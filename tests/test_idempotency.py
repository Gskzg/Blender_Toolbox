"""Idempotency behavior that can be tested without a Blender installation."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox.addon import ExecutorError, _CoreToolboxExecutor, _object_reference  # noqa: E402
from blender_toolbox.protocol import SCHEMA_VERSION  # noqa: E402


def _payload(*, request_id: str, args: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "session_id": "session-test",
        "episode_id": "episode-test",
        "step_id": 0,
        "action": "session.create",
        "args": args,
        "idempotency_key": "episode-test:0:session.create",
    }


def test_idempotency_replays_same_request_and_rejects_key_reuse(monkeypatch) -> None:
    executor = _CoreToolboxExecutor()
    monkeypatch.setattr(
        executor,
        "_state",
        lambda *, refresh=False: {
            "summary": {},
            "state_hash": "sha256:test",
        },
    )
    monkeypatch.setattr(
        executor,
        "_dispatch",
        lambda action, args, request=None, *, required_tags_lock=None: {"session": "Test"},
    )

    first = executor.execute(_payload(request_id="req-first", args={"mode": "resume"}))
    retry = executor.execute(_payload(request_id="req-retry", args={"mode": "resume"}))
    conflict = executor.execute(_payload(request_id="req-conflict", args={"mode": "new"}))

    assert first["ok"] is True
    assert retry == first
    assert conflict["ok"] is False
    assert conflict["error"]["code"] == "idempotency_conflict"


def test_idempotency_cache_isolated_from_nested_response_mutations(monkeypatch) -> None:
    """A caller must not be able to corrupt a later retry via a response alias."""
    executor = _CoreToolboxExecutor()
    monkeypatch.setattr(
        executor,
        "_state",
        lambda *, refresh=False: {
            "summary": {"objects": [{"uuid": "obj-1", "tags": ["primary"]}]},
            "state_hash": "sha256:test",
        },
    )
    monkeypatch.setattr(
        executor,
        "_dispatch",
        lambda action, args, request=None, *, required_tags_lock=None: {
            "session": "Test",
            "metadata": {"tags": ["primary"]},
        },
    )

    first = executor.execute(_payload(request_id="req-first", args={"mode": "resume"}))
    first["result"]["metadata"]["tags"].append("caller-mutated")
    first["state"]["summary"]["objects"][0]["tags"].clear()

    retry = executor.execute(_payload(request_id="req-retry", args={"mode": "resume"}))

    assert retry["result"]["metadata"]["tags"] == ["primary"]
    assert retry["state"]["summary"]["objects"][0]["tags"] == ["primary"]

    # Mutating the retry itself must not poison a subsequent retry either.
    retry["result"]["metadata"]["tags"].append("second-mutation")
    again = executor.execute(_payload(request_id="req-again", args={"mode": "resume"}))
    assert again["result"]["metadata"]["tags"] == ["primary"]


def test_failed_session_open_rolls_back_lifecycle_metadata(monkeypatch) -> None:
    """A failed open must not reserve a session or leak its task contract."""
    executor = _CoreToolboxExecutor()
    baseline = {
        "session_id": executor._session_id,
        "required_tags_lock": executor._required_tags_lock,
        "task_spec": executor._task_spec.copy(),
        "task_spec_frozen": executor._task_spec_frozen,
        "task_spec_hash": executor._task_spec_hash,
        "profile": executor._profile,
        "quality_contract": executor._quality_contract.copy(),
    }
    monkeypatch.setattr(
        executor,
        "_state",
        lambda *, refresh=False: {"summary": {}, "state_hash": "sha256:test"},
    )

    def fail_after_partial_open(action, args, request=None, *, required_tags_lock=None):
        executor._task_spec = {"leaked": True}
        executor._task_spec_frozen = True
        executor._task_spec_hash = "sha256:leaked"
        executor._profile = "leaked"
        executor._quality_contract = {"enforce": True, "leaked": True}
        raise ExecutorError("open failed", "execution_error")

    monkeypatch.setattr(executor, "_dispatch", fail_after_partial_open)
    response = executor.execute(
        {
            "session_id": "new-session",
            "episode_id": "episode",
            "step_id": 0,
            "action": "session.open",
            "args": {"mode": "resume"},
        }
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "execution_error"
    assert executor._session_id == baseline["session_id"]
    assert executor._required_tags_lock == baseline["required_tags_lock"]
    assert executor._task_spec == baseline["task_spec"]
    assert executor._task_spec_frozen is baseline["task_spec_frozen"]
    assert executor._task_spec_hash == baseline["task_spec_hash"]
    assert executor._profile == baseline["profile"]
    assert executor._quality_contract == baseline["quality_contract"]


def test_failed_verify_does_not_commit_required_tag_lock(monkeypatch) -> None:
    """A failed verification cannot make new tags sticky for the episode."""
    executor = _CoreToolboxExecutor()
    executor._session_id = "session-test"
    executor._required_tags_lock = frozenset({"existing"})
    monkeypatch.setattr(
        executor,
        "_state",
        lambda *, refresh=False: {"summary": {}, "state_hash": "sha256:test"},
    )
    monkeypatch.setattr(
        executor,
        "_dispatch",
        lambda action, args, request=None, *, required_tags_lock=None: (_ for _ in ()).throw(
            ExecutorError("verify failed", "execution_error")
        ),
    )

    response = executor.execute(
        {
            "session_id": "session-test",
            "episode_id": "episode",
            "step_id": 0,
            "action": "verify.run",
            "args": {"required_tags": ["new"]},
        }
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "execution_error"
    assert executor._required_tags_lock == frozenset({"existing"})


def test_post_dispatch_failure_before_response_restores_lifecycle_metadata(monkeypatch) -> None:
    """Census/response failures must not leave a partially opened contract."""
    executor = _CoreToolboxExecutor()
    baseline_contract = executor._quality_contract.copy()
    state_calls = 0

    def state(*, refresh=False):
        nonlocal state_calls
        state_calls += 1
        if state_calls == 1:
            return {"summary": {}, "state_hash": "sha256:before"}
        raise ExecutorError("post-dispatch census failed", "execution_error")

    monkeypatch.setattr(executor, "_state", state)

    def partial_open(action, args, request=None, *, required_tags_lock=None):
        executor._task_spec = {"partial": True}
        executor._task_spec_frozen = True
        executor._task_spec_hash = "sha256:partial"
        executor._profile = "partial"
        executor._quality_contract = {"partial": True}
        return {"session": "Test"}

    monkeypatch.setattr(executor, "_dispatch", partial_open)
    response = executor.execute(
        {
            "session_id": "new-session",
            "episode_id": "episode",
            "step_id": 0,
            "action": "session.open",
            "args": {"mode": "resume"},
        }
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "execution_error"
    assert executor._session_id is None
    assert executor._task_spec == {}
    assert executor._task_spec_frozen is False
    assert executor._task_spec_hash is None
    assert executor._profile is None
    assert executor._quality_contract == baseline_contract


def test_object_reference_aliases_are_normalized_and_conflicts_rejected() -> None:
    assert _object_reference({"id": "  body "}) == "body"
    assert _object_reference({"ref": "body"}) == "body"
    assert _object_reference({"id": " body ", "ref": "body"}) == "body"

    try:
        _object_reference({"id": "body", "ref": "other"})
    except ExecutorError as exc:
        assert exc.code == "invalid_args"
        assert "must match" in str(exc)
    else:  # pragma: no cover - assertion keeps the expected failure explicit
        raise AssertionError("conflicting id/ref aliases must be rejected")


def test_invalid_request_id_error_does_not_stringify_falsey_or_arbitrary_values() -> None:
    executor = _CoreToolboxExecutor()
    for value in (None, 0, False, {"untrusted": "value"}):
        response = executor.execute(
            {
                "request_id": value,
                "session_id": "session-test",
                "episode_id": "episode-test",
                "step_id": 0,
                "action": "model.plan",
                "args": {},
            }
        )
        assert response["ok"] is False
        assert response["request_id"] == ""
