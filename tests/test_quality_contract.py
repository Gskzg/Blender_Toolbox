"""Pure-Python contract tests for the quality-first modeling lifecycle.

These tests intentionally avoid ``bpy`` so the common quality boundary can be
checked in a regular Python/CI process. Blender smoke tests exercise the same
contract through the executor; this file focuses on normalization, workflow
inheritance, and the task-agnostic planner.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox.addon import _model_plan, _quality_contract  # noqa: E402
from blender_toolbox.workflows import quality_bar, recommended_contract  # noqa: E402

BASELINE_STAGES = ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]


def test_quality_first_cannot_be_weakened_by_task_fields() -> None:
    """The safety profile always retains the common gates and topology sanity."""
    contract = _quality_contract(
        "quality_first",
        {
            "quality": {
                "enforce": False,
                "required_stages": [],
                "technical": {"require_topology": False},
                "primary_refs": "body",
                "min_quality": float("nan"),
                # These are runtime-owned and must not be trusted from JSON.
                "configured": False,
                "profile": "advisory",
                "version": "forged.v0",
            }
        },
    )

    assert contract["enforce"] is True
    assert contract["required_stages"] == BASELINE_STAGES
    assert contract["technical"]["require_topology"] is True
    assert contract["technical"]["require_connected"] is True
    assert contract["primary_refs"] == ["body"]
    assert contract["representation"]["primary_refs"] == ["body"]
    assert contract["min_quality"] == 0.86
    assert contract["min_primary_vertices"] == 512
    assert contract["min_secondary_vertices"] == 256
    assert contract["min_tertiary_vertices"] == 192
    assert contract["min_samples_per_feature"] == 8
    assert contract["evidence"]["min_views"] == 4
    assert contract["version"] == "quality_contract.v1"
    assert contract["configured"] is True
    assert contract["profile"] == "quality_first"
    assert contract["requested_enforce"] is False
    assert contract["requested_required_stages"] == []


def test_advisory_contract_stays_advisory_even_when_requested_enforce() -> None:
    """Inspection/resume callers may opt into advisory mode without an export gate."""
    contract = _quality_contract(
        "advisory",
        {"quality": {"enforce": True, "required_stages": [], "technical": {"require_topology": False}}},
    )

    assert contract["enforce"] is False
    assert contract["profile"] == "advisory"
    assert contract["required_stages"] == []
    # Advisory does not erase the default technical policy from the normalized
    # object, but it must not turn that policy into a lifecycle gate.
    assert contract["technical"]["require_topology"] is False
    assert contract["requested_enforce"] is True


def test_low_resolution_exception_cannot_disable_connectivity() -> None:
    contract = _quality_contract(
        "quality_first",
        {
            "quality": {
                "allow_low_resolution": True,
                "resolution_exception": {"exception_reason": "deliberate low-poly game asset"},
                "technical": {"require_connected": False, "expected_shells": {"body": 2}},
            }
        },
    )

    assert contract["resolution"]["allow_low_resolution"] is True
    assert contract["technical"]["require_connected"] is True
    assert contract["technical"]["expected_shells"] == {"body": 2}


def test_unconfigured_contract_is_advisory_and_has_discovery_defaults() -> None:
    contract = _quality_contract()

    assert contract["enforce"] is False
    assert contract["configured"] is False
    assert contract["profile"] == "advisory"
    assert contract["required_stages"] == BASELINE_STAGES
    assert contract["reference_views"] == ["front", "three_quarter", "side", "top"]
    assert contract["evidence"]["min_views"] == 4


def test_profile_inheritance_is_deep_and_domain_specific_additions_survive() -> None:
    vehicle = recommended_contract("vehicle")
    general = recommended_contract("general")

    # Both profiles inherit the same quality-first recommendation.
    assert vehicle["quality"]["enforce"] is True
    assert general["quality"]["enforce"] is True
    assert vehicle["quality"]["required_stages"] == BASELINE_STAGES
    assert general["quality"]["required_stages"] == BASELINE_STAGES

    # A leaf profile can add requirements without mutating the shared base.
    assert vehicle["require_closed"] is True
    assert vehicle["required_tags"] == ["vehicle", "body", "wheel", "tire", "window", "light"]
    # The inherited generic profile explicitly opts out of closed-shell
    # enforcement; this is still domain-neutral and unlike vehicle's gate.
    assert general.get("require_closed") is False
    assert "required_tags" not in general

    vehicle["quality"]["representation"]["carrier_options"].append("test_only")
    fresh = recommended_contract("vehicle")
    assert "test_only" not in fresh["quality"]["representation"]["carrier_options"]


def test_quality_contract_applies_workflow_aliases_without_vehicle_lock_in() -> None:
    vehicle = _quality_contract("quality_first", workflow_profile="vehicle")
    general = _quality_contract("quality_first", workflow_profile="general")

    assert vehicle["require_closed"] is True
    assert vehicle["required_semantic_parts"] == ["vehicle", "body", "wheel", "tire", "window", "light"]
    assert vehicle["reference_views"] == ["front", "side", "top", "three_quarter"]
    assert general["require_closed"] is False
    assert general["required_semantic_parts"] == []
    assert all("vehicle" not in tag for tag in general["required_semantic_parts"])


def test_quality_contract_normalizes_lists_and_finite_feature_scales() -> None:
    contract = _quality_contract(
        "quality_first",
        {
            "quality": {
                "primary_refs": " shell ",
                "secondary_refs": "trim",
                "reference_views": "front",
                "feature_scales": [0.01, "0.2", 0, -1, float("inf"), "bad"],
                "required_tags": [" body ", "", 7],
                "min_evidence_views": 0,
                "evidence": {"min_views": 99},
            }
        },
    )

    assert contract["primary_refs"] == ["shell"]
    assert contract["secondary_refs"] == ["trim"]
    assert contract["reference_views"] == ["front"]
    assert contract["feature_scales"] == [0.01, 0.2]
    assert contract["required_semantic_parts"] == ["body", "7"]
    assert contract["min_evidence_views"] == 4
    assert contract["evidence"]["min_views"] == 32


def test_top_level_evidence_view_alias_controls_nested_minimum() -> None:
    contract = _quality_contract("quality_first", {"quality": {"min_evidence_views": 1}})
    assert contract["min_evidence_views"] == 4
    assert contract["evidence"]["min_views"] == 4


def test_quality_bar_is_domain_neutral_and_marks_optional_stages_adaptive() -> None:
    bar = quality_bar()

    assert bar["scope"] == "domain_neutral"
    assert bar["stage_policy"]["baseline"] == BASELINE_STAGES
    assert bar["stage_policy"]["optional"] == []
    assert bar["defaults"]["min_quality"] == 0.86
    assert bar["defaults"]["min_samples_per_feature"] == 8
    assert bar["defaults"]["require_high_resolution"] is True
    assert bar["defaults"]["require_connected"] is True
    assert "primary_primitive_pile" in bar["anti_slop"]
    assert bar["declaration_contract"]["unknown_is_not_pass"] is True


def test_model_plan_normalizes_string_refs_and_adapts_to_generic_signals() -> None:
    plan = _model_plan(
        {
            "intent": "organic repeated relief with an opening",
            "task_spec": {
                "identity": {"family": "ornamental"},
                "scale": {"unit": "m", "height": 1.0},
                "quality": {
                    "primary_refs": "envelope",
                    "secondary_refs": "rim",
                    "feature_scales": [0.02, "0.05", float("nan")],
                },
                "openings": True,
                "local_relief": True,
                "repeated": True,
            },
        }
    )

    assert plan["representation"]["primary_refs"] == ["envelope"]
    assert plan["quality_contract_template"]["primary_refs"] == ["envelope"]
    assert plan["signals"] == {
        "continuous_envelope": True,
        "repeated": True,
        "openings": True,
        "local_relief": True,
        "strict_topology": False,
    }
    assert plan["stages"] == ["structure", "primary", "secondary", "tertiary", "technical", "evidence"]
    assert plan["declaration_status"]["identity"] is True
    assert plan["declaration_status"]["scale"] is True
    assert plan["declaration_status"]["primary_refs"] is True
    assert plan["quality_contract_template"]["feature_scales"] == [0.02, 0.05]
    assert "vehicle" not in str(plan).lower()


def test_model_plan_reports_missing_declarations_instead_of_guessing() -> None:
    plan = _model_plan({"intent": "simple prop"})

    assert plan["missing_contract"] == [
        "quality.identity",
        "quality.scale",
        "quality.primary_refs",
        "quality.reference_views",
        "quality.secondary_refs",
        "quality.feature_scales",
        "quality.detail_regions",
    ]
    assert "quality.feature_scales" in plan["missing_contract"]
    assert "quality.detail_regions" in plan["missing_contract"]
    assert "quality.feature_scales (required for detail sampling)" in plan["recommended_missing"]
    assert plan["quality_contract_template"]["enforce"] is True
    assert plan["quality_contract_template"]["required_stages"] == BASELINE_STAGES


def test_low_resolution_requires_a_documented_exception_reason() -> None:
    without_reason = _quality_contract(
        "quality_first",
        {"quality": {"allow_low_resolution": True, "min_primary_vertices": 32}},
    )
    assert without_reason["resolution"]["allow_low_resolution"] is False
    assert without_reason["resolution"]["exception_reason"] == ""
    assert without_reason["min_primary_vertices"] == 512

    with_reason = _quality_contract(
        "quality_first",
        {
            "quality": {
                "allow_low_resolution": True,
                "resolution_exception": {"exception_reason": "deliberate low-poly reference study"},
                "min_primary_vertices": 32,
                "resolution": {"min_primary_vertices": 32, "min_samples_per_feature": 2},
            }
        },
    )
    assert with_reason["resolution"]["allow_low_resolution"] is True
    assert with_reason["resolution"]["exception_reason"] == "deliberate low-poly reference study"
    assert with_reason["resolution"]["min_primary_vertices"] == 32
    assert with_reason["resolution"]["min_samples_per_feature"] == 2


def test_model_plan_deep_merges_default_task_spec_without_mutating_input() -> None:
    defaults = {
        "quality": {"primary_refs": ["carrier"], "technical": {"strict_topology": True}},
        "identity": {"family": "widget"},
    }
    supplied = {"task_spec": {"quality": {"feature_scales": [0.01]}}}
    original = copy.deepcopy(defaults)

    plan = _model_plan(supplied, default_task_spec=defaults)

    assert defaults == original
    assert plan["quality_contract_template"]["primary_refs"] == ["carrier"]
    assert plan["quality_contract_template"]["feature_scales"] == [0.01]
    assert plan["quality_contract_template"]["technical"]["strict_topology"] is True
    assert plan["declaration_status"]["identity"] is True


@pytest.mark.parametrize("name", ["missing", "", "VEHICLE-ish"])
def test_unknown_workflow_profile_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="unknown workflow profile"):
        recommended_contract(name)
