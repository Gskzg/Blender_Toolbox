"""Pure-Python regression checks for creation-action transaction boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_toolbox import addon as addon_module  # noqa: E402
from blender_toolbox.addon import ExecutorError  # noqa: E402


def test_hair_validates_late_strands_before_allocating_curve(monkeypatch) -> None:
    class Curves:
        def new(self, *_args, **_kwargs):  # pragma: no cover - should never run
            raise AssertionError("curve datablock allocated before validation")

    monkeypatch.setattr(addon_module, "bpy", SimpleNamespace(data=SimpleNamespace(curves=Curves())))
    with pytest.raises(ExecutorError, match="each strand needs at least two points"):
        addon_module._hair_create_strands(
            {"strands": [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0]]]}
        )


def test_armature_validates_parent_before_allocating_datablock(monkeypatch) -> None:
    class Armatures:
        def new(self, *_args, **_kwargs):  # pragma: no cover - should never run
            raise AssertionError("armature datablock allocated before validation")

    monkeypatch.setattr(addon_module, "bpy", SimpleNamespace(data=SimpleNamespace(armatures=Armatures())))
    with pytest.raises(ExecutorError, match="parent bone not found"):
        addon_module._rig_create_armature(
            {
                "name": "Rig",
                "bones": [
                    {"name": "root", "head": [0, 0, 0], "tail": [0, 0, 1], "parent": "missing"},
                ],
            }
        )


def test_face_curve_resolves_surface_before_creation(monkeypatch) -> None:
    monkeypatch.setattr(addon_module, "bpy", SimpleNamespace())
    monkeypatch.setattr(addon_module, "_landmark_points", lambda _refs: [(0, 0, 0), (1, 0, 0)])

    def missing_surface(_ref):
        raise ExecutorError("surface not found", "not_found")

    monkeypatch.setattr(addon_module, "_object_by_ref", missing_surface)
    monkeypatch.setattr(addon_module, "_create_curve", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("curve was created")))
    with pytest.raises(ExecutorError, match="surface not found"):
        addon_module._face_curve_from_landmarks(
            {"name": "crease", "landmarks": ["a", "b"], "surface": "missing"}
        )


def test_landmark_resolves_parent_before_allocating_empty(monkeypatch) -> None:
    class Objects:
        def get(self, _name):
            return None

        def new(self, *_args, **_kwargs):  # pragma: no cover - should never run
            raise AssertionError("landmark allocated before parent validation")

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(
        addon_module,
        "bpy",
        SimpleNamespace(data=SimpleNamespace(objects=Objects())),
    )

    def missing_parent(_ref):
        raise ExecutorError("parent not found", "not_found")

    monkeypatch.setattr(addon_module, "_object_by_ref", missing_parent)
    with pytest.raises(ExecutorError, match="parent not found"):
        addon_module._landmark_create(
            {"name": "nose", "location": [0, 0, 0], "parent": "missing"}
        )


def test_landmark_update_restores_existing_object_on_dependency_failure(monkeypatch) -> None:
    class Landmark(dict):
        name = "nose"
        type = "EMPTY"

        def __init__(self):
            super().__init__()
            self.location = [1.0, 2.0, 3.0]
            self.parent = None

    landmark = Landmark()

    class Objects:
        def get(self, name):
            return landmark if name == landmark.name else None

        def __iter__(self):
            return iter((landmark,))

    class ViewLayer:
        def update(self):
            raise RuntimeError("dependency graph unavailable")

    fake_bpy = SimpleNamespace(
        data=SimpleNamespace(objects=Objects()),
        context=SimpleNamespace(
            scene=SimpleNamespace(objects=[landmark]),
            view_layer=ViewLayer(),
        ),
    )
    monkeypatch.setattr(addon_module, "bpy", fake_bpy)

    with pytest.raises(RuntimeError, match="dependency graph unavailable"):
        addon_module._landmark_create(
            {
                "name": "nose",
                "location": [9, 8, 7],
                "semantic_tags": ["changed"],
                "role": "feature",
            }
        )

    assert landmark.location == [1.0, 2.0, 3.0]
    assert "blender_toolbox_uuid" not in landmark
    assert "blender_toolbox_semantic_tags" not in landmark
    assert "blender_toolbox_role" not in landmark


def test_discard_created_object_removes_object_and_orphan_data(monkeypatch) -> None:
    class DataBlock:
        users = 0

    class Obj(dict):
        name = "Generated"

        def __init__(self, data):
            super().__init__()
            self.data = data

    data = DataBlock()
    obj = Obj(data)
    objects = [obj]
    meshes = [data]
    fake_bpy = SimpleNamespace(data=SimpleNamespace(objects=objects, meshes=meshes), context=SimpleNamespace())
    monkeypatch.setattr(addon_module, "bpy", fake_bpy)

    addon_module._discard_created_object(obj)

    assert objects == []
    assert meshes == []
