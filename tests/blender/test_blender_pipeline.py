"""End-to-end operators on the bridge demo: analyze, evolve, lines, sculpture, relief.

Low resolutions keep the suite fast; tests skip while a core module is missing.
"""

import json

import numpy as np
import pytest

import bpy
from bhelpers import bridge, mesh_attribute_values, require

from stress_sculpt import build, jobs, scene_io


def _result(scene, name):
    obj = build.find_result(scene, name)
    assert obj is not None, f"{name} was not built"
    return obj


def _setup(resolution=16, **overrides):
    settings = bridge(resolution=resolution)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


# ----------------------------------------------------------------------- analyze
def test_analyze_builds_heatmap(scene):
    require("fem", "mesher")
    settings = _setup()
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    obj = _result(scene, build.OBJ_STRESS)
    mesh = obj.data
    assert obj.type == "MESH" and len(mesh.vertices) > 8  # the box copy was refined
    assert mesh.color_attributes.active_color.name == "Stress"
    colors = mesh_attribute_values(mesh, "Stress", "color", 4)
    assert np.all((colors >= 0.0) & (colors <= 1.0)) and np.ptp(colors[:, :3]) > 0.1
    values = mesh_attribute_values(mesh, "stress", "value")
    assert values.max() > 0.0
    assert obj.material_slots[0].material.name == "SS_Heatmap"
    assert settings.last_max_stress > 0.0 and settings.last_max_displacement > 0.0
    assert settings.last_compliance > 0.0 and settings.last_kind == "ANALYSIS"
    assert settings.domain.display_type == "WIRE"
    cache = jobs.get_cache(scene).analysis
    assert cache is not None and cache["signature"] == scene_io.setup_signature(bpy.context)
    assert cache["result"].von_mises.shape == cache["grid"].shape
    # the vertices are in world space on the domain surface
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    assert co.min(axis=0) == pytest.approx([-2.0, -0.5, 0.0], abs=1e-5)
    assert co.max(axis=0) == pytest.approx([2.0, 0.5, 1.5], abs=1e-5)

    # re-running keeps the object and replaces its data
    previous = mesh.name
    settings.stress_measure = "PRINCIPAL_MAX"
    settings.colormap = "COOLWARM"
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    assert _result(scene, build.OBJ_STRESS) == obj and previous not in bpy.data.meshes


def test_analyze_voxel_surface_and_measures(scene):
    require("fem", "mesher")
    settings = _setup(heatmap_on_original=False)
    for measure in ("VON_MISES", "PRINCIPAL_MIN", "DISPLACEMENT"):
        settings.stress_measure = measure
        assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
        mesh = _result(scene, build.OBJ_STRESS).data
        assert len(mesh.polygons) > 0 and "Stress" in mesh.color_attributes
        if measure == "PRINCIPAL_MIN":
            assert mesh_attribute_values(mesh, "stress", "value").min() < 0.0  # compression


def test_analyze_reports_setup_errors(scene):
    require("fem", "mesher")
    with pytest.raises(RuntimeError, match="domain"):
        bpy.ops.stress_sculpt.analyze()


# ------------------------------------------------------------------------ evolve
@pytest.mark.parametrize("mode", ["SIMP", "RULES"])
def test_evolve_builds_shape(scene, mode):
    require("fem", "topopt", "mesher")
    settings = _setup(evo_mode=mode, max_iterations=8, volume_fraction=0.4, smooth_iterations=2)
    assert bpy.ops.stress_sculpt.evolve() == {"FINISHED"}
    obj = _result(scene, build.OBJ_EVOLVED)
    assert len(obj.data.polygons) > 0 and "Stress" in obj.data.color_attributes
    evolution = jobs.get_cache(scene).evolution
    assert evolution is not None and evolution["state"].iteration >= 1
    assert (
        settings.last_kind == "EVOLVE" and settings.last_iterations == evolution["state"].iteration
    )
    assert 0.0 < settings.last_volume <= 1.0 and settings.last_compliance > 0.0


# ------------------------------------------------------------------------- lines
def test_trace_lines_builds_curves(scene):
    require("fem", "streamlines")
    settings = _setup(n_seeds=12, branch_depth=1, radius_min=0.3, radius_max=1.0)
    assert jobs.get_cache(scene).analysis is None
    assert bpy.ops.stress_sculpt.trace_lines() == {"FINISHED"}  # analyzes on its own
    assert jobs.get_cache(scene).analysis is not None
    entry = jobs.get_cache(scene).lines
    assert entry is not None and entry["lines"]
    families = {line.family for line in entry["lines"]}
    h = entry["h"]
    for family, name in (
        ("TENSION", build.OBJ_LINES_TENSION),
        ("COMPRESSION", build.OBJ_LINES_COMPRESSION),
    ):
        if family not in families:
            continue
        obj = _result(scene, name)
        curve = obj.data
        assert obj.type == "CURVE" and len(curve.splines) > 0
        assert curve.bevel_depth == pytest.approx(1.0 * h)
        radii = np.array([p.radius for s in curve.splines for p in s.points])
        assert (
            radii.min() >= 0.3 * settings.branch_taper**settings.branch_depth - 1e-6
            and radii.max() <= 1.0 + 1e-6
        )
        assert all(s.type == "POLY" for s in curve.splines)
    assert settings.last_kind == "LINES"
    # only tension requested: the stale compression curve goes away
    settings.line_families = "TENSION"
    assert bpy.ops.stress_sculpt.trace_lines() == {"FINISHED"}
    assert build.find_result(scene, build.OBJ_LINES_COMPRESSION) is None


def test_trace_lines_from_evolved_requires_evolution(scene):
    require("fem", "streamlines")
    _setup(line_source="EVOLVED")
    with pytest.raises(RuntimeError, match="Evolve"):
        bpy.ops.stress_sculpt.trace_lines()


# ---------------------------------------------------------------------- sculpture
def test_sculpt_traces_if_needed(scene):
    require("fem", "streamlines", "sculpt", "mesher")
    settings = _setup(n_seeds=10, sculpt_detail=1.5, sculpt_smooth=2)
    assert bpy.ops.stress_sculpt.sculpt() == {"FINISHED"}
    obj = _result(scene, build.OBJ_SCULPTURE)
    assert obj.type == "MESH" and len(obj.data.polygons) > 0
    assert obj.material_slots[0].material.name == "SS_Sculpture"
    assert "Stress" in obj.data.color_attributes and "stress" in obj.data.attributes
    assert jobs.get_cache(scene).lines is not None  # traced on the way
    assert settings.last_kind == "SCULPT"


def test_sculpt_lines_and_solid(scene):
    require("fem", "topopt", "streamlines", "sculpt", "mesher")
    _setup(
        n_seeds=8,
        max_iterations=3,
        sculpt_detail=1.0,
        sculpt_smooth=0,
        sculpt_source="LINES_AND_SOLID",
    )
    assert bpy.ops.stress_sculpt.evolve() == {"FINISHED"}
    assert bpy.ops.stress_sculpt.trace_lines() == {"FINISHED"}
    assert bpy.ops.stress_sculpt.sculpt() == {"FINISHED"}
    assert len(_result(scene, build.OBJ_SCULPTURE).data.polygons) > 0


# ------------------------------------------------------------------------- relief
def test_relief(scene):
    require("fem", "mesher")
    settings = _setup(relief_amount=0.05, relief_subdivisions=0, relief_smooth=2)
    assert bpy.ops.stress_sculpt.relief() == {"FINISHED"}
    obj = _result(scene, build.OBJ_RELIEF)
    mesh = obj.data
    assert len(mesh.vertices) > 8 and "Stress" in mesh.color_attributes
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    # pushed out where stressed and carved where not
    lo, hi = co.min(axis=0), co.max(axis=0)
    assert np.any(lo < np.array([-2.0, -0.5, 0.0]) - 1e-4) or np.any(
        hi > np.array([2.0, 0.5, 1.5]) + 1e-4
    )
    assert settings.last_kind == "RELIEF"
    analysis = jobs.get_cache(scene).analysis
    assert analysis is not None
    # a second relief reuses the cached analysis
    settings.relief_hollow = False
    assert bpy.ops.stress_sculpt.relief() == {"FINISHED"}
    assert jobs.get_cache(scene).analysis is analysis


# ----------------------------------------------------------------------- variants
def test_variants_save_compare_restore_remove(scene):
    require("fem", "mesher")
    settings = _setup()
    with pytest.raises(RuntimeError, match="Nothing to save"):
        bpy.ops.stress_sculpt.variant_save()
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    assert len(settings.variants) == 1
    first = settings.variants[0]
    assert first.name == "Variant 1" and first.kind == "ANALYSIS"
    assert first.max_stress == pytest.approx(settings.last_max_stress)
    collection = first.collection
    assert collection is not None and collection.name.startswith("Variant 1")
    parent = build.find_child_collection(scene.collection, build.VARIANTS_COLLECTION)
    assert build.find_layer_collection(bpy.context.view_layer.layer_collection, parent).exclude
    copies = list(collection.objects)
    assert len(copies) == 1 and copies[0].data != _result(scene, build.OBJ_STRESS).data
    saved = json.loads(first.settings_json)
    assert saved["resolution"] == 16 and saved["domain"] == "SS_Domain" and "variants" not in saved
    setup = json.loads(first.setup_json)
    assert {entry["role"] for entry in setup} == {"SUPPORT", "LOAD"}
    assert all(
        {"name", "role", "location", "rotation", "magnitude", "enabled"} <= set(entry)
        for entry in setup
    )

    # a second configuration
    load = next(o for o in scene.objects if o.stress_sculpt.role == "LOAD")
    original_location = tuple(load.location)
    load.location.x = 1.0
    settings.resolution = 18
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    assert bpy.ops.stress_sculpt.variant_save(variant_name="Off centre") == {"FINISHED"}
    assert [v.name for v in settings.variants] == ["Variant 1", "Off centre"]
    assert settings.active_variant_index == 1

    assert bpy.ops.stress_sculpt.variant_compare() == {"FINISHED"}
    compare = build.find_child_collection(scene.collection, build.COMPARE_COLLECTION)
    instances = [o for o in compare.objects if o.instance_type == "COLLECTION"]
    labels = [o for o in compare.objects if o.type == "FONT"]
    assert len(instances) == 2 and len(labels) == 2
    assert {o.instance_collection for o in instances} == {v.collection for v in settings.variants}
    xs = sorted(o.location.x for o in instances)
    assert xs[0] > 0.0 and xs[1] - xs[0] >= 4.0 * settings.compare_spacing - 1e-3
    assert "Variant 1" in "".join(label.data.body for label in labels)

    settings.active_variant_index = 0
    assert bpy.ops.stress_sculpt.variant_restore() == {"FINISHED"}
    assert settings.resolution == 16
    assert tuple(load.location) == pytest.approx(original_location)

    assert bpy.ops.stress_sculpt.variant_move(direction="DOWN") == {"FINISHED"}
    assert [v.name for v in settings.variants] == ["Off centre", "Variant 1"]
    assert settings.active_variant_index == 1

    assert bpy.ops.stress_sculpt.variant_remove() == {"FINISHED"}
    assert [v.name for v in settings.variants] == ["Off centre"]
    compare = build.find_child_collection(scene.collection, build.COMPARE_COLLECTION)
    assert (
        compare is not None
        and len([o for o in compare.objects if o.instance_type == "COLLECTION"]) == 1
    )

    assert bpy.ops.stress_sculpt.variant_clear_compare() == {"FINISHED"}
    assert build.find_child_collection(scene.collection, build.COMPARE_COLLECTION) is None


def test_variant_remove_deletes_objects(scene):
    require("fem", "mesher")
    settings = _setup()
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    collection_name = settings.variants[0].collection.name
    copy_names = [o.name for o in settings.variants[0].collection.objects]
    mesh_names = [o.data.name for o in settings.variants[0].collection.objects]
    assert bpy.ops.stress_sculpt.variant_remove() == {"FINISHED"}
    assert collection_name not in bpy.data.collections
    assert not any(name in bpy.data.objects for name in copy_names)
    assert not any(name in bpy.data.meshes for name in mesh_names)
    assert len(settings.variants) == 0


# ------------------------------------------------------------------- clear results
def test_clear_results(scene):
    require("fem", "mesher")
    settings = _setup()
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    mesh_name = _result(scene, build.OBJ_STRESS).data.name
    assert bpy.ops.stress_sculpt.clear_results() == {"FINISHED"}
    assert build.find_result(scene, build.OBJ_STRESS) is None
    assert mesh_name not in bpy.data.meshes
    assert jobs.get_cache(scene).analysis is None
    assert settings.domain.display_type == "TEXTURED" and settings.last_kind == ""


def test_analyze_from_edit_mode(scene):
    require("fem", "mesher")
    settings = _setup()
    bpy.context.view_layer.objects.active = settings.domain
    settings.domain.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    assert bpy.context.mode == "EDIT_MESH"
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    assert bpy.context.mode == "OBJECT"
    assert build.find_result(scene, build.OBJ_STRESS) is not None


def test_results_are_per_scene(scene):
    require("fem", "mesher")
    _setup()
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    first = _result(scene, build.OBJ_STRESS)
    other = bpy.data.scenes.new("Other")
    bpy.context.window.scene = other
    assert bpy.context.scene == other
    _setup()
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    second = _result(other, build.OBJ_STRESS)
    assert second != first and build.base_name(second.name) == build.OBJ_STRESS
    assert build.results_collection(other, create=False) != build.results_collection(
        scene, create=False
    )
    assert jobs.get_cache(other).analysis is not jobs.get_cache(scene).analysis
    # re-running in the first scene updates its own object only
    bpy.context.window.scene = scene
    assert bpy.ops.stress_sculpt.analyze() == {"FINISHED"}
    assert _result(scene, build.OBJ_STRESS) == first and _result(other, build.OBJ_STRESS) == second


def test_sculpt_reduces_detail_to_fit_memory(scene, monkeypatch):
    require("fem", "streamlines", "sculpt", "mesher")
    from stress_sculpt import operators

    _setup(n_seeds=6, sculpt_detail=6.0, sculpt_smooth=0)
    monkeypatch.setattr(operators, "SCULPT_MAX_CELLS", 20000)
    assert bpy.ops.stress_sculpt.sculpt() == {"FINISHED"}
    assert len(_result(scene, build.OBJ_SCULPTURE).data.polygons) > 0
