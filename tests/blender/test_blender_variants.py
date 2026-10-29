"""Variants with synthetic results (no numpy core needed): save, restore, compare, remove."""

import json

import numpy as np
import pytest

import bpy
from bhelpers import bridge

from stress_sculpt import build, scene_io


def _fake_result(scene, name=build.OBJ_EVOLVED, size=(4.0, 1.0, 1.5), kind="EVOLVE", volume=0.3):
    """A box-shaped result object standing in for an evolved shape."""
    hx, hy, hz = (0.5 * s for s in size)
    verts = np.array(
        [
            [-hx, -hy, 0],
            [hx, -hy, 0],
            [hx, hy, 0],
            [-hx, hy, 0],
            [-hx, -hy, 2 * hz],
            [hx, -hy, 2 * hz],
            [hx, hy, 2 * hz],
            [-hx, hy, 2 * hz],
        ],
        dtype=float,
    )
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]]
    mesh = build.mesh_from_faces(name, verts, faces)
    obj = build.assign_result(scene, name, mesh, build.heatmap_material())
    settings = scene.stress_sculpt
    settings.last_kind = kind
    settings.last_volume = volume
    settings.last_compliance = 1.5
    settings.last_max_stress = 2.0e6
    settings.last_iterations = 12
    return obj


def test_variant_save_copies_visible_results(scene):
    settings = bridge(resolution=16)
    evolved = _fake_result(scene)
    hidden = _fake_result(scene, build.OBJ_STRESS)
    build.set_hidden(hidden, True)
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    variant = settings.variants[0]
    assert variant.name == "Variant 1" and variant.kind == "EVOLVE"
    assert variant.volume == pytest.approx(0.3) and variant.iterations == 12
    assert variant.collection.name == "Variant 1 – Evolved"
    copies = list(variant.collection.objects)
    assert [build.base_name(o.name) for o in copies] == ["V1 SS_Evolved"]  # hidden ones skipped
    assert copies[0].data != evolved.data and copies[0].data.materials[0].name == "SS_Heatmap"
    # the live result can change without touching the variant
    _fake_result(scene, size=(1.0, 1.0, 1.0))
    assert len(copies[0].data.vertices) == 8
    assert build.data_bounds(copies[0])[1][0] == pytest.approx(2.0)


def test_variant_restore_brings_back_the_setup(scene):
    settings = bridge(resolution=16)
    _fake_result(scene)
    load = next(o for o in scene.objects if o.stress_sculpt.role == "LOAD")
    support = next(o for o in scene.objects if o.stress_sculpt.role == "SUPPORT")
    support_name = support.name
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    saved = json.loads(settings.variants[0].setup_json)
    assert len(saved) == 3

    # change the configuration: move + weaken the load, delete a support, add a zone
    load.location.x = 1.2
    load.stress_sculpt.magnitude = 10.0
    build.remove_objects([support])
    bpy.ops.stress_sculpt.add_zone(role="HOLLOW")
    zone = bpy.context.view_layer.objects.active
    settings.resolution = 30
    settings.evo_mode = "RULES"

    assert bpy.ops.stress_sculpt.variant_restore() == {"FINISHED"}
    assert settings.resolution == 16 and settings.evo_mode == "SIMP"
    assert load.location.x == pytest.approx(0.0) and load.stress_sculpt.magnitude == 5000.0
    restored = scene.objects.get(support_name)
    assert restored is not None and restored.stress_sculpt.role == "SUPPORT"
    assert restored.empty_display_type == "CUBE"
    assert not zone.stress_sculpt.enabled  # not part of the saved configuration
    roles = sorted(o.stress_sculpt.role for o in scene_io.role_objects(bpy.context))
    assert roles == ["LOAD", "SUPPORT", "SUPPORT"]

    # settings only
    settings.resolution = 50
    assert bpy.ops.stress_sculpt.variant_restore(include_setup=False) == {"FINISHED"}
    assert settings.resolution == 16


def test_compare_layout_and_cleanup(scene):
    settings = bridge(resolution=16)
    settings.compare_spacing = 1.5
    _fake_result(scene, volume=0.3)
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    _fake_result(scene, size=(2.0, 1.0, 1.0), volume=0.2)
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    _fake_result(scene, size=(3.0, 1.0, 1.0), kind="SCULPT", volume=0.1)
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    assert [v.name for v in settings.variants] == ["Variant 1", "Variant 2", "Variant 3"]

    assert bpy.ops.stress_sculpt.variant_compare() == {"FINISHED"}
    compare = build.find_child_collection(scene.collection, build.COMPARE_COLLECTION)
    instances = sorted(
        (o for o in compare.objects if o.instance_type == "COLLECTION"), key=lambda o: o.location.x
    )
    labels = [o for o in compare.objects if o.type == "FONT"]
    assert len(instances) == 3 and len(labels) == 3
    # slots 4 wide (widest variant / domain) with gaps of 4 * (1.5 - 1) = 2, right of x = 2:
    # centres at 2 + 2 + 2 = 6, then every 6 (each variant is centred on x = 0)
    centres = [o.location.x for o in instances]
    assert centres == pytest.approx([6.0, 12.0, 18.0])
    assert np.diff(centres) == pytest.approx([6.0, 6.0])
    body = "\n".join(label.data.body for label in labels)
    assert "Variant 3" in body and "Sculpture" in body and "material 10%" in body
    # the variant collections stay excluded: only the instances show them
    parent = build.find_child_collection(scene.collection, build.VARIANTS_COLLECTION)
    assert build.find_layer_collection(bpy.context.view_layer.layer_collection, parent).exclude
    depsgraph = bpy.context.evaluated_depsgraph_get()
    instanced = [i.object.name for i in depsgraph.object_instances if i.is_instance]
    assert sum(1 for name in instanced if name.startswith("V")) == 3

    # removing a variant rebuilds the comparison without it
    settings.active_variant_index = 1
    assert bpy.ops.stress_sculpt.variant_remove() == {"FINISHED"}
    compare = build.find_child_collection(scene.collection, build.COMPARE_COLLECTION)
    assert len([o for o in compare.objects if o.instance_type == "COLLECTION"]) == 2
    assert [v.name for v in settings.variants] == ["Variant 1", "Variant 3"]
    assert settings.active_variant_index == 1
    # the next automatic name continues the numbering
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    assert settings.variants[-1].name == "Variant 4"

    assert bpy.ops.stress_sculpt.variant_clear_compare() == {"FINISHED"}
    assert build.find_child_collection(scene.collection, build.COMPARE_COLLECTION) is None
    assert not any(name.startswith("SS_Label") for name in bpy.data.curves.keys())


def test_variant_move_and_polls(scene):
    settings = bridge(resolution=16)
    assert not bpy.ops.stress_sculpt.variant_remove.poll()
    assert not bpy.ops.stress_sculpt.variant_compare.poll()
    _fake_result(scene)
    bpy.ops.stress_sculpt.variant_save()
    assert not bpy.ops.stress_sculpt.variant_move.poll()  # needs two
    bpy.ops.stress_sculpt.variant_save(variant_name="Second")
    assert bpy.ops.stress_sculpt.variant_move(direction="UP") == {"FINISHED"}
    assert [v.name for v in settings.variants] == ["Second", "Variant 1"]
    assert settings.active_variant_index == 0
    assert bpy.ops.stress_sculpt.variant_move(direction="UP") == {"CANCELLED"}  # already on top
