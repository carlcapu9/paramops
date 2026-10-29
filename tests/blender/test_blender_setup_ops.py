"""Setup operators: registration, helpers at the cursor, roles, domain and demo scenes."""

import json
import math

import numpy as np
import pytest

import bpy
from bhelpers import CLASSIC_PRESETS, box, bridge, classic_setup, require

from stress_sculpt import build, jobs, operators, scene_io

OPERATORS = (
    "set_domain",
    "add_support",
    "add_load",
    "add_zone",
    "set_role",
    "select_role",
    "set_target",
    "add_load_at_selection",
    "add_support_at_selection",
    "remove_role",
    "demo_scene",
    "generate",
    "verify",
    "analyze",
    "evolve",
    "trace_lines",
    "sculpt",
    "relief",
    "cancel",
    "clear_results",
    "variant_save",
    "variant_remove",
    "variant_restore",
    "variant_compare",
    "variant_clear_compare",
    "variant_move",
)


def test_operators_registered(scene):
    for name in OPERATORS:
        assert hasattr(bpy.types, f"STRESS_SCULPT_OT_{name}"), name
        assert getattr(bpy.ops.stress_sculpt, name).idname() == f"STRESS_SCULPT_OT_{name}"
    assert len(operators.classes) == len(OPERATORS)
    assert hasattr(operators, "register") and hasattr(operators, "unregister")
    assert jobs.RUNTIME.running is False


def test_operators_reregister_cleanly(scene):
    operators.unregister()
    assert not hasattr(bpy.types, "STRESS_SCULPT_OT_analyze")
    operators.register()
    assert hasattr(bpy.types, "STRESS_SCULPT_OT_analyze")
    assert (
        sum(1 for h in bpy.app.handlers.load_pre if getattr(h, "__module__", "") == jobs.__name__)
        == 1
    )


def test_full_addon_registration(addon):
    if addon != "full":
        pytest.skip("the package registration (__init__/ui/translations) is not available")
    assert hasattr(bpy.types, "STRESS_SCULPT_PT_structure")


def _selected(scene):
    return [obj for obj in scene_io.view_layer_objects(bpy.context) if obj.select_get()]


def test_add_support_without_and_with_domain(scene):
    scene.cursor.location = (1.0, 2.0, 3.0)
    assert bpy.ops.stress_sculpt.add_support() == {"FINISHED"}
    support = bpy.context.view_layer.objects.active
    assert support.name.startswith("SS_Support") and support.type == "EMPTY"
    assert support.stress_sculpt.role == "SUPPORT"
    assert support.empty_display_type == "SPHERE" and support.empty_display_size == pytest.approx(
        0.25
    )
    assert tuple(support.location) == pytest.approx((1.0, 2.0, 3.0))
    assert (
        support.name
        in build.find_child_collection(scene.collection, build.SETUP_COLLECTION).objects
    )
    assert _selected(scene) == [support]
    assert support.color[2] > support.color[0]  # blue

    domain = box("Domain", size=(3.0, 4.0, 0.0001 + 12.0))
    scene.stress_sculpt.domain = domain
    assert bpy.ops.stress_sculpt.add_support(shape="CUBE") == {"FINISHED"}
    cube = bpy.context.view_layer.objects.active
    diagonal = math.sqrt(3.0**2 + 4.0**2 + 12.0001**2)
    assert cube.empty_display_type == "CUBE"
    assert cube.empty_display_size == pytest.approx(0.05 * diagonal, rel=1e-4)
    assert _selected(scene) == [cube]


def test_add_load_points_down(scene):
    domain = box("Domain", size=(2.0, 2.0, 1.0))
    scene.stress_sculpt.domain = domain
    assert bpy.ops.stress_sculpt.add_load() == {"FINISHED"}
    load = bpy.context.view_layer.objects.active
    assert load.stress_sculpt.role == "LOAD" and load.empty_display_type == "SINGLE_ARROW"
    assert load.rotation_euler.x == pytest.approx(math.pi)
    assert load.empty_display_size == pytest.approx(0.15 * 3.0, rel=1e-5)
    bpy.context.view_layer.update()
    direction = np.array(load.matrix_world)[:3, 2]
    assert direction / np.linalg.norm(direction) == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)
    assert load.color[0] > load.color[2]  # red


def test_add_zone_names_and_roles(scene):
    for role, name in (
        ("KEEP_SOLID", "SS_KeepSolid"),
        ("KEEP_VOID", "SS_KeepVoid"),
        ("THICKEN", "SS_Thicken"),
        ("HOLLOW", "SS_Hollow"),
        ("BRANCH", "SS_Branch"),
    ):
        assert bpy.ops.stress_sculpt.add_zone(role=role) == {"FINISHED"}
        zone = bpy.context.view_layer.objects.active
        assert zone.name == name and zone.stress_sculpt.role == role
        assert zone.empty_display_type == "SPHERE"
    with pytest.raises(TypeError):
        bpy.ops.stress_sculpt.add_zone(role="SUPPORT")


def test_set_role_skips_domain(scene):
    domain = box("Domain")
    other = box("Other", location=(3.0, 0.0, 0.0))
    scene.stress_sculpt.domain = domain
    marker = bpy.data.objects.new("Marker", None)
    marker.empty_display_type = "PLAIN_AXES"
    scene.collection.objects.link(marker)
    for obj in (domain, other, marker):
        obj.select_set(True)
    assert bpy.ops.stress_sculpt.set_role(role="HOLLOW") == {"FINISHED"}
    assert domain.stress_sculpt.role == "NONE"
    assert other.stress_sculpt.role == "HOLLOW" and marker.stress_sculpt.role == "HOLLOW"
    assert marker.empty_display_type == "SPHERE"  # a region shape for zones
    assert bpy.ops.stress_sculpt.set_role(role="LOAD") == {"FINISHED"}
    assert marker.empty_display_type == "SINGLE_ARROW"
    for obj in (domain, other, marker):
        obj.select_set(False)
    with pytest.raises(RuntimeError, match="Select"):
        bpy.ops.stress_sculpt.set_role(role="SUPPORT")


def test_select_role(scene):
    bridge(resolution=16)
    assert bpy.ops.stress_sculpt.select_role(role="SUPPORT") == {"FINISHED"}
    selected = _selected(scene)
    assert len(selected) == 2 and all(o.stress_sculpt.role == "SUPPORT" for o in selected)
    assert bpy.ops.stress_sculpt.select_role(role="LOAD", extend=True) == {"FINISHED"}
    assert len(_selected(scene)) == 3


def test_set_domain(scene):
    mesh_obj = box("Mine")
    mesh_obj.stress_sculpt.role = "THICKEN"
    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.stress_sculpt.set_domain() == {"FINISHED"}
    assert scene.stress_sculpt.domain == mesh_obj and mesh_obj.stress_sculpt.role == "NONE"
    bpy.context.view_layer.objects.active = None
    assert not bpy.ops.stress_sculpt.set_domain.poll()


@pytest.mark.parametrize("preset", CLASSIC_PRESETS)
def test_classic_setups_are_valid(scene, preset):
    require("fem")
    settings = bridge(resolution=16, preset=preset)
    domain = settings.domain
    assert domain is not None and domain.name == "SS_Domain" and domain.type == "MESH"
    roles = [o.stress_sculpt.role for o in scene_io.role_objects(bpy.context)]
    assert roles.count("SUPPORT") >= 1 and roles.count("LOAD") >= 1
    problem, info = scene_io.gather_problem(bpy.context)
    assert not info["warnings"]
    assert problem.domain.sum() == problem.grid.n_elems  # box domains fill the grid
    if preset == "BRIDGE":
        lo = np.array(domain.matrix_world)[:3, 3] - 0.5 * np.array(domain.dimensions)
        assert lo == pytest.approx([-2.0, -0.5, 0.0])
        assert tuple(domain.dimensions) == pytest.approx((4.0, 1.0, 1.5))


def test_classic_setup_fits_a_user_domain(scene):
    require("fem")
    mine = box("Mine", size=(6.0, 2.0, 2.0), location=(10.0, 0.0, 1.0))
    classic_setup("BRIDGE", resolution=18, domain=mine)
    assert scene.stress_sculpt.domain == mine and "SS_Domain" not in bpy.data.objects
    problem, info = scene_io.gather_problem(bpy.context)
    assert not info["warnings"]
    assert problem.forces.reshape(-1, 3).sum(axis=0)[2] == pytest.approx(-5000.0)


def test_gather_problem_domain_override(scene):
    require("fem")
    settings = bridge(resolution=16)
    block = box("Block", size=(4.0, 1.0, 1.5), location=(0.0, 0.0, 0.75))
    block.stress_sculpt.role = "SUPPORT"  # the override object never acts as a helper
    problem, info = scene_io.gather_problem(bpy.context, domain=block)
    assert info["domain_name"] == "Block" and settings.domain.name == "SS_Domain"
    assert len(info["supports"]) == 2


def test_cancel_poll_and_clear_results_without_results(scene):
    assert not bpy.ops.stress_sculpt.cancel.poll()
    assert bpy.ops.stress_sculpt.clear_results() == {"FINISHED"}


def test_settings_json_round_trip_keeps_material(scene):
    settings = scene.stress_sculpt
    domain = box("Domain")
    settings.domain = domain
    settings.material_preset = "CUSTOM"
    settings.youngs_modulus = 12.5
    settings.poisson_ratio = 0.21
    settings.resolution = 33
    settings.symmetry_y = True
    data = json.loads(json.dumps(operators.settings_to_dict(settings), sort_keys=True))
    assert data["domain"] == "Domain" and data["material_preset"] == "CUSTOM"
    assert not any(key.startswith("last_") for key in data) and "variants" not in data
    settings.material_preset = "STEEL"
    settings.resolution = 40
    settings.symmetry_y = False
    settings.domain = None
    skipped = operators.apply_settings_dict(settings, dict(data, unknown_key=1))
    assert skipped == ["unknown_key"]
    assert settings.material_preset == "CUSTOM"
    assert settings.youngs_modulus == pytest.approx(
        12.5
    ) and settings.poisson_ratio == pytest.approx(0.21)
    assert settings.resolution == 33 and settings.symmetry_y and settings.domain == domain
    # a preset restores its own E and nu even when the JSON lists them first
    steel = dict(data, material_preset="STEEL", youngs_modulus=200.0, poisson_ratio=0.3)
    operators.apply_settings_dict(settings, {k: steel[k] for k in sorted(steel, reverse=True)})
    assert settings.material_preset == "STEEL" and settings.youngs_modulus == pytest.approx(200.0)
