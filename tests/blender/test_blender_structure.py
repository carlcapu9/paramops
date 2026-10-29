"""The Structure panel: weights / supports at the selection, Generate Structure, Check.

Tiny setups and low resolutions keep the suite fast; ORGANIC needs ``core/topopt.py``
and RIBS the surface mode of ``core/streamlines.py`` (skipped while missing).
"""

import json
import math

import numpy as np
import pytest

import bpy
import bmesh
from bhelpers import box, empty, mesh_attribute_values, require

from stress_sculpt import build, jobs, operators, scene_io


def _ribs_available():
    try:
        from stress_sculpt.core import streamlines
    except ImportError:
        return False
    return operators.ribs_supported(streamlines)


def _topopt_available():
    try:
        import stress_sculpt.core.topopt  # noqa: F401
    except ImportError:
        return False
    return True


def require_style(style, mode="SUPPORT", rib_source="EVOLVED"):
    require("fem", "mesher", "sculpt", "forms")
    if style == "ORGANIC" or (style == "RIBS" and not (mode == "CARVE" and rib_source == "SURFACE")):
        if not _topopt_available():
            pytest.skip("core/topopt.py not available yet")
    if style == "RIBS":
        require("streamlines")  # surface tracing: the core's, or the add-on's own fallback


def _structure(scene):
    obj = build.find_result(scene, build.OBJ_STRUCTURE)
    assert obj is not None, "SS_Structure was not built"
    return obj


def _world_verts(obj):
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    matrix = np.array(obj.matrix_world)
    return co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]


def _loads(scene):
    return sorted(
        (o for o in scene.objects if o.stress_sculpt.role == "LOAD"), key=lambda o: o.name
    )


def _supports(scene):
    return [o for o in scene.objects if o.stress_sculpt.role == "SUPPORT"]


def _select_verts(obj, predicate, mode="VERT"):
    """Edit Mode on ``obj`` with the vertices (or faces) matching ``predicate`` selected."""
    if bpy.context.mode != "OBJECT" and obj.mode != "EDIT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for other in bpy.context.view_layer.objects:
        if other != obj and other.select_get():
            other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if obj.mode != "EDIT":
        bpy.ops.object.mode_set(mode="EDIT")
    tool = bpy.context.tool_settings
    tool.mesh_select_mode = (mode == "VERT", False, mode == "FACE")
    bm = bmesh.from_edit_mesh(obj.data)
    for elem in list(bm.verts) + list(bm.edges) + list(bm.faces):
        elem.select = False
    if mode == "FACE":
        for face in bm.faces:
            if predicate(face.calc_center_median(), face):
                face.select = True
    else:
        for vert in bm.verts:
            if predicate(vert.co, vert):
                vert.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data)


def _floating_cube(size=1.0, bottom=1.5, name="Cube"):
    return box(name, size=(size, size, size), location=(0.0, 0.0, bottom + 0.5 * size))


# ------------------------------------------------------------------ registration
def test_structure_operators_registered(scene):
    for name in (
        "set_target",
        "add_load_at_selection",
        "add_support_at_selection",
        "remove_role",
        "generate",
        "verify",
    ):
        assert hasattr(bpy.types, f"STRESS_SCULPT_OT_{name}"), name


def test_set_target(scene):
    cube = _floating_cube()
    bpy.context.view_layer.objects.active = cube
    assert bpy.ops.stress_sculpt.set_target() == {"FINISHED"}
    assert scene.stress_sculpt.target == cube
    helper = empty("Helper", "LOAD", "SINGLE_ARROW")
    bpy.context.view_layer.objects.active = helper
    assert not bpy.ops.stress_sculpt.set_target.poll()  # not a mesh
    marked = box("Marked", location=(3.0, 0.0, 0.0))
    marked.stress_sculpt.role = "KEEP_SOLID"
    bpy.context.view_layer.objects.active = marked
    assert not bpy.ops.stress_sculpt.set_target.poll()  # a role object


# --------------------------------------------------------------- weights at selection
def test_weights_on_selected_vertices(scene):
    require("forms")
    cube = _floating_cube(size=2.0, bottom=3.0)
    bpy.context.view_layer.update()
    _select_verts(cube, lambda co, _v: co.z < 0.0 and co.x > 0.0 or (co.z < 0 and co.y > 0.5))
    assert bpy.ops.stress_sculpt.add_load_at_selection(force=1500.0) == {"FINISHED"}
    assert bpy.context.mode == "EDIT_MESH"  # stays in Edit Mode
    loads = _loads(scene)
    assert len(loads) == 3
    bpy.context.view_layer.update()
    setup = build.find_child_collection(scene.collection, build.SETUP_COLLECTION)
    diagonal = math.sqrt(3.0 * 4.0)
    for load in loads:
        assert load.name.startswith("SS_Load") and load.type == "EMPTY"
        assert load.parent == cube and load.name in setup.objects
        assert load.empty_display_type == "SINGLE_ARROW"
        assert load.empty_display_size == pytest.approx(0.08 * diagonal, rel=1e-3)
        assert load.stress_sculpt.magnitude == pytest.approx(1500.0)
        assert load.color[0] > load.color[2]  # red
        world = np.array(load.matrix_world)
        assert world[2, 3] == pytest.approx(3.0, abs=1e-5)  # on the bottom face
        direction = world[:3, 2] / np.linalg.norm(world[:3, 2])
        assert direction == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)
    assert scene.stress_sculpt.target == cube
    # the weights follow the object (parented, world transform kept at creation)
    bpy.ops.object.mode_set(mode="OBJECT")
    before = np.array([tuple(o.matrix_world.translation) for o in loads])
    cube.location.x += 1.0
    bpy.context.view_layer.update()
    after = np.array([tuple(o.matrix_world.translation) for o in loads])
    assert after - before == pytest.approx(np.tile([1.0, 0.0, 0.0], (3, 1)), abs=1e-5)
    # repeated calls add more (Object Mode uses the stored selection)
    bpy.context.view_layer.objects.active = cube
    assert bpy.ops.stress_sculpt.add_load_at_selection() == {"FINISHED"}
    assert len(_loads(scene)) == 6


def test_weights_total_on_faces_and_into_surface(scene):
    require("forms")
    # a 4 x 1 x 1 box: its top face is split 4 x 1, its long side faces too
    mesh = operators.grid_box_mesh("Slab", (4.0, 1.0, 1.0), (4, 1, 1))
    slab = bpy.data.objects.new("Slab", mesh)
    slab.location = (0.0, 0.0, 2.0)
    scene.collection.objects.link(slab)
    bpy.context.view_layer.update()
    # the two top faces at x > 0 plus the whole -X end face (area 1 each)
    _select_verts(
        slab,
        lambda c, f: (c.z > 0.49 and c.x > 0.0) or c.x < -1.99,
        mode="FACE",
    )
    assert bpy.ops.stress_sculpt.add_load_at_selection(
        force=900.0, distribute="TOTAL", direction="INTO_SURFACE"
    ) == {"FINISHED"}
    loads = _loads(scene)
    assert len(loads) == 3
    bpy.context.view_layer.update()
    total = sum(o.stress_sculpt.magnitude for o in loads)
    assert total == pytest.approx(900.0)
    by_x = {round(o.matrix_world.translation.x, 3): o for o in loads}
    assert set(by_x) == {0.5, 1.5, -2.0}  # face centres
    end = np.array(by_x[-2.0].matrix_world)[:3, 2]
    assert end / np.linalg.norm(end) == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)  # into -X face
    top = np.array(by_x[0.5].matrix_world)[:3, 2]
    assert top / np.linalg.norm(top) == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)
    assert by_x[0.5].stress_sculpt.magnitude == pytest.approx(300.0)  # equal areas


def test_weights_area_weighted_and_clustered(scene):
    require("forms")
    mesh = operators.grid_box_mesh("Plate", (3.0, 1.0, 0.2), (3, 1, 1))
    plate = bpy.data.objects.new("Plate", mesh)
    plate.scale = (1.0, 2.0, 1.0)  # world areas: top faces 2 m^2, the end face 0.4 m^2
    scene.collection.objects.link(plate)
    bpy.context.view_layer.update()
    _select_verts(plate, lambda c, f: c.z > 0.09 or c.x > 1.49, mode="FACE")
    assert bpy.ops.stress_sculpt.add_load_at_selection(force=640.0, distribute="TOTAL") == {
        "FINISHED"
    }
    mags = sorted(o.stress_sculpt.magnitude for o in _loads(scene))
    assert mags == pytest.approx([40.0, 200.0, 200.0, 200.0])  # 0.4 : 2 : 2 : 2
    bpy.ops.stress_sculpt.remove_role(role="LOAD")
    # many vertices merge into at most max_points weights, force preserved
    bpy.ops.object.mode_set(mode="OBJECT")
    sphere = bpy.data.objects.new("Ball", bpy.data.meshes.new("Ball"))
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
    bm.to_mesh(sphere.data)
    bm.free()
    sphere.location = (5.0, 0.0, 3.0)
    scene.collection.objects.link(sphere)
    bpy.context.view_layer.objects.active = sphere
    bpy.context.view_layer.update()
    _select_verts(sphere, lambda co, _v: co.z < -0.2)
    selected = sum(1 for v in sphere.data.vertices if v.co.z < -0.2)
    assert selected > 30
    assert bpy.ops.stress_sculpt.add_load_at_selection(force=10.0, max_points=12) == {"FINISHED"}
    loads = _loads(scene)
    assert len(loads) == 12
    assert sum(o.stress_sculpt.magnitude for o in loads) == pytest.approx(10.0 * selected)


def test_support_at_selection(scene):
    cube = _floating_cube(size=2.0, bottom=0.0)
    scene.stress_sculpt.structure_mode = "CARVE"
    scene.stress_sculpt.resolution = 20
    bpy.context.view_layer.update()
    _select_verts(cube, lambda co, _v: co.z < 0.0)
    assert bpy.ops.stress_sculpt.add_support_at_selection() == {"FINISHED"}
    supports = _supports(scene)
    assert len(supports) == 4
    diagonal = math.sqrt(12.0)
    for support in supports:
        assert support.empty_display_type == "SPHERE" and support.parent == cube
        assert support.empty_display_size == pytest.approx(max(1.5 * 2.0 / 20, 0.03 * diagonal))
        s = support.stress_sculpt
        assert s.fix_x and s.fix_y and s.fix_z
        assert support.color[2] > support.color[0]  # blue
    assert scene.stress_sculpt.target == cube  # set in Carve mode


def test_selection_errors(scene):
    cube = _floating_cube()
    _select_verts(cube, lambda co, _v: False)
    with pytest.raises(RuntimeError, match="Nothing selected"):
        bpy.ops.stress_sculpt.add_load_at_selection()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = None
    assert not bpy.ops.stress_sculpt.add_load_at_selection.poll()


def test_remove_role(scene):
    cube = _floating_cube()
    bpy.context.view_layer.update()
    _select_verts(cube, lambda co, _v: co.z < 0.0)
    bpy.ops.stress_sculpt.add_load_at_selection()
    bpy.ops.object.mode_set(mode="OBJECT")
    marked = box("Zone", location=(4.0, 0.0, 0.0))
    marked.stress_sculpt.role = "LOAD"
    assert len(_loads(scene)) == 5
    assert bpy.ops.stress_sculpt.remove_role(role="LOAD") == {"FINISHED"}
    assert _loads(scene) == []
    assert "Zone" in bpy.data.objects and marked.stress_sculpt.role == "NONE"
    assert "Cube" in bpy.data.objects


# ------------------------------------------------------------------- tiny setups
def _support_setup(settings_overrides=None):
    """A 1 m cube floating at 1.5 m with weights on its 4 bottom corners."""
    scene = bpy.context.scene
    settings = scene.stress_sculpt
    cube = _floating_cube(size=1.0, bottom=1.5)
    bpy.context.view_layer.update()
    corners = [(x, y, 1.5) for x in (-0.5, 0.5) for y in (-0.5, 0.5)]
    operators.add_role_empties(
        bpy.context, cube, "LOAD", corners, np.tile([0, 0, -1.0], (4, 1)), [500.0] * 4, 0.1
    )
    settings.target = cube
    settings.structure_mode = "SUPPORT"
    _fast(settings)
    for key, value in (settings_overrides or {}).items():
        setattr(settings, key, value)
    return settings


def _carve_setup(settings_overrides=None):
    """A 2 x 0.6 x 1 block on two end supports, two weights on top."""
    scene = bpy.context.scene
    settings = scene.stress_sculpt
    block = box("Block", size=(2.0, 0.6, 1.0), location=(0.0, 0.0, 0.5))
    bpy.context.view_layer.update()
    operators.add_role_empties(
        bpy.context, block, "SUPPORT", [(-0.95, 0.0, 0.0), (0.95, 0.0, 0.0)], size=0.2
    )
    operators.add_role_empties(
        bpy.context,
        block,
        "LOAD",
        [(-0.3, 0.0, 1.0), (0.3, 0.0, 1.0)],
        np.tile([0, 0, -1.0], (2, 1)),
        [800.0, 800.0],
        0.1,
    )
    settings.target = block
    settings.structure_mode = "CARVE"
    _fast(settings)
    for key, value in (settings_overrides or {}).items():
        setattr(settings, key, value)
    return settings


def _fast(settings):
    settings.resolution = 14
    settings.max_iterations = 12
    settings.tree_detail = 1.2
    settings.smooth_iterations = 2
    settings.sculpt_detail = 1.0
    settings.rib_spacing = 2.0


COMBOS = [(mode, style) for mode in ("SUPPORT", "CARVE") for style in ("ORGANIC", "BRANCHING", "RIBS")]


@pytest.mark.parametrize("mode,style", COMBOS)
def test_generate_every_mode_and_style(scene, mode, style):
    require_style(style, mode)
    settings = (_support_setup if mode == "SUPPORT" else _carve_setup)({"structure_style": style})
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    obj = _structure(scene)
    mesh = obj.data
    assert obj.type == "MESH" and len(mesh.polygons) > 50
    assert obj.material_slots[0].material.name == build.MAT_STRUCTURE
    assert mesh.color_attributes.active_color.name == "Stress"
    assert "stress" in mesh.attributes
    assert obj.name in build.results_collection(scene, create=False).objects
    assert settings.last_kind == style and settings.last_volume > 0.0
    assert obj[operators.PROP_MODE] == mode and obj[operators.PROP_STYLE] == style
    verts = _world_verts(obj)
    target = settings.target
    t_min = np.array(target.matrix_world)[:3, 3] - 0.5 * np.array(target.dimensions)
    t_max = np.array(target.matrix_world)[:3, 3] + 0.5 * np.array(target.dimensions)
    h = float(np.max(target.dimensions)) / settings.resolution
    if mode == "SUPPORT":
        # stands on the ground, reaches up to the object's underside and stops there
        assert verts[:, 2].min() == pytest.approx(0.0, abs=0.35 * h)
        assert verts[:, 2].max() == pytest.approx(t_min[2], abs=0.6 * h)
    else:
        # stays inside the block
        assert np.all(verts >= t_min - 0.6 * h) and np.all(verts <= t_max + 0.6 * h)
    # no floating crumbs: at most one part per footing / support
    labels = operators.face_components(
        len(mesh.vertices), np.array([tuple(p.vertices) for p in mesh.polygons])
    )
    footings = np.asarray(list(obj.get(operators.PROP_FOOTINGS, [0, 0, 0]))).reshape(-1, 3)
    assert np.unique(labels).size <= max(1, footings.shape[0])


def test_generate_errors(scene):
    require("forms", "sculpt", "mesher")
    settings = scene.stress_sculpt
    settings.structure_style = "BRANCHING"
    with pytest.raises(RuntimeError, match="No object"):
        bpy.ops.stress_sculpt.generate()
    cube = _floating_cube(size=1.0, bottom=1.5)
    settings.target = cube
    with pytest.raises(RuntimeError, match="No weights"):
        bpy.ops.stress_sculpt.generate()
    operators.add_role_empties(
        bpy.context, cube, "LOAD", [(0.0, 0.0, 1.5)], [(0.0, 0.0, -1.0)], [100.0], 0.1
    )
    settings.ground_height = 2.5
    with pytest.raises(RuntimeError, match="below the ground"):
        bpy.ops.stress_sculpt.generate()
    settings.ground_height = 0.0
    settings.structure_mode = "CARVE"
    with pytest.raises(RuntimeError, match="needs supports"):
        bpy.ops.stress_sculpt.generate()


def test_branching_support_contacts_and_footings(scene):
    require_style("BRANCHING")
    settings = _support_setup({"structure_style": "BRANCHING", "tree_detail": 1.5})
    # a weight on TOP of the cube is carried from its underside
    cube = settings.target
    operators.add_role_empties(
        bpy.context, cube, "LOAD", [(0.0, 0.0, 2.5)], [(0.0, 0.0, -1.0)], [500.0], 0.1
    )
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    obj = _structure(scene)
    verts = _world_verts(obj)
    assert verts[:, 2].max() < 1.5 + 0.05  # nothing inside the cube
    footings = np.asarray(list(obj[operators.PROP_FOOTINGS])).reshape(-1, 3)
    assert footings.shape == (1, 3) and footings[0, 2] == pytest.approx(0.0, abs=1e-6)
    cached = jobs.get_cache(scene).structure
    assert cached["kind"] == "BRANCHING" and cached["mode"] == "SUPPORT"
    # flat base on the ground: many vertices exactly at z = 0
    assert np.sum(np.abs(verts[:, 2]) < 1e-5) >= 8
    # rerun: same object, replaced data
    mesh_name = obj.data.name
    settings.tree_thickness = 0.07
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    assert _structure(scene) == obj and mesh_name not in bpy.data.meshes


def test_branching_with_explicit_footings(scene):
    require_style("BRANCHING")
    settings = _support_setup({"structure_style": "BRANCHING"})
    ground = box("Ground", size=(4.0, 4.0, 0.2), location=(0.0, 0.0, -0.1))
    bpy.context.view_layer.update()
    operators.add_role_empties(
        bpy.context, ground, "SUPPORT", [(-0.8, 0.0, 0.0), (0.8, 0.0, 0.0)], size=0.1
    )
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    obj = _structure(scene)
    footings = np.asarray(list(obj[operators.PROP_FOOTINGS])).reshape(-1, 3)
    assert sorted(footings[:, 0]) == pytest.approx([-0.8, 0.8])
    # supports above the weights are rejected with a clear message
    for support in _supports(scene):
        support.location.z += 3.0
    bpy.context.view_layer.update()
    with pytest.raises(RuntimeError, match="not below the weights"):
        bpy.ops.stress_sculpt.generate()


def test_generate_hides_other_results_and_clear(scene):
    require_style("BRANCHING")
    _support_setup({"structure_style": "BRANCHING"})
    fake = build.mesh_from_faces("x", np.eye(3), [[0, 1, 2]])
    other = build.assign_result(scene, build.OBJ_EVOLVED, fake, build.heatmap_material())
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    assert not other.visible_get()
    mesh_name = _structure(scene).data.name
    assert bpy.ops.stress_sculpt.clear_results() == {"FINISHED"}
    assert build.find_result(scene, build.OBJ_STRUCTURE) is None
    assert mesh_name not in bpy.data.meshes and jobs.get_cache(scene).structure is None


def test_carve_branching_stays_in_block_and_wires_it(scene):
    require_style("BRANCHING", "CARVE")
    settings = _carve_setup({"structure_style": "BRANCHING"})
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    assert settings.target.display_type == "WIRE"
    assert bpy.ops.stress_sculpt.clear_results() == {"FINISHED"}
    assert settings.target.display_type == "TEXTURED"


# ------------------------------------------------------------------------- verify
def test_verify_branching(scene):
    require_style("BRANCHING")
    settings = _support_setup({"structure_style": "BRANCHING", "resolution": 16})
    with pytest.raises(RuntimeError, match="Generate"):
        bpy.ops.stress_sculpt.verify()
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    obj = _structure(scene)
    assert bpy.ops.stress_sculpt.verify() == {"FINISHED"}
    assert obj.material_slots[0].material.name == build.MAT_HEATMAP
    values = mesh_attribute_values(obj.data, "stress", "value")
    assert values.shape[0] == len(obj.data.vertices) and values.max() > 0.0
    assert settings.last_max_stress > 0.0 and settings.last_max_displacement > 0.0
    assert settings.last_compliance > 0.0 and settings.last_kind == "BRANCHING"
    # generating again restores the structure material
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    assert obj.material_slots[0].material.name == build.MAT_STRUCTURE


def test_verify_carve(scene):
    require_style("BRANCHING", "CARVE")
    settings = _carve_setup({"structure_style": "BRANCHING", "resolution": 16})
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    assert bpy.ops.stress_sculpt.verify() == {"FINISHED"}
    assert settings.last_max_stress > 0.0


# ---------------------------------------------------------------- variants / demos
def test_variant_captures_structure(scene):
    require_style("BRANCHING")
    settings = _support_setup({"structure_style": "BRANCHING"})
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    assert bpy.ops.stress_sculpt.variant_save() == {"FINISHED"}
    variant = settings.variants[0]
    assert variant.kind == "BRANCHING" and variant.collection.name.endswith("Branching")
    copies = list(variant.collection.objects)
    assert [build.base_name(o.name) for o in copies] == ["V1 SS_Structure"]
    assert copies[0].data != _structure(scene).data
    saved = json.loads(variant.settings_json)
    assert saved["structure_style"] == "BRANCHING" and saved["target"] == settings.target.name
    settings.structure_style = "ORGANIC"
    assert bpy.ops.stress_sculpt.variant_restore() == {"FINISHED"}
    assert settings.structure_style == "BRANCHING"
    assert bpy.ops.stress_sculpt.variant_compare() == {"FINISHED"}


DEMOS = {
    "FLOATING_CUBE": ("SUPPORT", "BRANCHING", 8, 0),
    "CANOPY": ("SUPPORT", "ORGANIC", 25, 0),
    "PAVILION": ("SUPPORT", "RIBS", 12, 0),
    "BRIDGE": ("CARVE", "ORGANIC", 9, 4),
    "CARVED_CUBE": ("CARVE", "RIBS", 4, 4),
}


@pytest.mark.parametrize("preset", list(DEMOS))
def test_demo_presets_set_up_the_structure(scene, preset):
    mode, style, n_loads, n_supports = DEMOS[preset]
    assert bpy.ops.stress_sculpt.demo_scene(preset=preset) == {"FINISHED"}
    settings = scene.stress_sculpt
    target = settings.target
    assert target is not None and target.type == "MESH"
    assert settings.structure_mode == mode and settings.structure_style == style
    loads, supports = _loads(scene), _supports(scene)
    assert len(loads) == n_loads and len(supports) == n_supports
    assert all(o.parent == target for o in loads + supports)
    if mode == "CARVE":
        assert settings.domain == target
    # the setup is complete: the (fast) branching request gathers without errors
    require("forms", "sculpt", "mesher", "fem")
    settings.structure_style = "BRANCHING"
    request = operators.structure_request(bpy.context, settings)
    assert request["mode"] == mode and len(request["contacts"]) == n_loads


def test_demo_rerun_replaces_previous_demo(scene):
    bpy.ops.stress_sculpt.demo_scene(preset="FLOATING_CUBE")
    first = scene.stress_sculpt.target.name
    bpy.ops.stress_sculpt.demo_scene(preset="CANOPY")
    assert first not in bpy.data.objects
    assert len([o for o in scene.objects if o.type == "MESH"]) == 1
    assert len(_loads(scene)) == 25 and scene.stress_sculpt.footings == 4


def test_demo_keeps_user_objects(scene):
    mine = box("Mine", size=(1.0, 1.0, 1.0))
    helper = empty("MyLoad", "LOAD", "SINGLE_ARROW", location=(5.0, 0.0, 0.0))
    assert bpy.ops.stress_sculpt.demo_scene(preset="BRIDGE") == {"FINISHED"}
    assert "Mine" in bpy.data.objects and "MyLoad" in bpy.data.objects
    assert not helper.stress_sculpt.enabled  # not part of the example
    assert scene.stress_sculpt.target != mine


@pytest.mark.parametrize("preset", list(DEMOS))
def test_demo_presets_generate(scene, preset):
    mode, style, _n_loads, _n_supports = DEMOS[preset]
    settings = scene.stress_sculpt
    assert bpy.ops.stress_sculpt.demo_scene(preset=preset) == {"FINISHED"}
    require_style(style, mode, settings.rib_source)
    settings.resolution = 16
    settings.max_iterations = 10
    settings.tree_detail = 1.5
    settings.sculpt_detail = 1.0
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    assert len(_structure(scene).data.polygons) > 100
    assert settings.last_kind == style


def test_floating_cube_default_is_fast_and_clean(scene):
    require_style("BRANCHING")
    import time

    assert bpy.ops.stress_sculpt.demo_scene(preset="FLOATING_CUBE") == {"FINISHED"}
    start = time.monotonic()
    assert bpy.ops.stress_sculpt.generate() == {"FINISHED"}
    elapsed = time.monotonic() - start
    assert elapsed < 20.0  # target 10 s on 4 free cores
    obj = _structure(scene)
    verts = _world_verts(obj)
    assert verts[:, 2].min() == pytest.approx(0.0, abs=0.02)
    assert verts[:, 2].max() == pytest.approx(3.0, abs=0.02)  # flush under the cube
    # every weight is reached: a vertex of the flat capital right under it
    cube_min = np.array([-1.0, -1.0])
    for load in _loads(scene):
        p = np.array(load.matrix_world.translation)
        near = verts[np.abs(verts[:, 2] - 3.0) < 0.02]
        assert np.min(np.linalg.norm(near[:, :2] - p[:2], axis=1)) < 0.35
    # nothing sticks out beside the cube at the top
    top = verts[verts[:, 2] > 2.9]
    assert np.all(np.abs(top[:, :2]) <= 1.0 + 0.02)
    assert cube_min is not None
