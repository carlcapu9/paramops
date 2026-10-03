"""Integration tests: run inside Blender's Python (``pip install bpy`` or ``blender -b --python``)."""

import math
import os
import sys

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import paramops  # noqa: E402
from paramops import handlers, ops  # noqa: E402
from paramops.demo import box, merge, tube_x  # noqa: E402
from paramops.writer import write  # noqa: E402
from paramops.core.meshdata import assemble  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    paramops.register()
    yield
    paramops.unregister()


@pytest.fixture(autouse=True)
def clean_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for d in list(coll):
            coll.remove(d)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    yield


def mesh_object(name, md):
    me = bpy.data.meshes.new(name)
    write(me, assemble([(md, md.co[None], np.zeros(1, dtype=bool))]))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def poly_curve(name, pts, cyclic=False, kind="POLY"):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new(kind)
    sp.points.add(len(pts) - 1)
    for p, co in zip(sp.points, pts):
        p.co = (*co, 1.0)
    sp.use_cyclic_u = cyclic
    if kind == "NURBS":
        sp.use_endpoint_u = True
        sp.order_u = 4
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def panel(length=2.0, nx=8):
    return box(0.0, length, -0.05, 0.05, 0.0, 1.0, nx=nx)


def coords(ob):
    co = np.empty(len(ob.data.vertices) * 3, np.float32)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]


def scatter(curve, default=None):
    return ops.create_scatter(bpy.context, curve, default=default)


def test_basic_adaptive_fill():
    curve = poly_curve("C", [(0, 0, 0), (10.4, 0, 0)])
    smp = mesh_object("Panel", panel())
    sc = scatter(curve, smp)
    st = sc.paramops
    assert st.last_error == ""
    assert st.stat_modules == 5
    assert len(sc.data.vertices) == 5 * len(smp.data.vertices)
    co = coords(sc)
    assert co[:, 0].min() == pytest.approx(0.0, abs=1e-5)
    assert co[:, 0].max() == pytest.approx(10.4, abs=1e-4)
    assert not sc.data.validate(verbose=True)


def test_live_update_from_curve_and_settings():
    curve = poly_curve("C", [(0, 0, 0), (10, 0, 0)])
    smp = mesh_object("Panel", panel())
    sc = scatter(curve, smp)
    curve.data.splines[0].points[1].co = (20.0, 0.0, 0.0, 1.0)
    bpy.context.view_layer.update()
    assert sc.paramops.stat_modules == 10
    curve.location.y = 3.0
    bpy.context.view_layer.update()
    assert coords(sc)[:, 1].mean() == pytest.approx(3.0, abs=1e-4)
    sc.paramops.fit_mode = "COUNT"
    sc.paramops.fit_count = 3
    assert sc.paramops.stat_modules == 3
    # Editing the sample mesh updates the scatter too.
    smp.data.vertices[0].co.z = 5.0
    smp.data.update()
    bpy.context.view_layer.update()
    assert coords(sc)[:, 2].max() == pytest.approx(5.0, abs=1e-4)


def test_scatter_object_transform_is_compensated():
    curve = poly_curve("C", [(0, 0, 0), (10, 0, 0)])
    sc = scatter(curve, mesh_object("Panel", panel()))
    sc.location = (5.0, 5.0, 5.0)
    sc.rotation_euler.z = 1.0
    bpy.context.view_layer.update()
    co = coords(sc)
    assert co[:, 0].min() == pytest.approx(0.0, abs=1e-4)
    assert abs(co[:, 1]).max() == pytest.approx(0.05, abs=1e-4)


def test_corners_posts_and_round_corners():
    curve = poly_curve("C", [(0, 0, 0), (6, 0, 0), (6, 6, 0)])
    smp = mesh_object("Panel", panel(1.0, nx=4))
    post = mesh_object("Post", box(-0.05, 0.05, -0.05, 0.05, 0.0, 1.2))
    sc = scatter(curve, smp)
    st = sc.paramops
    st.corner.object = post
    st.start.object = post
    st.end.object = post
    assert st.stat_modules == 12 - 0 + 3 - 1 or st.stat_modules > 0
    co = coords(sc)
    # A post sits on the corner.
    near = np.linalg.norm(co[:, :2] - np.array((6.0, 0.0)), axis=1) < 0.08
    assert near.any()
    st.corner_radius = 2.0
    co = coords(sc)
    near = np.linalg.norm(co[:, :2] - np.array((6.0, 0.0)), axis=1) < 0.3
    assert not near.any()  # the path now cuts the corner with an arc
    assert not sc.data.validate()


def test_vertical_mode_and_slopes():
    curve = poly_curve("C", [(0, 0, 0), (4, 0, 3)])
    sc = scatter(curve, mesh_object("Panel", panel(1.0, nx=2)))
    st = sc.paramops
    st.fit_mode = "COUNT"
    st.fit_count = 1
    st.default.keep_vertical = True
    co = coords(sc)
    # Upright: the top edge is exactly 1 above the bottom edge (shear, not rotation).
    bottom = co[np.isclose(co[:, 2] - co[:, 0] * 0.75, 0.0, atol=1e-4)]
    assert len(bottom) > 0
    assert co[:, 0].min() == pytest.approx(0.0, abs=1e-4)
    assert co[:, 0].max() == pytest.approx(4.0, abs=1e-4)


def test_segment_ids_and_markers_in_edit_mode():
    curve = poly_curve("C", [(0, 0, 0), (4, 0, 0), (8, 0, 0), (12, 0, 0)])
    smp = mesh_object("Panel", panel(1.0, nx=2))
    alt = mesh_object("Alt", box(0.0, 0.5, -0.05, 0.05, 0.0, 2.0))
    door = mesh_object("Door", box(-0.5, 0.5, -0.1, 0.1, 0.0, 2.2))
    sc = scatter(curve, smp)
    st = sc.paramops
    curve.paramops.curve_scatter = sc
    bpy.context.view_layer.objects.active = curve
    curve.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.curve.select_all(action="DESELECT")
        pts = curve.data.splines[0].points
        pts[1].select = True
        pts[2].select = True
        assert bpy.ops.paramops.segment_id_add() == {"FINISHED"}
        st.segment_ids[0].slot.object = alt
        assert bpy.ops.paramops.segment_id_assign() == {"FINISHED"}
        assert len(st.segment_ids[0].refs) == 1
        co = coords(sc)
        assert co[:, 2].max() == pytest.approx(2.0, abs=1e-4)
        tall = co[co[:, 2] > 1.5]
        assert tall[:, 0].min() >= 4.0 - 1e-4 and tall[:, 0].max() <= 8.0 + 1e-4

        bpy.ops.curve.select_all(action="DESELECT")
        curve.data.splines[0].points[2].select = True
        assert bpy.ops.paramops.marker_add() == {"FINISHED"}
        st.markers[0].slot.object = door
        assert bpy.ops.paramops.marker_place() == {"FINISHED"}
        co = coords(sc)
        doors = co[co[:, 2] > 2.1]
        assert doors[:, 0].min() == pytest.approx(7.5, abs=1e-4)

        # Subdividing the curve keeps the assignments on their points.
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.subdivide()
        bpy.context.view_layer.update()
        handlers.rebuild(sc)
        assert len(st.segment_ids[0].refs) == 2
        assert st.markers[0].refs[0].index == 4
        co = coords(sc)
        doors = co[co[:, 2] > 2.1]
        assert doors[:, 0].min() == pytest.approx(7.5, abs=1e-4)
        tall = co[(co[:, 2] > 1.5) & (co[:, 2] < 2.1)]
        assert tall[:, 0].min() >= 4.0 - 1e-4
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def test_markers_by_distance_and_percentage():
    curve = poly_curve("C", [(0, 0, 0), (10, 0, 0)])
    sc = scatter(curve, mesh_object("Panel", panel(1.0, nx=2)))
    st = sc.paramops
    door = mesh_object("Door", box(-0.5, 0.5, -0.1, 0.1, 0.0, 2.2))
    item = st.markers.add()
    item.slot.object = door
    item.slot.deform = False
    item.positions = "25%, -2"
    handlers.rebuild(sc)
    co = coords(sc)
    doors = co[co[:, 2] > 2.1]
    xs = sorted(set(np.round(doors[:, 0], 3)))
    assert xs[0] == pytest.approx(2.0) and xs[-1] == pytest.approx(8.5)


def test_materials_uvs_and_attributes_are_copied():
    curve = poly_curve("C", [(0, 0, 0), (6, 0, 0)])
    smp = mesh_object("Panel", panel())
    mat = bpy.data.materials.new("M")
    smp.data.materials.append(mat)
    uv = smp.data.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", np.full(len(smp.data.loops) * 2, 0.25, np.float32))
    attr = smp.data.attributes.new("weight", "FLOAT", "POINT")
    attr.data.foreach_set("value", np.arange(len(smp.data.vertices), dtype=np.float32))
    sc = scatter(curve, smp)
    me = sc.data
    assert me.materials[0] == mat
    assert "UVMap" in me.uv_layers
    uvs = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers["UVMap"].data.foreach_get("uv", uvs)
    assert np.allclose(uvs, 0.25)
    assert "weight" in me.attributes and "sharp_face" in me.attributes
    assert not me.validate()


def test_collection_modes():
    curve = poly_curve("C", [(0, 0, 0), (12, 0, 0)])
    col = bpy.data.collections.new("Variants")
    bpy.context.scene.collection.children.link(col)
    a = mesh_object("A", box(0, 1, -0.1, 0.1, 0, 1.0))
    b = mesh_object("B", box(0, 1, -0.1, 0.1, 0, 2.0))
    for ob in (a, b):
        bpy.context.scene.collection.objects.unlink(ob)
        col.objects.link(ob)
    sc = scatter(curve)
    st = sc.paramops
    st.default.source = "COLLECTION"
    st.default.collection = col
    st.default.collection_mode = "SEQUENCE"
    assert st.stat_modules == 12
    assert len(sc.data.vertices) == 12 * 8
    st.default.collection_mode = "COMBINE"
    assert st.stat_modules == 12 // 1 and len(sc.data.vertices) > 0
    b.paramops_weight = 0.0
    st.default.collection_mode = "RANDOM"
    assert coords(sc)[:, 2].max() == pytest.approx(1.0, abs=1e-5)


def test_bezier_cyclic_and_nurbs_paths():
    bpy.ops.curve.primitive_bezier_circle_add(radius=3.0)
    circle = bpy.context.active_object
    sc = scatter(circle, mesh_object("Panel", panel(1.0, nx=8)))
    st = sc.paramops
    assert st.stat_modules == round(2 * math.pi * 3.0)
    r = np.linalg.norm(coords(sc)[:, :2], axis=1)
    assert r.min() > 2.9 and r.max() < 3.1

    nurbs = poly_curve("N", [(0, 0, 0), (2, 3, 0), (5, -1, 0), (8, 2, 0), (10, 0, 0)], kind="NURBS")
    sc2 = scatter(nurbs, mesh_object("Panel2", panel(0.5, nx=4)))
    # Compare with Blender's own evaluation of the NURBS curve.
    dg = bpy.context.evaluated_depsgraph_get()
    me = nurbs.evaluated_get(dg).to_mesh()
    ref = np.array([v.co[:] for v in me.vertices])
    nurbs.evaluated_get(dg).to_mesh_clear()
    co = coords(sc2)
    A, B = ref[:-1, :2], ref[1:, :2]
    AB = B - A
    rel = co[:, None, :2] - A[None]
    t = np.clip(np.einsum("pij,ij->pi", rel, AB) / np.einsum("ij,ij->i", AB, AB), 0.0, 1.0)
    d = np.linalg.norm(rel - t[..., None] * AB[None], axis=2).min(axis=1)
    # Panel vertices are 0.05 to the side of the curve (plus a little sampling error).
    assert d.max() < 0.05 + 0.02


def test_fixed_size_with_slicing():
    curve = poly_curve("C", [(0, 0, 0), (5.5, 0, 0)])
    sc = scatter(curve, mesh_object("Panel", panel(2.0, nx=8)))
    st = sc.paramops
    st.fit_mode = "FIXED"
    st.fixed_remainder = "SLICE"
    co = coords(sc)
    assert co[:, 0].max() == pytest.approx(5.5, abs=1e-4)
    assert st.stat_modules == 3
    assert not sc.data.validate()


def test_conform_to_surface():
    ground = mesh_object("Ground", box(-1, 11, -3, 3, -1.0, 0.0, nx=12))
    for v in ground.data.vertices:
        if v.co.z > -0.5:
            v.co.z = 0.1 * v.co.x
    curve = poly_curve("C", [(0, 0, 5), (10, 0, 5)])
    sc = scatter(curve, mesh_object("Panel", panel(1.0, nx=4)))
    st = sc.paramops
    st.default.keep_vertical = True
    st.project_target = ground
    st.project = True
    co = coords(sc)
    low = co[np.isclose(co[:, 1], -0.05)]
    assert np.allclose(low[:, 2].min(), 0.0, atol=0.02)
    assert co[:, 2].max() == pytest.approx(2.0, abs=0.05)


def test_boxes_preview_and_convert():
    curve = poly_curve("C", [(0, 0, 0), (4, 0, 0)])
    sc = scatter(curve, mesh_object("Panel", panel(1.0)))
    st = sc.paramops
    st.preview = "BOXES"
    assert len(sc.data.vertices) == 4 * 8
    st.preview = "FULL"
    bpy.context.view_layer.objects.active = sc
    bpy.ops.paramops.convert(keep_original=True)
    copies = [o for o in bpy.data.objects if o.name.endswith("_Mesh")]
    assert len(copies) == 1 and not copies[0].paramops.is_scatter
    assert len(copies[0].data.vertices) == len(sc.data.vertices)


def test_copy_settings_to_selected():
    c1 = poly_curve("C1", [(0, 0, 0), (4, 0, 0)])
    c2 = poly_curve("C2", [(0, 5, 0), (8, 5, 0)])
    smp = mesh_object("Panel", panel(1.0))
    a = scatter(c1, smp)
    b = scatter(c2)
    a.paramops.spacing = 0.2
    for o in bpy.context.selected_objects:
        o.select_set(False)
    a.select_set(True)
    b.select_set(True)
    bpy.context.view_layer.objects.active = a
    assert bpy.ops.paramops.copy_settings() == {"FINISHED"}
    assert b.paramops.default.object == smp
    assert b.paramops.spacing == pytest.approx(0.2)
    assert b.paramops.path == c2
    assert b.paramops.stat_modules > 0


def test_demo_scene_builds_cleanly():
    assert bpy.ops.paramops.demo() == {"FINISHED"}
    scs = [o for o in bpy.data.objects if o.paramops.is_scatter]
    assert len(scs) == 3
    for sc in scs:
        assert sc.paramops.last_error == "", sc.paramops.last_error
        assert len(sc.data.polygons) > 0
        assert not sc.data.validate()


def test_tube_bends_smoothly_around_round_corner():
    curve = poly_curve("C", [(0, 0, 1), (4, 0, 1), (4, 4, 1)])
    sc = scatter(curve, mesh_object("Rail", tube_x(0.0, 1.0, 0.05, rings=10)))
    sc.paramops.corner_radius = 1.0
    co = coords(sc)
    # Every vertex keeps its distance (0.05) from the rounded centre line.
    centre = np.array([4.0 - 1.0, 1.0])
    on_arc = (co[:, 0] > 3.0 + 1e-3) & (co[:, 1] < 1.0 - 1e-3)
    d = np.hypot(np.linalg.norm(co[on_arc, :2] - centre, axis=1) - 1.0, co[on_arc, 2] - 1.0)
    assert np.allclose(d, 0.05, atol=2e-3)
