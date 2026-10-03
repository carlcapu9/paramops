"""Integration tests of the CurveForge add-on inside Blender (bpy module)."""

import math
import os
import sys

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import curveforge  # noqa: E402
from curveforge import live, meshio, ops, output  # noqa: E402
from curveforge.engine.mesh import Assembler  # noqa: E402
from curveforge.templates import box  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    curveforge.register()
    yield
    curveforge.unregister()


@pytest.fixture(autouse=True)
def clean():
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                 bpy.data.node_groups, bpy.data.collections):
        for d in list(coll):
            coll.remove(d)
    yield


def mesh_obj(name, buf):
    asm = Assembler()
    asm.add(buf, buf.co[None])
    me = bpy.data.meshes.new(name)
    meshio.write_mesh(me, asm.build())
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def curve_obj(name, pts, cyclic=False, kind="POLY"):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new(kind)
    if kind == "BEZIER":
        sp.bezier_points.add(len(pts) - 1)
        for bp, co in zip(sp.bezier_points, pts):
            bp.co = co
            bp.handle_left_type = bp.handle_right_type = "VECTOR"
    else:
        sp.points.add(len(pts) - 1)
        for p, co in zip(sp.points, pts):
            p.co = (*co, 1.0)
    sp.use_cyclic_u = cyclic
    ob = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def panel(length=1.0, height=1.0, nx=4):
    return box(0.0, length, -0.05, 0.05, 0.0, height, nx=nx)


def coords(ob):
    live.flush()
    co = np.zeros(len(ob.data.vertices) * 3, np.float32)
    ob.data.vertices.foreach_get("co", co)
    m = np.array(ob.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def tall_modules(obj, threshold=1.5, count=6):
    """Indices k of the unit-length modules [k, k + 1] that have vertices above ``threshold``."""
    co = coords(obj)
    tall = co[co[:, 2] > threshold][:, 0]
    return [k for k in range(count) if np.any((tall > k + 0.1) & (tall < k + 0.9))]


def scatter(curve, segment=None):
    tree = ops.basic_style("Style", curve, segment)
    return ops.new_output(bpy.context, "Out", tree), tree


def node(tree, idname):
    return next(n for n in tree.nodes if n.bl_idname == idname)


def add(tree, idname, **props):
    n = tree.nodes.new(idname)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def test_basic_scatter_and_validity():
    c = curve_obj("C", [(0, 0, 0), (10.4, 0, 0)])
    s = mesh_obj("S", panel(2.0))
    obj, tree = scatter(c, s)
    st = obj.cf_scatter
    assert st.error == "" and st.stat_segments == 5
    co = coords(obj)
    assert co[:, 0].min() == pytest.approx(0.0, abs=1e-5) and co[:, 0].max() == pytest.approx(10.4, abs=1e-4)
    assert not obj.data.validate(verbose=True)


def test_live_updates_from_tree_curve_and_segment():
    c = curve_obj("C", [(0, 0, 0), (10, 0, 0)])
    s = mesh_obj("S", panel(2.0))
    obj, tree = scatter(c, s)
    gen = node(tree, "CF_NodeLinear")
    gen.fit = "COUNT"
    gen.count = 3
    assert obj.cf_scatter.stat_segments == 3
    gen.fit = "ROUND"
    c.data.splines[0].points[1].co = (20, 0, 0, 1)
    bpy.context.view_layer.update()
    assert obj.cf_scatter.stat_segments == 10
    s.data.vertices[0].co.z = 4.0
    s.data.update()
    bpy.context.view_layer.update()
    assert coords(obj)[:, 2].max() == pytest.approx(4.0, abs=1e-4)
    gen.inputs["Spacing"].default_value = 0.5
    assert obj.cf_scatter.stat_segments == 8


def test_generator_inputs_start_end_corner_evenly_marker():
    c = curve_obj("C", [(0, 0, 0), (6, 0, 0), (6, 6, 0)])
    obj, tree = scatter(c, mesh_obj("Panel", panel(1.0)))
    post = mesh_obj("Post", box(-0.05, 0.05, -0.05, 0.05, 0.0, 1.5))
    door = mesh_obj("Door", box(-0.5, 0.5, -0.1, 0.1, 0.0, 2.2))
    n_post = add(tree, "CF_NodeSegment", object=post, bend="OFF")
    n_door = add(tree, "CF_NodeSegment", object=door, bend="OFF")
    gen = node(tree, "CF_NodeLinear")
    for name in ("Start", "End", "Corner", "Evenly"):
        tree.links.new(n_post.outputs[0], gen.inputs[name])
    tree.links.new(n_door.outputs[0], gen.inputs["Marker"])
    gen.inputs["Evenly Distance"].default_value = 2.0
    gen.marker_mode = "DISTANCES"
    gen.marker_list = "9"
    co = coords(obj)
    posts = co[(co[:, 2] > 1.2) & (co[:, 2] < 1.6)]
    near_corner = np.linalg.norm(posts[:, :2] - np.array((6.0, 0.0)), axis=1) < 0.08
    assert near_corner.sum() >= 4  # the corner post
    doors = co[co[:, 2] > 2.1]
    assert doors[:, 1].mean() == pytest.approx(3.0, abs=1e-3)  # marker at 9 m = (6, 3)
    assert obj.cf_scatter.error == ""


def test_operators_sequence_randomize_conditional_selector():
    c = curve_obj("C", [(0, 0, 0), (6, 0, 0)])
    a = mesh_obj("A", panel(1.0, 1.0))
    b = mesh_obj("B", panel(1.0, 2.0))
    obj, tree = scatter(c, a)
    gen = node(tree, "CF_NodeLinear")
    na, nb = node(tree, "CF_NodeSegment"), add(tree, "CF_NodeSegment", object=b)
    seq = add(tree, "CF_NodeSequence")
    tree.links.new(na.outputs[0], seq.inputs[0])
    tree.links.new(nb.outputs[0], seq.inputs[1])
    tree.links.new(seq.outputs[0], gen.inputs["Default"])
    live.flush()
    assert len(seq.inputs) == 3  # a free socket was added
    seq.inputs[0].count = 2
    assert tall_modules(obj) == [2, 5]
    cond = add(tree, "CF_NodeConditional", variable="index", compare="EVEN")
    tree.links.new(na.outputs[0], cond.inputs["True"])
    tree.links.new(nb.outputs[0], cond.inputs["False"])
    tree.links.new(cond.outputs[0], gen.inputs["Default"])
    assert tall_modules(obj) == [1, 3, 5]
    expr = add(tree, "CF_NodeExpression", expression="index >= 4")
    tree.links.new(expr.outputs[0], cond.inputs["Condition"])
    assert tall_modules(obj) == [0, 1, 2, 3]
    rnd = add(tree, "CF_NodeRandomize", seed=3)
    tree.links.new(na.outputs[0], rnd.inputs[0])
    tree.links.new(nb.outputs[0], rnd.inputs[1])
    rnd.inputs[1].weight = 0.0
    tree.links.new(rnd.outputs[0], gen.inputs["Default"])
    assert coords(obj)[:, 2].max() == pytest.approx(1.0)
    sel = add(tree, "CF_NodeSelector", variable="index")
    tree.links.new(na.outputs[0], sel.inputs[1])
    tree.links.new(nb.outputs[0], sel.inputs[2])
    tree.links.new(sel.outputs[0], gen.inputs["Default"])
    assert tall_modules(obj) == [1, 3, 5]


def test_compose_mirror_transform_material_uv_and_numbers():
    c = curve_obj("C", [(0, 0, 0), (4, 0, 0)])
    a = mesh_obj("A", panel(1.0))
    b = mesh_obj("B", box(0.0, 1.0, -0.05, 0.05, 1.0, 1.2))
    obj, tree = scatter(c, a)
    gen = node(tree, "CF_NodeLinear")
    na, nb = node(tree, "CF_NodeSegment"), add(tree, "CF_NodeSegment", object=b)
    comp = add(tree, "CF_NodeCompose", mode="OVERLAP")
    tree.links.new(na.outputs[0], comp.inputs[0])
    tree.links.new(nb.outputs[0], comp.inputs[1])
    xf = add(tree, "CF_NodeTransform")
    tree.links.new(comp.outputs[0], xf.inputs["Segment"])
    xf.inputs["Offset"].default_value = (0.0, 0.0, 0.5)
    m1 = bpy.data.materials.new("M1")
    m2 = bpy.data.materials.new("M2")
    mat = add(tree, "CF_NodeMaterial", pick="SEQUENCE", count=2, material_0=m1, material_1=m2)
    tree.links.new(xf.outputs[0], mat.inputs["Segment"])
    mir = add(tree, "CF_NodeMirror", x=False, y=True, mode="ALTERNATE")
    tree.links.new(mat.outputs[0], mir.inputs["Segment"])
    uv = add(tree, "CF_NodeUVTransform", offset=(0.5, 0.0))
    tree.links.new(mir.outputs[0], uv.inputs["Segment"])
    tree.links.new(uv.outputs[0], gen.inputs["Default"])
    co = coords(obj)
    assert co[:, 2].min() == pytest.approx(0.5) and co[:, 2].max() == pytest.approx(1.7)
    assert [m.name for m in obj.data.materials] == ["M1", "M2"]
    mi = np.zeros(len(obj.data.polygons), np.int32)
    obj.data.polygons.foreach_get("material_index", mi)
    assert set(mi) == {0, 1}
    assert not obj.data.validate()
    # Numbers drive parameters.
    val = add(tree, "CF_NodeValue", value=0.25)
    math_n = add(tree, "CF_NodeMath", operation="MULTIPLY")
    tree.links.new(val.outputs[0], math_n.inputs["A"])
    math_n.inputs["B"].default_value = 2.0
    tree.links.new(math_n.outputs[0], gen.inputs["Spacing"])
    live.flush()
    assert obj.cf_scatter.stat_segments == 3


def test_segment_ids_on_curve_drive_selector():
    c = curve_obj("C", [(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)])
    a = mesh_obj("A", panel(1.0, 1.0))
    b = mesh_obj("B", panel(1.0, 2.0))
    obj, tree = scatter(c, a)
    gen = node(tree, "CF_NodeLinear")
    gen.corner_mode = "NONE"
    sel = add(tree, "CF_NodeSelector", variable="segment_id", mode="CLAMP")
    tree.links.new(node(tree, "CF_NodeSegment").outputs[0], sel.inputs[1])
    nb = add(tree, "CF_NodeSegment", object=b)
    tree.links.new(nb.outputs[0], sel.inputs[2])
    tree.links.new(sel.outputs[0], gen.inputs["Default"])
    bpy.context.view_layer.objects.active = c
    c.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.curve.select_all(action="DESELECT")
        pts = c.data.splines[0].points
        pts[1].select = pts[2].select = True
        assert bpy.ops.curveforge.segment_id_set(value=1) == {"FINISHED"}
        tall = coords(obj)
        tall = tall[tall[:, 2] > 1.5]
        assert tall[:, 0].min() >= 3.0 - 1e-6 and tall[:, 0].max() <= 6.0 + 1e-6
        bpy.ops.curve.select_all(action="SELECT")
        bpy.ops.curve.subdivide()
        bpy.context.view_layer.update()
        live.rebuild(obj)
        assert len(c.cf_curve.ids) == 2
        tall = coords(obj)
        tall = tall[tall[:, 2] > 1.5]
        assert tall[:, 0].min() >= 3.0 - 1e-6 and tall[:, 0].max() <= 6.0 + 1e-6
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def test_clipping_area_and_rail_uvs_and_weld():
    c = curve_obj("C", [(0, 0, 0), (10, 0, 0)])
    s = mesh_obj("S", panel(2.0, nx=8))
    s.data.uv_layers.new(name="UVMap")
    area = curve_obj("Area", [(2.5, -2, 0), (7.5, -2, 0), (7.5, 2, 0), (2.5, 2, 0)], cyclic=True)
    obj, tree = scatter(c, s)
    gen = node(tree, "CF_NodeLinear")
    gen.clip_curve = area
    co = coords(obj)
    assert co[:, 0].min() == pytest.approx(2.5, abs=1e-4) and co[:, 0].max() == pytest.approx(7.5, abs=1e-4)
    gen.clip_inside = False
    co = coords(obj)
    assert not np.any((co[:, 0] > 2.6) & (co[:, 0] < 7.4))
    gen.clip_curve = None
    gen.uv_mode = "RAIL"
    uvs = np.zeros(len(obj.data.loops) * 2, np.float32)
    obj.data.uv_layers[0].data.foreach_get("uv", uvs)
    assert uvs[0::2].max() == pytest.approx(10.0, abs=1e-4)
    before = len(obj.data.vertices)
    gen.weld = True
    assert len(obj.data.vertices) < before


def test_bend_vertical_and_rigid():
    c = curve_obj("C", [(0, 0, 0), (4, 0, 3)])
    obj, tree = scatter(c, mesh_obj("S", panel(1.0, nx=2)))
    gen = node(tree, "CF_NodeLinear")
    gen.fit = "COUNT"
    gen.count = 1
    gen.upright = True
    co = coords(obj)
    assert co[:, 0].min() == pytest.approx(0.0, abs=1e-5) and co[:, 0].max() == pytest.approx(4.0, abs=1e-5)
    seg = node(tree, "CF_NodeSegment")
    seg.bend = "OFF"
    seg.upright = "OFF"
    co = coords(obj)
    assert co[:, 2].max() > 3.5  # tilted with the slope


def test_collection_segment_modes():
    c = curve_obj("C", [(0, 0, 0), (12, 0, 0)])
    col = bpy.data.collections.new("Var")
    bpy.context.scene.collection.children.link(col)
    for name, h in (("A", 1.0), ("B", 2.0)):
        ob = mesh_obj(name, panel(1.0, h))
        bpy.context.scene.collection.objects.unlink(ob)
        col.objects.link(ob)
    obj, tree = scatter(c)
    seg = node(tree, "CF_NodeSegment")
    seg.source = "COLLECTION"
    seg.collection = col
    seg.collection_mode = "SEQUENCE"
    assert obj.cf_scatter.stat_segments == 12
    seg.collection_mode = "COMBINE"
    assert obj.cf_scatter.stat_segments == 12
    bpy.data.objects["B"].cf_weight = 0.0
    seg.collection_mode = "RANDOM"
    assert coords(obj)[:, 2].max() == pytest.approx(1.0)


def test_templates_build_cleanly():
    for key, _name, _desc in ops.TEMPLATE_ITEMS:
        assert bpy.ops.curveforge.add_template(template=key) == {"FINISHED"}
        obj = bpy.context.active_object
        assert output.is_output(obj)
        assert obj.cf_scatter.error == "", (key, obj.cf_scatter.error)
        assert len(obj.data.polygons) > 0
        assert not obj.data.validate()


def test_bake_and_box_display():
    c = curve_obj("C", [(0, 0, 0), (4, 0, 0)])
    obj, tree = scatter(c, mesh_obj("S", panel(1.0)))
    obj.cf_scatter.display = "BOX"
    assert len(obj.data.vertices) == 4 * 8
    obj.cf_scatter.display = "FULL"
    bpy.context.view_layer.objects.active = obj
    assert bpy.ops.curveforge.bake(keep=True) == {"FINISHED"}
    baked = bpy.context.active_object
    assert baked != obj and not output.is_output(baked)
    assert len(baked.data.vertices) == len(obj.data.vertices)


def test_muted_nodes_and_reroutes_pass_through():
    c = curve_obj("C", [(0, 0, 0), (4, 0, 0)])
    a = mesh_obj("A", panel(1.0))
    obj, tree = scatter(c, a)
    gen = node(tree, "CF_NodeLinear")
    seg = node(tree, "CF_NodeSegment")
    mir = add(tree, "CF_NodeMirror", x=False, y=False, z=True)
    reroute = tree.nodes.new("NodeReroute")
    tree.links.new(seg.outputs[0], mir.inputs[0])
    tree.links.new(mir.outputs[0], reroute.inputs[0])
    tree.links.new(reroute.outputs[0], gen.inputs["Default"])
    assert coords(obj)[:, 2].min() == pytest.approx(-1.0)
    mir.mute = True
    live.tree_changed(tree)
    assert coords(obj)[:, 2].min() == pytest.approx(0.0)


def test_instancing_rigid_segments():
    c = curve_obj("C", [(0, 0, 0), (6, 0, 0), (6, 6, 0)])
    obj, tree = scatter(c, mesh_obj("Panel", panel(1.0)))
    post = mesh_obj("Post", box(-0.05, 0.05, -0.05, 0.05, 0.0, 1.5))
    post.rotation_euler.z = 0.0
    n_post = add(tree, "CF_NodeSegment", object=post, bend="OFF", instance="ON")
    gen = node(tree, "CF_NodeLinear")
    for name in ("Start", "End", "Corner", "Evenly"):
        tree.links.new(n_post.outputs[0], gen.inputs[name])
    live.flush()
    mod = obj.modifiers.get("CurveForge Instances")
    assert mod is not None and mod.node_group is not None
    dg = bpy.context.evaluated_depsgraph_get()
    # Instance data is only valid while iterating: copy the matrices.
    mats = [np.array(inst.matrix_world) for inst in dg.object_instances
            if inst.is_instance and inst.parent and inst.parent.original == obj]
    assert len(mats) == 7  # start, end, corner and 2 evenly posts per 6 m side
    for m in mats:
        assert abs(m[2, 3]) < 1e-5  # posts stand on the curve
        assert np.allclose(m[:3, 2], (0, 0, 1))  # and are upright
    locs = {(round(float(m[0, 3]), 3) + 0.0, round(float(m[1, 3]), 3) + 0.0) for m in mats}
    # Start / End posts occupy the first / last 10 cm; evenly posts are measured from the points.
    assert {(0.05, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (6.0, 2.0), (6.0, 4.0), (6.0, 5.95)} == locs
    # No post geometry in the mesh itself (only panels + instance points).
    assert coords(obj)[:, 2].max() <= 1.0 + 1e-6
    n_post.instance = "OFF"
    assert obj.modifiers.get("CurveForge Instances") is None
    assert coords(obj)[:, 2].max() == pytest.approx(1.5)
