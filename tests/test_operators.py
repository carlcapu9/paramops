"""The operators of the sidebar and of the Add menu."""

import os
import sys

import pytest

bpy = pytest.importorskip("bpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import curveforge  # noqa: E402
from curveforge import curvedata, live, output  # noqa: E402
from curveforge.templates import box  # noqa: E402

from test_blender_nodes import curve_obj, mesh_obj  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    curveforge.register()
    yield
    curveforge.unregister()


@pytest.fixture(autouse=True)
def clean():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.node_groups):
        for d in list(coll):
            coll.remove(d)
    yield


def select_only(*objs, active=None):
    bpy.context.view_layer.update()
    for o in bpy.context.scene.objects:
        o.select_set(o in objs)
    bpy.context.view_layer.objects.active = active


def test_new_scatter_uses_active_curve_and_selected_mesh():
    curve = curve_obj("Path", [(0, 0, 0), (5, 0, 0), (5, 3, 0)])
    module = mesh_obj("Module", box(0.0, 1.0, -0.1, 0.1, 0.0, 1.0, nx=2))
    select_only(curve, module, active=curve)
    assert bpy.ops.curveforge.new_scatter() == {"FINISHED"}
    obj = bpy.context.active_object
    assert output.is_output(obj) and obj.select_get() and not curve.select_get()
    tree = obj.cf_scatter.style
    kinds = sorted(n.bl_idname for n in tree.nodes)
    assert kinds == ["CF_NodeLinear", "CF_NodeSegment", "CF_NodeSpline"]
    assert next(n for n in tree.nodes if n.bl_idname == "CF_NodeSpline").curve == curve
    assert next(n for n in tree.nodes if n.bl_idname == "CF_NodeSegment").object == module
    assert obj.cf_scatter.error == ""
    assert obj.cf_scatter.stat_segments == 8 and len(obj.data.polygons) == 8 * 10


def test_new_scatter_without_curve_adds_one():
    select_only(active=None)
    assert bpy.ops.curveforge.new_scatter() == {"FINISHED"}
    obj = bpy.context.active_object
    spline = next(n for n in obj.cf_scatter.style.nodes if n.bl_idname == "CF_NodeSpline")
    assert spline.curve is not None and spline.curve.type == "CURVE"
    # No segment yet: the panel says what is missing.
    assert obj.cf_scatter.stat_segments == 0
    assert obj.cf_scatter.error == "'Segment': choose an object"


def test_new_style_refresh_and_edit_style():
    bpy.ops.curveforge.add_template(template="FENCE")
    obj = bpy.context.active_object
    old = obj.cf_scatter.style
    assert bpy.ops.curveforge.refresh(all=True) == {"FINISHED"}
    # No node editor in background mode: Edit Style only reports it.
    assert bpy.ops.curveforge.edit_style() == {"FINISHED"}
    assert bpy.ops.curveforge.new_style() == {"FINISHED"}
    assert obj.cf_scatter.style is not None and obj.cf_scatter.style != old
    live.flush()
    # The new style has no curve and no segment object yet: the panel says so.
    assert obj.cf_scatter.error == "'Spline': choose a curve | 'Segment': choose an object"
    assert obj.cf_scatter.stat_segments == 0


def test_segment_id_operators_in_edit_mode():
    bpy.ops.curveforge.add_template(template="GARDEN")
    obj = bpy.context.active_object
    curve = next(n.curve for n in obj.cf_scatter.style.nodes if n.bl_idname == "CF_NodeSpline")
    select_only(curve, active=curve)
    bpy.ops.object.mode_set(mode="EDIT")
    pts = curve.data.splines[0].points
    for i, p in enumerate(pts):
        p.select = i in (3, 4)
    assert bpy.ops.curveforge.segment_id_set(value=3) == {"FINISHED"}
    for p in pts:
        p.select = False
    assert bpy.ops.curveforge.segment_id_select(value=1) == {"FINISHED"}
    assert [i for i, p in enumerate(pts) if p.select] == [1, 2]
    bpy.ops.object.mode_set(mode="OBJECT")
    ids = curvedata.sync_ids(curve)
    assert ids == {0: {1: 1, 2: 2, 3: 3}}
    live.rebuild(obj)
    assert obj.cf_scatter.error == ""
    select_only(curve, active=curve)
    bpy.ops.object.mode_set(mode="EDIT")
    assert bpy.ops.curveforge.segment_id_clear() == {"FINISHED"}
    bpy.ops.object.mode_set(mode="OBJECT")
    assert curvedata.sync_ids(curve) == {}
