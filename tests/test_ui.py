"""Draw every node and panel with a recording layout to catch wrong property / operator names."""

import os
import sys
from types import SimpleNamespace

import pytest

bpy = pytest.importorskip("bpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import curveforge  # noqa: E402
from curveforge import nodes, ops, ui  # noqa: E402


class Recorder:
    """Stand-in for ``bpy.types.UILayout`` that checks the names it receives."""

    def __init__(self, log):
        self.log = log
        self.active = True
        self.alert = False
        self.enabled = True
        self.scale_y = 1.0
        self.use_property_split = False
        self.use_property_decorate = True

    def _child(self, *args, **kw):
        return Recorder(self.log)

    row = column = box = split = grid_flow = column_flow = _child

    def prop(self, data, prop, **kw):
        assert prop in data.bl_rna.properties, "unknown property %r on %r" % (prop, data)
        self.log.append(prop)

    def operator(self, idname, **kw):
        mod, name = idname.split(".")
        assert hasattr(getattr(bpy.ops, mod), name), "unknown operator " + idname
        self.log.append(idname)
        return SimpleNamespace()

    def operator_menu_enum(self, idname, prop, **kw):
        mod, name = idname.split(".")
        op = getattr(getattr(bpy.ops, mod), name)
        assert prop in op.get_rna_type().properties
        self.log.append(idname)

    def template_ID(self, data, prop, **kw):
        assert prop in data.bl_rna.properties
        self.log.append(prop)

    def label(self, **kw):
        self.log.append(kw.get("text", ""))

    def separator(self, **kw):
        pass


@pytest.fixture(scope="module", autouse=True)
def addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    curveforge.register()
    yield
    curveforge.unregister()


def test_every_node_draws():
    tree = bpy.data.node_groups.new("UI", nodes.TREE_ID)
    ctx = SimpleNamespace(active_object=None, mode="OBJECT")
    for cls in nodes.node_classes:
        node = tree.nodes.new(cls.bl_idname)
        for fn in ("draw_buttons", "draw_buttons_ext", "draw_panel"):
            getattr(node, fn)(ctx, Recorder([]))
        for sock in list(node.inputs) + list(node.outputs):
            sock.draw(ctx, Recorder([]), node, sock.name)
    # Variants of the conditional / segment / material / linear drawings.
    cond = tree.nodes.new("CF_NodeConditional")
    for op in ("BETWEEN", "EVERY", "EVEN"):
        cond.compare = op
        cond.draw_buttons(ctx, Recorder([]))
    seg = tree.nodes.new("CF_NodeSegment")
    seg.source = "COLLECTION"
    seg.draw_buttons_ext(ctx, Recorder([]))
    gen = tree.nodes.new("CF_NodeLinear")
    for mode in ("VERTICES", "REPEAT"):
        gen.marker_mode = mode
        gen.adaptive = mode == "REPEAT"
        gen.draw_buttons_ext(ctx, Recorder([]))
    mat = tree.nodes.new("CF_NodeMaterial")
    mat.target, mat.pick, mat.count = "SLOT", "RANDOM", 8
    mat.draw_buttons(ctx, Recorder([]))


@pytest.mark.parametrize("active", ["output", "curve", "none"])
@pytest.mark.parametrize("mode", ["OBJECT", "EDIT_CURVE"])
def test_panels_draw(active, mode):
    bpy.ops.curveforge.add_template(template="WALL")
    obj = bpy.context.active_object
    curve = next(n.curve for n in obj.cf_scatter.style.nodes if n.bl_idname == "CF_NodeSpline")
    target = {"output": obj, "curve": curve, "none": None}[active]
    obj.cf_scatter.error = "example | message"
    obj.cf_scatter.stat_truncated = True
    tree = obj.cf_scatter.style
    tree.nodes.active = next(n for n in tree.nodes if n.bl_idname == "CF_NodeLinear")
    space = SimpleNamespace(tree_type=nodes.TREE_ID, node_tree=tree)
    ctx = SimpleNamespace(active_object=target, object=target, mode=mode, scene=bpy.context.scene,
                          space_data=space, window_manager=bpy.context.window_manager)
    drawn = 0
    for cls in ui.classes:
        if hasattr(cls, "poll") and not cls.poll(ctx):
            continue
        panel = SimpleNamespace(layout=Recorder([]))
        cls.draw(panel, ctx)
        drawn += 1
    assert drawn >= 1


def test_context_output_from_curve():
    bpy.ops.curveforge.add_template(template="FENCE")
    obj = bpy.context.active_object
    curve = next(n.curve for n in obj.cf_scatter.style.nodes if n.bl_idname == "CF_NodeSpline")
    assert ops.context_output(SimpleNamespace(active_object=curve)) == obj
    assert ops.context_output(SimpleNamespace(active_object=obj)) == obj


def test_overlay_midpoints():
    from curveforge import overlay
    bpy.ops.curveforge.add_template(template="GARDEN")
    obj = bpy.context.active_object
    curve = next(n.curve for n in obj.cf_scatter.style.nodes if n.bl_idname == "CF_NodeSpline")
    mids = list(overlay.midpoints(curve))
    assert len(mids) == 4
    ids = {(r.spline, r.index): r.value for r in curve.cf_curve.ids}
    assert sorted(ids.values()) == [1, 2]
    origin = curve.matrix_world.translation
    assert (mids[0][2] - origin - __import__("mathutils").Vector((3.0, 0.0, 0.0))).length < 1e-4
    overlay.draw()  # no 3D view region in background mode: nothing to draw, no error
