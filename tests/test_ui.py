"""Draws every panel with a recording layout to catch wrong property / operator names."""

import os
import sys
from types import SimpleNamespace

import pytest

bpy = pytest.importorskip("bpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import paramops  # noqa: E402
from paramops import ops, ui  # noqa: E402
from paramops.demo import box  # noqa: E402
from paramops.writer import write  # noqa: E402
from paramops.core.meshdata import assemble  # noqa: E402

import numpy as np  # noqa: E402


class Recorder:
    """Minimal stand-in for ``bpy.types.UILayout`` that validates its arguments."""

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
        self.log.append(("prop", prop))

    def operator(self, idname, **kw):
        mod, name = idname.split(".")
        assert hasattr(getattr(bpy.ops, mod), name), "unknown operator " + idname
        self.log.append(("op", idname))
        return SimpleNamespace()

    def label(self, **kw):
        self.log.append(("label", kw.get("text", "")))

    def separator(self, **kw):
        pass

    def template_list(self, list_type, list_id, data, prop, active_data, active_prop, **kw):
        assert prop in data.bl_rna.properties and active_prop in active_data.bl_rna.properties
        assert hasattr(bpy.types, list_type)
        self.log.append(("list", prop))


@pytest.fixture(scope="module", autouse=True)
def addon():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    paramops.register()
    yield
    paramops.unregister()


def make_scene():
    cu = bpy.data.curves.new("C", "CURVE")
    sp = cu.splines.new("POLY")
    sp.points.add(2)
    for p, co in zip(sp.points, [(0, 0, 0), (4, 0, 0), (4, 4, 0)]):
        p.co = (*co, 1.0)
    curve = bpy.data.objects.new("C", cu)
    bpy.context.scene.collection.objects.link(curve)
    md = box(0.0, 1.0, -0.1, 0.1, 0.0, 1.0, nx=2)
    me = bpy.data.meshes.new("S")
    write(me, assemble([(md, md.co[None], np.zeros(1, dtype=bool))]))
    smp = bpy.data.objects.new("S", me)
    bpy.context.scene.collection.objects.link(smp)
    sc = ops.create_scatter(bpy.context, curve, default=smp)
    st = sc.paramops
    for slot in (st.default, st.start, st.end, st.evenly, st.corner):
        slot.show_expanded = True
        slot.keep_vertical = True
    st.evenly.source = "COLLECTION"
    item = st.segment_ids.add()
    item.slot.show_expanded = True
    mk = st.markers.add()
    mk.slot.object = smp
    st.last_error = "example error"
    st.stat_truncated = True
    return curve, sc


@pytest.mark.parametrize("active", ["scatter", "curve", "none"])
@pytest.mark.parametrize("mode", ["OBJECT", "EDIT_CURVE"])
def test_all_panels_draw(active, mode):
    curve, sc = make_scene()
    obj = {"scatter": sc, "curve": curve, "none": None}[active]
    ctx = SimpleNamespace(active_object=obj, object=obj, mode=mode, scene=bpy.context.scene)
    for cls in ui.classes:
        if not issubclass(cls, bpy.types.Panel):
            continue
        if hasattr(cls, "poll") and not cls.poll(ctx):
            continue
        log = []
        panel = SimpleNamespace(layout=Recorder(log))
        if hasattr(cls, "draw_header"):
            cls.draw_header(panel, ctx)
        cls.draw(panel, ctx)
        assert log, cls.__name__


def test_ui_list_draw_item():
    _curve, sc = make_scene()
    item = sc.paramops.segment_ids[0]
    ui.POPS_UL_items.draw_item(None, bpy.context, Recorder([]), sc.paramops, item, 0, None, "", 0)
