# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators."""

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator

from . import curves, handlers
from .samples import MESH_LIKE
from .utils import context_scatter, copy_group, is_scatter

# Slots that are placed at a point (posts, doors...) start rigid and upright.
ANCHORED_DEFAULTS = dict(deform=False, keep_vertical=True, align_x="CENTER")


def init_slot(slot, anchored):
    if anchored:
        for k, v in ANCHORED_DEFAULTS.items():
            setattr(slot, k, v)
    else:
        slot.deform = True
        slot.keep_vertical = False


def init_defaults(st):
    init_slot(st.default, False)
    for slot in (st.start, st.end, st.evenly, st.corner):
        init_slot(slot, True)


def new_curve(context, name="ParamOps Curve"):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    pts = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0), (6.0, 4.0, 0.0)]
    sp.points.add(len(pts) - 1)
    for p, co in zip(sp.points, pts):
        p.co = (*co, 1.0)
    ob = bpy.data.objects.new(name, cu)
    context.collection.objects.link(ob)
    ob.location = context.scene.cursor.location
    return ob


def create_scatter(context, curve, default=None, name=None):
    name = name or "%s_Scatter" % curve.name
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    cols = list(curve.users_collection) or [context.scene.collection]
    cols[0].objects.link(obj)
    obj.matrix_world.translation = curve.matrix_world.translation
    with handlers.suspended():
        st = obj.paramops
        st.is_scatter = True
        st.path = curve
        init_defaults(st)
        if default is not None:
            st.default.object = default
    curve.paramops.curve_scatter = obj
    context.view_layer.update()
    handlers.rebuild(obj)
    return obj


def _scatter_and_curve(context):
    sc = context_scatter(context)
    if sc is None:
        return None, None
    return sc, sc.paramops.path


class POPS_OT_create(Operator):
    bl_idname = "paramops.create"
    bl_label = "New Scatter"
    bl_description = ("Create a scatter along the active curve (a new curve is added if none is active). "
                      "A selected mesh becomes the Default sample")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        curve = context.active_object
        meshes = [o for o in context.selected_objects
                  if o is not curve and o.type in MESH_LIKE and not is_scatter(o)]
        if curve is None or curve.type != "CURVE":
            curve = new_curve(context)
        obj = create_scatter(context, curve, default=meshes[0] if meshes else None)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {"FINISHED"}


class POPS_OT_refresh(Operator):
    bl_idname = "paramops.refresh"
    bl_label = "Refresh"
    bl_description = "Rebuild the scatter"
    bl_options = {"REGISTER", "UNDO"}

    all: BoolProperty(name="All", default=False, description="Rebuild every scatter of the scene")

    def execute(self, context):
        from . import builder, samples
        samples.clear_cache()
        builder.clear_caches()
        targets = handlers.scatters(context.scene) if self.all else [context_scatter(context)]
        n = sum(1 for ob in targets if ob is not None and handlers.rebuild(ob))
        self.report({"INFO"}, "Rebuilt %d scatter(s)" % n)
        return {"FINISHED"}


class POPS_OT_convert(Operator):
    bl_idname = "paramops.convert"
    bl_label = "Convert to Mesh"
    bl_description = "Turn the scatter into a regular mesh that no longer updates"
    bl_options = {"REGISTER", "UNDO"}

    keep_original: BoolProperty(name="Keep Scatter", default=True,
                                description="Create a mesh copy and keep the live scatter")

    @classmethod
    def poll(cls, context):
        return context_scatter(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        sc = context_scatter(context)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if self.keep_original:
            new = sc.copy()
            new.data = sc.data.copy()
            new.name = sc.name + "_Mesh"
            new.data.name = new.name
            for col in sc.users_collection:
                col.objects.link(new)
            target = new
        else:
            target = sc
        target.paramops.is_scatter = False
        for o in context.selected_objects:
            o.select_set(False)
        target.select_set(True)
        context.view_layer.objects.active = target
        return {"FINISHED"}


class POPS_OT_select(Operator):
    bl_idname = "paramops.select"
    bl_label = "Select"
    bl_description = "Select the curve or the scatter"
    bl_options = {"REGISTER", "UNDO"}

    target: StringProperty(default="PATH")

    def execute(self, context):
        sc = context_scatter(context)
        if sc is None:
            return {"CANCELLED"}
        ob = sc.paramops.path if self.target == "PATH" else sc
        if ob is None:
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        if ob.name not in context.view_layer.objects:
            self.report({"WARNING"}, "%s is not in the view layer" % ob.name)
            return {"CANCELLED"}
        ob.select_set(True)
        context.view_layer.objects.active = ob
        if ob.type == "CURVE":
            ob.paramops.curve_scatter = sc
        return {"FINISHED"}


class POPS_OT_edit_path(Operator):
    bl_idname = "paramops.edit_path"
    bl_label = "Edit Curve"
    bl_description = "Select the curve and enter Edit Mode (to assign Segment IDs and Markers)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        sc = context_scatter(context)
        if sc is None or sc.paramops.path is None:
            return {"CANCELLED"}
        bpy.ops.paramops.select(target="PATH")
        if context.mode != "EDIT_CURVE":
            bpy.ops.object.mode_set(mode="EDIT")
        return {"FINISHED"}


class POPS_OT_set_scatter(Operator):
    bl_idname = "paramops.set_scatter"
    bl_label = "Show Scatter"
    bl_description = "Choose which scatter of this curve is shown in the panel"

    name: StringProperty()

    def execute(self, context):
        ob = context.active_object
        sc = bpy.data.objects.get(self.name)
        if ob is None or ob.type != "CURVE" or not is_scatter(sc):
            return {"CANCELLED"}
        ob.paramops.curve_scatter = sc
        return {"FINISHED"}


class POPS_OT_copy_settings(Operator):
    bl_idname = "paramops.copy_settings"
    bl_label = "Copy to Selected"
    bl_description = "Copy samples and settings of the active scatter to the other selected scatters"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return is_scatter(context.active_object)

    def execute(self, context):
        src = context.active_object
        skip = {"is_scatter", "path", "topo_signature", "curve_scatter", "last_error", "stat_modules",
                "stat_verts", "stat_faces", "stat_time", "stat_truncated"}
        n = 0
        for ob in context.selected_objects:
            if ob is src or not is_scatter(ob):
                continue
            with handlers.suspended():
                copy_group(src.paramops, ob.paramops, skip)
                for item in ob.paramops.segment_ids:
                    item.refs.clear()
                for item in ob.paramops.markers:
                    item.refs.clear()
            handlers.rebuild(ob)
            n += 1
        self.report({"INFO"}, "Settings copied to %d scatter(s)" % n)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Segment IDs
# ---------------------------------------------------------------------------

def _in_curve_edit(context, curve):
    return curve is not None and context.mode == "EDIT_CURVE" and context.active_object == curve


class _ListOp:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context_scatter(context) is not None


class POPS_OT_segment_id_add(_ListOp, Operator):
    bl_idname = "paramops.segment_id_add"
    bl_label = "Add Segment ID"
    bl_description = "Add a Segment ID: an alternative Default sample for some segments of the curve"

    def execute(self, context):
        sc = context_scatter(context)
        st = sc.paramops
        with handlers.suspended():
            item = st.segment_ids.add()
            item.name = "ID_%d" % len(st.segment_ids)
            init_slot(item.slot, False)
            st.segment_ids_index = len(st.segment_ids) - 1
        return {"FINISHED"}


class POPS_OT_segment_id_remove(_ListOp, Operator):
    bl_idname = "paramops.segment_id_remove"
    bl_label = "Remove Segment ID"
    bl_description = "Remove the active Segment ID"

    def execute(self, context):
        sc = context_scatter(context)
        st = sc.paramops
        if 0 <= st.segment_ids_index < len(st.segment_ids):
            st.segment_ids.remove(st.segment_ids_index)
            st.segment_ids_index = max(0, st.segment_ids_index - 1)
            handlers.request_update(sc)
        return {"FINISHED"}


class _EditOp(_ListOp):
    """Operators working on the selected points of the curve in Edit Mode."""

    @classmethod
    def poll(cls, context):
        sc = context_scatter(context)
        return sc is not None and _in_curve_edit(context, sc.paramops.path)


def _active(collection, index):
    return collection[index] if 0 <= index < len(collection) else None


class POPS_OT_segment_id_assign(_EditOp, Operator):
    bl_idname = "paramops.segment_id_assign"
    bl_label = "Assign"
    bl_description = "Assign the selected segments (both end points selected) to the active Segment ID"

    def execute(self, context):
        sc, curve = _scatter_and_curve(context)
        st = sc.paramops
        item = _active(st.segment_ids, st.segment_ids_index)
        if item is None:
            return {"CANCELLED"}
        segs = curves.selected_segments(curve)
        if not segs:
            self.report({"WARNING"}, "Select two consecutive points of the curve")
            return {"CANCELLED"}
        pts = curves.local_points(curve)
        with handlers.suspended():
            chosen = set(segs)
            for other in st.segment_ids:
                for i in reversed(range(len(other.refs))):
                    r = other.refs[i]
                    if (r.spline, r.index) in chosen:
                        other.refs.remove(i)
            for si, i in segs:
                n = len(pts[si])
                ref = item.refs.add()
                ref.spline, ref.index = si, i
                ref.co_a, ref.co_b = pts[si][i], pts[si][(i + 1) % n]
        handlers.rebuild(sc)
        self.report({"INFO"}, "%d segment(s) assigned" % len(item.refs))
        return {"FINISHED"}


class POPS_OT_segment_id_unassign(_EditOp, Operator):
    bl_idname = "paramops.segment_id_unassign"
    bl_label = "Remove"
    bl_description = "Remove the selected segments from the active Segment ID"

    def execute(self, context):
        sc, curve = _scatter_and_curve(context)
        st = sc.paramops
        item = _active(st.segment_ids, st.segment_ids_index)
        if item is None:
            return {"CANCELLED"}
        chosen = set(curves.selected_segments(curve))
        with handlers.suspended():
            for i in reversed(range(len(item.refs))):
                r = item.refs[i]
                if (r.spline, r.index) in chosen:
                    item.refs.remove(i)
        handlers.rebuild(sc)
        return {"FINISHED"}


class POPS_OT_segment_id_select(_EditOp, Operator):
    bl_idname = "paramops.segment_id_select"
    bl_label = "Select"
    bl_description = "Select (or deselect) the points of the segments of the active Segment ID"

    deselect: BoolProperty(default=False)

    def execute(self, context):
        sc, curve = _scatter_and_curve(context)
        st = sc.paramops
        item = _active(st.segment_ids, st.segment_ids_index)
        if item is None:
            return {"CANCELLED"}
        pts = curves.local_points(curve)
        refs = []
        for r in item.refs:
            if 0 <= r.spline < len(pts):
                n = len(pts[r.spline])
                refs += [(r.spline, r.index), (r.spline, (r.index + 1) % n)]
        curves.set_point_selection(curve, refs, not self.deselect)
        curve.data.update_tag()
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

class POPS_OT_marker_add(_ListOp, Operator):
    bl_idname = "paramops.marker_add"
    bl_label = "Add Marker"
    bl_description = "Add a Marker type: a sample pinned to chosen points of the curve"

    def execute(self, context):
        sc = context_scatter(context)
        st = sc.paramops
        with handlers.suspended():
            item = st.markers.add()
            item.name = "Marker_%d" % len(st.markers)
            init_slot(item.slot, True)
            st.markers_index = len(st.markers) - 1
        return {"FINISHED"}


class POPS_OT_marker_remove(_ListOp, Operator):
    bl_idname = "paramops.marker_remove"
    bl_label = "Remove Marker"
    bl_description = "Remove the active Marker type"

    def execute(self, context):
        sc = context_scatter(context)
        st = sc.paramops
        if 0 <= st.markers_index < len(st.markers):
            st.markers.remove(st.markers_index)
            st.markers_index = max(0, st.markers_index - 1)
            handlers.request_update(sc)
        return {"FINISHED"}


class POPS_OT_marker_place(_EditOp, Operator):
    bl_idname = "paramops.marker_place"
    bl_label = "Place"
    bl_description = "Place the active Marker on the selected points of the curve"

    def execute(self, context):
        sc, curve = _scatter_and_curve(context)
        st = sc.paramops
        item = _active(st.markers, st.markers_index)
        if item is None:
            return {"CANCELLED"}
        sel = curves.selected_points(curve)
        if not sel:
            self.report({"WARNING"}, "Select one or more points of the curve")
            return {"CANCELLED"}
        pts = curves.local_points(curve)
        with handlers.suspended():
            chosen = set(sel)
            for other in st.markers:
                for i in reversed(range(len(other.refs))):
                    r = other.refs[i]
                    if (r.spline, r.index) in chosen:
                        other.refs.remove(i)
            for si, pi in sel:
                ref = item.refs.add()
                ref.spline, ref.index = si, pi
                ref.co = pts[si][pi]
        handlers.rebuild(sc)
        self.report({"INFO"}, "%d marker(s) placed" % len(item.refs))
        return {"FINISHED"}


class POPS_OT_marker_unplace(_EditOp, Operator):
    bl_idname = "paramops.marker_unplace"
    bl_label = "Remove"
    bl_description = "Remove the active Marker from the selected points"

    def execute(self, context):
        sc, curve = _scatter_and_curve(context)
        st = sc.paramops
        item = _active(st.markers, st.markers_index)
        if item is None:
            return {"CANCELLED"}
        chosen = set(curves.selected_points(curve))
        with handlers.suspended():
            for i in reversed(range(len(item.refs))):
                r = item.refs[i]
                if (r.spline, r.index) in chosen:
                    item.refs.remove(i)
        handlers.rebuild(sc)
        return {"FINISHED"}


class POPS_OT_marker_select(_EditOp, Operator):
    bl_idname = "paramops.marker_select"
    bl_label = "Select"
    bl_description = "Select (or deselect) the points where the active Marker is placed"

    deselect: BoolProperty(default=False)

    def execute(self, context):
        sc, curve = _scatter_and_curve(context)
        st = sc.paramops
        item = _active(st.markers, st.markers_index)
        if item is None:
            return {"CANCELLED"}
        curves.set_point_selection(curve, [(r.spline, r.index) for r in item.refs], not self.deselect)
        curve.data.update_tag()
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


class POPS_OT_demo(Operator):
    bl_idname = "paramops.demo"
    bl_label = "Demo Scene"
    bl_description = "Add example samples and scatters (fence on a slope, rounded wall, railing)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import demo
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        objs = demo.build_demo(context)
        if objs:
            for o in context.selected_objects:
                o.select_set(False)
            objs[0].select_set(True)
            context.view_layer.objects.active = objs[0]
        return {"FINISHED"}


classes = (
    POPS_OT_create, POPS_OT_refresh, POPS_OT_convert, POPS_OT_select, POPS_OT_edit_path,
    POPS_OT_set_scatter, POPS_OT_copy_settings,
    POPS_OT_segment_id_add, POPS_OT_segment_id_remove, POPS_OT_segment_id_assign,
    POPS_OT_segment_id_unassign, POPS_OT_segment_id_select,
    POPS_OT_marker_add, POPS_OT_marker_remove, POPS_OT_marker_place, POPS_OT_marker_unplace,
    POPS_OT_marker_select, POPS_OT_demo,
)


def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(POPS_OT_create.bl_idname, icon="CURVE_PATH")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_object.append(menu_func)
    bpy.types.VIEW3D_MT_add.append(menu_func)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(menu_func)
    bpy.types.VIEW3D_MT_object.remove(menu_func)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
