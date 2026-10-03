# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators."""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator

from . import curvedata, live, output
from .nodes import TREE_ID

MESH_LIKE = {"MESH", "CURVE", "SURFACE", "FONT", "META"}


def context_output(context):
    """CurveForge object for the panel: the active one, or one using the active curve."""
    ob = getattr(context, "active_object", None)
    if ob is None:
        return None
    if output.is_output(ob):
        return ob
    if ob.type == "CURVE":
        for o in bpy.data.objects:
            if output.is_output(o) and o.cf_scatter.style is not None:
                if any(n.bl_idname == "CF_NodeSpline" and n.curve == ob for n in o.cf_scatter.style.nodes):
                    return o
    return None


def add_node(tree, idname, location, label=None, **props):
    node = tree.nodes.new(idname)
    node.location = location
    if label:
        node.label = label
    for k, v in props.items():
        setattr(node, k, v)
    return node


def link(tree, a, a_sock, b, b_sock):
    out = a.outputs[a_sock] if isinstance(a_sock, str) else a.outputs[a_sock]
    inp = b.inputs[b_sock] if isinstance(b_sock, str) else b.inputs[b_sock]
    return tree.links.new(out, inp)


def basic_style(name, curve=None, segment=None):
    """Spline -> Linear Generator <- Segment (Default)."""
    tree = bpy.data.node_groups.new(name, TREE_ID)
    with live.suspended():
        spl = add_node(tree, "CF_NodeSpline", (-460, 160), curve=curve)
        seg = add_node(tree, "CF_NodeSegment", (-460, -60), object=segment)
        gen = add_node(tree, "CF_NodeLinear", (0, 120))
        link(tree, spl, "Spline", gen, "Spline")
        link(tree, seg, "Segment", gen, "Default")
    return tree


def new_output(context, name, style, location=None):
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)
    if location is not None:
        obj.location = location
    with live.suspended():
        obj.cf_scatter.enabled = True
        obj.cf_scatter.style = style
    context.view_layer.update()
    live.rebuild(obj)
    return obj


def show_style(context, tree):
    """Show ``tree`` in a node editor (reuse one, convert the timeline, or split the view)."""
    screen = context.screen
    if screen is None:
        return None
    area = next((a for a in screen.areas if a.type == "NODE_EDITOR"), None)
    if area is None:
        others = [a for a in screen.areas if a.type in {"DOPESHEET_EDITOR", "TIMELINE", "GRAPH_EDITOR",
                                                         "NLA_EDITOR", "TEXT_EDITOR", "CONSOLE", "INFO"}]
        if others:
            area = max(others, key=lambda a: a.width * a.height)
            area.type = "NODE_EDITOR"
        elif context.area is not None:
            before = set(screen.areas)
            with context.temp_override(area=context.area):
                bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
            new = [a for a in screen.areas if a not in before]
            area = new[0] if new else None
            if area is not None:
                area.type = "NODE_EDITOR"
    if area is None:
        return None
    space = area.spaces.active
    space.tree_type = TREE_ID
    space.node_tree = tree
    return area


class CF_OT_new_scatter(Operator):
    bl_idname = "curveforge.new_scatter"
    bl_label = "New Curve Scatter"
    bl_description = ("Scatter along the active curve with a new style (a selected mesh becomes the Default "
                      "segment). Without a curve, one is added")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        curve = context.active_object
        meshes = [o for o in context.selected_objects
                  if o is not curve and o.type in MESH_LIKE and not output.is_output(o)]
        if curve is None or curve.type != "CURVE":
            cu = bpy.data.curves.new("CF Curve", "CURVE")
            cu.dimensions = "3D"
            sp = cu.splines.new("POLY")
            sp.points.add(2)
            for p, co in zip(sp.points, ((0, 0, 0), (6, 0, 0), (6, 4, 0))):
                p.co = (*co, 1.0)
            curve = bpy.data.objects.new("CF Curve", cu)
            context.collection.objects.link(curve)
            curve.location = context.scene.cursor.location
        tree = basic_style("%s Style" % curve.name, curve, meshes[0] if meshes else None)
        obj = new_output(context, "%s_CurveForge" % curve.name, tree)
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {"FINISHED"}


class CF_OT_new_style(Operator):
    bl_idname = "curveforge.new_style"
    bl_label = "New Style"
    bl_description = "Create a new style for this object"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context_output(context)
        if obj is None:
            return {"CANCELLED"}
        obj.cf_scatter.style = basic_style("CurveForge Style")
        return {"FINISHED"}


class CF_OT_edit_style(Operator):
    bl_idname = "curveforge.edit_style"
    bl_label = "Edit Style"
    bl_description = "Open the style of this object in the node editor"

    def execute(self, context):
        obj = context_output(context)
        if obj is None or obj.cf_scatter.style is None:
            return {"CANCELLED"}
        if show_style(context, obj.cf_scatter.style) is None:
            self.report({"WARNING"}, "Open a Node Editor and choose 'CurveForge Style'")
        return {"FINISHED"}


class CF_OT_refresh(Operator):
    bl_idname = "curveforge.refresh"
    bl_label = "Refresh"
    bl_description = "Rebuild the CurveForge object"
    bl_options = {"REGISTER", "UNDO"}

    all: BoolProperty(name="All", default=False)

    def execute(self, context):
        from . import meshio
        meshio.clear_cache()
        targets = output.outputs(context.scene) if self.all else [context_output(context)]
        done = sum(1 for o in targets if o is not None and live.rebuild(o))
        self.report({"INFO"}, "Rebuilt %d object(s)" % done)
        return {"FINISHED"}


class CF_OT_bake(Operator):
    bl_idname = "curveforge.bake"
    bl_label = "Convert to Mesh"
    bl_description = "Make a regular mesh from the generated geometry"
    bl_options = {"REGISTER", "UNDO"}

    keep: BoolProperty(name="Keep Original", default=True, description="Keep the live CurveForge object")

    @classmethod
    def poll(cls, context):
        return context_output(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context_output(context)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        target = obj
        if self.keep:
            target = obj.copy()
            target.data = obj.data.copy()
            target.name = obj.name + "_Mesh"
            for col in obj.users_collection:
                col.objects.link(target)
        target.cf_scatter.enabled = False
        target.cf_scatter.style = None
        for o in context.selected_objects:
            o.select_set(False)
        target.select_set(True)
        context.view_layer.objects.active = target
        return {"FINISHED"}


class _CurveEditOp:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "CURVE" and context.mode == "EDIT_CURVE"

    def finish(self, context):
        for obj in output.outputs(context.scene):
            live.request(obj)


class CF_OT_segment_id_set(_CurveEditOp, Operator):
    bl_idname = "curveforge.segment_id_set"
    bl_label = "Assign Segment ID"
    bl_description = "Give the selected segments (both end points selected) this Segment ID (0 = none)"

    value: IntProperty(name="ID", default=1, min=0)

    def execute(self, context):
        curve = context.active_object
        segs = curvedata.selected_segments(curve)
        if not segs:
            self.report({"WARNING"}, "Select two or more consecutive points")
            return {"CANCELLED"}
        curvedata.set_ids(curve, segs, self.value)
        self.finish(context)
        self.report({"INFO"}, "%d segment(s) set to ID %d" % (len(segs), self.value))
        return {"FINISHED"}


class CF_OT_segment_id_select(_CurveEditOp, Operator):
    bl_idname = "curveforge.segment_id_select"
    bl_label = "Select Segment ID"
    bl_description = "Select the points of the segments with this Segment ID"

    value: IntProperty(name="ID", default=1, min=0)
    deselect: BoolProperty(default=False)

    def execute(self, context):
        curve = context.active_object
        ids = curvedata.sync_ids(curve)
        refs = []
        for si, sp in enumerate(curve.data.splines):
            n = len(sp.bezier_points if sp.type == "BEZIER" else sp.points)
            for k in range(n if sp.use_cyclic_u else n - 1):
                if ids.get(si, {}).get(k, 0) == self.value:
                    refs += [(si, k), (si, (k + 1) % n)]
        curvedata.select_points(curve, refs, not self.deselect)
        curve.data.update_tag()
        if context.area:
            context.area.tag_redraw()
        return {"FINISHED"}


class CF_OT_segment_id_clear(_CurveEditOp, Operator):
    bl_idname = "curveforge.segment_id_clear"
    bl_label = "Clear All Segment IDs"
    bl_description = "Remove every Segment ID of this curve"

    def execute(self, context):
        context.active_object.cf_curve.ids.clear()
        self.finish(context)
        return {"FINISHED"}


TEMPLATE_ITEMS = [
    ("FENCE", "Fence", "Panels, posts at start, end and corners, evenly spaced posts"),
    ("RAILING", "Railing", "Composed module with random balusters and handrail"),
    ("WALL", "Wall with Windows", "Conditional windows, corner pillars and a door marker"),
    ("KERB", "Random Kerb", "Random stones with random materials and rotations"),
    ("PATTERN", "Sequence Pattern", "Sequence with counts and empty gaps"),
]


class CF_OT_add_template(Operator):
    bl_idname = "curveforge.add_template"
    bl_label = "Add Example"
    bl_description = "Add a ready-made example (curve, segments and style)"
    bl_options = {"REGISTER", "UNDO"}

    template: EnumProperty(name="Example", items=TEMPLATE_ITEMS)

    def execute(self, context):
        from . import templates
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj = templates.build(context, self.template, context.scene.cursor.location.copy())
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return {"FINISHED"}


classes = (CF_OT_new_scatter, CF_OT_new_style, CF_OT_edit_style, CF_OT_refresh, CF_OT_bake,
           CF_OT_segment_id_set, CF_OT_segment_id_select, CF_OT_segment_id_clear, CF_OT_add_template)


def menu_add(self, context):
    self.layout.separator()
    self.layout.operator(CF_OT_new_scatter.bl_idname, icon="MOD_ARRAY")
    self.layout.operator_menu_enum(CF_OT_add_template.bl_idname, "template", text="CurveForge Example",
                                   icon="MOD_ARRAY")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(menu_add)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(menu_add)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
