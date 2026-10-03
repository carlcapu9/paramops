# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar panels: 3D View (object, exported parameters, Segment IDs) and Node Editor."""

import bpy
from bpy.types import Panel

from . import output
from .nodes import TREE_ID
from .ops import context_output

CATEGORY = "CurveForge"


class CF_PT_object(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_label = "CurveForge"

    def draw(self, context):
        layout = self.layout
        obj = context_output(context)
        active = context.active_object
        if obj is None:
            col = layout.column(align=True)
            if active is not None and active.type == "CURVE":
                col.label(text=active.name, icon="CURVE_DATA")
                col.operator("curveforge.new_scatter", icon="ADD")
            else:
                col.label(text="Select a curve to scatter along", icon="INFO")
                col.operator("curveforge.new_scatter", text="New Curve Scatter", icon="ADD")
            layout.operator_menu_enum("curveforge.add_template", "template", text="Add Example",
                                      icon="PRESET")
            return
        st = obj.cf_scatter
        row = layout.row(align=True)
        row.label(text=obj.name, icon="MOD_ARRAY")
        row.prop(st, "live", text="", icon="FILE_REFRESH" if st.live else "PAUSE")
        row.operator("curveforge.refresh", text="", icon="FILE_REFRESH")
        layout.template_ID(st, "style", new="curveforge.new_style")
        row = layout.row()
        row.scale_y = 1.3
        row.operator("curveforge.edit_style", icon="NODETREE")
        layout.prop(st, "display", expand=True)
        if st.error:
            box = layout.box()
            box.alert = True
            for msg in st.error.split(" | ")[:4]:
                box.label(text=msg, icon="ERROR")
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="%d segments  ·  %d faces  ·  %.1f ms" % (st.stat_segments, st.stat_faces,
                                                                  st.stat_time * 1000.0), icon="INFO")
        if st.stat_truncated:
            col.label(text="Max Segments reached", icon="ERROR")
        layout.prop(st, "update_on_frame")
        row = layout.row(align=True)
        row.operator("curveforge.bake", icon="MESH_DATA")
        row.operator_menu_enum("curveforge.add_template", "template", text="Examples", icon="PRESET")


class CF_PT_parameters(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_label = "Parameters"
    bl_parent_id = "CF_PT_object"

    @classmethod
    def poll(cls, context):
        obj = context_output(context)
        return obj is not None and obj.cf_scatter.style is not None

    def draw(self, context):
        layout = self.layout
        tree = context_output(context).cf_scatter.style
        nodes = [n for n in tree.nodes if getattr(n, "cf_panel", False)]
        if not nodes:
            layout.label(text="Turn on 'Show in Panel' on nodes", icon="INFO")
            layout.label(text="(Node Editor sidebar, CurveForge tab)")
            return
        for node in sorted(nodes, key=lambda n: (-n.location.y, n.location.x)):
            box = layout.box()
            box.label(text=node.label or node.name, icon=getattr(node, "bl_icon", "NODE"))
            node.draw_panel(context, box)


class CF_PT_segment_ids(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_label = "Segment IDs"

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "CURVE" and context.mode == "EDIT_CURVE"

    def draw(self, context):
        layout = self.layout
        curve = context.active_object
        data = curve.cf_curve
        layout.prop(context.window_manager, "cf_show_ids")
        layout.label(text="Select consecutive points, then:")
        row = layout.row(align=True)
        row.prop(data, "edit_value", text="ID")
        row.operator("curveforge.segment_id_set", text="Assign", icon="CHECKMARK").value = data.edit_value
        row = layout.row(align=True)
        op = row.operator("curveforge.segment_id_select", text="Select", icon="RESTRICT_SELECT_OFF")
        op.value, op.deselect = data.edit_value, False
        op = row.operator("curveforge.segment_id_select", text="Deselect", icon="RESTRICT_SELECT_ON")
        op.value, op.deselect = data.edit_value, True
        counts = {}
        for ref in data.ids:
            counts[ref.value] = counts.get(ref.value, 0) + 1
        col = layout.column(align=True)
        for value in sorted(counts):
            col.label(text="ID %d: %d segment(s)" % (value, counts[value]))
        if counts:
            layout.operator("curveforge.segment_id_clear", icon="X")


class CF_PT_node_editor(Panel):
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_label = "CurveForge"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return getattr(space, "tree_type", "") == TREE_ID and space.node_tree is not None

    def draw(self, context):
        layout = self.layout
        tree = context.space_data.node_tree
        node = tree.nodes.active
        users = output.users_of(tree)
        col = layout.column(align=True)
        col.label(text="Used by:", icon="OBJECT_DATA")
        for obj in users[:6]:
            col.label(text="   " + obj.name)
        if not users:
            col.label(text="   (no object)")
        layout.operator("curveforge.refresh", text="Refresh All", icon="FILE_REFRESH").all = True
        if node is None or not hasattr(node, "cf_panel"):
            return
        box = layout.box()
        box.label(text=node.label or node.name, icon=getattr(node, "bl_icon", "NODE"))
        box.prop(node, "cf_panel")
        node.draw_buttons_ext(context, box)


classes = (CF_PT_object, CF_PT_parameters, CF_PT_segment_ids, CF_PT_node_editor)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
