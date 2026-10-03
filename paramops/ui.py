# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar panels (3D View > Sidebar > ParamOps)."""

import bpy
from bpy.types import Panel, UIList

from .props import slot_objects
from .utils import context_scatter, is_scatter, scatters_using

CATEGORY = "ParamOps"
ANCHORED = {"evenly", "corner", "marker"}


def _slot_kind(key):
    return key.split(":")[0]


def draw_slot(layout, slot, label, key):
    """One sample slot: header row (like the five slots of the landing page) + options."""
    kind = _slot_kind(key)
    col = layout.column(align=True)
    row = col.row(align=True)
    row.prop(slot, "show_expanded", text="", emboss=False,
             icon="TRIA_DOWN" if slot.show_expanded else "TRIA_RIGHT")
    row.prop(slot, "enabled", text="")
    sub = row.row(align=True)
    sub.active = slot.enabled
    sub.prop(slot, "source", text="", icon_only=True)
    split = sub.split(factor=0.32, align=True)
    split.label(text=label)
    if slot.source == "OBJECT":
        split.prop(slot, "object", text="")
    else:
        split.prop(slot, "collection", text="")
    if kind in {"start", "end"}:
        sub.prop(slot, "mirror_x", text="", icon="MOD_MIRROR")
    if slot.show_expanded:
        box = col.box()
        box.active = slot.enabled
        draw_slot_options(box, slot, kind)


def draw_slot_options(layout, slot, kind):
    layout.use_property_split = True
    layout.use_property_decorate = False
    if slot.source == "COLLECTION":
        layout.prop(slot, "collection_mode")
        if slot.collection_mode == "RANDOM" and slot.collection is not None:
            sub = layout.column(align=True)
            for ob in slot_objects(slot)[:24]:
                sub.prop(ob, "paramops_weight", text=ob.name)
    col = layout.column(align=True)
    row = col.row(align=True)
    row.use_property_split = False
    row.prop(slot, "deform", toggle=True, icon="MOD_CURVE")
    row.prop(slot, "keep_vertical", toggle=True, icon="EMPTY_SINGLE_ARROW")
    if slot.keep_vertical:
        sub = layout.column(align=True)
        sub.prop(slot, "flat_top", icon="TRIA_UP")
        sub.prop(slot, "flat_center")
        sub.prop(slot, "flat_bottom", icon="TRIA_DOWN")
        sub.prop(slot, "flat_reference")

    col = layout.column(align=True)
    if kind in ANCHORED:
        col.prop(slot, "align_x")
    col.prop(slot, "align_y")
    col.prop(slot, "align_z")
    if kind == "corner" or (kind in ANCHORED and not slot.deform):
        col.prop(slot, "orient")

    col = layout.column(align=True)
    col.prop(slot, "offset")
    col = layout.column(align=True)
    col.prop(slot, "rotation")
    col = layout.column(align=True)
    col.prop(slot, "scale")
    row = layout.row(heading="Mirror", align=True)
    row.prop(slot, "mirror_x", text="X", toggle=True)
    row.prop(slot, "mirror_y", text="Y", toggle=True)
    col = layout.column(align=True)
    col.prop(slot, "padding_before")
    col.prop(slot, "padding_after")

    layout.label(text="Randomize", icon="RNDCURVE")
    col = layout.column(align=True)
    col.prop(slot, "random_offset")
    col = layout.column(align=True)
    col.prop(slot, "random_rotation")
    col = layout.column(align=True)
    col.prop(slot, "random_scale")
    col.prop(slot, "random_flip_x")
    col.prop(slot, "random_flip_y")
    col.prop(slot, "seed")


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY

    @classmethod
    def poll(cls, context):
        return context_scatter(context) is not None


class POPS_PT_main(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY
    bl_label = "ParamOps"

    def draw(self, context):
        layout = self.layout
        sc = context_scatter(context)
        ob = context.active_object
        if sc is None:
            col = layout.column(align=True)
            if ob is not None and ob.type == "CURVE":
                col.label(text=ob.name, icon="CURVE_DATA")
                col.operator("paramops.create", icon="ADD")
            else:
                col.label(text="Select a curve to scatter along", icon="INFO")
                col.operator("paramops.create", text="New Scatter + Curve", icon="ADD")
            layout.operator("paramops.demo", icon="SCENE_DATA")
            return

        st = sc.paramops
        row = layout.row(align=True)
        row.label(text=sc.name, icon="MOD_ARRAY")
        row.prop(st, "auto_update", text="", icon="FILE_REFRESH" if st.auto_update else "PAUSE")
        row.operator("paramops.refresh", text="", icon="FILE_REFRESH")

        if ob is not None and ob.type == "CURVE":
            users = scatters_using(ob)
            if len(users) > 1:
                box = layout.box()
                box.label(text="Scatters on this curve:")
                for u in users:
                    box.operator("paramops.set_scatter", text=u.name,
                                 icon="RADIOBUT_ON" if u == sc else "RADIOBUT_OFF").name = u.name
            layout.operator("paramops.create", text="Add Another Scatter", icon="ADD")

        row = layout.row(align=True)
        row.prop(st, "path", text="", icon="CURVE_DATA")
        row.operator("paramops.select", text="", icon="RESTRICT_SELECT_OFF").target = "PATH"
        row.operator("paramops.edit_path", text="", icon="EDITMODE_HLT")
        if ob is not None and ob.type == "CURVE" and ob != sc:
            layout.operator("paramops.select", text="Select Scatter", icon="RESTRICT_SELECT_OFF").target = "SCATTER"

        if st.last_error:
            box = layout.box()
            box.alert = True
            box.label(text=st.last_error, icon="ERROR")
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="%d samples  ·  %d verts  ·  %.1f ms" % (st.stat_modules, st.stat_verts,
                                                              st.stat_time * 1000.0), icon="INFO")
        if st.stat_truncated:
            col.label(text="Max Samples reached (see Options)", icon="ERROR")
        row = layout.row(align=True)
        row.operator("paramops.convert", icon="MESH_DATA")
        row.operator("paramops.copy_settings", icon="COPYDOWN", text="Copy to Selected")


class POPS_PT_samples(_Base, Panel):
    bl_label = "Samples"
    bl_parent_id = "POPS_PT_main"

    def draw(self, context):
        st = context_scatter(context).paramops
        layout = self.layout
        draw_slot(layout, st.start, "Start", "start")
        draw_slot(layout, st.end, "End", "end")
        draw_slot(layout, st.default, "Default", "default")
        draw_slot(layout, st.evenly, "Evenly", "evenly")
        draw_slot(layout, st.corner, "Corner", "corner")


class POPS_PT_fit(_Base, Panel):
    bl_label = "Default"
    bl_parent_id = "POPS_PT_main"

    def draw(self, context):
        st = context_scatter(context).paramops
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(st, "fit_mode")
        if st.fit_mode == "COUNT":
            layout.prop(st, "fit_count")
        if st.fit_mode == "FIXED":
            layout.prop(st, "fixed_align")
            if st.fixed_align != "DISTRIBUTE":
                layout.prop(st, "fixed_remainder")
        layout.prop(st, "spacing")


class POPS_PT_evenly(_Base, Panel):
    bl_label = "Evenly"
    bl_parent_id = "POPS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context_scatter(context).paramops
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        row = layout.row()
        row.prop(st, "evenly_mode", expand=True)
        col = layout.column(align=True)
        if st.evenly_mode == "COUNT":
            col.prop(st, "evenly_count")
        else:
            col.prop(st, "evenly_spacing")
        if st.evenly_mode == "SPACING":
            col.prop(st, "evenly_min_gap")
            col.prop(st, "evenly_align")
        if not st.evenly.enabled:
            layout.label(text="Enable the Evenly sample to use these settings", icon="INFO")


class POPS_PT_corners(_Base, Panel):
    bl_label = "Corners"
    bl_parent_id = "POPS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context_scatter(context).paramops
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=True)
        col.prop(st, "corner_threshold")
        col.prop(st, "corner_radius")
        col.prop(st, "corner_slide")


class POPS_UL_items(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item.slot, "enabled", text="")
        row.prop(item, "name", text="", emboss=False)
        ob = item.slot.object if item.slot.source == "OBJECT" else item.slot.collection
        row.label(text=ob.name if ob else "(empty)")
        row.label(text=str(len(item.refs)))


def _draw_list_section(layout, context, st, coll_name, index_name, add_op, remove_op, kind,
                       ops, unit):
    row = layout.row()
    row.template_list("POPS_UL_items", coll_name, st, coll_name, st, index_name, rows=3)
    col = row.column(align=True)
    col.operator(add_op, icon="ADD", text="")
    col.operator(remove_op, icon="REMOVE", text="")
    coll = getattr(st, coll_name)
    idx = getattr(st, index_name)
    if not (0 <= idx < len(coll)):
        return None
    item = coll[idx]
    draw_slot(layout, item.slot, item.name, kind)
    curve = st.path
    editing = curve is not None and context.mode == "EDIT_CURVE" and context.active_object == curve
    if editing:
        grid = layout.grid_flow(columns=2, align=True, even_columns=True)
        assign, unassign, select = ops
        grid.operator(assign, icon="CHECKMARK")
        grid.operator(unassign, icon="X")
        grid.operator(select, text="Select", icon="RESTRICT_SELECT_OFF").deselect = False
        grid.operator(select, text="Deselect", icon="RESTRICT_SELECT_ON").deselect = True
    else:
        layout.operator("paramops.edit_path", text="Edit Curve to Assign", icon="EDITMODE_HLT")
    layout.label(text="%d %s" % (len(item.refs), unit), icon="INFO")
    return item


class POPS_PT_segment_ids(_Base, Panel):
    bl_label = "Segment IDs"
    bl_parent_id = "POPS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context_scatter(context).paramops
        _draw_list_section(self.layout, context, st, "segment_ids", "segment_ids_index",
                           "paramops.segment_id_add", "paramops.segment_id_remove", "seg",
                           ("paramops.segment_id_assign", "paramops.segment_id_unassign",
                            "paramops.segment_id_select"), "segment(s) assigned")


class POPS_PT_markers(_Base, Panel):
    bl_label = "Markers"
    bl_parent_id = "POPS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context_scatter(context).paramops
        item = _draw_list_section(self.layout, context, st, "markers", "markers_index",
                                  "paramops.marker_add", "paramops.marker_remove", "marker",
                                  ("paramops.marker_place", "paramops.marker_unplace",
                                   "paramops.marker_select"), "marker(s) placed")
        if item is not None:
            self.layout.prop(item, "positions", text="At", icon="DRIVER_DISTANCE")


class POPS_PT_curve(_Base, Panel):
    bl_label = "Curve"
    bl_parent_id = "POPS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context_scatter(context).paramops
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(st, "reverse")
        col = layout.column(align=True)
        col.prop(st, "clip_mode")
        if st.clip_mode == "PERCENT":
            col.prop(st, "clip_start_pct")
            col.prop(st, "clip_end_pct")
        else:
            col.prop(st, "clip_start")
            col.prop(st, "clip_end")
        col = layout.column(align=True)
        col.prop(st, "offset_y")
        col.prop(st, "offset_z")
        col = layout.column(align=True)
        col.prop(st, "twist_mode")
        col.prop(st, "use_tilt")
        col.prop(st, "resolution")


class POPS_PT_surface(_Base, Panel):
    bl_label = "Conform to Surface"
    bl_parent_id = "POPS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context_scatter(context).paramops, "project", text="")

    def draw(self, context):
        st = context_scatter(context).paramops
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.active = st.project
        layout.prop(st, "project_target")
        layout.prop(st, "project_offset")
        layout.prop(st, "project_step")


class POPS_PT_options(_Base, Panel):
    bl_label = "Options"
    bl_parent_id = "POPS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context_scatter(context).paramops
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(st, "seed")
        layout.prop(st, "preview")
        layout.prop(st, "max_modules")
        layout.prop(st, "apply_sample_transform")
        layout.prop(st, "update_on_frame")
        layout.operator("paramops.refresh", text="Refresh All Scatters", icon="FILE_REFRESH").all = True


classes = (POPS_UL_items, POPS_PT_main, POPS_PT_samples, POPS_PT_fit, POPS_PT_evenly,
           POPS_PT_corners, POPS_PT_segment_ids, POPS_PT_markers, POPS_PT_curve, POPS_PT_surface,
           POPS_PT_options)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
