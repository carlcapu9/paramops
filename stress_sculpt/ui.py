"""Sidebar panels (View3D > Sidebar > Stress Sculpt), menus and the variants list.

Main flow (panel "Structure"): pick the object, choose Support Object / Carve
Inside and a style, put loads on selected vertices or faces, Generate. The other
panels hold the stress analysis, the variants and the lower-level tools.
"""

import bpy
from bpy.types import Menu, Panel, UIList

from . import jobs
from .props import ZONE_ROLES

CATEGORY = "Stress Sculpt"

ROLE_ICONS = {
    "SUPPORT": "PINNED",
    "LOAD": "FORCE_FORCE",
    "KEEP_SOLID": "MESH_CUBE",
    "KEEP_VOID": "SELECT_SUBTRACT",
    "THICKEN": "ADD",
    "HOLLOW": "REMOVE",
    "BRANCH": "OUTLINER_DATA_CURVES",
}

KIND_ICONS = {
    "ANALYSIS": "SHADING_RENDERED",
    "EVOLVE": "MOD_REMESH",
    "LINES": "OUTLINER_DATA_CURVES",
    "SCULPT": "SCULPTMODE_HLT",
    "RELIEF": "MOD_DISPLACE",
    "STRUCTURE": "MOD_BUILD",
    "ORGANIC": "MOD_FLUIDSIM",
    "BRANCHING": "OUTLINER_DATA_CURVES",
    "RIBS": "MOD_LATTICE",
}


# --------------------------------------------------------------------------- helpers
def _runtime(key, default=None):
    runtime = getattr(jobs, "RUNTIME", None)
    if runtime is None:
        return default
    if isinstance(runtime, dict):
        return runtime.get(key, default)
    return getattr(runtime, key, default)


def _running():
    return bool(_runtime("running", False))


def format_stress(pascal):
    """Stress in Pa -> readable string."""
    value = abs(pascal)
    if value >= 1e9:
        return f"{pascal / 1e9:.3g} GPa"
    if value >= 1e6:
        return f"{pascal / 1e6:.3g} MPa"
    if value >= 1e3:
        return f"{pascal / 1e3:.3g} kPa"
    return f"{pascal:.3g} Pa"


def format_length(meters):
    """Length in m -> readable string."""
    value = abs(meters)
    if value == 0.0:
        return "0 mm"
    if value < 1e-3:
        return f"{meters * 1e6:.3g} µm"
    if value < 1.0:
        return f"{meters * 1e3:.3g} mm"
    return f"{meters:.3g} m"


def format_force(newton):
    value = abs(newton)
    if value >= 1e6:
        return f"{newton / 1e6:.3g} MN"
    if value >= 1e3:
        return f"{newton / 1e3:.3g} kN"
    return f"{newton:.3g} N"


def _role_objects(context, role):
    """Enabled objects with ``role`` in the current view layer."""
    objects = getattr(context, "view_layer", None)
    pool = objects.objects if objects is not None else context.scene.objects
    out = []
    for obj in pool:
        settings = getattr(obj, "stress_sculpt", None)
        if settings is not None and settings.role == role and settings.enabled:
            out.append(obj)
    return out


def _role_counts(context):
    counts = {}
    objects = getattr(context, "view_layer", None)
    pool = objects.objects if objects is not None else context.scene.objects
    for obj in pool:
        settings = getattr(obj, "stress_sculpt", None)
        if settings is None or settings.role == "NONE" or not settings.enabled:
            continue
        counts[settings.role] = counts.get(settings.role, 0) + 1
    return counts


def draw_job_status(layout):
    """Progress bar + cancel button while a background job runs. Returns True if running."""
    if not _running():
        return False
    box = layout.box()
    name = _runtime("name", "") or "Working"
    message = _runtime("message", "") or ""
    progress = _runtime("progress", 0.0) or 0.0
    try:
        progress = min(max(float(progress), 0.0), 1.0)
    except (TypeError, ValueError):
        progress = 0.0
    text = f"{name}: {message}" if message else name
    if hasattr(box, "progress"):  # Blender 4.0+
        box.progress(factor=progress, type="BAR", text=text)
    else:
        box.label(text=f"{text} ({progress * 100:.0f}%)", icon="SORTTIME")
    box.operator("stress_sculpt.cancel", text="Cancel", icon="CANCEL")
    return True


def _big_button(layout, operator, text, icon, enabled=True):
    row = layout.row()
    row.scale_y = 1.6
    row.enabled = enabled and not _running()
    row.operator(operator, text=text, icon=icon)
    return row


def _draw_last_result(layout, settings):
    if not settings.last_kind:
        return
    box = layout.box()
    kind = settings.last_kind
    box.label(text=f"Last result: {kind.replace('_', ' ').title()}", icon=KIND_ICONS.get(kind, "INFO"))
    grid = box.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
    if settings.last_max_stress:
        grid.label(text="Max stress")
        grid.label(text=format_stress(settings.last_max_stress))
    if settings.last_max_displacement:
        grid.label(text="Max displacement")
        grid.label(text=format_length(settings.last_max_displacement))
    if settings.last_compliance:
        grid.label(text="Compliance")
        grid.label(text=f"{settings.last_compliance:.4g} J")
    if kind != "ANALYSIS" and settings.last_volume:
        grid.label(text="Material")
        grid.label(text=f"{settings.last_volume * 100:.1f} %")
    if settings.last_iterations:
        grid.label(text="Iterations")
        grid.label(text=str(settings.last_iterations))


class _StressSculptPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


# --------------------------------------------------------------------------- menus
class STRESS_SCULPT_MT_add_zone(Menu):
    bl_idname = "STRESS_SCULPT_MT_add_zone"
    bl_label = "Add Zone"

    def draw(self, _context):
        self.layout.operator_enum("stress_sculpt.add_zone", "role")


class STRESS_SCULPT_MT_demo(Menu):
    bl_idname = "STRESS_SCULPT_MT_demo"
    bl_label = "Examples"

    def draw(self, _context):
        self.layout.operator_enum("stress_sculpt.demo_scene", "preset")


class STRESS_SCULPT_MT_add(Menu):
    """Shift+A > Stress Sculpt."""

    bl_idname = "STRESS_SCULPT_MT_add"
    bl_label = "Stress Sculpt"

    def draw(self, _context):
        layout = self.layout
        layout.operator("stress_sculpt.add_load", text="Load", icon="FORCE_FORCE")
        layout.operator("stress_sculpt.add_support", text="Support", icon="PINNED")
        layout.separator()
        layout.operator_enum("stress_sculpt.add_zone", "role")
        layout.separator()
        layout.menu(STRESS_SCULPT_MT_demo.bl_idname, icon="PRESET")


class STRESS_SCULPT_MT_edit_mesh(Menu):
    """Edit Mode right-click menu > Stress Sculpt."""

    bl_idname = "STRESS_SCULPT_MT_edit_mesh"
    bl_label = "Stress Sculpt"

    def draw(self, _context):
        layout = self.layout
        layout.operator("stress_sculpt.add_load_at_selection", text="Put Weight on Selection", icon="FORCE_FORCE")
        layout.operator("stress_sculpt.add_support_at_selection", text="Support at Selection", icon="PINNED")


def _draw_add_menu(self, _context):
    self.layout.menu(STRESS_SCULPT_MT_add.bl_idname, icon="FORCE_FORCE")


def _draw_edit_menu(self, _context):
    self.layout.separator()
    self.layout.menu(STRESS_SCULPT_MT_edit_mesh.bl_idname, icon="FORCE_FORCE")


# --------------------------------------------------------------------------- structure (main)
class STRESS_SCULPT_PT_structure(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_structure"
    bl_label = "Structure"

    def draw_header_preset(self, _context):
        self.layout.menu(STRESS_SCULPT_MT_demo.bl_idname, text="", icon="PRESET")

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        running = draw_job_status(layout)

        # object
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(settings, "target", text="")
        row.operator("stress_sculpt.set_target", text="", icon="EYEDROPPER")
        if settings.target is None:
            box = layout.box()
            box.label(text="Pick the mesh to work with,", icon="INFO")
            box.label(text="or start from an example.")
            box.menu(STRESS_SCULPT_MT_demo.bl_idname, text="Examples", icon="PRESET")

        layout.row().prop(settings, "structure_mode", expand=True)
        layout.row().prop(settings, "structure_style", expand=True)

        counts = _role_counts(context)
        n_loads = counts.get("LOAD", 0)
        n_supports = counts.get("SUPPORT", 0)

        # loads
        box = layout.box()
        row = box.row(align=True)
        row.label(text=f"Weights: {n_loads}", icon="FORCE_FORCE")
        sub = row.row(align=True)
        sub.operator("stress_sculpt.select_role", text="", icon="RESTRICT_SELECT_OFF").role = "LOAD"
        sub.operator("stress_sculpt.remove_role", text="", icon="TRASH").role = "LOAD"
        col = box.column(align=True)
        col.enabled = not running
        in_edit = context.mode == "EDIT_MESH"
        row = col.row(align=True)
        row.scale_y = 1.2
        row.operator(
            "stress_sculpt.add_load_at_selection",
            text="Put Weight on Selection" if in_edit else "Weight on Selection (Edit Mode)",
            icon="FORCE_FORCE",
        )
        if not in_edit:
            col.label(text="Tab into Edit Mode, select points or faces", icon="INFO")
        loads = _role_objects(context, "LOAD")
        if loads:
            total = sum(obj.stress_sculpt.magnitude for obj in loads)
            box.label(text=f"Total weight {format_force(total)}")

        # supports
        box = layout.box()
        if settings.structure_mode == "SUPPORT":
            row = box.row(align=True)
            row.label(text="Ground", icon="VIEW_PERSPECTIVE")
            row.prop(settings, "ground_height", text="")
            box.row().prop(settings, "ground_contact", expand=True)
            if settings.ground_contact == "FOOTINGS":
                if n_supports:
                    row = box.row(align=True)
                    row.label(text=f"Footings: {n_supports} support(s)", icon="PINNED")
                    row.operator("stress_sculpt.remove_role", text="", icon="TRASH").role = "SUPPORT"
                else:
                    box.prop(settings, "footings", text="Footings (0 = auto)")
            box.operator("stress_sculpt.add_support_at_selection", text="Footing at Selection", icon="PINNED")
        else:
            row = box.row(align=True)
            row.label(text=f"Supports: {n_supports}", icon="PINNED")
            sub = row.row(align=True)
            sub.operator("stress_sculpt.select_role", text="", icon="RESTRICT_SELECT_OFF").role = "SUPPORT"
            sub.operator("stress_sculpt.remove_role", text="", icon="TRASH").role = "SUPPORT"
            box.operator("stress_sculpt.add_support_at_selection", text="Support at Selection", icon="PINNED")
            if not n_supports:
                box.label(text="Anchor the block where it rests", icon="ERROR")

        if settings.target is not None and not n_loads:
            layout.label(text="Put weight on at least one point", icon="ERROR")

        _big_button(layout, "stress_sculpt.generate", "Generate Structure", "MOD_BUILD")
        _draw_last_result(layout, settings)


class STRESS_SCULPT_PT_style(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_style"
    bl_label = "Style Settings"
    bl_parent_id = "STRESS_SCULPT_PT_structure"

    def draw_header(self, context):
        style = context.scene.stress_sculpt.structure_style
        self.layout.label(text="", icon=KIND_ICONS.get(style, "MOD_BUILD"))

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        style = settings.structure_style
        if style == "BRANCHING":
            self._branching(layout, settings)
        else:
            self._organic(layout, settings)
            if style == "RIBS":
                self._ribs(layout, settings)

    @staticmethod
    def _organic(layout, settings):
        col = layout.column(align=True)
        if settings.structure_mode == "SUPPORT":
            col.prop(settings, "support_material")
        else:
            col.prop(settings, "volume_fraction")
        col.prop(settings, "resolution")
        col.prop(settings, "filter_radius")
        col.prop(settings, "max_iterations")
        col.prop(settings, "load_pad")
        row = layout.row(align=True)
        row.label(text="Symmetry")
        row.prop(settings, "symmetry_x", toggle=True)
        row.prop(settings, "symmetry_y", toggle=True)
        row.prop(settings, "symmetry_z", toggle=True)
        col = layout.column(align=True)
        col.prop(settings, "smooth_iterations")
        col.prop(settings, "live_update")
        layout.prop(settings, "show_advanced", icon="PREFERENCES", toggle=True)
        if settings.show_advanced:
            col = layout.column(align=True)
            col.prop(settings, "evo_mode")
            if settings.evo_mode == "SIMP":
                col.prop(settings, "penalty")
                col.prop(settings, "move_limit")
            else:
                col.prop(settings, "grow_threshold")
                col.prop(settings, "hollow_threshold")
                col.prop(settings, "growth_rate")
                col.prop(settings, "keep_volume")
            col.prop(settings, "min_change")
            col.prop(settings, "iso_level")

    @staticmethod
    def _branching(layout, settings):
        col = layout.column(align=True)
        col.prop(settings, "tree_branching")
        col.prop(settings, "tree_split")
        col.prop(settings, "tree_levels")
        col.prop(settings, "tree_organic")
        col = layout.column(align=True)
        col.label(text="Members (sized by their force)")
        col.prop(settings, "tree_thickness")
        col.prop(settings, "tree_min_thickness")
        col.prop(settings, "tree_footing_flare")
        col.prop(settings, "tree_capital_flare")
        col = layout.column(align=True)
        col.prop(settings, "tree_blend")
        col.prop(settings, "tree_detail")
        col.prop(settings, "smooth_iterations")

    @staticmethod
    def _ribs(layout, settings):
        col = layout.column(align=True)
        col.label(text="Ribs", icon="MOD_LATTICE")
        if settings.structure_mode == "CARVE":
            col.prop(settings, "rib_source", text="")
        col.prop(settings, "line_families", text="")
        col.prop(settings, "rib_spacing")
        col.prop(settings, "rib_thickness")
        col.prop(settings, "rib_min_thickness")
        col.prop(settings, "rib_skin")
        col.prop(settings, "sculpt_blend")


class STRESS_SCULPT_PT_active_object(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_active_object"
    bl_label = "Active Object"
    bl_parent_id = "STRESS_SCULPT_PT_structure"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def draw_header(self, context):
        obj = context.active_object
        role = obj.stress_sculpt.role
        self.layout.label(text="", icon=ROLE_ICONS.get(role, "OBJECT_DATA"))

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = context.scene.stress_sculpt
        layout.label(text=obj.name)
        if obj == settings.target:
            layout.label(text="This is the working object", icon="MESH_CUBE")
            return
        if settings.domain is not None and obj == settings.domain:
            layout.label(text="This mesh is the analysis domain", icon="MESH_CUBE")
            return

        ob_settings = obj.stress_sculpt
        layout.prop(ob_settings, "role")
        role = ob_settings.role
        if role == "NONE":
            if obj.type == "MESH":
                layout.operator("stress_sculpt.set_target", text="Use as Object", icon="MESH_CUBE")
            return

        layout.prop(ob_settings, "enabled")
        col = layout.column(align=True)
        col.active = ob_settings.enabled
        if role == "LOAD":
            col.prop(ob_settings, "magnitude", text="Force (N)")
            col.prop(ob_settings, "load_radius")
            if obj.type == "EMPTY":
                col.label(text="The arrow (local +Z) is the force direction", icon="INFO")
        elif role == "SUPPORT":
            col.label(text="Fixed Axes")
            row = col.row(align=True)
            row.prop(ob_settings, "fix_x", toggle=True)
            row.prop(ob_settings, "fix_y", toggle=True)
            row.prop(ob_settings, "fix_z", toggle=True)
        elif role in {"THICKEN", "HOLLOW", "BRANCH"}:
            col.prop(ob_settings, "strength")

        if role != "LOAD" and obj.type == "EMPTY":
            col = layout.column(align=True)
            col.label(text="Region")
            col.prop(obj, "empty_display_type", text="Shape")
            col.prop(obj, "empty_display_size", text="Size")
        elif role in ZONE_ROLES and obj.type == "MESH":
            layout.label(text="The mesh interior is the zone", icon="INFO")


# --------------------------------------------------------------------------- analysis
class STRESS_SCULPT_PT_analyze(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_analyze"
    bl_label = "Stress Analysis"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(settings, "domain", text="Analyze")
        row.operator("stress_sculpt.set_domain", text="", icon="EYEDROPPER")
        col.prop(settings, "resolution")
        col.prop(settings, "voxelize_method", text="")
        col = layout.column(align=True)
        col.prop(settings, "stress_measure")
        col.prop(settings, "colormap")
        col.prop(settings, "color_clip")
        col.prop(settings, "heatmap_on_original")
        _big_button(layout, "stress_sculpt.analyze", "Analyze Stress", "PLAY")
        layout.operator("stress_sculpt.verify", text="Check Generated Structure", icon="CHECKMARK")


# --------------------------------------------------------------------------- variants
class STRESS_SCULPT_UL_variants(UIList):
    bl_idname = "STRESS_SCULPT_UL_variants"

    def draw_item(
        self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index=0, _flt_flag=0
    ):
        icon = KIND_ICONS.get(item.kind, "OBJECT_DATA")
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            row.prop(item, "name", text="", emboss=False, icon=icon)
            sub = row.row(align=True)
            sub.alignment = "RIGHT"
            if item.kind and item.kind != "ANALYSIS" and item.volume:
                sub.label(text=f"{item.volume * 100:.0f}%")
            if item.compliance:
                sub.label(text=f"C {item.compliance:.3g}")
        else:
            layout.alignment = "CENTER"
            layout.label(text="", icon=icon)


class STRESS_SCULPT_PT_variants(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_variants"
    bl_label = "Variants"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt

        row = layout.row()
        row.template_list(
            STRESS_SCULPT_UL_variants.bl_idname, "", settings, "variants",
            settings, "active_variant_index", rows=3,
        )
        col = row.column(align=True)
        col.operator("stress_sculpt.variant_save", text="", icon="ADD")
        col.operator("stress_sculpt.variant_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("stress_sculpt.variant_move", text="", icon="TRIA_UP").direction = "UP"
        col.operator("stress_sculpt.variant_move", text="", icon="TRIA_DOWN").direction = "DOWN"

        index = settings.active_variant_index
        if 0 <= index < len(settings.variants):
            variant = settings.variants[index]
            box = layout.box()
            grid = box.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
            if variant.kind:
                grid.label(text="Kind")
                grid.label(text=variant.kind.replace("_", " ").title())
            if variant.kind != "ANALYSIS" and variant.volume:
                grid.label(text="Material")
                grid.label(text=f"{variant.volume * 100:.1f} %")
            if variant.compliance:
                grid.label(text="Compliance")
                grid.label(text=f"{variant.compliance:.4g} J")
            if variant.max_stress:
                grid.label(text="Max stress")
                grid.label(text=format_stress(variant.max_stress))
            if variant.max_displacement:
                grid.label(text="Max displacement")
                grid.label(text=format_length(variant.max_displacement))
            if variant.iterations:
                grid.label(text="Iterations")
                grid.label(text=str(variant.iterations))
            box.prop(variant, "notes", text="", icon="TEXT")
            box.operator("stress_sculpt.variant_restore", text="Restore Settings", icon="RECOVER_LAST")

        col = layout.column(align=True)
        row = col.row(align=True)
        row.enabled = len(settings.variants) > 0
        row.operator("stress_sculpt.variant_compare", text="Compare Side by Side", icon="VIEW_ORTHO")
        row.operator("stress_sculpt.variant_clear_compare", text="", icon="X")
        col.prop(settings, "compare_spacing")


# --------------------------------------------------------------------------- more tools
class STRESS_SCULPT_PT_tools(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_tools"
    bl_label = "More Tools"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt

        col = layout.column(align=True)
        col.label(text="Zones", icon="MOD_MASK")
        row = col.row(align=True)
        row.menu(STRESS_SCULPT_MT_add_zone.bl_idname, text="Add Zone", icon="ADD")
        row.operator("stress_sculpt.add_load", text="Load", icon="FORCE_FORCE")
        row.operator("stress_sculpt.add_support", text="Support", icon="PINNED")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Stress Lines", icon="OUTLINER_DATA_CURVES")
        col.prop(settings, "line_source", text="")
        col.prop(settings, "line_families", text="")
        col.prop(settings, "n_seeds")
        col.prop(settings, "line_separation")
        col.prop(settings, "line_min_stress")
        col.prop(settings, "branch_depth")
        row = col.row(align=True)
        row.prop(settings, "radius_min", text="Min")
        row.prop(settings, "radius_max", text="Max")
        row = layout.row(align=True)
        row.enabled = not _running()
        row.operator("stress_sculpt.trace_lines", text="Trace Lines", icon="OUTLINER_DATA_CURVES")
        row.operator("stress_sculpt.sculpt", text="Fuse Lines", icon="SCULPTMODE_HLT")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Relief", icon="MOD_DISPLACE")
        col.prop(settings, "relief_amount")
        col.prop(settings, "relief_hollow")
        col.prop(settings, "relief_threshold")
        col.prop(settings, "relief_subdivisions")
        row = layout.row()
        row.enabled = not _running()
        row.operator("stress_sculpt.relief", text="Thicken Surface by Stress", icon="MOD_DISPLACE")


# --------------------------------------------------------------------------- solver
class STRESS_SCULPT_PT_solver(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_solver"
    bl_label = "Material & Solver"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        col = layout.column(align=True)
        col.prop(settings, "material_preset")
        if settings.material_preset == "CUSTOM":
            col.prop(settings, "youngs_modulus")
            col.prop(settings, "poisson_ratio")
        else:
            col.label(text=f"E = {settings.youngs_modulus:g} GPa   ν = {settings.poisson_ratio:.2f}")
        col = layout.column(align=True)
        col.prop(settings, "solver")
        col.prop(settings, "solver_tolerance")
        col.prop(settings, "max_solver_iterations")
        layout.prop(settings, "domain_as_wire")
        row = layout.row()
        row.enabled = not _running()
        row.operator("stress_sculpt.clear_results", text="Clear Results", icon="TRASH")


classes = (
    STRESS_SCULPT_MT_add_zone,
    STRESS_SCULPT_MT_demo,
    STRESS_SCULPT_MT_add,
    STRESS_SCULPT_MT_edit_mesh,
    STRESS_SCULPT_UL_variants,
    STRESS_SCULPT_PT_structure,
    STRESS_SCULPT_PT_style,
    STRESS_SCULPT_PT_active_object,
    STRESS_SCULPT_PT_analyze,
    STRESS_SCULPT_PT_variants,
    STRESS_SCULPT_PT_tools,
    STRESS_SCULPT_PT_solver,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_draw_add_menu)
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.append(_draw_edit_menu)


def unregister():
    bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(_draw_edit_menu)
    bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
