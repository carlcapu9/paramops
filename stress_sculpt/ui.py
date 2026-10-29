"""Sidebar panels (View3D > Sidebar > Stress Sculpt), menus and the variants list."""

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


def _role_counts(scene):
    counts = {}
    for obj in scene.objects:
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
    row.scale_y = 1.5
    row.enabled = enabled and not _running()
    row.operator(operator, text=text, icon=icon)
    return row


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
    bl_label = "Demo Scene"

    def draw(self, _context):
        self.layout.operator_enum("stress_sculpt.demo_scene", "preset")


class STRESS_SCULPT_MT_add(Menu):
    """Shift+A > Stress Sculpt."""

    bl_idname = "STRESS_SCULPT_MT_add"
    bl_label = "Stress Sculpt"

    def draw(self, _context):
        layout = self.layout
        layout.operator("stress_sculpt.add_support", text="Support", icon="PINNED")
        layout.operator("stress_sculpt.add_load", text="Load", icon="FORCE_FORCE")
        layout.separator()
        layout.operator_enum("stress_sculpt.add_zone", "role")
        layout.separator()
        layout.menu(STRESS_SCULPT_MT_demo.bl_idname, icon="PRESET")


def _draw_add_menu(self, _context):
    self.layout.menu(STRESS_SCULPT_MT_add.bl_idname, icon="FORCE_FORCE")


# --------------------------------------------------------------------------- setup
class STRESS_SCULPT_PT_setup(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_setup"
    bl_label = "Setup"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        settings = context.scene.stress_sculpt
        running = draw_job_status(layout)

        col = layout.column(align=True)
        col.label(text="Design Domain", icon="MESH_CUBE")
        row = col.row(align=True)
        row.prop(settings, "domain", text="")
        row.operator("stress_sculpt.set_domain", text="", icon="EYEDROPPER")
        if settings.domain is None:
            box = layout.box()
            box.label(text="Pick a closed mesh as domain,", icon="INFO")
            box.label(text="or start from a demo scene.")
            box.menu(STRESS_SCULPT_MT_demo.bl_idname, text="Demo Scene", icon="PRESET")

        col = layout.column(align=True)
        col.enabled = not running
        col.prop(settings, "resolution")
        col.prop(settings, "voxelize_method", text="")

        col = layout.column(align=True)
        col.prop(settings, "material_preset")
        if settings.material_preset == "CUSTOM":
            col.prop(settings, "youngs_modulus")
            col.prop(settings, "poisson_ratio")
        else:
            col.label(text=f"E = {settings.youngs_modulus:g} GPa   ν = {settings.poisson_ratio:.2f}")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Forces & Zones", icon="FORCE_FORCE")
        row = col.row(align=True)
        row.operator("stress_sculpt.add_support", text="Support", icon="PINNED")
        row.operator("stress_sculpt.add_load", text="Load", icon="FORCE_FORCE")
        row = col.row(align=True)
        row.menu(STRESS_SCULPT_MT_add_zone.bl_idname, text="Zone", icon="ADD")
        row.menu(STRESS_SCULPT_MT_demo.bl_idname, text="Demo", icon="PRESET")

        counts = _role_counts(context.scene)
        n_zones = sum(counts.get(role, 0) for role in ZONE_ROLES)
        row = layout.row(align=True)
        row.label(text=str(counts.get("SUPPORT", 0)), icon="PINNED")
        row.label(text=str(counts.get("LOAD", 0)), icon="FORCE_FORCE")
        row.label(text=str(n_zones), icon="MOD_MASK")
        if settings.domain is not None and counts.get("SUPPORT", 0) == 0:
            layout.label(text="Add at least one support", icon="ERROR")
        if settings.domain is not None and counts.get("LOAD", 0) == 0:
            layout.label(text="Add at least one load", icon="ERROR")


class STRESS_SCULPT_PT_active_object(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_active_object"
    bl_label = "Active Object"
    bl_parent_id = "STRESS_SCULPT_PT_setup"

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
        if settings.domain is not None and obj == settings.domain:
            layout.label(text="This mesh is the design domain", icon="MESH_CUBE")
            return

        ob_settings = obj.stress_sculpt
        layout.prop(ob_settings, "role")
        role = ob_settings.role
        if role == "NONE":
            if obj.type == "MESH":
                layout.operator("stress_sculpt.set_domain", text="Use as Domain", icon="MESH_CUBE")
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


# --------------------------------------------------------------------------- analyze
class STRESS_SCULPT_PT_analyze(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_analyze"
    bl_label = "Analyze"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        col = layout.column(align=True)
        col.prop(settings, "stress_measure")
        col.prop(settings, "colormap")
        col.prop(settings, "color_clip")
        col.prop(settings, "heatmap_on_original")
        _big_button(layout, "stress_sculpt.analyze", "Analyze Stress", "PLAY")

        if settings.last_kind:
            box = layout.box()
            box.label(text=f"Last result: {settings.last_kind.title()}", icon=KIND_ICONS.get(settings.last_kind, "INFO"))
            grid = box.grid_flow(row_major=True, columns=2, even_columns=True, align=True)
            grid.label(text="Max stress")
            grid.label(text=format_stress(settings.last_max_stress))
            grid.label(text="Max displacement")
            grid.label(text=format_length(settings.last_max_displacement))
            grid.label(text="Compliance")
            grid.label(text=f"{settings.last_compliance:.4g} J")
            if settings.last_kind != "ANALYSIS":
                grid.label(text="Material")
                grid.label(text=f"{settings.last_volume * 100:.1f} %")
            if settings.last_iterations:
                grid.label(text="Iterations")
                grid.label(text=str(settings.last_iterations))


# --------------------------------------------------------------------------- evolve
class STRESS_SCULPT_PT_evolve(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_evolve"
    bl_label = "Evolve"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        layout.row().prop(settings, "evo_mode", expand=True)

        col = layout.column(align=True)
        col.prop(settings, "volume_fraction")
        col.prop(settings, "max_iterations")
        col.prop(settings, "filter_radius")

        if settings.evo_mode == "RULES":
            col = layout.column(align=True)
            col.label(text="Growth Rules")
            col.prop(settings, "grow_threshold")
            col.prop(settings, "hollow_threshold")
            col.prop(settings, "growth_rate")
            col.prop(settings, "keep_volume")
            if settings.hollow_threshold >= settings.grow_threshold:
                col.label(text="'Hollow Below' should be lower than 'Thicken Above'", icon="ERROR")

        row = layout.row(align=True)
        row.label(text="Symmetry")
        row.prop(settings, "symmetry_x", toggle=True)
        row.prop(settings, "symmetry_y", toggle=True)
        row.prop(settings, "symmetry_z", toggle=True)

        col = layout.column(align=True)
        col.prop(settings, "iso_level")
        col.prop(settings, "smooth_iterations")
        col.prop(settings, "live_update")

        layout.prop(settings, "show_advanced", icon="PREFERENCES", toggle=True)
        if settings.show_advanced:
            col = layout.column(align=True)
            if settings.evo_mode == "SIMP":
                col.prop(settings, "penalty")
                col.prop(settings, "move_limit")
            col.prop(settings, "min_change")

        _big_button(layout, "stress_sculpt.evolve", "Evolve Shape", "MOD_REMESH")


# --------------------------------------------------------------------------- tension lines
class STRESS_SCULPT_PT_lines(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_lines"
    bl_label = "Tension Lines"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        col = layout.column(align=True)
        col.prop(settings, "line_source")
        col.prop(settings, "line_families")

        col = layout.column(align=True)
        col.prop(settings, "n_seeds")
        col.prop(settings, "seed_from_loads")
        col.prop(settings, "line_separation")
        col.prop(settings, "line_min_stress")

        col = layout.column(align=True)
        col.label(text="Branching")
        col.prop(settings, "branch_depth")
        sub = col.column(align=True)
        sub.active = settings.branch_depth > 0
        sub.prop(settings, "branch_spacing")
        sub.prop(settings, "branch_ratio")
        sub.prop(settings, "branch_taper")

        col = layout.column(align=True)
        col.label(text="Thickness")
        row = col.row(align=True)
        row.prop(settings, "radius_min", text="Min")
        row.prop(settings, "radius_max", text="Max")

        if settings.show_advanced:
            col = layout.column(align=True)
            col.prop(settings, "line_step")
            col.prop(settings, "random_seed")

        _big_button(layout, "stress_sculpt.trace_lines", "Trace Tension Lines", "OUTLINER_DATA_CURVES")


# --------------------------------------------------------------------------- sculpture
class STRESS_SCULPT_PT_sculpture(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_sculpture"
    bl_label = "Sculpture"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        col = layout.column(align=True)
        col.prop(settings, "sculpt_source")
        col.prop(settings, "sculpt_detail")
        col.prop(settings, "sculpt_blend")
        col.prop(settings, "sculpt_smooth")
        layout.label(text="Traces the lines first if needed", icon="INFO")
        _big_button(layout, "stress_sculpt.sculpt", "Sculpt from Lines", "SCULPTMODE_HLT")


# --------------------------------------------------------------------------- relief
class STRESS_SCULPT_PT_relief(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_relief"
    bl_label = "Relief"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        col = layout.column(align=True)
        col.prop(settings, "relief_amount")
        col.prop(settings, "relief_hollow")
        col.prop(settings, "relief_threshold")
        col.prop(settings, "relief_gamma")
        col.prop(settings, "relief_subdivisions")
        col.prop(settings, "relief_smooth")
        _big_button(layout, "stress_sculpt.relief", "Thicken Surface", "MOD_DISPLACE")


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
            if item.kind and item.kind != "ANALYSIS":
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
            settings, "active_variant_index", rows=4,
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
                grid.label(text=variant.kind.title())
            if variant.kind != "ANALYSIS":
                grid.label(text="Material")
                grid.label(text=f"{variant.volume * 100:.1f} %")
            grid.label(text="Compliance")
            grid.label(text=f"{variant.compliance:.4g} J")
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
        row.operator("stress_sculpt.variant_compare", text="Compare", icon="VIEW_ORTHO")
        row.operator("stress_sculpt.variant_clear_compare", text="", icon="X")
        col.prop(settings, "compare_spacing")


# --------------------------------------------------------------------------- solver
class STRESS_SCULPT_PT_solver(_StressSculptPanel, Panel):
    bl_idname = "STRESS_SCULPT_PT_solver"
    bl_label = "Solver"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        settings = context.scene.stress_sculpt
        col = layout.column(align=True)
        col.prop(settings, "solver")
        col.prop(settings, "solver_tolerance")
        col.prop(settings, "max_solver_iterations")
        layout.prop(settings, "domain_as_wire")
        layout.prop(settings, "show_advanced")
        row = layout.row()
        row.enabled = not _running()
        row.operator("stress_sculpt.clear_results", text="Clear Results", icon="TRASH")


classes = (
    STRESS_SCULPT_MT_add_zone,
    STRESS_SCULPT_MT_demo,
    STRESS_SCULPT_MT_add,
    STRESS_SCULPT_UL_variants,
    STRESS_SCULPT_PT_setup,
    STRESS_SCULPT_PT_active_object,
    STRESS_SCULPT_PT_analyze,
    STRESS_SCULPT_PT_evolve,
    STRESS_SCULPT_PT_lines,
    STRESS_SCULPT_PT_sculpture,
    STRESS_SCULPT_PT_relief,
    STRESS_SCULPT_PT_variants,
    STRESS_SCULPT_PT_solver,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_add.append(_draw_add_menu)


def unregister():
    bpy.types.VIEW3D_MT_add.remove(_draw_add_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
