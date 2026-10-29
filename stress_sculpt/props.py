"""Blender properties of Stress Sculpt (the shared data contract of the add-on).

* ``Scene.stress_sculpt``  -> ``StressSculptSceneSettings``
* ``Object.stress_sculpt`` -> ``StressSculptObjectSettings``

Registered properties must always be read with attribute access
(``scene.stress_sculpt.resolution``), never with ``obj["..."]``: since
Blender 5.0 RNA-defined properties are not exposed as ID custom properties.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


ROLE_ITEMS = [
    ("NONE", "None", "Not used by Stress Sculpt", "BLANK1", 0),
    ("SUPPORT", "Support", "Anchor: nodes inside this empty cannot move along the fixed axes", "PINNED", 1),
    ("LOAD", "Load", "Force applied along the empty's local +Z axis (the arrow)", "FORCE_FORCE", 2),
    ("KEEP_SOLID", "Keep Solid", "Material is always kept inside this zone", "MESH_CUBE", 3),
    ("KEEP_VOID", "Keep Void", "Material is never placed inside this zone", "SELECT_SUBTRACT", 4),
    ("THICKEN", "Thicken", "Zone where the structure tends to grow thicker", "ADD", 5),
    ("HOLLOW", "Hollow", "Zone where the structure tends to be carved out", "REMOVE", 6),
    ("BRANCH", "Branch", "Zone where tension lines are seeded densely and branch more", "OUTLINER_DATA_CURVES", 7),
]

ZONE_ROLES = {"KEEP_SOLID", "KEEP_VOID", "THICKEN", "HOLLOW", "BRANCH"}

# name, E [GPa], nu
MATERIAL_PRESETS = {
    "PLA": (3.5, 0.36),
    "STEEL": (200.0, 0.30),
    "ALUMINIUM": (69.0, 0.33),
    "CONCRETE": (30.0, 0.20),
    "WOOD": (11.0, 0.30),
    "BONE": (18.0, 0.30),
}


def _poll_mesh(_self, obj):
    return obj is not None and obj.type == "MESH"


def _apply_material_preset(self, _context):
    preset = MATERIAL_PRESETS.get(self.material_preset)
    if preset is None:
        return
    E, nu = preset
    if abs(self.youngs_modulus - E) > 1e-9:
        self.youngs_modulus = E
    if abs(self.poisson_ratio - nu) > 1e-9:
        self.poisson_ratio = nu


class StressSculptObjectSettings(PropertyGroup):
    role: EnumProperty(name="Role", items=ROLE_ITEMS, default="NONE")
    enabled: BoolProperty(
        name="Enabled",
        default=True,
        description="Include this support / load / zone in the computation (handy to compare configurations)",
    )
    # --- loads
    magnitude: FloatProperty(
        name="Force",
        default=1000.0,
        min=0.0,
        soft_max=1.0e6,
        unit="NONE",
        description="Force magnitude in newtons, applied along the arrow (local +Z)",
    )
    load_radius: FloatProperty(
        name="Spread Radius",
        default=0.0,
        min=0.0,
        subtype="DISTANCE",
        description="The force is shared by the nodes within this distance. 0 = automatic (about 1.5 voxels)",
    )
    # --- supports
    fix_x: BoolProperty(name="X", default=True, description="Block movement along world X")
    fix_y: BoolProperty(name="Y", default=True, description="Block movement along world Y")
    fix_z: BoolProperty(name="Z", default=True, description="Block movement along world Z")
    # --- zones
    strength: FloatProperty(
        name="Strength",
        default=0.5,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        description="How strongly the zone rule acts (Thicken / Hollow / Branch)",
    )


class StressSculptVariant(PropertyGroup):
    # ``name`` is inherited from PropertyGroup
    collection: PointerProperty(type=bpy.types.Collection, name="Collection")
    kind: StringProperty(name="Kind", default="")
    compliance: FloatProperty(name="Compliance", default=0.0)
    volume: FloatProperty(name="Volume", default=0.0, subtype="FACTOR")
    max_stress: FloatProperty(name="Max Stress", default=0.0)
    max_displacement: FloatProperty(name="Max Displacement", default=0.0)
    iterations: IntProperty(name="Iterations", default=0)
    settings_json: StringProperty(name="Settings", default="")
    setup_json: StringProperty(name="Setup", default="")
    notes: StringProperty(name="Notes", default="")


class StressSculptSceneSettings(PropertyGroup):
    # ------------------------------------------------------------------ setup
    domain: PointerProperty(
        name="Domain",
        type=bpy.types.Object,
        poll=_poll_mesh,
        description="Closed mesh that bounds the space where material may exist",
    )
    resolution: IntProperty(
        name="Resolution",
        default=40,
        min=8,
        soft_max=120,
        max=256,
        description="Number of voxels along the longest side of the domain (higher = finer but slower)",
    )
    voxelize_method: EnumProperty(
        name="Voxelize",
        items=[
            ("RAY", "Mesh Interior", "Voxels inside the closed domain mesh (ray parity test)"),
            ("BOUNDS", "Bounding Box", "Fill the whole bounding box of the domain"),
        ],
        default="RAY",
    )
    material_preset: EnumProperty(
        name="Material",
        items=[
            ("CUSTOM", "Custom", "Set Young's modulus and Poisson ratio manually"),
            ("PLA", "PLA (3D print)", ""),
            ("STEEL", "Steel", ""),
            ("ALUMINIUM", "Aluminium", ""),
            ("CONCRETE", "Concrete", ""),
            ("WOOD", "Wood", ""),
            ("BONE", "Bone", ""),
        ],
        default="PLA",
        update=_apply_material_preset,
    )
    youngs_modulus: FloatProperty(
        name="Young's Modulus (GPa)", default=3.5, min=1e-6, soft_max=400.0,
        description="Stiffness of the material in gigapascal",
    )
    poisson_ratio: FloatProperty(name="Poisson Ratio", default=0.36, min=0.0, max=0.49)
    solver: EnumProperty(
        name="Solver",
        items=[
            ("MG", "Multigrid", "Conjugate gradient with a geometric multigrid preconditioner (fast)"),
            ("JACOBI", "Jacobi", "Conjugate gradient with a diagonal preconditioner (simple, slower)"),
        ],
        default="MG",
    )
    solver_tolerance: FloatProperty(
        name="Tolerance", default=1e-5, min=1e-10, max=1e-1, precision=7,
        description="Relative residual at which the linear solver stops",
    )
    max_solver_iterations: IntProperty(name="Max Solver Iterations", default=4000, min=10, max=100000)
    domain_as_wire: BoolProperty(
        name="Show Domain as Wire",
        default=True,
        description="Display the domain object as wireframe when results are generated",
    )

    # ---------------------------------------------------------------- analysis
    stress_measure: EnumProperty(
        name="Show",
        items=[
            ("VON_MISES", "Von Mises", "Equivalent stress"),
            ("PRINCIPAL_MAX", "Max Principal (tension)", "Largest principal stress"),
            ("PRINCIPAL_MIN", "Min Principal (compression)", "Smallest principal stress"),
            ("DISPLACEMENT", "Displacement", "Magnitude of the displacement"),
        ],
        default="VON_MISES",
    )
    colormap: EnumProperty(
        name="Colors",
        items=[
            ("HEAT", "Heat", "Black - red - yellow - white"),
            ("COOLWARM", "Cool / Warm", "Blue (compression) - white - red (tension)"),
            ("VIRIDIS", "Viridis", "Perceptually uniform"),
        ],
        default="HEAT",
    )
    color_clip: FloatProperty(
        name="Clip Percentile", default=98.0, min=50.0, max=100.0,
        description="Colors saturate above this percentile (hides stress peaks under point loads)",
    )
    heatmap_on_original: BoolProperty(
        name="Color Original Shape",
        default=True,
        description="Color a copy of the domain mesh; otherwise show the voxel surface the solver sees",
    )

    # --------------------------------------------------------------- evolution
    evo_mode: EnumProperty(
        name="Method",
        items=[
            ("SIMP", "Optimize", "Topology optimization (SIMP): stiffest shape for the given amount of material"),
            ("RULES", "Growth Rules", "Material thickens where stress is high and is carved out where it is low"),
        ],
        default="SIMP",
    )
    volume_fraction: FloatProperty(
        name="Material", default=0.3, min=0.02, max=1.0, subtype="FACTOR",
        description="Fraction of the domain volume that the final structure may use",
    )
    penalty: FloatProperty(name="Penalty", default=3.0, min=1.0, max=6.0, description="SIMP penalization power")
    filter_radius: FloatProperty(
        name="Smoothness", default=1.5, min=1.0, max=6.0,
        description="Filter radius in voxels: larger values give thicker, smoother members",
    )
    move_limit: FloatProperty(name="Move Limit", default=0.2, min=0.01, max=1.0)
    max_iterations: IntProperty(name="Iterations", default=40, min=1, max=500)
    min_change: FloatProperty(
        name="Convergence", default=0.01, min=0.0, max=0.5, precision=3,
        description="Stop when the largest density change falls below this value",
    )
    grow_threshold: FloatProperty(
        name="Thicken Above", default=0.6, min=0.0, max=2.0, subtype="FACTOR",
        description="Relative stress above which material grows",
    )
    hollow_threshold: FloatProperty(
        name="Hollow Below", default=0.25, min=0.0, max=2.0, subtype="FACTOR",
        description="Relative stress below which material is carved out",
    )
    growth_rate: FloatProperty(name="Rate", default=0.15, min=0.01, max=1.0, subtype="FACTOR")
    keep_volume: BoolProperty(
        name="Steer to Material Amount",
        default=True,
        description="Growth rules converge towards the requested material fraction",
    )
    symmetry_x: BoolProperty(name="X", default=False, description="Mirror symmetry across the domain's X mid-plane")
    symmetry_y: BoolProperty(name="Y", default=False, description="Mirror symmetry across the domain's Y mid-plane")
    symmetry_z: BoolProperty(name="Z", default=False, description="Mirror symmetry across the domain's Z mid-plane")
    iso_level: FloatProperty(name="Surface Level", default=0.5, min=0.05, max=0.95, subtype="FACTOR")
    smooth_iterations: IntProperty(name="Smoothing", default=12, min=0, max=200)
    live_update: BoolProperty(name="Live Preview", default=True, description="Update the shape at every iteration")

    # ----------------------------------------------------------- tension lines
    line_source: EnumProperty(
        name="Stress From",
        items=[
            ("ANALYSIS", "Full Domain", "Stress of the solid domain (runs an analysis if needed)"),
            ("EVOLVED", "Evolved Shape", "Stress inside the last evolved structure"),
        ],
        default="ANALYSIS",
    )
    line_families: EnumProperty(
        name="Lines",
        items=[
            ("BOTH", "Tension + Compression", ""),
            ("TENSION", "Tension", "Lines along the largest principal stress"),
            ("COMPRESSION", "Compression", "Lines along the smallest principal stress"),
        ],
        default="BOTH",
    )
    n_seeds: IntProperty(name="Seeds", default=40, min=1, max=2000, description="Number of stress-weighted seed points")
    seed_from_loads: BoolProperty(name="Seed at Loads & Supports", default=True)
    line_step: FloatProperty(name="Step", default=0.5, min=0.1, max=2.0, description="Integration step in voxels")
    line_separation: FloatProperty(
        name="Separation", default=2.0, min=0.5, max=20.0, description="Minimum distance between lines in voxels",
    )
    line_min_stress: FloatProperty(
        name="Min Stress", default=0.02, min=0.0, max=1.0, subtype="FACTOR",
        description="Lines stop where the stress drops below this fraction of the peak",
    )
    branch_depth: IntProperty(name="Branch Levels", default=1, min=0, max=4)
    branch_spacing: FloatProperty(name="Branch Spacing", default=4.0, min=1.0, max=50.0, description="In voxels")
    branch_ratio: FloatProperty(
        name="Branch Threshold", default=0.3, min=0.0, max=1.0, subtype="FACTOR",
        description="A branch sprouts where the crossing stress is at least this fraction of the line's stress",
    )
    random_seed: IntProperty(name="Random Seed", default=0, min=0)
    radius_min: FloatProperty(name="Min Radius", default=0.25, min=0.01, max=10.0, description="Tube radius in voxels")
    radius_max: FloatProperty(name="Max Radius", default=1.2, min=0.01, max=20.0, description="Tube radius in voxels")
    branch_taper: FloatProperty(
        name="Branch Taper", default=0.7, min=0.1, max=1.0, subtype="FACTOR",
        description="Radius multiplier applied at every branching generation",
    )

    # --------------------------------------------------------------- sculpture
    sculpt_source: EnumProperty(
        name="Fuse",
        items=[
            ("LINES", "Lines", "Fuse the tension lines into one organic mesh"),
            ("LINES_AND_SOLID", "Lines + Evolved", "Fuse the lines with the last evolved structure"),
        ],
        default="LINES",
    )
    sculpt_detail: FloatProperty(
        name="Detail", default=2.0, min=0.5, max=6.0,
        description="Sculpture voxels per analysis voxel (higher = finer surface, slower)",
    )
    sculpt_blend: FloatProperty(
        name="Blend", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
        description="How softly crossing members melt into each other",
    )
    sculpt_smooth: IntProperty(name="Smoothing", default=8, min=0, max=200)

    # ------------------------------------------------------------------ relief
    relief_amount: FloatProperty(
        name="Thickness", default=0.03, min=0.0, max=0.5, subtype="FACTOR",
        description="Maximum offset as a fraction of the domain size",
    )
    relief_hollow: BoolProperty(name="Carve Low Stress", default=True, description="Push the surface inwards where stress is low")
    relief_threshold: FloatProperty(
        name="Neutral Stress", default=0.3, min=0.0, max=1.0, subtype="FACTOR",
        description="Relative stress that is neither thickened nor carved",
    )
    relief_gamma: FloatProperty(name="Contrast", default=0.7, min=0.1, max=4.0)
    relief_subdivisions: IntProperty(name="Subdivisions", default=1, min=0, max=4)
    relief_smooth: IntProperty(name="Smoothing", default=5, min=0, max=100)

    # ---------------------------------------------------------------- variants
    variants: CollectionProperty(type=StressSculptVariant)
    active_variant_index: IntProperty(name="Active Variant", default=0, min=0)
    compare_spacing: FloatProperty(name="Spacing", default=1.3, min=1.0, max=5.0, description="Gap between compared variants")

    # ------------------------------------------------------------------ last results (persisted summary)
    last_max_stress: FloatProperty(name="Max Stress", default=0.0)
    last_max_displacement: FloatProperty(name="Max Displacement", default=0.0)
    last_compliance: FloatProperty(name="Compliance", default=0.0)
    last_volume: FloatProperty(name="Volume", default=0.0, subtype="FACTOR")
    last_iterations: IntProperty(name="Iterations", default=0)
    last_kind: StringProperty(name="Last Result", default="")
    show_advanced: BoolProperty(name="Advanced", default=False)


classes = (
    StressSculptObjectSettings,
    StressSculptVariant,
    StressSculptSceneSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.stress_sculpt = PointerProperty(type=StressSculptSceneSettings)
    bpy.types.Object.stress_sculpt = PointerProperty(type=StressSculptObjectSettings)


def unregister():
    del bpy.types.Object.stress_sculpt
    del bpy.types.Scene.stress_sculpt
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
