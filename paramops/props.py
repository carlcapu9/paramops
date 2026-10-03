# SPDX-License-Identifier: GPL-3.0-or-later
"""Settings stored on scatter objects (``Object.paramops``)."""

import math

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)
from bpy.types import PropertyGroup

from .samples import MESH_LIKE


def _update(self, context):
    from . import handlers
    handlers.request_update(self.id_data)


def _update_weight(self, context):
    from . import handlers
    handlers.request_update_users(self)


def depends_on(obj, target, _seen=None):
    """True when scatter ``obj`` (directly or through other scatters) uses ``target``."""
    if obj is None or not getattr(obj, "paramops", None) or not obj.paramops.is_scatter:
        return False
    _seen = _seen or set()
    if obj.name in _seen:
        return False
    _seen.add(obj.name)
    for dep in scatter_dependencies(obj):
        if dep == target or depends_on(dep, target, _seen):
            return True
    return False


def iter_slots(st):
    """(key, slot) pairs of every sample slot of a scatter."""
    yield "default", st.default
    yield "start", st.start
    yield "end", st.end
    yield "evenly", st.evenly
    yield "corner", st.corner
    for i, item in enumerate(st.segment_ids):
        yield "seg:%d" % i, item.slot
    for i, item in enumerate(st.markers):
        yield "marker:%d" % i, item.slot


def slot_objects(slot):
    if slot.source == "COLLECTION":
        if slot.collection is None:
            return []
        return sorted((o for o in slot.collection.all_objects if o.type in MESH_LIKE),
                      key=lambda o: o.name)
    return [slot.object] if slot.object is not None else []


def scatter_dependencies(obj):
    """Objects whose changes affect the scatter ``obj``."""
    st = obj.paramops
    deps = []
    if st.path is not None:
        deps.append(st.path)
    for _, slot in iter_slots(st):
        if slot.enabled:
            deps.extend(slot_objects(slot))
    if st.project and st.project_target is not None:
        deps.append(st.project_target)
    return deps


def _poll_sample(self, obj):
    owner = self.id_data
    if obj == owner or obj.type not in MESH_LIKE:
        return False
    if isinstance(owner, bpy.types.Object) and owner.paramops.path == obj:
        return False
    return not depends_on(obj, owner)


def _poll_curve(self, obj):
    return obj.type == "CURVE" and obj != self.id_data


def _poll_target(self, obj):
    return obj.type == "MESH" and obj != self.id_data


ALIGN_X = [
    ("CENTER", "Center", "Centre of the sample's bounding box on the anchor point"),
    ("PIVOT", "Pivot", "Object origin on the anchor point"),
    ("START", "Start", "Sample starts at the anchor point"),
    ("END", "End", "Sample ends at the anchor point"),
]
ALIGN_Y = [
    ("PIVOT", "Pivot", "Object origin on the curve"),
    ("CENTER", "Center", "Bounding box centred on the curve"),
    ("LEFT", "Left", "Sample entirely on the left side of the curve"),
    ("RIGHT", "Right", "Sample entirely on the right side of the curve"),
]
ALIGN_Z = [
    ("PIVOT", "Pivot", "Object origin on the curve"),
    ("BOTTOM", "Bottom", "Bottom of the bounding box on the curve"),
    ("CENTER", "Center", "Bounding box centred on the curve"),
    ("TOP", "Top", "Top of the bounding box on the curve"),
]


class POPS_SampleSlot(PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=True, update=_update)
    source: EnumProperty(
        name="Source",
        items=[("OBJECT", "Object", "Use a single object", "OBJECT_DATA", 0),
               ("COLLECTION", "Collection", "Use the objects of a collection", "OUTLINER_COLLECTION", 1)],
        default="OBJECT", update=_update)
    object: PointerProperty(name="Object", type=bpy.types.Object, poll=_poll_sample, update=_update)
    collection: PointerProperty(name="Collection", type=bpy.types.Collection, update=_update)
    collection_mode: EnumProperty(
        name="Mode",
        items=[("RANDOM", "Random", "Pick a random object for every module (weighted)"),
               ("SEQUENCE", "Sequence", "Use the objects one after the other (alphabetical order)"),
               ("COMBINE", "Combine", "All objects together form a single module (collection origin = pivot)")],
        default="RANDOM", update=_update)
    seed: IntProperty(name="Seed", default=0, update=_update)

    deform: BoolProperty(name="Deform to Curve", default=True, update=_update,
                         description="Bend the sample along the curve (otherwise it is placed rigidly)")
    keep_vertical: BoolProperty(name="Keep Vertical (slopes)", default=False, update=_update,
                                description="On slopes keep the sample upright and shear it instead of tilting it")
    flat_top: FloatProperty(name="Flat Top", default=0.0, min=0.0, subtype="DISTANCE", unit="LENGTH",
                            update=_update,
                            description="Height of the top band that stays horizontal on slopes")
    flat_center: FloatProperty(name="Flat Center", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                               update=_update,
                               description="How much the rest of the sample is levelled on slopes")
    flat_bottom: FloatProperty(name="Flat Bottom", default=0.0, min=0.0, subtype="DISTANCE", unit="LENGTH",
                               update=_update,
                               description="Height of the bottom band that stays horizontal on slopes")
    flat_reference: EnumProperty(
        name="Level At",
        items=[("CENTER", "Center", "Level at the height of the sample centre"),
               ("LOW", "Lowest", "Level at the lowest end of the sample"),
               ("HIGH", "Highest", "Level at the highest end of the sample")],
        default="CENTER", update=_update)

    align_x: EnumProperty(name="Anchor", items=ALIGN_X, default="CENTER", update=_update,
                          description="Which point of the sample sits on its anchor (corners, evenly, markers)")
    align_y: EnumProperty(name="Side", items=ALIGN_Y, default="PIVOT", update=_update)
    align_z: EnumProperty(name="Height", items=ALIGN_Z, default="PIVOT", update=_update)
    orient: EnumProperty(
        name="Orientation",
        items=[("AUTO", "Bisector", "At corners follow the bisector of the two directions"),
               ("INCOMING", "Incoming", "Follow the incoming direction"),
               ("OUTGOING", "Outgoing", "Follow the outgoing direction")],
        default="AUTO", update=_update)

    offset: FloatVectorProperty(name="Offset", size=3, subtype="TRANSLATION", unit="LENGTH",
                                default=(0.0, 0.0, 0.0), update=_update,
                                description="Offset along the curve (X), to the side (Y) and up (Z)")
    rotation: FloatVectorProperty(name="Rotation", size=3, subtype="EULER", unit="ROTATION",
                                  default=(0.0, 0.0, 0.0), update=_update)
    scale: FloatVectorProperty(name="Scale", size=3, subtype="XYZ", default=(1.0, 1.0, 1.0), update=_update)
    mirror_x: BoolProperty(name="Mirror X", default=False, update=_update,
                           description="Flip the sample along the curve direction")
    mirror_y: BoolProperty(name="Mirror Y", default=False, update=_update,
                           description="Flip the sample to the other side of the curve")
    padding_before: FloatProperty(name="Padding Before", default=0.0, subtype="DISTANCE", unit="LENGTH",
                                  update=_update, description="Extra space before the sample")
    padding_after: FloatProperty(name="Padding After", default=0.0, subtype="DISTANCE", unit="LENGTH",
                                 update=_update, description="Extra space after the sample")

    random_offset: FloatVectorProperty(name="Random Offset", size=3, subtype="TRANSLATION", unit="LENGTH",
                                       min=0.0, default=(0.0, 0.0, 0.0), update=_update)
    random_rotation: FloatVectorProperty(name="Random Rotation", size=3, subtype="EULER", unit="ROTATION",
                                         min=0.0, max=math.pi, default=(0.0, 0.0, 0.0), update=_update)
    random_scale: FloatProperty(name="Random Scale", default=0.0, min=0.0, max=0.95, subtype="FACTOR",
                                update=_update)
    random_flip_x: FloatProperty(name="Random Flip X", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                                 update=_update, description="Probability of mirroring along the curve")
    random_flip_y: FloatProperty(name="Random Flip Y", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                                 update=_update, description="Probability of mirroring across the curve")

    show_expanded: BoolProperty(name="Show Options", default=False)


class POPS_SegmentRef(PropertyGroup):
    spline: IntProperty()
    index: IntProperty()
    co_a: FloatVectorProperty(size=3)
    co_b: FloatVectorProperty(size=3)


class POPS_PointRef(PropertyGroup):
    spline: IntProperty()
    index: IntProperty()
    co: FloatVectorProperty(size=3)


class POPS_SegmentID(PropertyGroup):
    name: StringProperty(name="Name", default="ID")
    slot: PointerProperty(type=POPS_SampleSlot)
    refs: CollectionProperty(type=POPS_SegmentRef)


class POPS_Marker(PropertyGroup):
    name: StringProperty(name="Name", default="Marker")
    slot: PointerProperty(type=POPS_SampleSlot)
    refs: CollectionProperty(type=POPS_PointRef)
    positions: StringProperty(
        name="Positions", default="", update=_update,
        description="Extra marker positions along every spline, separated by commas. "
                    "Distances (2.5), percentages (50%) or negative values measured from the end (-1)")


class POPS_Settings(PropertyGroup):
    is_scatter: BoolProperty(default=False)
    path: PointerProperty(name="Curve", type=bpy.types.Object, poll=_poll_curve, update=_update,
                          description="Curve the samples are scattered along")
    auto_update: BoolProperty(name="Live Update", default=True, update=_update,
                              description="Rebuild automatically when the curve, the samples or the settings change")

    default: PointerProperty(type=POPS_SampleSlot)
    start: PointerProperty(type=POPS_SampleSlot)
    end: PointerProperty(type=POPS_SampleSlot)
    evenly: PointerProperty(type=POPS_SampleSlot)
    corner: PointerProperty(type=POPS_SampleSlot)
    segment_ids: CollectionProperty(type=POPS_SegmentID)
    segment_ids_index: IntProperty(default=0)
    markers: CollectionProperty(type=POPS_Marker)
    markers_index: IntProperty(default=0)

    fit_mode: EnumProperty(
        name="Fit",
        items=[("ROUND", "Adaptive", "Stretch or squash the Default sample to fit a whole number of copies"),
               ("CEIL", "Adaptive (Shrink)", "Never longer than the original: add copies and squash them"),
               ("FLOOR", "Adaptive (Stretch)", "Never shorter than the original: remove copies and stretch them"),
               ("COUNT", "Count", "Fixed number of copies in every section"),
               ("FIXED", "Real Size", "Keep the real size and handle the remaining space")],
        default="ROUND", update=_update)
    fit_count: IntProperty(name="Count", default=3, min=1, soft_max=100, update=_update)
    spacing: FloatProperty(name="Spacing", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_update,
                           description="Gap between consecutive Default samples (negative overlaps them)")
    fixed_align: EnumProperty(
        name="Align",
        items=[("START", "Start", "Copies start at the beginning of each section"),
               ("CENTER", "Center", "Copies are centred in each section"),
               ("END", "End", "Copies end at the end of each section"),
               ("DISTRIBUTE", "Distribute", "Spread the remaining space between the copies")],
        default="START", update=_update)
    fixed_remainder: EnumProperty(
        name="Remainder",
        items=[("NONE", "Empty", "Leave the remaining space empty"),
               ("SCALE", "Scale", "Fill it with a squashed copy"),
               ("SLICE", "Slice", "Fill it with a copy cut to size")],
        default="SLICE", update=_update)

    evenly_mode: EnumProperty(
        name="Mode",
        items=[("SPACING", "Spacing", "Exact distance between Evenly samples"),
               ("FIT", "Fit", "Equal intervals, never longer than the spacing"),
               ("COUNT", "Count", "Fixed number of Evenly samples per section")],
        default="SPACING", update=_update)
    evenly_spacing: FloatProperty(name="Spacing", default=2.0, min=0.01, subtype="DISTANCE", unit="LENGTH",
                                  update=_update)
    evenly_count: IntProperty(name="Count", default=1, min=0, soft_max=100, update=_update)
    evenly_min_gap: FloatProperty(name="Min Gap", default=0.5, min=0.0, subtype="DISTANCE", unit="LENGTH",
                                  update=_update,
                                  description="Minimum distance between the last Evenly sample and the end of the section")
    evenly_align: EnumProperty(
        name="Align",
        items=[("START", "Start", "Measure from the start of each section"),
               ("CENTER", "Center", "Centre the pattern in each section"),
               ("END", "End", "Measure from the end of each section")],
        default="START", update=_update)

    corner_threshold: FloatProperty(name="Corner Angle", default=math.radians(10.0), min=0.0,
                                    max=math.radians(179.0), subtype="ANGLE", update=_update,
                                    description="Control points that turn more than this angle are corners")
    corner_radius: FloatProperty(name="Corner Radius", default=0.0, min=0.0, subtype="DISTANCE",
                                 unit="LENGTH", update=_update,
                                 description="Round the corners of the curve (non-destructive)")
    corner_slide: FloatProperty(name="Corner Slide", default=0.0, subtype="DISTANCE", unit="LENGTH",
                                update=_update,
                                description="Move the section break (and the Corner sample) along the curve")

    reverse: BoolProperty(name="Reverse Direction", default=False, update=_update)
    clip_mode: EnumProperty(name="Trim", items=[("DISTANCE", "Distance", ""), ("PERCENT", "Percentage", "")],
                            default="DISTANCE", update=_update)
    clip_start: FloatProperty(name="Start", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_update,
                              description="Trim (or extend, if negative) the start of open curves")
    clip_end: FloatProperty(name="End", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_update,
                            description="Trim (or extend, if negative) the end of open curves")
    clip_start_pct: FloatProperty(name="Start", default=0.0, min=-100.0, max=100.0, subtype="PERCENTAGE",
                                  update=_update)
    clip_end_pct: FloatProperty(name="End", default=0.0, min=-100.0, max=100.0, subtype="PERCENTAGE",
                                update=_update)
    resolution: IntProperty(name="Curve Resolution", default=0, min=0, max=128, update=_update,
                            description="Steps per curved segment (0 = automatic)")
    twist_mode: EnumProperty(
        name="Twist",
        items=[("Z_UP", "Z-Up", "Samples stay level (sideways axis horizontal)"),
               ("MINIMUM", "Minimum", "Minimum twist, for 3D paths such as loops and helices")],
        default="Z_UP", update=_update)
    use_tilt: BoolProperty(name="Use Tilt", default=True, update=_update,
                           description="Rotate samples with the tilt of the curve points")
    offset_y: FloatProperty(name="Side Offset", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_update)
    offset_z: FloatProperty(name="Height Offset", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_update)

    project: BoolProperty(name="Conform to Surface", default=False, update=_update,
                          description="Drop the curve onto a surface (e.g. terrain) along -Z")
    project_target: PointerProperty(name="Surface", type=bpy.types.Object, poll=_poll_target, update=_update)
    project_offset: FloatProperty(name="Height", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_update)
    project_step: FloatProperty(name="Step", default=0.25, min=0.01, subtype="DISTANCE", unit="LENGTH",
                                update=_update, description="Sampling distance used to follow the surface")

    seed: IntProperty(name="Seed", default=0, update=_update)
    preview: EnumProperty(
        name="Display",
        items=[("FULL", "Full", "Real geometry"),
               ("BOXES", "Boxes", "Bounding boxes only (fast preview for heavy scenes)")],
        default="FULL", update=_update)
    max_modules: IntProperty(name="Max Samples", default=20000, min=1, soft_max=200000, update=_update,
                             description="Safety limit for the number of placed samples")
    apply_sample_transform: BoolProperty(
        name="Use Sample Rotation/Scale", default=True, update=_update,
        description="Use the rotation and scale of the sample objects (the location is always ignored)")
    update_on_frame: BoolProperty(name="Update on Frame Change", default=False,
                                  description="Rebuild on every frame (animated curves or samples)")

    stat_modules: IntProperty(default=0)
    stat_verts: IntProperty(default=0)
    stat_faces: IntProperty(default=0)
    stat_time: FloatProperty(default=0.0)
    stat_truncated: BoolProperty(default=False)
    last_error: StringProperty(default="")
    topo_signature: StringProperty(default="")

    # Used on curve objects: the scatter shown in the panel when the curve is active.
    curve_scatter: PointerProperty(type=bpy.types.Object)


classes = (POPS_SampleSlot, POPS_SegmentRef, POPS_PointRef, POPS_SegmentID, POPS_Marker, POPS_Settings)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.paramops = PointerProperty(type=POPS_Settings)
    bpy.types.Object.paramops_weight = FloatProperty(
        name="Weight", default=1.0, min=0.0, soft_max=10.0, update=_update_weight,
        description="Probability weight when this object is picked at random from a collection")


def unregister():
    del bpy.types.Object.paramops_weight
    del bpy.types.Object.paramops
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
