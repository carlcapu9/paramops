# SPDX-License-Identifier: GPL-3.0-or-later
"""The CurveForge style editor: node tree and nodes."""

import math

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Node, NodeTree

from .engine.graph import MATH_OPS, VARIABLES

TREE_ID = "CF_StyleTree"
MESH_LIKE = {"MESH", "CURVE", "SURFACE", "FONT", "META"}


_registering = [False]
MESH_TYPES = {"MESH", "CURVE", "SURFACE", "FONT", "META"}


def _changed(self, context):
    from . import live
    live.tree_changed(self.id_data)


class CF_StyleTree(NodeTree):
    """CurveForge style: segments and operators feeding Linear generators"""

    bl_idname = TREE_ID
    bl_label = "CurveForge Style"
    bl_icon = "MOD_ARRAY"

    def update(self):
        from . import live
        if _registering[0]:
            # Blender updates the trees while each node class is (un)registered: the
            # nodes are only half defined then.
            return
        for node in self.nodes:
            if hasattr(node, "cf_sync_inputs"):
                node.cf_sync_inputs()
        live.tree_changed(self, deferred=True)


TRI_STATE = [("DEFAULT", "Default", "Use the setting of the generator"),
             ("ON", "On", "Always on for this segment"),
             ("OFF", "Off", "Always off for this segment")]
VARIABLE_ITEMS = [(name, name.replace("_", " ").title().replace(" Id", " ID"), desc) for name, desc in VARIABLES]
COMPARE_ITEMS = [
    ("EQUAL", "Equal", "Variable == Value"), ("NOT_EQUAL", "Not Equal", "Variable != Value"),
    ("LESS", "Less Than", "Variable < Value"), ("LESS_EQUAL", "Less or Equal", "Variable <= Value"),
    ("GREATER", "Greater Than", "Variable > Value"), ("GREATER_EQUAL", "Greater or Equal", "Variable >= Value"),
    ("BETWEEN", "Between", "Value <= Variable <= Value 2"), ("EVEN", "Even", "Variable is even"),
    ("ODD", "Odd", "Variable is odd"), ("EVERY", "Every N", "Every N-th value, starting at Value 2"),
]
MATH_ITEMS = [(k, k.replace("_", " ").title(), "") for k in MATH_OPS]


CATEGORY_COLORS = {
    "INPUT": (0.16, 0.30, 0.28),
    "GENERATOR": (0.17, 0.24, 0.40),
    "OPERATOR": (0.38, 0.25, 0.12),
    "NUMBER": (0.24, 0.24, 0.26),
}


class CFNode:
    bl_width_default = 170
    cf_category = ""

    cf_panel: BoolProperty(name="Show in Panel", default=False, update=_changed,
                           description="Show the settings of this node in the 3D View sidebar")

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_ID

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)

    def draw_panel(self, context, layout):
        """Settings shown in the 3D View sidebar when ``cf_panel`` is on."""
        self.draw_buttons_ext(context, layout)

    def cf_colorize(self):
        color = CATEGORY_COLORS.get(self.cf_category)
        if color is not None:
            self.use_custom_color = True
            self.color = color


class DynamicInputs:
    """Nodes with a growing list of Segment inputs (one free socket at the end)."""

    cf_input_label = "Input"
    cf_input_extra = ""

    def init(self, context):
        self.cf_colorize()
        for _ in range(2):
            self.inputs.new("CF_SocketSegment", self.cf_input_label)
        self.outputs.new("CF_SocketSegment", "Segment")
        self.cf_sync_inputs()

    def segment_inputs(self):
        return [s for s in self.inputs if s.bl_idname == "CF_SocketSegment"]

    def cf_sync_inputs(self):
        ins = self.segment_inputs()
        if ins and ins[-1].links:
            self.inputs.new("CF_SocketSegment", self.cf_input_label)
        else:
            while len(ins) > 2 and not ins[-1].links and not ins[-2].links:
                self.inputs.remove(ins[-1])
                ins = self.segment_inputs()
        for i, sock in enumerate(self.segment_inputs()):
            name = self.input_name(i)
            if sock.name != name:
                sock.name = name

    def input_name(self, i):
        return "%s %d" % (self.cf_input_label, i + 1)

    def update(self):
        self.cf_sync_inputs()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _poll_curve(self, obj):
    return obj.type == "CURVE"


def _poll_mesh(self, obj):
    return obj.type in MESH_LIKE


class CF_NodeSpline(CFNode, Node):
    """A curve object: every spline of it (or the chosen ones) is used by the generator"""

    bl_idname = "CF_NodeSpline"
    bl_label = "Spline"
    bl_icon = "CURVE_DATA"
    cf_category = "INPUT"

    curve: PointerProperty(name="Curve", type=bpy.types.Object, poll=_poll_curve, update=_changed)
    splines: StringProperty(name="Splines", default="", update=_changed,
                            description="Spline numbers to use, e.g. '0, 2, 4-6' (empty = all)")
    reverse: BoolProperty(name="Reverse", default=False, update=_changed,
                          description="Run along the splines in the opposite direction")
    resolution: IntProperty(name="Resolution", default=0, min=0, max=128, update=_changed,
                            description="Steps per curved segment (0 = automatic)")
    twist: EnumProperty(name="Twist", items=[("Z_UP", "Z Up", "Segments stay level"),
                                             ("MINIMUM", "Minimum", "Minimum twist, for 3D paths")],
                        default="Z_UP", update=_changed)
    use_tilt: BoolProperty(name="Use Tilt", default=True, update=_changed,
                           description="Rotate the segments with the tilt of the curve points")

    def init(self, context):
        self.cf_colorize()
        self.outputs.new("CF_SocketSpline", "Spline")

    def draw_buttons(self, context, layout):
        layout.prop(self, "curve", text="")
        row = layout.row(align=True)
        row.prop(self, "reverse", toggle=True, icon="ARROW_LEFTRIGHT")
        row.prop(self, "use_tilt", toggle=True)

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, "curve")
        layout.prop(self, "splines")
        layout.prop(self, "reverse")
        layout.prop(self, "twist")
        layout.prop(self, "use_tilt")
        layout.prop(self, "resolution")


class CF_NodeSegment(CFNode, Node):
    """A piece of geometry (an object or a collection) to repeat along the curve"""

    bl_idname = "CF_NodeSegment"
    bl_label = "Segment"
    bl_icon = "MESH_CUBE"
    bl_width_default = 190
    cf_category = "INPUT"

    source: EnumProperty(name="Source", items=[("OBJECT", "Object", "", "OBJECT_DATA", 0),
                                               ("COLLECTION", "Collection", "", "OUTLINER_COLLECTION", 1)],
                         default="OBJECT", update=_changed)
    object: PointerProperty(name="Object", type=bpy.types.Object, poll=_poll_mesh, update=_changed)
    collection: PointerProperty(name="Collection", type=bpy.types.Collection, update=_changed)
    collection_mode: EnumProperty(
        name="Use", items=[("RANDOM", "Random", "A random object of the collection for every segment"),
                           ("SEQUENCE", "Sequence", "The objects one after the other (by name)"),
                           ("COMBINE", "Combine", "All the objects together as one segment")],
        default="RANDOM", update=_changed)
    seed: IntProperty(name="Seed", default=0, update=_changed)
    use_transform: BoolProperty(name="Object Rotation & Scale", default=True, update=_changed,
                                description="Use the rotation and scale of the object (its location is ignored)")

    bend: EnumProperty(name="Bend", items=TRI_STATE, default="DEFAULT", update=_changed,
                       description="Deform the segment along the curve")
    upright: EnumProperty(name="Vertical", items=TRI_STATE, default="DEFAULT", update=_changed,
                          description="Keep the segment vertical on slopes (shear instead of tilt)")
    slice: EnumProperty(name="Slice", items=TRI_STATE, default="DEFAULT", update=_changed,
                        description="Allow cutting the segment at the ends and at the clipping area")
    adaptive: EnumProperty(name="Adaptive", items=[("ON", "On", "Can be stretched to fit"),
                                                   ("OFF", "Off", "Keeps its length when fitting")],
                           default="ON", update=_changed)
    align_x: EnumProperty(name="X", items=[("AUTO", "Auto", "Start for repeated segments, pivot for "
                                                            "corners, centre for evenly and markers"),
                                           ("PIVOT", "Pivot", ""), ("MIN", "Left", ""),
                                           ("CENTER", "Center", ""), ("MAX", "Right", "")],
                          default="AUTO", update=_changed)
    align_y: EnumProperty(name="Y", items=[("PIVOT", "Pivot", ""), ("MIN", "Right", ""),
                                           ("CENTER", "Center", ""), ("MAX", "Left", "")],
                          default="PIVOT", update=_changed)
    align_z: EnumProperty(name="Z", items=[("PIVOT", "Pivot", ""), ("MIN", "Bottom", ""),
                                           ("CENTER", "Center", ""), ("MAX", "Top", "")],
                          default="PIVOT", update=_changed)
    offset: FloatVectorProperty(name="Offset", size=3, subtype="TRANSLATION", unit="LENGTH",
                                default=(0.0, 0.0, 0.0), update=_changed)
    rotation: FloatVectorProperty(name="Rotation", size=3, subtype="EULER", unit="ROTATION",
                                  default=(0.0, 0.0, 0.0), update=_changed)
    scale: FloatVectorProperty(name="Scale", size=3, subtype="XYZ", default=(1.0, 1.0, 1.0), update=_changed)
    mirror_x: BoolProperty(name="Mirror X", default=False, update=_changed)
    mirror_y: BoolProperty(name="Mirror Y", default=False, update=_changed)
    mirror_z: BoolProperty(name="Mirror Z", default=False, update=_changed)
    use_size: BoolProperty(name="Custom Size", default=False, update=_changed,
                           description="Occupy a custom length along the curve instead of the bounding box")
    size: FloatProperty(name="Size", default=1.0, min=0.001, subtype="DISTANCE", unit="LENGTH", update=_changed)
    padding: FloatVectorProperty(name="Padding", size=2, subtype="NONE", default=(0.0, 0.0), update=_changed,
                                 description="Extra space before and after the segment")
    orient: EnumProperty(name="Corner Orientation",
                         items=[("BISECTOR", "Bisector", "Halfway between the two directions"),
                                ("INCOMING", "Incoming", "Direction before the corner"),
                                ("OUTGOING", "Outgoing", "Direction after the corner")],
                         default="BISECTOR", update=_changed)
    instance: EnumProperty(name="Instance", items=TRI_STATE, default="DEFAULT", update=_changed,
                           description="Place rigid copies as instances (lighter scenes)")

    def init(self, context):
        self.cf_colorize()
        self.outputs.new("CF_SocketSegment", "Segment")

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "source", text="", icon_only=True)
        row.prop(self, "object" if self.source == "OBJECT" else "collection", text="")
        row = layout.row(align=True)
        row.prop(self, "bend", text="Bend")
        row = layout.row(align=True)
        row.prop(self, "upright", text="Vertical")

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, "source", expand=True)
        if self.source == "OBJECT":
            layout.prop(self, "object")
        else:
            layout.prop(self, "collection")
            layout.prop(self, "collection_mode")
            if self.collection_mode == "RANDOM":
                layout.prop(self, "seed")
                if self.collection is not None:
                    col = layout.column(align=True)
                    col.label(text="Weights")
                    objs = sorted((o for o in self.collection.all_objects if o.type in MESH_TYPES),
                                  key=lambda o: o.name)
                    for ob in objs[:24]:
                        col.prop(ob, "cf_weight", text=ob.name)
        layout.prop(self, "use_transform")
        col = layout.column(align=True)
        col.label(text="Deformation")
        col.prop(self, "bend")
        col.prop(self, "upright")
        col.prop(self, "slice")
        col.prop(self, "adaptive")
        col.prop(self, "instance")
        col = layout.column(align=True)
        col.label(text="Alignment")
        row = col.row(align=True)
        row.prop(self, "align_x", text="")
        row.prop(self, "align_y", text="")
        row.prop(self, "align_z", text="")
        col.prop(self, "offset", text="")
        layout.prop(self, "orient")
        col = layout.column(align=True)
        col.label(text="Transform")
        col.prop(self, "rotation", text="")
        col.prop(self, "scale", text="")
        row = layout.row(align=True)
        row.label(text="Mirror")
        row.prop(self, "mirror_x", text="X", toggle=True)
        row.prop(self, "mirror_y", text="Y", toggle=True)
        row.prop(self, "mirror_z", text="Z", toggle=True)
        row = layout.row(align=True)
        row.prop(self, "use_size")
        sub = row.row()
        sub.active = self.use_size
        sub.prop(self, "size", text="")
        layout.prop(self, "padding")


class CF_NodeGap(CFNode, Node):
    """An empty segment: leaves a gap of the given length"""

    bl_idname = "CF_NodeGap"
    bl_label = "Empty Segment"
    bl_icon = "SELECT_SET"
    cf_category = "INPUT"

    def init(self, context):
        self.cf_colorize()
        sock = self.inputs.new("CF_SocketDistance", "Length")
        sock.default_value = 1.0
        self.outputs.new("CF_SocketSegment", "Segment")

    def draw_buttons(self, context, layout):
        pass

    def draw_panel(self, context, layout):
        sock = self.inputs.get("Length")
        if sock is not None and not sock.is_linked:
            layout.prop(sock, "default_value", text="Length")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GEN_SEGMENT_INPUTS = ("Start", "End", "Corner", "Evenly", "Default", "Marker")


class CF_NodeLinear(CFNode, Node):
    """Linear generator: lays the segments out along the splines"""

    bl_idname = "CF_NodeLinear"
    bl_label = "Linear Generator"
    bl_icon = "CURVE_PATH"
    bl_width_default = 230
    cf_category = "GENERATOR"

    enabled: BoolProperty(name="Enabled", default=True, update=_changed)
    # Bounds
    clip_percent: BoolProperty(name="Clip in %", default=False, update=_changed,
                               description="Clip Start / End are percentages of the spline length")
    extend: BoolProperty(name="Extend", default=False, update=_changed,
                         description="Negative clipping extends the splines beyond their ends")
    # Default segments
    adaptive: BoolProperty(name="Adaptive", default=True, update=_changed,
                           description="Stretch the Default segments so a whole number fits each section")
    fit: EnumProperty(name="Fit", items=[("ROUND", "Nearest", "Closest whole number of segments"),
                                         ("CEIL", "Shrink", "Segments never longer than the original"),
                                         ("FLOOR", "Stretch", "Segments never shorter than the original"),
                                         ("COUNT", "Count", "Fixed number of segments per section")],
                      default="ROUND", update=_changed)
    count: IntProperty(name="Count", default=3, min=1, soft_max=100, update=_changed)
    align: EnumProperty(name="Align", items=[("START", "Start", ""), ("CENTER", "Center", ""),
                                             ("END", "End", ""), ("SPREAD", "Spread", "Distribute the "
                                                                                       "remaining space")],
                        default="START", update=_changed)
    remainder: EnumProperty(name="Remainder", items=[("SLICE", "Slice", "Fill with a segment cut to size"),
                                                     ("SCALE", "Scale", "Fill with a squashed segment"),
                                                     ("EMPTY", "Empty", "Leave the space empty")],
                            default="SLICE", update=_changed)
    # Corners
    corner_mode: EnumProperty(name="Corners", items=[("SHARP", "Sharp", "Points that turn more than the angle"),
                                                     ("ALL", "All Points", "Every point of the curve"),
                                                     ("NONE", "None", "No corners")],
                              default="SHARP", update=_changed)
    corner_angle: FloatProperty(name="Angle", default=math.radians(2.0), min=0.0, max=math.pi,
                                subtype="ANGLE", update=_changed)
    corner_split: BoolProperty(name="Split at Corners", default=True, update=_changed,
                               description="Corners split the sections even without a Corner segment")
    split_ids: BoolProperty(name="Split at ID Changes", default=True, update=_changed,
                            description="Segment ID changes split the sections")
    # Evenly
    evenly_mode: EnumProperty(name="Evenly", items=[("DISTANCE", "Distance", ""), ("COUNT", "Count", "")],
                              default="DISTANCE", update=_changed)
    evenly_offset: FloatProperty(name="Offset", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_changed)
    evenly_scope: EnumProperty(name="Measure", items=[("SECTION", "Per Section", "Restart at every corner"),
                                                      ("SPLINE", "Whole Spline", "Ignore corners")],
                               default="SECTION", update=_changed)
    # Markers
    marker_mode: EnumProperty(name="Markers", items=[("NONE", "None", ""),
                                                     ("VERTICES", "Points", "At curve points: '1, 3, 5-7, all'"),
                                                     ("DISTANCES", "Distances", "At distances: '2.5, 50%, -1'"),
                                                     ("REPEAT", "Repeat", "Every Step, starting at Offset")],
                              default="NONE", update=_changed)
    marker_list: StringProperty(name="List", default="", update=_changed)
    marker_step: FloatProperty(name="Step", default=5.0, min=0.01, subtype="DISTANCE", unit="LENGTH",
                               update=_changed)
    marker_offset: FloatProperty(name="Offset", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_changed)
    # Deformation defaults
    bend: BoolProperty(name="Bend", default=True, update=_changed,
                       description="Deform segments along the curve (segments can override it)")
    upright: BoolProperty(name="Vertical", default=False, update=_changed,
                          description="Keep segments vertical on slopes (segments can override it)")
    slice: BoolProperty(name="Slice", default=True, update=_changed,
                        description="Allow cutting segments (segments can override it)")
    instance: BoolProperty(name="Instancing", default=False, update=_changed,
                           description="Place rigid segments as instances (segments can override it)")
    # Output
    uv_mode: EnumProperty(name="UV", items=[("KEEP", "Keep", "Keep the UVs of the segments"),
                                            ("RAIL", "Along Curve", "U follows the distance along the curve")],
                          default="KEEP", update=_changed)
    uv_scale: FloatProperty(name="UV Scale", default=1.0, update=_changed)
    weld: BoolProperty(name="Weld", default=False, update=_changed,
                       description="Merge vertices closer than the distance")
    weld_distance: FloatProperty(name="Weld Distance", default=0.001, min=0.0, subtype="DISTANCE",
                                 unit="LENGTH", update=_changed)
    clip_curve: PointerProperty(name="Clipping Area", type=bpy.types.Object, poll=_poll_curve, update=_changed,
                                description="Closed curve: keep only the geometry inside (or outside) of it")
    clip_inside: BoolProperty(name="Keep Inside", default=True, update=_changed)
    max_segments: IntProperty(name="Max Segments", default=50000, min=1, update=_changed)
    seed: IntProperty(name="Seed", default=0, update=_changed)

    def init(self, context):
        self.cf_colorize()
        spl = self.inputs.new("CF_SocketSpline", "Spline")
        spl.link_limit = 4095
        for name in GEN_SEGMENT_INPUTS:
            self.inputs.new("CF_SocketSegment", name)
        for sock_type, name, value in (("CF_SocketDistance", "Spacing", 0.0),
                                       ("CF_SocketDistance", "Evenly Distance", 2.0),
                                       ("CF_SocketInt", "Evenly Count", 1),
                                       ("CF_SocketFloat", "Clip Start", 0.0),
                                       ("CF_SocketFloat", "Clip End", 0.0),
                                       ("CF_SocketDistance", "Offset Y", 0.0),
                                       ("CF_SocketDistance", "Offset Z", 0.0)):
            self.inputs.new(sock_type, name).default_value = value

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "enabled", text="")
        row.prop(self, "adaptive", toggle=True)
        row.prop(self, "fit", text="")
        row = layout.row(align=True)
        row.prop(self, "bend", toggle=True)
        row.prop(self, "upright", toggle=True)
        row.prop(self, "slice", toggle=True)
        layout.prop(self, "marker_mode", text="Markers")
        if self.marker_mode in {"VERTICES", "DISTANCES"}:
            layout.prop(self, "marker_list", text="")

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, "enabled")
        box = layout.box()
        box.label(text="Bounds", icon="ARROW_LEFTRIGHT")
        box.prop(self, "clip_percent")
        box.prop(self, "extend")
        box.prop(self, "clip_curve")
        if self.clip_curve is not None:
            box.prop(self, "clip_inside")
        box = layout.box()
        box.label(text="Default Segments", icon="MOD_ARRAY")
        box.prop(self, "adaptive")
        if self.adaptive:
            box.prop(self, "fit")
            if self.fit == "COUNT":
                box.prop(self, "count")
        else:
            box.prop(self, "align")
            box.prop(self, "remainder")
        box = layout.box()
        box.label(text="Corners", icon="MOD_SIMPLIFY")
        box.prop(self, "corner_mode")
        if self.corner_mode == "SHARP":
            box.prop(self, "corner_angle")
        box.prop(self, "corner_split")
        box.prop(self, "split_ids")
        box = layout.box()
        box.label(text="Evenly", icon="ALIGN_JUSTIFY")
        box.prop(self, "evenly_mode")
        box.prop(self, "evenly_offset")
        box.prop(self, "evenly_scope")
        box = layout.box()
        box.label(text="Markers", icon="PINNED")
        box.prop(self, "marker_mode")
        if self.marker_mode in {"VERTICES", "DISTANCES"}:
            box.prop(self, "marker_list")
        elif self.marker_mode == "REPEAT":
            box.prop(self, "marker_step")
            box.prop(self, "marker_offset")
        box = layout.box()
        box.label(text="Deformation", icon="MOD_CURVE")
        row = box.row(align=True)
        row.prop(self, "bend", toggle=True)
        row.prop(self, "upright", toggle=True)
        row.prop(self, "slice", toggle=True)
        box.prop(self, "instance")
        box = layout.box()
        box.label(text="Output", icon="MESH_DATA")
        box.prop(self, "uv_mode")
        if self.uv_mode == "RAIL":
            box.prop(self, "uv_scale")
        row = box.row(align=True)
        row.prop(self, "weld")
        sub = row.row()
        sub.active = self.weld
        sub.prop(self, "weld_distance", text="")
        box.prop(self, "max_segments")
        box.prop(self, "seed")

    def draw_panel(self, context, layout):
        col = layout.column(align=True)
        for name in ("Spacing", "Evenly Distance", "Evenly Count", "Clip Start", "Clip End"):
            sock = self.inputs.get(name)
            if sock is not None and not sock.is_linked:
                col.prop(sock, "default_value", text=name)
        layout.prop(self, "adaptive")
        layout.prop(self, "fit")
        row = layout.row(align=True)
        row.prop(self, "bend", toggle=True)
        row.prop(self, "upright", toggle=True)
        layout.prop(self, "seed")


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class CF_NodeCompose(DynamicInputs, CFNode, Node):
    """Joins the segments of the inputs into one segment"""

    bl_idname = "CF_NodeCompose"
    bl_label = "Compose"
    bl_icon = "LINKED"
    cf_category = "OPERATOR"
    cf_input_label = "Part"

    mode: EnumProperty(name="Mode", items=[("SEQUENCE", "One After the Other", "Parts are placed in a row"),
                                           ("OVERLAP", "Overlapped", "Parts share the same origin")],
                       default="SEQUENCE", update=_changed)
    gap: FloatProperty(name="Gap", default=0.0, subtype="DISTANCE", unit="LENGTH", update=_changed)

    def draw_buttons(self, context, layout):
        layout.prop(self, "mode", text="")
        if self.mode == "SEQUENCE":
            layout.prop(self, "gap")


class CF_NodeSequence(DynamicInputs, CFNode, Node):
    """Uses the inputs one after the other (each Count times)"""

    bl_idname = "CF_NodeSequence"
    bl_label = "Sequence"
    bl_icon = "SEQUENCE"
    cf_category = "OPERATOR"
    cf_input_extra = "count"

    scope: EnumProperty(name="Restart", items=[("SPLINE", "Every Spline", ""),
                                               ("SECTION", "Every Section", "Restart after corners and markers"),
                                               ("INPUT", "Every Input", "Separate count per generator input"),
                                               ("STYLE", "Never", "One count for the whole style")],
                        default="SPLINE", update=_changed)
    offset: IntProperty(name="Offset", default=0, update=_changed, description="Start the sequence later")

    def draw_buttons(self, context, layout):
        layout.prop(self, "scope", text="")
        layout.prop(self, "offset")


class CF_NodeRandomize(DynamicInputs, CFNode, Node):
    """Picks a random input for every segment (with weights)"""

    bl_idname = "CF_NodeRandomize"
    bl_label = "Randomize"
    bl_icon = "RNDCURVE"
    cf_category = "OPERATOR"
    cf_input_extra = "weight"

    seed: IntProperty(name="Seed", default=0, update=_changed)
    per: EnumProperty(name="Per", items=[("SEGMENT", "Segment", "A new choice for every segment"),
                                         ("SPLINE", "Spline", "The same choice for a whole spline")],
                      default="SEGMENT", update=_changed)

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "seed")
        row.prop(self, "per", text="")


class CF_NodeConditional(CFNode, Node):
    """Chooses between two inputs with a condition on the segment variables"""

    bl_idname = "CF_NodeConditional"
    bl_label = "Conditional"
    bl_icon = "DECORATE_DRIVER"
    bl_width_default = 200
    cf_category = "OPERATOR"

    variable: EnumProperty(name="Variable", items=VARIABLE_ITEMS, default="index", update=_changed)
    compare: EnumProperty(name="Is", items=COMPARE_ITEMS, default="EQUAL", update=_changed)
    value: FloatProperty(name="Value", default=0.0, update=_changed)
    value2: FloatProperty(name="Value 2", default=0.0, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketSegment", "True")
        self.inputs.new("CF_SocketSegment", "False")
        self.inputs.new("CF_SocketBool", "Condition")
        self.outputs.new("CF_SocketSegment", "Segment")

    def draw_buttons(self, context, layout):
        cond = self.inputs.get("Condition")
        if cond is not None and cond.is_linked:
            layout.label(text="Condition from socket", icon="LINKED")
            return
        layout.prop(self, "variable", text="")
        layout.prop(self, "compare", text="")
        if self.compare in {"EVEN", "ODD"}:
            return
        row = layout.row(align=True)
        row.prop(self, "value", text="N" if self.compare == "EVERY" else "Value")
        if self.compare in {"BETWEEN", "EVERY"}:
            row.prop(self, "value2", text="Start" if self.compare == "EVERY" else "Max")


class CF_NodeSelector(DynamicInputs, CFNode, Node):
    """Uses the input whose number is given by a variable or by the Index socket"""

    bl_idname = "CF_NodeSelector"
    bl_label = "Selector"
    bl_icon = "PRESET"
    cf_category = "OPERATOR"

    variable: EnumProperty(name="Index", items=VARIABLE_ITEMS, default="segment_id", update=_changed)
    mode: EnumProperty(name="Out of Range", items=[("WRAP", "Wrap", "Start again from the first input"),
                                                   ("CLAMP", "Clamp", "Use the first / last input"),
                                                   ("NONE", "Nothing", "No segment")],
                       default="WRAP", update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketInt", "Index")
        DynamicInputs.init(self, context)

    def input_name(self, i):
        return str(i)

    def draw_buttons(self, context, layout):
        idx = self.inputs.get("Index")
        if idx is None or not idx.is_linked:
            layout.prop(self, "variable", text="")
        layout.prop(self, "mode", text="")


class CF_NodeMirror(CFNode, Node):
    """Mirrors the segments"""

    bl_idname = "CF_NodeMirror"
    bl_label = "Mirror"
    bl_icon = "MOD_MIRROR"
    cf_category = "OPERATOR"

    x: BoolProperty(name="X", default=True, update=_changed)
    y: BoolProperty(name="Y", default=False, update=_changed)
    z: BoolProperty(name="Z", default=False, update=_changed)
    mode: EnumProperty(name="When", items=[("ALWAYS", "Always", ""),
                                           ("ALTERNATE", "Alternate", "Every other segment"),
                                           ("RANDOM", "Random", "With a probability")],
                       default="ALWAYS", update=_changed)
    probability: FloatProperty(name="Probability", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
                               update=_changed)
    seed: IntProperty(name="Seed", default=0, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketSegment", "Segment")
        self.outputs.new("CF_SocketSegment", "Segment")

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "x", toggle=True)
        row.prop(self, "y", toggle=True)
        row.prop(self, "z", toggle=True)
        layout.prop(self, "mode", text="")
        if self.mode == "RANDOM":
            row = layout.row(align=True)
            row.prop(self, "probability")
            row.prop(self, "seed")


class CF_NodeTransform(CFNode, Node):
    """Moves, rotates and scales the segments (with optional randomness)"""

    bl_idname = "CF_NodeTransform"
    bl_label = "Transform"
    bl_icon = "ORIENTATION_GLOBAL"
    bl_width_default = 200
    cf_category = "OPERATOR"

    rand_offset: FloatVectorProperty(name="Random Offset", size=3, min=0.0, subtype="TRANSLATION",
                                     unit="LENGTH", update=_changed)
    rand_rotation: FloatVectorProperty(name="Random Rotation", size=3, min=0.0, max=math.pi,
                                       subtype="EULER", unit="ROTATION", update=_changed)
    rotation_step: FloatProperty(name="Rotation Step", default=0.0, min=0.0, max=math.pi, subtype="ANGLE",
                                 update=_changed, description="Snap random rotations to multiples of this angle")
    uniform: BoolProperty(name="Uniform Scale", default=True, update=_changed)
    rand_scale: FloatVectorProperty(name="Random Scale", size=3, min=0.0, max=0.95, update=_changed)
    seed: IntProperty(name="Seed", default=0, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketSegment", "Segment")
        self.inputs.new("CF_SocketVector", "Offset")
        self.inputs.new("CF_SocketVector", "Rotation")
        self.inputs.new("CF_SocketVector", "Scale").default_value = (1.0, 1.0, 1.0)
        self.outputs.new("CF_SocketSegment", "Segment")

    def draw_buttons(self, context, layout):
        layout.prop(self, "seed")

    def draw_buttons_ext(self, context, layout):
        layout.label(text="Rotation socket values are in degrees")
        layout.prop(self, "rand_offset")
        layout.prop(self, "rand_rotation")
        layout.prop(self, "rotation_step")
        layout.prop(self, "uniform")
        if self.uniform:
            layout.prop(self, "rand_scale", index=0, text="Random Scale")
        else:
            layout.prop(self, "rand_scale")
        layout.prop(self, "seed")

    def draw_panel(self, context, layout):
        for name in ("Offset", "Rotation", "Scale"):
            sock = self.inputs.get(name)
            if sock is not None and not sock.is_linked:
                layout.prop(sock, "default_value", text=name)
        self.draw_buttons_ext(context, layout)


class CF_NodeMaterial(CFNode, Node):
    """Assigns materials to the segments (one, random, in sequence or by index)"""

    bl_idname = "CF_NodeMaterial"
    bl_label = "Material"
    bl_icon = "MATERIAL"
    bl_width_default = 200
    cf_category = "OPERATOR"

    target: EnumProperty(name="Replace", items=[("ALL", "All Faces", ""), ("SLOT", "One Slot", "")],
                         default="ALL", update=_changed)
    slot: IntProperty(name="Slot", default=0, min=0, update=_changed)
    pick: EnumProperty(name="Pick", items=[("FIRST", "First", "Always the first material"),
                                           ("RANDOM", "Random", ""), ("SEQUENCE", "Sequence", ""),
                                           ("INDEX", "By Index", "Use the Index socket")],
                       default="FIRST", update=_changed)
    count: IntProperty(name="Materials", default=1, min=1, max=8, update=_changed)
    seed: IntProperty(name="Seed", default=0, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketSegment", "Segment")
        self.inputs.new("CF_SocketInt", "Index")
        self.outputs.new("CF_SocketSegment", "Segment")

    def materials(self):
        return [getattr(self, "material_%d" % i) for i in range(self.count)]

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "target", text="")
        if self.target == "SLOT":
            row.prop(self, "slot", text="")
        row = layout.row(align=True)
        row.prop(self, "pick", text="")
        row.prop(self, "count", text="")
        for i in range(self.count):
            layout.prop(self, "material_%d" % i, text="")
        if self.pick == "RANDOM":
            layout.prop(self, "seed")


for _i in range(8):
    CF_NodeMaterial.__annotations__["material_%d" % _i] = PointerProperty(
        name="Material %d" % (_i + 1), type=bpy.types.Material, update=_changed)


class CF_NodeUVTransform(CFNode, Node):
    """Moves, scales and rotates the UVs of the segments"""

    bl_idname = "CF_NodeUVTransform"
    bl_label = "UV Transform"
    bl_icon = "UV"
    cf_category = "OPERATOR"

    offset: FloatVectorProperty(name="Offset", size=2, default=(0.0, 0.0), update=_changed)
    scale: FloatVectorProperty(name="Scale", size=2, default=(1.0, 1.0), update=_changed)
    rotation: FloatProperty(name="Rotation", default=0.0, subtype="ANGLE", update=_changed)
    rand_offset: FloatVectorProperty(name="Random Offset", size=2, min=0.0, default=(0.0, 0.0), update=_changed)
    seed: IntProperty(name="Seed", default=0, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketSegment", "Segment")
        self.outputs.new("CF_SocketSegment", "Segment")

    def draw_buttons(self, context, layout):
        layout.prop(self, "offset")
        layout.prop(self, "scale")
        layout.prop(self, "rotation")
        layout.prop(self, "rand_offset")
        layout.prop(self, "seed")


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

class CF_NodeValue(CFNode, Node):
    """A number"""

    bl_idname = "CF_NodeValue"
    bl_label = "Value"
    bl_icon = "DRIVER_TRANSFORM"
    bl_width_default = 140
    cf_category = "NUMBER"

    value: FloatProperty(name="Value", default=1.0, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.outputs.new("CF_SocketFloat", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value", text="")


class CF_NodeInteger(CFNode, Node):
    """A whole number"""

    bl_idname = "CF_NodeInteger"
    bl_label = "Integer"
    bl_icon = "DRIVER_TRANSFORM"
    bl_width_default = 140
    cf_category = "NUMBER"

    value: IntProperty(name="Integer", default=1, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.outputs.new("CF_SocketInt", "Integer")

    def draw_buttons(self, context, layout):
        layout.prop(self, "value", text="")


class CF_NodeInfo(CFNode, Node):
    """A variable of the segment being placed (index, distance, spline, slope...)"""

    bl_idname = "CF_NodeInfo"
    bl_label = "Segment Info"
    bl_icon = "INFO"
    bl_width_default = 160
    cf_category = "NUMBER"

    variable: EnumProperty(name="Variable", items=VARIABLE_ITEMS, default="index", update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.outputs.new("CF_SocketFloat", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "variable", text="")


class CF_NodeMath(CFNode, Node):
    """Math and comparisons (comparisons give 1 or 0)"""

    bl_idname = "CF_NodeMath"
    bl_label = "Math"
    bl_icon = "CON_TRANSFORM"
    bl_width_default = 150
    cf_category = "NUMBER"

    operation: EnumProperty(name="Operation", items=MATH_ITEMS, default="ADD", update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketFloat", "A")
        self.inputs.new("CF_SocketFloat", "B")
        self.inputs.new("CF_SocketFloat", "C")
        self.outputs.new("CF_SocketFloat", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "operation", text="")


class CF_NodeRandom(CFNode, Node):
    """A random number between Min and Max, different for every segment (or spline)"""

    bl_idname = "CF_NodeRandom"
    bl_label = "Random Value"
    bl_icon = "RNDCURVE"
    bl_width_default = 150
    cf_category = "NUMBER"

    seed: IntProperty(name="Seed", default=0, update=_changed)
    per: EnumProperty(name="Per", items=[("SEGMENT", "Segment", ""), ("SPLINE", "Spline", ""),
                                         ("INDEX", "Index", "Same value for the same index")],
                      default="SEGMENT", update=_changed)
    integer: BoolProperty(name="Integer", default=False, update=_changed)

    def init(self, context):
        self.cf_colorize()
        self.inputs.new("CF_SocketFloat", "Min")
        self.inputs.new("CF_SocketFloat", "Max").default_value = 1.0
        self.outputs.new("CF_SocketFloat", "Value")

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, "per", text="")
        row.prop(self, "integer", toggle=True, text="Int")
        layout.prop(self, "seed")


class CF_NodeExpression(CFNode, Node):
    """A formula using the segment variables and the inputs a, b, c, d"""

    bl_idname = "CF_NodeExpression"
    bl_label = "Expression"
    bl_icon = "SCRIPT"
    bl_width_default = 220
    cf_category = "NUMBER"

    expression: StringProperty(name="Expression", default="index % 2", update=_changed,
                               description="Python-like formula, e.g. 'index % 3 == 0 and slope < 10'")

    def init(self, context):
        self.cf_colorize()
        for name in ("a", "b", "c", "d"):
            self.inputs.new("CF_SocketFloat", name)
        self.outputs.new("CF_SocketFloat", "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "expression", text="")


class CF_NodeCombine(CFNode, Node):
    """Builds a vector from three numbers"""

    bl_idname = "CF_NodeCombine"
    bl_label = "Combine XYZ"
    bl_icon = "EMPTY_ARROWS"
    bl_width_default = 140
    cf_category = "NUMBER"

    def init(self, context):
        self.cf_colorize()
        for name in ("X", "Y", "Z"):
            self.inputs.new("CF_SocketFloat", name)
        self.outputs.new("CF_SocketVector", "Vector")

    def draw_buttons(self, context, layout):
        pass


node_classes = (CF_NodeSpline, CF_NodeSegment, CF_NodeGap, CF_NodeLinear, CF_NodeCompose, CF_NodeSequence,
                CF_NodeRandomize, CF_NodeConditional, CF_NodeSelector, CF_NodeMirror, CF_NodeTransform,
                CF_NodeMaterial, CF_NodeUVTransform, CF_NodeValue, CF_NodeInteger, CF_NodeInfo, CF_NodeMath,
                CF_NodeRandom, CF_NodeExpression, CF_NodeCombine)
classes = (CF_StyleTree,) + node_classes


def register():
    _registering[0] = True
    try:
        for cls in classes:
            bpy.utils.register_class(cls)
    finally:
        _registering[0] = False


def unregister():
    _registering[0] = True
    try:
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
    finally:
        _registering[0] = False
