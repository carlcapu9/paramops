# SPDX-License-Identifier: GPL-3.0-or-later
"""Socket types of the CurveForge node editor."""

import bpy
from bpy.props import BoolProperty, FloatProperty, FloatVectorProperty, IntProperty
from bpy.types import NodeSocket


def _changed(self, context):
    from . import live
    live.tree_changed(self.id_data)


class _Socket:
    color = (0.6, 0.6, 0.6, 1.0)

    def draw_color(self, context, node):
        return self.color

    @classmethod
    def draw_color_simple(cls):
        return cls.color


class _ValueSocket(_Socket):
    """Number sockets show their value when nothing is connected."""

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "default_value", text=text)


class CF_SocketSpline(_Socket, NodeSocket):
    bl_idname = "CF_SocketSpline"
    bl_label = "Spline"
    color = (0.15, 0.78, 0.74, 1.0)

    def draw(self, context, layout, node, text):
        layout.label(text=text)


class CF_SocketSegment(_Socket, NodeSocket):
    bl_idname = "CF_SocketSegment"
    bl_label = "Segment"
    color = (1.0, 0.56, 0.16, 1.0)

    weight: FloatProperty(name="Weight", default=1.0, min=0.0, soft_max=10.0, update=_changed,
                          description="Probability of this input")
    count: IntProperty(name="Count", default=1, min=0, soft_max=20, update=_changed,
                       description="How many times this input is used before the next one")

    def draw(self, context, layout, node, text):
        if self.is_output:
            layout.label(text=text)
            return
        kind = getattr(node, "cf_input_extra", "")
        if kind and self.is_linked:
            row = layout.row(align=True)
            row.label(text=text)
            row.prop(self, kind, text="")
        else:
            layout.label(text=text)


class CF_SocketFloat(_ValueSocket, NodeSocket):
    bl_idname = "CF_SocketFloat"
    bl_label = "Value"
    color = (0.63, 0.63, 0.63, 1.0)
    default_value: FloatProperty(default=0.0, update=_changed)


class CF_SocketDistance(_ValueSocket, NodeSocket):
    bl_idname = "CF_SocketDistance"
    bl_label = "Distance"
    color = (0.63, 0.63, 0.63, 1.0)
    default_value: FloatProperty(default=0.0, subtype="DISTANCE", unit="LENGTH", update=_changed)


class CF_SocketAngle(_ValueSocket, NodeSocket):
    bl_idname = "CF_SocketAngle"
    bl_label = "Angle"
    color = (0.63, 0.63, 0.63, 1.0)
    default_value: FloatProperty(default=0.0, subtype="ANGLE", unit="ROTATION", update=_changed)


class CF_SocketInt(_ValueSocket, NodeSocket):
    bl_idname = "CF_SocketInt"
    bl_label = "Integer"
    color = (0.35, 0.55, 0.36, 1.0)
    default_value: IntProperty(default=0, update=_changed)


class CF_SocketBool(_ValueSocket, NodeSocket):
    bl_idname = "CF_SocketBool"
    bl_label = "Boolean"
    color = (0.8, 0.65, 0.84, 1.0)
    default_value: BoolProperty(default=False, update=_changed)


class CF_SocketVector(_ValueSocket, NodeSocket):
    bl_idname = "CF_SocketVector"
    bl_label = "Vector"
    color = (0.39, 0.39, 0.78, 1.0)
    default_value: FloatVectorProperty(size=3, default=(0.0, 0.0, 0.0), subtype="XYZ", update=_changed)

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            col = layout.column(align=True)
            col.label(text=text)
            col.prop(self, "default_value", text="")


NUMBER_SOCKETS = {"CF_SocketFloat", "CF_SocketDistance", "CF_SocketAngle", "CF_SocketInt", "CF_SocketBool"}

classes = (CF_SocketSpline, CF_SocketSegment, CF_SocketFloat, CF_SocketDistance, CF_SocketAngle,
           CF_SocketInt, CF_SocketBool, CF_SocketVector)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
