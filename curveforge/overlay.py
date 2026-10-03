# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport overlay: the Segment ID of every segment of the curve being edited."""

import blf
import bpy
import gpu
from bpy.props import BoolProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader

COLORS = ((0.62, 0.62, 0.62, 0.9), (1.0, 0.56, 0.16, 1.0), (0.32, 0.74, 1.0, 1.0), (0.52, 0.94, 0.4, 1.0),
          (1.0, 0.42, 0.62, 1.0), (0.9, 0.8, 0.3, 1.0), (0.72, 0.52, 1.0, 1.0), (0.36, 0.94, 0.86, 1.0))

_handle = [None]


def midpoints(curve_obj):
    """(spline, segment, world position) of the middle of every segment."""
    mw = curve_obj.matrix_world
    for si, sp in enumerate(curve_obj.data.splines):
        bezier = sp.type == "BEZIER"
        pts = sp.bezier_points if bezier else sp.points
        n = len(pts)
        for k in range(n if sp.use_cyclic_u else n - 1):
            a, b = pts[k], pts[(k + 1) % n]
            if bezier:
                mid = (a.co + 3.0 * a.handle_right + 3.0 * b.handle_left + b.co) / 8.0
            else:
                mid = (a.co.xyz + b.co.xyz) / 2.0
            yield si, k, mw @ mid


def draw():
    context = bpy.context
    ob = context.active_object
    if ob is None or ob.type != "CURVE" or context.mode != "EDIT_CURVE":
        return
    if not context.window_manager.cf_show_ids:
        return
    region, rv3d = context.region, context.region_data
    if region is None or rv3d is None:
        return
    ids = {(r.spline, r.index): r.value for r in ob.cf_curve.ids}
    scale = context.preferences.system.ui_scale
    font = 0
    blf.size(font, round(13 * scale))
    _, h = blf.dimensions(font, "ID 0123456789")
    pad = 4.0 * scale
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    for si, k, co in midpoints(ob):
        p = location_3d_to_region_2d(region, rv3d, co)
        if p is None:
            continue
        value = ids.get((si, k), 0)
        color = COLORS[value % len(COLORS)]
        text = "ID %d" % value
        w = blf.dimensions(font, text)[0]
        x, y = p.x - w / 2.0, p.y + 10.0 * scale
        _rect(shader, x - pad, y - pad, x + w + pad, y + h + pad, (0.04, 0.04, 0.04, 0.78))
        _rect(shader, x - pad, y - pad, x + w + pad, y - pad + 2.0 * scale, color)
        blf.color(font, *color)
        blf.position(font, x, y, 0.0)
        blf.draw(font, text)
    gpu.state.blend_set("NONE")


def _rect(shader, x0, y0, x1, y1, color):
    batch = batch_for_shader(shader, "TRIS", {"pos": ((x0, y0), (x1, y0), (x1, y1), (x0, y1))},
                             indices=((0, 1, 2), (2, 3, 0)))
    shader.uniform_float("color", color)
    batch.draw(shader)


def register():
    bpy.types.WindowManager.cf_show_ids = BoolProperty(
        name="Show in Viewport", default=True,
        description="Show the Segment ID of every segment of the curve in Edit Mode")
    _handle[0] = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_PIXEL")


def unregister():
    if _handle[0] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle[0], "WINDOW")
        _handle[0] = None
    del bpy.types.WindowManager.cf_show_ids
