# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading Blender curve objects (works in object and edit mode)."""

import numpy as np

from .core.pathmath import SplineInput


def _points(sp):
    return sp.bezier_points if sp.type == "BEZIER" else sp.points


def spline_local_points(sp):
    pts = _points(sp)
    n = len(pts)
    if sp.type == "BEZIER":
        co = np.empty(n * 3, np.float32)
        pts.foreach_get("co", co)
        return co.reshape(-1, 3).astype(np.float64)
    co = np.empty(n * 4, np.float32)
    pts.foreach_get("co", co)
    return co.reshape(-1, 4)[:, :3].astype(np.float64)


def local_points(curve_obj):
    """Control point positions per spline, in the curve's local space."""
    return [spline_local_points(sp) for sp in curve_obj.data.splines]


def topology_signature(curve_obj):
    return ";".join("%d%s" % (len(_points(sp)), "c" if sp.use_cyclic_u else "o")
                    for sp in curve_obj.data.splines)


def read_splines(curve_obj):
    """World-space :class:`SplineInput` list for every spline of a curve object."""
    mw = np.array(curve_obj.matrix_world, dtype=np.float64)
    m3, t = mw[:3, :3], mw[:3, 3]

    def xf(a):
        return a @ m3.T + t

    out = []
    for sp in curve_obj.data.splines:
        pts = _points(sp)
        n = len(pts)
        if n == 0:
            out.append(None)
            continue
        tilt = np.empty(n, np.float32)
        pts.foreach_get("tilt", tilt)
        if sp.type == "BEZIER":
            co = np.empty(n * 3, np.float32)
            hl = np.empty(n * 3, np.float32)
            hr = np.empty(n * 3, np.float32)
            pts.foreach_get("co", co)
            pts.foreach_get("handle_left", hl)
            pts.foreach_get("handle_right", hr)
            out.append(SplineInput("BEZIER", xf(co.reshape(-1, 3).astype(np.float64)),
                                   cyclic=sp.use_cyclic_u,
                                   handle_left=xf(hl.reshape(-1, 3).astype(np.float64)),
                                   handle_right=xf(hr.reshape(-1, 3).astype(np.float64)),
                                   tilt=tilt, resolution=sp.resolution_u))
        else:
            co4 = np.empty(n * 4, np.float32)
            pts.foreach_get("co", co4)
            co4 = co4.reshape(-1, 4).astype(np.float64)
            out.append(SplineInput("NURBS" if sp.type == "NURBS" else "POLY", xf(co4[:, :3]),
                                   cyclic=sp.use_cyclic_u, tilt=tilt, weights=co4[:, 3],
                                   order=sp.order_u, use_endpoint=sp.use_endpoint_u,
                                   use_bezier=sp.use_bezier_u, resolution=sp.resolution_u))
    return out


def selected_points(curve_obj):
    """List of (spline_index, point_index) for selected control points."""
    out = []
    for si, sp in enumerate(curve_obj.data.splines):
        if sp.type == "BEZIER":
            for pi, p in enumerate(sp.bezier_points):
                if p.select_control_point:
                    out.append((si, pi))
        else:
            for pi, p in enumerate(sp.points):
                if p.select:
                    out.append((si, pi))
    return out


def selected_segments(curve_obj):
    """List of (spline_index, segment_index): segments whose two ends are selected."""
    sel = set(selected_points(curve_obj))
    out = []
    for si, sp in enumerate(curve_obj.data.splines):
        n = len(_points(sp))
        segs = n if sp.use_cyclic_u else n - 1
        for i in range(max(segs, 0)):
            if (si, i) in sel and (si, (i + 1) % n) in sel:
                out.append((si, i))
    return out


def set_point_selection(curve_obj, refs, state):
    """Select / deselect control points given as (spline, index) pairs."""
    splines = curve_obj.data.splines
    for si, pi in refs:
        if si < 0 or si >= len(splines):
            continue
        sp = splines[si]
        pts = _points(sp)
        if pi < 0 or pi >= len(pts):
            continue
        p = pts[pi]
        if sp.type == "BEZIER":
            p.select_control_point = state
            p.select_left_handle = state
            p.select_right_handle = state
        else:
            p.select = state
