# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading Blender curves, plus per-segment IDs stored on curve objects.

Segment IDs are the equivalent of spline material IDs: an integer per curve
segment (between two control points) that segments can be chosen by (Selector,
Conditional, Segment Info). They are assigned in Edit Mode and follow their
points when the curve is edited (points moved, added, removed or reversed).
"""

import bpy
import numpy as np
from bpy.props import CollectionProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup

from .engine.spline import SplineSpec


class CF_SegmentIDRef(PropertyGroup):
    spline: IntProperty()
    index: IntProperty()
    value: IntProperty()
    co_a: FloatVectorProperty(size=3)
    co_b: FloatVectorProperty(size=3)


class CF_CurveData(PropertyGroup):
    ids: CollectionProperty(type=CF_SegmentIDRef)
    signature: StringProperty()
    edit_value: IntProperty(name="ID", default=1, min=0, description="Segment ID to assign")


def _points(sp):
    return sp.bezier_points if sp.type == "BEZIER" else sp.points


def local_points(curve_obj):
    out = []
    for sp in curve_obj.data.splines:
        pts = _points(sp)
        n = len(pts)
        if sp.type == "BEZIER":
            co = np.zeros(n * 3, np.float32)
            pts.foreach_get("co", co)
            out.append(co.reshape(-1, 3).astype(np.float64))
        else:
            co = np.zeros(n * 4, np.float32)
            pts.foreach_get("co", co)
            out.append(co.reshape(-1, 4)[:, :3].astype(np.float64))
    return out


def signature(curve_obj):
    return ";".join("%d%s" % (len(_points(sp)), "c" if sp.use_cyclic_u else "o") for sp in curve_obj.data.splines)


def sync_ids(curve_obj):
    """Re-attach stored IDs to their points after edits. Returns {spline: {segment: id}}."""
    data = curve_obj.cf_curve
    if not len(data.ids):
        return {}
    pts = local_points(curve_obj)
    cyclic = [sp.use_cyclic_u for sp in curve_obj.data.splines]
    sig = signature(curve_obj)
    changed = sig != data.signature
    tol = 1e-5

    def find(co, prefer):
        co = np.asarray(co)
        order = [prefer] + [i for i in range(len(pts)) if i != prefer] if 0 <= prefer < len(pts) else range(len(pts))
        for si in order:
            if len(pts[si]):
                d = np.linalg.norm(pts[si] - co, axis=1)
                j = int(np.argmin(d))
                if d[j] <= tol:
                    return si, j
        return None

    drop, add = [], []
    for r_i, ref in enumerate(data.ids):
        si, i = ref.spline, ref.index
        ok_index = 0 <= si < len(pts) and 0 <= i < len(pts[si])
        if ok_index:
            n = len(pts[si])
            j = (i + 1) % n
            if (np.linalg.norm(pts[si][i] - np.array(ref.co_a)) <= tol
                    and np.linalg.norm(pts[si][j] - np.array(ref.co_b)) <= tol):
                continue
        ha, hb = find(ref.co_a, si), find(ref.co_b, si)
        if ha is not None and hb is not None and ha[0] == hb[0]:
            s2, a = ha
            b = hb[1]
            n = len(pts[s2])
            if b == (a + 1) % n:
                ref.spline, ref.index = s2, a
                continue
            lo, hi = (a, b) if a < b else (b, a)
            segs = list(range(lo, hi))
            if segs and len(segs) < n:
                ref.spline, ref.index = s2, segs[0]
                ref.co_a, ref.co_b = pts[s2][segs[0]], pts[s2][(segs[0] + 1) % n]
                add.extend((s2, k, ref.value) for k in segs[1:])
                continue
        if not changed and ok_index:
            n = len(pts[si])
            if i < n - 1 or cyclic[si]:
                ref.co_a, ref.co_b = pts[si][i], pts[si][(i + 1) % n]
                continue
        drop.append(r_i)
    for r_i in reversed(drop):
        data.ids.remove(r_i)
    for s2, k, value in add:
        n = len(pts[s2])
        ref = data.ids.add()
        ref.spline, ref.index, ref.value = s2, k, value
        ref.co_a, ref.co_b = pts[s2][k], pts[s2][(k + 1) % n]
    if data.signature != sig:
        data.signature = sig
    out = {}
    for ref in data.ids:
        out.setdefault(ref.spline, {})[ref.index] = ref.value
    return out


def set_ids(curve_obj, segments, value):
    """Assign ``value`` to (spline, segment) pairs (0 removes the assignment)."""
    data = curve_obj.cf_curve
    pts = local_points(curve_obj)
    chosen = set(segments)
    for i in reversed(range(len(data.ids))):
        r = data.ids[i]
        if (r.spline, r.index) in chosen:
            data.ids.remove(i)
    if value:
        for si, k in segments:
            n = len(pts[si])
            ref = data.ids.add()
            ref.spline, ref.index, ref.value = si, k, value
            ref.co_a, ref.co_b = pts[si][k], pts[si][(k + 1) % n]
    data.signature = signature(curve_obj)


def selected_points(curve_obj):
    out = []
    for si, sp in enumerate(curve_obj.data.splines):
        if sp.type == "BEZIER":
            out += [(si, i) for i, p in enumerate(sp.bezier_points) if p.select_control_point]
        else:
            out += [(si, i) for i, p in enumerate(sp.points) if p.select]
    return out


def selected_segments(curve_obj):
    sel = set(selected_points(curve_obj))
    out = []
    for si, sp in enumerate(curve_obj.data.splines):
        n = len(_points(sp))
        for i in range(n if sp.use_cyclic_u else n - 1):
            if (si, i) in sel and (si, (i + 1) % n) in sel:
                out.append((si, i))
    return out


def select_points(curve_obj, refs, state=True):
    splines = curve_obj.data.splines
    for si, i in refs:
        if 0 <= si < len(splines):
            sp = splines[si]
            pts = _points(sp)
            if 0 <= i < len(pts):
                if sp.type == "BEZIER":
                    p = pts[i]
                    p.select_control_point = p.select_left_handle = p.select_right_handle = state
                else:
                    pts[i].select = state


def read_splines(curve_obj, only=None):
    """World-space :class:`SplineSpec` list (``only``: set of spline numbers or None)."""
    m = np.array(curve_obj.matrix_world, dtype=np.float64)
    m3, t = m[:3, :3], m[:3, 3]
    ids = sync_ids(curve_obj)
    specs = []
    for si, sp in enumerate(curve_obj.data.splines):
        if only is not None and si not in only:
            continue
        pts = _points(sp)
        n = len(pts)
        if n < 2:
            continue
        tilt = np.zeros(n, np.float32)
        pts.foreach_get("tilt", tilt)
        n_seg = n if sp.use_cyclic_u else n - 1
        seg_ids = np.zeros(n_seg, np.int64)
        for k, v in ids.get(si, {}).items():
            if 0 <= k < n_seg:
                seg_ids[k] = v
        common = dict(cyclic=sp.use_cyclic_u, tilt=tilt, index=si, material=sp.material_index,
                      seg_ids=seg_ids, name=curve_obj.name, resolution=sp.resolution_u)
        if sp.type == "BEZIER":
            co = np.zeros(n * 3, np.float32)
            hl = np.zeros(n * 3, np.float32)
            hr = np.zeros(n * 3, np.float32)
            pts.foreach_get("co", co)
            pts.foreach_get("handle_left", hl)
            pts.foreach_get("handle_right", hr)

            def xf(a):
                return a.reshape(-1, 3).astype(np.float64) @ m3.T + t
            specs.append(SplineSpec("BEZIER", xf(co), left=xf(hl), right=xf(hr), **common))
        else:
            co = np.zeros(n * 4, np.float32)
            pts.foreach_get("co", co)
            co = co.reshape(-1, 4).astype(np.float64)
            specs.append(SplineSpec("NURBS" if sp.type == "NURBS" else "POLY", co[:, :3] @ m3.T + t,
                                    order=sp.order_u, endpoint=sp.use_endpoint_u, bezier=sp.use_bezier_u,
                                    weights=co[:, 3], **common))
    return specs


def polygons(curve_obj):
    """Closed splines of a curve as world-space 2D polygons (for clipping areas)."""
    from .engine.spline import sample
    out = []
    for spec in read_splines(curve_obj):
        s = sample(spec)
        if s is not None and spec.cyclic:
            out.append(s.points[:-1, :2])
    return out


classes = (CF_SegmentIDRef, CF_CurveData)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.cf_curve = PointerProperty(type=CF_CurveData)


def unregister():
    del bpy.types.Object.cf_curve
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
