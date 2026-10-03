# SPDX-License-Identifier: GPL-3.0-or-later
"""Spline description and dense sampling (pure numpy).

A :class:`SplineSpec` holds one spline in world space. :func:`sample` turns it
into a :class:`Sampled` polyline whose ``param`` array is the *control
parameter*: control point ``k`` is at ``param == k``, and the value grows
monotonically along the polyline. Closed splines repeat their first point at
the end of the polyline.
"""

import math

import numpy as np

CURVE_STEP = math.radians(5.0)   # max turning per sampled step on curved Bezier segments


def _length(v):
    return float(np.sqrt(np.dot(v, v)))


def turn_angle(a, b):
    """Angle between two direction vectors (0 if one is degenerate)."""
    la, lb = _length(a), _length(b)
    if la < 1e-12 or lb < 1e-12:
        return 0.0
    c = float(np.dot(a, b)) / (la * lb)
    return math.acos(min(1.0, max(-1.0, c)))


class SplineSpec:
    """One spline in world space plus the per-spline data the generator needs."""

    def __init__(self, kind, points, cyclic=False, left=None, right=None, tilt=None,
                 order=4, endpoint=False, bezier=False, resolution=12, index=0,
                 material=0, seg_ids=None, name="", weights=None):
        self.kind = kind
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n = len(self.points)
        self.cyclic = bool(cyclic)
        self.left = None if left is None else np.asarray(left, dtype=np.float64).reshape(-1, 3)
        self.right = None if right is None else np.asarray(right, dtype=np.float64).reshape(-1, 3)
        self.tilt = np.zeros(n) if tilt is None else np.asarray(tilt, dtype=np.float64).reshape(-1)
        self.order = int(order)
        self.endpoint = bool(endpoint)
        self.bezier = bool(bezier)
        self.resolution = int(resolution)
        self.index = int(index)
        self.material = int(material)
        n_seg = n if self.cyclic else max(n - 1, 0)
        self.seg_ids = (np.zeros(n_seg, dtype=np.int64) if seg_ids is None
                        else np.asarray(seg_ids, dtype=np.int64).reshape(-1))
        self.name = name
        self.weights = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)

    @property
    def n_points(self):
        return len(self.points)

    @property
    def n_segments(self):
        return self.n_points if self.cyclic else max(self.n_points - 1, 0)


class Sampled:
    __slots__ = ("points", "tilt", "param", "cyclic", "n_ctrl", "vertex_angle")

    def __init__(self, points, tilt, param, cyclic, n_ctrl, vertex_angle):
        self.points = points
        self.tilt = tilt
        self.param = param
        self.cyclic = cyclic
        self.n_ctrl = n_ctrl
        self.vertex_angle = vertex_angle


# ---------------------------------------------------------------------------
# Bezier
# ---------------------------------------------------------------------------

def cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (np.outer(u * u * u, p0) + np.outer(3.0 * u * u * t, p1)
            + np.outer(3.0 * u * t * t, p2) + np.outer(t * t * t, p3))


def bezier_steps(p0, p1, p2, p3, resolution=0):
    """Pieces used for one Bezier segment: 1 if it is straight, else adaptive or fixed."""
    chord = p3 - p0
    clen = _length(chord)
    scale = max(clen, _length(p1 - p0), _length(p2 - p3), 1e-9)
    tol = 1e-6 * scale
    if clen > tol:
        d = chord / clen
        straight = True
        for q in (p1, p2):
            w = q - p0
            along = float(np.dot(w, d))
            if _length(w - along * d) > tol or not -tol <= along <= clen + tol:
                straight = False
                break
        if straight:
            return 1
    elif _length(p1 - p0) <= tol and _length(p2 - p3) <= tol:
        return 1
    if resolution > 0:
        return int(resolution)
    legs = [leg for leg in (p1 - p0, p2 - p1, p3 - p2) if _length(leg) > tol]
    turning = sum(turn_angle(a, b) for a, b in zip(legs, legs[1:]))
    return int(min(64, max(3, math.ceil(turning / CURVE_STEP))))


def _bezier(spec, resolution):
    co, left, right, tilt = spec.points, spec.left, spec.right, spec.tilt
    n = len(co)
    pts, tilts, params, first_dense = [], [], [], []
    count = 0
    for i in range(spec.n_segments):
        j = (i + 1) % n
        steps = bezier_steps(co[i], right[i], left[j], co[j], resolution)
        t = np.arange(steps, dtype=np.float64) / steps
        first_dense.append(count)
        pts.append(cubic(co[i], right[i], left[j], co[j], t))
        tilts.append(tilt[i] + (tilt[j] - tilt[i]) * t)
        params.append(i + t)
        count += steps
    last = 0 if spec.cyclic else n - 1
    pts.append(co[last][None])
    tilts.append(tilt[last:last + 1])
    params.append(np.array([float(spec.n_segments)]))
    pts = np.concatenate(pts)
    tilts = np.concatenate(tilts)
    params = np.concatenate(params)

    angles = np.zeros(n)
    for k in range(n):
        if not spec.cyclic and k in (0, n - 1):
            continue
        d = first_dense[k] if k < len(first_dense) else len(pts) - 1
        into = co[k] - left[k]
        if _length(into) < 1e-9:
            into = co[k] - pts[d - 1 if d > 0 else len(pts) - 2]
        out = right[k] - co[k]
        if _length(out) < 1e-9:
            out = pts[d + 1] - co[k]
        angles[k] = turn_angle(into, out)
    return pts, tilts, params, angles


# ---------------------------------------------------------------------------
# Poly
# ---------------------------------------------------------------------------

def _poly(spec):
    co, tilt = spec.points, spec.tilt
    n = len(co)
    params = np.arange(n, dtype=np.float64)
    pts, tilts = co.copy(), tilt.copy()
    if spec.cyclic:
        pts = np.vstack([pts, co[:1]])
        tilts = np.append(tilts, tilt[0])
        params = np.append(params, float(n))
    angles = np.zeros(n)
    for k in range(n):
        if not spec.cyclic and k in (0, n - 1):
            continue
        into = out = None
        for step in range(1, n):
            if not spec.cyclic and k - step < 0:
                break
            v = co[k] - co[(k - step) % n]
            if _length(v) > 1e-9:
                into = v
                break
        for step in range(1, n):
            if not spec.cyclic and k + step >= n:
                break
            v = co[(k + step) % n] - co[k]
            if _length(v) > 1e-9:
                out = v
                break
        if into is not None and out is not None:
            angles[k] = turn_angle(into, out)
    return pts, tilts, params, angles


# ---------------------------------------------------------------------------
# NURBS
# ---------------------------------------------------------------------------

def knot_vector(pnts, order, cyclic, endpoint, bezier):
    """Same knot layout as Blender's NURBS evaluation."""
    inner = order - 1 if bezier else 1
    if endpoint:
        head = order - (1 if cyclic else 0)
    else:
        head = min(2, inner) if bezier else 1
    tail = 2 * order - 1 if cyclic else (order if endpoint else 0)
    size = pnts + order + (order - 1 if cyclic else 0)
    knots = np.zeros(size)
    left = head
    value = 0.0
    start = 1 if endpoint and cyclic else 0
    if start:
        value = 1.0
    for i in range(start, size - tail):
        knots[i] = value
        left -= 1
        if left == 0:
            value += 1.0
            left = inner
    for i in range(tail):
        knots[size - tail + i] = value + (knots[i] - knots[0])
    return knots


def basis_functions(knots, order, count, t):
    """B-spline basis values, shape (len(t), count)."""
    t = np.asarray(t, dtype=np.float64)
    spans = len(knots) - 1
    b = np.zeros((len(t), spans))
    for i in range(spans):
        if knots[i + 1] > knots[i]:
            b[:, i] = (t >= knots[i]) & (t < knots[i + 1])
    t_end = knots[count] if count < len(knots) else knots[-1]
    filled = [i for i in range(min(count, spans)) if knots[i + 1] > knots[i]]
    at_end = t >= t_end - 1e-12
    if at_end.any() and filled:
        b[at_end] = 0.0
        b[at_end, filled[-1]] = 1.0
    for degree in range(1, order):
        nb = np.zeros((len(t), spans - degree))
        for i in range(spans - degree):
            d0 = knots[i + degree] - knots[i]
            d1 = knots[i + degree + 1] - knots[i + 1]
            acc = np.zeros(len(t))
            if d0 > 0:
                acc += (t - knots[i]) / d0 * b[:, i]
            if d1 > 0:
                acc += (knots[i + degree + 1] - t) / d1 * b[:, i + 1]
            nb[:, i] = acc
        b = nb
    return b[:, :count]


def _nurbs(spec, resolution):
    co, tilt = spec.points, spec.tilt
    n = len(co)
    order = max(2, min(spec.order, n + (1 if spec.cyclic else 0)))
    res = resolution if resolution > 0 else max(spec.resolution, 12)
    total = max(2, res * spec.n_segments)
    wrap = order - 1 if spec.cyclic else 0
    count = n + wrap
    knots = knot_vector(n, order, spec.cyclic, spec.endpoint, spec.bezier)
    lo = knots[order - 1]
    hi = knots[n + order - 1] if spec.cyclic else knots[n]
    if spec.cyclic:
        t = lo + (hi - lo) * np.arange(total) / total
    else:
        t = lo + (hi - lo) * np.arange(total) / (total - 1)
    idx = np.arange(count) % n
    w = spec.weights
    bw = basis_functions(knots, order, count, t) * w[idx][None, :]
    norm = bw.sum(axis=1)
    norm[norm == 0.0] = 1.0
    pts = (bw @ co[idx]) / norm[:, None]
    tilts = (bw @ tilt[idx]) / norm
    greville = np.array([knots[i + 1:i + order].mean() for i in range(count)])
    params = np.interp(t, greville, np.arange(count, dtype=np.float64))
    if spec.cyclic:
        pts = np.vstack([pts, pts[:1]])
        tilts = np.append(tilts, tilts[0])
        params = np.append(params, params[0] + n)
    else:
        params[0], params[-1] = 0.0, float(n - 1)
    params = np.maximum.accumulate(params)
    return pts, tilts, params, np.zeros(n)


# ---------------------------------------------------------------------------

def _clean(pts, tilts, params, cyclic, tol=1e-7):
    """Drop consecutive duplicates and force a strictly increasing parameter."""
    keep = np.ones(len(pts), dtype=bool)
    ref = pts[0]
    for i in range(1, len(pts)):
        if _length(pts[i] - ref) <= tol:
            keep[i] = False
        else:
            ref = pts[i]
    if cyclic and not keep[-1]:
        keep[-1] = True
        j = len(pts) - 2
        while j > 0 and not keep[j]:
            j -= 1
        if j > 0:
            keep[j] = False
    pts, tilts, params = pts[keep], tilts[keep], params[keep].copy()
    for i in range(1, len(params)):
        if params[i] <= params[i - 1]:
            params[i] = params[i - 1] + 1e-9
    return pts, tilts, params


def sample(spec, resolution=0):
    """Dense polyline for a :class:`SplineSpec` (``None`` when degenerate)."""
    if spec.n_points < 2:
        return None
    if spec.kind == "BEZIER" and spec.left is not None and spec.right is not None:
        pts, tilts, params, angles = _bezier(spec, resolution)
    elif spec.kind == "NURBS" and spec.n_points > 2:
        pts, tilts, params, angles = _nurbs(spec, resolution)
    else:
        pts, tilts, params, angles = _poly(spec)
    pts, tilts, params = _clean(pts, tilts, params, spec.cyclic)
    if len(pts) < (3 if spec.cyclic else 2):
        return None
    return Sampled(pts, tilts, params, spec.cyclic, spec.n_points, angles)


def reverse(sampled):
    """Same polyline travelled backwards (``param`` keeps naming the same points)."""
    return Sampled(sampled.points[::-1].copy(), sampled.tilt[::-1].copy(), sampled.param[::-1].copy(),
                   sampled.cyclic, sampled.n_ctrl, sampled.vertex_angle.copy())
