# SPDX-License-Identifier: GPL-3.0-or-later
"""Curve sampling and path evaluation (pure numpy, no ``bpy``).

Conventions
-----------
* A :class:`Polyline` is a dense sampling of one spline. Closed (cyclic)
  polylines repeat their first point at the end, so segment ``i`` always
  joins points ``i`` and ``i + 1``.
* ``u`` is the *control parameter*: control point ``k`` sits at ``u == k``.
  It is monotonic along the polyline and lets the layout map control points,
  control segments, corners and markers to arc-length positions even after
  the polyline has been rounded, projected or reversed.
* Sample-local axes: +X runs along the path, +Y is to the left, +Z is up.
"""

import math

import numpy as np

UP = np.array((0.0, 0.0, 1.0))
MAX_MITER = 8.0
DEDUPE_TOL = 1e-7
ADAPTIVE_STEP = math.radians(4.0)
ARC_STEP = math.radians(4.0)


# ---------------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------------

def _norm(v):
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def _normalize(v):
    """Normalize rows, returning ``(unit_rows, valid_mask)``."""
    n = _norm(v)
    ok = n > 1e-12
    out = np.zeros_like(v)
    out[ok] = v[ok] / n[ok, None]
    return out, ok


def _fill_invalid(vecs, ok):
    """Replace invalid rows by the nearest preceding valid row (or the first valid one)."""
    if ok.all() or not ok.any():
        return vecs
    idx = np.where(ok, np.arange(len(ok)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = int(np.argmax(ok))
    return vecs[idx]


def _angle(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = float(np.dot(a, b) / (na * nb))
    return math.acos(max(-1.0, min(1.0, c)))


def _rotate(v, axis, angle):
    """Rotate vectors ``v`` (n, 3) around unit ``axis`` rows (n, 3) by ``angle`` (n,)."""
    c = np.cos(angle)[:, None]
    s = np.sin(angle)[:, None]
    d = np.einsum("ij,ij->i", axis, v)[:, None]
    return v * c + np.cross(axis, v) * s + axis * d * (1.0 - c)


# ---------------------------------------------------------------------------
# Input / output containers
# ---------------------------------------------------------------------------

class SplineInput:
    """Plain description of one spline, already in world space."""

    __slots__ = ("kind", "co", "cyclic", "handle_left", "handle_right", "tilt",
                 "weights", "order", "use_endpoint", "use_bezier", "resolution")

    def __init__(self, kind, co, cyclic=False, handle_left=None, handle_right=None,
                 tilt=None, weights=None, order=4, use_endpoint=False,
                 use_bezier=False, resolution=12):
        self.kind = kind
        self.co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
        n = len(self.co)
        self.cyclic = bool(cyclic)
        self.handle_left = None if handle_left is None else np.asarray(handle_left, np.float64).reshape(-1, 3)
        self.handle_right = None if handle_right is None else np.asarray(handle_right, np.float64).reshape(-1, 3)
        self.tilt = np.zeros(n) if tilt is None else np.asarray(tilt, np.float64).reshape(-1)
        self.weights = np.ones(n) if weights is None else np.asarray(weights, np.float64).reshape(-1)
        self.order = int(order)
        self.use_endpoint = bool(use_endpoint)
        self.use_bezier = bool(use_bezier)
        self.resolution = int(resolution)


class Polyline:
    """Dense sampling of a spline (see module docstring for conventions)."""

    __slots__ = ("pts", "tilt", "u", "cyclic", "n_ctrl", "corner_angle")

    def __init__(self, pts, tilt, u, cyclic, n_ctrl, corner_angle=None):
        self.pts = np.asarray(pts, dtype=np.float64)
        self.tilt = np.asarray(tilt, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float64)
        self.cyclic = bool(cyclic)
        self.n_ctrl = int(n_ctrl)
        if corner_angle is None:
            corner_angle = np.zeros(self.n_ctrl)
        self.corner_angle = np.asarray(corner_angle, dtype=np.float64)

    def copy(self):
        return Polyline(self.pts.copy(), self.tilt.copy(), self.u.copy(), self.cyclic,
                        self.n_ctrl, self.corner_angle.copy())

    @property
    def n_segments(self):
        """Number of control segments."""
        return self.n_ctrl if self.cyclic else self.n_ctrl - 1


# ---------------------------------------------------------------------------
# Spline sampling
# ---------------------------------------------------------------------------

def _bezier_eval(p0, p1, p2, p3, t):
    mt = 1.0 - t
    return ((mt ** 3)[:, None] * p0 + (3.0 * mt * mt * t)[:, None] * p1
            + (3.0 * mt * t * t)[:, None] * p2 + (t ** 3)[:, None] * p3)


def bezier_segment_steps(p0, p1, p2, p3, resolution=0, max_angle=ADAPTIVE_STEP):
    """Number of linear pieces used for one cubic Bezier segment.

    Straight segments always use a single piece so sharp, straight paths stay
    exact and light. ``resolution > 0`` forces a fixed count for curved
    segments, otherwise the count adapts to the turning of the control polygon.
    """
    chord = p3 - p0
    clen = float(np.linalg.norm(chord))
    span = max(clen, float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(p2 - p3)), 1e-9)
    tol = 1e-6 * span
    if clen > tol:
        d = chord / clen
        straight = True
        for q in (p1, p2):
            w = q - p0
            along = float(w @ d)
            if np.linalg.norm(w - along * d) > tol or along < -tol or along > clen + tol:
                straight = False
                break
        if straight:
            return 1
    elif np.linalg.norm(p1 - p0) <= tol and np.linalg.norm(p2 - p3) <= tol:
        return 1
    if resolution > 0:
        return int(resolution)
    legs = [leg for leg in (p1 - p0, p2 - p1, p3 - p2) if np.linalg.norm(leg) > tol]
    turn = sum(_angle(a, b) for a, b in zip(legs, legs[1:]))
    return int(min(64, max(2, math.ceil(turn / max_angle))))


def _sample_bezier(sp, resolution):
    co, hl, hr, tilt = sp.co, sp.handle_left, sp.handle_right, sp.tilt
    n = len(co)
    segs = n if sp.cyclic else n - 1
    pts, tilts, us = [], [], []
    ctrl_dense = [0]
    for i in range(segs):
        j = (i + 1) % n
        steps = bezier_segment_steps(co[i], hr[i], hl[j], co[j], resolution)
        t = np.arange(steps, dtype=np.float64) / steps
        pts.append(_bezier_eval(co[i], hr[i], hl[j], co[j], t))
        tilts.append(tilt[i] + (tilt[j] - tilt[i]) * t)
        us.append(i + t)
        ctrl_dense.append(ctrl_dense[-1] + steps)
    last = 0 if sp.cyclic else n - 1
    pts.append(co[last][None, :])
    tilts.append(np.array([tilt[last]]))
    us.append(np.array([float(n if sp.cyclic else n - 1)]))
    pts = np.concatenate(pts)
    tilts = np.concatenate(tilts)
    us = np.concatenate(us)

    # Corner angles from the analytic tangents (handles) at each control point.
    angles = np.zeros(n)
    for k in range(n):
        if not sp.cyclic and (k == 0 or k == n - 1):
            continue
        d = ctrl_dense[k]
        tin = co[k] - hl[k]
        if np.linalg.norm(tin) < 1e-9:
            tin = co[k] - pts[d - 1] if d > 0 else co[k] - pts[-2]
        tout = hr[k] - co[k]
        if np.linalg.norm(tout) < 1e-9:
            tout = pts[d + 1] - co[k]
        angles[k] = _angle(tin, tout)
    return pts, tilts, us, angles


def _sample_poly(sp):
    co, tilt = sp.co, sp.tilt
    n = len(co)
    pts = co.copy()
    tilts = tilt.copy()
    us = np.arange(n, dtype=np.float64)
    if sp.cyclic:
        pts = np.vstack([pts, co[:1]])
        tilts = np.append(tilts, tilt[0])
        us = np.append(us, float(n))
    angles = np.zeros(n)
    for k in range(n):
        if not sp.cyclic and (k == 0 or k == n - 1):
            continue
        # Skip coincident neighbours to find meaningful directions.
        tin = None
        for step in range(1, n):
            prev = co[(k - step) % n]
            if not sp.cyclic and k - step < 0:
                break
            if np.linalg.norm(co[k] - prev) > 1e-9:
                tin = co[k] - prev
                break
        tout = None
        for step in range(1, n):
            if not sp.cyclic and k + step >= n:
                break
            nxt = co[(k + step) % n]
            if np.linalg.norm(nxt - co[k]) > 1e-9:
                tout = nxt - co[k]
                break
        if tin is not None and tout is not None:
            angles[k] = _angle(tin, tout)
    return pts, tilts, us, angles


def nurbs_knots(pnts, order, cyclic, endpoint, bezier):
    """Knot vector generation, ported from Blender's ``calculate_knots``."""
    repeat_inner = order - 1 if bezier else 1
    if endpoint:
        head = order - (1 if cyclic else 0)
    else:
        head = min(2, repeat_inner) if bezier else 1
    tail = 2 * order - 1 if cyclic else (order if endpoint else 0)
    count = pnts + order + (order - 1 if cyclic else 0)
    knots = [0.0] * count
    r = head
    current = 0.0
    offset = 1 if (endpoint and cyclic) else 0
    if offset:
        knots[0] = current
        current += 1.0
    for i in range(offset, count - tail):
        knots[i] = current
        r -= 1
        if r == 0:
            current += 1.0
            r = repeat_inner
    tail_index = count - tail
    for i in range(tail):
        knots[tail_index + i] = current + (knots[i] - knots[0])
    return np.array(knots, dtype=np.float64)


def _bspline_basis(knots, order, n_ctrl, t):
    """Cox-de Boor basis functions, shape (len(t), n_ctrl)."""
    t = np.asarray(t, dtype=np.float64)
    k = knots
    nb = len(k) - 1
    basis = np.zeros((len(t), nb))
    # Degree 0: half-open spans; the very last parameter goes into the last non-empty span.
    for i in range(nb):
        if k[i + 1] > k[i]:
            basis[:, i] = (t >= k[i]) & (t < k[i + 1])
    t_max = k[n_ctrl] if n_ctrl < len(k) else k[-1]
    at_end = t >= t_max - 1e-12
    spans = [i for i in range(min(n_ctrl, nb)) if k[i + 1] > k[i]]
    if at_end.any() and spans:
        basis[at_end, :] = 0.0
        basis[at_end, spans[-1]] = 1.0
    for d in range(1, order):
        nxt = np.zeros((len(t), nb - d))
        for i in range(nb - d):
            left = k[i + d] - k[i]
            right = k[i + d + 1] - k[i + 1]
            term = 0.0
            if left > 0:
                term = term + (t - k[i]) / left * basis[:, i]
            if right > 0:
                term = term + (k[i + d + 1] - t) / right * basis[:, i + 1]
            nxt[:, i] = term
        basis = nxt
    return basis[:, :n_ctrl]


def _sample_nurbs(sp, resolution):
    co, tilt, w = sp.co, sp.tilt, sp.weights
    n = len(co)
    order = max(2, min(sp.order, n if not sp.cyclic else n + 1))
    if n < 2:
        return None
    res = resolution if resolution > 0 else max(sp.resolution, 12)
    segs = n if sp.cyclic else n - 1
    total = max(2, res * segs)
    cyc = order - 1 if sp.cyclic else 0
    pnts = n + cyc
    knots = nurbs_knots(n, order, sp.cyclic, sp.use_endpoint, sp.use_bezier)
    if sp.cyclic:
        t0, t1 = knots[order - 1], knots[n + order - 1]
        t = t0 + (t1 - t0) * np.arange(total) / total
    else:
        t0, t1 = knots[order - 1], knots[n]
        t = t0 + (t1 - t0) * np.arange(total) / (total - 1)
    idx = np.arange(pnts) % n
    basis = _bspline_basis(knots, order, pnts, t) * w[idx][None, :]
    denom = basis.sum(axis=1)
    denom[denom == 0.0] = 1.0
    pts = (basis @ co[idx]) / denom[:, None]
    tilts = (basis @ tilt[idx]) / denom
    # Control parameter: map knot parameter through the Greville abscissae.
    grev = np.array([knots[i + 1:i + order].mean() for i in range(pnts)])
    if sp.cyclic:
        # Cyclic NURBS start somewhere inside control span 0; ``u`` simply keeps
        # increasing by ``n`` over the loop (see Path.s_of_u for the wrap).
        u = np.interp(t, grev, np.arange(pnts, dtype=np.float64))
        pts = np.vstack([pts, pts[:1]])
        tilts = np.append(tilts, tilts[0])
        u = np.append(u, u[0] + n)
    else:
        u = np.interp(t, grev, np.arange(n, dtype=np.float64))
        u[0], u[-1] = 0.0, float(n - 1)
    u = np.maximum.accumulate(u)
    return pts, tilts, u, np.zeros(n)


def _dedupe(pts, tilts, us, cyclic, tol=DEDUPE_TOL):
    """Remove consecutive coincident points (keeps the closing point of loops)."""
    if len(pts) < 2:
        return pts, tilts, us
    keep = np.ones(len(pts), dtype=bool)
    last = pts[0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - last) <= tol:
            keep[i] = False
        else:
            last = pts[i]
    if cyclic:
        # The closing duplicate must survive; drop the point before it instead.
        if not keep[-1]:
            keep[-1] = True
            j = len(pts) - 2
            while j > 0 and not keep[j]:
                j -= 1
            if j > 0:
                keep[j] = False
    pts, tilts, us = pts[keep], tilts[keep], us[keep]
    # Make sure u is strictly increasing (needed for interpolation).
    for i in range(1, len(us)):
        if us[i] <= us[i - 1]:
            us[i] = us[i - 1] + 1e-9
    return pts, tilts, us


def sample_spline(sp, resolution=0):
    """Sample a :class:`SplineInput` into a :class:`Polyline` (``None`` if degenerate)."""
    n = len(sp.co)
    if n < 2:
        return None
    if sp.kind == "BEZIER" and sp.handle_left is not None and sp.handle_right is not None:
        res = _sample_bezier(sp, resolution)
    elif sp.kind == "NURBS" and n > 2:
        res = _sample_nurbs(sp, resolution)
    else:
        res = _sample_poly(sp)
    if res is None:
        return None
    pts, tilts, us, angles = res
    pts, tilts, us = _dedupe(pts, tilts, us, sp.cyclic)
    if len(pts) < (3 if sp.cyclic else 2):
        return None
    return Polyline(pts, tilts, us, sp.cyclic, n, angles)


# ---------------------------------------------------------------------------
# Polyline operations
# ---------------------------------------------------------------------------

def arclength(pts):
    seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def reverse_polyline(poly):
    """Reverse travel direction; ``u`` keeps referring to the same control points."""
    return Polyline(poly.pts[::-1].copy(), poly.tilt[::-1].copy(), poly.u[::-1].copy(),
                    poly.cyclic, poly.n_ctrl, poly.corner_angle.copy())


def densify_polyline(poly, step):
    """Insert points so that no segment is longer than ``step``."""
    if step <= 0:
        return poly
    pts, tilt, u = poly.pts, poly.tilt, poly.u
    out_p, out_t, out_u = [pts[:1]], [tilt[:1]], [u[:1]]
    for i in range(len(pts) - 1):
        seg_len = float(np.linalg.norm(pts[i + 1] - pts[i]))
        n = max(1, int(math.ceil(seg_len / step - 1e-9)))
        f = np.arange(1, n + 1, dtype=np.float64) / n
        out_p.append(pts[i] + (pts[i + 1] - pts[i]) * f[:, None])
        out_t.append(tilt[i] + (tilt[i + 1] - tilt[i]) * f)
        out_u.append(u[i] + (u[i + 1] - u[i]) * f)
    return Polyline(np.concatenate(out_p), np.concatenate(out_t), np.concatenate(out_u),
                    poly.cyclic, poly.n_ctrl, poly.corner_angle)


def corner_indices(poly, threshold):
    """Control points whose turning angle exceeds ``threshold`` (radians)."""
    return [k for k in range(poly.n_ctrl) if poly.corner_angle[k] > threshold + 1e-12]


def _point_and_dir(pts, S, s):
    """Point at arc length ``s`` and the direction of the containing segment."""
    i = int(np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(pts) - 2))
    seg = pts[i + 1] - pts[i]
    ln = S[i + 1] - S[i]
    t = 0.0 if ln <= 0 else (s - S[i]) / ln
    return pts[i] + seg * t, seg / max(ln, 1e-12), i, t


def fillet_polyline(poly, corner_ks, radius):
    """Round the given control-point corners with circular arcs.

    Returns ``(new_polyline, rounded_set)``. The effective radius is limited so
    neighbouring fillets never overlap (each corner gets at most half of the
    distance to the next rounded corner, or the whole distance to an open end).
    """
    if radius <= 0 or not corner_ks:
        return poly, set()
    pts, tilt, u = poly.pts, poly.tilt, poly.u
    S = arclength(pts)
    L = float(S[-1])
    n_ctrl = poly.n_ctrl
    cyclic = poly.cyclic

    # Locate corners (exact dense points where u == k) and their angles.
    corners = []
    for k in corner_ks:
        hits = np.where(np.abs(u - k) < 1e-7)[0]
        if len(hits) == 0:
            continue
        i = int(hits[0])
        if cyclic and i == len(pts) - 1:
            i = 0
        if not cyclic and (i == 0 or i == len(pts) - 1):
            continue
        prev_i = i - 1 if i > 0 else len(pts) - 2
        next_i = i + 1
        tin = pts[i] - pts[prev_i]
        tout = pts[next_i] - pts[i]
        theta = _angle(tin, tout)
        if theta < 1e-4 or theta > math.pi - 1e-3:
            continue
        corners.append([float(S[i]), k, theta, radius * math.tan(theta / 2.0)])
    if not corners:
        return poly, set()
    corners.sort(key=lambda c: c[0])

    m = len(corners)
    for j, c in enumerate(corners):
        s_j = c[0]
        if m > 1 or cyclic:
            if j > 0:
                before = (s_j - corners[j - 1][0]) * 0.5
            elif cyclic:
                before = (s_j - (corners[-1][0] - L)) * 0.5 if m > 1 else L * 0.5
            else:
                before = s_j
            if j < m - 1:
                after = (corners[j + 1][0] - s_j) * 0.5
            elif cyclic:
                after = ((corners[0][0] + L) - s_j) * 0.5 if m > 1 else L * 0.5
            else:
                after = L - s_j
        else:
            before, after = s_j, L - s_j
        c[3] = max(0.0, min(c[3], before, after) * (1.0 - 1e-6))

    def wrap_s(s):
        return s % L if cyclic else min(max(s, 0.0), L)

    def u_at(s):
        return float(np.interp(s, S, u))

    def tilt_at(s):
        return float(np.interp(s, S, tilt))

    keep = np.ones(len(pts), dtype=bool)
    if cyclic:
        keep[-1] = False  # closing duplicate is rebuilt below
    extra_p, extra_t, extra_u = [], [], []
    rounded = set()
    for s_c, k, theta, d in corners:
        if d <= 1e-9:
            continue
        sa, sb = s_c - d, s_c + d
        # Remove original points strictly inside the window (wrap-aware).
        if cyclic:
            Sm = S[:-1]
            if sa < 0:
                inside = (Sm > sa + L) | (Sm < sb)
            elif sb > L:
                inside = (Sm > sa) | (Sm < sb - L)
            else:
                inside = (Sm > sa) & (Sm < sb)
            keep[:-1] &= ~inside
        else:
            keep &= ~((S > sa) & (S < sb))
        A, TA, _, _ = _point_and_dir(pts, S, wrap_s(sa) if sa >= 0 or cyclic else 0.0)
        B, TB, _, _ = _point_and_dir(pts, S, wrap_s(sb))
        if cyclic and sa < 0:
            ua = u_at(sa + L) - n_ctrl
        else:
            ua = u_at(sa)
        if cyclic and sb > L:
            ub = u_at(sb - L) + n_ctrl
        else:
            ub = u_at(sb)
        ta, tb = tilt_at(wrap_s(sa)), tilt_at(wrap_s(sb))
        phi = _angle(TA, TB)
        chord = float(np.linalg.norm(B - A))
        if phi < 1e-6 or chord < 1e-12:
            h = chord / 3.0
        else:
            r_arc = chord / (2.0 * math.sin(phi / 2.0))
            h = 4.0 / 3.0 * math.tan(phi / 4.0) * r_arc
        steps = max(2, int(math.ceil(phi / ARC_STEP)))
        tt = np.arange(0, steps + 1, dtype=np.float64) / steps
        arc = _bezier_eval(A, A + TA * h, B - TB * h, B, tt)
        au = ua + (ub - ua) * tt
        at = ta + (tb - ta) * tt
        if cyclic:
            # Wrap control parameters into [0, n_ctrl); add an exact seam point if crossed.
            seam = None
            if au[0] < 0.0 <= au[-1]:
                seam = 0.0
            elif au[0] < n_ctrl <= au[-1]:
                seam = float(n_ctrl)
            if seam is not None:
                f = (seam - ua) / (ub - ua)
                p_seam = _bezier_eval(A, A + TA * h, B - TB * h, B, np.array([f]))[0]
                extra_p.append(p_seam[None, :])
                extra_t.append(np.array([ta + (tb - ta) * f]))
                extra_u.append(np.array([0.0]))
            au = np.mod(au, n_ctrl)
            if seam is not None:
                near = np.abs(au) < 1e-9
                arc, at, au = arc[~near], at[~near], au[~near]
        extra_p.append(arc)
        extra_t.append(at)
        extra_u.append(au)
        rounded.add(k)

    all_p = np.concatenate([pts[keep]] + extra_p)
    all_t = np.concatenate([tilt[keep]] + extra_t)
    all_u = np.concatenate([u[keep]] + extra_u)
    order = np.argsort(all_u, kind="stable")
    all_p, all_t, all_u = all_p[order], all_t[order], all_u[order]
    if cyclic:
        all_p = np.vstack([all_p, all_p[:1]])
        all_t = np.append(all_t, all_t[0])
        all_u = np.append(all_u, all_u[0] + n_ctrl)
    all_p, all_t, all_u = _dedupe(all_p, all_t, all_u, cyclic)
    out = Polyline(all_p, all_t, all_u, cyclic, n_ctrl, poly.corner_angle.copy())
    for k in rounded:
        out.corner_angle[k] = 0.0
    return out, rounded


# ---------------------------------------------------------------------------
# Path: arc length, frames and deformation
# ---------------------------------------------------------------------------

def _insert_miter_zones(P, u, tilt, cyclic, extent):
    """Insert straight-frame vertices close to sharp corners.

    Offsets are interpolated linearly between vertex axes, so without extra
    vertices the miter of a corner would shear the geometry over the whole
    adjacent segment. Two extra vertices at distance ``Z`` from each corner
    confine the miter blend to a short zone (``Z`` grows with the geometry
    extent and the sharpness of the corner).
    """
    n = len(P)
    if n < 3:
        return P, u, tilt
    seg = P[1:] - P[:-1]
    seglen = np.maximum(_norm(seg), 1e-12)
    T = seg / seglen[:, None]
    if cyclic:
        verts = np.arange(0, n - 1)
        j_in = np.where(verts == 0, n - 2, verts - 1)
    else:
        verts = np.arange(1, n - 1)
        j_in = verts - 1
    j_out = verts
    a, b = T[j_in], T[j_out]
    theta = np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))
    ah, bh = a.copy(), b.copy()
    ah[:, 2] = 0.0
    bh[:, 2] = 0.0
    ah, ok_a = _normalize(ah)
    bh, ok_b = _normalize(bh)
    theta_h = np.where(ok_a & ok_b, np.arccos(np.clip(np.einsum("ij,ij->i", ah, bh), -1.0, 1.0)), 0.0)
    th = np.maximum(theta, theta_h)
    sharp = th >= math.radians(2.0)
    Z = np.maximum(4.0 * extent * np.tan(np.minimum(th, math.radians(170.0)) * 0.5), 1e-4)
    extra = []
    for jin, jout, z in zip(j_in[sharp], j_out[sharp], Z[sharp]):
        if z < 0.45 * seglen[jin]:
            extra.append((int(jin), 1.0 - z / seglen[jin]))
        if z < 0.45 * seglen[jout]:
            extra.append((int(jout), z / seglen[jout]))
    if not extra:
        return P, u, tilt
    ej = np.array([j for j, _ in extra], dtype=np.int64)
    ef = np.array([f for _, f in extra], dtype=np.float64)
    keys = np.concatenate([np.arange(n, dtype=np.float64), ej + ef])
    pts = np.vstack([P, P[ej] + seg[ej] * ef[:, None]])
    us = np.concatenate([u, u[ej] + (u[ej + 1] - u[ej]) * ef])
    ts = np.concatenate([tilt, tilt[ej] + (tilt[ej + 1] - tilt[ej]) * ef])
    order = np.argsort(keys, kind="stable")
    return pts[order], us[order], ts[order]


class Path:
    """Arc-length parameterised polyline with miter-aware frames.

    Two placement modes are supported:

    * *follow* (default): the sample's Z axis follows the path normal, so
      geometry tilts with slopes and banks with the curve tilt.
    * *vertical*: the sample stays upright (world Z) and is sheared along
      slopes, like a fence on a hill.

    Offsets are interpolated between per-vertex axes that are scaled by the
    miter factor, so geometry wrapped around a sharp corner closes exactly on
    the bisector plane.
    """

    def __init__(self, poly, twist="Z_UP", use_tilt=True, miter_extent=0.0):
        P = np.asarray(poly.pts, dtype=np.float64)
        if len(P) < 2:
            raise ValueError("path needs at least two points")
        self.cyclic = poly.cyclic
        u = np.asarray(poly.u, dtype=np.float64)
        tilt = np.asarray(poly.tilt, dtype=np.float64)
        if miter_extent > 0.0:
            P, u, tilt = _insert_miter_zones(P, u, tilt, poly.cyclic, miter_extent)
        self.P = P
        self.u = u
        self.n_ctrl = poly.n_ctrl
        seg = P[1:] - P[:-1]
        seglen = _norm(seg)
        seglen = np.maximum(seglen, 1e-12)
        self.seglen = seglen
        self.S = np.concatenate(([0.0], np.cumsum(seglen)))
        self.L = float(self.S[-1])
        self.Tseg = seg / seglen[:, None]
        self.tilt = tilt if use_tilt else np.zeros(len(P))
        self._build_frames(twist)

    # -- construction -------------------------------------------------------

    def _build_frames(self, twist):
        P, Tseg = self.P, self.Tseg
        n = len(P)
        tin = np.empty((n, 3))
        tout = np.empty((n, 3))
        tin[1:] = Tseg
        tout[:-1] = Tseg
        if self.cyclic:
            tin[0] = Tseg[-1]
            tout[-1] = Tseg[0]
        else:
            tin[0] = Tseg[0]
            tout[-1] = Tseg[-1]
        self.tin, self.tout = tin, tout
        T, ok = _normalize(tin + tout)
        T[~ok] = tout[~ok]
        self.Tv = T

        # Lateral axis (left) and normal (up) per vertex.
        if twist == "MINIMUM":
            L = self._minimum_twist_laterals(T)
        else:
            L, okl = _normalize(np.cross(UP, T))
            if not okl.any():
                L[:] = (0.0, 1.0, 0.0)
                okl[:] = True
            L = _fill_invalid(L, okl)
            L, _ = _normalize(L - np.einsum("ij,ij->i", L, T)[:, None] * T)
        N = np.cross(T, L)
        if np.any(self.tilt != 0.0):
            a = self.tilt
            ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
            L, N = L * ca + N * sa, N * ca - L * sa
        self.Lv, self.Nv = L, N

        # Miter-scaled offset axes for the follow mode.
        c = np.einsum("ij,ij->i", T, tin)
        m = 1.0 / np.clip(c, 1.0 / MAX_MITER, 1.0)
        K, okk = _normalize(tout - tin)
        ky = np.einsum("ij,ij->i", L, K) * (m - 1.0) * okk
        kz = np.einsum("ij,ij->i", N, K) * (m - 1.0) * okk
        self.Ay = L + ky[:, None] * K
        self.Az = N + kz[:, None] * K

        # Horizontal frames for the vertical mode.
        def horiz(v):
            h = v.copy()
            h[:, 2] = 0.0
            return _normalize(h)
        hin, ok_in = horiz(tin)
        hout, ok_out = horiz(tout)
        hin[~ok_in] = hout[~ok_in]
        hout[~ok_out] = hin[~ok_out]
        okh = ok_in | ok_out
        if not okh.any():
            hin[:] = (1.0, 0.0, 0.0)
            hout[:] = (1.0, 0.0, 0.0)
            okh[:] = True
        hin = _fill_invalid(hin, okh)
        hout = _fill_invalid(hout, okh)
        Th, okt = _normalize(hin + hout)
        Th[~okt] = hout[~okt]
        self.Thv = Th
        Lh = np.stack([-Th[:, 1], Th[:, 0], np.zeros(n)], axis=1)
        self.Lhv = Lh
        ch = np.einsum("ij,ij->i", Th, hin)
        mh = 1.0 / np.clip(ch, 1.0 / MAX_MITER, 1.0)
        self.Hy = Lh * mh[:, None]

    def _minimum_twist_laterals(self, T):
        n = len(T)
        L = np.empty((n, 3))
        first = np.cross(UP, T[0])
        if np.linalg.norm(first) < 1e-6:
            first = np.cross((0.0, 1.0, 0.0), T[0])
            first = np.cross(T[0], first)
        L[0] = first / np.linalg.norm(first)
        for i in range(1, n):
            a, b = T[i - 1], T[i]
            v = L[i - 1]
            axis = np.cross(a, b)
            s = np.linalg.norm(axis)
            if s > 1e-12:
                axis /= s
                ang = math.atan2(s, float(a @ b))
                v = _rotate(v[None, :], axis[None, :], np.array([ang]))[0]
            v = v - (v @ b) * b
            nv = np.linalg.norm(v)
            L[i] = v / nv if nv > 1e-12 else L[i - 1]
        if self.cyclic and n > 2:
            # Distribute the closing twist mismatch along the loop.
            a = L[-1]
            b = L[0]
            ang = math.atan2(float(np.cross(a, b) @ T[0]), float(a @ b))
            frac = self.S / max(self.L, 1e-12)
            L = _rotate(L, T, ang * frac)
        return L

    # -- evaluation ---------------------------------------------------------

    def _locate(self, s):
        s = np.asarray(s, dtype=np.float64)
        if self.cyclic:
            s = np.mod(s, self.L)
        i = np.searchsorted(self.S, s, side="right") - 1
        i = np.clip(i, 0, len(self.P) - 2)
        t = (s - self.S[i]) / self.seglen[i]
        if self.cyclic:
            t = np.clip(t, 0.0, 1.0)
        return s, i, t

    def point(self, s):
        """Centre-line positions at arc lengths ``s`` (extrapolated past open ends)."""
        s, i, t = self._locate(np.atleast_1d(s))
        return self.P[i] + (self.P[i + 1] - self.P[i]) * t[:, None]

    def elevation(self, s):
        return self.point(s)[:, 2]

    def evaluate(self, s, y, z, vertical=False):
        """Map sample coordinates (arc length, lateral, up) to world positions."""
        s, i, t = self._locate(np.atleast_1d(s))
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        tc = np.clip(t, 0.0, 1.0)
        w0 = (1.0 - tc)[:, None]
        w1 = tc[:, None]
        base = self.P[i] + (self.P[i + 1] - self.P[i]) * t[:, None]
        if vertical:
            off = (w0 * self.Hy[i] + w1 * self.Hy[i + 1]) * y[:, None]
            out = base + off
            out[:, 2] += z
        else:
            out = base + (w0 * self.Ay[i] + w1 * self.Ay[i + 1]) * y[:, None] \
                + (w0 * self.Az[i] + w1 * self.Az[i + 1]) * z[:, None]
        return out

    def frames(self, s, vertical=False, orient="AUTO"):
        """Orthonormal frames ``(P, T, L, N)`` at arc lengths ``s``.

        At path vertices the tangent is the corner bisector (``AUTO``) or the
        incoming / outgoing direction.
        """
        s, i, t = self._locate(np.atleast_1d(s))
        P = self.P[i] + (self.P[i + 1] - self.P[i]) * t[:, None]
        T = self.Tseg[i].copy()
        tc = np.clip(t, 0.0, 1.0)[:, None]
        L = self.Lv[i] * (1.0 - tc) + self.Lv[i + 1] * tc
        tol = 1e-6
        at0 = np.abs(s - self.S[i]) < tol
        at1 = np.abs(s - self.S[i + 1]) < tol
        vtx = np.where(at1, i + 1, i)
        on_vertex = at0 | at1
        if on_vertex.any():
            vi = vtx[on_vertex]
            if orient == "INCOMING":
                T[on_vertex] = self.tin[vi]
            elif orient == "OUTGOING":
                T[on_vertex] = self.tout[vi]
            else:
                T[on_vertex] = self.Tv[vi]
            L[on_vertex] = self.Lv[vi]
        if vertical:
            Th = T.copy()
            Th[:, 2] = 0.0
            Th, ok = _normalize(Th)
            if (~ok).any():
                fallback = self.Thv[i] * (1.0 - tc) + self.Thv[i + 1] * tc
                Th[~ok] = _normalize(fallback[~ok])[0]
            Lh = np.stack([-Th[:, 1], Th[:, 0], np.zeros(len(Th))], axis=1)
            N = np.broadcast_to(UP, Th.shape).copy()
            return P, Th, Lh, N
        L = L - np.einsum("ij,ij->i", L, T)[:, None] * T
        L, ok = _normalize(L)
        if (~ok).any():
            L[~ok] = _normalize(np.cross(UP, T[~ok]))[0]
        N = np.cross(T, L)
        return P, T, L, N

    # -- control parameter mapping -------------------------------------------

    def s_of_u(self, k):
        """Arc length where the control parameter equals ``k``."""
        u, S = self.u, self.S
        if self.cyclic:
            lo = min(u[0], u[-1])
            k = lo + np.mod(np.asarray(k, dtype=np.float64) - lo, self.n_ctrl)
        if u[-1] >= u[0]:
            return np.interp(k, u, S)
        return np.interp(k, u[::-1], S[::-1])

    def u_of_s(self, s):
        s = np.mod(s, self.L) if self.cyclic else s
        return np.interp(s, self.S, self.u)
