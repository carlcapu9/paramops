"""Implicit tube field of the sculpture: members fused into one organic solid.

Every :class:`Tube` (a polyline with a radius and a carried value per point) adds
a smooth implicit field on a regular, element-centred grid (``origin + (ijk + 0.5)
* h``, like every field of the add-on); meshing the result at :data:`TUBE_ISO`
(``mesher.surface_nets``) gives one watertight surface.

Field of one tube
    For a grid point ``p`` and a polyline segment ``[a, b]`` with end radii
    ``r_a, r_b``::

        t = clamp(dot(p - a, b - a) / |b - a|^2, 0, 1)      # closest point parameter
        d = |p - (a + t (b - a))|                             # capsule distance
        r = r_a + t (r_b - r_a)                               # radius, linear along the segment
        f = phi(T * d / r),   phi(u) = (1 - u^2)^3 for u < 1, else 0   (Wyvill kernel)

    with ``T = sqrt(1 - TUBE_ISO ** (1/3))`` so that ``f == TUBE_ISO`` exactly at
    ``d == r``. The kernel reaches zero at ``TUBE_SUPPORT * r`` (~2.2 r). Each
    segment is evaluated only inside its *slab* (between the planes through its
    ends, perpendicular to it), extended by the full support at the ends of the
    polyline (round caps) and, at a joint, by ``support * sin(bend / 2)`` (the full
    support for bends over 90 degrees), just enough to cover the outer wedge of the
    bend; a tube is the maximum of its restricted segments. On a straight run this
    is exactly the surface of revolution of the linearly interpolated radius - no
    beads at the joints of a tapering polyline (an unrestricted capsule union would
    let the joint spheres poke out of the narrowing segment by up to ~7 % of the
    radius); bends get round outer corners. (The restriction can leave steps of
    up to about 1 % of the radius on the inside of bends of strongly tapering
    tubes.) ``TUBE_ISO = 0.5`` puts the surface next to the inflection point of
    ``phi`` (``u^2 = 1/5``), where the field is almost linear across the surface, so
    the linear edge interpolation of the mesher is very accurate.

Smooth union (``blend`` in [0, 1])
    Tubes meet in a soft maximum whose temperature follows the angle of the crease
    they form, so creases get round fillets while overlapping or coaxial members
    do not swell. Per grid point, over the tubes ``i`` that reach it::

        tau = TAU_MAX * blend                              # 0: plain maximum
        F = max_i f_i,   S = sum_i (exp(f_i / tau) - 1)    # soft-max: tau ln(1 + S)
        theta = crease angle (angle between the tubes' field gradients)
        tau' = tau * sin(theta) for theta < 90 deg, else tau
        U = tau' ln(exp(F / tau') + exp(G / tau') - 1)

    ``G = tau ln(1 + S - exp(F / tau))`` is the soft maximum of the other tubes (the
    second tube itself when two meet; clamped towards ``F`` as ``tau'`` vanishes),
    and ``theta`` comes from the weighted mean of the pairwise misalignments
    ``1 - cos(theta_ij)`` of the unit gradients, with weights that do not depend on
    ``blend`` (exactly the angle between the two gradients when two tubes meet). The
    gradient of a tube's field points to the closest point of its axis.

    * ``blend = 0`` gives the plain maximum (hard union, sharp creases),
    * a tube alone keeps its exact field for every blend: a lone tube's surface is
      always at its radius, whatever the blend,
    * the extra material is bounded: ``F <= U <= tau ln(1 + S) <= F + tau ln(n)``;
      at a crease of 90 degrees or more (crossings, the crotch between branches)
      ``U`` is the full soft maximum (the old metaball-like fillet, ~0.15 r deep
      at ``blend = 0.45`` for a right-angle crossing), while for a crease of angle
      ``theta`` both the fillet's footprint (~``2 tau / |grad f|``, ~0.3 r at
      ``blend = 0.45``) and its depth (``prop. to sin(theta / 2)``) behave like a
      rolling-ball fillet: coaxial or touching-parallel overlaps (``theta = 0``) and
      the tops of crossings get no extra material, a shallow junction gets a short,
      shallow fillet instead of a bulge,
    * symmetric in the tubes and independent of their order and of the chunking
      (every per-point quantity is a sum or a maximum over tubes), continuous, and
      non-decreasing in ``blend`` point by point.

    ``value_field`` is the average of the tubes' values weighted by
    ``exp(f_i / tau) - 1`` (the value of the dominant tube when ``blend = 0``):
    colors follow the members and blend only where they meet.
    :func:`union_fields` combines two whole fields with the same rule (crease
    angle from their central-difference gradients).

Memory is bounded: the grid is processed in slabs of whole x-layers with at most
``_SLAB_CELLS`` cells (the per-point accumulators exist for one slab only) and the
kernel evaluations in batches of at most ``_BATCH`` points, about 150 MB of
temporaries at most; besides the two output arrays nothing scales with the grid.
Pure numpy, no global mutable state.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid import ProgressFn, StopFn, check_stop


__all__ = [
    "Tube",
    "TUBE_ISO",
    "TUBE_SUPPORT",
    "TAU_MAX",
    "MAX_CELLS",
    "blend_tau",
    "tubes_field",
    "evaluate_tubes",
    "resample_to",
    "union_fields",
]

TUBE_ISO = 0.5  # iso level to mesh tubes_field with
_T_ISO = float(np.sqrt(1.0 - TUBE_ISO ** (1.0 / 3.0)))  # kernel argument at the tube surface
TUBE_SUPPORT = 1.0 / _T_ISO  # radius of influence / tube radius (~2.2018)
TAU_MAX = 0.25  # soft-max temperature at blend = 1
_TAU_MIN = 0.005  # below this the union is the exact maximum
_TAU_ANGLE = 0.1  # fixed temperature of the weights that average the crease angle
_PAIR_TOL = 1e-9  # pair weight (relative) below which one tube dominates alone
_MIS_DEAD = 1e-10  # misalignment 1 - cos(theta) treated as parallel (rounding noise)
_BATCH = 1 << 18  # grid-point evaluations per batch (bounds the temporaries)
_MAX_SEGMENT_VOXELS = 32.0  # longer segments are cut into collinear pieces
_COLUMN_BATCH = 1 << 18  # grid columns per batch of segments
_SLAB_CELLS = 1 << 20  # cells per slab of accumulators (8 float64 arrays: 64 MB)
_RESAMPLE_BLOCK = 1 << 22  # output elements per block in resample_to
_UNION_BLOCK = 1 << 20  # elements per block in union_fields (gradients need ~12 temporaries)
MAX_CELLS = 60_000_000  # refuse larger grids (the two float64 outputs alone ~1 GB)


@dataclass
class Tube:
    """A tube along a polyline.

    ``points (n, 3)`` world positions, ``radius (n,)`` world radii and ``value (n,)``
    a scalar carried to the surface (e.g. stress magnitude, any units). ``radius``
    and ``value`` may be scalars (broadcast); ``value`` defaults to 0. Points with
    non-finite coordinates or a radius <= 0 interrupt the tube.
    """

    points: np.ndarray
    radius: np.ndarray
    value: Optional[np.ndarray] = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        n = self.points.shape[0]
        self.radius = _per_point(self.radius, n, "radius")
        self.value = _per_point(0.0 if self.value is None else self.value, n, "value")


def _per_point(values, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == n:
        return arr.reshape(n).copy()
    if arr.size == 1:
        return np.full(n, float(arr.reshape(-1)[0]))
    raise ValueError(f"Tube.{name} must be a scalar or have one entry per point")


def blend_tau(blend) -> float:
    """Soft-max temperature of a blend factor (0 means the exact maximum)."""
    b = float(blend)
    b = 0.0 if not np.isfinite(b) else min(max(b, 0.0), 1.0)
    tau = TAU_MAX * b
    return tau if tau >= _TAU_MIN else 0.0


# ============================================================================ segments
class _Segments:
    """Flat arrays of all tube segments, in tube order (``owner`` = tube index).

    Segment ``k`` starts at ``a[k]`` and runs ``length[k]`` along the unit direction
    ``u[k]`` (``length = 0``, ``u = +z`` for an isolated point: a ball). Its slab
    ``0 <= (p - a) . u <= length`` is extended by ``ext0`` before the start and
    ``ext1`` after the end: by the full support at the ends of a polyline (round
    caps) and by ``support * sin(bend / 2)`` at a joint (the full support for bends
    over 90 degrees). The wedge on the outer side of a bend lies in no slab; split
    by the bisector plane, each half is within that distance of its segment's slab.
    """

    def __init__(self, a, u, length, ra, rb, va, vb, ext0, ext1, owner):
        self.n = a.shape[0]
        self.a = a
        self.u = u
        self.length = length
        self.inv_length = np.zeros(self.n)
        np.divide(1.0, length, out=self.inv_length, where=length > 0.0)
        self.ra = ra
        self.dr = rb - ra
        self.support = np.maximum(ra, rb) * TUBE_SUPPORT
        self.va = va
        self.dv = vb - va
        self.ext0 = ext0
        self.ext1 = ext1
        self.owner = owner


def _as_tube(tube) -> Tube:
    if isinstance(tube, Tube):
        return tube
    return Tube(points=tube.points, radius=tube.radius, value=getattr(tube, "value", None))


def _polyline_segments(p, r, v, max_length=np.inf) -> Optional[tuple]:
    """Segments of one tube: ``(a, u, length, ra, rb, va, vb, ext0, ext1)``.

    Invalid points (non-finite, radius <= 0) split the polyline; repeated points are
    merged; a valid point without a valid neighbour becomes a ball. Segments longer
    than ``max_length`` are cut into equal collinear pieces (same field: collinear
    joints get no slab extension and radius / value stay linear), which bounds the
    number of grid columns a single segment needs.
    """
    valid = np.isfinite(p).all(axis=1) & np.isfinite(r) & (r > 0.0)
    if not valid.any():
        return None
    v = np.where(np.isfinite(v), v, 0.0)
    piece = np.cumsum(~valid)
    tol = 1e-9 * float(np.max(r[valid]))
    dup = np.zeros(p.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        dup[1:] = valid[1:] & valid[:-1] & (np.linalg.norm(p[1:] - p[:-1], axis=1) <= tol)
    keep = np.flatnonzero(valid & ~dup)
    q, rq, vq, pc = p[keep], r[keep], v[keep], piece[keep]
    if q.shape[0] > 1 and np.isfinite(max_length) and max_length > 0.0:
        seg_len = np.linalg.norm(q[1:] - q[:-1], axis=1)
        pieces = np.where(pc[1:] == pc[:-1], np.ceil(seg_len / max_length), 1.0)
        reps = np.append(np.maximum(pieces, 1.0).astype(np.int64), 1)
        if np.any(reps > 1):
            src = np.repeat(np.arange(q.shape[0]), reps)
            frac = np.arange(src.size) - np.repeat(np.cumsum(reps) - reps, reps)
            frac = frac / reps[src]
            nxt = np.minimum(src + 1, q.shape[0] - 1)
            q = q[src] + frac[:, None] * (q[nxt] - q[src])
            rq = rq[src] + frac * (rq[nxt] - rq[src])
            vq = vq[src] + frac * (vq[nxt] - vq[src])
            pc = pc[src]
    support = rq * TUBE_SUPPORT
    m = q.shape[0]
    link = pc[1:] == pc[:-1]  # q[j] -> q[j + 1] is a segment
    step = q[1:] - q[:-1]
    length = np.linalg.norm(step, axis=1)
    direction = np.zeros((m - 1, 3))
    np.divide(step, length[:, None], out=direction, where=link[:, None])
    seg_support = np.maximum(support[:-1], support[1:])

    j = np.flatnonzero(link)
    ext0 = seg_support[j].copy()
    ext1 = seg_support[j].copy()
    for side, ext in ((-1, ext0), (1, ext1)):
        nb = j + side
        has = (nb >= 0) & (nb < m - 1)
        has[has] = link[nb[has]]
        if has.any():
            k, n = j[has], nb[has]
            cos = np.clip(np.einsum("ij,ij->i", direction[k], direction[n]), -1.0, 1.0)
            half_sin = np.sqrt(0.5 * (1.0 - cos))  # sin(bend / 2)
            reach = np.maximum(seg_support[k], seg_support[n])
            ext[has] = np.where(cos > 0.0, reach * half_sin, reach)
    lone = np.ones(m, dtype=bool)
    lone[:-1] &= ~link
    lone[1:] &= ~link
    i = np.flatnonzero(lone)
    up = np.zeros((i.size, 3))
    up[:, 2] = 1.0
    return (
        np.concatenate([q[j], q[i]]),
        np.concatenate([direction[j], up]),
        np.concatenate([length[j], np.zeros(i.size)]),
        np.concatenate([rq[j], rq[i]]),
        np.concatenate([rq[j + 1], rq[i]]),
        np.concatenate([vq[j], vq[i]]),
        np.concatenate([vq[j + 1], vq[i]]),
        np.concatenate([ext0, support[i]]),
        np.concatenate([ext1, support[i]]),
    )


def _segments(tubes, max_length=np.inf) -> Optional[_Segments]:
    """Segments of all tubes in tube order (none longer than ``max_length``);
    ``None`` if there are none."""
    parts, owners = [], []
    for index, tube in enumerate(tubes):
        tube = _as_tube(tube)
        part = _polyline_segments(tube.points, tube.radius, tube.value, max_length)
        if part is not None and part[0].shape[0]:
            parts.append(part)
            owners.append(np.full(part[0].shape[0], index, dtype=np.int64))
    if not parts:
        return None
    cols = [np.concatenate([part[c] for part in parts]) for c in range(9)]
    return _Segments(*cols, np.concatenate(owners))


def _segment_boxes(seg: _Segments, ids, origin, h) -> tuple:
    """Index ranges ``(lo, hi)`` ``(len(ids), 3)`` int64 of the cells whose centres may lie
    in the extended slab + support cylinder of segments ``ids`` (not clipped to a grid)."""
    u = seg.u[ids]
    a = seg.a[ids]
    start = a - seg.ext0[ids, None] * u
    end = a + (seg.length[ids] + seg.ext1[ids])[:, None] * u
    half = seg.support[ids, None] * np.sqrt(np.maximum(1.0 - u * u, 0.0))
    lo = np.ceil((np.minimum(start, end) - half - origin) / h - 0.5).astype(np.int64)
    hi = np.floor((np.maximum(start, end) + half - origin) / h - 0.5).astype(np.int64)
    return lo, hi


class _Columns:
    """Grid columns (lines of grid points along one axis) crossing segment slabs.

    For column ``c`` of segment ``sid[c]``: grid points ``cell0[c] + k * stride[c]``
    for ``k = 0 .. count[c] - 1``, at offsets ``z0[c] + k * h`` along the column
    axis ``axis[c]`` from the segment start; ``w1[c]``, ``w2[c]`` are the offsets
    along the two other axes (``axis + 1``, ``axis + 2`` mod 3), ``p_dot[c]`` /
    ``q_perp2[c]`` their projection on the segment direction and squared length,
    ``u_col``, ``u1``, ``u2`` the direction components along the three axes.
    """

    def __init__(self, sid, cell0, stride, z0, count, p_dot, q_perp2, u_col, axis, w1, w2, u1, u2):
        self.sid = sid
        self.cell0 = cell0
        self.stride = stride
        self.z0 = z0
        self.count = count
        self.p_dot = p_dot
        self.q_perp2 = q_perp2
        self.u_col = u_col
        self.axis = axis
        self.w1 = w1
        self.w2 = w2
        self.u1 = u1
        self.u2 = u2


def _segment_columns(seg: _Segments, ids, shape, origin, h, lo_clip, hi_clip) -> _Columns:
    """Columns of the extended slabs of segments ``ids`` within the cell box
    ``lo_clip .. hi_clip`` (inclusive index bounds per axis); empty columns dropped.

    Each segment uses the grid axis closest to its direction as column axis, so
    the slab crosses every column over a short interval; the interval of a column
    is the intersection of the slab (two planes) with the cylinder of radius
    ``support`` around the segment axis (a quadratic). Columns keep the segment
    order.
    """
    lo_clip = np.asarray(lo_clip, dtype=np.int64)
    hi_clip = np.asarray(hi_clip, dtype=np.int64)
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    u = seg.u[ids]
    rows = np.arange(ids.size)
    axis_c = np.argmax(np.abs(u), axis=1)
    axis_1 = (axis_c + 1) % 3
    axis_2 = (axis_c + 2) % 3
    a = seg.a[ids]
    s = seg.support[ids]
    lo_i, hi_i = _segment_boxes(seg, ids, origin, h)
    lo_i = np.clip(lo_i, lo_clip, hi_clip + 1)
    hi_i = np.clip(hi_i, lo_clip - 1, hi_clip)
    n_ax = np.maximum(hi_i - lo_i + 1, 0)
    n1 = n_ax[rows, axis_1]
    n2 = n_ax[rows, axis_2]
    n_cols = n1 * n2
    # per-segment constants in the (column, 1, 2) axis frame
    lo1 = lo_i[rows, axis_1]
    lo2 = lo_i[rows, axis_2]
    w1_0 = origin[axis_1] + (lo1 + 0.5) * h - a[rows, axis_1]
    w2_0 = origin[axis_2] + (lo2 + 0.5) * h - a[rows, axis_2]
    a_rel = a[rows, axis_c] - origin[axis_c]  # segment start along the column axis

    def per_column(values):
        return np.repeat(values, n_cols)

    local = np.arange(int(n_cols.sum()), dtype=np.int64)
    local -= per_column(np.cumsum(n_cols) - n_cols)
    n2_c = per_column(n2)
    m1 = local // n2_c
    m2 = local - m1 * n2_c
    del local, n2_c
    w1 = per_column(w1_0) + m1 * h
    w2 = per_column(w2_0) + m2 * h
    u1 = per_column(u[rows, axis_1])
    u2 = per_column(u[rows, axis_2])
    p_dot = w1 * u1
    p_dot += w2 * u2
    q_perp2 = w1 * w1
    q_perp2 += w2 * w2
    uc = per_column(u[rows, axis_c])
    tol = 1e-6 * h
    # slab: -ext0 <= p_dot + z * uc <= length + ext1   (|uc| >= 1/sqrt(3))
    za = (per_column(-seg.ext0[ids] - tol) - p_dot) / uc
    zb = (per_column(seg.length[ids] + seg.ext1[ids] + tol) - p_dot) / uc
    z_lo = np.minimum(za, zb)
    z_hi = np.maximum(za, zb)
    del za, zb
    # cylinder: (q_perp2 + z^2) - (p_dot + z uc)^2 <= s^2, i.e. qa z^2 + qb z + qc <= 0
    qa = 1.0 - uc * uc
    qb = -2.0 * p_dot * uc
    qc = q_perp2 - p_dot * p_dot - per_column(s * s)
    flat_axis = qa <= 1e-12
    ok = np.where(flat_axis, qc <= 0.0, True)
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = qb * qb - 4.0 * qa * qc
        ok &= flat_axis | (disc >= 0.0)
        root = np.sqrt(np.maximum(disc, 0.0))
        np.maximum(z_lo, np.where(flat_axis, -np.inf, (-qb - root) / (2.0 * qa)), out=z_lo)
        np.minimum(z_hi, np.where(flat_axis, np.inf, (-qb + root) / (2.0 * qa)), out=z_hi)
    del qa, qb, qc, disc, root, flat_axis
    a_rel_c = per_column(a_rel)
    k_lo = per_column(lo_clip[axis_c])
    k_hi = per_column(hi_clip[axis_c])
    with np.errstate(invalid="ignore"):
        k0 = np.clip(np.ceil((a_rel_c + z_lo) / h - 0.5), k_lo, k_hi + 1).astype(np.int64)
        k1 = np.clip(np.floor((a_rel_c + z_hi) / h - 0.5), k_lo - 1, k_hi).astype(np.int64)
    count = np.where(ok & (z_lo <= z_hi), np.maximum(k1 - k0 + 1, 0), 0)
    del z_lo, z_hi, k1, k_lo, k_hi, ok
    keep = np.flatnonzero(count)
    col = np.repeat(rows, n_cols)[keep]
    stride = strides[axis_c][col]
    k0 = k0[keep]
    cell0 = (lo1 * strides[axis_1] + lo2 * strides[axis_2])[col]
    cell0 += m1[keep] * strides[axis_1][col] + m2[keep] * strides[axis_2][col] + k0 * stride
    z0 = (k0 + 0.5) * h - a_rel_c[keep]
    return _Columns(
        ids[col], cell0, stride, z0, count[keep], p_dot[keep], q_perp2[keep], uc[keep],
        axis_c[col].astype(np.int8), w1[keep], w2[keep], u1[keep], u2[keep],
    )


def _evaluate_points(seg: _Segments, cols: _Columns, sl: slice, h: float, with_dir: bool) -> tuple:
    """Kernel values at the grid points of columns ``sl``.

    Returns ``(cell, owner, f, value, direction)`` for the points with ``f > 0``;
    ``direction (K, 3)`` (or ``None`` unless ``with_dir``) is the unit vector from
    the point to the closest point of the segment's axis (the direction of the
    kernel's gradient; zero on the axis).
    """
    count = cols.count[sl]
    sid = cols.sid[sl]
    total = int(count.sum())

    def rep(values):
        return np.repeat(values, count)

    step = np.arange(total, dtype=np.int64)
    step -= rep(np.cumsum(count) - count)
    z = step * h  # offset from the segment start along the column axis
    z += rep(cols.z0[sl])
    proj = z * rep(cols.u_col[sl])
    proj += rep(cols.p_dot[sl])  # (p - a) . u
    t = proj * rep(seg.inv_length[sid])
    np.clip(t, 0.0, 1.0, out=t)
    tl = t * rep(seg.length[sid])
    # squared distance to the closest point a + t * length * u
    d2 = z * z
    d2 += rep(cols.q_perp2[sl])
    proj *= 2.0
    proj -= tl
    proj *= tl
    d2 -= proj
    del proj
    r = t * rep(seg.dr[sid])
    r += rep(seg.ra[sid])
    r *= r
    r *= 1.0 / (_T_ISO * _T_ISO)
    u2 = np.divide(d2, r, out=r)
    keep = np.flatnonzero(u2 < 1.0)
    f = 1.0 - np.maximum(u2[keep], 0.0)
    f *= f * f
    del u2, r
    item = np.searchsorted(np.cumsum(count), keep, side="right")  # column of every kept point
    sid_k = sid[item]
    value = t[keep]
    value *= seg.dv[sid_k]
    value += seg.va[sid_k]
    cell = step[keep]
    del step, t
    cell *= cols.stride[sl][item]
    cell += cols.cell0[sl][item]
    direction = None
    if with_dir:
        tk = tl[keep]
        # (p - closest) in the (column, 1, 2) frame, scaled to -unit
        dist = np.sqrt(np.maximum(d2[keep], 0.0))
        scale = np.zeros(dist.size)
        np.divide(-1.0, dist, out=scale, where=dist > 0.0)
        c0 = z[keep] - tk * cols.u_col[sl][item]
        c1 = cols.w1[sl][item] - tk * cols.u1[sl][item]
        c2 = cols.w2[sl][item] - tk * cols.u2[sl][item]
        c0 *= scale
        c1 *= scale
        c2 *= scale
        ax = cols.axis[sl][item]
        direction = np.empty((keep.size, 3))
        # world component a takes c0 if a == axis, c1 if a == axis + 1, c2 if a == axis + 2
        direction[:, 0] = np.where(ax == 0, c0, np.where(ax == 2, c1, c2))
        direction[:, 1] = np.where(ax == 1, c0, np.where(ax == 0, c1, c2))
        direction[:, 2] = np.where(ax == 2, c0, np.where(ax == 1, c1, c2))
    return cell, seg.owner[sid_k], f, value, direction


def _run_starts(keys: np.ndarray) -> np.ndarray:
    """Start positions of the runs of equal values of sorted ``keys``."""
    if keys.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))


def _last_max(starts: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Position of the last maximum of every run ``f[starts[k]:starts[k + 1]]``.

    Only runs longer than one element are scanned (usually a small minority).
    """
    counts = np.diff(np.append(starts, f.size))
    multi = np.flatnonzero(counts > 1)
    pick = starts.copy()
    if multi.size:
        m_counts = counts[multi]
        m_first = np.cumsum(m_counts) - m_counts
        pos = np.arange(int(m_counts.sum()), dtype=np.int64)
        pos += np.repeat(starts[multi] - m_first, m_counts)  # positions of the long runs
        fm = f[pos]
        best = np.maximum.reduceat(fm, m_first)
        pos[fm != np.repeat(best, m_counts)] = -1
        pick[multi] = np.maximum.reduceat(pos, m_first)
    return pick


def _tube_maxima(cell, owner, f, value, direction, open_owner: int) -> tuple:
    """One row per (cell, tube): the tube's own field (maximum over its segments) and the
    value / direction of the maximizing segment.

    Rows come sorted by cell, then tube - except the rows of tube ``open_owner``
    (still continuing, ``-1``: none), which come last. Returns the rows and the
    number of those trailing rows.
    """
    o0 = int(owner.min())
    span = int(owner.max()) - o0 + 1
    keys = cell * span
    keys += owner - o0
    held = None
    if open_owner >= 0:
        held = owner == open_owner
        keys += held * ((int(cell.max()) + 1) * span)
    order = np.argsort(keys)
    starts = _run_starts(keys[order])
    if starts.size < order.size:
        order = order[_last_max(starts, f[order])]
    n_held = 0 if held is None else int(np.count_nonzero(held[order]))
    rows = (
        cell[order],
        owner[order],
        f[order],
        value[order],
        None if direction is None else direction[order],
    )
    return rows, n_held


class _Accumulator:
    """Per-cell sums over tubes for one slab of cells (``n`` cells from flat id ``base``).

    ``tau == 0``: maximum and the value at the maximum. ``tau > 0``: maximum, soft
    sum ``sum(exp(f / tau) - 1)``, weighted value sum, and for the crease angle the
    sums of the angle weights ``w = exp(f / _TAU_ANGLE) - 1``, of ``w ** 2`` and of
    ``w * direction``. The arrays are reused from slab to slab (:meth:`finish`
    resets the touched cells): fresh pages are expensive to fault in.
    """

    def __init__(self, n: int, tau: float):
        self.base = 0
        self.tau = float(tau)
        self.fmax = np.zeros(n)
        self.val = np.zeros(n)
        if self.tau > 0.0:
            self.soft = np.zeros(n)
            self.wsum = np.zeros(n)
            self.wsq = np.zeros(n)
            self.wdir = np.zeros((n, 3))

    def start(self, base: int) -> None:
        """Start a slab whose first cell has flat id ``base`` (accumulators are all zero)."""
        self.base = int(base)

    def add(self, cell, f, value, direction) -> None:
        """Add rows sorted by cell, at most one per (cell, tube), each tube complete."""
        if cell.size == 0:
            return
        starts = _run_starts(cell)
        ucell = cell[starts] - self.base
        single = starts.size == cell.size  # every cell reached by one tube only

        def total(x):
            return x if single else np.add.reduceat(x, starts, axis=0)

        if self.tau == 0.0:
            top = starts if single else _last_max(starts, f)
            fmax = f[top]
            better = fmax > self.fmax[ucell]
            self.fmax[ucell[better]] = fmax[better]
            self.val[ucell[better]] = value[top[better]]
            return
        fm = f if single else np.maximum.reduceat(f, starts)
        self.fmax[ucell] = np.maximum(self.fmax[ucell], fm)
        w = np.expm1(f / self.tau)
        self.soft[ucell] += total(w)
        w *= value
        self.val[ucell] += total(w)
        wa = np.expm1(f / _TAU_ANGLE)
        wa[np.einsum("ij,ij->i", direction, direction) < 0.5] = 0.0  # on an axis: no direction
        self.wdir[ucell] += total(direction * wa[:, None])
        self.wsum[ucell] += total(wa)
        wa *= wa
        self.wsq[ucell] += total(wa)

    def finish(self, field_out: np.ndarray, value_out: np.ndarray) -> None:
        """Write the blended field and the value field of the slab into the (zero)
        outputs (only cells reached by a tube are written), then reset."""
        n = field_out.size
        if self.tau == 0.0:
            hit = np.flatnonzero(self.fmax[:n] > 0.0)
            field_out[hit] = self.fmax[hit]
            value_out[hit] = self.val[hit]
            self.fmax[hit] = 0.0
            self.val[hit] = 0.0
            return
        hit = np.flatnonzero(self.soft[:n] > 0.0)  # cells reached by some tube
        if hit.size == 0:
            return
        tau = self.tau
        F = self.fmax[hit]
        S = self.soft[hit]
        value_out[hit] = self.val[hit] / S
        # crease angle: weighted mean pairwise misalignment 1 - cos(theta_ij)
        W = self.wsum[hit]
        W *= W
        pairs = W - self.wsq[hit]
        d = self.wdir[hit]
        spread = W - np.einsum("ij,ij->i", d, d)
        del d
        for arr in (self.fmax, self.val, self.soft, self.wsum, self.wsq, self.wdir):
            arr[hit] = 0.0
        mis = np.zeros(hit.size)
        np.divide(spread, pairs, out=mis, where=pairs > _PAIR_TOL * W)
        del W, pairs, spread
        g = _crease_factor(mis)
        del mis
        # soft maximum G of the other tubes, kept below F + g (G - F) as g vanishes
        rest = (1.0 + S) - np.exp(F / tau)
        np.maximum(rest, 0.0, out=rest)
        G = np.log1p(rest)
        G *= tau
        del rest, S
        over = np.maximum(G - F, 0.0)
        over *= g
        over += F
        np.minimum(G, over, out=G)
        del over
        g *= tau
        field_out[hit] = _soft_max2(F, G, g)


def _crease_factor(mis: np.ndarray) -> np.ndarray:
    """Temperature factor ``sin(theta)`` (1 beyond 90 degrees) of a crease whose gradients
    are misaligned by ``mis = 1 - cos(theta)``; C1 at 90 degrees. Misalignments below
    ``_MIS_DEAD`` (rounding noise of identical directions) count as parallel."""
    m = np.clip(np.asarray(mis, dtype=np.float64) - _MIS_DEAD, 0.0, 2.0)
    return np.where(m >= 1.0, 1.0, np.sqrt(m * (2.0 - m)))


def _soft_max2(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """``t * ln(exp(a / t) + exp(b / t) - 1)`` (overflow free; the maximum where ``t <= 0``)."""
    a, b, t = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(t, dtype=np.float64)
    )
    shape = a.shape
    a, b, t = (x.reshape(-1) for x in (a, b, t))
    hi = np.maximum(a, b)
    lo = np.minimum(a, b)
    pos = t > 0.0
    safe = np.where(pos, t, 1.0)
    with np.errstate(under="ignore", over="ignore"):
        extra = np.exp((lo - hi) / safe)
        extra -= np.exp(-hi / safe)
    np.maximum(extra, 0.0, out=extra)
    np.log1p(extra, out=extra)
    extra *= t
    extra[~pos] = 0.0
    extra += hi
    return extra.reshape(shape)


def _aligned_end(end: int, start: int, first_of_run: np.ndarray, owner: np.ndarray) -> int:
    """Move a batch end back to the start of the tube it would split, if that keeps the
    batch non-empty (only tubes larger than a whole batch are split)."""
    if end < owner.size and owner[end] == owner[end - 1]:
        first = int(first_of_run[end - 1])
        if first > start:
            return first
    return end


def _run_firsts(owner: np.ndarray) -> np.ndarray:
    """For every position of a sequence of runs, the position where its run starts."""
    starts = _run_starts(owner)
    return np.repeat(starts, np.diff(np.append(starts, owner.size)))


def _segment_batches(seg: _Segments, ids: np.ndarray, h: float, max_extent: int) -> list:
    """Split ``ids`` into consecutive ranges with at most ~``_COLUMN_BATCH`` columns each,
    cut between tubes whenever possible.

    The column count of a segment is bounded by its cylinder's extent; a range
    always holds at least one segment. Returns boundaries into ``ids``.
    """
    extent = 2.0 * seg.support[ids] + seg.length[ids] + seg.ext0[ids] + seg.ext1[ids]
    cols = (np.minimum(extent / h + 2.0, max_extent)) ** 2
    cum = np.cumsum(cols)
    owner = seg.owner[ids]
    firsts = _run_firsts(owner)
    bounds = [0]
    while bounds[-1] < ids.size:
        start = bounds[-1]
        base = cum[start - 1] if start else 0.0
        end = int(np.searchsorted(cum, base + _COLUMN_BATCH, side="right"))
        end = min(max(end, start + 1), ids.size)
        bounds.append(_aligned_end(end, start, firsts, owner))
    return bounds


def _field_on_grid(seg, shape, origin, h, tau, progress, should_stop) -> tuple:
    n_cells = int(np.prod(shape))
    field = np.zeros(n_cells, dtype=np.float64)
    value = np.zeros(n_cells, dtype=np.float64)
    if seg is None:
        return field.reshape(shape), value.reshape(shape)
    layer = shape[1] * shape[2]
    thick = max(1, _SLAB_CELLS // layer)
    hi_grid = np.asarray(shape, dtype=np.int64) - 1
    box_lo, box_hi = _segment_boxes(seg, np.arange(seg.n), origin, h)
    in_grid = np.all((box_hi >= 0) & (box_lo <= hi_grid), axis=1)
    slabs = list(range(0, shape[0], thick))
    with_dir = tau > 0.0
    acc = _Accumulator(min(thick, shape[0]) * layer, tau)
    for s_index, x0 in enumerate(slabs):
        x1 = min(x0 + thick, shape[0])
        check_stop(should_stop)
        ids = np.flatnonzero(in_grid & (box_lo[:, 0] < x1) & (box_hi[:, 0] >= x0))
        acc.start(x0 * layer)
        if ids.size:
            lo_clip = np.array([x0, 0, 0], dtype=np.int64)
            hi_clip = np.array([x1 - 1, shape[1] - 1, shape[2] - 1], dtype=np.int64)
            bounds = _segment_batches(seg, ids, h, max(shape))
            carry = None  # rows of a tube that continues into the next batch
            for b0, b1 in zip(bounds[:-1], bounds[1:]):
                check_stop(should_stop)
                cols = _segment_columns(seg, ids[b0:b1], shape, origin, h, lo_clip, hi_clip)
                n_cols = cols.count.size
                cum = np.cumsum(cols.count)
                owners = seg.owner[cols.sid]
                firsts = _run_firsts(owners)
                next_owner = int(seg.owner[ids[b1]]) if b1 < ids.size else -1
                start = 0
                while start < n_cols:
                    check_stop(should_stop)
                    base = int(cum[start - 1]) if start else 0
                    end = int(np.searchsorted(cum, base + _BATCH, side="right"))
                    end = min(max(end, start + 1), n_cols)
                    end = _aligned_end(end, start, firsts, owners)
                    rows = _evaluate_points(seg, cols, slice(start, end), h, with_dir)
                    if carry is not None:
                        rows = tuple(
                            None if new is None else np.concatenate([old, new])
                            for old, new in zip(carry, rows)
                        )
                        carry = None
                    last = int(owners[end - 1])
                    follows = int(owners[end]) if end < n_cols else next_owner
                    if rows[0].size:
                        rows, n_held = _tube_maxima(*rows, last if follows == last else -1)
                        if n_held:  # the last tube continues: hold its rows back
                            cut = rows[0].size - n_held
                            carry = tuple(None if r is None else r[cut:] for r in rows)
                            rows = tuple(None if r is None else r[:cut] for r in rows)
                        acc.add(rows[0], rows[2], rows[3], rows[4])
                    start = end
                if progress is not None:
                    progress((s_index + b1 / ids.size) / len(slabs), "Fusing tubes")
            if carry is not None:  # cannot happen: the last tube closes the stream
                acc.add(carry[0], carry[2], carry[3], carry[4])
        sl = slice(x0 * layer, x1 * layer)
        acc.finish(field[sl], value[sl])
    return field.reshape(shape), value.reshape(shape)


def _check_grid(shape, h) -> tuple:
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("grid shape must be three positive integers")
    if not float(h) > 0.0:
        raise ValueError("voxel size must be positive")
    cells = int(np.prod(shape))
    if cells > MAX_CELLS:
        raise ValueError(
            f"The tube field would need {cells:,} voxels (limit {MAX_CELLS:,}): use a larger voxel size"
        )
    return shape


# ============================================================================ public API
def evaluate_tubes(
    tubes,
    shape,
    origin,
    h,
    blend=0.5,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
) -> tuple:
    """Tube field on a given element-centred grid ``(shape, origin, h)``.

    Same field as :func:`tubes_field` (see the module docstring), but on the caller's
    grid (tubes may extend beyond it). Returns ``(field, value_field)``, both
    ``shape`` float64.
    """
    shape = _check_grid(shape, h)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    tubes = [_as_tube(t) for t in tubes]
    return _field_on_grid(
        _segments(tubes, _MAX_SEGMENT_VOXELS * float(h)), shape, origin, float(h), blend_tau(blend),
        progress, should_stop,
    )


def tubes_field(
    tubes,
    voxel_size,
    blend=0.5,
    pad=2,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
) -> tuple:
    """Implicit field of the smooth union of ``tubes`` (mesh it at :data:`TUBE_ISO`).

    The grid (spacing ``voxel_size``, element-centred) covers the tubes' supports
    (radius x :data:`TUBE_SUPPORT`) plus ``pad`` voxels per side. ``blend`` in
    [0, 1]: 0 = hard union, 1 = softest fillets (see the module docstring).
    ``progress(fraction, message)`` / ``should_stop()`` are polled per batch (~0.1 s).

    Returns ``(field, origin, h, value_field)``: ``field (nx, ny, nz)`` >= 0, the grid
    origin ``(3,)``, ``h = voxel_size`` and the blended tube values ``(nx, ny, nz)``
    (0 where no tube reaches). Without any valid tube: a ``(1, 1, 1)`` zero field.
    Raises ``ValueError`` if the grid would exceed :data:`MAX_CELLS` cells.
    """
    h = float(voxel_size)
    if not h > 0.0:
        raise ValueError("voxel_size must be positive")
    pad = max(0, int(pad))
    seg = _segments([_as_tube(t) for t in tubes], _MAX_SEGMENT_VOXELS * h)
    if seg is None:
        return np.zeros((1, 1, 1)), np.zeros(3), h, np.zeros((1, 1, 1))
    end = seg.a + seg.length[:, None] * seg.u
    bmin = (np.minimum(seg.a, end) - seg.support[:, None]).min(axis=0)
    bmax = (np.maximum(seg.a, end) + seg.support[:, None]).max(axis=0)
    counts = np.maximum(1, np.ceil((bmax - bmin) / h - 1e-9).astype(np.int64)) + 2 * pad
    origin = 0.5 * (bmin + bmax) - 0.5 * counts * h
    shape = _check_grid(counts, h)
    field, value_field = _field_on_grid(
        seg, shape, origin, h, blend_tau(blend), progress, should_stop
    )
    return field, origin, h, value_field


def _resample_axis(a: np.ndarray, axis: int, src_o: float, src_h: float, n_dst: int,
                   dst_o: float, dst_h: float) -> np.ndarray:
    """Linear resampling of element-centred samples along one axis (clamped ends)."""
    n_src = a.shape[axis]
    u = (dst_o + (np.arange(n_dst) + 0.5) * dst_h - src_o) / src_h - 0.5
    u = np.clip(u, 0.0, n_src - 1.0)
    i0 = np.minimum(np.floor(u).astype(np.int64), max(n_src - 2, 0))
    i1 = np.minimum(i0 + 1, n_src - 1)
    w_shape = [1] * a.ndim
    w_shape[axis] = n_dst
    w1 = (u - i0).reshape(w_shape)
    w0 = 1.0 - w1
    out_shape = list(a.shape)
    out_shape[axis] = n_dst
    out = np.empty(out_shape, dtype=np.float64)
    other = 1 if axis == 0 else 0
    per_row = max(1, int(np.prod(out_shape)) // max(out_shape[other], 1))
    step = max(1, _RESAMPLE_BLOCK // per_row)
    for s0 in range(0, out_shape[other], step):
        sl = [slice(None)] * a.ndim
        sl[other] = slice(s0, s0 + step)
        sl = tuple(sl)
        block = a[sl]
        lo = np.take(block, i0, axis=axis)
        lo *= w0
        hi = np.take(block, i1, axis=axis)
        hi *= w1
        lo += hi
        out[sl] = lo
    return out


def resample_to(field, src_origin, src_h, dst_shape, dst_origin, dst_h) -> np.ndarray:
    """Trilinear resampling of an element-centred field onto another grid.

    Source samples at ``src_origin + (ijk + 0.5) * src_h``, destination centres at
    ``dst_origin + (ijk + 0.5) * dst_h``; points beyond the source grid take the
    nearest border value (clamped, like ``mesher.sample_trilinear``). Separable
    (exactly trilinear), done axis by axis in blocks; trailing channel axes are
    kept. Returns ``dst_shape (+ channels)`` float64.
    """
    out = np.asarray(field, dtype=np.float64)
    if out.ndim < 3:
        raise ValueError("field must have shape (nx, ny, nz) or (nx, ny, nz, C...)")
    src_origin = np.asarray(src_origin, dtype=np.float64).reshape(3)
    dst_origin = np.asarray(dst_origin, dtype=np.float64).reshape(3)
    dst_shape = tuple(int(n) for n in dst_shape)
    if len(dst_shape) != 3 or min(dst_shape) < 1:
        raise ValueError("dst_shape must be three positive integers")
    src_h = float(src_h)
    dst_h = float(dst_h)
    if not (src_h > 0.0 and dst_h > 0.0):
        raise ValueError("grid spacings must be positive")
    # Shrinking axes first keeps the intermediate arrays small.
    order = sorted(range(3), key=lambda a: dst_shape[a] / out.shape[a])
    for axis in order:
        out = _resample_axis(
            out, axis, src_origin[axis], src_h, dst_shape[axis], dst_origin[axis], dst_h
        )
    return out


def _unit_gradient(block: np.ndarray, axes: tuple) -> np.ndarray:
    """Unit central-difference gradient ``(len(axes), ...)`` of ``block`` (0 where flat)."""
    grads = np.gradient(block, axis=axes)
    if len(axes) == 1:
        grads = [grads]
    g = np.stack(grads)
    norm = np.sqrt(np.einsum("i...,i...->...", g, g))
    np.divide(g, norm, out=g, where=norm > 0.0)
    g[:, norm <= 0.0] = 0.0
    return g


def union_fields(a, b, blend) -> np.ndarray:
    """Smooth union of two non-negative fields with the rule of :func:`tubes_field`.

    ``U = tau' ln(exp(a / tau') + exp(b / tau') - 1)`` with ``tau' = TAU_MAX * blend *
    sin(theta)``, ``theta`` = angle between the central-difference gradients of ``a``
    and ``b`` (capped at 90 degrees): the maximum for ``blend = 0`` and wherever a
    gradient vanishes (no direction, like a point on a tube's axis; also for inputs
    without any axis of two or more samples). So ``max(a, b) <= U <= soft-max``,
    ``U(a, 0) = a`` exactly, ``U(a, b) = U(b, a)``, non-decreasing in ``blend``;
    fillets where the surfaces cross, no swelling where they overlap with parallel
    gradients. Both fields must be sampled on the same regular grid (same spacing
    on every axis); negative inputs count as 0. Arrays broadcast; returns float64.
    """
    tau = blend_tau(blend)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=np.float64)
    if out.size == 0:
        return out
    axes = tuple(ax for ax in range(out.ndim) if out.shape[ax] >= 2)
    if tau == 0.0 or not axes:
        out[...] = np.maximum(np.maximum(a, b), 0.0)
        return out
    n0 = out.shape[0]
    per_row = max(1, out.size // n0)
    step = max(1, _UNION_BLOCK // per_row)
    for s0 in range(0, n0, step):
        s1 = min(s0 + step, n0)
        h0 = max(s0 - 1, 0)  # one halo row on each side for the central differences
        h1 = min(s1 + 1, n0)
        x = np.maximum(a[h0:h1], 0.0)
        y = np.maximum(b[h0:h1], 0.0)
        inner = slice(s0 - h0, s0 - h0 + (s1 - s0))
        nx = _unit_gradient(x, axes)[(slice(None), inner)]
        ny = _unit_gradient(y, axes)[(slice(None), inner)]
        defined = (np.einsum("i...,i...->...", nx, nx) > 0.5) & (np.einsum("i...,i...->...", ny, ny) > 0.5)
        mis = np.where(defined, 1.0 - np.einsum("i...,i...->...", nx, ny), 0.0)
        del nx, ny, defined
        g = _crease_factor(mis)
        g *= tau
        out[s0:s1] = _soft_max2(x[inner], y[inner], g)
    return out
