"""Implicit tube field of the sculpture: stress lines fused into one organic solid.

Every :class:`Tube` (a polyline with a radius and a carried value per point) adds
a smooth implicit field on a regular, element-centred grid (``origin + (ijk + 0.5)
* h``, like every field of the add-on); meshing the result at :data:`TUBE_ISO`
(``mesher.surface_nets``) gives one watertight surface.

Field of one tube
    For a grid point ``p`` and a polyline segment ``[a, b]`` with end radii
    ``r_a, r_b``::

        t = clamp(dot(p - a, b - a) / |b - a|^2, 0, 1)      # closest point parameter
        d = |p - (a + t (b - a))|                             # exact capsule distance
        r = r_a + t (r_b - r_a)                               # radius, linear along the segment
        f = phi(T * d / r),   phi(u) = (1 - u^2)^3 for u < 1, else 0   (Wyvill kernel)

    with ``T = sqrt(1 - TUBE_ISO ** (1/3))`` so that ``f == TUBE_ISO`` exactly at
    ``d == r``. A tube takes the maximum over its own segments (exact union: no
    bulges at the joints of the polyline). The kernel reaches zero at
    ``TUBE_SUPPORT * r`` (~2.2 r); only grid points inside each segment's bounding
    box grown by that support are evaluated. ``TUBE_ISO = 0.5`` puts the surface
    next to the inflection point of ``phi`` (``u^2 = 1/5``), where the field is
    almost linear across the surface, so the linear edge interpolation of the
    mesher is very accurate.

Smooth union (``blend`` in [0, 1])
    ::

        U(f_1, ..., f_n) = tau * ln(1 + sum_i (exp(f_i / tau) - 1)),   tau = TAU_MAX * blend

    a soft maximum with 0 as neutral element:

    * ``blend = 0`` gives the plain maximum (hard union, sharp creases),
    * a tube alone keeps its exact field for every blend: a lone tube's surface is
      always at its radius, whatever the blend,
    * ``max_i f_i <= U <= max_i f_i + tau ln(n)``: the extra material is bounded
      (unlike a metaball sum): at ``blend = 1`` two coincident tubes grow ~18% in
      radius (a plain sum: +34%) and two parallel tubes do not merge into a slab,
      while the creases where members cross get round fillets (~0.6 r),
    * associative and commutative (``U(a, b, c) = U(U(a, b), c)``) and increasing in
      ``blend`` point by point, so tubes are accumulated in any order and chunking,
      and :func:`union_fields` combines whole fields with exactly the same rule.

    ``value_field`` is the average of the tubes' values weighted by the same
    soft-max weights ``exp(f_i / tau) - 1`` (the value of the dominant tube when
    ``blend = 0``): colors follow the members and blend only in the fillets.

Memory is bounded: the work is split into batches of at most ``_BATCH`` grid-point
evaluations (~60 MB of temporaries); besides the two output arrays nothing scales
with the grid. Pure numpy, no global mutable state.
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
_BATCH = 1 << 19  # grid-point evaluations per batch (bounds the temporaries)
_COLUMN_BATCH = 1 << 18  # grid columns per batch of segments
_RESAMPLE_BLOCK = 1 << 22  # output elements per block in resample_to / union_fields
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


def _polyline_segments(p, r, v) -> Optional[tuple]:
    """Segments of one tube: ``(a, u, length, ra, rb, va, vb, ext0, ext1)``.

    Invalid points (non-finite, radius <= 0) split the polyline; repeated points are
    merged; a valid point without a valid neighbour becomes a ball.
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
    support = rq * TUBE_SUPPORT
    m = keep.size
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


def _segments(tubes) -> Optional[_Segments]:
    """Segments of all tubes in tube order; ``None`` if there are none."""
    parts, owners = [], []
    for index, tube in enumerate(tubes):
        tube = _as_tube(tube)
        part = _polyline_segments(tube.points, tube.radius, tube.value)
        if part is not None and part[0].shape[0]:
            parts.append(part)
            owners.append(np.full(part[0].shape[0], index, dtype=np.int64))
    if not parts:
        return None
    cols = [np.concatenate([part[c] for part in parts]) for c in range(9)]
    return _Segments(*cols, np.concatenate(owners))


class _Columns:
    """Grid columns (lines of grid points along one axis) crossing segment slabs.

    For column ``c`` of segment ``sid[c]``: grid points ``cell0[c] + k * stride[c]``
    for ``k = 0 .. count[c] - 1``, at offsets ``z0[c] + k * h`` along the column
    axis from the segment start; ``p_dot[c]`` / ``q_perp2[c]`` are the projection on
    the segment direction and the squared length of the column's offset in the two
    other axes, ``u_col[c]`` the direction component along the column.
    """

    def __init__(self, sid, cell0, stride, z0, count, p_dot, q_perp2, u_col):
        self.sid = sid
        self.cell0 = cell0
        self.stride = stride
        self.z0 = z0
        self.count = count
        self.p_dot = p_dot
        self.q_perp2 = q_perp2
        self.u_col = u_col


def _segment_columns(seg: _Segments, ids, shape, origin, h) -> _Columns:
    """Columns of the extended slabs of segments ``ids`` (empty columns dropped).

    Each segment uses the grid axis closest to its direction as column axis, so
    the slab crosses every column over a short interval; the interval of a column
    is the intersection of the slab (two planes) with the cylinder of radius
    ``support`` around the segment axis (a quadratic). Columns keep the segment
    order.
    """
    shape_a = np.asarray(shape, dtype=np.int64)
    strides = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    u = seg.u[ids]
    rows = np.arange(ids.size)
    axis_c = np.argmax(np.abs(u), axis=1)
    axis_1 = (axis_c + 1) % 3
    axis_2 = (axis_c + 2) % 3
    a = seg.a[ids]
    s = seg.support[ids]
    start = a - seg.ext0[ids, None] * u
    end = a + (seg.length[ids] + seg.ext1[ids])[:, None] * u
    half = s[:, None] * np.sqrt(np.maximum(1.0 - u * u, 0.0))
    lo_i = np.ceil((np.minimum(start, end) - half - origin) / h - 0.5)
    hi_i = np.floor((np.maximum(start, end) + half - origin) / h - 0.5)
    lo_i = np.clip(lo_i, 0, shape_a).astype(np.int64)
    hi_i = np.clip(hi_i, -1, shape_a - 1).astype(np.int64)
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
    p_dot = w1 * per_column(u[rows, axis_1])
    p_dot += w2 * per_column(u[rows, axis_2])
    q_perp2 = w1 * w1
    q_perp2 += w2 * w2
    del w1, w2
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
    n_c = per_column(shape_a[axis_c])
    k0 = np.clip(np.ceil((a_rel_c + z_lo) / h - 0.5), 0, n_c).astype(np.int64)
    k1 = np.clip(np.floor((a_rel_c + z_hi) / h - 0.5), -1, n_c - 1).astype(np.int64)
    count = np.where(ok & (z_lo <= z_hi), np.maximum(k1 - k0 + 1, 0), 0)
    del z_lo, z_hi, k1, n_c, ok
    keep = np.flatnonzero(count)
    col = np.repeat(rows, n_cols)[keep]
    stride = strides[axis_c][col]
    k0 = k0[keep]
    cell0 = (lo1 * strides[axis_1] + lo2 * strides[axis_2])[col]
    cell0 += m1[keep] * strides[axis_1][col] + m2[keep] * strides[axis_2][col] + k0 * stride
    z0 = (k0 + 0.5) * h - a_rel_c[keep]
    return _Columns(ids[col], cell0, stride, z0, count[keep], p_dot[keep], q_perp2[keep], uc[keep])


def _evaluate_points(seg: _Segments, cols: _Columns, sl: slice, h: float) -> tuple:
    """Kernel values at the grid points of columns ``sl``.

    Returns ``(cell, owner, f, value)`` for the points with ``f > 0``.
    """
    count = cols.count[sl]
    sid = cols.sid[sl]
    total = int(count.sum())
    step = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(count) - count, count)
    cell = np.repeat(cols.cell0[sl], count) + step * np.repeat(cols.stride[sl], count)
    z = np.repeat(cols.z0[sl], count) + step * h
    del step
    proj = np.repeat(cols.u_col[sl], count) * z
    proj += np.repeat(cols.p_dot[sl], count)
    t = proj * np.repeat(seg.inv_length[sid], count)
    np.clip(t, 0.0, 1.0, out=t)
    tl = t * np.repeat(seg.length[sid], count)
    # squared distance to the closest point a + t * length * u
    d2 = z * z
    d2 += np.repeat(cols.q_perp2[sl], count)
    proj *= 2.0
    proj -= tl
    proj *= tl
    d2 -= proj
    del z, proj, tl
    r = t * np.repeat(seg.dr[sid], count)
    r += np.repeat(seg.ra[sid], count)
    u2 = d2 / (r * r)
    u2 *= _T_ISO * _T_ISO
    del d2, r
    keep = np.flatnonzero(u2 < 1.0)
    f = 1.0 - np.maximum(u2[keep], 0.0)
    f = f * f * f
    owner_col = seg.owner[sid]
    item = np.repeat(np.arange(sid.size), count)[keep]
    value = seg.va[sid[item]] + t[keep] * seg.dv[sid[item]]
    return cell[keep], owner_col[item], f, value


def _group_max(keys: np.ndarray, f: np.ndarray, v: np.ndarray) -> tuple:
    """Per run of equal sorted ``keys``: max of ``f`` and the ``v`` at that max.

    (Ties average their values.) Returns ``(starts, fmax, vmax)``.
    """
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    if starts.size == keys.size:
        return starts, f, v
    fmax = np.maximum.reduceat(f, starts)
    at_max = f == np.repeat(fmax, np.diff(np.append(starts, keys.size)))
    v_sum = np.add.reduceat(np.where(at_max, v, 0.0), starts)
    n_max = np.add.reduceat(at_max.astype(np.float64), starts)
    return starts, fmax, v_sum / n_max


def _combine(cell, owner, f, v, tau, acc, val, open_owner) -> Optional[tuple]:
    """Fold kernel values into the accumulators.

    First the maximum per (cell, tube) (the tube's own field), then the soft-max sum
    (``tau > 0``: ``acc += exp(f / tau) - 1``, ``val += weight * value``) or the hard
    maximum (``acc = max``, ``val = value of the max``) over tubes. Rows of tube
    ``open_owner`` (continued in the next batch) are returned as the carry instead.
    """
    if cell.size == 0:
        return None
    o0 = int(owner.min())
    span = int(owner.max()) - o0 + 1
    keys = cell * span + (owner - o0)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts, f, v = _group_max(keys, f[order], v[order])
    keys = keys[starts]
    owner = keys % span + o0
    cell = keys // span
    carry = None
    if open_owner >= 0:
        held = owner == open_owner
        if held.any():
            carry = (cell[held], owner[held], f[held], v[held])
            rest = ~held
            cell, f, v = cell[rest], f[rest], v[rest]
    if cell.size:
        if tau > 0.0:
            cstarts = np.flatnonzero(np.concatenate(([True], cell[1:] != cell[:-1])))
            ucell = cell[cstarts]
            w = np.expm1(f / tau)
            acc[ucell] += np.add.reduceat(w, cstarts)
            val[ucell] += np.add.reduceat(w * v, cstarts)
        else:
            cstarts, fmax, vmax = _group_max(cell, f, v)
            ucell = cell[cstarts]
            better = fmax > acc[ucell]
            acc[ucell[better]] = fmax[better]
            val[ucell[better]] = vmax[better]
    return carry


def _segment_batches(seg: _Segments, shape, h) -> list:
    """Consecutive segment ranges with at most ~``_COLUMN_BATCH`` columns each.

    The column count of a segment is bounded by its cylinder's extent; a range
    always holds at least one segment.
    """
    extent = 2.0 * seg.support + seg.length + seg.ext0 + seg.ext1
    cols = (np.minimum(extent / h + 2.0, max(shape))) ** 2
    cum = np.cumsum(cols)
    bounds = [0]
    while bounds[-1] < seg.n:
        base = cum[bounds[-1] - 1] if bounds[-1] else 0.0
        end = int(np.searchsorted(cum, base + _COLUMN_BATCH, side="right"))
        bounds.append(min(max(end, bounds[-1] + 1), seg.n))
    return bounds


def _field_on_grid(seg, shape, origin, h, tau, progress, should_stop) -> tuple:
    n_cells = int(np.prod(shape))
    acc = np.zeros(n_cells, dtype=np.float64)
    val = np.zeros(n_cells, dtype=np.float64)
    if seg is not None:
        bounds = _segment_batches(seg, shape, h)
        carry = None
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            check_stop(should_stop)
            cols = _segment_columns(seg, np.arange(b0, b1), shape, origin, h)
            n_cols = cols.count.size
            cum = np.cumsum(cols.count)
            owners = seg.owner[cols.sid]
            next_owner = int(seg.owner[b1]) if b1 < seg.n else -1
            start = 0
            while start < n_cols:
                check_stop(should_stop)
                base = int(cum[start - 1]) if start else 0
                end = int(np.searchsorted(cum, base + _BATCH, side="right"))
                end = min(max(end, start + 1), n_cols)
                cell, owner, f, v = _evaluate_points(seg, cols, slice(start, end), h)
                if carry is not None:
                    cell, owner, f, v = (
                        np.concatenate(pair) for pair in zip(carry, (cell, owner, f, v))
                    )
                last = int(owners[end - 1])
                follows = int(owners[end]) if end < n_cols else next_owner
                carry = _combine(cell, owner, f, v, tau, acc, val, last if follows == last else -1)
                start = end
            if progress is not None:
                progress(b1 / seg.n, "Fusing tubes")
        if carry is not None:  # cannot happen: the last tube closes the stream
            _combine(*carry, tau, acc, val, -1)
    if tau > 0.0:
        np.divide(val, acc, out=val, where=acc > 0.0)
        np.log1p(acc, out=acc)
        acc *= tau
    return acc.reshape(shape), val.reshape(shape)


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
def evaluate_tubes(tubes, shape, origin, h, blend=0.5, progress=None, should_stop=None) -> tuple:
    """Tube field on a given element-centred grid ``(shape, origin, h)``.

    Same field as :func:`tubes_field` (see the module docstring), but on the caller's
    grid. Returns ``(field, value_field)``, both ``shape`` float64.
    """
    shape = _check_grid(shape, h)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    tubes = [_as_tube(t) for t in tubes]
    return _field_on_grid(
        _segments(tubes), shape, origin, float(h), blend_tau(blend), progress, should_stop
    )


def tubes_field(tubes, voxel_size, blend=0.5, pad=2, progress=None, should_stop=None) -> tuple:
    """Implicit field of the smooth union of ``tubes`` (mesh it at :data:`TUBE_ISO`).

    The grid (spacing ``voxel_size``, element-centred) covers the tubes' supports
    (radius x :data:`TUBE_SUPPORT`) plus ``pad`` voxels per side. ``blend`` in
    [0, 1]: 0 = hard union, 1 = softest fillets (see the module docstring).
    ``progress(fraction, message)`` / ``should_stop()`` are polled per batch (~0.05 s).

    Returns ``(field, origin, h, value_field)``: ``field (nx, ny, nz)`` >= 0, the grid
    origin ``(3,)``, ``h = voxel_size`` and the blended tube values ``(nx, ny, nz)``
    (0 where no tube reaches). Without any valid tube: a ``(1, 1, 1)`` zero field.
    """
    h = float(voxel_size)
    if not h > 0.0:
        raise ValueError("voxel_size must be positive")
    pad = max(0, int(pad))
    seg = _segments([_as_tube(t) for t in tubes])
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
    # Shrinking axes first keeps the intermediate arrays small.
    order = sorted(range(3), key=lambda a: dst_shape[a] / out.shape[a])
    for axis in order:
        out = _resample_axis(
            out, axis, src_origin[axis], src_h, dst_shape[axis], dst_origin[axis], dst_h
        )
    return out


def union_fields(a, b, blend) -> np.ndarray:
    """Smooth union of two non-negative fields with the rule of :func:`tubes_field`.

    ``U(a, b) = tau * ln(exp(a / tau) + exp(b / tau) - 1)`` (``tau = TAU_MAX * blend``,
    the maximum for ``blend = 0``), evaluated in the overflow-free form
    ``m + tau * log1p(exp((n - m) / tau) - exp(-m / tau))`` with ``m = max``,
    ``n = min``. ``U(a, 0) = a`` exactly. Negative inputs are treated as 0. Arrays
    broadcast; returns float64.
    """
    tau = blend_tau(blend)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=np.float64)
    if out.size == 0:
        return out
    if not (a.flags.c_contiguous and b.flags.c_contiguous):
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
    fa, fb, fo = a.reshape(-1), b.reshape(-1), out.reshape(-1)
    for s0 in range(0, fo.size, _RESAMPLE_BLOCK):
        sl = slice(s0, s0 + _RESAMPLE_BLOCK)
        x = np.maximum(fa[sl], 0.0)
        y = np.maximum(fb[sl], 0.0)
        hi = np.maximum(x, y)
        if tau == 0.0:
            fo[sl] = hi
            continue
        lo = np.minimum(x, y)
        extra = np.exp((lo - hi) / tau)
        extra -= np.exp(-hi / tau)
        np.maximum(extra, 0.0, out=extra)
        np.log1p(extra, out=extra)
        extra *= tau
        extra += hi
        fo[sl] = extra
    return out
