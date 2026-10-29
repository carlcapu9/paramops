"""Principal stress trajectories ("tension lines") traced through a voxel stress field.

The stress of an analysis (Voigt values at element centres, see ``core/grid.py``)
has at every point three orthogonal principal directions. A *trajectory* is a curve
that is everywhere tangent to one of them:

* ``"TENSION"`` lines follow the eigenvector of the largest principal stress,
* ``"COMPRESSION"`` lines follow the eigenvector of the smallest principal stress.

How :func:`trace` works:

* The stress TENSOR (not its eigenvectors) is interpolated trilinearly between the
  element centres, using only solid elements (weights renormalized), and is
  eigen-decomposed in closed form. Eigenvectors have no orientation: every new
  direction is flipped to agree with the previous one.
* Lines are integrated with RK4 (default) or RK2 (midpoint) at a fixed step, in both
  directions from the seed; the two halves are joined into one polyline.
* A half stops when it leaves the solid or the grid, when ``|sigma_family|`` drops
  below ``min_stress`` times a robust reference (99th percentile of the family in the
  solid), after ``max_steps``, or when it comes closer than ``separation`` to an
  accepted line of the same family. A line whose two halves meet head-on is a closed
  loop (its last point repeats the first one); a line that curls back onto itself is
  cut where it comes within ``separation`` of its own earlier part.
* Where the direction is undefined (isotropic / degenerate points: the eigen-gap of
  the family is ~0, or the direction would turn by more than 60 degrees in one step)
  a line keeps its previous direction for at most ~2 voxels, then stops.
* Seeds: ``seed_points`` first, then ``n_seeds`` stress-weighted random samples per
  family. Seeds closer than ``separation`` to an accepted line of their family are
  skipped, so no line ever comes closer than ``separation`` to another one of its
  family (the seed vicinity needs no exemption).
* Branching: lines of the other family sprout from accepted lines where the crossing
  principal stress is significant, recursively up to ``branch_depth``.

The result is the one of the classic sequential evenly-spaced streamline algorithm
(seed after seed, every line stopping at the lines accepted before it), but lines are
integrated in batches: all candidates of a round advance simultaneously (vectorized
interpolation and eigen-analysis), then they are accepted one by one, each one clipped
where it comes closer than ``separation`` to a line accepted before it.

Internally positions are in voxel units ``u = (p - grid.origin) / grid.h`` (element
``(i, j, k)`` spans ``[i, i + 1]`` along each axis, its centre is at ``i + 0.5``); all
lengths in :class:`TraceSettings` are in voxels. Results are in world coordinates.

Pure numpy, no global mutable state: independent calls may run in parallel threads.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .grid import Grid, ProgressFn, StopFn, check_stop


FAMILIES = ("TENSION", "COMPRESSION")
INTEGRATORS = ("RK4", "RK2")

# Column (in ascending eigenvalue order) followed by each family, indexed like FAMILIES.
_FAMILY_COLUMN = np.array([2, 0], dtype=np.int64)
_FAMILY_COLUMN.setflags(write=False)

# Voigt [xx, yy, zz, xy, yz, zx] -> symmetric 3x3 tensor (fancy index of the 6 values).
_VOIGT_TO_TENSOR = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]], dtype=np.int64)
_VOIGT_TO_TENSOR.setflags(write=False)

_TWO_THIRDS_PI = 2.0 * math.pi / 3.0

_REF_PERCENTILE = 99.0  # robust "peak" of a family's stress magnitude in the solid
_REF_FLOOR = 0.05  # a family's reference is at least this fraction of the larger one
_GAP_TOL = 1e-2  # eigen-gap (relative to max |principal|) below which a direction is undefined
_COS_MAX_TURN = 0.5  # cos(60 deg): sharper turns within one step count as degenerate
_DEGENERATE_LENGTH = 2.0  # voxels travelled straight through a degenerate region at most
_CLOSE_ALIGN = 0.7  # a closing head must see the other head ahead (within ~45 degrees)
_SNAP_RADIUS = 2.0  # explicit seeds outside the solid move to a solid centre this close
_BATCH_SEEDS = 128  # candidate lines integrated together in one round
_SCAN_FACTOR = 4  # seeds tested for admissibility per round, in units of the batch size
_STAMP_CELL = 0.8  # visit-stamp cell size, in units of the separation
_MAX_STAMP_CELLS = 1 << 22
_EIG_CHUNK = 1 << 18  # elements per chunk when analysing the whole field


# ----------------------------------------------------------------- data classes
@dataclass(eq=False)
class StressField:
    """Stress tensors at element centres, interpolated inside the solid elements.

    ``stress`` holds Voigt stresses ``[xx, yy, zz, xy, yz, zx]`` (``(nx, ny, nz, 6)``),
    ``solid`` the elements lines may travel through: the analysis domain, or the
    ``density > 0.5`` part of an evolved shape. Non-finite stresses are replaced by
    zeros and their elements removed from ``solid``. Both arrays are treated as
    read-only once the field is built.
    """

    grid: Grid
    stress: np.ndarray  # (nx, ny, nz, 6) Voigt
    solid: np.ndarray  # (nx, ny, nz) bool: where lines may travel

    def __post_init__(self):
        g = self.grid
        stress = np.asarray(self.stress, dtype=np.float64)
        if stress.size != 6 * g.n_elems:
            raise ValueError("stress must have the shape grid.shape + (6,)")
        stress = stress.reshape(g.shape + (6,))
        solid = np.asarray(self.solid, dtype=bool)
        if solid.size != g.n_elems:
            raise ValueError("solid must have the shape grid.shape")
        solid = solid.reshape(g.shape)
        finite = np.isfinite(stress).all(axis=-1)
        if not finite.all():
            stress = np.where(finite[..., None], stress, 0.0)
            solid = solid & finite
        self.stress = stress
        self.solid = solid

        # Interpolation table: 6 solid-masked stress channels + the mask itself, padded
        # with one empty layer on the upper side of every axis so that the 8 corners
        # i0 + {0, 1} of a sample always exist (padding cells have zero weight).
        nx, ny, nz = g.shape
        table = np.zeros((nx + 1, ny + 1, nz + 1, 7), dtype=np.float64)
        table[:nx, :ny, :nz, :6] = np.where(solid[..., None], stress, 0.0)
        table[:nx, :ny, :nz, 6] = solid
        self._table = table.reshape(-1, 7)
        sy = nz + 1
        sx = (ny + 1) * sy
        self._strides = np.array([sx, sy, 1], dtype=np.int64)
        self._corners = np.array(
            [a * sx + b * sy + c for a in (0, 1) for b in (0, 1) for c in (0, 1)],
            dtype=np.int64,
        )
        self._centre_max = np.asarray(g.shape, dtype=np.float64) - 1.0
        self._shape = np.asarray(g.shape, dtype=np.int64)
        self._solid_strides = np.array([ny * nz, nz, 1], dtype=np.int64)
        self._solid_flat = solid.ravel()

    # ------------------------------------------------------------ public API
    def sample_tensor(self, points) -> np.ndarray:
        """Stress tensors ``(..., 3, 3)`` at world ``points`` ``(..., 3)``.

        Masked trilinear interpolation of the tensor between element centres: only
        solid elements contribute and their weights are renormalized. Points outside
        the grid are clamped onto it; points with no solid element among their 8
        surrounding centres (or non-finite points) get a zero tensor.
        """
        p = np.asarray(points, dtype=np.float64)
        u, bad = self._to_voxels(p)
        s6 = self._sample_voigt(u)
        s6[bad] = 0.0
        return s6[:, _VOIGT_TO_TENSOR].reshape(p.shape[:-1] + (3, 3))

    def inside(self, points) -> np.ndarray:
        """``(...)`` bool: the point lies in the grid and its element is solid."""
        p = np.asarray(points, dtype=np.float64)
        u, bad = self._to_voxels(p)
        out = self._inside_voxels(u)
        out[bad] = False
        return out.reshape(p.shape[:-1])

    # ------------------------------------------------- voxel-space internals
    def _to_voxels(self, p: np.ndarray) -> tuple:
        if p.shape[-1:] != (3,):
            raise ValueError("points must have a trailing dimension of 3")
        u = (p.reshape(-1, 3) - self.grid.origin) / self.grid.h
        bad = ~np.isfinite(u).all(axis=1)
        if bad.any():
            u[bad] = 0.0
        return u, bad

    def _sample_voigt(self, u: np.ndarray) -> np.ndarray:
        """Masked trilinear interpolation of the Voigt stress at voxel coords ``(P, 3)``."""
        q = np.clip(u - 0.5, 0.0, self._centre_max)
        i0 = np.floor(q)
        t = q - i0
        base = i0.astype(np.int64) @ self._strides
        tx, ty, tz = t[:, 0:1], t[:, 1:2], t[:, 2:3]
        wx = np.concatenate([1.0 - tx, tx], axis=1)
        wy = np.concatenate([1.0 - ty, ty], axis=1)
        wz = np.concatenate([1.0 - tz, tz], axis=1)
        w = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)
        acc = np.einsum("pc,pck->pk", w, self._table[base[:, None] + self._corners])
        den = acc[:, 6:7]
        return np.divide(acc[:, :6], den, out=np.zeros((len(u), 6)), where=den > 0.0)

    def _inside_voxels(self, u: np.ndarray) -> np.ndarray:
        """``(P,)`` bool for finite voxel coords: inside the grid and in a solid element."""
        e = np.floor(u).astype(np.int64)
        ok = np.all((e >= 0) & (e < self._shape), axis=1)
        out = np.zeros(len(u), dtype=bool)
        if ok.any():
            out[ok] = self._solid_flat[e[ok] @ self._solid_strides]
        return out


@dataclass(eq=False)
class Polyline:
    """One traced trajectory (world coordinates)."""

    points: np.ndarray  # (n, 3) world
    magnitude: np.ndarray  # (n,) |principal stress of the family| at each point [Pa]
    family: str  # "TENSION" (max principal) | "COMPRESSION" (min principal)
    generation: int  # 0 = primary, 1.. = branches
    sign: np.ndarray  # (n,) +1 where the family's principal stress is >= 0, else -1
    closed: bool = False  # closed loop: the last point repeats the first one


@dataclass(eq=False)
class TraceSettings:
    """Parameters of :func:`trace`; lengths are in voxels."""

    families: tuple = ("TENSION", "COMPRESSION")
    n_seeds: int = 40  # stress-weighted random seeds per family
    seed_points: Optional[np.ndarray] = None  # (k, 3) world: extra seeds, traced first
    step: float = 0.5  # integration step (capped at separation / 2)
    max_steps: int = 4000  # per direction
    min_stress: float = 0.02  # stop below this fraction of the family's reference stress
    separation: float = 2.0  # minimum distance between lines of the same family
    branch_depth: int = 1
    branch_spacing: float = 4.0  # distance between candidate branch points along a line
    branch_ratio: float = 0.3  # branch where |sigma_other| >= ratio * |sigma_this|
    branch_weight: Optional[np.ndarray] = None  # (nx, ny, nz) in [0, 1]: BRANCH zones
    min_points: int = 5
    rng_seed: int = 0
    max_lines: int = 2000
    integrator: str = "RK4"  # "RK4" | "RK2" (midpoint)


# ------------------------------------------------------------ eigen-analysis
def _deviator_eigen(s6: np.ndarray) -> tuple:
    """Closed-form eigenvalues of symmetric tensors given as Voigt rows ``(P, 6)``.

    Returns ``(m, dev, d)``: the mean normal stress ``m`` (P,), the diagonal of the
    deviator ``dev`` (P, 3) and the deviator eigenvalues ``d`` (P, 3) ascending
    (eigenvalues of the tensor = ``m + d``). Trigonometric solution of the deviator's
    characteristic cubic: exact to rounding for distinct eigenvalues; nearly repeated
    ones lose ~half the digits, which is irrelevant here since such directions are
    treated as degenerate.
    """
    xy, yz, zx = s6[:, 3], s6[:, 4], s6[:, 5]
    m = (s6[:, 0] + s6[:, 1] + s6[:, 2]) / 3.0
    dev = s6[:, :3] - m[:, None]
    a, b, c = dev[:, 0], dev[:, 1], dev[:, 2]
    p = (a * a + b * b + c * c + 2.0 * (xy * xy + yz * yz + zx * zx)) / 6.0
    det = a * (b * c - yz * yz) - xy * (xy * c - yz * zx) + zx * (xy * yz - b * zx)
    sp = np.sqrt(p)
    den = p * sp
    r = np.divide(0.5 * det, den, out=np.zeros_like(den), where=den > 0.0)
    phi = np.arccos(np.clip(r, -1.0, 1.0)) / 3.0
    d_max = 2.0 * sp * np.cos(phi)
    d_min = 2.0 * sp * np.cos(phi + _TWO_THIRDS_PI)
    d = np.stack([d_min, -(d_max + d_min), d_max], axis=1)
    return m, dev, d


def _principal_values(s6: np.ndarray) -> np.ndarray:
    """Principal stresses ``(P, 3)`` (ascending) of Voigt rows ``(P, 6)``."""
    m, _, d = _deviator_eigen(s6)
    return m[:, None] + d


def _principal_frame(s6: np.ndarray, col) -> tuple:
    """Principal values and the eigenvector of one principal stress per row.

    ``col`` (scalar or (P,)) selects the eigenvalue: 2 = largest, 0 = smallest.
    Returns ``(values, vectors, well)``: principal stresses ``(P, 3)`` ascending, unit
    eigenvectors ``(P, 3)`` of the selected eigenvalue (zero where undefined) and
    ``well`` ``(P,)`` bool, False where the direction is undefined (eigen-gap below
    ``_GAP_TOL`` times the largest |principal stress|).

    The eigenvector comes from the rows of ``deviator - d I``: their cross products are
    the columns of the adjugate, all parallel to the eigenvector; the longest one is
    used (its length is ~ the product of the two eigen-gaps, so it is accurate
    whenever the direction is well defined).
    """
    m, dev, d = _deviator_eigen(s6)
    n = len(s6)
    col = np.broadcast_to(np.asarray(col, dtype=np.int64), (n,))
    top = col == 2
    mu = np.where(top, d[:, 2], d[:, 0])
    gap = np.where(top, d[:, 2] - d[:, 1], d[:, 1] - d[:, 0])
    values = m[:, None] + d
    scale = np.maximum(np.abs(values[:, 0]), np.abs(values[:, 2]))
    well = gap > _GAP_TOL * scale

    a = dev[:, 0] - mu
    b = dev[:, 1] - mu
    c = dev[:, 2] - mu
    xy, yz, zx = s6[:, 3], s6[:, 4], s6[:, 5]
    cand = np.empty((n, 3, 3), dtype=np.float64)
    cand[:, 0, 0] = xy * yz - zx * b  # row0 x row1
    cand[:, 0, 1] = zx * xy - a * yz
    cand[:, 0, 2] = a * b - xy * xy
    cand[:, 1, 0] = xy * c - zx * yz  # row0 x row2
    cand[:, 1, 1] = zx * zx - a * c
    cand[:, 1, 2] = a * yz - xy * zx
    cand[:, 2, 0] = b * c - yz * yz  # row1 x row2
    cand[:, 2, 1] = yz * zx - xy * c
    cand[:, 2, 2] = xy * yz - b * zx
    n2 = np.einsum("pij,pij->pi", cand, cand)
    best = np.argmax(n2, axis=1)
    rows = np.arange(n)
    vec = cand[rows, best]
    norm = np.sqrt(n2[rows, best])[:, None]
    vec = np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0.0)
    return values, vec, well


def _field_magnitudes(field: StressField) -> tuple:
    """``(ids, mags)``: flat ids of solid elements and ``|sigma|`` per family ``(2, n)``."""
    ids = np.flatnonzero(field.solid.ravel())
    s6 = field.stress.reshape(-1, 6)
    mags = np.empty((2, ids.size), dtype=np.float64)
    for start in range(0, ids.size, _EIG_CHUNK):
        sl = slice(start, start + _EIG_CHUNK)
        values = _principal_values(s6[ids[sl]])
        mags[0, sl] = np.abs(values[:, 2])
        mags[1, sl] = np.abs(values[:, 0])
    return ids, mags


# ---------------------------------------------------------- spatial index
class _CellIndex:
    """Points bucketed in cubic cells and kept sorted by cell key (voxel units).

    Supports exact fixed-radius queries for radii up to the cell size (the 27
    neighbouring cells are scanned as 9 contiguous key ranges). Memory is O(points):
    cells are virtual, nothing is allocated per cell.
    """

    def __init__(self, origin, extent, cell: float):
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.cell = float(cell)
        extent = np.maximum(np.asarray(extent, dtype=np.float64).reshape(3), 0.0)
        # one empty margin cell below and above the occupied range
        self.dims = np.floor(extent / self.cell).astype(np.int64) + 3
        d1, d2 = int(self.dims[1]), int(self.dims[2])
        self._mult = np.array([d1 * d2, d2, 1], dtype=np.int64)
        # first key of each of the 9 (dx, dy) columns of 3 cells (dz = -1, 0, 1)
        self._columns = np.array(
            [dx * d1 * d2 + dy * d2 - 1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
            dtype=np.int64,
        )
        self.keys = np.zeros(0, dtype=np.int64)
        self.points = np.zeros((0, 3), dtype=np.float64)
        self.ids = np.zeros(0, dtype=np.int64)
        self._next_id = 0

    @property
    def size(self) -> int:
        return int(self.keys.size)

    def _keys(self, u: np.ndarray) -> np.ndarray:
        c = np.floor((u - self.origin) / self.cell).astype(np.int64) + 1
        np.clip(c, 1, self.dims - 2, out=c)
        return c @ self._mult

    def insert(self, u) -> None:
        u = np.asarray(u, dtype=np.float64).reshape(-1, 3)
        if not len(u):
            return
        keys = self._keys(u)
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        at = np.searchsorted(self.keys, keys, side="right")
        self.keys = np.insert(self.keys, at, keys)
        self.points = np.insert(self.points, at, u[order], axis=0)
        self.ids = np.insert(self.ids, at, self._next_id + order)
        self._next_id += len(u)

    def _candidates(self, u: np.ndarray) -> tuple:
        """``(query, stored)`` index pairs of every stored point in the 27 cells around."""
        first = self._keys(u)[:, None] + self._columns[None, :]
        lo = np.searchsorted(self.keys, first, side="left").ravel()
        hi = np.searchsorted(self.keys, first + 2, side="right").ravel()
        counts = hi - lo
        nz = np.flatnonzero(counts)
        if not nz.size:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        counts = counts[nz]
        ends = np.cumsum(counts)
        stored = np.arange(int(ends[-1]), dtype=np.int64) + np.repeat(lo[nz] - ends + counts, counts)
        query = np.repeat(nz // 9, counts)
        return query, stored

    def near(self, u, radius: float) -> np.ndarray:
        """``(Q,)`` bool: a stored point lies closer than ``radius`` (<= cell)."""
        u = np.asarray(u, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(len(u), dtype=bool)
        if not self.size or not len(u):
            return out
        query, stored = self._candidates(u)
        if query.size:
            diff = self.points[stored] - u[query]
            close = np.einsum("ij,ij->i", diff, diff) < radius * radius
            out[query[close]] = True
        return out

    def pairs(self, radius: float) -> tuple:
        """Insertion ids ``(i, j)``, ``i < j``, of stored points closer than ``radius``."""
        if self.size < 2:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        query, stored = self._candidates(self.points)
        a, b = self.ids[query], self.ids[stored]
        keep = a < b
        query, stored, a, b = query[keep], stored[keep], a[keep], b[keep]
        diff = self.points[stored] - self.points[query]
        close = np.einsum("ij,ij->i", diff, diff) < radius * radius
        return a[close], b[close]


# ---------------------------------------------------------------- the tracer
class _Batch:
    """Half-lines of one round: candidate ``k`` has halves ``k`` (forward) and ``K + k``."""

    def __init__(self, seeds, fams, val0, vec0, uid0: int, max_points: int):
        k = len(seeds)
        self.K = k
        n = 2 * k
        self.fams = np.asarray(fams, dtype=np.int64)
        self.val0 = np.asarray(val0, dtype=np.float64)
        self.fam_h = np.concatenate([self.fams, self.fams])
        self.col_h = _FAMILY_COLUMN[self.fam_h]
        self.partner = np.concatenate([np.arange(k, n), np.arange(k)])
        self.uid = uid0 + np.arange(n, dtype=np.int64)
        self.pos = np.concatenate([seeds, seeds]).astype(np.float64)
        self.dirn = np.concatenate([vec0, -vec0]).astype(np.float64)
        self.max_points = int(max_points)
        cap = min(self.max_points, 128)
        self.trail_u = np.empty((n, cap, 3), dtype=np.float64)
        self.trail_v = np.empty((n, cap), dtype=np.float64)
        self.trail_u[:, 0] = self.pos
        self.trail_v[:, 0] = np.concatenate([self.val0, self.val0])
        self.n_pts = np.ones(n, dtype=np.int64)
        self.alive = np.ones(n, dtype=bool)
        self.degenerate = np.zeros(n, dtype=np.int64)
        self.closed = np.zeros(k, dtype=bool)

    def ensure_capacity(self, needed: int) -> None:
        cap = self.trail_u.shape[1]
        if needed <= cap:
            return
        new_cap = min(self.max_points, max(needed, 2 * cap))
        n = len(self.pos)
        trail_u = np.empty((n, new_cap, 3), dtype=np.float64)
        trail_v = np.empty((n, new_cap), dtype=np.float64)
        trail_u[:, :cap] = self.trail_u
        trail_v[:, :cap] = self.trail_v
        self.trail_u, self.trail_v = trail_u, trail_v


class _Tracer:
    """State of one :func:`trace` call (never shared between calls)."""

    def __init__(self, field: StressField, settings: TraceSettings, progress, should_stop):
        if not isinstance(field, StressField):
            raise TypeError("field must be a StressField")
        self.field = field
        self.progress = progress
        self.should_stop = should_stop
        g = field.grid
        self.grid = g
        self.shape = np.asarray(g.shape, dtype=np.int64)
        self._resolve(settings)
        self.lines = []  # accepted Polylines
        self.line_u = []  # their points in voxel units (closed loops without the repeat)
        self.line_fam = []  # their family ids
        extent = np.asarray(g.shape, dtype=np.float64)
        self.index = [_CellIndex(np.zeros(3), extent, self.sep) for _ in FAMILIES]
        self._next_uid = 0
        self._init_stamps()

    # ------------------------------------------------------------ settings
    def _resolve(self, st: TraceSettings) -> None:
        families = st.families
        if isinstance(families, str):
            families = (families,)
        families = tuple(dict.fromkeys(str(f).upper() for f in families))
        for fam in families:
            if fam not in FAMILIES:
                raise ValueError(f"Unknown line family {fam!r} (use {', '.join(FAMILIES)})")
        self.order = [FAMILIES.index(f) for f in families]  # requested family ids, in order
        self.sep = float(st.separation)
        if not (math.isfinite(self.sep) and self.sep > 0.0):
            raise ValueError("separation must be a positive number of voxels")
        step = float(st.step)
        if not (math.isfinite(step) and step > 0.0):
            raise ValueError("step must be a positive number of voxels")
        self.s = min(step, 0.5 * self.sep)  # keeps the separation test exact to a few %
        self.max_steps = int(st.max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.spacing = float(st.branch_spacing)
        if not (math.isfinite(self.spacing) and self.spacing > 0.0):
            raise ValueError("branch_spacing must be a positive number of voxels")
        integrator = str(st.integrator).upper()
        if integrator not in INTEGRATORS:
            raise ValueError(f"Unknown integrator {st.integrator!r} (use RK4 or RK2)")
        self.rk4 = integrator == "RK4"
        self.n_seeds = max(0, int(st.n_seeds))
        self.min_stress = max(0.0, float(st.min_stress))
        self.depth = max(0, int(st.branch_depth))
        self.ratio = max(0.0, float(st.branch_ratio))
        self.min_points = max(2, int(st.min_points))
        self.max_lines = max(0, int(st.max_lines))
        self.rng = np.random.default_rng(None if st.rng_seed is None else int(st.rng_seed))

        g = self.field.grid
        if st.branch_weight is None:
            self.branch_weight = None
        else:
            bw = np.asarray(st.branch_weight, dtype=np.float64)
            if bw.size != g.n_elems:
                raise ValueError("branch_weight must have the shape grid.shape")
            bw = np.clip(np.nan_to_num(bw.reshape(g.shape), nan=0.0), 0.0, 1.0)
            self.branch_weight = bw if bw.any() else None

        if st.seed_points is None:
            self.seed_world = np.zeros((0, 3), dtype=np.float64)
        else:
            sp = np.asarray(st.seed_points, dtype=np.float64)
            if sp.size % 3:
                raise ValueError("seed_points must have shape (k, 3)")
            sp = sp.reshape(-1, 3)
            self.seed_world = sp[np.isfinite(sp).all(axis=1)]

        s = self.s
        self.close_dist = max(2.5 * s, 0.5 * self.sep)  # head-to-head distance closing a loop
        self.close_len = max(math.pi * self.sep, 4.0 * self.close_dist)  # shortest loop
        self.lag_self = max(2, int(math.ceil(0.5 * math.pi * self.sep / s)))
        self.max_degenerate = max(1, int(math.ceil(_DEGENERATE_LENGTH / s)))

    def _init_stamps(self) -> None:
        """Visit stamps: coarse cells remember the last half-line (and time) inside them.

        A half re-entering a cell it left ``lag_coil`` or more steps earlier, or ending
        closer than ``separation`` to where it was ``lag_coil`` steps ago, is curling
        onto itself: it stops early (the exact self-clipping happens at acceptance).
        """
        cell = _STAMP_CELL * self.sep
        dims = np.ceil(self.shape / cell).astype(np.int64) + 1
        while int(np.prod(dims)) > _MAX_STAMP_CELLS:
            cell *= 1.25
            dims = np.ceil(self.shape / cell).astype(np.int64) + 1
        self.stamp_cell = cell
        self.stamp_dims = dims
        self.stamp_mult = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
        n = int(np.prod(dims))
        self.stamp_uid = np.full(n, -1, dtype=np.int64)
        self.stamp_time = np.zeros(n, dtype=np.int64)
        self.lag_coil = max(2, int(math.ceil(2.0 * math.sqrt(3.0) * cell / self.s)))

    # -------------------------------------------------------------- driver
    def run(self) -> list:
        self._report(0.0)
        if self.max_lines == 0 or not self.order or not self.field.solid.any():
            self._report(1.0)
            return []
        self._references()
        if not any(self.active):
            self._report(1.0)
            return []
        primary_end = 0.6 if self.depth > 0 else 1.0
        u, fams = self._primary_seeds()
        self._process(u, fams, 0, 0.0, primary_end)
        for gen in range(self.depth):
            if len(self.lines) >= self.max_lines:
                break
            check_stop(self.should_stop)
            u, fams = self._branch_seeds(gen)
            if not len(u):
                break
            lo = primary_end + (1.0 - primary_end) * gen / self.depth
            hi = primary_end + (1.0 - primary_end) * (gen + 1) / self.depth
            self._process(u, fams, gen + 1, lo, hi)
        self._report(1.0)
        return self.lines

    def _report(self, fraction: float) -> None:
        if self.progress is not None:
            n = len(self.lines)
            self.progress(min(max(float(fraction), 0.0), 1.0), f"Tracing lines: {n} lines")

    def _references(self) -> None:
        """Per-family stop thresholds from the 99th percentile of |sigma| in the solid."""
        self.solid_ids, self.solid_mags = _field_magnitudes(self.field)
        refs = np.array(
            [float(np.percentile(m, _REF_PERCENTILE)) if m.size else 0.0 for m in self.solid_mags]
        )
        top = float(refs.max())
        # a family carrying (almost) no stress at all must not trace numerical noise
        refs = np.maximum(refs, _REF_FLOOR * top)
        self.threshold = self.min_stress * refs
        self.active = [top > 0.0 and f in self.order for f in range(len(FAMILIES))]

    # -------------------------------------------------------------- seeding
    def _primary_seeds(self) -> tuple:
        """Explicit seeds (every requested family), then stress-weighted samples."""
        order = [f for f in self.order if self.active[f]]
        u_parts, f_parts = [], []
        explicit = self._snap_seeds((self.seed_world - self.grid.origin) / self.grid.h)
        if len(explicit):
            u_parts.append(np.repeat(explicit, len(order), axis=0))
            f_parts.append(np.tile(np.asarray(order, dtype=np.int64), len(explicit)))
        samples = [self._sample_seeds(f) for f in order]
        if samples and self.n_seeds:
            rank = np.concatenate([np.arange(len(s)) for s in samples])
            slot = np.concatenate([np.full(len(s), i) for i, s in enumerate(samples)])
            interleave = np.lexsort((slot, rank))
            u_parts.append(np.concatenate(samples)[interleave])
            f_parts.append(np.asarray(order, dtype=np.int64)[slot[interleave]])
        if not u_parts:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
        return np.concatenate(u_parts), np.concatenate(f_parts)

    def _sample_seeds(self, fam: int) -> np.ndarray:
        """``n_seeds`` voxel-space seeds drawn ∝ |sigma_fam| * (1 + 3 branch_weight)."""
        mags = self.solid_mags[fam]
        weights = np.where(mags >= self.threshold[fam], mags, 0.0)
        if self.branch_weight is not None:
            weights = weights * (1.0 + 3.0 * self.branch_weight.ravel()[self.solid_ids])
        count = min(self.n_seeds, int(np.count_nonzero(weights)))
        if count == 0:
            return np.zeros((0, 3))
        picks = self.rng.choice(weights.size, size=count, replace=False, p=weights / weights.sum())
        ijk = np.stack(np.unravel_index(self.solid_ids[picks], self.field.grid.shape), axis=1)
        return ijk + self.rng.random((count, 3))

    def _snap_seeds(self, u: np.ndarray) -> np.ndarray:
        """Seeds outside the solid move to the nearest solid centre within ``_SNAP_RADIUS``."""
        if not len(u):
            return np.zeros((0, 3))
        inside = self.field._inside_voxels(u)
        solid = self.field.solid
        out = []
        for p, ok in zip(u, inside):
            if ok:
                out.append(p)
                continue
            lo = np.maximum(np.floor(p - _SNAP_RADIUS).astype(np.int64), 0)
            hi = np.minimum(np.floor(p + _SNAP_RADIUS).astype(np.int64) + 1, self.shape)
            if np.any(hi <= lo):
                continue
            cells = np.argwhere(solid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
            if not len(cells):
                continue
            centres = cells + lo + 0.5
            dist = np.linalg.norm(centres - p, axis=1)
            j = int(np.argmin(dist))
            if dist[j] <= _SNAP_RADIUS:
                out.append(centres[j])
        return np.asarray(out, dtype=np.float64).reshape(-1, 3)

    def _branch_seeds(self, gen: int) -> tuple:
        """Seeds of generation ``gen + 1``: points of generation-``gen`` lines, every
        ``branch_spacing`` voxels, where the other family's stress is significant.

        Ordered by decreasing crossing stress (strongest branches first)."""
        pts, parent = [], []
        for idx, line in enumerate(self.lines):
            if line.generation != gen or not self.active[1 - self.line_fam[idx]]:
                continue
            u = self.line_u[idx]
            if len(u) < 2:
                continue
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(u, axis=0), axis=1))])
            at = np.arange(0.5 * self.spacing, cum[-1], self.spacing)
            if not at.size:
                continue
            pts.append(np.stack([np.interp(at, cum, u[:, a]) for a in range(3)], axis=1))
            parent.append(np.full(at.size, self.line_fam[idx], dtype=np.int64))
        if not pts:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
        u = np.concatenate(pts)
        this_fam = np.concatenate(parent)
        other_fam = 1 - this_fam
        values = _principal_values(self.field._sample_voigt(u))
        rows = np.arange(len(u))
        this = np.abs(values[rows, _FAMILY_COLUMN[this_fam]])
        other = np.abs(values[rows, _FAMILY_COLUMN[other_fam]])
        ratio = np.full(len(u), self.ratio)
        if self.branch_weight is not None:
            e = np.clip(np.floor(u).astype(np.int64), 0, self.shape - 1)
            ratio = ratio * (1.0 - 0.5 * self.branch_weight[e[:, 0], e[:, 1], e[:, 2]])
        ok = (other >= ratio * this) & (other >= self.threshold[other_fam]) & (other > 0.0)
        order = np.argsort(-other[ok], kind="stable")
        return u[ok][order], other_fam[ok][order]

    # ----------------------------------------------------------- processing
    def _process(self, seeds: np.ndarray, fams: np.ndarray, gen: int, lo: float, hi: float):
        """Trace the seeds of one generation in order (batches of admissible seeds)."""
        total = len(seeds)
        pos = 0
        while pos < total and len(self.lines) < self.max_lines:
            check_stop(self.should_stop)
            remaining = self.max_lines - len(self.lines)
            batch = min(_BATCH_SEEDS, max(16, 2 * remaining))
            end = min(total, pos + _SCAN_FACTOR * batch)
            ok, val0, vec0 = self._admissible(seeds[pos:end], fams[pos:end])
            sel = np.flatnonzero(ok)
            nxt = end
            if sel.size > batch:
                sel = sel[:batch]
                nxt = pos + int(sel[-1]) + 1
            if sel.size:
                b = _Batch(
                    seeds[pos + sel],
                    fams[pos + sel],
                    val0[sel],
                    vec0[sel],
                    self._next_uid,
                    self.max_steps + 1,
                )
                self._next_uid += 2 * b.K
                self._integrate(b)
                self._accept(b, gen)
            pos = nxt
            self._report(lo + (hi - lo) * pos / max(total, 1))

    def _admissible(self, u: np.ndarray, fams: np.ndarray) -> tuple:
        """Seeds that may start a line now: inside, stressed, non-degenerate, separated."""
        cols = _FAMILY_COLUMN[fams]
        values, vec, well = _principal_frame(self.field._sample_voigt(u), cols)
        val = values[np.arange(len(u)), cols]
        active = np.asarray(self.active, dtype=bool)[fams]
        mag = np.abs(val)
        ok = self.field._inside_voxels(u) & well & active & (mag >= self.threshold[fams])
        ok &= mag > 0.0
        for fam in range(len(FAMILIES)):
            m = ok & (fams == fam)
            if self.index[fam].size and m.any():
                ok[m] = ~self.index[fam].near(u[m], self.sep)
        return ok, val, vec

    # ---------------------------------------------------------- integration
    def _direction(self, x: np.ndarray, cols: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Family direction at ``x`` oriented like ``ref`` (``ref`` where undefined)."""
        _, vec, well = _principal_frame(self.field._sample_voigt(x), cols)
        dot = np.einsum("ij,ij->i", vec, ref)
        vec = np.where((dot < 0.0)[:, None], -vec, vec)
        good = well & (np.abs(dot) >= _COS_MAX_TURN)
        return np.where(good[:, None], vec, ref)

    def _increment(self, x: np.ndarray, d: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Unit RK4 / RK2 direction of the next step from ``x`` (current direction ``d``)."""
        half = 0.5 * self.s
        k2 = self._direction(x + half * d, cols, d)
        if self.rk4:
            k3 = self._direction(x + half * k2, cols, d)
            k4 = self._direction(x + self.s * k3, cols, d)
            inc = d + 2.0 * (k2 + k3) + k4
        else:
            inc = k2
        norm = np.linalg.norm(inc, axis=1)[:, None]
        return np.where(norm > 1e-12, inc / np.where(norm > 1e-12, norm, 1.0), d)

    def _integrate(self, b: _Batch) -> None:
        """Advance every half-line of the batch until all of them have stopped."""
        sep = self.sep
        while True:
            ia = np.flatnonzero(b.alive)
            if not ia.size:
                return
            check_stop(self.should_stop)
            b.ensure_capacity(int(b.n_pts[ia].max()) + 1)
            x = b.pos[ia]
            d = b.dirn[ia]
            cols = b.col_h[ia]
            fam = b.fam_h[ia]
            inc = self._increment(x, d, cols)
            x_new = x + self.s * inc

            values, vec, well = _principal_frame(self.field._sample_voigt(x_new), cols)
            val = values[np.arange(ia.size), cols]
            dot = np.einsum("ij,ij->i", vec, inc)
            vec = np.where((dot < 0.0)[:, None], -vec, vec)
            good = well & (np.abs(dot) >= _COS_MAX_TURN)
            new_dir = np.where(good[:, None], vec, inc)  # degenerate: keep going straight
            degenerate = np.where(good, 0, b.degenerate[ia] + 1)

            stop = ~self.field._inside_voxels(x_new)
            stop |= np.abs(val) < self.threshold[fam]
            stop |= degenerate > self.max_degenerate
            for f in range(len(FAMILIES)):
                if self.index[f].size:
                    m = ~stop & (fam == f)
                    if m.any():
                        stop[m] = self.index[f].near(x_new[m], sep)
            t_new = b.n_pts[ia]
            stop |= self._curling(b, ia, x_new, t_new, ~stop)

            rec = ~stop
            ir = ia[rec]
            tr = t_new[rec]
            b.trail_u[ir, tr] = x_new[rec]
            b.trail_v[ir, tr] = val[rec]
            b.n_pts[ir] += 1
            b.pos[ir] = x_new[rec]
            b.dirn[ir] = new_dir[rec]
            b.degenerate[ir] = degenerate[rec]
            b.alive[ia[stop]] = False
            if ir.size:
                self._close_loops(b, ir)
                b.alive[ir[b.n_pts[ir] > self.max_steps]] = False

    def _curling(self, b: _Batch, ia, x_new, t_new, candidates) -> np.ndarray:
        """Halves curling onto their line (coarse, cheap safeguard; see ``_init_stamps``).

        A cell last visited by the same half ``lag_coil`` steps ago or more, or by the
        other half of the same line at a point that half left ``lag_coil`` steps ago or
        more (its trail, not the region around its head where loops close), counts as
        a self-approach; so does a chord over the last ``lag_coil`` steps shorter than
        ``separation``.
        """
        curl = np.zeros(ia.size, dtype=bool)
        if not candidates.any():
            return curl
        back = t_new - self.lag_coil
        m = candidates & (back >= 0)
        if m.any():
            chord = x_new[m] - b.trail_u[ia[m], back[m]]
            curl[m] = np.einsum("ij,ij->i", chord, chord) < self.sep * self.sep
        c = np.floor(x_new / self.stamp_cell).astype(np.int64)
        np.clip(c, 0, self.stamp_dims - 1, out=c)
        cells = c @ self.stamp_mult
        uid = b.uid[ia]
        partner = b.partner[ia]
        owner = self.stamp_uid[cells]
        stamped = self.stamp_time[cells]
        own = (owner == uid) & (t_new - stamped >= self.lag_coil)
        other = (owner == b.uid[partner]) & (b.n_pts[partner] - 1 - stamped >= self.lag_coil)
        curl |= candidates & (own | other)
        w = candidates & ~curl
        self.stamp_uid[cells[w]] = uid[w]
        self.stamp_time[cells[w]] = t_new[w]
        return curl

    def _close_loops(self, b: _Batch, ir: np.ndarray) -> None:
        """Stop both halves of lines whose heads meet (closed loops).

        The heads must be closer than ``close_dist`` after a total length of at least
        ``close_len``, and either the other head lies ahead or the two heads travel in
        opposite directions (they may pass each other within one step, also with a
        small lateral offset when the trajectory does not close exactly).
        """
        pr = b.partner[ir]
        gap = b.pos[pr] - b.pos[ir]
        dist = np.linalg.norm(gap, axis=1)
        ahead = np.einsum("ij,ij->i", gap, b.dirn[ir]) >= _CLOSE_ALIGN * dist - 1e-12
        opposite = np.einsum("ij,ij->i", b.dirn[pr], b.dirn[ir]) <= -_COS_MAX_TURN
        length = (b.n_pts[ir] + b.n_pts[pr] - 2) * self.s
        closing = (dist < self.close_dist) & (ahead | opposite) & (length >= self.close_len)
        if closing.any():
            b.alive[ir[closing]] = False
            b.alive[pr[closing]] = False
            b.closed[ir[closing] % b.K] = True

    # ------------------------------------------------------------ acceptance
    def _accept(self, b: _Batch, gen: int) -> None:
        """Accept the candidates in seed order, clipped against the lines before them."""
        K = b.K
        origin, h = self.grid.origin, self.grid.h
        for k in range(K):
            if len(self.lines) >= self.max_lines:
                return
            check_stop(self.should_stop)
            fam = int(b.fams[k])
            index = self.index[fam]
            n_f, n_b = int(b.n_pts[k]), int(b.n_pts[K + k])
            seed = b.trail_u[k, 0]
            fu, fv = b.trail_u[k, 1:n_f], b.trail_v[k, 1:n_f]
            bu, bv = b.trail_u[K + k, 1:n_b], b.trail_v[K + k, 1:n_b]
            keep_f, keep_b = len(fu), len(bu)
            if index.size:
                hits = index.near(np.concatenate([seed[None], fu, bu]), self.sep)
                if hits[0]:
                    continue
                hf, hb = hits[1 : 1 + keep_f], hits[1 + keep_f :]
                if hf.any():
                    keep_f = int(np.argmax(hf))
                if hb.any():
                    keep_b = int(np.argmax(hb))
            closed = bool(b.closed[k]) and keep_f == len(fu) and keep_b == len(bu)
            if not closed:
                keep_f, keep_b = self._self_clip(seed, fu[:keep_f], bu[:keep_b])
            if 1 + keep_f + keep_b + int(closed) < self.min_points:
                continue
            u = np.concatenate([bu[:keep_b][::-1], seed[None], fu[:keep_f]])
            v = np.concatenate([bv[:keep_b][::-1], b.val0[k : k + 1], fv[:keep_f]])
            index.insert(u)
            self.line_u.append(u)
            self.line_fam.append(fam)
            if closed:
                u = np.concatenate([u, u[:1]])
                v = np.concatenate([v, v[:1]])
            self.lines.append(
                Polyline(
                    points=origin + u * h,
                    magnitude=np.abs(v),
                    family=FAMILIES[fam],
                    generation=int(gen),
                    sign=np.where(v < 0.0, -1.0, 1.0),
                    closed=closed,
                )
            )

    def _self_clip(self, seed: np.ndarray, fu: np.ndarray, bu: np.ndarray) -> tuple:
        """Cut the halves where the line comes within ``separation`` of its own earlier part.

        Point ``i`` of a half was reached at time ``i + 1`` (both halves advance
        together from the seed at time 0). Two points closer than ``separation`` whose
        distance *along the line* is at least a half circle of diameter ``separation``
        form a conflict, resolved at the time the later of them was reached: that half
        is cut there (both when they were reached together). Returns the number of
        points kept on each half.
        """
        n_f, n_b = len(fu), len(bu)
        if n_f + n_b < self.lag_self:
            return n_f, n_b
        pts = np.concatenate([seed[None], fu, bu])
        half = np.concatenate([[0], np.ones(n_f, np.int64), np.full(n_b, 2, np.int64)])
        time = np.concatenate([[0], np.arange(1, n_f + 1), np.arange(1, n_b + 1)])
        lo = pts.min(axis=0)
        index = _CellIndex(lo, pts.max(axis=0) - lo, self.sep)
        index.insert(pts)
        i, j = index.pairs(self.sep)
        if not i.size:
            return n_f, n_b
        hi_, hj, ti, tj = half[i], half[j], time[i], time[j]
        along = np.where((hi_ == hj) & (hi_ > 0), np.abs(ti - tj), ti + tj)
        keep = along >= self.lag_self
        if not keep.any():
            return n_f, n_b
        hi_, hj, ti, tj = hi_[keep], hj[keep], ti[keep], tj[keep]
        when = np.maximum(ti, tj)
        cut = np.array([np.iinfo(np.int64).max, n_f + 1, n_b + 1], dtype=np.int64)
        for _ in range(2):  # the first conflict cuts one half (or both), the next the other
            valid = (ti < cut[hi_]) & (tj < cut[hj])
            if not valid.any():
                break
            first = when[valid].min()
            event = valid & (when == first)
            for hh in (1, 2):
                late = ((hi_ == hh) & (ti == first)) | ((hj == hh) & (tj == first))
                if np.any(event & late):
                    cut[hh] = min(cut[hh], first)
        return int(cut[1] - 1), int(cut[2] - 1)


def trace(
    field: StressField,
    settings: Optional[TraceSettings] = None,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
) -> list:
    """Principal stress trajectories of ``field`` (see the module docstring).

    Returns a list of :class:`Polyline` in acceptance order: primary lines (generation
    0, explicit seeds first) then the branches of each generation; at most
    ``settings.max_lines``. Deterministic for a fixed ``settings.rng_seed``. Raises
    :class:`~.grid.Cancelled` when ``should_stop()`` returns True and ``ValueError``
    for invalid settings.
    """
    if settings is None:
        settings = TraceSettings()
    return _Tracer(field, settings, progress, should_stop).run()
