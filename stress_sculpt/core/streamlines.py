"""Principal stress trajectories ("tension lines") traced through a voxel stress field.

The stress of an analysis (Voigt values at element centres, see ``core/grid.py``)
has at every point three orthogonal principal directions. A *trajectory* is a curve
that is everywhere tangent to one of them:

* ``"TENSION"`` lines follow the eigenvector of the largest principal stress,
* ``"COMPRESSION"`` lines follow the eigenvector of the smallest principal stress.

How :func:`trace` works (volume mode, the default):

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

Surface mode (``TraceSettings.surface``, a :class:`SurfaceConstraint`): the lines run
on the iso-surface of a scalar field (an optimized density, a smoothed domain
indicator ...) along the principal directions of the stress *in the surface*, the
isostatic ribs of a shell (Nervi):

* The normal ``n`` is the normalized gradient of the trilinear field; the tangential
  stress is ``P sigma P`` with ``P = I - n n^T``. Its eigenvector along ``n`` is
  discarded; ``"TENSION"`` follows the larger of the two remaining principal stresses,
  ``"COMPRESSION"`` the smaller one (closed form in a tangent basis). Magnitudes and
  signs of the polylines are those tangential principal stresses.
* After every step the point is projected back onto the iso-surface with 1-3 Newton
  steps on the trilinear field (``p -= (f(p) - iso) grad f / |grad f|^2``). A line stops
  when that fails or needs a move longer than ``tolerance`` voxels, when it leaves the
  box spanned by the field's element centres (where the trilinear field is defined:
  pad the field when the surface would touch the grid border), where no solid element
  is near, where the tangential stress drops below ``min_stress`` times the family's
  99th percentile over the surface, and at degenerate points / sharp turns as above.
  Turns are measured after carrying the previous direction along with the rotation of
  the normal, so lines may wrap around rounded edges. Lines are not confined to the
  solid elements (the surface usually bounds them; the masked interpolation gives the
  stress there).
* Lines are spaced evenly (Jobard & Lefer, "Creating evenly-spaced streamlines of
  arbitrary density", 1997): every accepted line spawns side seeds 1, 2 and 3
  ``separation`` to both of its sides (walked along the surface, perpendicular to the
  line); a line stops when it comes closer than ``SURFACE_STOP_RATIO * separation`` to a
  line of its family, and seeds need ``SURFACE_SEED_RATIO * separation`` of clearance,
  so neighbours run parallel at ``separation`` and only end where the stress lines
  converge. Fronts start at the ``seed_points`` (moved onto the surface), then at
  ``n_seeds`` stress-weighted random points of the surface (the iso crossings between
  neighbouring element centres, i.e. near-surface centres moved onto it): new fronts
  start far from the lines and pending seeds of their family, or once its fronts have
  stopped. Lines shorter than ``separation`` and loops shorter than ``pi *
  separation`` are dropped; branching works as in the volume mode (the filling leaves
  room for branches only where its fronts did not reach). All distances are 3-D: the
  two faces of a thin wall count as close.
* The field may live on its own, e.g. finer and smoother, grid (``SurfaceConstraint.grid``);
  optional smoothing of the traced lines (``SurfaceConstraint.smoothing``) irons out
  wiggles of noisy element stresses.

The result is the one of the classic sequential evenly-spaced streamline algorithm
(seed after seed, every line stopping at the lines accepted before it), but lines are
integrated in batches: all candidates of a round advance simultaneously (vectorized
interpolation and eigen-analysis), then they are accepted one by one, each one clipped
where it comes closer than the stop distance to a line accepted before it.

Internally positions are in voxel units ``u = (p - grid.origin) / grid.h`` of the
stress grid (element ``(i, j, k)`` spans ``[i, i + 1]`` along each axis, its centre is at
``i + 0.5``); all lengths in :class:`TraceSettings` and :class:`SurfaceConstraint` are in
those voxels. Results are in world coordinates.

Pure numpy, no global mutable state: independent calls may run in parallel threads.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .grid import Grid, ProgressFn, StopFn, check_stop


FAMILIES = ("TENSION", "COMPRESSION")
INTEGRATORS = ("RK4", "RK2")

# Surface mode: a line stops closer than this fraction of ``separation`` to a line of its
# family; a seed needs this fraction of ``separation`` of clearance (side seeds are
# placed at ``separation``, the margin absorbs their projection onto curved surfaces).
SURFACE_STOP_RATIO = 0.6
SURFACE_SEED_RATIO = 0.9

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
_MAX_STAMP_CELLS = 1 << 22
_EIG_CHUNK = 1 << 18  # elements per chunk when analysing the whole field

# surface mode
_SURFACE_REACH = 8.0  # voxels: explicit seeds this close to the surface are moved onto it
_SAMPLE_REACH = 1.5  # voxels: jittered random seeds are moved back at most this far
_PROJ_EPS = 1e-2  # voxels: residual distance to the surface that counts as "on it"
_SEED_EPS = 5e-2  # voxels: seeds must be at least this close to the surface
_NEWTON_STEP = 1.0  # voxels: longest move of one Newton iteration (damping far away)
_MARCH_STEP = 0.5  # voxels: side seeds are walked along the surface in steps this long
_SIDE_SPACING = 1.0  # x separation: distance between the side seeds along a line
_SIDE_LEVELS = 3  # side seeds after 1, 2, 3 separations: fronts advance 3 lines per round
_FIRST_FRONTS = 1  # random seeds starting the first fronts of a family
_FRONTS = 4  # random seeds starting new fronts, per family and round
_FAR_CELLS = 6  # a new front starts this many separation cells away from its family
_GAP_SPREAD = 3.0  # x separation: random seeds of one round filling gaps are this far apart
_MIN_LENGTH = 1.0  # x separation: shorter surface lines (loops: x pi) are dropped
_COVERAGE_SAMPLES = 512  # surface samples used to estimate the progress of the filling
_SAMPLE_CHUNK = 1 << 15  # points per chunk when sampling the whole surface
_MAX_SMOOTH_PASSES = 64  # binomial smoothing passes of a surface line at most


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
        w = _trilinear_weights(t)
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
class SurfaceConstraint:
    """Confines the lines of :func:`trace` to an iso-surface (surface mode).

    ``field`` is an element-centred scalar ``(nx, ny, nz)`` (an optimized density, a
    smoothed domain indicator, a distance ...) on the grid of the :class:`StressField`,
    or on its own ``grid`` (e.g. a finer, smoother copy of a voxel density); the surface
    is its ``iso`` level under trilinear interpolation between the element centres. It
    must lie inside the box spanned by those centres and inside the stress grid: pad the
    field (and the stress) when the surface would touch the grid border. Lengths are in
    voxels of the stress grid. ``tolerance``: a step whose end cannot be brought back
    onto the surface within this distance stops the line. ``smoothing``: traced lines
    are smoothed over about this length (binomial filter of their points, ends kept) and
    projected back onto the surface before they are clipped and accepted, which irons
    out wiggles of noisy element stresses on smooth surfaces (it cannot make the lines
    smoother than the surface itself: give staircase-like voxel fields a smoothed,
    preferably finer, copy); 0 keeps the exact trajectories. Arrays are treated as
    read-only.
    """

    field: np.ndarray  # (nx, ny, nz) element-centred scalar on ``grid``
    iso: float
    tolerance: float = 0.75  # voxels: a point further than this from the surface stops the line
    smoothing: float = 0.0  # voxels: smoothing length of the traced lines (0 = exact trajectories)
    grid: Optional[Grid] = None  # grid of ``field`` (None: the StressField's grid)

    def __post_init__(self):
        self.field = np.asarray(self.field, dtype=np.float64)
        if self.field.ndim != 3:
            raise ValueError("surface field must be a 3-D array (nx, ny, nz)")
        if not np.isfinite(self.field).all():
            raise ValueError("surface field must be finite")
        self.iso = float(self.iso)
        if not math.isfinite(self.iso):
            raise ValueError("surface iso level must be finite")
        self.tolerance = float(self.tolerance)
        if not (math.isfinite(self.tolerance) and self.tolerance > 0.0):
            raise ValueError("surface tolerance must be a positive number of voxels")
        self.smoothing = float(self.smoothing)
        if not (math.isfinite(self.smoothing) and self.smoothing >= 0.0):
            raise ValueError("surface smoothing must be a non-negative number of voxels")
        if self.grid is not None:
            if not isinstance(self.grid, Grid):
                raise TypeError("surface grid must be a Grid (or None)")
            if self.field.shape != self.grid.shape:
                raise ValueError("surface field must have the shape of its grid")


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
    """Parameters of :func:`trace`; lengths are in voxels.

    ``separation``: volume mode, the minimum distance between lines of a family; surface
    mode, the distance between neighbouring lines (they stop at ``SURFACE_STOP_RATIO``
    of it). ``surface``: trace on an iso-surface instead of through the volume.
    """

    families: tuple = ("TENSION", "COMPRESSION")
    n_seeds: int = 40  # stress-weighted random seeds per family
    seed_points: Optional[np.ndarray] = None  # (k, 3) world: extra seeds, traced first
    step: float = 0.5  # integration step (capped at half the stop distance)
    max_steps: int = 4000  # per direction
    min_stress: float = 0.02  # stop below this fraction of the family's reference stress
    separation: float = 2.0  # distance between lines of the same family (see module doc)
    branch_depth: int = 1
    branch_spacing: float = 4.0  # distance between candidate branch points along a line
    branch_ratio: float = 0.3  # branch where |sigma_other| >= ratio * |sigma_this|
    branch_weight: Optional[np.ndarray] = None  # (nx, ny, nz) in [0, 1]: BRANCH zones
    min_points: int = 5
    rng_seed: int = 0
    max_lines: int = 2000
    integrator: str = "RK4"  # "RK4" | "RK2" (midpoint)
    surface: Optional[SurfaceConstraint] = None  # trace on an iso-surface (surface mode)


# ------------------------------------------------------------ small helpers
def _trilinear_weights(t: np.ndarray) -> np.ndarray:
    """``(P, 8)`` trilinear weights of fractions ``t`` ``(P, 3)``, corner order a*4+b*2+c."""
    tx, ty, tz = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    wx = np.concatenate([1.0 - tx, tx], axis=1)
    wy = np.concatenate([1.0 - ty, ty], axis=1)
    wz = np.concatenate([1.0 - tz, tz], axis=1)
    return (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)


def _trilinear_value_grad(c: np.ndarray, t: np.ndarray) -> tuple:
    """Value ``(P,)`` and gradient ``(P, 3)`` (per voxel) of the trilinear interpolant of
    corner values ``c`` ``(P, 8)`` (order a*4+b*2+c) at fractions ``t`` ``(P, 3)``."""
    c = c.reshape(-1, 2, 2, 2)
    tx = t[:, 0]
    ty = t[:, 1:2]
    tz = t[:, 2:3, None]
    d_z = c[:, :, :, 1] - c[:, :, :, 0]  # (P, 2, 2)
    cz = c[:, :, :, 0] + tz * d_z
    d_y = cz[:, :, 1] - cz[:, :, 0]  # (P, 2)
    cy = cz[:, :, 0] + ty * d_y
    gx = cy[:, 1] - cy[:, 0]
    value = cy[:, 0] + tx * gx
    gy = d_y[:, 0] + tx * (d_y[:, 1] - d_y[:, 0])
    dz_y = d_z[:, :, 0] + ty * (d_z[:, :, 1] - d_z[:, :, 0])  # (P, 2)
    gz = dz_y[:, 0] + tx * (dz_y[:, 1] - dz_y[:, 0])
    return value, np.stack([gx, gy, gz], axis=1)


def _rowdot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product of ``(P, 3)`` arrays (cheaper than ``np.cross`` for small P)."""
    out = np.empty(np.broadcast(a, b).shape, dtype=np.float64)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _spread(points: np.ndarray, radius: float, cap: int) -> np.ndarray:
    """Indices (in order) of a greedy subset of ``points`` at least ``radius`` apart, at
    most ``cap`` of them."""
    cap = min(int(cap), len(points))
    if cap <= 0:
        return np.zeros(0, dtype=np.int64)
    kept = np.empty((cap, 3), dtype=np.float64)
    out = []
    r2 = radius * radius
    for i, p in enumerate(points):
        m = len(out)
        if m:
            d = kept[:m] - p
            if np.min(_rowdot(d, d)) < r2:
                continue
        kept[m] = p
        out.append(i)
        if len(out) >= cap:
            break
    return np.asarray(out, dtype=np.int64)


def _dilate_box(mask: np.ndarray, k: int) -> np.ndarray:
    """``mask`` grown by ``k`` cells along every axis (a (2k+1)^3 box, zero padded)."""
    out = np.asarray(mask, dtype=bool)
    for axis in range(out.ndim):
        n = out.shape[axis]
        acc = out.copy()
        for s in range(1, min(int(k), n - 1) + 1):
            lo = [slice(None)] * out.ndim
            hi = [slice(None)] * out.ndim
            lo[axis] = slice(0, n - s)
            hi[axis] = slice(s, n)
            acc[tuple(lo)] |= out[tuple(hi)]
            acc[tuple(hi)] |= out[tuple(lo)]
        out = acc
    return out


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


def _tangent_eigen(s6: np.ndarray, n: np.ndarray) -> tuple:
    """Principal stresses of ``sigma`` projected onto the planes normal to ``n``.

    ``s6`` Voigt rows ``(P, 6)``, ``n`` unit normals ``(P, 3)``. The projected tensor
    ``P sigma P`` (``P = I - n n^T``) has ``n`` as an eigenvector (eigenvalue 0,
    discarded); the two others lie in the plane: with a tangent basis ``(t1, t2)`` the
    2x2 tensor ``A_ij = t_i . sigma t_j`` is diagonalized in closed form. Returns
    ``(hi, lo, e_hi, e_lo, well)``: the larger / smaller tangential principal stress
    ``(P,)``, their unit directions ``(P, 3)`` and ``well`` ``(P,)`` bool, False where the
    two are (nearly) equal so that the directions are undefined (gap below
    ``_GAP_TOL`` times the larger magnitude).

    The tangent basis is the branchless one of Duff et al. ("Building an orthonormal
    basis, revisited", 2017); it jumps where ``n_z`` changes sign, which does not
    matter since the principal directions do not depend on the basis.
    """
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    sg = np.copysign(1.0, nz)
    a = -1.0 / (sg + nz)
    b = nx * ny * a
    t1x = 1.0 + sg * nx * nx * a
    t1y = sg * b
    t1z = -sg * nx
    t2x = b
    t2y = sg + ny * ny * a
    t2z = -ny
    sxx, syy, szz, sxy, syz, szx = s6[:, 0], s6[:, 1], s6[:, 2], s6[:, 3], s6[:, 4], s6[:, 5]
    u = sxx * t1x + sxy * t1y + szx * t1z  # sigma t1
    v = sxy * t1x + syy * t1y + syz * t1z
    w = szx * t1x + syz * t1y + szz * t1z
    a11 = t1x * u + t1y * v + t1z * w
    a12 = t2x * u + t2y * v + t2z * w
    u = sxx * t2x + sxy * t2y + szx * t2z  # sigma t2
    v = sxy * t2x + syy * t2y + syz * t2z
    w = szx * t2x + syz * t2y + szz * t2z
    a22 = t2x * u + t2y * v + t2z * w
    mean = 0.5 * (a11 + a22)
    half = 0.5 * (a11 - a22)
    r = np.hypot(half, a12)
    theta = 0.5 * np.arctan2(a12, half)
    c = np.cos(theta)
    s = np.sin(theta)
    e_hi = np.empty_like(n)
    e_lo = np.empty_like(n)
    e_hi[:, 0] = c * t1x + s * t2x
    e_hi[:, 1] = c * t1y + s * t2y
    e_hi[:, 2] = c * t1z + s * t2z
    e_lo[:, 0] = c * t2x - s * t1x
    e_lo[:, 1] = c * t2y - s * t1y
    e_lo[:, 2] = c * t2z - s * t1z
    hi = mean + r
    lo = mean - r
    well = 2.0 * r > _GAP_TOL * np.maximum(np.abs(hi), np.abs(lo))
    return hi, lo, e_hi, e_lo, well


def _unit(v: np.ndarray, eps: float) -> tuple:
    """``(v / |v|, |v|)`` row-wise; rows with ``|v| <= eps`` give a zero vector."""
    norm = np.sqrt(_rowdot(v, v))
    out = np.divide(v, norm[:, None], out=np.zeros_like(v), where=norm[:, None] > eps)
    return out, norm


def _transport(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Directions ``v`` tangent where the unit normal is ``a``, carried to the plane of
    the unit normal ``b``: the minimal rotation taking ``a`` to ``b`` (Rodrigues with
    ``w = a x b``: ``R v = c v + w x v + w (w . v) / (1 + c)``, ``c = a . b``), then a
    projection onto that plane. Unit vectors (``v`` itself where the result degenerates;
    normals turning by more than ~150 degrees are not rotated, only projected)."""
    w = _cross(a, b)
    c = _rowdot(a, b)
    k = _rowdot(w, v) / np.maximum(1.0 + c, 0.1)
    r = v * c[:, None] + _cross(w, v) + w * k[:, None]
    r = np.where((c > -0.9)[:, None], r, v)
    r -= _rowdot(r, b)[:, None] * b
    out, norm = _unit(r, 1e-9)
    return np.where((norm > 1e-9)[:, None], out, v)


def _trim_overlap(u: np.ndarray, v: np.ndarray, k0: int) -> tuple:
    """A closed loop ``u`` (``v`` its values, ``k0`` its seed index) whose two heads
    passed each other within the last step: drop head points until the closing segment
    ``u[-1] -> u[0]`` no longer runs backwards. Returns ``(u, v, k0)``."""
    for _ in range(8):
        if len(u) < 4:
            break
        gap = u[0] - u[-1]
        if float(gap @ (u[-1] - u[-2])) < 0.0 and k0 < len(u) - 1:  # forward head overshot
            u, v = u[:-1], v[:-1]
        elif float((u[1] - u[0]) @ gap) < 0.0 and k0 > 0:  # backward head overshot
            u, v, k0 = u[1:], v[1:], k0 - 1
        else:
            break
    return u, v, k0


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


# ------------------------------------------------------------ surface sampling
class _Surface:
    """The iso-surface of a :class:`SurfaceConstraint`, in voxel units ``u`` of the
    stress grid.

    On a shared grid one gather of the 8 surrounding element centres serves the masked
    stress (the 7 channels of the :class:`StressField` table) and the surface field (an
    8th channel); a field on its own grid is gathered separately (its voxel coords are
    ``v = u * scale + offset``, gradients are converted to stress-grid voxels). The field
    is padded like the stress table, but by repeating its last layer, so its gradient
    vanishes beyond the last centres instead of reading the padding.
    """

    def __init__(self, field: StressField, constraint: SurfaceConstraint):
        g = field.grid
        fg = g if constraint.grid is None else constraint.grid
        f = np.asarray(constraint.field, dtype=np.float64)
        if f.size != fg.n_elems:
            raise ValueError(
                "surface field must have the shape of its grid (the StressField grid "
                "unless SurfaceConstraint.grid is given)"
            )
        f = f.reshape(fg.shape)
        self.f = f
        self.iso = float(constraint.iso)
        self.tolerance = float(constraint.tolerance)
        self.smoothing = float(constraint.smoothing)
        nx, ny, nz = fg.shape
        padded = np.pad(f, ((0, 1), (0, 1), (0, 1)), mode="edge")
        self.f_flat = np.ascontiguousarray(padded.ravel())
        sy = nz + 1
        sx = (ny + 1) * sy
        self.f_strides = np.array([sx, sy, 1], dtype=np.int64)
        self.f_corners = np.array(
            [a * sx + b * sy + c for a in (0, 1) for b in (0, 1) for c in (0, 1)],
            dtype=np.int64,
        )
        self.f_centre_max = np.asarray(fg.shape, dtype=np.float64) - 1.0
        self.shared = fg.shape == g.shape and fg.h == g.h and np.array_equal(fg.origin, g.origin)
        self.scale = g.h / fg.h  # field voxels per stress voxel
        self.offset = (g.origin - fg.origin) / fg.h
        self.stress_table = field._table
        self.strides = field._strides
        self.corners = field._corners
        self.centre_max = field._centre_max
        if self.shared:
            self.table = np.concatenate([field._table, self.f_flat[:, None]], axis=1)
        # where the field lives (its centre box), within the stress grid, in u
        lo = (0.5 - self.offset) / self.scale
        hi = (np.asarray(fg.shape, dtype=np.float64) - 0.5 - self.offset) / self.scale
        self.hull_lo = np.maximum(lo, 0.0)
        self.hull_hi = np.minimum(hi, np.asarray(g.shape, dtype=np.float64))
        if self.shared:  # exact bounds, free of rounding
            self.hull_lo = np.full(3, 0.5)
            self.hull_hi = np.asarray(g.shape, dtype=np.float64) - 0.5
        span = float(f.max() - f.min()) if f.size else 0.0
        self.grad_eps = 1e-9 * max(span, abs(self.iso), 1e-300)  # |grad f| per voxel

    def _locate_field(self, u: np.ndarray) -> tuple:
        v = u if self.shared else u * self.scale + self.offset
        q = np.clip(v - 0.5, 0.0, self.f_centre_max)
        i0 = np.floor(q)
        idx = (i0.astype(np.int64) @ self.f_strides)[:, None] + self.f_corners
        return idx, q - i0

    def on_hull(self, u: np.ndarray) -> np.ndarray:
        """``(P,)`` bool: inside the box of the field's element centres (where it lives)
        and inside the stress grid."""
        return np.all((u >= self.hull_lo) & (u <= self.hull_hi), axis=1)

    def sample(self, u: np.ndarray) -> tuple:
        """``(s6, den, value, grad)`` at voxel coords ``(P, 3)``: masked stress (zero where
        no solid element is near), solid weight, field value and gradient (per voxel)."""
        q = np.clip(u - 0.5, 0.0, self.centre_max)
        i0 = np.floor(q)
        t = q - i0
        idx = (i0.astype(np.int64) @ self.strides)[:, None] + self.corners
        if self.shared:
            c = self.table[idx]  # (P, 8 corners, 8 channels)
            acc = np.einsum("pc,pck->pk", _trilinear_weights(t), c[:, :, :7])
            value, grad = _trilinear_value_grad(c[:, :, 7], t)
        else:
            acc = np.einsum("pc,pck->pk", _trilinear_weights(t), self.stress_table[idx])
            value, grad = self.value_grad(u)
        den = acc[:, 6]
        s6 = np.divide(
            acc[:, :6], den[:, None], out=np.zeros((len(u), 6)), where=den[:, None] > 0.0
        )
        return s6, den, value, grad

    def value_grad(self, u: np.ndarray) -> tuple:
        """Field value ``(P,)`` and gradient ``(P, 3)`` (per stress voxel) at ``u``."""
        idx, t = self._locate_field(u)
        value, grad = _trilinear_value_grad(self.f_flat[idx], t)
        if not self.shared:
            grad *= self.scale
        return value, grad

    def newton_delta(self, value: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Damped Newton move ``(f - iso) grad / |grad|^2`` (at most ``_NEWTON_STEP``)."""
        g2 = _rowdot(grad, grad)
        k = np.divide(
            value - self.iso, g2, out=np.zeros_like(g2), where=g2 > self.grad_eps ** 2
        )
        delta = k[:, None] * grad
        length = np.sqrt(_rowdot(delta, delta))
        scale = np.minimum(1.0, _NEWTON_STEP / np.maximum(length, 1e-300))
        return delta * scale[:, None]

    def residual(self, value: np.ndarray, grad: np.ndarray) -> tuple:
        """``(distance, |grad|)``: first-order distance (voxels) to the surface."""
        gn = np.sqrt(_rowdot(grad, grad))
        dist = np.abs(value - self.iso) / np.maximum(gn, self.grad_eps)
        return np.where(gn > self.grad_eps, dist, np.inf), gn

    def project(self, u: np.ndarray, max_move: float, iterations: int) -> tuple:
        """Newton iterations onto the surface: ``(points, ok)``.

        ``ok``: within ``_PROJ_EPS`` of the surface, inside the field's centre box and
        moved at most ``max_move`` voxels.
        """
        start = np.asarray(u, dtype=np.float64).reshape(-1, 3)
        p = start.copy()
        if not len(p):
            return p, np.zeros(0, dtype=bool)
        for _ in range(int(iterations)):
            value, grad = self.value_grad(p)
            dist, _gn = self.residual(value, grad)
            todo = dist > 0.1 * _PROJ_EPS
            if not todo.any():
                break
            p[todo] -= self.newton_delta(value[todo], grad[todo])
        value, grad = self.value_grad(p)
        dist, _gn = self.residual(value, grad)
        move = p - start
        ok = (dist <= _PROJ_EPS) & self.on_hull(p)
        ok &= _rowdot(move, move) <= float(max_move) ** 2
        return p, ok

    def settle(self, x: np.ndarray) -> tuple:
        """Bring the ends of integration steps back onto the surface (1-3 Newton steps).

        Returns ``(points, s6, den, grad, ok)`` with the masked stress, solid weight and
        field gradient at the projected points; ``ok``: on the surface, inside the
        field's centre box and moved at most ``tolerance`` voxels.
        """
        value, grad = self.value_grad(x)
        p = x - self.newton_delta(value, grad)
        s6, den, value, grad = self.sample(p)
        for _ in range(2):
            dist, gn = self.residual(value, grad)
            bad = (dist > _PROJ_EPS) & (gn > self.grad_eps)
            if not bad.any():
                break
            p[bad] -= self.newton_delta(value[bad], grad[bad])
            s6[bad], den[bad], value[bad], grad[bad] = self.sample(p[bad])
        dist, gn = self.residual(value, grad)
        move = p - x
        ok = (dist <= _PROJ_EPS) & self.on_hull(p)
        ok &= _rowdot(move, move) <= self.tolerance ** 2
        return p, s6, den, grad, ok

    def crossings(self) -> np.ndarray:
        """``(P, 3)`` voxel coords ``u`` where the field crosses the iso level on the
        segments between neighbouring element centres of its grid: near-surface element
        centres moved onto the surface along the grid axis (exact: the trilinear field is
        linear there)."""
        f, iso = self.f, self.iso
        out = []
        for axis in range(3):
            if f.shape[axis] < 2:
                continue
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, -1)
            hi[axis] = slice(1, None)
            fa, fb = f[tuple(lo)], f[tuple(hi)]
            where = np.nonzero((fa >= iso) != (fb >= iso))
            a, b = fa[where], fb[where]
            p = np.stack(where, axis=1).astype(np.float64) + 0.5
            p[:, axis] += (iso - a) / (b - a)
            out.append(p)
        v = np.concatenate(out) if out else np.zeros((0, 3))
        return v if self.shared else (v - self.offset) / self.scale


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
        start = np.repeat(lo[nz] - ends + counts, counts)
        stored = np.arange(int(ends[-1]), dtype=np.int64) + start
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

    def __init__(self, seeds, fams, val0, vec0, uid0: int, max_points: int, nrm0=None):
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
        # surface normals at the heads (surface mode only)
        self.normal = None if nrm0 is None else np.concatenate([nrm0, nrm0]).astype(np.float64)
        self.max_points = int(max_points)
        cap = min(self.max_points, 128)
        self.trail_u = np.zeros((n, cap, 3), dtype=np.float64)
        self.trail_v = np.zeros((n, cap), dtype=np.float64)
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
        trail_u = np.zeros((n, new_cap, 3), dtype=np.float64)
        trail_v = np.zeros((n, new_cap), dtype=np.float64)
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
        # surface filling (Jobard-Lefer): pending side seeds and random seeds per family
        self.filling = False
        self.fresh = []  # (line id, seed index) of lines accepted in the current batch
        self.side_u = [np.zeros((0, 3)) for _ in FAMILIES]
        self.side_group = [np.zeros(0, dtype=np.int64) for _ in FAMILIES]
        self.random_u = [np.zeros((0, 3)) for _ in FAMILIES]

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
        if st.surface is None:
            self.surface = None
            self.sep_seed = self.sep_stop = self.sep
        else:
            if not isinstance(st.surface, SurfaceConstraint):
                raise TypeError("surface must be a SurfaceConstraint")
            self.surface = _Surface(self.field, st.surface)
            self.sep_seed = SURFACE_SEED_RATIO * self.sep
            self.sep_stop = SURFACE_STOP_RATIO * self.sep
        step = float(st.step)
        if not (math.isfinite(step) and step > 0.0):
            raise ValueError("step must be a positive number of voxels")
        self.s = min(step, 0.5 * self.sep_stop)  # keeps the separation test exact to a few %
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
        self.close_dist = max(2.5 * s, 0.5 * self.sep_stop)  # head-to-head distance closing a loop
        self.close_len = max(math.pi * self.sep_stop, 4.0 * self.close_dist)  # shortest loop
        self.lag_self = max(2, int(math.ceil(0.5 * math.pi * self.sep_stop / s)))
        self.max_degenerate = max(1, int(math.ceil(_DEGENERATE_LENGTH / s)))
        self.min_length = _MIN_LENGTH * self.sep if self.surface is not None else 0.0
        # binomial [1, 2, 1] passes with a standard deviation of ~``smoothing`` voxels
        sigma = 0.0 if self.surface is None else self.surface.smoothing
        self.smooth_passes = min(_MAX_SMOOTH_PASSES, int(round(2.0 * (sigma / s) ** 2)))

    def _init_stamps(self) -> None:
        """Visit stamps: cells (>= ``separation``) remember the last half-line inside them.

        They detect cheaply when a half may be coming back near its own line (see
        :meth:`_self_approach`). Stamps are only a filter: an exact distance test
        decides, so a stamp overwritten by another line merely delays the stop (the
        exact self-clipping at acceptance stays authoritative).
        """
        cell = self.sep
        dims = np.ceil(self.shape / cell).astype(np.int64) + 2  # + one margin cell per side
        while int(np.prod(dims)) > _MAX_STAMP_CELLS:
            cell *= 1.25
            dims = np.ceil(self.shape / cell).astype(np.int64) + 2
        self.stamp_cell = cell
        mult = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
        self.stamp_mult = mult
        self.stamp_dims = dims
        self.stamp_nb = np.array(
            [dx * mult[0] + dy * mult[1] + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1)
             for dz in (-1, 0, 1)],
            dtype=np.int64,
        )
        n = int(np.prod(dims))
        self.stamp_uid = np.full(n, -1, dtype=np.int64)
        self.stamp_time = np.zeros(n, dtype=np.int64)

    # -------------------------------------------------------------- driver
    def run(self) -> list:
        self._report(0.0)
        if self.max_lines == 0 or not self.order or not self.field.solid.any():
            self._report(1.0)
            return []
        if self.surface is None:
            self._references()
        else:
            self._surface_references()
        if not any(self.active):
            self._report(1.0)
            return []
        primary_end = 0.6 if self.depth > 0 else 1.0
        if self.surface is None:
            u, fams = self._primary_seeds()
            self._process(u, fams, 0, 0.0, primary_end)
        else:
            self._fill_surface(0.0, primary_end)
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

    def _set_references(self, mags: np.ndarray) -> None:
        """Per-family stop thresholds from the 99th percentile of ``mags`` ``(2, n)``."""
        refs = np.array(
            [float(np.percentile(m, _REF_PERCENTILE)) if m.size else 0.0 for m in mags]
        )
        top = float(refs.max())
        # a family carrying (almost) no stress at all must not trace numerical noise
        refs = np.maximum(refs, _REF_FLOOR * top)
        self.threshold = self.min_stress * refs
        self.active = [top > 0.0 and f in self.order for f in range(len(FAMILIES))]

    def _references(self) -> None:
        """Volume mode: references from the principal stresses of the solid elements."""
        self.solid_ids, self.solid_mags = _field_magnitudes(self.field)
        self._set_references(self.solid_mags)

    def _surface_references(self) -> None:
        """Surface mode: the iso crossings between neighbouring element centres are the
        samples of the references and of the random seeds."""
        surf = self.surface
        u = surf.crossings()
        keep = np.zeros(len(u), dtype=bool)
        vals = np.zeros((len(u), 2))
        for start in range(0, len(u), _SAMPLE_CHUNK):
            sl = slice(start, start + _SAMPLE_CHUNK)
            s6, den, _value, grad = surf.sample(u[sl])
            n, gn = _unit(grad, surf.grad_eps)
            hi, lo, _e_hi, _e_lo, _well = _tangent_eigen(s6, n)
            keep[sl] = surf.on_hull(u[sl]) & (den > 0.0) & (gn > surf.grad_eps)
            vals[sl] = np.stack([hi, lo], axis=1)
        self.samples_u = u[keep]
        elems = np.clip(np.floor(self.samples_u).astype(np.int64), 0, self.shape - 1)
        self.samples_elem = np.ravel_multi_index(tuple(elems.T), self.field.grid.shape)
        self.samples_mag = np.abs(vals[keep]).T
        self._set_references(self.samples_mag)

    # ------------------------------------------------------------ evaluation
    def _family_values(self, u: np.ndarray) -> np.ndarray:
        """Signed stress ``(P, 2)`` of both families (TENSION, COMPRESSION) at ``u``."""
        if self.surface is None:
            values = _principal_values(self.field._sample_voigt(u))
            return np.stack([values[:, 2], values[:, 0]], axis=1)
        s6, _den, _value, grad = self.surface.sample(u)
        n, _gn = _unit(grad, self.surface.grad_eps)
        hi, lo, _e_hi, _e_lo, _well = _tangent_eigen(s6, n)
        return np.stack([hi, lo], axis=1)

    def _surface_frame(self, u: np.ndarray, cols: np.ndarray) -> tuple:
        """Surface mode at points ``u``: ``(values, vectors, well, normals, valid)``.

        ``valid``: inside the centre box, near a solid element and on the surface.
        """
        surf = self.surface
        s6, den, value, grad = surf.sample(u)
        n, gn = _unit(grad, surf.grad_eps)
        hi, lo, e_hi, e_lo, well = _tangent_eigen(s6, n)
        top = cols == 2
        val = np.where(top, hi, lo)
        vec = np.where(top[:, None], e_hi, e_lo)
        dist, gn = surf.residual(value, grad)
        valid = surf.on_hull(u) & (den > 0.0) & (dist <= _SEED_EPS)
        return val, vec, well & (gn > surf.grad_eps), n, valid

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

    def _surface_explicit(self) -> np.ndarray:
        """Explicit seeds moved onto the surface (Newton, else the nearest surface sample),
        within ``_SURFACE_REACH`` voxels; farther ones are dropped."""
        u = (self.seed_world - self.grid.origin) / self.grid.h
        if not len(u):
            return np.zeros((0, 3))
        p, ok = self.surface.project(u, _SURFACE_REACH, 12)
        samples = self.samples_u
        for i in np.flatnonzero(~ok):
            if not len(samples):
                break
            d = samples - u[i]
            d2 = _rowdot(d, d)
            j = int(np.argmin(d2))
            if d2[j] <= _SURFACE_REACH ** 2:
                p[i] = samples[j]
                ok[i] = True
        return p[ok]

    def _surface_random_seeds(self, fam: int) -> np.ndarray:
        """``n_seeds`` surface seeds drawn ∝ |sigma_fam| * (1 + 3 branch_weight) among the
        surface samples, jittered within a voxel and moved back onto the surface."""
        mags = self.samples_mag[fam]
        weights = np.where(mags >= self.threshold[fam], mags, 0.0)
        if self.branch_weight is not None:
            weights = weights * (1.0 + 3.0 * self.branch_weight.ravel()[self.samples_elem])
        count = min(self.n_seeds, int(np.count_nonzero(weights)))
        if count == 0:
            return np.zeros((0, 3))
        picks = self.rng.choice(weights.size, size=count, replace=False, p=weights / weights.sum())
        base = self.samples_u[picks]
        jittered, ok = self.surface.project(
            base + self.rng.random((count, 3)) - 0.5, _SAMPLE_REACH, 4
        )
        return np.where(ok[:, None], jittered, base)

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
        values = self._family_values(u)
        rows = np.arange(len(u))
        this = np.abs(values[rows, this_fam])
        other = np.abs(values[rows, other_fam])
        ratio = np.full(len(u), self.ratio)
        if self.branch_weight is not None:
            e = np.clip(np.floor(u).astype(np.int64), 0, self.shape - 1)
            ratio = ratio * (1.0 - 0.5 * self.branch_weight[e[:, 0], e[:, 1], e[:, 2]])
        ok = (other >= ratio * this) & (other >= self.threshold[other_fam]) & (other > 0.0)
        order = np.argsort(-other[ok], kind="stable")
        return u[ok][order], other_fam[ok][order]

    # ----------------------------------------------------- surface filling
    def _fill_surface(self, lo: float, hi: float) -> None:
        """Generation 0 of the surface mode: explicit seeds, then evenly spaced fronts."""
        order = [f for f in self.order if self.active[f]]
        self.filling = True
        explicit = self._surface_explicit()
        if len(explicit):
            u = np.repeat(explicit, len(order), axis=0)
            fams = np.tile(np.asarray(order, dtype=np.int64), len(explicit))
            self._process(u, fams, 0, lo, lo)
        for f in order:
            self.random_u[f] = self._surface_random_seeds(f)
        n_cov = min(_COVERAGE_SAMPLES, len(self.samples_u))
        self.coverage_ids = np.unique(
            np.linspace(0, len(self.samples_u) - 1, n_cov).astype(np.int64)
        ) if n_cov else np.zeros(0, dtype=np.int64)
        while len(self.lines) < self.max_lines:
            check_stop(self.should_stop)
            u, fams = self._fill_round(order)
            if not len(u):
                break
            self._trace_seeds(u, fams, 0)
            self._report(lo + (hi - lo) * self._coverage(order))
        self.filling = False

    def _fill_round(self, order: list) -> tuple:
        """Seeds of the next filling round, per family: side seeds of accepted lines
        (Jobard-Lefer: the first admissible one of every line side and level, mutually
        spread), plus random seeds starting new fronts. While a family's fronts are
        moving, new fronts start only far from its lines and pending side seeds; once
        they have stopped, random seeds fill the remaining gaps."""
        cap = max(1, _BATCH_SEEDS // max(len(order), 1))
        out_u, out_f = [], []
        for fam in order:
            index = self.index[fam]
            q, groups = self.side_u[fam], self.side_group[fam]
            if len(q) and index.size:
                keep = ~index.near(q, self.sep_seed)
                q, groups = q[keep], groups[keep]
            chosen = np.zeros((0, 3))
            if len(q):
                _uniq, first = np.unique(groups, return_index=True)
                first = np.sort(first)
                pick = first[_spread(q[first], self.sep_seed, cap)]
                chosen = q[pick]
                rest = np.ones(len(q), dtype=bool)
                rest[pick] = False
                q, groups = q[rest], groups[rest]
            self.side_u[fam], self.side_group[fam] = q, groups

            r = self.random_u[fam]
            if len(r) and index.size:
                r = r[~index.near(r, self.sep_seed)]
            room = cap - len(chosen)
            if len(r) and room > 0:
                if len(q) or len(chosen):  # fronts are moving: new ones far away only
                    cand = np.flatnonzero(self._far_mask(fam, r, np.concatenate([q, chosen])))
                    radius, count = _FAR_CELLS * self.sep, _FRONTS
                elif not index.size:  # the very first fronts
                    cand = np.arange(len(r))
                    radius, count = _FAR_CELLS * self.sep, _FIRST_FRONTS
                else:  # the fronts have stopped: fill the gaps
                    cand = np.arange(len(r))
                    radius, count = _GAP_SPREAD * self.sep, _FRONTS
                take = cand[_spread(r[cand], radius, min(room, count))]
                if take.size:
                    chosen = np.concatenate([chosen, r[take]])
                    rest = np.ones(len(r), dtype=bool)
                    rest[take] = False
                    r = r[rest]
            self.random_u[fam] = r
            out_u.append(chosen)
            out_f.append(np.full(len(chosen), fam, dtype=np.int64))
        return np.concatenate(out_u), np.concatenate(out_f)

    def _far_mask(self, fam: int, u: np.ndarray, pending: np.ndarray) -> np.ndarray:
        """``(P,)`` bool: ``u`` lies at least ~``_FAR_CELLS`` cells of ``separation`` away
        from every accepted line of ``fam`` and every ``pending`` seed."""
        index = self.index[fam]
        occ = np.zeros(int(np.prod(index.dims)), dtype=bool)
        if index.size:
            occ[index.keys] = True
        if len(pending):
            occ[index._keys(pending)] = True
        occ = _dilate_box(occ.reshape(tuple(int(d) for d in index.dims)), _FAR_CELLS)
        return ~occ.ravel()[index._keys(u)]

    def _coverage(self, order: list) -> float:
        """Share of the (stressed) surface samples within ``separation`` of a line."""
        shares = []
        for fam in order:
            ids = self.coverage_ids[self.samples_mag[fam][self.coverage_ids] >= self.threshold[fam]]
            if ids.size:
                shares.append(float(np.mean(self.index[fam].near(self.samples_u[ids], self.sep))))
        return float(np.mean(shares)) if shares else 1.0

    def _queue_sides(self, fresh: list) -> None:
        """Side seeds of newly accepted lines (Jobard-Lefer).

        From points every ``_SIDE_SPACING * separation`` along each line (nearest to
        its middle first; for loops, to its seed) the surface is walked perpendicular to
        the line, to both sides, in steps of ``_MARCH_STEP`` (each projected onto the
        surface, the direction carried along with the normal); seeds are taken after
        1, 2, ... ``_SIDE_LEVELS`` separations. Several levels let a front advance by
        several lines per round; the lines they start are the ones the sequential
        algorithm would trace next where neighbours run parallel, and are clipped like
        any other where they do not.
        """
        surf = self.surface
        spacing = _SIDE_SPACING * self.sep
        pts, tans, owner = [], [], []
        for line_id, k0 in fresh:
            u = self.line_u[line_id]
            closed = self.lines[line_id].closed
            uu = np.concatenate([u, u[:1]]) if closed else u
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(uu, axis=0), axis=1))])
            total = float(cum[-1])
            if total <= 1e-9:
                continue
            if closed:
                m = max(1, int(round(total / spacing)))
                j = np.arange(m)
                at = (float(cum[k0]) + j[np.lexsort((j, np.minimum(j, m - j)))] * (total / m))
                at %= total
                lo_at = (at - 0.25 * spacing) % total
                hi_at = (at + 0.25 * spacing) % total
            else:
                mid = 0.5 * total
                j = np.arange(-int(mid // spacing), int(mid // spacing) + 1)
                at = mid + j[np.lexsort((j, np.abs(j)))] * spacing
                lo_at = np.maximum(at - 0.25 * spacing, 0.0)
                hi_at = np.minimum(at + 0.25 * spacing, total)
            pts.append(np.stack([np.interp(at, cum, uu[:, a]) for a in range(3)], axis=1))
            tans.append(
                np.stack([np.interp(hi_at, cum, uu[:, a]) - np.interp(lo_at, cum, uu[:, a])
                          for a in range(3)], axis=1)
            )
            owner.append(np.full(len(at), line_id, dtype=np.int64))
        if not pts:
            return
        p = np.concatenate(pts)
        owner = np.concatenate(owner)
        _value, grad = surf.value_grad(p)
        n, gn = _unit(grad, surf.grad_eps)
        t = np.concatenate(tans)
        t, tn = _unit(t - _rowdot(t, n)[:, None] * n, 1e-9)
        good = (gn > surf.grad_eps) & (tn > 1e-9)
        b = _cross(n, t)
        # walks: (point, side) pairs, side 0 = +b, side 1 = -b
        c = np.repeat(p, 2, axis=0)
        d = np.stack([b, -b], axis=1).reshape(-1, 3)
        nn = np.repeat(n, 2, axis=0)
        ok = np.repeat(good, 2)
        per_level = max(1, int(math.ceil(self.sep / _MARCH_STEP)))
        step = self.sep / per_level
        levels = []
        for _level in range(_SIDE_LEVELS):
            for _ in range(per_level):
                c, moved = surf.project(c + step * d, 2.0 * step, 3)
                ok &= moved
                _value, grad = surf.value_grad(c)
                n_new, gn = _unit(grad, surf.grad_eps)
                ok &= gn > surf.grad_eps
                d = _transport(d, nn, n_new)
                nn = n_new
            levels.append((c.copy(), ok.copy()))
            if not ok.any():
                break
        n_lev = len(levels)
        # order: point (line by line, middle first), side, level
        cand = np.stack([lv[0] for lv in levels], axis=1).reshape(-1, 3)
        valid = np.stack([lv[1] for lv in levels], axis=1).ravel()
        walk = np.repeat(np.arange(len(c)), n_lev)
        level = np.tile(np.arange(n_lev), len(c))
        line = owner[walk // 2]
        group = (2 * line + walk % 2) * _SIDE_LEVELS + level
        fams = np.asarray(self.line_fam, dtype=np.int64)[line]
        cand, group, fams = cand[valid], group[valid], fams[valid]
        if not len(cand):
            return
        cols = _FAMILY_COLUMN[fams]
        val, _vec, well, _n, on = self._surface_frame(cand, cols)
        ok = on & well & (np.abs(val) >= self.threshold[fams]) & (np.abs(val) > 0.0)
        for fam in range(len(FAMILIES)):
            m = ok & (fams == fam)
            if m.any():
                self.side_u[fam] = np.concatenate([self.side_u[fam], cand[m]])
                self.side_group[fam] = np.concatenate([self.side_group[fam], group[m]])

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
            ok, val0, vec0, nrm0 = self._admissible(seeds[pos:end], fams[pos:end])
            sel = np.flatnonzero(ok)
            nxt = end
            if sel.size > batch:
                sel = sel[:batch]
                nxt = pos + int(sel[-1]) + 1
            if sel.size:
                self._trace_batch(
                    seeds[pos + sel],
                    fams[pos + sel],
                    val0[sel],
                    vec0[sel],
                    None if nrm0 is None else nrm0[sel],
                    gen,
                )
            pos = nxt
            self._report(lo + (hi - lo) * pos / max(total, 1))

    def _trace_seeds(self, seeds: np.ndarray, fams: np.ndarray, gen: int) -> None:
        """Trace the admissible ones of ``seeds`` as one batch."""
        ok, val0, vec0, nrm0 = self._admissible(seeds, fams)
        sel = np.flatnonzero(ok)
        if sel.size:
            self._trace_batch(
                seeds[sel], fams[sel], val0[sel], vec0[sel],
                None if nrm0 is None else nrm0[sel], gen,
            )

    def _trace_batch(self, seeds, fams, val0, vec0, nrm0, gen: int) -> None:
        """Integrate admissible seeds together, accept their lines in order (surface
        filling: and queue the side seeds of the new lines)."""
        b = _Batch(seeds, fams, val0, vec0, self._next_uid, self.max_steps + 1, nrm0)
        self._next_uid += 2 * b.K
        if self.surface is None:
            self._integrate(b)
        else:
            self._integrate_surface(b)
        self.fresh = []
        self._accept(b, gen)
        if self.filling and self.fresh:
            self._queue_sides(self.fresh)
        self.fresh = []

    def _admissible(self, u: np.ndarray, fams: np.ndarray) -> tuple:
        """Seeds that may start a line now: inside, stressed, non-degenerate, separated.

        Returns ``(ok, values, vectors, normals)`` (``normals`` is None in volume mode).
        """
        cols = _FAMILY_COLUMN[fams]
        if self.surface is None:
            values, vec, well = _principal_frame(self.field._sample_voigt(u), cols)
            val = values[np.arange(len(u)), cols]
            inside = self.field._inside_voxels(u)
            nrm = None
        else:
            val, vec, well, nrm, inside = self._surface_frame(u, cols)
        active = np.asarray(self.active, dtype=bool)[fams]
        mag = np.abs(val)
        ok = inside & well & active & (mag >= self.threshold[fams])
        ok &= mag > 0.0
        for fam in range(len(FAMILIES)):
            m = ok & (fams == fam)
            if self.index[fam].size and m.any():
                ok[m] = ~self.index[fam].near(u[m], self.sep_seed)
        return ok, val, vec, nrm

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
        sep = self.sep_stop
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
            stop |= self._self_approach(b, ia, x_new, t_new, ~stop)
            self._record(b, ia, stop, x_new, val, new_dir, degenerate, t_new)

    def _record(self, b: _Batch, ia, stop, x_new, val, new_dir, degenerate, t_new,
                normal=None) -> None:
        """Append the new points of the halves that go on; stop the others."""
        rec = ~stop
        ir = ia[rec]
        tr = t_new[rec]
        b.trail_u[ir, tr] = x_new[rec]
        b.trail_v[ir, tr] = val[rec]
        b.n_pts[ir] += 1
        b.pos[ir] = x_new[rec]
        b.dirn[ir] = new_dir[rec]
        b.degenerate[ir] = degenerate[rec]
        if normal is not None:
            b.normal[ir] = normal[rec]
        b.alive[ia[stop]] = False
        if ir.size:
            self._close_loops(b, ir)
            b.alive[ir[b.n_pts[ir] > self.max_steps]] = False

    def _surface_direction(self, x, cols, ref) -> np.ndarray:
        """Surface mode, one integration stage: family direction in the tangent plane of
        the level set through ``x``, oriented like ``ref`` (``ref`` where undefined).

        Stages stay within one step of the current point, where the normal turns
        little, so ``ref`` is compared as it is (the end of the step carries it along
        the normal properly, see :meth:`_integrate_surface`)."""
        surf = self.surface
        s6, _den, _value, grad = surf.sample(x)
        n, gn = _unit(grad, surf.grad_eps)
        hi, lo, e_hi, e_lo, well = _tangent_eigen(s6, n)
        vec = np.where((cols == 2)[:, None], e_hi, e_lo)
        dot = _rowdot(vec, ref)
        vec = np.where((dot < 0.0)[:, None], -vec, vec)
        good = well & (gn > surf.grad_eps) & (np.abs(dot) >= _COS_MAX_TURN)
        return np.where(good[:, None], vec, ref)

    def _surface_increment(self, x, d, n0, cols) -> np.ndarray:
        """Surface mode: unit RK4 / RK2 direction of the next step (stages off the surface
        use the tangent plane of the level set through them)."""
        half = 0.5 * self.s
        k2 = self._surface_direction(x + half * d, cols, d)
        if self.rk4:
            k3 = self._surface_direction(x + half * k2, cols, d)
            k4 = self._surface_direction(x + self.s * k3, cols, d)
            inc = d + 2.0 * (k2 + k3) + k4
        else:
            inc = k2
        inc -= _rowdot(inc, n0)[:, None] * n0
        out, norm = _unit(inc, 1e-12)
        return np.where((norm > 1e-12)[:, None], out, d)

    def _integrate_surface(self, b: _Batch) -> None:
        """Surface mode: advance every half-line, projecting each step onto the surface."""
        surf = self.surface
        sep = self.sep_stop
        while True:
            ia = np.flatnonzero(b.alive)
            if not ia.size:
                return
            check_stop(self.should_stop)
            b.ensure_capacity(int(b.n_pts[ia].max()) + 1)
            x = b.pos[ia]
            d = b.dirn[ia]
            n0 = b.normal[ia]
            cols = b.col_h[ia]
            fam = b.fam_h[ia]
            inc = self._surface_increment(x, d, n0, cols)
            x_new, s6, den, grad, on = surf.settle(x + self.s * inc)

            n_new, gn = _unit(grad, surf.grad_eps)
            n_new = np.where((gn > surf.grad_eps)[:, None], n_new, n0)
            hi, lo, e_hi, e_lo, well = _tangent_eigen(s6, n_new)
            top = cols == 2
            val = np.where(top, hi, lo)
            vec = np.where(top[:, None], e_hi, e_lo)
            ref = _transport(inc, n0, n_new)
            dot = _rowdot(vec, ref)
            vec = np.where((dot < 0.0)[:, None], -vec, vec)
            good = well & (np.abs(dot) >= _COS_MAX_TURN)
            new_dir = np.where(good[:, None], vec, ref)  # degenerate: keep going straight
            degenerate = np.where(good, 0, b.degenerate[ia] + 1)

            stop = ~on | ~(den > 0.0)
            stop |= np.abs(val) < self.threshold[fam]
            stop |= degenerate > self.max_degenerate
            for f in range(len(FAMILIES)):
                if self.index[f].size:
                    m = ~stop & (fam == f)
                    if m.any():
                        stop[m] = self.index[f].near(x_new[m], sep)
            t_new = b.n_pts[ia]
            stop |= self._self_approach(b, ia, x_new, t_new, ~stop)
            self._record(b, ia, stop, x_new, val, new_dir, degenerate, t_new, n_new)

    def _self_approach(self, b: _Batch, ia, x_new, t_new, candidates) -> np.ndarray:
        """Halves whose new point comes closer than the stop distance to their own line.

        Counted are the points of the same half at least ``lag_self`` steps back, and
        the points of the other half that it left at least ``lag_self`` steps ago (its
        trail, not the region around its head, where closing loops meet). Candidates
        are filtered first (old stamps of either half in the 27 surrounding cells, or a
        chord over the last ``lag_self`` steps shorter than the stop distance), then
        tested exactly against the stored trails.
        """
        found = np.zeros(ia.size, dtype=bool)
        if not candidates.any():
            return found
        lag = self.lag_self
        r2 = self.sep_stop * self.sep_stop
        uid = b.uid[ia]
        partner = b.partner[ia]
        n_partner = b.n_pts[partner]
        c = np.floor(x_new / self.stamp_cell).astype(np.int64) + 1
        np.clip(c, 1, self.stamp_dims - 2, out=c)
        cell = c @ self.stamp_mult
        around = cell[:, None] + self.stamp_nb[None, :]
        owner = self.stamp_uid[around]
        age = self.stamp_time[around]
        flag = (owner == uid[:, None]) & (t_new[:, None] - age >= lag)
        flag |= (owner == b.uid[partner][:, None]) & ((n_partner - 1)[:, None] - age >= lag)
        flag = flag.any(axis=1)
        back = t_new - lag
        m = candidates & ~flag & (back >= 0)
        if m.any():
            chord = x_new[m] - b.trail_u[ia[m], back[m]]
            flag[m] = np.einsum("ij,ij->i", chord, chord) < r2
        flag &= candidates
        if flag.any():
            fi = np.flatnonzero(flag)
            halves, others = ia[fi], partner[fi]
            n = int(max(b.n_pts[halves].max(), n_partner[fi].max()))
            steps = np.arange(n)
            x = x_new[fi][:, None, :]
            d_own = b.trail_u[halves, :n] - x
            hit = (np.einsum("fti,fti->ft", d_own, d_own) < r2) & (
                steps[None, :] <= (t_new[fi] - lag)[:, None]
            )
            d_other = b.trail_u[others, :n] - x
            hit |= (np.einsum("fti,fti->ft", d_other, d_other) < r2) & (
                steps[None, :] <= (n_partner[fi] - 1 - lag)[:, None]
            )
            found[fi] = hit.any(axis=1)
        w = candidates & ~found
        self.stamp_uid[cell[w]] = uid[w]
        self.stamp_time[cell[w]] = t_new[w]
        return found

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
            if self.smooth_passes:
                bu, seed, fu = self._smooth_candidate(bu, seed, fu, bool(b.closed[k]))
            keep_f, keep_b = len(fu), len(bu)
            if index.size:
                if index.near(seed[None], self.sep_seed)[0]:
                    continue
                hits = index.near(np.concatenate([fu, bu]), self.sep_stop)
                hf, hb = hits[:keep_f], hits[keep_f:]
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
            if closed:
                u, v, keep_b = _trim_overlap(u, v, keep_b)
            if self.min_length > 0.0:  # surface mode: no stubs, no loops tighter than the spacing
                length = float(np.linalg.norm(np.diff(u, axis=0), axis=1).sum())
                if closed:
                    length += float(np.linalg.norm(u[0] - u[-1]))
                if length < (math.pi if closed else 1.0) * self.min_length:
                    continue
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
            if self.filling:
                self.fresh.append((len(self.lines) - 1, keep_b))

    def _smooth_candidate(self, bu, seed, fu, closed: bool) -> tuple:
        """Surface mode: the candidate line (``bu`` reversed, seed, ``fu``) smoothed with
        ``smooth_passes`` binomial passes (ends fixed; cyclic for loops) and projected
        back onto the surface (points that cannot be keep their place). Returns the
        smoothed ``(bu, seed, fu)``."""
        raw = np.concatenate([bu[::-1], seed[None], fu])
        if len(raw) < 3:
            return bu, seed, fu
        p = raw.copy()
        for _ in range(self.smooth_passes):
            if closed:
                p = 0.5 * p + 0.25 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0))
            else:
                p[1:-1] = 0.5 * p[1:-1] + 0.25 * (p[:-2] + p[2:])
        proj, ok = self.surface.project(p, self.surface.tolerance, 3)
        p = np.where(ok[:, None], proj, raw)
        nb = len(bu)
        return p[:nb][::-1], p[nb], p[nb + 1 :]

    def _self_clip(self, seed: np.ndarray, fu: np.ndarray, bu: np.ndarray) -> tuple:
        """Cut the halves where the line comes within the stop distance of its own
        earlier part.

        Point ``i`` of a half was reached at time ``i + 1`` (both halves advance
        together from the seed at time 0). Two points closer than the stop distance
        whose distance *along the line* is at least a half circle of that diameter
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
        index = _CellIndex(lo, pts.max(axis=0) - lo, self.sep_stop)
        index.insert(pts)
        i, j = index.pairs(self.sep_stop)
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

    With ``settings.surface`` the lines run on that iso-surface (surface mode).
    Returns a list of :class:`Polyline` in acceptance order: primary lines (generation
    0, explicit seeds first) then the branches of each generation; at most
    ``settings.max_lines``. Deterministic for a fixed ``settings.rng_seed``. Raises
    :class:`~.grid.Cancelled` when ``should_stop()`` returns True and ``ValueError``
    for invalid settings.
    """
    if settings is None:
        settings = TraceSettings()
    return _Tracer(field, settings, progress, should_stop).run()
