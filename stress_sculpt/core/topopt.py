"""Shape evolution on the voxel grid: SIMP topology optimization and growth rules.

:class:`Optimizer` lets the design domain of a :class:`~.grid.Problem` evolve under
its loads, one iteration per :meth:`Optimizer.step`. Two modes
(``OptSettings.mode``):

* ``"SIMP"`` ("Optimize") - minimum compliance topology optimization. The design
  variables ``x`` in [0, 1] live on the *design elements* (domain minus passive
  solid / void). The physical density seen by the FE model is

      rho = passives(sym(H(F x)))

  with ``F`` the cone :class:`DensityFilter` (over the domain only), ``H`` the
  smoothed Heaviside projection of Wang, Lazarov & Sigmund (2011)
  ``H(t) = (tanh(beta eta) + tanh(beta (t - eta))) / (tanh(beta eta) + tanh(beta (1 - eta)))``
  (``OptSettings.projection``; identity when off), ``sym`` the mirror averaging of
  the enabled symmetry planes and ``passives`` forcing 1 on passive solid and 0 on
  passive void elements. The stiffness scale is
  ``emin + (1 - emin) * rho**penal`` (:meth:`~.fem.FEModel.stiffness_scale`).
  One iteration = an optimality-criteria (OC) update of ``x`` - multiplicative
  heuristic ``x * sqrt(-dc / (lambda dv))`` clipped to the move limit and to
  [0, 1], with the Lagrange multiplier found by a safeguarded regula falsi so that
  the mean *physical* (projected) density over the domain equals ``volfrac`` - then
  a warm-started FE solve of the new design and the exact adjoint sensitivities
  through the chain above (zone bias multiplies the compliance sensitivity by
  ``1 + b``). Continuation: ``penal`` ramps linearly from ``penal_start`` (convex
  problem at 1) to ``penal`` over the first ``continuation_iters`` iterations
  (at most half of ``max_iter``), then ``beta`` grows geometrically from
  ``beta_start`` to ``beta_max`` over half of the remaining iterations (but at
  most by a factor 1.3 per iteration, so a short budget ends at a milder
  ``beta``); the last iterations run at the final values and converge on crisp
  0/1 members, so the 0.5 iso-surface of the density is a clean mesh even at very
  low material fractions (a few % of a mostly empty box).
* ``"RULES"`` ("Growth Rules") - stress-driven growth and erosion (BESO flavoured
  but continuous): starting from ``initial_density`` (default solid), every
  iteration solves and takes the stress stimulus ``s = vm * rho**((penal - 1) / 2)``
  (``vm``: von Mises of the solid-equivalent stress ``E D B u_e``, i.e. the stress
  the material would carry at full stiffness), filters it, normalizes it by its
  90th percentile over the material (``x > 0.5``), applies the zone bias
  ``s *= 1 + b``, averages it with the stimulus of the previous iteration (BESO
  stabilization) and moves ``x`` by
  ``rate * (2 clip((s - hollow) / (grow - hollow), 0, 1) - 1)`` (``-rate`` below
  ``hollow_threshold``, ``+rate`` above ``grow_threshold``). With ``keep_volume``
  a uniform shift of all design variables (found by root finding) moves the mean
  density towards ``volfrac`` by at most ``rate`` per iteration. The physical
  density is ``passives(sym(H(F x)))`` with the same ``beta`` ramp as SIMP (over
  the first half of the iterations). For solid material the stimulus is its von
  Mises stress; the density weighting is the stress-like square root of the SIMP
  compliance sensitivity and makes partially carved material lose the competition
  for stress. (The bare solid-equivalent stress falls as material is added -
  ``~ x**-penal`` in a member - so grey "fully stressed" states are stable
  fixed points: nearly all-grey fields with ~4x the compliance of SIMP in tests;
  with the weighting and the projection the rules land within ~10 % of SIMP.)

Symmetry: the mirror is the mid-plane of the grid along each enabled axis; the
average of a cell and its mirror is taken only where both lie in the domain. The
design variables are averaged after every update and the physical density is
averaged at the end of the forward chain, so a symmetric problem gives an exactly
(bitwise) symmetric density.

Cost: an iteration is one warm-started FE solve (two in the first step, which also
analyzes the initial design) plus ~0.1 s of filtering, root finding and stress
recovery. Measured on 4 cores for a SUPPORT-mode box of 44 x 44 x 34 = 66k elements
at 5 % material: ~20 MG-PCG iterations to the 1e-4 residual, 3 - 4 s per iteration
(of which ~0.8 s is the multigrid set-up the solver redoes for every new density).

Conventions as in ``core/grid.py``: grid-shaped arrays are ``(nx, ny, nz)`` and
zero outside the domain; per-element FE arrays follow ``FEModel.elems`` (C-order of
the domain). Pure numpy (float64), no global mutable state: independent optimizers
may run in different threads (one optimizer must not be stepped concurrently).
Long phases poll ``should_stop`` (the FE solver polls it every PCG iteration) and
raise :class:`~.grid.Cancelled`; a cancelled step leaves the optimizer at its last
completed iteration.
"""

from dataclasses import dataclass, replace
import math
import time
from typing import Callable, Optional

import numpy as np

from . import fem
from .grid import Problem, StopFn, check_stop

MODES = ("SIMP", "RULES")

# Multiplicative OC updates cannot leave exactly zero: design variables below this
# floor are updated as if they had this value (they may then regrow next to
# material; elements with zero sensitivity still shrink to 0).
_OC_REVIVE = 1e-3
# Grey band used by IterationState.grey.
_GREY_LO, _GREY_HI = 0.1, 0.9
# Largest growth factor of the projection sharpness between two iterations: a short
# iteration budget then ends at a milder beta instead of projecting a rough design
# so hard that thin, still grey members break apart.
_BETA_GROWTH = 1.3


# ----------------------------------------------------------------------- settings
@dataclass
class OptSettings:
    """Parameters of :class:`Optimizer` (fields up to ``initial_density`` = SPEC §4).

    * ``mode`` ``"SIMP"`` | ``"RULES"``; ``volfrac`` target mean physical density
      over the domain elements (passive solid counts 1, passive void 0; >= 1 gives
      the full domain),
    * ``penal`` stiffness penalization (SIMP: final value of the ramp from
      ``penal_start`` over ``continuation_iters`` iterations, capped at half of
      ``max_iter``; RULES: used throughout), ``move`` (SIMP OC move limit),
    * ``projection`` (both modes) with ``beta_start`` -> ``beta_max`` (geometric
      ramp after the penalty ramp over half of the remaining iterations, at most
      x1.3 per iteration) and threshold ``eta``,
    * ``rmin`` density filter radius in voxels (cone weights ``rmin - distance``;
      < 1 means no filtering; >= 1.5 prevents checkerboards; 2 - 3 give rounder,
      thicker members), ``emin`` stiffness of void,
    * ``max_iter`` iterations (updates), ``tol_change`` convergence on the largest
      design change once the continuation is complete (SIMP: once; RULES: three
      times in a row),
    * RULES: ``grow_threshold``, ``hollow_threshold`` (relative stress), ``rate``,
      ``keep_volume``,
    * ``symmetry`` mirror planes (x, y, z), ``solver`` ``"MG"`` | ``"JACOBI"``,
      ``solver_tol`` / ``solver_maxiter`` of the FE solves. Inside the iterations
      the solves use the looser of ``solver_tol`` and ``iter_solver_tol`` (warm
      started; 1e-4 keeps the sensitivities far more accurate than an OC step
      needs); ``iter_solver_tol = None`` uses ``solver_tol`` as given,
    * ``initial_density`` of the design elements: ``None`` = uniform ``volfrac``
      (SIMP, corrected for the passive solid) or 1 (RULES).
    """

    mode: str = "SIMP"
    volfrac: float = 0.3
    penal: float = 3.0
    penal_start: float = 1.0
    continuation_iters: int = 15
    rmin: float = 1.5
    move: float = 0.2
    emin: float = 1e-3
    max_iter: int = 40
    tol_change: float = 0.01
    grow_threshold: float = 0.6
    hollow_threshold: float = 0.25
    rate: float = 0.15
    keep_volume: bool = True
    symmetry: tuple = (False, False, False)
    solver: str = "MG"
    solver_tol: float = 1e-5
    solver_maxiter: int = 4000
    initial_density: Optional[float] = None
    # Additions (crisp members and speed), all optional.
    projection: bool = True
    beta_start: float = 1.0
    beta_max: float = 16.0
    eta: float = 0.5
    iter_solver_tol: Optional[float] = 1e-4


def checked_settings(settings: Optional[OptSettings]) -> OptSettings:
    """Validated copy of ``settings`` (defaults when ``None``); raises ``ValueError``.

    Normalizes the mode to upper case, clips ``volfrac`` / ``initial_density`` to
    [0, 1], turns ``symmetry`` into a tuple of three bools and checks the ranges of
    the numeric parameters.
    """
    s = OptSettings() if settings is None else settings
    mode = str(s.mode).upper()
    if mode not in MODES:
        raise ValueError(f"unknown optimization mode {s.mode!r} (expected one of {MODES})")
    sym = tuple(bool(v) for v in tuple(s.symmetry)[:3])
    if len(sym) != 3:
        raise ValueError("symmetry must have three entries (x, y, z)")
    numbers = {
        "volfrac": s.volfrac,
        "penal": s.penal,
        "penal_start": s.penal_start,
        "rmin": s.rmin,
        "move": s.move,
        "emin": s.emin,
        "tol_change": s.tol_change,
        "grow_threshold": s.grow_threshold,
        "hollow_threshold": s.hollow_threshold,
        "rate": s.rate,
        "solver_tol": s.solver_tol,
        "beta_start": s.beta_start,
        "beta_max": s.beta_max,
        "eta": s.eta,
    }
    for name, value in numbers.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    if not s.penal > 0.0 or not s.penal_start > 0.0:
        raise ValueError("penal and penal_start must be positive")
    if not 0.0 < s.move <= 1.0:
        raise ValueError("move must be in (0, 1]")
    if not 0.0 < s.emin < 1.0:
        raise ValueError("emin must be in (0, 1)")
    if not s.rate > 0.0:
        raise ValueError("rate must be positive")
    if not (s.beta_start > 0.0 and s.beta_max > 0.0):
        raise ValueError("beta_start and beta_max must be positive")
    if not 0.0 < s.eta < 1.0:
        raise ValueError("eta must be in (0, 1)")
    initial = s.initial_density
    if initial is not None:
        initial = float(np.clip(float(initial), 0.0, 1.0))
    iter_tol = s.iter_solver_tol
    if iter_tol is not None:
        iter_tol = max(float(iter_tol), 0.0)
    return replace(
        s,
        mode=mode,
        volfrac=float(np.clip(float(s.volfrac), 0.0, 1.0)),
        penal=float(s.penal),
        penal_start=float(s.penal_start),
        continuation_iters=max(0, int(s.continuation_iters)),
        rmin=max(0.0, float(s.rmin)),
        max_iter=max(1, int(s.max_iter)),
        tol_change=max(0.0, float(s.tol_change)),
        symmetry=sym,
        solver=str(s.solver).upper(),
        solver_tol=max(float(s.solver_tol), 0.0),
        solver_maxiter=max(1, int(s.solver_maxiter)),
        initial_density=initial,
        projection=bool(s.projection),
        beta_max=max(float(s.beta_max), float(s.beta_start)),
        iter_solver_tol=iter_tol,
    )


@dataclass
class IterationState:
    """Snapshot after an iteration (arrays are fresh copies owned by the caller).

    ``density`` is the physical density (filtered, projected, symmetrized,
    passives applied; 0 outside the domain) the FE model analyzed, ``stress`` /
    ``von_mises`` / ``compliance`` belong to that very design. ``volume`` is the
    mean density over the domain elements, ``change`` the largest change of a
    design variable in this iteration, ``solver_iterations`` the PCG iterations of
    the step. Extra fields: ``beta`` (projection sharpness, 0 = no projection),
    ``grey`` (fraction of the material, i.e. of the density sum, held by grey
    elements with 0.1 < density < 0.9), ``solver_residual`` (relative residual of
    the last solve) and ``time`` (seconds spent in the step).
    """

    iteration: int  # 1-based count of completed iterations (design updates)
    density: np.ndarray  # (nx, ny, nz) physical density, 0 outside the domain
    stress: np.ndarray  # (nx, ny, nz, 6) physical stress of the current design [Pa]
    von_mises: np.ndarray  # (nx, ny, nz)
    compliance: float  # forces . u [J]
    volume: float  # mean density over domain elements
    change: float  # max |dx| of the design variables this iteration
    converged: bool
    solver_iterations: int
    penal: float
    beta: float = 0.0
    grey: float = 0.0
    solver_residual: float = 0.0
    time: float = 0.0


# ------------------------------------------------------------------------ filter
def _cone_groups(rmin: float) -> tuple:
    """Offsets of the cone filter grouped by distance.

    Returns ``(radius, centre_weight, groups)``: ``groups`` is a list of
    ``(weight, offsets (m, 3) int)`` for every non-zero distance ``d < rmin``
    with weight ``rmin - d``; ``radius`` is the largest offset component. A radius
    below one voxel keeps only the centre (identity filter).
    """
    if rmin <= 1.0:
        return 0, 1.0, []
    reach = math.ceil(rmin)
    r = np.arange(-reach, reach + 1)
    di, dj, dk = np.meshgrid(r, r, r, indexing="ij")
    d2 = (di * di + dj * dj + dk * dk).ravel()
    offsets = np.stack([di.ravel(), dj.ravel(), dk.ravel()], axis=1)
    groups = []
    radius = 0
    for value in np.unique(d2):
        if value == 0:
            continue
        weight = rmin - math.sqrt(float(value))
        if weight <= 1e-12 * rmin:
            continue
        offs = offsets[d2 == value]
        radius = max(radius, int(np.abs(offs).max()))
        groups.append((weight, offs))
    return radius, float(rmin), groups


class DensityFilter:
    """Linear cone ("hat") density filter restricted to the domain.

    ``forward``: ``y_e = sum_j w_ej x_j / sum_j w_ej`` over the domain neighbours
    ``j`` of a domain element ``e`` with ``w_ej = max(0, rmin - |c_e - c_j|)``
    (distances in voxels between element centres), 0 outside the domain.
    ``backward`` is its exact adjoint (``<F x, y> = <x, F^T y>``), i.e. the chain
    rule for sensitivities. The convolution adds zero-padded shifted views of the
    field (never ``np.roll``: nothing wraps around the grid borders), offsets of
    equal distance are summed before one multiplication; the normalization is
    precomputed. Constant fields stay constant inside the domain (also next to its
    borders and holes). Immutable after construction (thread-safe).
    """

    def __init__(self, domain: np.ndarray, rmin: float):
        domain = np.asarray(domain, dtype=bool)
        if domain.ndim != 3:
            raise ValueError("domain must be a 3-D boolean array")
        self.domain = domain
        self.shape = domain.shape
        self.rmin = max(0.0, float(rmin))
        self.radius, self._w0, self._groups = _cone_groups(self.rmin)
        self._mask = domain.astype(np.float64)
        norm = self._convolve(self._mask)
        self.norm = np.where(domain, norm, 0.0)  # weighted neighbour count (domain only)
        self._norm_safe = np.where(domain, norm, 1.0)
        self._inv_norm = self._mask / self._norm_safe

    @property
    def n_neighbours(self) -> int:
        """Number of cone offsets (including the centre)."""
        return 1 + sum(int(offs.shape[0]) for _w, offs in self._groups)

    def _convolve(self, a: np.ndarray) -> np.ndarray:
        """``W a``: symmetric cone convolution with zero padding (no normalization)."""
        out = a * self._w0
        r = self.radius
        if r == 0:
            return out
        nx, ny, nz = self.shape
        padded = np.zeros((nx + 2 * r, ny + 2 * r, nz + 2 * r), dtype=np.float64)
        padded[r : r + nx, r : r + ny, r : r + nz] = a
        acc = np.empty(self.shape, dtype=np.float64)
        for weight, offs in self._groups:
            for n, (di, dj, dk) in enumerate(offs):
                view = padded[r + di : r + di + nx, r + dj : r + dj + ny, r + dk : r + dk + nz]
                if n == 0:
                    acc[...] = view
                else:
                    acc += view
            acc *= weight
            out += acc
        return out

    def _checked(self, a) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        if a.shape != self.shape:
            raise ValueError(f"field must have the grid shape {self.shape}")
        return a

    def forward(self, x_grid) -> np.ndarray:
        """Filtered field (nx, ny, nz); values of ``x_grid`` outside the domain are ignored."""
        x = self._checked(x_grid)
        out = self._convolve(x * self._mask)
        out /= self._norm_safe  # a division keeps an all-ones field exactly 1
        out *= self._mask
        return out

    def backward(self, dx_grid) -> np.ndarray:
        """Adjoint of :meth:`forward`: gradient w.r.t. ``x`` from a gradient w.r.t. ``F x``."""
        g = self._checked(dx_grid)
        return self._convolve(g * self._inv_norm) * self._mask


# -------------------------------------------------------------------- projection
def _projection_constants(beta: float, eta: float) -> tuple:
    tb = float(np.tanh(beta * eta))
    return tb, tb + float(np.tanh(beta * (1.0 - eta)))


def project(rho_tilde, beta: float, eta: float = 0.5) -> np.ndarray:
    """Smoothed Heaviside ``H(t)`` (maps 0 -> 0, 1 -> 1, ``eta`` stays put); clipped to [0, 1]."""
    tb, denom = _projection_constants(beta, eta)
    out = (tb + np.tanh(beta * (np.asarray(rho_tilde, dtype=np.float64) - eta))) / denom
    return np.clip(out, 0.0, 1.0)


def project_derivative(rho_tilde, beta: float, eta: float = 0.5) -> np.ndarray:
    """Derivative ``dH/dt`` of :func:`project` (always positive)."""
    _tb, denom = _projection_constants(beta, eta)
    t = np.tanh(beta * (np.asarray(rho_tilde, dtype=np.float64) - eta))
    return beta * (1.0 - t * t) / denom


# ------------------------------------------------------------------ root finding
def _bracketed_root(fun: Callable, a: float, b: float, fa: float, fb: float, ftol: float):
    """Root of a continuous monotone ``fun`` with ``fa``, ``fb`` of opposite signs.

    Illinois regula falsi (superlinear) with a bisection step whenever the bracket
    did not halve over two steps. Returns ``(t, f(t))`` of the last evaluation
    (``|f| <= ftol``, or the bracket collapsed to round-off).
    """
    side = 0
    width = abs(b - a)
    slow = 0
    t, ft = a, fa
    for _ in range(200):
        if slow >= 2 or fa == fb:
            t = 0.5 * (a + b)
            slow = 0
        else:
            t = (a * fb - b * fa) / (fb - fa)
            lo, hi = min(a, b), max(a, b)
            if not lo < t < hi:
                t = 0.5 * (a + b)
        ft = fun(t)
        if abs(ft) <= ftol:
            break
        if (ft > 0.0) == (fa > 0.0):
            a, fa = t, ft
            if side == 1:
                fb *= 0.5
            side = 1
        else:
            b, fb = t, ft
            if side == -1:
                fa *= 0.5
            side = -1
        new_width = abs(b - a)
        slow = slow + 1 if new_width > 0.5 * width else 0
        if slow == 0:
            width = new_width
        if new_width <= 1e-13 * max(1.0, abs(a), abs(b)):
            break
    return t, ft


@dataclass
class _Analysis:
    """FE solution of one design (internal)."""

    u: np.ndarray
    scale: np.ndarray  # (n_act,) stiffness scales
    energy: np.ndarray  # (n_act,) unscaled element energies
    compliance: float
    iterations: int
    residual: float


# --------------------------------------------------------------------- optimizer
class Optimizer:
    """Iterative evolution of a problem's domain (see the module docstring).

    Public attributes (read-only by convention): ``problem``, ``settings`` (the
    validated :class:`OptSettings`), ``model`` (:class:`~.fem.FEModel`, also handy
    for a final analysis), ``filter`` (:class:`DensityFilter`), ``x`` (design
    variables on the grid, passives at 1 / 0, 0 outside the domain), ``u``
    (displacement of the last analyzed design, ``(n_dofs,)``), ``iteration``
    (completed iterations) and ``state`` (last :class:`IterationState` or ``None``).

    The first :meth:`step` analyzes the initial design before its update (two FE
    solves), every later step one solve.
    """

    def __init__(self, problem: Problem, settings: Optional[OptSettings] = None):
        s = checked_settings(settings)
        self.problem = problem
        self.settings = s
        self.grid = problem.grid
        self.model = fem.FEModel(problem, solver=s.solver, emin=s.emin)
        domain = np.asarray(problem.domain, dtype=bool)
        self._domain = domain
        self._solid = problem.passive_solid_or_empty()
        self._void = problem.passive_void_or_empty()
        design = domain & ~self._solid & ~self._void
        self._design_idx = np.flatnonzero(design.ravel())
        self._passive = self._solid | self._void
        self._n_dom = int(np.count_nonzero(domain))
        self._n_solid = int(np.count_nonzero(self._solid))
        self.filter = DensityFilter(domain, s.rmin)
        bias = problem.bias_or_zero()
        self._bias_factor = 1.0 + bias.ravel()[self._design_idx]
        self._sym = [
            (axis, domain & np.flip(domain, axis)) for axis in range(3) if s.symmetry[axis]
        ]
        self._max_iter = s.max_iter
        tol = s.solver_tol
        if s.iter_solver_tol is not None:
            tol = max(tol, s.iter_solver_tol)
        self._tol = tol

        # Continuation schedule (in numbers of updates).
        has_design = self._design_idx.size > 0
        if s.mode == "SIMP" and has_design and s.penal_start != s.penal:
            self._n_penal = min(s.continuation_iters, self._max_iter // 2)
        else:
            self._n_penal = 0
        if s.projection and has_design and s.beta_max > s.beta_start:
            budget = int(0.5 * (self._max_iter - self._n_penal) + 0.5)
            ratio = math.log(s.beta_max / s.beta_start) / math.log(_BETA_GROWTH)
            shortest = math.ceil(ratio - 1e-9)  # ramp length with the largest allowed growth
            self._n_beta = max(1, budget, shortest)
        else:
            self._n_beta = 0

        self.x = self._initial_design()
        self.u = np.zeros(self.grid.n_dofs, dtype=np.float64)
        self.iteration = 0
        self.state: Optional[IterationState] = None
        self._pending = None  # sensitivities / growth signal of the last analysis
        self._converged = False
        self._quiet = 0  # consecutive small changes (RULES)

    # ---------------------------------------------------------------- set-up
    def _initial_design(self) -> np.ndarray:
        s = self.settings
        n_design = self._design_idx.size
        if s.initial_density is not None:
            value = s.initial_density
        elif s.mode == "RULES":
            value = 1.0
        elif n_design:
            # uniform design that meets the volume target together with the solids
            value = (s.volfrac * self._n_dom - self._n_solid) / n_design
        else:
            value = 0.0
        x = np.zeros(self.grid.shape, dtype=np.float64)
        x.ravel()[self._design_idx] = float(np.clip(value, 0.0, 1.0))
        x[self._solid] = 1.0
        return x

    def _schedule(self, k: int) -> tuple:
        """``(penal, beta)`` used to analyze the design after ``k`` updates."""
        s = self.settings
        if self._n_penal > 0:
            t = min(1.0, k / self._n_penal)
            penal = s.penal_start + (s.penal - s.penal_start) * t
        else:
            penal = s.penal
        if not s.projection:
            return penal, 0.0
        if self._n_beta <= 0 or k >= self._n_penal + self._n_beta:
            return penal, s.beta_max
        if k <= self._n_penal:
            return penal, s.beta_start
        t = (k - self._n_penal) / self._n_beta
        return penal, s.beta_start * (s.beta_max / s.beta_start) ** t

    def _continuation_done(self, k: int) -> bool:
        return k >= self._n_penal + self._n_beta

    @property
    def done(self) -> bool:
        """True once converged or after ``max_iter`` iterations."""
        return self._converged or self.iteration >= self._max_iter

    # ----------------------------------------------------------- design maps
    def _symmetrize(self, a: np.ndarray, adjoint: bool = False) -> np.ndarray:
        """Mirror averaging of the enabled planes (its own adjoint in reverse order)."""
        planes = reversed(self._sym) if adjoint else self._sym
        for axis, both in planes:
            a = np.where(both, 0.5 * (a + np.flip(a, axis)), a)
        return a

    def _forward(self, x: np.ndarray, beta: float) -> tuple:
        """Physical density and filtered field ``(rho, rho_tilde)`` of design ``x``."""
        rho_t = self.filter.forward(x)
        rho = project(rho_t, beta, self.settings.eta) if beta > 0.0 else rho_t
        rho = self._symmetrize(rho)
        rho = np.where(self._domain, rho, 0.0)
        rho[self._solid] = 1.0
        rho[self._void] = 0.0
        return rho, rho_t

    def _backward(self, g: np.ndarray, rho_t: np.ndarray, beta: float) -> np.ndarray:
        """Chain rule of :meth:`_forward`: gradient w.r.t. ``x`` from one w.r.t. ``rho``."""
        g = np.where(self._passive, 0.0, g)
        g = self._symmetrize(g, adjoint=True)
        if beta > 0.0:
            g = g * project_derivative(rho_t, beta, self.settings.eta)
        return self.filter.backward(g)

    def physical_density(self, x=None, beta: Optional[float] = None) -> np.ndarray:
        """Physical density of design ``x`` (default: the current one).

        ``beta`` defaults to the projection sharpness of the current iteration
        (ignored without projection).
        """
        x = self.x if x is None else np.asarray(x, dtype=np.float64).reshape(self.grid.shape)
        if beta is None:
            beta = self._schedule(self.iteration)[1]
        if not self.settings.projection:
            beta = 0.0
        return self._forward(x, float(beta))[0]

    def _volume(self, rho: np.ndarray) -> float:
        if self._n_dom == 0:
            return 0.0
        return float(rho.sum()) / self._n_dom

    # -------------------------------------------------------------- analysis
    def _analyze(self, rho: np.ndarray, penal: float, should_stop, tol=None) -> _Analysis:
        model = self.model
        s = self.settings
        rho_act = rho.ravel()[model.elems]
        scale = model.stiffness_scale(rho_act, penal)
        u, info = model.solve(
            scale,
            x0=self.u,
            tol=self._tol if tol is None else float(tol),
            maxiter=s.solver_maxiter,
            should_stop=should_stop,
        )
        energy = model.element_energy(u)
        return _Analysis(
            u=u,
            scale=scale,
            energy=energy,
            compliance=float(self.problem.forces @ u),
            iterations=int(info.get("iterations", 0)),
            residual=float(info.get("residual", 0.0)),
        )

    def _simp_sensitivities(self, an: _Analysis, rho, rho_t, penal: float, beta: float):
        """Compliance and volume gradients on the design elements (bias applied to dc)."""
        model = self.model
        rho_act = np.clip(rho.ravel()[model.elems], 0.0, 1.0)
        dc_act = -(1.0 - model.emin) * penal * rho_act ** (penal - 1.0) * an.energy
        dc = self._backward(model.to_grid(dc_act), rho_t, beta)
        dv = self._backward(self._domain / max(self._n_dom, 1), rho_t, beta)
        idx = self._design_idx
        return dc.ravel()[idx] * self._bias_factor, dv.ravel()[idx]

    def compliance_gradient(self, x=None, penal=None, beta=None, tol=1e-10) -> tuple:
        """Compliance ``f . u`` of design ``x`` and its exact gradient w.r.t. ``x`` (grid).

        SIMP chain (filter, projection, symmetry, passives; no zone bias). Defaults:
        the current design and continuation parameters; ``tol`` of the solve. The
        gradient is zero outside the design elements. Meant for checks and tools;
        it does not change the optimizer state.
        """
        x = self.x if x is None else np.asarray(x, dtype=np.float64).reshape(self.grid.shape)
        p_k, b_k = self._schedule(self.iteration)
        penal = p_k if penal is None else float(penal)
        beta = b_k if beta is None else float(beta)
        if not self.settings.projection:
            beta = 0.0
        rho, rho_t = self._forward(x, beta)
        model = self.model
        rho_act = np.clip(rho.ravel()[model.elems], 0.0, 1.0)
        scale = model.stiffness_scale(rho_act, penal)
        u, _info = model.solve(scale, tol=tol, maxiter=self.settings.solver_maxiter)
        energy = model.element_energy(u)
        dc_act = -(1.0 - model.emin) * penal * rho_act ** (penal - 1.0) * energy
        dc = self._backward(model.to_grid(dc_act), rho_t, beta)
        out = np.zeros(self.grid.shape, dtype=np.float64)
        out.ravel()[self._design_idx] = dc.ravel()[self._design_idx]
        return float(self.problem.forces @ u), out

    # --------------------------------------------------------------- updates
    def _design_volume_fn(self, beta: float, should_stop: Optional[StopFn]):
        """``volume(x_design)`` of the physical density for a candidate design."""
        work = self.x.copy()
        flat = work.ravel()
        idx = self._design_idx

        def volume(xd):
            check_stop(should_stop)
            flat[idx] = xd
            return self._volume(self._forward(work, beta)[0])

        return volume

    def _oc_update(self, dc: np.ndarray, dv: np.ndarray, beta: float, should_stop=None):
        """Optimality-criteria update of the design elements (exact volume constraint)."""
        s = self.settings
        x = self.x.ravel()[self._design_idx]
        if x.size == 0:
            return x.copy()
        lo = np.maximum(0.0, x - s.move)
        hi = np.minimum(1.0, x + s.move)
        ratio = np.maximum(-dc, 0.0) / np.maximum(dv, 1e-300)
        ratio[~np.isfinite(ratio)] = 0.0
        r_max = float(ratio.max())
        if not r_max > 0.0:
            return x.copy()  # no load information: keep the design
        ratio /= r_max  # lambda is then O(1) whatever the units
        ratio[ratio < 1e-30] = 0.0  # such elements sit on their lower bound anyway
        # x_new(mu) = clip(a * exp(-mu / 2), lo, hi) with mu = log(lambda), in log space
        pos = ratio > 0.0
        log_a = np.full(x.shape, -np.inf)
        log_a[pos] = np.log(np.maximum(x[pos], _OC_REVIVE)) + 0.5 * np.log(ratio[pos])
        mu_lo = 2.0 * float(np.min(log_a[pos] - np.log(hi[pos]))) - 1e-9  # all at upper bounds
        mu_hi = 2.0 * float(np.max(log_a[pos] - np.log(np.maximum(lo[pos], 1e-12)))) + 1e-9

        def candidate(mu):
            with np.errstate(over="ignore"):
                return np.clip(np.exp(log_a - 0.5 * mu), lo, hi)

        volume = self._design_volume_fn(beta, should_stop)
        target = s.volfrac
        v_lo = volume(candidate(mu_lo))
        if v_lo <= target:
            return candidate(mu_lo)
        v_hi = volume(candidate(mu_hi))
        if v_hi >= target:
            return candidate(mu_hi)
        ftol = 1e-9 + 1e-6 * target
        mu, _f = _bracketed_root(
            lambda m: volume(candidate(m)) - target,
            mu_lo,
            mu_hi,
            v_lo - target,
            v_hi - target,
            ftol,
        )
        return candidate(mu)

    def _rules_signal(self, an: _Analysis, x, rho, penal: float, previous) -> np.ndarray:
        """Growth stimulus on the design elements (normalized, filtered, biased, averaged).

        ``x`` / ``rho`` are the analyzed design and its physical density (the
        ``x > 0.5`` cells are the material whose 90th percentile is the reference),
        ``previous`` the stimulus of the previous iteration (or ``None``).
        """
        model = self.model
        vm = fem.von_mises(model.element_stress(an.u, np.ones(model.n_act)))  # solid equivalent
        rho_act = np.clip(rho.ravel()[model.elems], 0.0, 1.0)
        stimulus = vm * np.sqrt(rho_act ** max(penal - 1.0, 0.0))
        s_f = self._symmetrize(self.filter.forward(model.to_grid(stimulus)))
        material = self._domain & (x > 0.5)
        values = s_f[material] if np.any(material) else s_f[self._domain]
        ref = float(np.percentile(values, 90.0)) if values.size else 0.0
        if not (np.isfinite(ref) and ref > 0.0):
            signal = np.zeros(self._design_idx.size)
        else:
            signal = s_f.ravel()[self._design_idx] / ref
        signal = signal * self._bias_factor
        if previous is not None:
            signal = 0.5 * (signal + previous)
        return signal

    def _rules_update(self, signal: np.ndarray, beta: float, should_stop=None) -> np.ndarray:
        s = self.settings
        x = self.x.ravel()[self._design_idx]
        if x.size == 0:
            return x.copy()
        span = s.grow_threshold - s.hollow_threshold
        if span > 1e-12:
            t = np.clip((signal - s.hollow_threshold) / span, 0.0, 1.0)
        else:
            t = (signal >= s.grow_threshold).astype(np.float64)
        dx = s.rate * (2.0 * t - 1.0)
        if not s.keep_volume:
            return np.clip(x + dx, 0.0, 1.0)
        # The mean density moves towards volfrac by at most ``rate`` per iteration; the
        # uniform shift itself is free, so saturated growth (dx = +rate everywhere the
        # stress is high) can still be thinned out when there is too much material.
        volume = self._design_volume_fn(beta, should_stop)
        current = self._volume(self._forward(self.x, beta)[0])
        target = current + float(np.clip(s.volfrac - current, -s.rate, s.rate))

        def excess(shift):
            return volume(np.clip(x + dx + shift, 0.0, 1.0)) - target

        reach = 1.0 + s.rate  # shifts beyond this clip every design variable to a bound
        f_lo = excess(-reach)
        if f_lo >= 0.0:
            return np.clip(x + dx - reach, 0.0, 1.0)
        f_hi = excess(reach)
        if f_hi <= 0.0:
            return np.clip(x + dx + reach, 0.0, 1.0)
        shift, _f = _bracketed_root(excess, -reach, reach, f_lo, f_hi, 1e-9 + 1e-6 * target)
        return np.clip(x + dx + shift, 0.0, 1.0)

    def _apply_design(self, xd: np.ndarray) -> np.ndarray:
        """Grid design from design-element values: symmetry, then passives."""
        x = self.x.copy()
        x.ravel()[self._design_idx] = xd
        x = self._symmetrize(x)
        x = np.where(self._domain, x, 0.0)
        x[self._solid] = 1.0
        x[self._void] = 0.0
        return x

    # ------------------------------------------------------------------ step
    def _signal(self, an: _Analysis, x, rho, rho_t, penal: float, beta: float):
        """What the next update needs from analysis ``an`` of design ``x``."""
        if self.settings.mode == "SIMP":
            return self._simp_sensitivities(an, rho, rho_t, penal, beta)
        return self._rules_signal(an, x, rho, penal, self._pending)

    def step(self, should_stop: Optional[StopFn] = None) -> IterationState:
        """One design update followed by the analysis of the new design.

        Returns the new :class:`IterationState` (also kept as ``self.state``).
        Raises :class:`~.grid.Cancelled` when ``should_stop()`` returns True; the
        optimizer then stays at its last completed iteration. Stepping past
        :attr:`done` keeps iterating with the final continuation parameters.
        """
        t_start = time.perf_counter()
        check_stop(should_stop)
        s = self.settings
        k = self.iteration + 1
        solver_its = 0
        if self._pending is None:
            # Analysis of the initial design: gives the first update its gradient.
            penal0, beta0 = self._schedule(self.iteration)
            rho0, rho_t0 = self._forward(self.x, beta0)
            an0 = self._analyze(rho0, penal0, should_stop)
            solver_its += an0.iterations
            self._pending = self._signal(an0, self.x, rho0, rho_t0, penal0, beta0)
            self.u = an0.u
        penal, beta = self._schedule(k)
        check_stop(should_stop)
        if s.mode == "SIMP":
            dc, dv = self._pending
            xd = self._oc_update(dc, dv, beta, should_stop)
        else:
            xd = self._rules_update(self._pending, beta, should_stop)
        x_new = self._apply_design(xd)
        idx = self._design_idx
        change = (
            float(np.max(np.abs(x_new.ravel()[idx] - self.x.ravel()[idx]))) if idx.size else 0.0
        )
        check_stop(should_stop)
        rho, rho_t = self._forward(x_new, beta)
        an = self._analyze(rho, penal, should_stop)
        solver_its += an.iterations
        signal = self._signal(an, x_new, rho, rho_t, penal, beta)

        # commit (nothing above modified the optimizer state except the first analysis)
        self.x = x_new
        self.u = an.u
        self.iteration = k
        self._pending = signal
        if idx.size == 0:
            converged = True
        elif s.mode == "SIMP":
            converged = change < s.tol_change and self._continuation_done(k)
        else:
            self._quiet = self._quiet + 1 if change < s.tol_change else 0
            converged = self._quiet >= 3 and self._continuation_done(k)
        self._converged = bool(converged)
        self.state = self._make_state(
            k,
            rho,
            an,
            change,
            self._converged,
            solver_its,
            penal,
            beta,
            time.perf_counter() - t_start,
        )
        return self.state

    def _make_state(self, k, rho, an, change, converged, its, penal, beta, elapsed):
        model = self.model
        stress = model.to_grid(model.element_stress(an.u, an.scale))
        dom_rho = rho[self._domain]
        total = float(dom_rho.sum())
        grey_mask = (dom_rho > _GREY_LO) & (dom_rho < _GREY_HI)
        grey = float(dom_rho[grey_mask].sum()) / total if total > 0.0 else 0.0
        return IterationState(
            iteration=int(k),
            density=rho,  # fresh array, not kept by the optimizer
            stress=stress,
            von_mises=fem.von_mises(stress),
            compliance=an.compliance,
            volume=self._volume(rho),
            change=float(change),
            converged=bool(converged),
            solver_iterations=int(its),
            penal=float(penal),
            beta=float(beta),
            grey=grey,
            solver_residual=an.residual,
            time=float(elapsed),
        )

    def run(
        self,
        callback: Optional[Callable] = None,
        should_stop: Optional[StopFn] = None,
    ) -> IterationState:
        """Iterate until :attr:`done`; returns the last state.

        ``callback(state)`` is called after every iteration; returning ``False``
        (exactly) stops early. Raises :class:`~.grid.Cancelled` on ``should_stop``.
        """
        state = self.state
        while not self.done or state is None:
            state = self.step(should_stop=should_stop)
            if callback is not None and callback(state) is False:
                break
        return state
