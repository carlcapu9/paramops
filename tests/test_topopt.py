"""Tests of the shape evolution core (``stress_sculpt.core.topopt``)."""

import threading

import numpy as np
import pytest

from stress_sculpt.core import fem, topopt
from stress_sculpt.core.grid import Cancelled, Grid, Problem
from stress_sculpt.core.topopt import DensityFilter, OptSettings, Optimizer


# ---------------------------------------------------------------------------- helpers
def node_ids(grid):
    return np.arange(grid.n_nodes).reshape(grid.nodes_shape)


def all_dofs(nodes):
    nodes = np.asarray(nodes).ravel()
    return (3 * nodes[:, None] + np.arange(3)).ravel()


def cantilever(shape=(24, 8, 8), load_z=None, **kwargs):
    """Clamped at x = 0, downward unit load on a line along y at the free end."""
    g = Grid(shape, (0.0, 0.0, 0.0), 1.0)
    nodes = node_ids(g)
    nx, _ny, nz = shape
    kz = nz // 2 if load_z is None else load_z
    f = np.zeros(g.n_dofs)
    tip = nodes[nx, :, kz].ravel()
    f[3 * tip + 2] = -1.0 / tip.size
    return Problem(g, np.ones(shape, dtype=bool), all_dofs(nodes[0]), f, **kwargs)


def mbb(shape=(36, 6, 12), **kwargs):
    """Full MBB beam: rollers at both bottom ends, load at the top centre line."""
    g = Grid(shape, (0.0, 0.0, 0.0), 1.0)
    nodes = node_ids(g)
    nx, _ny, nz = shape
    left = nodes[0, :, 0].ravel()
    right = nodes[nx, :, 0].ravel()
    fixed = np.concatenate([all_dofs(left), 3 * right + 2, 3 * right + 1])
    f = np.zeros(g.n_dofs)
    top = nodes[nx // 2, :, nz].ravel()
    f[3 * top + 2] = -1.0 / top.size
    return Problem(g, np.ones(shape, dtype=bool), fixed, f, **kwargs)


def support_problem(res=22, footings=((0.0, 0.0),), pad=1.6, foot_pad=2.0):
    """SUPPORT-mode like problem: a floating 2 m cube carried from the ground.

    Eight 2000 N loads on the bottom edges of the cube (corners and edge mid-points,
    bottom at z = 3 m) with passive-solid pads of ``pad`` voxels, the cube interior
    passive void, footing pads (half balls of ``foot_pad`` voxels on the ground,
    all dofs of their bottom nodes fixed). The design domain is the whole box from
    the ground to one voxel above the cube bottom, grown by 0.35 H in plan.
    """
    loads = np.array(
        [(x, y, 3.0) for x in (-1.0, 1.0) for y in (-1.0, 1.0)]
        + [(0.0, -1.0, 3.0), (0.0, 1.0, 3.0), (-1.0, 0.0, 3.0), (1.0, 0.0, 3.0)]
    )
    feet = np.array([(fx, fy, 0.0) for fx, fy in footings])
    grow = 0.35 * 3.0
    lo = np.minimum(loads.min(0), feet.min(0)) - [grow, grow, 0.0]
    hi = np.maximum(loads.max(0), feet.max(0)) + [grow, grow, 0.0]
    lo[2], hi[2] = 0.0, 3.0
    hi[2] += float((hi - lo).max()) / res
    g0 = Grid.from_bounds(lo, hi, res)
    g = Grid(g0.shape, (g0.origin[0], g0.origin[1], 0.0), g0.h)
    h = g.h
    centres = g.element_centers()
    void = (np.abs(centres[..., 0]) < 1.0) & (np.abs(centres[..., 1]) < 1.0)
    void &= centres[..., 2] > 3.0
    solid = np.zeros(g.shape, dtype=bool)
    for p in loads:
        solid |= np.linalg.norm(centres - p, axis=-1) <= pad * h
    for p in feet:
        solid |= np.linalg.norm(centres - p, axis=-1) <= foot_pad * h
    coords = g.node_coords()
    forces = np.zeros(g.n_dofs)
    f3 = forces.reshape(-1, 3)
    for p in loads:
        near = np.flatnonzero(np.linalg.norm(coords - p, axis=1) <= pad * h)
        f3[near, 2] -= 2000.0 / near.size
    fixed = []
    for p in feet:
        on = (np.linalg.norm(coords - p, axis=1) <= foot_pad * h) & (coords[:, 2] <= 1e-9)
        fixed.append(all_dofs(np.flatnonzero(on)))
    problem = Problem(
        g,
        np.ones(g.shape, dtype=bool),
        np.concatenate(fixed),
        forces,
        passive_solid=solid,
        passive_void=void & ~solid,
        E=3.5e9,
        nu=0.36,
    )
    return problem, loads, feet


def dense_filter(domain, rmin):
    """Brute-force filter matrix over the domain cells (rows normalized)."""
    idx = np.argwhere(domain)
    d = np.linalg.norm(idx[:, None, :] - idx[None, :, :], axis=-1)
    w = np.maximum(0.0, rmin - d)
    if rmin <= 1.0:
        w = np.eye(idx.shape[0])
    return idx, w / w.sum(axis=1, keepdims=True)


def holey_domain(shape=(9, 7, 8), seed=0):
    rng = np.random.default_rng(seed)
    dom = rng.random(shape) > 0.25
    dom[:, :, 0] = True  # a solid floor so the domain touches every border
    return dom


def run(problem, n_iter=None, **settings):
    opt = Optimizer(problem, OptSettings(**settings))
    states = []
    for _ in range(n_iter if n_iter is not None else opt.settings.max_iter):
        states.append(opt.step())
    return opt, states


# ----------------------------------------------------------------------------- filter
@pytest.mark.parametrize("rmin", [0.8, 1.5, 2.0, 2.6])
def test_filter_matches_brute_force_and_is_adjoint(rmin):
    dom = holey_domain()
    filt = DensityFilter(dom, rmin)
    idx, W = dense_filter(dom, rmin)
    rng = np.random.default_rng(1)
    x = rng.random(dom.shape)
    y = rng.random(dom.shape)
    fx = filt.forward(x)
    fty = filt.backward(y)
    cells = tuple(idx.T)
    np.testing.assert_allclose(fx[cells], W @ x[cells], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(fty[cells], W.T @ y[cells], rtol=1e-12, atol=1e-14)
    assert np.all(fx[~dom] == 0.0) and np.all(fty[~dom] == 0.0)
    lhs = float(np.sum(fx * y))
    rhs = float(np.sum(x * fty))
    assert abs(lhs - rhs) <= 1e-13 * abs(lhs)


@pytest.mark.parametrize("rmin", [1.5, 2.3, 3.0])
def test_filter_preserves_constants_inside_the_domain(rmin):
    dom = holey_domain(shape=(10, 6, 9), seed=4)
    filt = DensityFilter(dom, rmin)
    x = np.full(dom.shape, 0.37)
    x[~dom] = 5.0  # ignored: only domain neighbours count
    out = filt.forward(x)
    np.testing.assert_allclose(out[dom], 0.37, rtol=0, atol=1e-14)
    assert np.all(out[~dom] == 0.0)


def test_filter_never_wraps_around():
    dom = np.ones((7, 6, 5), dtype=bool)
    filt = DensityFilter(dom, 2.5)
    for corner in [(0, 0, 0), (6, 5, 4), (0, 5, 2)]:
        x = np.zeros(dom.shape)
        x[corner] = 1.0
        for out in (filt.forward(x), filt.backward(x)):
            ijk = np.argwhere(out != 0.0)
            dist = np.linalg.norm(ijk - np.asarray(corner), axis=1)
            assert dist.max() < 2.5  # only true neighbours, nothing on the far side
    # an impulse at the low-x border must leave the high-x border untouched
    x = np.zeros(dom.shape)
    x[0] = 1.0
    assert np.all(filt.forward(x)[-3:] == 0.0)


def test_filter_neighbourhood_counts():
    dom = np.ones((5, 5, 5), dtype=bool)
    assert DensityFilter(dom, 0.5).n_neighbours == 1
    assert DensityFilter(dom, 1.0).n_neighbours == 1
    assert DensityFilter(dom, 1.5).n_neighbours == 19  # centre + 6 faces + 12 edges
    assert DensityFilter(dom, 2.0).n_neighbours == 27
    x = np.random.default_rng(0).random(dom.shape)
    np.testing.assert_array_equal(DensityFilter(dom, 0.9).forward(x), x)


# ------------------------------------------------------------------------- projection
def test_projection_maps_bounds_and_has_exact_derivative():
    t = np.linspace(0.0, 1.0, 41)
    for beta in (1.0, 4.0, 8.0, 16.0):
        h = topopt.project(t, beta)
        assert h[0] == 0.0 and h[-1] == 1.0
        assert abs(topopt.project(0.5, beta) - 0.5) < 1e-12
        assert np.all(np.diff(h) > 0.0)
        eps = 1e-6
        fd = (topopt.project(t + eps, beta) - topopt.project(t - eps, beta)) / (2 * eps)
        inner = slice(1, -1)  # clipping makes the one-sided ends differ
        np.testing.assert_allclose(
            topopt.project_derivative(t, beta)[inner], fd[inner], rtol=1e-5, atol=1e-9
        )


# ------------------------------------------------------------------- sensitivities
def _fd_problem():
    problem = cantilever((6, 3, 4), E=7.0, nu=0.3)
    solid = np.zeros(problem.grid.shape, dtype=bool)
    solid[-1, :, 1:3] = True  # pad under the load
    void = np.zeros(problem.grid.shape, dtype=bool)
    void[2, 1, 1] = True
    problem.passive_solid = solid
    problem.passive_void = void
    return problem


@pytest.mark.parametrize(
    "case",
    [
        {"projection": False, "penal": 3.0, "rmin": 1.5},
        {"projection": True, "beta": 4.0, "penal": 3.0, "rmin": 1.5},
        {
            "projection": True,
            "beta": 8.0,
            "penal": 2.0,
            "rmin": 2.2,
            "symmetry": (False, True, True),
        },
    ],
)
def test_compliance_sensitivity_matches_finite_differences(case):
    case = dict(case)
    beta = case.pop("beta", None)
    penal = case.pop("penal")
    opt = Optimizer(_fd_problem(), OptSettings(**case))
    rng = np.random.default_rng(7)
    x = opt.x.copy()
    design = opt.problem.domain & ~opt.problem.passive_solid & ~opt.problem.passive_void
    x[design] = rng.uniform(0.15, 0.85, int(design.sum()))
    c0, grad = opt.compliance_gradient(x, penal=penal, beta=beta, tol=1e-13)
    assert c0 > 0.0 and np.all(grad[~design] == 0.0) and np.all(grad[design] <= 0.0)
    cells = np.argwhere(design)
    eps = 1e-6
    for cell in cells[rng.choice(len(cells), 12, replace=False)]:
        cell = tuple(cell)
        xp, xm = x.copy(), x.copy()
        xp[cell] += eps
        xm[cell] -= eps
        cp = opt.compliance_gradient(xp, penal=penal, beta=beta, tol=1e-13)[0]
        cm = opt.compliance_gradient(xm, penal=penal, beta=beta, tol=1e-13)[0]
        fd = (cp - cm) / (2 * eps)
        assert grad[cell] == pytest.approx(fd, rel=1e-5, abs=1e-9 * abs(c0))


# ------------------------------------------------------------------------------- SIMP
def test_simp_cantilever_decreases_compliance_and_meets_volume():
    opt, states = run(
        cantilever(),
        n_iter=14,
        volfrac=0.3,
        max_iter=14,
        continuation_iters=0,
        projection=False,
    )
    comp = np.array([s.compliance for s in states])
    assert comp[-1] < 0.75 * comp[0]
    assert np.all(np.diff(comp[2:]) < 1e-3 * comp[0])  # monotone once the design settled
    for s in states:
        assert abs(s.volume - 0.3) < 1e-3
        assert np.all((s.density >= 0.0) & (s.density <= 1.0))
        assert s.stress.shape == opt.grid.shape + (6,) and s.von_mises.shape == opt.grid.shape
    last = states[-1]
    # the reported state is consistent: compliance of the reported density, u exposed
    assert last.compliance == pytest.approx(float(opt.problem.forces @ opt.u), rel=1e-12)
    np.testing.assert_array_equal(opt.physical_density(), last.density)
    rho = last.density.ravel()[opt.model.elems]
    scale = opt.model.stiffness_scale(rho, last.penal)
    np.testing.assert_allclose(
        last.stress, opt.model.to_grid(opt.model.element_stress(opt.u, scale)), rtol=1e-12
    )


def test_simp_mbb_with_continuation_and_projection_gets_crisp():
    opt, states = run(mbb(), n_iter=24, volfrac=0.3, max_iter=24, continuation_iters=8)
    for s in states:
        assert abs(s.volume - 0.3) < 1e-3
    assert states[0].penal < states[-1].penal == 3.0
    assert states[0].beta == 1.0 and states[-1].beta == opt.settings.beta_max
    assert states[-1].compliance < states[0].compliance
    assert states[-1].grey < 0.25 < states[0].grey
    assert opt.done
    # material at the load and at both supports, nothing in the middle of the bottom-centre
    rho = states[-1].density
    assert rho[17:19, :, -2:].min() > 0.5
    assert rho[:2, :, :2].mean() > 0.5 and rho[-2:, :, :2].mean() > 0.5


def test_continuation_schedule():
    problem = cantilever((6, 3, 3))
    opt = Optimizer(problem, OptSettings(max_iter=40))  # defaults: 15 penalty iterations
    sched = [opt._schedule(k) for k in range(41)]
    penal = np.array([p for p, _b in sched])
    beta = np.array([b for _p, b in sched])
    assert penal[0] == 1.0 and penal[15] == 3.0 and np.all(np.diff(penal) >= 0.0)
    assert np.all(beta[:16] == 1.0) and beta[-1] == opt.settings.beta_max
    ramp = beta[15:][beta[15:] < opt.settings.beta_max]
    assert np.all(beta[1:] / beta[:-1] <= 1.3 + 1e-12)
    assert ramp.size >= 8 and np.argmax(beta == opt.settings.beta_max) <= 30  # time to settle
    # a short budget compresses the penalty ramp and ends at a milder sharpness
    short = Optimizer(problem, OptSettings(max_iter=6))
    assert short._schedule(3)[0] == 3.0
    assert 1.0 < short._schedule(6)[1] < 3.0
    # without projection or in RULES mode there is no sharpness
    assert Optimizer(problem, OptSettings(projection=False))._schedule(30)[1] == 0.0
    assert Optimizer(problem, OptSettings(mode="RULES"))._schedule(0)[0] == 3.0
    s = topopt.checked_settings(OptSettings(mode="simp", volfrac=1.7, symmetry=[1, 0, 0]))
    assert s.mode == "SIMP" and s.volfrac == 1.0 and s.symmetry == (True, False, False)


def test_passive_regions_are_respected():
    problem = cantilever((20, 6, 8))
    solid = np.zeros(problem.grid.shape, dtype=bool)
    solid[8:11, :, 6:] = True
    void = np.zeros(problem.grid.shape, dtype=bool)
    void[3:7, :, 2:5] = True
    problem.passive_solid, problem.passive_void = solid, void
    for mode in ("SIMP", "RULES"):
        opt, states = run(problem, n_iter=6, mode=mode, volfrac=0.35, max_iter=6)
        for s in states:
            assert np.all(s.density[solid] == 1.0)
            assert np.all(s.density[void] == 0.0)
            assert np.all(opt.x[solid] == 1.0) and np.all(opt.x[void] == 0.0)


def test_domain_with_holes_and_outside_is_zero():
    problem = cantilever((16, 6, 8))
    dom = np.ones(problem.grid.shape, dtype=bool)
    dom[6:10, :, 3:5] = False  # a tunnel through the beam
    problem.domain = dom
    problem, _ = fem.remove_unsupported_components(problem)
    _opt, states = run(problem, n_iter=5, volfrac=0.4, max_iter=5)
    for s in states:
        assert np.all(s.density[~dom] == 0.0) and np.all(s.von_mises[~dom] == 0.0)
        assert abs(s.volume - 0.4) < 1e-3
        assert np.isfinite(s.compliance) and s.compliance > 0.0


def test_symmetry_is_exact_even_for_an_asymmetric_load():
    problem = mbb((24, 8, 10))
    # make the load asymmetric in y and in x: the enforced symmetry must still be exact
    f3 = problem.forces.reshape(-1, 3)
    nodes = node_ids(problem.grid)
    f3[nodes[7, 2, 10], 2] -= 0.3
    for sym, axes in (((True, False, False), (0,)), ((True, True, False), (0, 1))):
        opt, states = run(problem, n_iter=6, volfrac=0.3, max_iter=6, symmetry=sym)
        for s in states:
            for axis in axes:
                np.testing.assert_array_equal(s.density, np.flip(s.density, axis))
                np.testing.assert_array_equal(opt.x, np.flip(opt.x, axis))
            assert abs(s.volume - 0.3) < 1e-3


def test_zone_bias_attracts_and_repels_material():
    problem = cantilever((20, 6, 10))
    zone = np.zeros(problem.grid.shape, dtype=bool)
    zone[4:9, :, 6:] = True  # upper region near the support (normally solid in tension)
    base_opt, _ = run(problem, n_iter=8, volfrac=0.3, max_iter=8)
    base = base_opt.x[zone].mean()
    problem.bias = np.where(zone, -1.0, 0.0)
    hollow_opt, _ = run(problem, n_iter=8, volfrac=0.3, max_iter=8)
    assert hollow_opt.x[zone].max() < 1e-12  # b = -1: never material
    problem.bias = np.where(zone, 1.0, 0.0)
    zone2 = np.zeros(problem.grid.shape, dtype=bool)
    zone2[10:16, :, 3:7] = True  # mid-height band: normally the void core of the beam
    problem.bias = np.where(zone2, 1.0, 0.0)
    thick_opt, _ = run(problem, n_iter=8, volfrac=0.3, max_iter=8)
    assert thick_opt.x[zone2].mean() > base_opt.x[zone2].mean() + 0.05
    assert base > 0.3


def test_robust_corner_cases():
    # all passive: nothing to design, converges at once
    problem = cantilever((8, 4, 4))
    problem.passive_solid = np.ones(problem.grid.shape, dtype=bool)
    opt = Optimizer(problem, OptSettings(max_iter=5))
    s = opt.step()
    assert s.converged and opt.done and s.volume == 1.0 and s.change == 0.0
    # volfrac >= 1: the full domain
    opt = Optimizer(cantilever((8, 4, 4)), OptSettings(volfrac=1.5, max_iter=3))
    s = opt.run()
    assert np.all(s.density == 1.0) and s.volume == 1.0
    # no load: no NaN, design kept
    problem = cantilever((8, 4, 4))
    problem.forces[:] = 0.0
    for mode in ("SIMP", "RULES"):
        opt = Optimizer(problem, OptSettings(mode=mode, max_iter=3))
        s = opt.run()
        assert np.all(np.isfinite(s.density)) and s.compliance == 0.0
    # empty domain
    problem = cantilever((4, 3, 3))
    problem.domain = np.zeros(problem.grid.shape, dtype=bool)
    s = Optimizer(problem, OptSettings(max_iter=2)).run()
    assert s.volume == 0.0 and np.all(s.density == 0.0)
    with pytest.raises(ValueError):
        Optimizer(cantilever((4, 3, 3)), OptSettings(mode="MAGIC"))


def test_run_callback_done_and_iteration_bookkeeping():
    opt = Optimizer(cantilever((12, 4, 6)), OptSettings(volfrac=0.4, max_iter=5))
    seen = []
    state = opt.run(callback=lambda s: seen.append(s.iteration))
    assert seen == [1, 2, 3, 4, 5] and state.iteration == 5 and opt.done
    assert opt.run() is state  # already done: nothing more
    opt = Optimizer(cantilever((12, 4, 6)), OptSettings(volfrac=0.4, max_iter=5))
    state = opt.run(callback=lambda s: None)  # only an explicit False stops
    assert state.iteration == 5
    opt = Optimizer(cantilever((12, 4, 6)), OptSettings(volfrac=0.4, max_iter=5))
    state = opt.run(callback=lambda s: False)
    assert state.iteration == 1 and not opt.done
    assert opt.u.shape == (opt.grid.n_dofs,)


def test_should_stop_raises_cancelled_and_keeps_the_last_state():
    opt = Optimizer(cantilever((16, 6, 6)), OptSettings(volfrac=0.4, max_iter=10))
    with pytest.raises(Cancelled):
        opt.run(should_stop=lambda: True)
    assert opt.iteration == 0 and opt.state is None
    opt.step()
    x_before = opt.x.copy()
    calls = {"n": 0}

    def stop_soon():
        calls["n"] += 1
        return calls["n"] > 3  # inside the next solve

    with pytest.raises(Cancelled):
        opt.run(should_stop=stop_soon)
    assert opt.iteration == 1
    np.testing.assert_array_equal(opt.x, x_before)
    assert opt.step().iteration == 2  # can continue afterwards


def test_optimizers_are_independent_across_threads():
    problem = cantilever((14, 5, 6))
    settings = OptSettings(volfrac=0.35, max_iter=4)
    reference = Optimizer(problem, settings).run().density
    results = [None, None]

    def work(i):
        results[i] = Optimizer(problem, settings).run().density

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        np.testing.assert_array_equal(r, reference)


# ------------------------------------------------------------------------------ RULES
def test_rules_reaches_volfrac_and_keeps_the_load_path():
    problem = cantilever((24, 6, 10), load_z=0)  # load at the bottom of the free end
    _opt, states = run(problem, n_iter=24, mode="RULES", volfrac=0.4, max_iter=24)
    vols = [s.volume for s in states]
    assert vols[0] < 1.0 and all(abs(v - 0.4) < 2e-3 for v in vols[6:])
    rho = states[-1].density
    solid = rho >= 0.5
    assert rho[:3, :, -1].mean() > 0.9 and rho[:3, :, 0].mean() > 0.9  # flanges at the clamp
    assert rho[-2:, :, 0].mean() > 0.9  # material at the load
    labels, n = fem.connected_components(solid, 26)
    assert n == 1 and labels[0, 0, -1] == labels[-1, 0, 0] == 1  # one piece, load to support
    assert rho[:12, :, 3:7].mean() < 0.5 * rho[:12, :, [0, -1]].mean()  # web lighter than flanges
    assert states[-1].grey < 0.2
    # nearly as stiff as SIMP with the same material
    simp = Optimizer(problem, OptSettings(volfrac=0.4, max_iter=24)).run()
    assert states[-1].compliance < 1.2 * simp.compliance
    # without keep_volume the rules decide the volume freely
    _, free_states = run(problem, n_iter=4, mode="RULES", keep_volume=False, max_iter=4)
    assert free_states[-1].volume != pytest.approx(0.4, abs=1e-2)


def test_rules_convergence_needs_three_quiet_iterations():
    problem = cantilever((16, 4, 8))
    settings = OptSettings(
        mode="RULES", volfrac=0.5, max_iter=60, tol_change=0.02, rate=0.2, projection=False
    )
    opt = Optimizer(problem, settings)
    changes = []
    state = opt.run(callback=lambda s: changes.append(s.change))
    assert state.converged and opt.done and state.iteration < 60
    assert all(c < 0.02 for c in changes[-3:]) and not all(c < 0.02 for c in changes[-4:])
    assert state.beta == 0.0 and state.penal == 3.0
    assert abs(state.volume - 0.5) < 2e-3


# --------------------------------------------------------------- architecture (SUPPORT)
def _components(problem, density):
    labels, n = fem.connected_components(density >= 0.5, 26)
    pads, n_pads = fem.connected_components(problem.passive_solid, 26)
    return labels, n, [np.unique(labels[pads == i]) for i in range(1, n_pads + 1)]


def test_support_mode_low_volfrac_gives_one_connected_crisp_structure():
    problem, _loads, _feet = support_problem(res=22)
    opt = Optimizer(problem, OptSettings(volfrac=0.08, max_iter=30, rmin=1.5))
    states = [opt.step() for _ in range(30)]
    final = states[-1]
    assert abs(final.volume - 0.08) < 1e-3
    assert final.compliance < states[0].compliance
    _labels, n, pad_labels = _components(problem, final.density)
    assert n == 1  # one piece, no floating islands
    assert all(lab.tolist() == [1] for lab in pad_labels)  # every pad belongs to it
    assert final.grey < 0.12
