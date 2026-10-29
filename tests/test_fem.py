"""Tests of the voxel FEM core (``stress_sculpt.core.fem``)."""

import collections
import itertools
import time

import numpy as np
import pytest

from stress_sculpt.core import fem
from stress_sculpt.core.grid import Cancelled, Grid, HEX_NODE_OFFSETS, Problem


# ---------------------------------------------------------------------------- helpers
def node_ids(grid):
    return np.arange(grid.n_nodes).reshape(grid.nodes_shape)


def all_dofs(nodes):
    nodes = np.asarray(nodes).ravel()
    return (3 * nodes[:, None] + np.arange(3)).ravel()


def cantilever(shape, h=1.0, E=1.0, nu=0.3, domain=None, total_force=-1.0):
    """Clamped at x = 0, downward (z) load spread over the end face x = L."""
    g = Grid(shape, (0.0, 0.0, 0.0), h)
    nodes = node_ids(g)
    if domain is None:
        domain = np.ones(shape, dtype=bool)
    domain = np.asarray(domain, dtype=bool)
    touched = g.nodes_of_elements(domain).reshape(g.nodes_shape)
    end = nodes[shape[0]][touched[shape[0]]]
    f = np.zeros(g.n_dofs)
    f[3 * end + 2] = total_force / end.size
    return Problem(g, domain, all_dofs(nodes[0]), f, E=E, nu=nu)


def irregular_problem(shape=(7, 5, 6), seed=3):
    """Small non-box domain with holes, odd sizes and a partial support."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    i, j, k = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    dom = np.ones(shape, dtype=bool)
    dom &= ~((i - 3.2) ** 2 + (j - 2.1) ** 2 + (k - 3.0) ** 2 < 2.3)  # inner hole
    dom &= ~((i >= nx - 2) & (k >= nz - 2))  # notch
    dom &= (i + j) < (nx + ny - 3)  # chamfer
    dom[rng.random(shape) < 0.05] = False  # a few random voids
    g = Grid(shape, (0.3, -0.2, 1.0), 0.25)
    nodes = node_ids(g)
    touched = g.nodes_of_elements(dom).reshape(g.nodes_shape)
    fixed = all_dofs(nodes[0][touched[0]])
    fixed = np.concatenate([fixed, 3 * nodes[:, 0, 0].ravel() + 2])  # roller line
    f = np.zeros(g.n_dofs)
    fnodes = np.flatnonzero(touched.ravel() & (np.arange(g.n_nodes) % 3 == 0))
    f[3 * fnodes + 2] = -1.0 + rng.random(fnodes.size)
    f[3 * fnodes + 0] = 0.3 * rng.standard_normal(fnodes.size)
    return Problem(g, dom, fixed, f, E=7.0, nu=0.33)


def dense_stiffness(problem, scale=None):
    """Explicitly assembled dense K (n_dofs, n_dofs), independent of FEModel."""
    g = problem.grid
    elems = np.flatnonzero(problem.domain.ravel())
    edof = g.element_dofs(elems)
    ke = problem.E * g.h * fem.hex8_ke(problem.nu)
    if scale is None:
        scale = np.ones(elems.size)
    K = np.zeros((g.n_dofs, g.n_dofs))
    for e in range(elems.size):
        K[np.ix_(edof[e], edof[e])] += scale[e] * ke
    return K


def dense_solve(problem, model, scale):
    K = dense_stiffness(problem, scale)
    free = np.flatnonzero(model.free)
    u = np.zeros(problem.grid.n_dofs)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], problem.forces[free])
    return u


def simp_scale(model, seed=0, emin=1e-3, penal=3.0):
    rho = np.random.default_rng(seed).uniform(0.0, 1.0, model.n_act)
    return emin + (1.0 - emin) * rho ** penal


def rigid_modes(coords):
    """(n_nodes * 3, 6) rigid body modes (3 translations, 3 rotations about 0)."""
    n = coords.shape[0]
    modes = np.zeros((n, 3, 6))
    for a in range(3):
        modes[:, a, a] = 1.0
    for a in range(3):
        w = np.zeros(3)
        w[a] = 1.0
        modes[:, :, 3 + a] = np.cross(w, coords)
    return modes.reshape(3 * n, 6)


# ------------------------------------------------------------------------ element
@pytest.mark.parametrize("nu", [0.0, 0.2, 0.3, 0.45, 0.49])
def test_ke_symmetric_psd_with_six_rigid_modes(nu):
    ke = fem.hex8_ke(nu)
    assert ke.shape == (24, 24)
    np.testing.assert_allclose(ke, ke.T, atol=1e-14)
    w = np.linalg.eigvalsh(ke)
    tol = 1e-10 * w[-1]
    assert np.all(w > -tol)
    assert int(np.sum(np.abs(w) < tol)) == 6
    assert w[6] > 1e-3 * w[-1]
    # the rigid modes span the null space
    R = rigid_modes(HEX_NODE_OFFSETS.astype(float))
    np.testing.assert_allclose(ke @ R, 0.0, atol=1e-12)


def test_ke_reproduces_exact_energy_of_linear_fields():
    """u = A x has uniform strain: u^T KE u = volume * eps^T D eps exactly."""
    rng = np.random.default_rng(1)
    nu = 0.27
    ke = fem.hex8_ke(nu)
    D = fem.elasticity_matrix(nu)
    X = HEX_NODE_OFFSETS.astype(float)
    B0 = fem.hex8_b_center()
    for _ in range(5):
        A = rng.standard_normal((3, 3))
        u = (X @ A.T).ravel()
        eps = np.array(
            [A[0, 0], A[1, 1], A[2, 2], A[0, 1] + A[1, 0], A[1, 2] + A[2, 1], A[2, 0] + A[0, 2]]
        )
        assert u @ ke @ u == pytest.approx(eps @ D @ eps, rel=1e-12)
        np.testing.assert_allclose(B0 @ u, eps, atol=1e-13)


def test_box_element_used_by_coarse_levels():
    """Coarse levels use box elements: KE(cube h) = h KE, exact energy of linear fields."""
    nu = 0.31
    np.testing.assert_allclose(fem._hex8_ke_box(nu, (2.0, 2.0, 2.0)), 2.0 * fem.hex8_ke(nu))
    sizes = np.array([4.0, 2.0, 1.0])
    ke = fem._hex8_ke_box(nu, sizes)
    D = fem.elasticity_matrix(nu)
    X = HEX_NODE_OFFSETS * sizes
    rng = np.random.default_rng(3)
    for _ in range(3):
        A = rng.standard_normal((3, 3))
        u = (X @ A.T).ravel()
        eps = np.array(
            [A[0, 0], A[1, 1], A[2, 2], A[0, 1] + A[1, 0], A[1, 2] + A[2, 1], A[2, 0] + A[0, 2]]
        )
        assert u @ ke @ u == pytest.approx(np.prod(sizes) * (eps @ D @ eps), rel=1e-12)
    w = np.linalg.eigvalsh(ke)
    assert int(np.sum(np.abs(w) < 1e-10 * w[-1])) == 6
    # the rigorous smoother bound really bounds lambda_max(D^-1 K) of an assembly
    lam = fem._element_lambda_bound(fem.hex8_ke(nu))
    model = fem.FEModel(cantilever((6, 4, 4), nu=nu))
    scale = simp_scale(model, seed=8)
    K = dense_stiffness(model.problem, scale)
    f = np.flatnonzero(model.free)
    Kf = K[np.ix_(f, f)]
    d = 1.0 / np.sqrt(np.diag(Kf))
    assert np.linalg.eigvalsh(d[:, None] * Kf * d[None, :])[-1] <= lam * (1.0 + 1e-12)


def test_elasticity_matrix():
    nu = 0.3
    D = fem.elasticity_matrix(nu)
    np.testing.assert_allclose(D, D.T)
    # uniaxial stress: strain = (1, -nu, -nu) gives stress (1, 0, 0) for E = 1
    s = D @ np.array([1.0, -nu, -nu, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(s, [1.0, 0, 0, 0, 0, 0], atol=1e-14)
    # engineering shear: tau = G * gamma
    assert D[3, 3] == pytest.approx(1.0 / (2.0 * (1.0 + nu)))
    with pytest.raises(ValueError):
        fem.elasticity_matrix(0.5)


# -------------------------------------------------------------- stress measures
def test_von_mises_and_principal_sanity():
    s = 3.0e6
    assert fem.von_mises([s, 0, 0, 0, 0, 0]) == pytest.approx(s)
    assert fem.von_mises([0, 0, 0, s, 0, 0]) == pytest.approx(np.sqrt(3.0) * s)
    assert fem.von_mises([0, 0, 0, 0, 0, s]) == pytest.approx(np.sqrt(3.0) * s)
    assert fem.von_mises([s, s, s, 0, 0, 0]) == pytest.approx(0.0, abs=1e-9)
    assert fem.von_mises(np.zeros((4, 5, 6))).shape == (4, 5)

    # pure shear in xy: principal stresses -tau, 0, +tau at 45 degrees
    w, v = fem.principal([0, 0, 0, 2.0, 0, 0])
    np.testing.assert_allclose(w, [-2.0, 0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(np.abs(v[:, 2]), [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-12)

    # random rotated tensors: ascending values, orthonormal columns, V diag(w) V^T = T
    rng = np.random.default_rng(0)
    s6 = rng.standard_normal((3, 4, 6))
    w, v = fem.principal(s6)
    assert w.shape == (3, 4, 3) and v.shape == (3, 4, 3, 3)
    assert np.all(np.diff(w, axis=-1) >= 0.0)
    t = fem.voigt_to_tensor(s6)
    np.testing.assert_allclose(np.swapaxes(t, -1, -2), t)
    np.testing.assert_allclose(np.einsum("...ik,...k,...jk->...ij", v, w, v), t, atol=1e-12)
    eye = np.broadcast_to(np.eye(3), t.shape)
    np.testing.assert_allclose(np.einsum("...ki,...kj->...ij", v, v), eye, atol=1e-12)
    # von Mises from the principal values
    d01, d12, d20 = (w[..., 0] - w[..., 1], w[..., 1] - w[..., 2], w[..., 2] - w[..., 0])
    vm = np.sqrt(0.5 * (d01**2 + d12**2 + d20**2))
    np.testing.assert_allclose(fem.von_mises(s6), vm, rtol=1e-10)

    # tensor layout
    t = fem.voigt_to_tensor([1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(t, [[1, 4, 6], [4, 2, 5], [6, 5, 3]])
    # non-finite input -> NaN rows only
    s6 = np.array([[1.0, 0, 0, 0, 0, 0], [np.nan, 0, 0, 0, 0, 0]])
    w, v = fem.principal(s6)
    assert np.all(np.isfinite(w[0])) and np.all(np.isnan(w[1])) and np.all(np.isnan(v[1]))


# -------------------------------------------------------------------- operators
def test_apply_K_matches_dense_assembly_on_irregular_domain():
    prob = irregular_problem()
    model = fem.FEModel(prob)
    rng = np.random.default_rng(5)
    scale = rng.uniform(0.05, 2.0, model.n_act)
    K = dense_stiffness(prob, scale)
    free = model.free
    for _ in range(3):
        u = rng.standard_normal(prob.grid.n_dofs)
        expected = np.where(free, K @ u, 0.0)
        np.testing.assert_allclose(model.apply_K(u, scale), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.diagonal(scale), np.diag(K), rtol=1e-12, atol=1e-14)
    # attributes of the contract
    assert model.elems.dtype == np.int64 and model.edof.shape == (model.n_act, 24)
    np.testing.assert_array_equal(model.elems, np.flatnonzero(prob.domain))
    np.testing.assert_allclose(model.KE, prob.E * prob.grid.h * fem.hex8_ke(prob.nu))
    touched = np.zeros(prob.grid.n_dofs, dtype=bool)
    touched[model.edof.ravel()] = True
    fixed = np.zeros(prob.grid.n_dofs, dtype=bool)
    fixed[prob.fixed_dofs] = True
    np.testing.assert_array_equal(model.free, touched & ~fixed)


def test_blocked_element_loops_match_single_block(monkeypatch):
    """Large models are processed in element blocks: results must not change."""
    prob = irregular_problem()
    rng = np.random.default_rng(9)
    u = rng.standard_normal(prob.grid.n_dofs)
    model = fem.FEModel(prob)
    scale = rng.uniform(0.1, 1.0, model.n_act)
    ref = (
        model.apply_K(u, scale),
        model.diagonal(scale),
        model.element_energy(u),
        model.element_stress(u, scale),
        model.solve(scale, tol=1e-10)[0],
    )
    monkeypatch.setattr(fem, "_ELEM_BLOCK", 37)  # odd block size, many blocks
    model = fem.FEModel(prob)
    got = (
        model.apply_K(u, scale),
        model.diagonal(scale),
        model.element_energy(u),
        model.element_stress(u, scale),
        model.solve(scale, tol=1e-10)[0],
    )
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12 * np.abs(a).max())


@pytest.mark.parametrize("solver", ["MG", "JACOBI"])
@pytest.mark.parametrize("simp", [False, True])
def test_pcg_matches_dense_direct_solve(solver, simp):
    prob = irregular_problem()
    model = fem.FEModel(prob, solver)
    scale = simp_scale(model, seed=2) if simp else np.ones(model.n_act)
    u_ref = dense_solve(prob, model, scale)
    u, info = model.solve(scale, tol=1e-11, maxiter=20000)
    assert info["converged"] and info["solver"] == solver
    assert info["residual"] <= 1e-11
    np.testing.assert_allclose(u, u_ref, rtol=0, atol=1e-8 * np.abs(u_ref).max())
    assert np.all(u[~model.free] == 0.0)
    # compliance identities
    energy = model.element_energy(u)
    assert np.sum(scale * energy) == pytest.approx(prob.forces @ u, rel=1e-8)


def test_multigrid_hierarchy_on_odd_irregular_domain_matches_dense():
    """Bigger odd-sized domain with holes: three levels, both solvers agree with dense K."""
    shape = (11, 7, 9)
    rng = np.random.default_rng(11)
    dom = rng.random(shape) > 0.15
    dom[:2] = True
    prob = cantilever(shape, h=0.5, E=3.0, nu=0.36, domain=dom)
    # make sure the load sits on the domain
    model = fem.FEModel(prob, "MG", mg_options=fem.MGOptions(coarse_dofs=100))
    scale = simp_scale(model, seed=4)
    u_ref = dense_solve(prob, model, scale)
    u, info = model.solve(scale, tol=1e-11, maxiter=5000)
    assert info["solver"] == "MG" and info["converged"]
    assert len(model._levels) >= 3
    np.testing.assert_allclose(u, u_ref, rtol=0, atol=1e-8 * np.abs(u_ref).max())


def test_prolongation_restriction_are_exact_transposes():
    prob = cantilever((5, 4, 3))
    model = fem.FEModel(prob, "MG", mg_options=fem.MGOptions(coarse_dofs=1))
    levels = model._mg_geometry()
    fine, coarse = levels[0], levels[1]
    assert fine.factors == (2, 2, 2) and coarse.shape == (3, 2, 2)
    P = np.stack([fem._prolong(e, fine, coarse) for e in np.eye(coarse.n_dofs)], axis=1)
    R = np.stack([fem._restrict(e, fine, coarse) for e in np.eye(fine.n_dofs)], axis=1)
    np.testing.assert_array_equal(R, P.T)
    # trilinear: reproduces linear fields
    gc = Grid(coarse.shape, (0, 0, 0), 2.0)
    gf = Grid(fine.shape, (0, 0, 0), 1.0)
    lin = lambda X: (X @ np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 3.0], [-2.0, 1.0, 1.0]])).ravel()
    np.testing.assert_allclose(P @ lin(gc.node_coords()), lin(gf.node_coords()), atol=1e-12)
    # coarse fixed dofs: every coarse dof whose column touches a fine fixed dof
    touches = (np.abs(P[fine.fixed]) > 0).any(axis=0)
    np.testing.assert_array_equal(coarse.fixed, touches)


@pytest.mark.parametrize(
    "case",
    ["solid", "simp", "irregular", "slab"],
)
def test_vcycle_is_symmetric_positive_definite(case):
    rng = np.random.default_rng(7)
    if case == "slab":
        g = Grid((16, 12, 2), (0, 0, 0), 1.0)
        nodes = node_ids(g)
        fixed = all_dofs(np.concatenate([nodes[0].ravel(), nodes[-1].ravel()]))
        f = np.zeros(g.n_dofs)
        f[3 * nodes[8, 6, 2] + 2] = -1.0
        prob = Problem(g, np.ones(g.shape, bool), fixed, f)
    elif case == "irregular":
        dom = rng.random((13, 9, 11)) > 0.2
        dom[0] = True
        prob = cantilever((13, 9, 11), domain=dom)
    else:
        prob = cantilever((16, 8, 8))
    model = fem.FEModel(prob, "MG", mg_options=fem.MGOptions(coarse_dofs=200))
    scale = simp_scale(model, seed=1) if case == "simp" else np.ones(model.n_act)
    M = model._mg_preconditioner(scale)
    assert len(model._levels) >= 2
    free = model.free
    for _ in range(4):
        x = rng.standard_normal(prob.grid.n_dofs) * free
        y = rng.standard_normal(prob.grid.n_dofs) * free
        mx, my = M(x), M(y)
        assert np.all(mx[~free] == 0.0)
        assert abs(y @ mx - x @ my) <= 1e-10 * abs(x @ mx) + 1e-300
        assert x @ mx > 0.0 and y @ my > 0.0
    # positive definiteness along the extreme directions: power / inverse estimates
    # of the spectrum of M K (all eigenvalues must be real and positive).
    K = lambda v: model.apply_K(v, scale)
    v = rng.standard_normal(prob.grid.n_dofs) * free
    for _ in range(30):
        w = M(K(v))
        v = w / np.linalg.norm(w)
    lam_max = v @ K(M(K(v))) / (v @ K(v))  # Rayleigh quotient in the K inner product
    assert 0.0 < lam_max < 2.0 + 1e-9  # V-cycle is a contraction: 0 < lambda(MK) < 2


def test_mg_dense_coarse_inverse_is_spd():
    model = fem.FEModel(cantilever((12, 6, 6)), "MG")
    model._mg_preconditioner(np.ones(model.n_act))
    G = model._ops[-1].coarse_inv
    Minv = G.T @ G
    np.testing.assert_allclose(Minv, Minv.T, rtol=0, atol=1e-12 * np.abs(Minv).max())
    assert np.linalg.eigvalsh(0.5 * (Minv + Minv.T))[0] > 0.0


# --------------------------------------------------------------------- physics
@pytest.mark.parametrize("solver", ["MG", "JACOBI"])
def test_patch_test_uniaxial_tension(solver):
    """Symmetry BCs + consistent nodal tractions: uniform stress, linear displacement."""
    shape = (6, 3, 4)
    h, E, nu, sigma = 0.05, 2.0e11, 0.3, 1.0e6
    g = Grid(shape, (0.0, 0.0, 0.0), h)
    nodes = node_ids(g)
    fixed = np.concatenate(
        [3 * nodes[0].ravel(), 3 * nodes[:, 0, :].ravel() + 1, 3 * nodes[:, :, 0].ravel() + 2]
    )
    f = np.zeros(g.n_dofs)
    wj = np.ones(shape[1] + 1)
    wj[[0, -1]] = 0.5
    wk = np.ones(shape[2] + 1)
    wk[[0, -1]] = 0.5
    f[3 * nodes[shape[0]].ravel()] = sigma * h * h * np.outer(wj, wk).ravel()
    area = shape[1] * h * shape[2] * h
    assert f.sum() == pytest.approx(sigma * area)
    prob = Problem(g, np.ones(shape, bool), fixed, f, E=E, nu=nu)
    res = fem.analyze(prob, solver=solver, tol=1e-12)
    assert res.info["converged"]
    st = res.stress.reshape(-1, 6)
    np.testing.assert_allclose(st[:, 0], sigma, rtol=1e-7)
    np.testing.assert_allclose(st[:, 1:], 0.0, atol=1e-7 * sigma)
    X = g.node_coords()
    U = res.u.reshape(-1, 3)
    np.testing.assert_allclose(U[:, 0], sigma * X[:, 0] / E, atol=1e-8 * sigma * 0.3 / E)
    np.testing.assert_allclose(U[:, 1], -nu * sigma * X[:, 1] / E, atol=1e-8 * sigma * 0.3 / E)
    np.testing.assert_allclose(U[:, 2], -nu * sigma * X[:, 2] / E, atol=1e-8 * sigma * 0.3 / E)
    np.testing.assert_allclose(res.von_mises, sigma, rtol=1e-7)
    assert res.compliance == pytest.approx(f @ res.u)


def test_cantilever_tip_deflection_vs_timoshenko():
    nx, ny, nz = 32, 3, 6
    h, E, nu, P = 0.1, 2.0e9, 0.3, 1000.0
    g = Grid((nx, ny, nz), (0.0, 0.0, 0.0), h)
    nodes = node_ids(g)
    wj = np.ones(ny + 1)
    wj[[0, -1]] = 0.5
    wk = np.ones(nz + 1)
    wk[[0, -1]] = 0.5
    w = np.outer(wj, wk).ravel()
    w /= w.sum()
    f = np.zeros(g.n_dofs)
    f[3 * nodes[nx].ravel() + 2] = -P * w
    prob = Problem(g, np.ones(g.shape, bool), all_dofs(nodes[0]), f, E=E, nu=nu)
    res = fem.analyze(prob, tol=1e-9)
    tip = -np.sum(res.u.reshape(-1, 3)[nodes[nx].ravel(), 2] * w)
    L, b, t = nx * h, ny * h, nz * h
    inertia = b * t**3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    timoshenko = P * L**3 / (3.0 * E * inertia) + P * L / (5.0 / 6.0 * G * b * t)
    ratio = tip / timoshenko
    assert 0.85 < ratio < 1.05  # hex8 is slightly too stiff (shear locking, clamp)
    # bending stress at the clamp ~ M c / I (away from the singular clamp corners)
    sxx = res.stress[nx // 2, ny // 2, :, 0]
    expected = P * (L - (nx // 2 + 0.5) * h) * ((np.arange(nz) + 0.5) * h - t / 2) / inertia
    np.testing.assert_allclose(sxx, expected, rtol=0.15, atol=0.05 * np.abs(expected).max())


# ---------------------------------------------------------------- solver behaviour
def test_multigrid_needs_far_fewer_iterations_than_jacobi():
    prob = cantilever((32, 16, 16))
    for simp in (False, True):
        its = {}
        sols = {}
        for solver in ("MG", "JACOBI"):
            model = fem.FEModel(prob, solver)
            scale = simp_scale(model, seed=3) if simp else np.ones(model.n_act)
            u, info = model.solve(scale, tol=1e-6, maxiter=20000)
            assert info["converged"] and info["solver"] == solver
            its[solver] = info["iterations"]
            sols[solver] = u
        assert its["MG"] * 8 < its["JACOBI"], its
        assert its["MG"] <= (25 if not simp else 60), its
        err = np.abs(sols["MG"] - sols["JACOBI"]).max() / np.abs(sols["JACOBI"]).max()
        assert err < 1e-4


def test_solve_thin_slab_semicoarsening():
    g = Grid((24, 24, 2), (0, 0, 0), 1.0)
    nodes = node_ids(g)
    corners = np.concatenate([nodes[i, j, :].ravel() for i in (0, 24) for j in (0, 24)])
    f = np.zeros(g.n_dofs)
    f[3 * nodes[12, 12, 2] + 2] = -1.0
    prob = Problem(g, np.ones(g.shape, bool), all_dofs(corners), f)
    mg = fem.FEModel(prob, "MG", mg_options=fem.MGOptions(coarse_dofs=100))
    u, info = mg.solve(None, tol=1e-8)
    assert info["converged"] and info["solver"] == "MG"
    # the thickness is halved once (2 -> 1 element), then only in-plane coarsening
    assert [lv.shape for lv in mg._levels[:4]] == [(24, 24, 2), (12, 12, 1), (6, 6, 1), (3, 3, 1)]
    np.testing.assert_allclose(mg._levels[2].sizes, [4.0, 4.0, 2.0])
    u2, info2 = fem.FEModel(prob, "JACOBI").solve(None, tol=1e-8, maxiter=20000)
    assert info["iterations"] * 5 < info2["iterations"]
    np.testing.assert_allclose(u, u2, atol=1e-5 * np.abs(u2).max())
    u3, info3 = fem.FEModel(prob, "MG").solve(None, tol=1e-8)  # default options
    assert info3["converged"] and info3["iterations"] * 5 < info2["iterations"]
    np.testing.assert_allclose(u3, u2, atol=1e-5 * np.abs(u2).max())


def test_solve_edge_cases():
    prob = cantilever((6, 3, 3))
    g = prob.grid
    # zero force -> zeros at once
    zero = Problem(g, prob.domain, prob.fixed_dofs, np.zeros(g.n_dofs))
    for solver in fem.SOLVERS:
        u, info = fem.FEModel(zero, solver).solve(None)
        assert np.all(u == 0.0) and info["iterations"] == 0 and info["converged"]
    # no free dofs
    every = Problem(g, prob.domain, np.arange(g.n_dofs), prob.forces)
    u, info = fem.FEModel(every).solve(None)
    assert np.all(u == 0.0) and info["converged"]
    # empty domain
    empty = Problem(g, np.zeros(g.shape, bool), prob.fixed_dofs, prob.forces)
    res = fem.analyze(empty)
    assert np.all(res.u == 0.0) and res.stress.shape == g.shape + (6,) and res.compliance == 0.0
    # non-convergence is reported, not raised
    u, info = fem.FEModel(prob, "JACOBI").solve(None, tol=1e-14, maxiter=3)
    assert not info["converged"] and info["iterations"] == 3 and np.all(np.isfinite(u))
    # invalid input
    model = fem.FEModel(prob)
    bad = np.ones(model.n_act)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        model.solve(bad)
    with pytest.raises(ValueError):
        model.solve(-np.ones(model.n_act))
    with pytest.raises(ValueError):
        model.solve(np.ones(3))
    with pytest.raises(ValueError):
        fem.FEModel(prob, "CHOLESKY")
    nan_force = prob.forces.copy()
    nan_force[-1] = np.inf
    with pytest.raises(ValueError):
        fem.FEModel(Problem(g, prob.domain, prob.fixed_dofs, nan_force)).solve(None)


def test_mechanism_solve_returns_finite_unconverged_result():
    g = Grid((6, 3, 3), (0, 0, 0), 1.0)
    f = np.zeros(g.n_dofs)
    f[-1] = 1.0
    prob = Problem(g, np.ones(g.shape, bool), np.arange(3), f)  # one pinned node
    for solver in fem.SOLVERS:
        u, info = fem.FEModel(prob, solver).solve(None, maxiter=300)
        assert not info["converged"]
        assert np.all(np.isfinite(u))


def test_warm_start_and_caching():
    prob = cantilever((16, 6, 6))
    model = fem.FEModel(prob, "MG")
    u, info = model.solve(None, tol=1e-8)
    u2, info2 = model.solve(None, x0=u, tol=1e-8)
    assert info2["iterations"] == 0 and info2["converged"]
    np.testing.assert_allclose(u2, u)
    # stiffer material, warm start from the softer solution
    u3, info3 = model.solve(np.full(model.n_act, 2.0), x0=u, tol=1e-8)
    assert info3["converged"] and info3["iterations"] < info["iterations"] + 2
    np.testing.assert_allclose(u3, 0.5 * u, rtol=0, atol=1e-6 * np.abs(u).max())
    # a useless warm start (wrong sign, huge) is ignored
    u4, info4 = model.solve(None, x0=-1e3 * u, tol=1e-8)
    assert info4["converged"]
    np.testing.assert_allclose(u4, u, rtol=0, atol=1e-6 * np.abs(u).max())


def test_mg_falls_back_to_jacobi_when_hierarchy_unusable():
    prob = cantilever((20, 10, 10))
    model = fem.FEModel(prob, "MG", mg_options=fem.MGOptions(max_levels=1, coarse_dofs=10))
    u, info = model.solve(None, tol=1e-8)
    assert info["solver"] == "JACOBI" and "fallback" in info and info["converged"]
    u_mg, _ = fem.FEModel(prob, "MG").solve(None, tol=1e-8)
    np.testing.assert_allclose(u, u_mg, rtol=0, atol=1e-6 * np.abs(u_mg).max())


def test_should_stop_raises_cancelled():
    prob = cantilever((16, 8, 8))
    calls = {"n": 0}

    def stop_after_three():
        calls["n"] += 1
        return calls["n"] > 3

    for solver in fem.SOLVERS:
        calls["n"] = 0
        with pytest.raises(Cancelled):
            fem.FEModel(prob, solver).solve(None, should_stop=stop_after_three)
    with pytest.raises(Cancelled):
        fem.analyze(prob, should_stop=lambda: True)


def test_analyze_result_fields_and_progress():
    prob = cantilever((12, 4, 5), h=0.2, E=5.0e9)
    dom = prob.domain.copy()
    dom[6:9, 1:3, 2:4] = False  # an interior hole
    prob = cantilever((12, 4, 5), h=0.2, E=5.0e9, domain=dom)
    seen = []
    res = fem.analyze(prob, progress=lambda f, m: seen.append((f, m)))
    assert res.info["converged"] and res.info["time"] > 0.0
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0
    assert all(0.0 <= f <= 1.0 for f, _ in seen)
    g = prob.grid
    assert res.u.shape == (g.n_dofs,)
    assert res.stress.shape == g.shape + (6,)
    assert res.von_mises.shape == g.shape and res.displacement.shape == g.shape
    assert np.all(res.stress[~dom] == 0.0) and np.all(res.displacement[~dom] == 0.0)
    assert np.all(res.von_mises[dom] > 0.0)
    unode = np.linalg.norm(res.u.reshape(-1, 3), axis=1)
    assert res.max_displacement == pytest.approx(unode.max())
    assert res.compliance == pytest.approx(prob.forces @ res.u) and res.compliance > 0.0
    model = fem.FEModel(prob)
    np.testing.assert_allclose(res.von_mises, fem.von_mises(res.stress))
    enodes = model.edof[:, 0::3] // 3
    np.testing.assert_allclose(model.to_grid(unode[enodes].mean(axis=1)), res.displacement)


def test_stiffness_scale_and_to_grid():
    prob = cantilever((4, 3, 2))
    dom = prob.domain.copy()
    dom[1, 1, 1] = False
    prob = cantilever((4, 3, 2), domain=dom)
    model = fem.FEModel(prob, emin=1e-3)
    rho = np.linspace(-0.1, 1.2, model.n_act)
    s = model.stiffness_scale(rho, 3.0)
    assert s.min() == pytest.approx(1e-3) and s.max() == pytest.approx(1.0)
    assert np.all(np.isfinite(model.stiffness_scale(rho, 2.7)))
    grid_vals = model.to_grid(np.arange(model.n_act, dtype=float), fill=-1.0)
    assert grid_vals.shape == prob.grid.shape and grid_vals[1, 1, 1] == -1.0
    np.testing.assert_array_equal(grid_vals[dom], np.arange(model.n_act))
    assert model.to_grid(np.ones((model.n_act, 6))).shape == prob.grid.shape + (6,)


# ----------------------------------------------------------------- connectivity
def reference_components(mask, connectivity):
    """Plain BFS labelling, components numbered in C-order of their first voxel."""
    offsets = [
        d
        for d in itertools.product((-1, 0, 1), repeat=3)
        if d != (0, 0, 0) and sum(map(abs, d)) <= {6: 1, 18: 2, 26: 3}[connectivity]
    ]
    labels = np.zeros(mask.shape, dtype=np.int64)
    n = 0
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        n += 1
        labels[start] = n
        queue = collections.deque([start])
        while queue:
            p = queue.popleft()
            for d in offsets:
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if all(0 <= q[a] < mask.shape[a] for a in range(3)) and mask[q] and not labels[q]:
                    labels[q] = n
                    queue.append(q)
    return labels, n


@pytest.mark.parametrize("connectivity", [6, 18, 26])
@pytest.mark.parametrize("density", [0.25, 0.45, 0.7])
def test_connected_components_matches_bfs(connectivity, density):
    rng = np.random.default_rng(int(density * 100) + connectivity)
    mask = rng.random((11, 9, 7)) < density
    labels, n = fem.connected_components(mask, connectivity)
    ref, n_ref = reference_components(mask, connectivity)
    assert n == n_ref
    np.testing.assert_array_equal(labels, ref)


def test_connected_components_special_cases():
    labels, n = fem.connected_components(np.zeros((3, 4, 5), bool))
    assert n == 0 and labels.shape == (3, 4, 5) and not labels.any()
    m = np.zeros((3, 3, 3), bool)
    m[0, 0, 0] = m[1, 1, 1] = m[2, 2, 2] = True  # corner contacts only
    assert fem.connected_components(m, 26)[1] == 1
    assert fem.connected_components(m, 18)[1] == 3
    m = np.zeros((3, 3, 1), bool)
    m[0, 0, 0] = m[1, 1, 0] = True  # edge contact
    assert fem.connected_components(m, 18)[1] == 1
    assert fem.connected_components(m, 6)[1] == 2
    assert fem.connected_components(np.ones((1, 1, 1), bool))[1] == 1
    with pytest.raises(ValueError):
        fem.connected_components(np.ones((3, 3)), 26)


def test_connected_components_large_serpentine_is_fast():
    """One long folded path through a 100^3 grid (worst case for label propagation)."""
    n = 100
    mask = np.zeros((n, n, n), bool)
    for i in range(0, n, 2):
        mask[i] = True
        if i + 1 < n:
            mask[i + 1, 0 if (i // 2) % 2 == 0 else n - 1, :] = True
    t0 = time.perf_counter()
    labels, count = fem.connected_components(mask)
    elapsed = time.perf_counter() - t0
    assert count == 1 and np.all(labels[mask] == 1) and np.all(labels[~mask] == 0)
    assert elapsed < 10.0  # typically ~0.15 s; generous bound for loaded machines
    mask[51, :, :] = False  # cut the connection between plane 50 and plane 52
    assert fem.connected_components(mask)[1] == 2


def two_blocks_problem():
    g = Grid((9, 3, 3), (0, 0, 0), 1.0)
    dom = np.zeros(g.shape, bool)
    dom[0:4] = True
    dom[5:9] = True
    nodes = node_ids(g)
    f = np.zeros(g.n_dofs)
    f[3 * nodes[4, :, :].ravel() + 2] = -1.0  # end of block 1 (x = 4)
    f[3 * nodes[9, :, :].ravel() + 2] = -2.0  # end of block 2
    return Problem(g, dom, all_dofs(nodes[0]), f, bias=np.full(g.shape, 0.5))


def test_remove_unsupported_components():
    prob = two_blocks_problem()
    new, removed = fem.remove_unsupported_components(prob)
    assert removed == 1
    np.testing.assert_array_equal(new.domain[0:4], True)
    assert not new.domain[4:].any()
    nodes = node_ids(prob.grid)
    f3 = new.forces.reshape(-1, 3)
    assert np.all(f3[nodes[9].ravel()] == 0.0)  # load of the removed block dropped
    assert np.all(f3[nodes[4].ravel(), 2] == -1.0)  # load of the kept block kept
    assert new.bias is not None and np.array_equal(new.fixed_dofs, prob.fixed_dofs)
    assert prob.domain[5:].all()  # input untouched
    # nothing to remove -> same object
    same, removed = fem.remove_unsupported_components(new)
    assert removed == 0 and same is new
    # no support at all: nothing removed (check_problem reports it)
    nosup = Problem(prob.grid, prob.domain, np.zeros(0, np.int64), prob.forces)
    same, removed = fem.remove_unsupported_components(nosup)
    assert removed == 0 and same is nosup
    # the solve of the cleaned problem works
    u, info = fem.FEModel(new).solve(None)
    assert info["converged"]


# ---------------------------------------------------------------- problem checks
def test_check_problem_clean_problem_has_no_warnings():
    assert fem.check_problem(cantilever((8, 3, 3))) == []
    # the irregular test problem also loads some nodes of its roller line
    w = fem.check_problem(irregular_problem())
    assert len(w) == 1 and "acts directly on supported nodes" in w[0]


def test_check_problem_mechanisms():
    g = Grid((8, 3, 3), (0, 0, 0), 1.0)
    nodes = node_ids(g)
    f = np.zeros(g.n_dofs)
    f[3 * nodes[8].ravel() + 2] = -1.0
    dom = np.ones(g.shape, bool)

    def warnings_for(fixed):
        return [w for w in fem.check_problem(Problem(g, dom, fixed, f)) if "rigid motion" in w]

    # hinge: one line of nodes along y, fully fixed -> rotation about Y is free
    w = warnings_for(all_dofs(nodes[0, :, 0]))
    assert len(w) == 1 and "rotation about Y" in w[0]
    # a single pinned node: three rotations free
    w = warnings_for(all_dofs(nodes[0, 0, 0]))
    assert len(w) == 1 and w[0].count("rotation") == 3
    # roller: only z fixed on the bottom face -> x/y translations + rotation about Z
    w = warnings_for(3 * nodes[:, :, 0].ravel() + 2)
    assert len(w) == 1
    assert "translation along X" in w[0] and "translation along Y" in w[0]
    assert "rotation about Z" in w[0] and "rotation about X" not in w[0]
    # bridge with a pin and a roller (z + y) at the bottom ends: stable
    fixed = np.concatenate([all_dofs(nodes[0, :, 0]), 3 * nodes[8, :, 0].ravel() + 2,
                            3 * nodes[8, :, 0].ravel() + 1])
    assert warnings_for(fixed) == []
    # clamped face: stable
    assert warnings_for(all_dofs(nodes[0])) == []


def test_check_problem_other_warnings():
    g = Grid((6, 3, 3), (0, 0, 0), 1.0)
    nodes = node_ids(g)
    dom = np.ones(g.shape, bool)
    clamp = all_dofs(nodes[0])
    f = np.zeros(g.n_dofs)
    f[3 * nodes[6].ravel() + 2] = -1.0

    w = fem.check_problem(Problem(g, np.zeros(g.shape, bool), clamp, f))
    assert len(w) == 1 and "empty" in w[0].lower()
    w = fem.check_problem(Problem(g, dom, np.zeros(0, np.int64), f))
    assert any("no supports" in x.lower() for x in w)
    w = fem.check_problem(Problem(g, dom, clamp, np.zeros(g.n_dofs)))
    assert any("no loads" in x.lower() for x in w)
    f_fixed = np.zeros(g.n_dofs)
    f_fixed[clamp] = 1.0
    w = fem.check_problem(Problem(g, dom, clamp, f_fixed))
    assert any("supported (fixed) nodes" in x for x in w)
    f_nan = f.copy()
    f_nan[-1] = np.nan
    assert any("not valid numbers" in x for x in fem.check_problem(Problem(g, dom, clamp, f_nan)))
    dom_part = dom.copy()
    dom_part[4:] = False  # load nodes at x = 6 are no longer on the domain
    w = fem.check_problem(Problem(g, dom_part, clamp, f))
    assert any("outside the domain" in x for x in w)
    w = fem.check_problem(two_blocks_problem())
    assert any("disconnected part" in x and "touches no support" in x for x in w)
    assert any("67% of the load" in x for x in w)
