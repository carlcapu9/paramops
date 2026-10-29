"""Principal stress trajectories (``core/streamlines.py``): volume and surface modes."""

import threading
import time

import numpy as np
import pytest

from stress_sculpt.core import fem, streamlines as sl, topopt
from stress_sculpt.core.grid import Cancelled, Grid, Problem


# ------------------------------------------------------------------ helpers
def _voigt(T) -> np.ndarray:
    """(..., 3, 3) symmetric tensors -> (..., 6) Voigt [xx, yy, zz, xy, yz, zx]."""
    T = np.asarray(T, dtype=np.float64)
    return np.stack(
        [T[..., 0, 0], T[..., 1, 1], T[..., 2, 2], T[..., 0, 1], T[..., 1, 2], T[..., 2, 0]],
        axis=-1,
    )


def _uniform(shape, s6, origin=(0.0, 0.0, 0.0), h=1.0):
    g = Grid(shape, origin, h)
    stress = np.zeros(g.shape + (6,))
    stress[...] = s6
    return g, stress


def _voxels(grid, points):
    return (np.asarray(points) - grid.origin) / grid.h


def _length(points) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _point_line_distance(points, lines) -> np.ndarray:
    """Distance from every point ``(P, 3)`` to the nearest segment of ``lines``."""
    pts = [np.asarray(ln.points) for ln in lines if len(ln.points) > 1]
    segs = [(p[:-1], p[1:]) for p in pts]
    if not segs:
        return np.full(len(points), np.inf)
    a = np.concatenate([s[0] for s in segs])
    b = np.concatenate([s[1] for s in segs])
    ab = b - a
    l2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    out = np.empty(len(points))
    for start in range(0, len(points), 256):
        p = points[start : start + 256]
        ap = p[:, None, :] - a[None]
        t = np.clip(np.einsum("pmk,mk->pm", ap, ab) / l2, 0.0, 1.0)
        d = ap - t[..., None] * ab
        out[start : start + 256] = np.sqrt(np.einsum("pmk,pmk->pm", d, d).min(axis=1))
    return out


def _same_family_distances(lines, family) -> np.ndarray:
    """Per point of every ``family`` line: distance to the nearest other line of it."""
    group = [ln for ln in lines if ln.family == family]
    out = []
    for i, line in enumerate(group):
        others = group[:i] + group[i + 1 :]
        if others:
            out.append(_point_line_distance(np.asarray(line.points), others))
    return np.concatenate(out) if out else np.zeros(0)


def _unit_rows(d):
    return d / np.linalg.norm(d, axis=1)[:, None]


def _identical(a, b) -> bool:
    return len(a) == len(b) and all(
        x.family == y.family
        and x.generation == y.generation
        and x.closed == y.closed
        and np.array_equal(x.points, y.points)
        and np.array_equal(x.magnitude, y.magnitude)
        and np.array_equal(x.sign, y.sign)
        for x, y in zip(a, b)
    )


def _ring_field(n=40, inner=5.0, outer=18.0, hoop=1.0, radial=-0.3):
    """Annulus around the z axis: hoop tension, radial compression (voxel units)."""
    g = Grid((n, n, 4), (-n / 2, -n / 2, 0.0), 1.0)
    c = g.element_centers()
    x, y = c[..., 0], c[..., 1]
    r = np.hypot(x, y)
    ct, st = x / np.maximum(r, 1e-9), y / np.maximum(r, 1e-9)
    s6 = np.zeros(g.shape + (6,))
    s6[..., 0] = radial * ct * ct + hoop * st * st
    s6[..., 1] = radial * st * st + hoop * ct * ct
    s6[..., 3] = (radial - hoop) * ct * st
    return g, s6, (r > inner) & (r < outer)


def _cantilever_field(nx=60, ny=30, nz=30):
    """Analytic bending + shear stress of a cantilever (beam theory, along x)."""
    g = Grid((nx, ny, nz), (0.0, 0.0, 0.0), 1.0)
    c = g.element_centers()
    x, z = c[..., 0], c[..., 2] - nz / 2
    half = nz / 2
    inertia = 2 * half ** 3 / 3
    s6 = np.zeros(g.shape + (6,))
    s6[..., 0] = -(nx - x) * z / inertia
    s6[..., 5] = -(half ** 2 - z ** 2) / (2 * inertia)
    return g, s6


def _plate(n=40, nz=6, alpha=0.3, s1=2.0, s2=-1.0, szz=5.0, h=0.5, origin=(1.0, -2.0, 0.5)):
    """Thin slab whose mid-plane is the iso level of a linear field ``f = z`` (voxels);
    uniform stress: in-plane principal stresses s1, s2 rotated by ``alpha`` about z
    plus a normal stress ``szz`` (which the surface mode must ignore)."""
    g = Grid((n, n, nz), origin, h)
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    stress = np.zeros(g.shape + (6,))
    stress[...] = _voigt(R @ np.diag([s1, s2, szz]) @ R.T)
    f = np.broadcast_to((np.arange(nz) + 0.5)[None, None, :], g.shape).copy()
    iso = nz / 2 + 0.2
    return g, stress, f, iso, R


def _sphere(n=36, R=12.0, hoop=2.0, meridian=-1.0, radial=4.0, h=0.25):
    """Ball of radius R (voxels) with hoop / meridian / radial principal stresses."""
    g = Grid((n, n, n), (-0.5 * n * h,) * 3, h)
    c = _voxels(g, g.element_centers()) - 0.5 * n
    x, y = c[..., 0], c[..., 1]
    r = np.linalg.norm(c, axis=-1)
    rho = np.hypot(x, y)
    er = c / np.maximum(r, 1e-9)[..., None]
    ephi = np.stack([-y, x, np.zeros_like(x)], axis=-1) / np.maximum(rho, 1e-9)[..., None]
    eth = np.cross(ephi, er)
    T = (
        hoop * ephi[..., :, None] * ephi[..., None, :]
        + meridian * eth[..., :, None] * eth[..., None, :]
        + radial * er[..., :, None] * er[..., None, :]
    )
    return g, _voigt(T), r, R


def _block_problem(nx=40, ny=10, nz=10):
    """Cantilever block clamped at x = 0, pushed down at the middle of the free end
    (voxels of 5 cm, E = 1 GPa)."""
    g = Grid((nx, ny, nz), (0.0, 0.0, 0.0), 0.05)
    nodes = np.arange(g.n_nodes).reshape(g.nodes_shape)
    fixed = (3 * nodes[0].ravel()[:, None] + np.arange(3)).ravel()
    forces = np.zeros(g.n_dofs)
    tip = nodes[nx, :, nz // 2 - 1 : nz // 2 + 2].ravel()
    forces[3 * tip + 2] = -1000.0 / tip.size
    return Problem(g, np.ones(g.shape, bool), fixed, forces, E=1e9, nu=0.3)


def _padded_surface_case(problem, stress, pad=2):
    """Surface field of a block as the add-on builds it: the density-filtered domain
    indicator padded with empty layers (so the surface closes inside the grid)."""
    g = problem.grid
    indicator = topopt.DensityFilter(problem.domain, 1.5).forward(problem.domain.astype(float))
    f = np.pad(indicator, pad)
    grid = Grid(f.shape, g.origin - pad * g.h, g.h)
    stress = np.pad(stress, ((pad, pad),) * 3 + ((0, 0),))
    return grid, f, stress, np.pad(problem.domain, pad)


def _trilinear(f, u):
    """Reference trilinear interpolation of an element-centred field at voxel coords."""
    q = np.clip(u - 0.5, 0.0, np.asarray(f.shape) - 1.0)
    i0 = np.minimum(np.floor(q).astype(int), np.asarray(f.shape) - 2)
    t = q - i0
    out = np.zeros(len(u))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (t[:, 0] if a else 1 - t[:, 0]) * (t[:, 1] if b else 1 - t[:, 1])
                w = w * (t[:, 2] if c else 1 - t[:, 2])
                out += w * f[i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c]
    return out


# ------------------------------------------------------------ the stress field
def test_sample_tensor_is_exact_for_linear_fields_and_masked():
    g = Grid((6, 5, 4), (1.0, -2.0, 0.5), 0.5)
    c = g.element_centers()
    coeff = np.arange(18, dtype=float).reshape(6, 3) / 7.0
    s6 = c @ coeff.T + np.arange(6)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    rng = np.random.default_rng(1)
    p = g.origin + (0.5 + rng.random((50, 3)) * (np.array(g.shape) - 1.0)) * g.h
    T = field.sample_tensor(p)
    assert T.shape == (50, 3, 3)
    np.testing.assert_allclose(T, np.swapaxes(T, 1, 2))
    np.testing.assert_allclose(_voigt(T), p @ coeff.T + np.arange(6), atol=1e-12)

    # masked: only solid centres contribute (renormalized), no solid near -> zero
    solid = np.zeros(g.shape, bool)
    solid[:3] = True
    stress = np.zeros(g.shape + (6,))
    stress[:3] = [1.0, 2.0, 3.0, 0.5, 0.25, 0.125]
    stress[3:] = 100.0  # ignored: not solid
    field = sl.StressField(g, stress, solid)
    border = g.origin + np.array([[3.0, 2.0, 2.0], [2.6, 1.1, 0.7]]) * g.h
    np.testing.assert_allclose(_voigt(field.sample_tensor(border)), [stress[0, 0, 0]] * 2)
    far = g.origin + np.array([[5.5, 2.0, 2.0]]) * g.h
    np.testing.assert_allclose(field.sample_tensor(far), 0.0)
    assert field.sample_tensor(np.full((2, 4, 3), 1.2)).shape == (2, 4, 3, 3)


def test_inside_and_non_finite_handling():
    g = Grid((4, 4, 4), (0.0, 0.0, 0.0), 1.0)
    solid = np.ones(g.shape, bool)
    solid[2:, 2:, 2:] = False
    stress = np.ones(g.shape + (6,))
    stress[0, 0, 0, 2] = np.nan
    field = sl.StressField(g, stress, solid)
    assert not field.solid[0, 0, 0] and field.stress[0, 0, 0, 2] == 0.0
    pts = np.array(
        [[1.5, 0.5, 0.5], [3.5, 3.5, 3.5], [-0.1, 1.0, 1.0], [4.0, 1.0, 1.0], [np.nan, 1.0, 1.0],
         [0.5, 0.5, 0.5]]
    )
    np.testing.assert_array_equal(field.inside(pts), [True, False, False, False, False, False])
    with pytest.raises(ValueError):
        sl.StressField(g, np.ones((4, 4, 4, 5)), solid)
    with pytest.raises(ValueError):
        sl.StressField(g, stress, np.ones((3, 4, 4), bool))


def test_closed_form_principal_frame_matches_eigh():
    rng = np.random.default_rng(3)
    T = rng.normal(size=(500, 3, 3))
    T = T + np.swapaxes(T, 1, 2)
    s6 = _voigt(T)
    w, v = np.linalg.eigh(T)
    np.testing.assert_allclose(sl._principal_values(s6), w, atol=1e-10)
    for col in (0, 2):
        values, vec, well = sl._principal_frame(s6, col)
        np.testing.assert_allclose(values, w, atol=1e-10)
        assert well.all()
        np.testing.assert_allclose(np.abs(np.einsum("pi,pi->p", vec, v[:, :, col])), 1.0, atol=1e-8)
    # isotropic tensors have no defined direction
    _values, _vec, well = sl._principal_frame(_voigt(np.eye(3)[None] * 2.0), 2)
    assert not well[0]


# ------------------------------------------------------------- volume mode
def test_uniaxial_field_gives_straight_lines_along_the_principal_axes():
    g, s6 = _uniform((24, 16, 8), [1.0, -0.5, 0.0, 0.0, 0.0, 0.0], origin=(1.0, 2.0, 3.0), h=0.1)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    lines = sl.trace(field, sl.TraceSettings(n_seeds=12, branch_depth=0))
    for family, axis, mag, sign, extent in (
        ("TENSION", 0, 1.0, 1.0, 24), ("COMPRESSION", 1, 0.5, -1.0, 16),
    ):
        group = [ln for ln in lines if ln.family == family]
        assert len(group) >= 4
        for line in group:
            d = _unit_rows(np.diff(line.points, axis=0))
            np.testing.assert_allclose(np.abs(d[:, axis]), 1.0, atol=1e-9)
            np.testing.assert_allclose(line.magnitude, mag, rtol=1e-9)
            np.testing.assert_array_equal(line.sign, sign)
            assert line.generation == 0 and not line.closed
            assert line.points.shape == (len(line.magnitude), 3)
            assert field.inside(line.points).all()
            # runs through the whole solid (stops only when leaving it)
            assert _length(line.points) / g.h > extent - 1.0


def test_pure_shear_rotated_45_degrees():
    g, s6 = _uniform((24, 24, 6), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    lines = sl.trace(field, sl.TraceSettings(n_seeds=10, branch_depth=0))
    expected = {
        "TENSION": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        "COMPRESSION": np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
    }
    for line in lines:
        d = _unit_rows(np.diff(line.points, axis=0))
        np.testing.assert_allclose(np.abs(d @ expected[line.family]), 1.0, atol=1e-9)
        np.testing.assert_allclose(line.magnitude, 1.0, rtol=1e-9)
    assert {ln.family for ln in lines} == {"TENSION", "COMPRESSION"}


def test_ring_field_gives_closed_circles_and_radial_lines():
    g, s6, solid = _ring_field()
    field = sl.StressField(g, s6, solid)
    lines = sl.trace(field, sl.TraceSettings(n_seeds=30, branch_depth=0, separation=2.0))
    hoops = [ln for ln in lines if ln.family == "TENSION"]
    radial = [ln for ln in lines if ln.family == "COMPRESSION"]
    assert len(hoops) >= 4 and len(radial) >= 10
    for line in hoops:
        r = np.hypot(line.points[:, 0], line.points[:, 1])
        assert np.ptp(r) < 0.05
        assert line.closed
        np.testing.assert_array_equal(line.points[0], line.points[-1])
        assert _length(line.points) > 0.95 * 2 * np.pi * r.mean()
        # smooth all around, also where the two halves met (no backward closing segment)
        d = _unit_rows(np.diff(line.points, axis=0))
        turns = np.einsum("ij,ij->i", d, np.roll(d, -1, axis=0))
        assert turns.min() > np.cos(np.radians(20.0))
    for line in radial:
        az = np.unwrap(np.arctan2(line.points[:, 1], line.points[:, 0]))
        assert np.ptp(az) < 0.01 and not line.closed
    for line in lines:
        assert field.inside(line.points).all()


def test_closing_a_loop_never_runs_backwards():
    # the two heads of a loop passed each other within the last step: the forward head
    # (last point) lies beyond the start, the closing segment would turn back by ~180 deg
    theta = np.concatenate([np.arange(0.0, 2 * np.pi, 0.1), [2 * np.pi + 0.03]])
    u = np.stack([10 * np.cos(theta), 10 * np.sin(theta), np.zeros_like(theta)], axis=1)
    v = np.arange(len(u), dtype=float)
    out, vals, k0 = sl._trim_overlap(u, v, 20)
    assert len(out) == len(u) - 1 and k0 == 20
    np.testing.assert_array_equal(vals, v[:-1])
    closing = out[0] - out[-1]
    assert closing @ (out[-1] - out[-2]) > 0.0 and (out[1] - out[0]) @ closing > 0.0
    # the backward head overshot instead: its point goes, the seed index shifts
    u2 = np.concatenate([u[-1:], u[:-1]])
    out, vals, k0 = sl._trim_overlap(u2, v, 20)
    assert len(out) == len(u) - 1 and k0 == 19
    # a clean loop is left alone
    out, vals, k0 = sl._trim_overlap(u[:-1], v[:-1], 5)
    assert len(out) == len(u) - 1 and k0 == 5


def test_lines_stay_inside_the_solid_and_respect_the_separation():
    g, s6 = _cantilever_field(40, 12, 20)
    solid = np.ones(g.shape, bool)
    solid[15:25, :, 7:13] = False  # a hole
    field = sl.StressField(g, s6, solid)
    sep = 2.0
    lines = sl.trace(field, sl.TraceSettings(n_seeds=60, branch_depth=1, separation=sep))
    assert len(lines) > 30
    for line in lines:
        assert field.inside(line.points).all()
        assert len(line.points) >= 5
    for family in sl.FAMILIES:
        d = _same_family_distances(lines, family)
        assert d.size and d.min() >= 0.95 * sep


def test_tracing_is_deterministic():
    g, s6 = _cantilever_field(30, 8, 16)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    st = sl.TraceSettings(n_seeds=30, branch_depth=1, rng_seed=7)
    a = sl.trace(field, st)
    b = sl.trace(field, st)
    assert len(a) > 10 and _identical(a, b)
    c = sl.trace(field, sl.TraceSettings(n_seeds=30, branch_depth=1, rng_seed=8))
    assert not _identical(a, c)


def test_branches_are_generation_one_lines_of_the_other_family():
    g, s6 = _uniform((30, 30, 4), [1.0, -0.6, 0.0, 0.0, 0.0, 0.0])
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    st = sl.TraceSettings(n_seeds=2, branch_depth=1, branch_spacing=6.0, separation=2.0)
    lines = sl.trace(field, st)
    primary = [ln for ln in lines if ln.generation == 0]
    branches = [ln for ln in lines if ln.generation == 1]
    assert len(branches) >= 4
    assert lines.index(branches[0]) >= len(primary)  # branches come after the primary lines
    for line in branches:
        parents = [ln for ln in primary if ln.family != line.family]
        assert _point_line_distance(np.asarray(line.points), parents).min() < 0.75
    # the ratio is |sigma_branch| / |sigma_parent|: 0.6 from tension lines, 1/0.6 from
    # compression lines; a ratio of 0.7 lets only tension branches sprout ...
    some = sl.trace(field, sl.TraceSettings(n_seeds=2, branch_depth=1, branch_ratio=0.7))
    assert {ln.family for ln in some if ln.generation > 0} == {"TENSION"}
    # ... and 2.0 none at all ...
    none = sl.trace(field, sl.TraceSettings(n_seeds=2, branch_depth=1, branch_ratio=2.0))
    assert not [ln for ln in none if ln.generation > 0]
    # ... unless a BRANCH zone lowers it there: ratio * (1 - 0.5 * weight) = 1.0. One
    # explicit seed: a tension line at y = 7.3 and a compression line at x = 5.
    seed = np.array([[5.0, 7.3, 2.0]])
    for zone, expected in ((slice(0, 15), True), (slice(15, 30), False)):
        weight = np.zeros(g.shape)
        weight[zone] = 1.0
        zoned = sl.trace(field, sl.TraceSettings(n_seeds=0, seed_points=seed, branch_depth=1,
                                                 branch_ratio=2.0, branch_spacing=6.0,
                                                 branch_weight=weight))
        assert [ln.family for ln in zoned if ln.generation == 0] == ["TENSION", "COMPRESSION"]
        sprouts = [ln for ln in zoned if ln.generation == 1]
        assert bool(sprouts) == expected
        assert {ln.family for ln in sprouts} <= {"TENSION"}


def test_lines_stop_where_the_stress_vanishes():
    g = Grid((40, 10, 4), (0.0, 0.0, 0.0), 1.0)
    s6 = np.zeros(g.shape + (6,))
    s6[..., 0] = np.clip(1.0 - g.element_centers()[..., 0] / 20.0, 0.0, None)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    st = sl.TraceSettings(n_seeds=10, branch_depth=0, min_stress=0.2, families=("TENSION",))
    lines = sl.trace(field, st)
    assert lines
    ref = np.percentile(np.abs(s6[..., 0]), 99.0)
    x_stop = 20.0 * (1.0 - 0.2 * ref)
    for line in lines:
        assert line.points[:, 0].max() < x_stop + 0.01
        assert line.magnitude.min() >= 0.2 * ref - 1e-9
    # nothing at all where there is no stress
    assert sl.trace(sl.StressField(g, np.zeros(g.shape + (6,)), np.ones(g.shape, bool))) == []


def test_explicit_seed_points_are_traced_first():
    g, s6 = _uniform((30, 20, 6), [1.0, -0.5, 0.0, 0.0, 0.0, 0.0], origin=(0.0, 0.0, 0.0), h=0.5)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    seeds = np.array([[5.2, 3.3, 1.4], [11.0, 7.7, 1.6]])
    lines = sl.trace(field, sl.TraceSettings(n_seeds=5, branch_depth=0, seed_points=seeds))
    first = lines[:4]
    assert [ln.family for ln in first] == ["TENSION", "COMPRESSION"] * 2
    for k, line in enumerate(first):
        seed = seeds[k // 2]
        assert _point_line_distance(seed[None], [line])[0] < 1e-9


def test_max_lines_families_and_invalid_settings():
    g, s6 = _cantilever_field(30, 8, 16)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    assert len(sl.trace(field, sl.TraceSettings(n_seeds=50, max_lines=7))) == 7
    only = sl.trace(field, sl.TraceSettings(families=("compression",), n_seeds=10, branch_depth=0))
    assert only and {ln.family for ln in only} == {"COMPRESSION"}
    assert sl.trace(field, sl.TraceSettings(max_lines=0)) == []
    for bad in (
        dict(families=("SHEAR",)), dict(separation=0.0), dict(step=-1.0), dict(max_steps=0),
        dict(integrator="EULER"), dict(branch_spacing=0.0), dict(seed_points=np.ones(4)),
        dict(branch_weight=np.ones(3)),
    ):
        with pytest.raises(ValueError):
            sl.trace(field, sl.TraceSettings(**bad))
    with pytest.raises(TypeError):
        sl.trace("not a field", sl.TraceSettings())


def test_should_stop_and_progress():
    g, s6 = _cantilever_field(30, 8, 16)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    calls = []
    with pytest.raises(Cancelled):
        sl.trace(field, sl.TraceSettings(), should_stop=lambda: calls.append(1) or len(calls) > 5)
    reports = []
    sl.trace(field, sl.TraceSettings(n_seeds=20), progress=lambda f, m: reports.append((f, m)))
    fractions = [f for f, _m in reports]
    assert fractions[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fractions)
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert all(isinstance(m, str) for _f, m in reports)


def test_rk2_integrator_traces_the_same_straight_lines():
    g, s6 = _uniform((20, 20, 4), [1.0, -0.5, 0.0, 0.0, 0.0, 0.0])
    lines = sl.trace(sl.StressField(g, s6, np.ones(g.shape, bool)),
                     sl.TraceSettings(n_seeds=6, branch_depth=0, integrator="rk2"))
    assert lines
    for line in lines:
        axis = 0 if line.family == "TENSION" else 1
        np.testing.assert_allclose(np.abs(_unit_rows(np.diff(line.points, axis=0))[:, axis]), 1.0)


def test_volume_tracing_performance():
    g, s6 = _cantilever_field(60, 30, 30)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    start = time.perf_counter()
    lines = sl.trace(field, sl.TraceSettings(n_seeds=200, branch_depth=1, max_lines=300))
    elapsed = time.perf_counter() - start
    assert len(lines) == 300
    assert elapsed < 10.0, f"300 lines took {elapsed:.1f} s"


# ------------------------------------------------------------ surface mode
def test_surface_constraint_validation():
    f = np.zeros((4, 4, 4))
    assert sl.TraceSettings().surface is None
    c = sl.SurfaceConstraint(f, 0.5)
    assert c.tolerance == 0.75 and c.smoothing == 0.0 and c.grid is None
    assert c.field.dtype == np.float64
    for bad in (
        dict(field=np.zeros((4, 4)), iso=0.5), dict(field=np.full((2, 2, 2), np.nan), iso=0.5),
        dict(field=f, iso=np.inf), dict(field=f, iso=0.5, tolerance=0.0),
        dict(field=f, iso=0.5, tolerance=np.nan), dict(field=f, iso=0.5, smoothing=-1.0),
        dict(field=f, iso=0.5, grid=Grid((4, 4, 5), (0, 0, 0), 1.0)),
    ):
        with pytest.raises(ValueError):
            sl.SurfaceConstraint(**bad)
    with pytest.raises(TypeError):
        sl.SurfaceConstraint(f, 0.5, grid=(4, 4, 4))
    g, s6 = _uniform((5, 5, 5), [1.0, 0, 0, 0, 0, 0])
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    with pytest.raises(ValueError):
        sl.trace(field, sl.TraceSettings(surface=sl.SurfaceConstraint(np.zeros((4, 5, 5)), 0.5)))
    with pytest.raises(TypeError):
        sl.trace(field, sl.TraceSettings(surface=(np.zeros((5, 5, 5)), 0.5)))
    # no iso crossing: nothing to trace
    flat = sl.SurfaceConstraint(np.ones((5, 5, 5)), 0.5)
    assert sl.trace(field, sl.TraceSettings(surface=flat)) == []


def test_tangential_principal_stresses_match_the_projected_tensor():
    rng = np.random.default_rng(5)
    T = rng.normal(size=(400, 3, 3))
    T = T + np.swapaxes(T, 1, 2)
    n = _unit_rows(rng.normal(size=(400, 3)))
    n[:3] = [[0, 0, 1], [0, 0, -1], [1, 0, 0]]
    hi, lo, e_hi, e_lo, well = sl._tangent_eigen(_voigt(T), n)
    P = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
    w, v = np.linalg.eigh(P @ T @ P)
    drop = np.argmax(np.abs(np.einsum("pi,pij->pj", n, v)), axis=1)
    keep = np.array([[j for j in range(3) if j != d] for d in drop])
    rows = np.arange(len(n))
    w2 = w[rows[:, None], keep]  # ascending
    np.testing.assert_allclose(lo, w2[:, 0], atol=1e-10)
    np.testing.assert_allclose(hi, w2[:, 1], atol=1e-10)
    assert well.all()
    for e, col in ((e_lo, 0), (e_hi, 1)):
        np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0)
        np.testing.assert_allclose(np.einsum("pi,pi->p", e, n), 0.0, atol=1e-12)
        ref = v[rows, :, keep[:, col]]
        np.testing.assert_allclose(np.abs(np.einsum("pi,pi->p", e, ref)), 1.0, atol=1e-7)


def test_surface_projection_honours_the_tolerance():
    g, s6, f, iso, _R = _plate(n=12, nz=6)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    for tol, ok in ((0.75, [True, False]), (1.5, [True, True])):
        surf = sl._Surface(field, sl.SurfaceConstraint(f, iso, tolerance=tol))
        trial = np.array([[5.0, 6.0, iso + 0.5], [5.0, 6.0, iso - 1.0]])
        p, _s6, den, grad, good = surf.settle(trial)
        np.testing.assert_array_equal(good, ok)
        np.testing.assert_allclose(p[good, 2], iso, atol=1e-9)
        np.testing.assert_allclose(p[:, :2], trial[:, :2])
        assert (den > 0).all() and np.allclose(grad, [0.0, 0.0, 1.0])
    # exact crossings between neighbouring centres
    surf = sl._Surface(field, sl.SurfaceConstraint(f, iso))
    cross = surf.crossings()
    assert len(cross) == 12 * 12
    np.testing.assert_allclose(cross[:, 2], iso)


def test_surface_plate_gives_straight_orthogonal_evenly_spaced_lines():
    g, s6, f, iso, R = _plate()
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    sep = 3.0
    lines = sl.trace(field, sl.TraceSettings(separation=sep, n_seeds=20, branch_depth=0,
                                             surface=sl.SurfaceConstraint(f, iso)))
    z0 = g.origin[2] + iso * g.h
    expected = (("TENSION", R[:, 0], 2.0, 1.0), ("COMPRESSION", R[:, 1], 1.0, -1.0))
    for family, axis, mag, sign in expected:
        group = [ln for ln in lines if ln.family == family]
        assert len(group) >= 12
        offsets = []
        for line in group:
            d = _unit_rows(np.diff(line.points, axis=0))
            np.testing.assert_allclose(np.abs(d @ axis), 1.0, atol=1e-9)  # straight, on axis
            np.testing.assert_allclose(line.points[:, 2], z0, atol=1e-9)  # on the mid-plane
            # in-plane principal stress, the normal one (5) is discarded
            np.testing.assert_allclose(line.magnitude, mag, rtol=1e-9)
            np.testing.assert_array_equal(line.sign, sign)
            u = _voxels(g, line.points)
            assert (u[:, :2] >= 0.5 - 1e-9).all() and (u[:, :2] <= 40 - 0.5 + 1e-9).all()
            offsets.append(np.mean(u @ np.cross([0.0, 0.0, 1.0], axis)))
        spacing = np.diff(np.sort(offsets))
        assert spacing.min() >= sl.SURFACE_SEED_RATIO * sep - 1e-6  # evenly spaced ...
        assert abs(np.median(spacing) - sep) < 0.02 * sep  # ... at the separation
        assert np.mean(np.abs(spacing - sep) < 0.05 * sep) > 0.8
    # the two families cross at right angles
    t = _unit_rows(np.diff([ln for ln in lines if ln.family == "TENSION"][0].points, axis=0))[0]
    c = _unit_rows(np.diff([ln for ln in lines if ln.family == "COMPRESSION"][0].points, axis=0))[0]
    assert abs(t @ c) < 1e-9


def test_surface_field_on_its_own_finer_grid():
    """A surface sampled on a finer grid than the stress (``SurfaceConstraint.grid``)."""
    g, s6, _f, _iso, R = _plate(n=30, nz=6, h=0.5)
    up = 3
    fine = Grid(tuple(n * up for n in g.shape), g.origin, g.h / up)
    z = (np.arange(fine.shape[2]) + 0.5) / up  # height in coarse voxels
    f = np.broadcast_to(z[None, None, :], fine.shape).copy()
    iso = 3.2
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    lines = sl.trace(field, sl.TraceSettings(separation=3.0, n_seeds=10, branch_depth=0,
                                             surface=sl.SurfaceConstraint(f, iso, grid=fine)))
    z0 = g.origin[2] + iso * g.h
    assert {ln.family for ln in lines} == {"TENSION", "COMPRESSION"}
    for line in lines:
        axis = R[:, 0] if line.family == "TENSION" else R[:, 1]
        d = _unit_rows(np.diff(line.points, axis=0))
        np.testing.assert_allclose(np.abs(d @ axis), 1.0, atol=1e-9)
        np.testing.assert_allclose(line.points[:, 2], z0, atol=1e-9)
        u = _voxels(fine, line.points)  # stays inside the fine field's centre box
        assert (u[:, :2] >= 0.5 - 1e-9).all() and (u[:, :2] <= 90 - 0.5 + 1e-9).all()
    # spacing in voxels of the stress grid
    offsets = np.sort([np.mean(_voxels(g, ln.points) @ np.cross([0.0, 0.0, 1.0], R[:, 0]))
                       for ln in lines if ln.family == "TENSION"])
    assert abs(np.median(np.diff(offsets)) - 3.0) < 0.06
    # a sphere given on a finer grid than its stress
    gs, s6s, r, Rs = _sphere(n=28, R=9.0)
    fine = Grid(tuple(2 * n for n in gs.shape), gs.origin, gs.h / 2)
    centres = _voxels(gs, fine.element_centers()) - 14.0
    rf = np.linalg.norm(centres, axis=-1)
    lines = sl.trace(sl.StressField(gs, s6s, r <= Rs + 1.0),
                     sl.TraceSettings(separation=2.5, n_seeds=10, branch_depth=0,
                                      surface=sl.SurfaceConstraint(rf, Rs, grid=fine)))
    centre = gs.origin + 14.0 * gs.h
    assert len(lines) >= 10
    for line in lines:
        assert np.abs(np.linalg.norm(line.points - centre, axis=1) / gs.h - Rs).max() < 0.05


def test_surface_line_smoothing_irons_out_stress_noise():
    g, s6, f, iso, _R = _plate(n=40, nz=6, alpha=0.0, h=1.0, origin=(0.0, 0.0, 0.0))
    rng = np.random.default_rng(0)
    noisy = s6 + 0.25 * rng.normal(size=s6.shape)  # element stresses with voxel noise
    field = sl.StressField(g, noisy, np.ones(g.shape, bool))

    def wiggle(lines):  # turning per voxel of length (the exact lines are straight)
        turn, length = 0.0, 0.0
        for line in lines:
            d = np.diff(line.points, axis=0)
            n = np.linalg.norm(d, axis=1)
            d = d / n[:, None]
            turn += np.arccos(np.clip(np.einsum("ij,ij->i", d[:-1], d[1:]), -1.0, 1.0)).sum()
            length += n.sum()
        return turn / length

    out = {}
    for smoothing in (0.0, 1.0):
        st = sl.TraceSettings(separation=3.0, n_seeds=10, branch_depth=0,
                              surface=sl.SurfaceConstraint(f, iso, smoothing=smoothing))
        lines = sl.trace(field, st)
        assert len(lines) >= 16
        for line in lines:  # on the plane, never closer than the stop distance
            np.testing.assert_allclose(line.points[:, 2], iso, atol=1e-6)
        for family in sl.FAMILIES:
            d = _same_family_distances(lines, family)
            assert d.min() >= sl.SURFACE_STOP_RATIO * 3.0 * 0.97
        out[smoothing] = wiggle(lines)
    assert out[1.0] < 0.5 * out[0.0]


def test_surface_sphere_hoops_and_meridians_stay_on_the_sphere():
    g, s6, r, R = _sphere()
    field = sl.StressField(g, s6, r <= R + 1.0)
    sep = 2.5
    lines = sl.trace(field, sl.TraceSettings(separation=sep, n_seeds=20, branch_depth=0,
                                             surface=sl.SurfaceConstraint(r, R)))
    centre = g.origin + 0.5 * np.asarray(g.shape) * g.h
    hoops = [ln for ln in lines if ln.family == "TENSION"]
    meridians = [ln for ln in lines if ln.family == "COMPRESSION"]
    assert len(hoops) >= 10 and len(meridians) >= 15
    for line in lines:
        p = (line.points - centre) / g.h
        assert np.abs(np.linalg.norm(p, axis=1) - R).max() < 0.2
    latitudes = []
    for line in hoops:  # circles of latitude (the stress is singular at the poles)
        p = (line.points - centre) / g.h
        assert np.ptp(p[:, 2]) < 0.1
        rho = np.hypot(p[:, 0], p[:, 1])
        if rho.mean() > 2.0:
            assert line.closed
            np.testing.assert_allclose(rho, rho.mean(), atol=0.1)
        if rho.min() > 4.0:
            np.testing.assert_allclose(line.magnitude, 2.0, rtol=0.05)
        latitudes.append(np.arcsin(np.clip(p[:, 2].mean() / R, -1.0, 1.0)))
    gaps = np.diff(np.sort(latitudes)) * R  # geodesic distance between neighbouring hoops
    assert np.all(np.abs(gaps - sep) < 0.25 * sep)
    for line in meridians:  # meridians: constant azimuth away from the poles
        p = (line.points - centre) / g.h
        away = np.hypot(p[:, 0], p[:, 1]) > 4.0
        runs = np.split(np.arange(len(p)), np.flatnonzero(np.diff(away.astype(int))) + 1)
        for run in runs:
            if away[run[0]] and len(run) > 2:
                az = np.unwrap(np.arctan2(p[run, 1], p[run, 0]))
                assert np.ptp(az) < 0.02
        np.testing.assert_array_equal(line.sign[away], -1.0)


def test_surface_lines_on_a_real_cantilever_block():
    problem = _block_problem()
    result = fem.analyze(problem)
    grid, f, stress, solid = _padded_surface_case(problem, result.stress)
    field = sl.StressField(grid, stress, solid)
    sep = 2.0
    start = time.perf_counter()
    lines = sl.trace(field, sl.TraceSettings(separation=sep, n_seeds=40, branch_depth=0,
                                             surface=sl.SurfaceConstraint(f, 0.5)))
    assert time.perf_counter() - start < 10.0
    h = grid.h
    hi = np.asarray(problem.grid.shape, float)
    for line in lines:
        u = _voxels(problem.grid, line.points)
        # on the block's surface (its voxel edges are rounded by < 0.3 voxel) ...
        inside = np.all((u > 0) & (u < hi), axis=1)
        to_face = np.min(np.concatenate([u, hi - u], axis=1), axis=1)
        outside = np.linalg.norm(np.maximum(np.maximum(-u, u - hi), 0.0), axis=1)
        assert np.where(inside, to_face, outside).max() < 0.35
        # ... exactly on the iso-surface of the trilinear field
        np.testing.assert_allclose(_trilinear(f, _voxels(grid, line.points)), 0.5, atol=0.011)
    counts = {fam: [ln for ln in lines if ln.family == fam] for fam in sl.FAMILIES}
    assert len(counts["TENSION"]) >= 15 and len(counts["COMPRESSION"]) >= 15
    for family, group in counts.items():
        d = _same_family_distances(lines, family) / h
        assert d.min() >= sl.SURFACE_STOP_RATIO * sep * 0.97  # never closer than the stop distance
        assert 0.75 * sep < np.median(d) < 1.15 * sep  # neighbours run at ~the separation
        lengths = np.array([_length(ln.points) / h for ln in group])
        assert lengths.min() >= sep - 1e-6 and lengths.max() > 30.0
    # top fibres in tension along the beam, bottom fibres in compression
    top = [ln for ln in counts["TENSION"] if np.all(_voxels(problem.grid, ln.points)[:, 2] > 9.9)]
    bottom = [
        ln for ln in counts["COMPRESSION"] if np.all(_voxels(problem.grid, ln.points)[:, 2] < 0.1)
    ]
    assert top and bottom
    for line in top + bottom:
        d = _unit_rows(np.diff(line.points, axis=0))
        assert np.median(np.abs(d[:, 0])) > 0.95


def test_surface_explicit_seeds_are_projected_and_traced_first():
    g, s6, f, iso, R = _plate(alpha=0.0)
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    z0 = g.origin[2] + iso * g.h
    seeds = np.array([[6.0, 4.0, z0 + 2.0 * g.h], [12.0, 9.0, z0 - 1.5 * g.h], [50.0, 50.0, 50.0]])
    st = sl.TraceSettings(separation=3.0, n_seeds=10, branch_depth=0, seed_points=seeds,
                          surface=sl.SurfaceConstraint(f, iso))
    lines = sl.trace(field, st)
    first = lines[:4]
    assert [ln.family for ln in first] == ["TENSION", "COMPRESSION"] * 2
    for k, line in enumerate(first):
        target = np.array([seeds[k // 2, 0], seeds[k // 2, 1], z0])  # moved onto the plane
        assert _point_line_distance(target[None], [line])[0] < 1e-9


def test_surface_lines_respect_the_stop_distance_and_are_deterministic():
    g, s6 = _cantilever_field(30, 12, 16)
    f = np.pad(np.ones(g.shape), 2)
    grid = Grid(f.shape, g.origin - 2.0, 1.0)
    stress = np.pad(s6, ((2, 2),) * 3 + ((0, 0),))
    field = sl.StressField(grid, stress, f > 0.5)
    sep = 2.5
    st = sl.TraceSettings(separation=sep, n_seeds=30, branch_depth=1, rng_seed=3,
                          surface=sl.SurfaceConstraint(f, 0.5))
    a = sl.trace(field, st)
    b = sl.trace(field, st)
    assert len(a) > 20 and _identical(a, b)
    for family in sl.FAMILIES:
        d = _same_family_distances(a, family)
        assert d.size and d.min() >= sl.SURFACE_STOP_RATIO * sep * 0.97
    for line in a:
        np.testing.assert_allclose(_trilinear(f, _voxels(grid, line.points)), 0.5, atol=0.011)


def test_surface_lines_stop_in_low_stress_regions():
    g, s6, f, iso, _R = _plate(alpha=0.0, szz=0.0)
    scale = np.clip(1.0 - (np.arange(40) + 0.5) / 20.0, 0.0, None)  # vanishes beyond x = 20
    s6 = s6 * scale[:, None, None, None]
    field = sl.StressField(g, s6, np.ones(g.shape, bool))
    st = sl.TraceSettings(separation=3.0, n_seeds=10, branch_depth=0, min_stress=0.3,
                          surface=sl.SurfaceConstraint(f, iso))
    lines = sl.trace(field, st)
    assert lines
    for line in lines:
        u = _voxels(g, line.points)
        assert u[:, 0].max() < 20.0 * (1.0 - 0.3 * 0.99) + 0.6
        ref = 2.0 if line.family == "TENSION" else 1.0
        assert line.magnitude.min() >= 0.3 * 0.98 * ref * 0.97


def test_surface_should_stop_progress_and_threads():
    g, s6, r, R = _sphere(n=28, R=9.0)
    field = sl.StressField(g, s6, r <= R + 1.0)
    st = sl.TraceSettings(separation=2.5, n_seeds=10, branch_depth=0,
                          surface=sl.SurfaceConstraint(r, R))
    calls = []
    with pytest.raises(Cancelled):
        sl.trace(field, st, should_stop=lambda: calls.append(1) or len(calls) > 3)
    reports = []
    reference = sl.trace(field, st, progress=lambda frac, msg: reports.append(frac))
    assert reports[-1] == 1.0 and all(b >= a for a, b in zip(reports, reports[1:]))
    results = [None, None]

    def work(k):
        results[k] = sl.trace(field, st)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _identical(results[0], reference) and _identical(results[1], reference)


def test_surface_tracing_performance():
    problem = _block_problem(56, 26, 26)
    result = fem.analyze(problem)
    grid, f, stress, solid = _padded_surface_case(problem, result.stress)
    assert grid.shape == (60, 30, 30)
    field = sl.StressField(grid, stress, solid)
    st = sl.TraceSettings(separation=1.5, n_seeds=60, branch_depth=0, max_lines=300,
                          surface=sl.SurfaceConstraint(f, 0.5))
    start = time.perf_counter()
    lines = sl.trace(field, st)
    elapsed = time.perf_counter() - start
    assert len(lines) == 300
    assert {ln.family for ln in lines} == {"TENSION", "COMPRESSION"}
    assert elapsed < 15.0, f"300 surface lines took {elapsed:.1f} s"
