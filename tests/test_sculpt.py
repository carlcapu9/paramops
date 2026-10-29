"""Tests of the implicit tube field of the sculpture (``stress_sculpt.core.sculpt``)."""

import threading
import time
import tracemalloc

import numpy as np
import pytest

from stress_sculpt.core import mesher, sculpt
from stress_sculpt.core.grid import Cancelled, Grid

ISO = sculpt.TUBE_ISO


# ---------------------------------------------------------------------------- helpers
def line(a, b, n=20):
    return np.linspace(np.asarray(a, dtype=float), np.asarray(b, dtype=float), n)


def mesh(field, origin, h):
    return mesher.surface_nets(field, ISO, origin, h)


def n_components(verts, faces):
    return np.unique(mesher.face_components(verts.shape[0], faces)).size


def signed_volume(verts, faces):
    tris = mesher.triangulate(faces)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def field_at(tubes, points, h, blend):
    """Field values exactly at ``points`` (which must lie on the lattice origin + (k + 1/2) h)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    origin = points.min(axis=0) - 0.5 * h - 6 * h
    idx = np.rint((points - origin) / h - 0.5).astype(int)
    assert np.allclose(origin + (idx + 0.5) * h, points)
    shape = tuple(idx.max(axis=0) + 7)
    f, v = sculpt.evaluate_tubes(tubes, shape, origin, h, blend)
    return f[tuple(idx.T)], v[tuple(idx.T)]


def random_tubes(n_tubes, n_points, seed, size=(2.0, 1.0, 1.0), r=(0.03, 0.08)):
    rng = np.random.default_rng(seed)
    tubes = []
    for _ in range(n_tubes):
        p = rng.random(3) * size
        d = rng.normal(size=3)
        pts = [p]
        for _k in range(n_points - 1):
            d = d / np.linalg.norm(d) + 0.3 * rng.normal(size=3)
            p = p + 0.04 * d / np.linalg.norm(d)
            pts.append(p)
        radius = r[0] + (r[1] - r[0]) * rng.random(n_points)
        tubes.append(sculpt.Tube(np.array(pts), radius, rng.random(n_points) * 10.0))
    return tubes


# ----------------------------------------------------------- independent brute force
def reference_field(tubes, shape, origin, h, blend):
    """Direct evaluation of the documented field (every segment at every grid point)."""
    tau = sculpt.blend_tau(blend)
    idx = np.indices(shape).reshape(3, -1).T
    P = np.asarray(origin, dtype=float) + (idx + 0.5) * h
    per_tube = []
    for tube in tubes:
        best = np.zeros(len(P))
        val = np.zeros(len(P))
        direction = np.zeros((len(P), 3))
        part = sculpt._polyline_segments(tube.points, tube.radius, tube.value)
        if part is not None:
            for a, u, L, ra, rb, va, vb, e0, e1 in zip(*part):
                support = max(ra, rb) * sculpt.TUBE_SUPPORT
                rel = P - a
                proj = rel @ u
                perp2 = np.einsum("ij,ij->i", rel, rel) - proj * proj
                in_slab = (proj >= -e0 - 1e-6 * h) & (proj <= L + e1 + 1e-6 * h)
                in_slab &= perp2 <= support * support * (1 + 1e-12)
                t = np.clip(proj / L, 0.0, 1.0) if L > 0 else np.zeros(len(P))
                closest = a + (t * L)[:, None] * u
                d = np.linalg.norm(P - closest, axis=1)
                u2 = (sculpt._T_ISO * d / (ra + t * (rb - ra))) ** 2
                f = np.where(in_slab & (u2 < 1.0), (1.0 - np.minimum(u2, 1.0)) ** 3, 0.0)
                better = f > best
                best = np.where(better, f, best)
                val = np.where(better, va + t * (vb - va), val)
                unit = (closest - P) / np.maximum(d, 1e-300)[:, None]
                direction = np.where(better[:, None], np.where(d[:, None] > 0, unit, 0.0), direction)
        per_tube.append((best, val, direction))
    F = np.max([p[0] for p in per_tube], axis=0)
    if tau == 0.0:
        return F.reshape(shape)
    S = sum(np.expm1(p[0] / tau) for p in per_tube)
    # angle weights; points on an axis have no direction and do not count
    wa = [np.expm1(p[0] / sculpt._TAU_ANGLE) * (np.abs(p[2]).sum(1) > 0) for p in per_tube]
    W = sum(wa)
    Q = sum(w * w for w in wa)
    N = sum(w[:, None] * p[2] for w, p in zip(wa, per_tube))
    pairs = W * W - Q
    ok = pairs > sculpt._PAIR_TOL * W * W
    mis = np.where(ok, (W * W - (N * N).sum(1)) / np.where(ok, pairs, 1.0), 0.0)
    mis = np.clip(mis - sculpt._MIS_DEAD, 0.0, 2.0)
    g = np.where(mis >= 1.0, 1.0, np.sqrt(mis * (2.0 - mis)))  # sin(crease angle), 1 past 90 deg
    G = tau * np.log1p(np.maximum(1.0 + S - np.exp(F / tau), 0.0))
    G = np.minimum(G, F + g * np.maximum(G - F, 0.0))
    t = g * tau
    hi, lo = np.maximum(F, G), np.minimum(F, G)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        extra = np.exp((lo - hi) / t) - np.exp(-hi / t)
    U = hi + np.where(t > 0, t * np.log1p(np.maximum(np.nan_to_num(extra), 0.0)), 0.0)
    return U.reshape(shape)


def mixed_tubes():
    rng = np.random.default_rng(7)
    tubes = []
    for _ in range(4):
        n = int(rng.integers(2, 9))
        p = np.cumsum(rng.normal(size=(n, 3)) * 0.12, axis=0) + 0.3 + 0.2 * rng.random(3)
        tubes.append(sculpt.Tube(p, 0.04 + 0.05 * rng.random(n), rng.random(n) * 10))
    tubes.append(sculpt.Tube([[0.5, 0.5, 0.5]], 0.07, 3.0))  # a lone point: a ball
    pts = [[0.2, 0.2, 0.2], [0.4, 0.2, 0.2], [np.nan, 0, 0], [0.5, 0.3, 0.3], [0.5, 0.3, 0.3], [0.7, 0.4, 0.3]]
    tubes.append(sculpt.Tube(pts, 0.05, 1.0))  # a NaN break and a repeated point
    tubes.append(sculpt.Tube([[0.0, 0.0125, 0.0125], [0.6, 0.0125, 0.0125]], 0.06, 2.0))  # through centres
    return tubes


# ---------------------------------------------------------------------------- single tubes
def test_straight_tube_mesh_radius():
    R = 0.1
    tube = sculpt.Tube(line([0, 0, 0], [1.0, 0, 0], 12), R, 2.0)
    field, origin, h, values = sculpt.tubes_field([tube], R / 3.0, blend=0.5)
    assert h == pytest.approx(R / 3.0)
    verts, quads = mesh(field, origin, h)
    assert n_components(verts, quads) == 1
    assert signed_volume(verts, quads) > 0.0
    body = (verts[:, 0] > 0.15) & (verts[:, 0] < 0.85)
    radius = np.linalg.norm(verts[body, 1:], axis=1)
    assert np.all(np.abs(radius - R) < 0.1 * R)  # within 10 %
    assert abs(radius.mean() - R) < 0.02 * R
    # round caps reach R beyond the ends; the value is carried everywhere
    assert verts[:, 0].min() == pytest.approx(-R, abs=0.1 * R)
    assert verts[:, 0].max() == pytest.approx(1.0 + R, abs=0.1 * R)
    assert np.allclose(values[field > 0.0], 2.0)
    # the grid covers the support (~2.2 R) plus the padding
    assert field.max() <= 1.0 + 1e-12 and field.min() >= 0.0
    assert np.all(field[0] == 0.0) and np.all(field[-1] == 0.0)


@pytest.mark.parametrize("blend", [0.0, 0.02, 0.3, 0.7, 1.0])
def test_lone_tube_surface_exactly_at_radius_for_any_blend(blend):
    R = 0.08
    h = 0.01
    tube = sculpt.Tube(line([0.0, 0.0, 0.0], [0.6, 0.0, 0.0], 7), R)
    # points at distance exactly R from the axis (cell centres of a lattice)
    xs = np.arange(10, 50) * h + 0.5 * h
    pts = np.concatenate([np.c_[xs, np.full_like(xs, R), np.zeros_like(xs)],
                          np.c_[xs, np.zeros_like(xs), np.full_like(xs, -R)]])
    f, _ = field_at([tube], pts, h, blend)
    assert np.allclose(f, ISO, atol=1e-12)
    # and the whole field equals the blend-0 field
    f0, origin, hh, _ = sculpt.tubes_field([tube], 0.02, blend=0.0)
    fb, _ = sculpt.evaluate_tubes([tube], f0.shape, origin, hh, blend)
    assert np.allclose(fb, f0, atol=1e-12)


def test_varying_radius_along_a_segment_and_no_beads_at_joints():
    # A cone sampled at many points: the surface follows the linear radius exactly,
    # also at the joints (no bead where a joint sphere would poke out).
    z = np.linspace(0.0, 0.5, 9)
    r = np.linspace(0.04, 0.12, 9)
    tube = sculpt.Tube(np.c_[0 * z, 0 * z, z], r)
    h = 0.005
    zq = np.arange(20, 96) * h + 0.5 * h  # heights along the cone
    for zz in zq[::5]:
        rad = np.interp(zz, z, r)
        f, _ = field_at([tube], [[rad, 0.0, zz]], 1e-3, 0.5)  # exactly on the cone
        assert f[0] == pytest.approx(ISO, abs=1e-9)
    # surface radius along x at many heights (fine 1-D grid rows)
    xs_n = 400
    hx = 0.15 / xs_n
    for zz in zq:
        f, _ = sculpt.evaluate_tubes([tube], (xs_n, 1, 1), (0.0, -0.5 * hx, zz - 0.5 * hx), hx, 0.0)
        f = f[:, 0, 0]
        k = np.flatnonzero(f >= ISO).max()
        x0 = (k + 0.5) * hx
        crossing = x0 + (f[k] - ISO) / (f[k] - f[k + 1]) * hx
        assert crossing == pytest.approx(np.interp(zz, z, r), abs=2e-4)
    # a single segment with different end radii: f == ISO at d == r(t)
    seg = sculpt.Tube([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.05, 0.15])
    for x in (0.1, 0.35, 0.8):
        rad = 0.05 + 0.1 * x
        f, _ = field_at([seg], [[x, 0.0, rad]], 1e-3, 0.0)
        assert f[0] == pytest.approx(ISO, abs=1e-9)


def test_long_segments_are_split_without_changing_the_field():
    a = sculpt.Tube([[0.0, 0.0, 0.0], [1.2, 1.0, 0.8]], [0.03, 0.08], [0.0, 10.0])
    n = 100
    dense = sculpt.Tube(np.linspace([0, 0, 0], [1.2, 1.0, 0.8], n), np.linspace(0.03, 0.08, n),
                        np.linspace(0.0, 10.0, n))
    f1, origin, h, v1 = sculpt.tubes_field([a], 0.012, blend=0.5)
    f2, v2 = sculpt.evaluate_tubes([dense], f1.shape, origin, h, 0.5)
    assert np.abs(f1 - f2).max() < 1e-12
    assert np.abs(v1 - v2)[f1 > 0].max() < 1e-9
    part = sculpt._polyline_segments(a.points, a.radius, a.value, 0.1)
    length, ext0, ext1 = part[2], part[7], part[8]
    assert length.size == 18 and length.max() <= 0.1 + 1e-12
    assert np.allclose(ext0[1:], 0.0) and np.allclose(ext1[:-1], 0.0)  # collinear joints
    assert ext0[0] > 0.0 and ext1[-1] > 0.0  # round caps at the ends


def test_single_point_tube_is_a_ball():
    R = 0.1
    ball = sculpt.Tube([[0.2, -0.3, 0.5]], R, 4.0)
    field, origin, h, values = sculpt.tubes_field([ball], R / 10.0)
    verts, quads = mesh(field, origin, h)
    dist = np.linalg.norm(verts - [0.2, -0.3, 0.5], axis=1)
    assert np.abs(dist - R).max() < 0.05 * R
    assert signed_volume(verts, quads) == pytest.approx(4.0 / 3.0 * np.pi * R**3, rel=0.03)
    assert np.allclose(values[field > 0], 4.0)


def test_invalid_points_break_the_tube():
    pts = [[0, 0, 0], [0.5, 0, 0], [np.nan, 0, 0], [1.0, 0, 0], [1.5, 0, 0], [1.5, 0, 0]]
    tube = sculpt.Tube(pts, 0.05)
    field, origin, h, _ = sculpt.tubes_field([tube], 0.02)
    verts, quads = mesh(field, origin, h)
    assert n_components(verts, quads) == 2  # [0, 0.5] and [1.0, 1.5]
    radii = np.array([0.05, 0.05, 0.05, -1.0, 0.05, 0.05])
    tube = sculpt.Tube(line([0, 0, 0], [2.5, 0, 0], 6), radii)
    field, origin, h, _ = sculpt.tubes_field([tube], 0.02)
    assert n_components(*mesh(field, origin, h)) == 2
    with pytest.raises(ValueError):
        sculpt.Tube(np.zeros((3, 3)), [0.1, 0.2])


def test_no_valid_tube_and_grid_limit():
    for tubes in ([], [sculpt.Tube(np.full((3, 3), np.nan), 0.1)], [sculpt.Tube(np.zeros((2, 3)), 0.0)]):
        field, origin, h, values = sculpt.tubes_field(tubes, 0.1)
        assert field.shape == (1, 1, 1) and not field.any() and not values.any()
        assert origin.shape == (3,) and h == 0.1
    with pytest.raises(ValueError):
        sculpt.tubes_field([sculpt.Tube(line([0, 0, 0], [10, 10, 10]), 1.0)], 1e-3)
    with pytest.raises(ValueError):
        sculpt.tubes_field([sculpt.Tube(line([0, 0, 0], [1, 0, 0]), 0.1)], 0.0)
    with pytest.raises(ValueError):
        sculpt.evaluate_tubes([], (0, 2, 2), (0, 0, 0), 0.1)


# ---------------------------------------------------------------------------- unions
@pytest.mark.parametrize("blend", [0.0, 0.3, 0.45, 1.0])
def test_matches_brute_force_reference(blend):
    tubes = mixed_tubes()
    f0, origin, h, _ = sculpt.tubes_field(tubes, 0.04, blend=0.0)
    field, _ = sculpt.evaluate_tubes(tubes, f0.shape, origin, h, blend)
    ref = reference_field(tubes, f0.shape, origin, h, blend)
    assert np.abs(field - ref).max() < 1e-9


def test_slabs_batches_and_tube_order_do_not_matter(monkeypatch):
    tubes = mixed_tubes() + random_tubes(6, 12, seed=2)
    f0, origin, h, _ = sculpt.tubes_field(tubes, 0.03, blend=0.0)
    expected = {b: sculpt.evaluate_tubes(tubes, f0.shape, origin, h, b) for b in (0.0, 0.6)}
    shuffled = [tubes[i] for i in np.random.default_rng(0).permutation(len(tubes))]
    for blend, (field, values) in expected.items():
        f2, v2 = sculpt.evaluate_tubes(shuffled, f0.shape, origin, h, blend)
        assert np.allclose(f2, field, atol=1e-10)
        assert np.allclose(v2, values, atol=1e-9)
    monkeypatch.setattr(sculpt, "_SLAB_CELLS", 3000)
    monkeypatch.setattr(sculpt, "_BATCH", 700)
    monkeypatch.setattr(sculpt, "_COLUMN_BATCH", 40)
    for blend, (field, values) in expected.items():
        f3, v3 = sculpt.evaluate_tubes(tubes, f0.shape, origin, h, blend)
        assert np.allclose(f3, field, atol=1e-12)
        assert np.allclose(v3, values, atol=1e-10)


@pytest.mark.parametrize("blend", [0.0, 0.25, 0.5, 1.0])
def test_two_crossing_tubes_form_one_component(blend):
    R = 0.06
    a = sculpt.Tube(line([-0.4, 0, 0], [0.4, 0, 0]), R)
    b = sculpt.Tube(line([0, -0.4, 0.02], [0, 0.4, 0.02]), 0.8 * R)
    field, origin, h, _ = sculpt.tubes_field([a, b], R / 3.0, blend=blend)
    verts, quads = mesh(field, origin, h)
    assert n_components(verts, quads) == 1
    assert verts.shape[0] - mesher.mesh_edges(quads).shape[0] + quads.shape[0] == 2  # a sphere


def test_blend_bridges_a_small_gap_only_when_blending():
    # Two tubes passing 0.3 R apart (3.6 voxels): the soft union bridges the gap at a
    # high blend (opposite gradients count as a crease), the hard union does not.
    R = 0.05
    a = sculpt.Tube(line([-0.3, 0, 0], [0.3, 0, 0]), R)
    b = sculpt.Tube(line([0, -0.3, 2.3 * R], [0, 0.3, 2.3 * R]), R)
    for blend, comps in ((0.0, 2), (0.45, 2), (1.0, 1)):
        field, origin, h, _ = sculpt.tubes_field([a, b], R / 12.0, blend=blend)
        assert n_components(*mesh(field, origin, h)) == comps


def test_crossing_gets_fillets_but_no_bump():
    R = 0.05
    h = R / 10.0
    a = sculpt.Tube(line([-0.4, 0, 0], [0.4, 0, 0]), R)
    b = sculpt.Tube(line([0, -0.4, 0], [0, 0.4, 0]), R)
    top = [[0.5 * h, 0.5 * h, R]]  # next to the top of the crossing (gradients ~4 deg apart)
    crease = [[1.1 * R, 1.1 * R, 0.0]]  # outside both tubes, near their crease at (R, R, 0)
    hard_top = field_at([a, b], top, h, 0.0)[0][0]
    hard_crease = field_at([a, b], crease, h, 0.0)[0][0]
    assert hard_top == pytest.approx(ISO, abs=0.01) and hard_crease < ISO - 0.05
    prev = hard_crease
    for blend in (0.3, 0.6, 1.0):
        tau = sculpt.blend_tau(blend)
        soft_excess = tau * np.log(2.0 * np.exp(hard_top / tau) - 1.0) - hard_top  # metaball-like
        f_top = field_at([a, b], top, h, blend)[0][0]
        assert hard_top <= f_top < hard_top + 0.1 * soft_excess  # the top does not swell
        f_crease = field_at([a, b], crease, h, blend)[0][0]
        assert f_crease > prev  # the fillet grows with the blend ...
        prev = f_crease
    assert prev > ISO  # ... and covers the crease point at blend 1
    # exactly on top of the crossing both gradients are parallel: the hard union, exactly
    origin = np.array([0.0, 0.0, R]) - 6.5 * h  # cell (6, 6, 6) is centred at (0, 0, R)
    shape = (13, 13, 13)
    for blend in (0.4, 1.0):
        f, _ = sculpt.evaluate_tubes([a, b], shape, origin, h, blend)
        assert f[6, 6, 6] == pytest.approx(ISO, abs=1e-12)


def test_coincident_and_coaxial_tubes_do_not_swell():
    R = 0.07
    t = sculpt.Tube(line([0, 0, 0], [0.5, 0, 0], 9), R)
    f1, origin, h, _ = sculpt.tubes_field([t], R / 4.0, blend=0.0)
    for blend in (0.3, 1.0):
        f2, _ = sculpt.evaluate_tubes([t, t, t], f1.shape, origin, h, blend)
        assert np.allclose(f2, f1, atol=1e-12)
    # a thinner coaxial tube inside a thicker one: the outer surface stays put
    inner = sculpt.Tube(line([0.1, 0, 0], [0.4, 0, 0], 5), 0.8 * R)
    for blend in (0.3, 1.0):
        f3, _ = sculpt.evaluate_tubes([t, inner], f1.shape, origin, h, blend)
        verts, _quads = mesh(f3, origin, h)
        body = (verts[:, 0] > 0.15) & (verts[:, 0] < 0.35)
        radius = np.linalg.norm(verts[body, 1:], axis=1)
        assert radius.max() < 1.02 * R


def test_y_junction_keeps_the_trunk_and_fills_the_crotch():
    R = 0.08
    th = np.radians(25)
    trunk = sculpt.Tube(line([0, 0, -0.6], [0, 0, 0], 7), R)
    left = sculpt.Tube(line([0, 0, 0], [-0.6 * np.sin(th), 0, 0.6 * np.cos(th)], 7), 0.75 * R)
    right = sculpt.Tube(line([0, 0, 0], [0.6 * np.sin(th), 0, 0.6 * np.cos(th)], 7), 0.75 * R)
    tubes = [trunk, left, right]
    vol = {}
    for blend in (0.0, 0.45, 1.0):
        field, origin, h, _ = sculpt.tubes_field(tubes, R / 5.0, blend=blend)
        verts, quads = mesh(field, origin, h)
        assert n_components(verts, quads) == 1
        vol[blend] = signed_volume(verts, quads)
        below = (verts[:, 2] > -0.45) & (verts[:, 2] < -0.25)  # trunk below the node
        radius = np.linalg.norm(verts[below, :2], axis=1)
        assert np.abs(radius - R).max() < 0.06 * R
        # just below the node on the outer side: no knuckle
        near = (verts[:, 2] > -0.12) & (verts[:, 2] < -0.02)
        assert np.linalg.norm(verts[near, :2], axis=1).max() < 1.04 * R
    # the crotch fillet adds material, bounded
    assert vol[0.0] < vol[0.45] < vol[1.0] < 1.06 * vol[0.0]


def test_blend_is_monotonic_and_bounded():
    tubes = random_tubes(8, 10, seed=4) + [
        sculpt.Tube(line([0, 0, 0], [0, 0, 0.5]), 0.08),
        sculpt.Tube(line([0, 0, 0.5], [0.3, 0, 0.9]), 0.06),
        sculpt.Tube(line([0, 0, 0.5], [-0.3, 0, 0.9]), 0.06),
    ]
    f0, origin, h, _ = sculpt.tubes_field(tubes, 0.03, blend=0.0)
    prev = f0
    n_max = len(tubes)
    for blend in np.linspace(0.05, 1.0, 9):
        f, _ = sculpt.evaluate_tubes(tubes, f0.shape, origin, h, blend)
        assert np.all(f >= prev - 1e-12)
        assert np.all(f <= f0 + sculpt.blend_tau(blend) * np.log(n_max) + 1e-12)
        prev = f
    assert sculpt.blend_tau(0.0) == 0.0 and sculpt.blend_tau(0.01) == 0.0
    assert sculpt.blend_tau(1.0) == sculpt.TAU_MAX and sculpt.blend_tau(5.0) == sculpt.TAU_MAX
    assert sculpt.blend_tau(float("nan")) == 0.0


def test_value_field_follows_the_dominant_tube():
    R = 0.06
    a = sculpt.Tube(line([-0.5, 0, 0], [0.5, 0, 0]), R, 1.0)
    b = sculpt.Tube(line([0, -0.5, 0], [0, 0.5, 0]), R, np.linspace(5.0, 7.0, 20))
    h = R / 4.0
    for blend in (0.0, 0.5):
        field, origin, hh, values = sculpt.tubes_field([a, b], h, blend=blend)
        centres = Grid(field.shape, origin, hh).element_centers()
        far_a = (np.abs(centres[..., 0]) > 0.3) & (field > 0.05)
        far_b = (np.abs(centres[..., 1]) > 0.3) & (np.abs(centres[..., 1]) <= 0.5) & (field > 0.05)
        assert np.allclose(values[far_a], 1.0)
        expected_b = 6.0 + 2.0 * centres[..., 1] / 1.0  # linear along b: 5 at y=-0.5, 7 at 0.5
        assert np.allclose(values[far_b], expected_b[far_b], atol=1e-9)
        near_crossing = (np.abs(centres[..., 0]) < 0.05) & (np.abs(centres[..., 1]) < 0.05) & (field > 0.1)
        v = values[near_crossing]
        assert np.all((v >= 1.0 - 1e-9) & (v <= 7.0 + 1e-9))
        if blend == 0.0:  # the value of the tube with the larger field
            fa, va = sculpt.evaluate_tubes([a], field.shape, origin, hh, 0.0)
            fb, vb = sculpt.evaluate_tubes([b], field.shape, origin, hh, 0.0)
            pick = np.where(fb > fa, vb, va)
            reached = (field > 0) & (fa != fb)
            assert np.allclose(values[reached], pick[reached])
        else:
            assert v.min() < 4.0 < v.max()  # colors blend at the crossing
    assert np.all(values[field == 0.0] == 0.0)


def test_union_fields_matches_the_tubes_rule():
    R = 0.08
    cases = [
        (sculpt.Tube(line([-0.4, 0, 0], [0.4, 0, 0]), R), sculpt.Tube(line([0, -0.4, 0], [0, 0.4, 0]), R)),
        (sculpt.Tube(line([-0.4, 0, 0], [0.4, 0, 0]), R),
         sculpt.Tube(line([-0.35, -0.2, 0.02], [0.35, 0.2, 0.02]), 0.8 * R)),
        (sculpt.Tube(line([0, 0, -0.4], [0, 0, 0]), R), sculpt.Tube(line([0, 0, -0.1], [0, 0, 0.4]), 0.8 * R)),
    ]
    for ta, tb in cases:
        f0, origin, h, _ = sculpt.tubes_field([ta, tb], R / 5.0, blend=0.0)
        fa, _ = sculpt.evaluate_tubes([ta], f0.shape, origin, h, 0.0)
        fb, _ = sculpt.evaluate_tubes([tb], f0.shape, origin, h, 0.0)
        assert np.array_equal(sculpt.union_fields(fa, fb, 0.0), np.maximum(fa, fb))
        for blend in (0.3, 1.0):
            both, _ = sculpt.evaluate_tubes([ta, tb], f0.shape, origin, h, blend)
            u = sculpt.union_fields(fa, fb, blend)
            near = (both > 0.3) & (both < 0.7)
            # analytic vs central-difference gradients: close at the surface
            assert np.abs(u - both)[near].max() < 3e-3
            assert signed_volume(*mesh(u, origin, h)) == pytest.approx(signed_volume(*mesh(both, origin, h)), rel=2e-3)


def test_union_fields_properties():
    rng = np.random.default_rng(3)
    a = rng.random((12, 13, 14))
    b = rng.random((12, 13, 14))
    assert np.array_equal(sculpt.union_fields(a, np.zeros_like(a), 0.7), a)
    assert np.allclose(sculpt.union_fields(a, b, 0.7), sculpt.union_fields(b, a, 0.7), atol=1e-15)
    hi = np.maximum(a, b)
    prev = hi
    for blend in np.linspace(0.0, 1.0, 11):
        u = sculpt.union_fields(a, b, blend)
        assert u.dtype == np.float64 and u.shape == a.shape
        assert np.all(u >= prev - 1e-12)
        assert np.all(u <= hi + sculpt.blend_tau(blend) * np.log(2.0) + 1e-12)
        prev = u
    assert np.array_equal(sculpt.union_fields(-a, -b, 0.5), np.zeros_like(a))  # negatives count as 0
    # broadcasting, scalars, empty arrays, axes of length 1
    assert sculpt.union_fields(np.zeros((3, 1, 4)), np.ones((1, 2, 1)), 0.5).shape == (3, 2, 4)
    assert sculpt.union_fields(np.zeros((0, 3)), np.zeros((0, 3)), 0.5).shape == (0, 3)
    s = sculpt.union_fields(0.5, 0.4, 1.0)  # no gradient (no axis of 2 samples): the maximum
    assert s.shape == () and float(s) == 0.5
    # parallel gradients (1-D): no extra material
    assert np.allclose(sculpt.union_fields([0.5, 0.4, 0.3], [0.5, 0.45, 0.4], 1.0), [0.5, 0.45, 0.4])
    # independent of the blocking
    big = rng.random((70, 40, 41))
    big2 = rng.random((70, 40, 41))
    ref = sculpt.union_fields(big, big2, 0.8)
    old = sculpt._UNION_BLOCK
    try:
        sculpt._UNION_BLOCK = 4000
        assert np.allclose(sculpt.union_fields(big, big2, 0.8), ref, atol=1e-14)
    finally:
        sculpt._UNION_BLOCK = old


# ---------------------------------------------------------------------------- resampling
def test_resample_to_is_exact_for_linear_fields():
    src = Grid((9, 7, 8), (0.1, -0.3, 0.2), 0.13)
    A = np.array([0.7, -1.3, 2.1])
    lin = src.element_centers() @ A + 0.5
    dst = Grid((15, 11, 13), src.origin + 0.5 * src.h, 0.071)  # inside the source centres
    out = sculpt.resample_to(lin, src.origin, src.h, dst.shape, dst.origin, dst.h)
    assert out.shape == dst.shape and out.dtype == np.float64
    centres = dst.element_centers()
    lo = src.origin + 0.5 * src.h
    hi = src.origin + (np.asarray(src.shape) - 0.5) * src.h
    exact = np.clip(centres, lo, hi) @ A + 0.5  # clamped beyond the source centres
    assert np.allclose(out, exact, atol=1e-12)
    # trailing channels, coarser target, far away (clamped) target
    lin2 = np.stack([lin, 2 * lin, -lin], axis=-1)
    out2 = sculpt.resample_to(lin2, src.origin, src.h, (4, 3, 5), src.origin, 0.3)
    c2 = Grid((4, 3, 5), src.origin, 0.3).element_centers()
    e2 = np.clip(c2, lo, hi) @ A + 0.5
    assert np.allclose(out2, np.stack([e2, 2 * e2, -e2], axis=-1), atol=1e-12)
    far = sculpt.resample_to(lin, src.origin, src.h, (2, 2, 2), src.origin + 50.0, 0.1)
    assert np.allclose(far, hi @ A + 0.5)
    # identity
    same = sculpt.resample_to(lin, src.origin, src.h, src.shape, src.origin, src.h)
    assert np.allclose(same, lin, atol=1e-12)
    with pytest.raises(ValueError):
        sculpt.resample_to(lin[0], src.origin, src.h, (2, 2, 2), src.origin, 0.1)
    with pytest.raises(ValueError):
        sculpt.resample_to(lin, src.origin, src.h, (2, 0, 2), src.origin, 0.1)


def test_resample_of_a_tube_field_keeps_the_surface():
    R = 0.08
    tube = sculpt.Tube(line([0, 0, 0], [0.5, 0.2, 0.1], 6), R)
    field, origin, h, _ = sculpt.tubes_field([tube], R / 6.0)
    dst_h = R / 4.0
    shape = tuple(int(n) for n in np.ceil(np.asarray(field.shape) * h / dst_h))
    out = sculpt.resample_to(field, origin, h, shape, origin, dst_h)
    v1 = signed_volume(*mesh(field, origin, h))
    v2 = signed_volume(*mesh(out, origin, dst_h))
    assert v2 == pytest.approx(v1, rel=0.03)


# ---------------------------------------------------------------------------- behaviour
def test_progress_and_should_stop(monkeypatch):
    tubes = random_tubes(20, 15, seed=1, size=(1.0, 0.5, 0.5))
    calls = []
    monkeypatch.setattr(sculpt, "_SLAB_CELLS", 20000)  # several slabs and batches
    monkeypatch.setattr(sculpt, "_BATCH", 5000)
    sculpt.tubes_field(tubes, 0.01, progress=lambda f, m: calls.append((f, m)))
    fractions = [f for f, _ in calls]
    assert fractions and all(0.0 <= f <= 1.0 + 1e-12 for f in fractions)
    assert all(b >= a - 1e-12 for a, b in zip(fractions, fractions[1:]))
    assert fractions[-1] == pytest.approx(1.0)
    with pytest.raises(Cancelled):
        sculpt.tubes_field(tubes, 0.01, should_stop=lambda: True)
    counter = {"n": 0}

    def stop_later():
        counter["n"] += 1
        return counter["n"] > 3

    with pytest.raises(Cancelled):
        sculpt.tubes_field(tubes, 0.01, blend=0.5, should_stop=stop_later)
    assert counter["n"] == 4


def test_thread_safe():
    tubes = random_tubes(10, 12, seed=8)
    f0 = sculpt.tubes_field(tubes, 0.015, blend=0.5)[0]
    results = [None] * 4

    def work(k):
        results[k] = sculpt.tubes_field(tubes, 0.015, blend=0.5)[0]

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        assert np.array_equal(r, f0)


def test_duck_typed_tubes_and_forms_trees():
    class Line:
        def __init__(self, points, radius):
            self.points = points
            self.radius = radius

    duck = Line(line([0, 0, 0], [0.4, 0, 0]), 0.05)
    f1 = sculpt.tubes_field([duck], 0.02)[0]
    f2 = sculpt.tubes_field([sculpt.Tube(duck.points, 0.05)], 0.02)[0]
    assert np.array_equal(f1, f2)
    from stress_sculpt.core import forms

    pts = np.array([(x, y, 3.0) for x in (-1, 1) for y in (-1, 1)], dtype=float)
    tree = forms.grow_tree(pts, np.tile([0.0, 0.0, -1000.0], (4, 1)), ground=0.0)
    tubes = tree.tubes()
    field, origin, h, values = sculpt.tubes_field(tubes, float(tree.radii.min()) / 2.0, blend=0.45)
    verts, quads = mesh(field, origin, h)
    assert n_components(verts, quads) == 1  # one connected structure, footing to every load
    assert values.max() == pytest.approx(tree.forces.max(), rel=1e-9)


def test_branching_structure_surface_is_clean():
    """A form-found tree, meshed and smoothed as the add-on does: one closed, smooth piece."""
    from stress_sculpt.core import forms

    pts = np.array([(x, y, 3.0) for x in (-1, 1) for y in (-1, 1)] + [(0, -1, 3.0), (0, 1, 3.0)], dtype=float)
    settings = forms.TreeSettings(radius_min=0.04, radius_max=0.15, samples=10)
    tree = forms.grow_tree(pts, np.tile([0.0, 0.0, -2000.0], (len(pts), 1)), settings, ground=0.0)
    for blend in (0.0, 0.45, 1.0):
        field, origin, h, _values = sculpt.tubes_field(tree.tubes(), 0.04 / 2.0, blend=blend)
        verts, quads = mesh(field, origin, h)
        verts, quads = mesher.remove_small_components(verts, quads, 16)
        assert n_components(verts, quads) == 1
        assert verts.shape[0] - mesher.mesh_edges(quads).shape[0] + quads.shape[0] == 2  # no handles
        smooth = mesher.taubin_smooth(verts, quads, iterations=12)
        assert signed_volume(smooth, quads) == pytest.approx(signed_volume(verts, quads), rel=0.02)
        # every quad faces outwards (along -grad f): no folds, no inverted spikes
        normal = np.cross(smooth[quads[:, 2]] - smooth[quads[:, 0]], smooth[quads[:, 3]] - smooth[quads[:, 1]])
        centre = smooth[quads].mean(axis=1)
        eps = 0.1 * h
        grad = np.stack([mesher.sample_trilinear(field, origin, h, centre + eps * e)
                         - mesher.sample_trilinear(field, origin, h, centre - eps * e) for e in np.eye(3)], 1)
        assert np.all(np.einsum("ij,ij->i", normal, -grad) > 0.0)
        # smooth: sharp creases only where members meet hard (blend 0)
        tris = mesher.triangulate(quads)
        n = np.cross(smooth[tris[:, 1]] - smooth[tris[:, 0]], smooth[tris[:, 2]] - smooth[tris[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        key = np.minimum(edges[:, 0], edges[:, 1]) * smooth.shape[0] + np.maximum(edges[:, 0], edges[:, 1])
        pair = np.tile(np.arange(tris.shape[0]), 3)[np.argsort(key, kind="stable")].reshape(-1, 2)
        dihedral = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", n[pair[:, 0]], n[pair[:, 1]]), -1, 1)))
        assert np.percentile(dihedral, 99) < 25.0
        if blend > 0.0:
            assert dihedral.max() < 75.0


def test_memory_and_time_on_200_tubes():
    tubes = random_tubes(200, 40, seed=0, size=(3.0, 1.0, 1.2), r=(0.02, 0.05))
    voxel = 0.02
    tracemalloc.start()
    t0 = time.perf_counter()
    field, origin, h, values = sculpt.tubes_field(tubes, voxel, blend=0.5)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    outputs = field.nbytes + values.nbytes
    assert field.size > 500_000
    assert peak - outputs < 160e6  # temporaries are bounded, whatever the grid size
    assert elapsed < 20.0
    _verts, quads = mesh(field, origin, h)
    assert quads.shape[0] > 10_000
