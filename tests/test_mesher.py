"""Tests of the isosurface mesher and mesh utilities (``stress_sculpt.core.mesher``)."""

import time

import numpy as np
import pytest

from stress_sculpt.core import mesher
from stress_sculpt.core.grid import Grid


# ---------------------------------------------------------------------------- helpers
def centred_field(n, fn, lo=-1.0, hi=1.0):
    """Field ``fn(X, Y, Z)`` sampled at the element centres of a cube grid."""
    if np.isscalar(n):
        n = (n, n, n)
    h = (hi - lo) / max(n)
    axes = [lo + (np.arange(k) + 0.5) * h for k in n]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return fn(X, Y, Z), np.full(3, lo), h


def sphere_sdf(r, c=(0.0, 0.0, 0.0)):
    return lambda x, y, z: r - np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)


def signed_volume(verts, faces):
    tris = mesher.triangulate(faces) if faces.shape[1] == 4 else faces
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def surface_area(verts, faces):
    tris = mesher.triangulate(faces) if faces.shape[1] == 4 else faces
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def euler_characteristic(verts, faces):
    return verts.shape[0] - mesher.mesh_edges(faces).shape[0] + faces.shape[0]


def n_components(verts, faces):
    labels = mesher._component_labels(verts.shape[0], mesher.mesh_edges(faces))
    return np.unique(labels[np.unique(faces)]).size


def edge_face_counts(faces):
    """Counts of faces per undirected edge and per directed edge."""
    a = faces.ravel()
    b = np.roll(faces, -1, axis=1).ravel()
    n = int(faces.max()) + 1
    undirected = np.unique(np.minimum(a, b) * n + np.maximum(a, b), return_counts=True)[1]
    directed = np.unique(a * n + b, return_counts=True)[1]
    return undirected, directed


def vertex_link_cycles(verts, faces):
    """Number of closed fans (link cycles) around every vertex; 1 everywhere = manifold."""
    k = faces.shape[1]
    v = faces.ravel()
    nxt = np.roll(faces, -1, axis=1).ravel()
    prv = np.roll(faces, 1, axis=1).ravel()
    n = verts.shape[0]
    keys, inverse = np.unique(np.concatenate([v * n + nxt, v * n + prv]), return_inverse=True)
    m = v.size
    arcs = np.stack([inverse[:m], inverse[m:]], axis=1)  # corner joins its two edges at v
    labels = mesher._component_labels(keys.size, arcs)
    owner = keys // n
    roots = np.unique(labels)
    return np.bincount(owner[roots], minlength=n), k


def upsample_trilinear(field, s):
    """Trilinear interpolant of a sample lattice at the centres of ``s**3`` sub-cells per cube."""
    out = np.asarray(field, dtype=np.float64)
    for axis in range(3):
        out = np.moveaxis(out, axis, 0)
        w = ((np.arange(s) + 0.5) / s).reshape((1, s) + (1,) * (out.ndim - 1))
        up = out[:-1][:, None] * (1.0 - w) + out[1:][:, None] * w  # (n - 1, s, ...)
        out = np.moveaxis(up.reshape((-1,) + out.shape[1:]), 0, axis)
    return out


def solid_euler(mask):
    """Euler characteristic of the union of the closed unit cubes of a 3-D boolean mask."""
    p = np.pad(np.asarray(mask, dtype=bool), 1)
    a, b, c = p.shape
    faces = sum(
        int(np.count_nonzero(np.take(p, range(n - 1), axis=ax) | np.take(p, range(1, n), axis=ax)))
        for ax, n in enumerate(p.shape)
    )
    edges = 0
    for ax in range(3):
        q = np.moveaxis(p, ax, 0)
        edges += int(np.count_nonzero(q[:, :-1, :-1] | q[:, 1:, :-1] | q[:, :-1, 1:] | q[:, 1:, 1:]))
    corners = np.zeros((a - 1, b - 1, c - 1), dtype=bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corners |= p[dx : dx + a - 1, dy : dy + b - 1, dz : dz + c - 1]
    return int(np.count_nonzero(corners)) - edges + faces - int(p.sum())


def assert_closed_manifold(verts, faces):
    """Closed, consistently oriented, edge- and vertex-manifold, no degenerate faces."""
    assert verts.dtype == np.float64 and faces.dtype == np.int64
    assert faces.ndim == 2 and faces.shape[1] == 4 and verts.shape[1] == 3
    assert np.isfinite(verts).all()
    assert faces.min() >= 0 and faces.max() < verts.shape[0]
    assert np.array_equal(np.unique(faces), np.arange(verts.shape[0])), "unused vertices"
    srt = np.sort(faces, axis=1)
    assert np.all(srt[:, 1:] != srt[:, :-1]), "a face repeats a vertex"
    undirected, directed = edge_face_counts(faces)
    assert np.all(undirected == 2), f"edges not in exactly 2 faces: {np.bincount(undirected)}"
    assert np.all(directed == 1), "inconsistent orientation"
    cycles, _k = vertex_link_cycles(verts, faces)
    assert np.all(cycles == 1), f"non-manifold vertices: {int(np.sum(cycles != 1))}"


# ---------------------------------------------------------------------------- surface nets
def test_sphere_closed_manifold_volume_area_normals():
    r = 0.7
    field, origin, h = centred_field(40, sphere_sdf(r))
    verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    assert_closed_manifold(verts, quads)
    assert euler_characteristic(verts, quads) == 2
    assert n_components(verts, quads) == 1
    volume = signed_volume(verts, quads)
    assert volume > 0.0  # outward normals
    assert volume == pytest.approx(4.0 / 3.0 * np.pi * r**3, rel=0.02)
    assert surface_area(verts, quads) == pytest.approx(4.0 * np.pi * r**2, rel=0.02)
    radius = np.linalg.norm(verts, axis=1)
    assert np.abs(radius - r).max() < 0.25 * h
    normals = mesher.vertex_normals(verts, quads)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.min(np.einsum("ij,ij->i", normals, verts / radius[:, None])) > 0.95


def test_sphere_coarse_and_offset_grid():
    # Non-cubic grid, odd sizes, arbitrary origin and spacing.
    g = Grid((23, 17, 19), (3.0, -2.0, 0.5), 0.13)
    c = g.origin + 0.5 * g.size
    pts = g.element_centers()
    field = 0.8 - np.linalg.norm(pts - c, axis=-1)
    verts, quads = mesher.surface_nets(field, 0.0, g.origin, g.h)
    assert_closed_manifold(verts, quads)
    assert euler_characteristic(verts, quads) == 2
    assert signed_volume(verts, quads) == pytest.approx(4.0 / 3.0 * np.pi * 0.8**3, rel=0.05)
    assert np.allclose(verts.mean(axis=0), c, atol=0.02)


def test_torus_genus_one():
    R, r = 0.6, 0.25
    field, origin, h = centred_field(
        48, lambda x, y, z: r - np.sqrt((np.sqrt(x * x + y * y) - R) ** 2 + z * z)
    )
    verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    assert_closed_manifold(verts, quads)
    assert euler_characteristic(verts, quads) == 0
    assert signed_volume(verts, quads) == pytest.approx(2 * np.pi**2 * R * r * r, rel=0.03)


def test_binary_box_meshes_on_voxel_faces():
    field = np.zeros((10, 12, 8))
    field[2:7, 3:9, 1:5] = 1.0
    origin = np.array([0.5, -1.0, 2.0])
    h = 0.25
    verts, quads = mesher.surface_nets(field, 0.5, origin, h)
    assert_closed_manifold(verts, quads)
    assert euler_characteristic(verts, quads) == 2
    lo = origin + np.array([2, 3, 1]) * h
    hi = origin + np.array([7, 9, 5]) * h
    assert np.allclose(verts.min(axis=0), lo) and np.allclose(verts.max(axis=0), hi)
    # Flat faces, bevelled edges and corners: a bit less than the voxel volume.
    volume = signed_volume(verts, quads)
    assert 0.85 * 120 * h**3 < volume < 120 * h**3


def test_full_field_closes_at_the_grid_border():
    origin = np.array([-1.0, 0.0, 2.0])
    h = 0.5
    verts, quads = mesher.surface_nets(np.ones((4, 3, 5)), 0.5, origin, h)
    assert_closed_manifold(verts, quads)
    assert np.allclose(verts.min(axis=0), origin)
    assert np.allclose(verts.max(axis=0), origin + np.array([4, 3, 5]) * h)
    # One single voxel: a closed cube-like octagon-faced blob (8 vertices, 6 quads).
    verts, quads = mesher.surface_nets(np.ones((1, 1, 1)), 0.5, (0, 0, 0), 1.0)
    assert_closed_manifold(verts, quads)
    assert verts.shape == (8, 3) and quads.shape == (6, 4)
    assert signed_volume(verts, quads) > 0.0


@pytest.mark.parametrize(
    "field",
    [np.zeros((5, 6, 7)), np.full((3, 3, 3), 0.2), np.zeros((0, 4, 4)), np.full((2, 2, 2), np.nan)],
)
def test_empty_results(field):
    verts, quads = mesher.surface_nets(field, 0.5, (0, 0, 0), 1.0)
    assert verts.shape == (0, 3) and quads.shape == (0, 4)
    assert verts.dtype == np.float64 and quads.dtype == np.int64


def test_iso_exactly_on_samples_and_nan():
    field, origin, h = centred_field(20, sphere_sdf(0.6))
    field = np.round(field / h) * h  # many samples exactly on the iso level
    field[3, 4, 5] = np.nan
    field[10, 10, 10] = np.nan  # a NaN inside the sphere is a tiny hole (outside)
    verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    assert_closed_manifold(verts, quads)
    assert n_components(verts, quads) == 2  # the sphere and the NaN cavity
    assert signed_volume(verts, quads) > 0.0


def test_inside_padding_encloses_the_outside_cells():
    field = np.zeros((6, 6, 6))
    field[:3] = 1.0
    verts, quads = mesher.surface_nets(field, 0.5, (0, 0, 0), 1.0, pad_value=1.0)
    assert_closed_manifold(verts, quads)
    # The outside block x in [3, 6] is now a cavity: normals point into it.
    assert np.allclose(verts.min(axis=0), (3.0, 0.0, 0.0))
    assert np.allclose(verts.max(axis=0), (6.0, 6.0, 6.0))
    assert -3 * 6 * 6 < signed_volume(verts, quads) < -0.85 * 3 * 6 * 6
    verts, quads = mesher.surface_nets(field, 0.5, (0, 0, 0), 1.0, pad_value=0.5)
    assert_closed_manifold(verts, quads)
    assert signed_volume(verts, quads) < 0.0


def touching_blobs(offset, direction, gap, n=26):
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    r = 0.35
    c1 = np.asarray(offset) - (r + gap / 2) * d
    c2 = np.asarray(offset) + (r + gap / 2) * d
    return centred_field(
        n, lambda x, y, z: np.maximum(sphere_sdf(r, c1)(x, y, z), sphere_sdf(r, c2)(x, y, z))
    )


@pytest.mark.parametrize("direction", [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0.3, 0.1)])
@pytest.mark.parametrize("gap", [-0.02, 0.0, 0.01])
@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (0.021, 0.013, -0.017)])
def test_touching_blobs_stay_manifold(direction, gap, offset):
    field, origin, h = touching_blobs(offset, direction, gap)
    verts, quads, info = mesher._surface_nets(field, 0.0, origin, h)
    assert_closed_manifold(verts, quads)  # never a pinch (an edge or vertex shared by 4 faces)
    assert info["residual_conflicts"] == 0
    comps = n_components(verts, quads)
    chi = euler_characteristic(verts, quads)
    assert comps in (1, 2) and chi % 2 == 0
    if gap > 0.5 * h:
        assert comps == 2
    if any(offset):
        # generic sampling: one or two spheres, no handles
        assert chi == 2 * comps
    # The contact region has exactly the topology of the trilinear interpolant of the
    # samples (supersampled). With samples symmetric about the contact point the
    # interpolant really has handles there (the (1, 1, 0) cases have genus 2).
    centre = np.rint((np.asarray(offset) - origin) / h - 0.5).astype(int)
    window = field[tuple(slice(c - 4, c + 4) for c in centre)]
    pad = mesher._default_pad(window, 0.0)
    wv, wq = mesher.surface_nets(window, 0.0, (0, 0, 0), 1.0)
    dense = upsample_trilinear(np.pad(window, 1, constant_values=pad), 16)
    assert euler_characteristic(wv, wq) == 2 * solid_euler(dense >= 0.0)


def test_diagonal_voxels_binary():
    # Voxels touching only along an edge or a corner: separate closed blobs.
    field = np.zeros((4, 4, 4))
    field[1, 1, 1] = field[2, 2, 1] = 1.0  # edge contact
    verts, quads = mesher.surface_nets(field, 0.5, (0, 0, 0), 1.0)
    assert_closed_manifold(verts, quads)
    assert n_components(verts, quads) == 2
    field = np.zeros((4, 4, 4))
    field[1, 1, 1] = field[2, 2, 2] = 1.0  # corner contact
    verts, quads = mesher.surface_nets(field, 0.5, (0, 0, 0), 1.0)
    assert_closed_manifold(verts, quads)
    assert n_components(verts, quads) == 2
    # A diagonal chain whose two ends are also joined around: the face is connected
    # so the surface is a ring (torus), not a pinched edge.
    field = np.zeros((5, 5, 4))
    field[1, 1, 1] = field[2, 2, 1] = 1.0
    field[1, 1, 2] = field[2, 1, 2] = field[2, 2, 2] = 1.0  # path around, one layer up
    field[1, 1, 0] = field[2, 1, 0] = field[2, 2, 0] = 1.0  # and one layer down
    verts, quads, info = mesher._surface_nets(field, 0.5, (0, 0, 0), 1.0)
    assert_closed_manifold(verts, quads)
    assert info["flipped_faces"] >= 1


def test_thin_diagonal_tube_stays_connected():
    # A thin smooth tube along a face diagonal, its axis half-way between two sample
    # planes: only the diagonal samples of both planes are inside, and they must join
    # through the faces (asymptotic decider), not fall apart into beads.
    n = 24
    h = 2.0 / n  # spacing of centred_field(n, ...); z = 0 lies between two sample planes

    def tube(axis_z):
        return lambda x, y, z: 0.7 * h - np.sqrt(((x - y) / np.sqrt(2)) ** 2 + (z - axis_z) ** 2)

    field, origin, h = centred_field(n, tube(0.0))
    verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    assert_closed_manifold(verts, quads)
    assert n_components(verts, quads) == 1
    assert euler_characteristic(verts, quads) == 2
    # Axis on a sample plane: a single chain of diagonal samples. Joining them would
    # need two patches meeting twice through each face (an edge in four quads), so the
    # faces are flipped: separate beads, but never a non-manifold pinch.
    field, origin, h = centred_field(n, tube(0.5 * h))
    verts, quads, info = mesher._surface_nets(field, 0.0, origin, h)
    assert_closed_manifold(verts, quads)
    assert info["flipped_faces"] > 0 and info["residual_conflicts"] == 0


@pytest.mark.parametrize("seed", range(6))
def test_random_noise_fields_closed_manifold(seed):
    rng = np.random.default_rng(seed)
    shape = (11 + seed, 9, 10)
    for field, iso in (
        (rng.random(shape), 0.5),  # white noise: ambiguous everywhere
        ((rng.random(shape) > 0.5).astype(float), 0.5),  # binary noise
        (np.round(rng.random(shape) * 4) / 4, 0.5),  # many samples exactly at iso
    ):
        verts, quads, info = mesher._surface_nets(field, iso, (0.0, 0.0, 0.0), 1.0)
        assert_closed_manifold(verts, quads)
        assert info["residual_conflicts"] == 0
        assert info["ambiguous_faces"] > 0
        # Euler characteristic = 2 * (components - handles): consistent with a closed surface
        chi = euler_characteristic(verts, quads)
        assert chi % 2 == 0


def test_smooth_noise_closed_manifold():
    rng = np.random.default_rng(42)
    f = rng.standard_normal((30, 28, 26))
    for _ in range(3):  # cheap low-pass
        f = (f + np.roll(f, 1, 0) + np.roll(f, 1, 1) + np.roll(f, 1, 2)) / 4.0
    verts, quads, info = mesher._surface_nets(f, 0.0, (0.0, 0.0, 0.0), 0.1)
    assert_closed_manifold(verts, quads)
    assert info["residual_conflicts"] == 0


def test_surface_nets_speed():
    field, origin, h = centred_field(96, sphere_sdf(0.8))
    t0 = time.perf_counter()
    _verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    assert time.perf_counter() - t0 < 5.0
    assert quads.shape[0] > 10000


# ---------------------------------------------------------------------------- mesh utilities
def test_triangulate_and_edges():
    quads = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    tris = mesher.triangulate(quads)
    assert tris.shape == (4, 3) and tris.dtype == np.int64
    assert np.array_equal(tris, [[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    edges = mesher.mesh_edges(quads)
    assert edges.shape == (7, 2) and np.all(edges[:, 0] < edges[:, 1])
    assert len({tuple(e) for e in edges}) == 7
    assert mesher.mesh_edges(np.zeros((0, 4), dtype=np.int64)).shape == (0, 2)
    field, origin, h = centred_field(16, sphere_sdf(0.6))
    verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    tris = mesher.triangulate(quads)
    assert signed_volume(verts, tris) == pytest.approx(signed_volume(verts, quads))
    assert edge_face_counts(tris)[1].max() == 1


def test_vertex_normals_triangles_and_isolated_vertex():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float)
    normals = mesher.vertex_normals(verts, np.array([[0, 1, 2]]))
    assert np.allclose(normals[:3], [0, 0, 1])
    assert np.allclose(normals[3], [0, 0, 1])  # no faces: default
    # area weighting: a big face dominates a small one
    verts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 4]])
    n0 = mesher.vertex_normals(verts, faces)[0]
    assert n0[2] > 0.99


def test_taubin_preserves_volume_and_smooths():
    field, origin, h = centred_field(32, sphere_sdf(0.7))
    verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    rng = np.random.default_rng(0)
    noisy = verts + rng.normal(scale=0.2 * h, size=verts.shape)
    smooth = mesher.taubin_smooth(noisy, quads, iterations=10)
    assert smooth.shape == verts.shape and not np.shares_memory(smooth, noisy)
    v0 = signed_volume(noisy, quads)
    assert signed_volume(smooth, quads) == pytest.approx(v0, rel=0.03)
    rough = lambda v: np.std(np.linalg.norm(v, axis=1))
    assert rough(smooth) < 0.5 * rough(noisy)
    # plain Laplacian shrinks, Taubin does not (much)
    lap = noisy.copy()
    for _ in range(10):
        lap = lap + 0.5 * mesher._UmbrellaOperator(quads, lap.shape[0]).laplacian(lap)
    assert signed_volume(lap, quads) < signed_volume(smooth, quads)
    # pinned vertices stay put (mask and index forms)
    pinned = np.zeros(verts.shape[0], dtype=bool)
    pinned[::7] = True
    out = mesher.taubin_smooth(noisy, quads, iterations=5, pinned=pinned)
    assert np.array_equal(out[pinned], noisy[pinned])
    out2 = mesher.taubin_smooth(noisy, quads, iterations=5, pinned=np.flatnonzero(pinned))
    assert np.array_equal(out, out2)
    assert np.array_equal(mesher.taubin_smooth(noisy, quads, iterations=0), noisy)
    empty = mesher.taubin_smooth(np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64))
    assert empty.shape == (0, 3)


def test_laplacian_smooth_scalar():
    field, origin, h = centred_field(16, sphere_sdf(0.6))
    verts, quads = mesher.surface_nets(field, 0.0, origin, h)
    n = verts.shape[0]
    tris = mesher.triangulate(quads)
    const = mesher.laplacian_smooth_scalar(np.full(n, 3.0), tris, n, 5)
    assert np.allclose(const, 3.0)
    rng = np.random.default_rng(1)
    values = rng.random(n)
    smooth = mesher.laplacian_smooth_scalar(values, tris, n, 5)
    assert smooth.std() < 0.5 * values.std()
    assert values.min() - 1e-12 <= smooth.min() and smooth.max() <= values.max() + 1e-12
    two = mesher.laplacian_smooth_scalar(np.stack([values, 2 * values], 1), quads, n, 5)
    assert np.allclose(two[:, 1], 2 * two[:, 0])
    # an isolated extra vertex keeps its value
    out = mesher.laplacian_smooth_scalar(np.append(values, 7.0), tris, n + 1, 3)
    assert out[-1] == 7.0
    with pytest.raises(ValueError):
        mesher.laplacian_smooth_scalar(values[:-1], tris, n, 1)


def test_remove_small_components():
    big, o1, h1 = centred_field(20, sphere_sdf(0.7))
    vb, qb = mesher.surface_nets(big, 0.0, o1, h1)
    small, o2, h2 = centred_field(6, sphere_sdf(0.6), lo=2.0, hi=2.5)
    vs, qs = mesher.surface_nets(small, 0.0, o2, h2)
    verts = np.concatenate([vs, vb, [[9.0, 9.0, 9.0]]])  # + one unused vertex
    quads = np.concatenate([qs, qb + vs.shape[0]])
    out_v, out_q = mesher.remove_small_components(verts, quads, min_faces=qs.shape[0] + 1)
    assert out_q.shape == qb.shape
    assert out_v.shape == vb.shape and np.allclose(np.sort(out_v, 0), np.sort(vb, 0))
    assert_closed_manifold(out_v, out_q)
    kept_v, kept_q = mesher.remove_small_components(verts, quads, min_faces=1)
    assert kept_q.shape[0] == quads.shape[0] and kept_v.shape[0] == verts.shape[0] - 1
    # everything below the threshold: the largest component survives
    _lone_v, lone_q = mesher.remove_small_components(verts, quads, min_faces=10**9)
    assert lone_q.shape == qb.shape
    ev, eq = mesher.remove_small_components(np.zeros((3, 3)), np.zeros((0, 4), dtype=np.int64), 5)
    assert ev.shape == (0, 3) and eq.shape == (0, 4)


def test_remove_small_components_relative_threshold_index_and_validation():
    parts = []
    for k, n in enumerate((20, 8, 12)):  # three separate spheres
        f, o, h = centred_field(n, sphere_sdf(0.6))
        v, q = mesher.surface_nets(f, 0.0, o + 3.0 * k, h)
        parts.append((v, q))
    sizes = [q.shape[0] for _, q in parts]
    offsets = np.cumsum([0] + [v.shape[0] for v, _ in parts])
    verts = np.concatenate([v for v, _ in parts])
    quads = np.concatenate([q + off for (_, q), off in zip(parts, offsets)])
    # keep components with at least 40 % of the largest one's faces
    frac = (sizes[2] / sizes[0] + sizes[1] / sizes[0]) / 2.0
    v2, q2, kept = mesher.remove_small_components(verts, quads, 1, min_fraction=frac, return_index=True)
    assert q2.shape[0] == sizes[0] + sizes[2] and q2.dtype == np.int64
    assert np.array_equal(kept, np.r_[offsets[0]:offsets[1], offsets[2]:offsets[3]])
    assert np.array_equal(v2, verts[kept])
    assert_closed_manifold(v2, q2)
    # the absolute threshold still applies; triangles work too
    tris = mesher.triangulate(quads)
    _v3, t3 = mesher.remove_small_components(verts, tris, 2 * sizes[2] + 1)
    assert t3.shape == (2 * sizes[0], 3)
    # ties: the component with the lowest vertex index wins
    v4, _q4 = mesher.remove_small_components(
        np.concatenate([parts[0][0], parts[0][0] + 5.0]),
        np.concatenate([parts[0][1], parts[0][1] + parts[0][0].shape[0]]),
        10**9,
    )
    assert np.allclose(v4, parts[0][0])
    with pytest.raises(ValueError):
        mesher.remove_small_components(verts, np.array([[0, 1, 2, verts.shape[0]]]), 1)
    with pytest.raises(ValueError):
        mesher.remove_small_components(verts, np.array([[0, 1, -1]]), 1)
    ev, eq, ei = mesher.remove_small_components(verts, np.zeros((0, 3), dtype=np.int64), 1, return_index=True)
    assert ev.shape == (0, 3) and eq.shape == (0, 3) and ei.shape == (0,)


def test_face_components():
    quads = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [6, 7, 8, 9], [9, 10, 11, 12]])
    labels = mesher.face_components(14, quads)  # vertex 13 unused
    assert labels.dtype == np.int64 and np.array_equal(labels, [0, 0, 6, 6])
    tris = np.array([[5, 6, 7], [0, 1, 2], [7, 8, 9]])  # touching at vertex 7 only
    assert np.array_equal(mesher.face_components(10, tris), [5, 0, 5])
    assert mesher.face_components(3, np.zeros((0, 3), dtype=np.int64)).shape == (0,)
    with pytest.raises(ValueError):
        mesher.face_components(3, np.array([[0, 1, 3]]))


def test_component_labels_long_chain():
    n = 5000
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    rng = np.random.default_rng(3)
    perm = rng.permutation(n)
    labels = mesher._component_labels(n, perm[edges])
    assert np.unique(labels).size == 1
    edges2 = np.array([[0, 1], [2, 3], [3, 4]])
    labels2 = mesher._component_labels(6, edges2)
    assert np.array_equal(labels2, [0, 0, 2, 2, 2, 5])


# ---------------------------------------------------------------------------- sampling
def linear_grid_field(channels=None):
    g = Grid((5, 4, 6), (0.3, -0.7, 1.1), 0.2)
    A = np.array([1.5, -2.0, 0.7])
    centres = g.element_centers()
    lin = centres @ A + 0.25
    if channels:
        lin = np.stack([lin * (c + 1) - c for c in range(channels)], axis=-1)
    return g, A, lin


def test_sample_trilinear_exact_for_linear_fields():
    g, A, lin = linear_grid_field()
    rng = np.random.default_rng(0)
    lo = g.origin + 0.5 * g.h
    hi = g.origin + (np.asarray(g.shape) - 0.5) * g.h
    pts = lo + rng.random((500, 3)) * (hi - lo)
    out = mesher.sample_trilinear(lin, g.origin, g.h, pts)
    assert out.shape == (500,)
    assert np.allclose(out, pts @ A + 0.25, atol=1e-12)
    # at the centres themselves
    centres = g.element_centers().reshape(-1, 3)
    assert np.allclose(mesher.sample_trilinear(lin, g.origin, g.h, centres), lin.ravel())
    # channels
    g, A, lin3 = linear_grid_field(channels=3)
    out3 = mesher.sample_trilinear(lin3, g.origin, g.h, pts)
    assert out3.shape == (500, 3)
    exact = pts @ A + 0.25
    assert np.allclose(out3, np.stack([exact * (c + 1) - c for c in range(3)], axis=-1))


def test_sample_trilinear_clamps_outside_and_handles_bad_points():
    g, A, lin = linear_grid_field()
    far = np.array([[-100.0, 0.0, 1.5], [100.0, 100.0, 100.0], [np.nan, 0.0, 0.0]])
    out = mesher.sample_trilinear(lin, g.origin, g.h, far)
    lo = g.origin + 0.5 * g.h
    hi = g.origin + (np.asarray(g.shape) - 0.5) * g.h
    clamped = np.clip(far[:2], lo, hi)
    assert np.allclose(out[:2], clamped @ A + 0.25)
    assert out[2] == 0.0
    # 1-voxel-thick grids
    flat = np.arange(6.0).reshape(1, 2, 3)
    p = np.array([[0.5, 1.0, 1.5]])
    assert mesher.sample_trilinear(flat, (0, 0, 0), 1.0, p)[0] == pytest.approx(3.0 * 0.5 + 1.0)
    assert mesher.sample_trilinear(lin, g.origin, g.h, np.zeros((0, 3))).shape == (0,)


def test_sample_trilinear_mask_renormalizes():
    g, A, lin = linear_grid_field()
    mask = np.zeros(g.shape, dtype=bool)
    mask[:3] = True  # first three x-layers valid
    garbage = np.where(mask, lin, 1e6)
    rng = np.random.default_rng(5)
    lo = g.origin + 0.5 * g.h
    hi = g.origin + (np.asarray(g.shape) - 0.5) * g.h
    pts = lo + rng.random((400, 3)) * (hi - lo)
    out = mesher.sample_trilinear(garbage, g.origin, g.h, pts, mask=mask)
    assert np.all(np.abs(out) < 1e3)  # invalid cells never leak in
    u = (pts[:, 0] - g.origin[0]) / g.h - 0.5
    inside = u <= 2.0  # all eight neighbours valid: exact
    assert np.allclose(out[inside], pts[inside] @ A + 0.25)
    # beyond the mask: renormalized weights of the valid layer (x index 2)
    near = (u > 2.0) & (u < 3.0)
    x2 = g.origin[0] + 2.5 * g.h
    assert np.allclose(out[near], np.c_[np.full(near.sum(), x2), pts[near, 1:]] @ A + 0.25)
    # points with no valid neighbour: extrapolated if close, else 0
    single = np.zeros(g.shape, dtype=bool)
    single[0, 0, 0] = True
    far_pts = np.array([g.origin + (np.array([2, 0, 0]) + 0.5) * g.h,
                        g.origin + (np.array([4, 3, 5]) + 0.5) * g.h])
    val = mesher.sample_trilinear(np.where(single, 7.0, -5.0), g.origin, g.h, far_pts, mask=single)
    assert val[0] == pytest.approx(7.0) and val[1] == 0.0
    nothing = mesher.sample_trilinear(lin, g.origin, g.h, pts[:3], mask=np.zeros(g.shape, bool))
    assert np.all(nothing == 0.0)


def test_extrapolate_reach_values_and_channels():
    shape = (7, 6, 5)
    mask = np.zeros(shape, dtype=bool)
    mask[0, 2, 2] = True
    field = np.where(mask, 5.0, -1.0)
    out = mesher.extrapolate(field, mask, iterations=2)
    assert out is not field and field[1, 2, 2] == -1.0
    idx = np.indices(shape)
    l1 = np.abs(idx[0]) + np.abs(idx[1] - 2) + np.abs(idx[2] - 2)
    assert np.all(out[l1 <= 2] == 5.0) and np.all(out[l1 > 2] == -1.0)
    assert out[6, 2, 2] == -1.0  # no wrap-around to the far side
    # mean of valid neighbours, channels
    mask = np.zeros(shape, dtype=bool)
    mask[3, 0, 0] = mask[3, 2, 0] = True
    field = np.zeros(shape + (2,))
    field[3, 0, 0] = (1.0, 10.0)
    field[3, 2, 0] = (3.0, 30.0)
    out = mesher.extrapolate(field, mask, iterations=1)
    assert np.allclose(out[3, 1, 0], (2.0, 20.0))
    assert np.allclose(out[3, 0, 0], (1.0, 10.0))
    _values, valid = mesher._extrapolate(field, mask, 1)
    # (3,0,0) has 4 in-grid neighbours, (3,2,0) has 5; (3,1,0) is shared
    assert valid.sum() == 2 + 4 + 5 - 1
    assert np.array_equal(mesher.extrapolate(field, np.zeros(shape, bool), 3), field)


# ---------------------------------------------------------------------------- colormaps
@pytest.mark.parametrize("name", mesher.COLORMAPS)
def test_colormap_ranges(name):
    values = np.linspace(-1.0, 3.0, 41)
    colors = mesher.colormap(values, 0.0, 2.0, name=name)
    assert colors.shape == (41, 4) and colors.dtype == np.float64
    assert np.all((colors >= 0.0) & (colors <= 1.0)) and np.all(colors[:, 3] == 1.0)
    below = mesher.colormap([-5.0, 0.0], 0.0, 2.0, name=name)
    above = mesher.colormap([2.0, 9.0], 0.0, 2.0, name=name)
    assert np.allclose(below[0], below[1]) and np.allclose(above[0], above[1])
    assert not np.allclose(below[0], above[0])
    assert np.allclose(mesher.colormap([np.nan], 0.0, 2.0, name=name), below[:1])
    same = mesher.colormap([1.0, 2.0], 1.0, 1.0, name=name)
    assert np.all(np.isfinite(same))
    assert mesher.colormap(np.zeros((2, 3)), 0, 1, name=name.lower()).shape == (2, 3, 4)


def test_colormap_named_maps_are_linear_rgb():
    heat = mesher.colormap([0.0, 1.0], 0.0, 1.0, "HEAT")
    assert np.allclose(heat[0, :3], 0.0) and np.allclose(heat[1, :3], 1.0)
    lum = mesher.colormap(np.linspace(0, 1, 50), 0, 1, "HEAT")[:, :3] @ [0.2126, 0.7152, 0.0722]
    assert np.all(np.diff(lum) >= -1e-12)  # heat gets brighter
    vir = mesher.colormap([0.0, 1.0], 0.0, 1.0, "VIRIDIS")
    assert np.allclose(vir[0, :3], mesher.srgb_to_linear([0.267, 0.005, 0.329]))
    assert np.allclose(vir[1, :3], mesher.srgb_to_linear([0.993, 0.906, 0.144]))
    # sRGB -> linear: mid grey 0.5 is ~0.214 linear
    assert mesher.srgb_to_linear(0.5) == pytest.approx(0.21404, abs=1e-4)
    # diverging: 0 is the white centre even for an asymmetric range
    cw = mesher.colormap([-1.0, 0.0, 3.0], -1.0, 3.0, "COOLWARM")
    centre = mesher.colormap([0.5], 0.0, 1.0, "COOLWARM")[0]
    assert np.allclose(cw[1], centre)
    assert cw[0, 2] > cw[0, 0] and cw[2, 0] > cw[2, 2]  # blue low end, red high end
    with pytest.raises(ValueError):
        mesher.colormap([0.0], 0.0, 1.0, "RAINBOW")
