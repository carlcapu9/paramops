import numpy as np
import pytest

from stress_sculpt.core import forms
from stress_sculpt.core.grid import Cancelled


def _grid_loads(nx=3, ny=3, size=2.0, z=3.0, force=1000.0):
    xs = np.linspace(-size / 2, size / 2, nx)
    ys = np.linspace(-size / 2, size / 2, ny)
    pts = np.array([[x, y, z] for x in xs for y in ys], dtype=np.float64)
    return pts, np.tile([0.0, 0.0, -force], (pts.shape[0], 1))


def test_single_tree_is_in_equilibrium_and_carries_the_weight():
    pts, F = _grid_loads()
    tree = forms.grow_tree(pts, F, ground=0.0)
    assert tree.equilibrium_residual() < 1e-9
    reactions = tree.base_reactions()
    assert reactions.shape == (1, 3)
    np.testing.assert_allclose(reactions[0], [0.0, 0.0, -9000.0], atol=1e-6)
    # every load point is a leaf of the tree
    leaves = tree.nodes[tree.kind == 2]
    for p in pts:
        assert np.min(np.linalg.norm(leaves - p, axis=1)) < 1e-9
    # the base stands on the ground
    assert np.allclose(tree.nodes[tree.kind == 0][:, 2], 0.0)


def test_members_are_sized_by_force():
    pts, F = _grid_loads()
    st = forms.TreeSettings(radius_min=0.01, radius_max=0.2)
    tree = forms.grow_tree(pts, F, st, ground=0.0)
    trunk = np.argmax(tree.forces)
    assert tree.radii[trunk] == pytest.approx(0.2)
    assert tree.forces[trunk] == pytest.approx(9000.0)
    # radius ~ sqrt(force) where not clamped
    free = tree.radii > 0.0100001
    ratio = tree.radii[free] / np.sqrt(tree.forces[free])
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_inner_nodes_rise_monotonically():
    pts, F = _grid_loads(4, 4)
    tree = forms.grow_tree(pts, F, forms.TreeSettings(branching=2), ground=0.0)
    for parent, child in tree.edges:
        assert tree.nodes[child, 2] > tree.nodes[parent, 2]


def test_automatic_footings_are_symmetric_for_a_symmetric_grid():
    pts, F = _grid_loads(4, 4, size=4.0)
    feet = forms.auto_footings(pts, F, ground=0.0, n=4)
    assert feet.shape == (4, 3)
    np.testing.assert_allclose(np.sort(np.abs(feet[:, 0])), np.full(4, feet[0, 0].__abs__()), atol=1e-9)
    np.testing.assert_allclose(feet[:, 2], 0.0)
    tree = forms.grow_tree(pts, F, forms.TreeSettings(n_trunks=4), ground=0.0)
    assert (tree.kind == 0).sum() == 4
    assert tree.equilibrium_residual() < 1e-9
    np.testing.assert_allclose(tree.base_reactions().sum(0), [0, 0, -16000.0], atol=1e-6)
    # vertical trunks: no horizontal thrust at the automatic footings
    np.testing.assert_allclose(tree.base_reactions()[:, :2], 0.0, atol=1e-6)


def test_explicit_supports_and_sideways_loads():
    pts, F = _grid_loads()
    supports = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tree = forms.grow_tree(pts, F, supports=supports)
    bases = tree.nodes[tree.kind == 0]
    assert bases.shape[0] == 2
    np.testing.assert_allclose(np.sort(bases[:, 0]), [-2.0, 2.0])
    assert tree.equilibrium_residual() < 1e-9
    np.testing.assert_allclose(tree.base_reactions().sum(0)[2], -9000.0, rtol=1e-9)
    # loads pushing along +X grow the tree along -X
    side = forms.grow_tree(pts[:, [2, 1, 0]], np.tile([1000.0, 0, 0], (9, 1)), ground=None)
    np.testing.assert_allclose(side.up, [-1.0, 0.0, 0.0], atol=1e-12)
    assert side.equilibrium_residual() < 1e-9


def test_loads_below_explicit_support_are_rejected():
    pts, F = _grid_loads(z=0.5)
    with pytest.raises(ValueError):
        forms.grow_tree(pts, F, supports=np.array([[0.0, 0.0, 1.0]]))


def test_many_load_points_are_clustered():
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-1, 1, (500, 2)), np.full(500, 3.0)]
    F = np.tile([0, 0, -10.0], (500, 1))
    tree = forms.grow_tree(pts, F, forms.TreeSettings(max_leaves=32), ground=0.0)
    assert (tree.kind == 2).sum() == 32
    assert tree.total_load == pytest.approx(5000.0)
    assert any("merged" in w for w in tree.warnings)


def test_determinism_and_tubes():
    pts, F = _grid_loads(4, 3)
    a = forms.grow_tree(pts, F, ground=0.0)
    b = forms.grow_tree(pts, F, ground=0.0)
    np.testing.assert_array_equal(a.nodes, b.nodes)
    tubes = a.tubes()
    assert len(tubes) == a.edges.shape[0]
    for tube, curve in zip(tubes, a.curves):
        assert tube.points.shape == curve.shape
        assert np.all(tube.radius > 0)


def test_constrain_keeps_inner_nodes_inside():
    pts, F = _grid_loads(3, 3, size=4.0)

    def clamp(p):  # allowed region: |x|, |y| <= 0.5
        q = p.copy()
        q[:, :2] = np.clip(q[:, :2], -0.5, 0.5)
        return q

    tree = forms.grow_tree(pts, F, supports=np.array([[0.0, 0.0, 0.0]]), constrain=clamp)
    inner = tree.nodes[tree.kind == 1]
    assert np.all(np.abs(inner[:, :2]) <= 0.5 + 1e-9)


def test_should_stop_cancels():
    pts, F = _grid_loads(4, 4)
    with pytest.raises(Cancelled):
        forms.grow_tree(pts, F, ground=0.0, should_stop=lambda: True)


def test_weighted_kmeans_is_deterministic_and_complete():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 2))
    w = rng.uniform(0.5, 2.0, 200)
    a = forms.weighted_kmeans(pts, w, 5)
    b = forms.weighted_kmeans(pts, w, 5)
    np.testing.assert_array_equal(a, b)
    assert set(np.unique(a)) == set(range(5))
    assert forms.weighted_kmeans(pts[:3], w[:3], 10).max() == 2
