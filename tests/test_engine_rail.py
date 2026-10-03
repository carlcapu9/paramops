import math

import numpy as np
import pytest

from engine import spline as sp
from engine.rail import Rail
from engine.expr import ExpressionError, compile_expression


def poly(points, cyclic=False):
    return sp.sample(sp.SplineSpec("POLY", points, cyclic=cyclic))


def test_poly_sampling_and_vertex_angles():
    s = poly([(0, 0, 0), (4, 0, 0), (4, 4, 0)])
    assert s.points.shape == (3, 3)
    assert list(s.param) == [0.0, 1.0, 2.0]
    assert s.vertex_angle[1] == pytest.approx(math.pi / 2)


def test_bezier_straight_and_curved_steps():
    a, d = np.array((0.0, 0, 0)), np.array((3.0, 0, 0))
    assert sp.bezier_steps(a, np.array((1.0, 0, 0)), np.array((2.0, 0, 0)), d) == 1
    assert sp.bezier_steps(a, np.array((1.0, 2, 0)), np.array((2.0, 2, 0)), d) >= 3


def test_bezier_smooth_vertex_has_no_corner():
    co = [(0, 0, 0), (2, 2, 0), (4, 0, 0)]
    left = [(-1, -1, 0), (1, 2, 0), (3, 1, 0)]
    right = [(1, 1, 0), (3, 2, 0), (5, -1, 0)]
    s = sp.sample(sp.SplineSpec("BEZIER", co, left=left, right=right))
    assert s.vertex_angle[1] == pytest.approx(0.0, abs=1e-9)


def test_rail_offsets_extrapolation_and_miter():
    straight = Rail(poly([(0, 0, 0), (10, 0, 0)]))
    assert np.allclose(straight.deform([2.0], [1.0], [0.5]), [[2, 1, 0.5]])
    rail = Rail(poly([(0, 0, 0), (4, 0, 0), (4, 4, 0)]))
    assert rail.length == pytest.approx(8.0)
    assert np.allclose(rail.deform([-1.0], [0.0], [0.0]), [[-1, 0, 0]])
    assert np.allclose(rail.deform([4.0], [1.0], [0.0]), [[3, 1, 0]])
    assert np.allclose(rail.deform([4.0], [-1.0], [0.0]), [[5, -1, 0]])


def test_joint_zone_limits_miter_blend():
    rail = Rail(poly([(0, 0, 0), (4, 0, 0), (4, 4, 0)]), joint_zone=0.1)
    assert np.allclose(rail.deform([4.0], [0.1], [0.0]), [[3.9, 0.1, 0.0]])
    assert np.allclose(rail.deform([6.0], [0.1], [0.0]), [[3.9, 2.0, 0.0]])
    assert rail.length == pytest.approx(8.0)


def test_upright_mode_shears():
    rail = Rail(poly([(0, 0, 0), (4, 0, 3)]))
    assert np.allclose(rail.deform([2.5], [0.0], [1.0], upright=True), [[2.0, 0.0, 2.5]])
    assert np.allclose(rail.deform([2.5], [0.0], [1.0]), [[1.4, 0.0, 2.3]])
    assert math.degrees(rail.slope([1.0])[0]) == pytest.approx(36.8699, abs=1e-3)


def test_frames_at_vertex_follow_orientation():
    rail = Rail(poly([(0, 0, 0), (4, 0, 0), (4, 4, 0)]))
    _, f, _, _ = rail.frames([4.0])
    assert np.allclose(f, [[math.sqrt(0.5), math.sqrt(0.5), 0]])
    assert np.allclose(rail.frames([4.0], orient="INCOMING")[1], [[1, 0, 0]])
    assert np.allclose(rail.frames([4.0], orient="OUTGOING")[1], [[0, 1, 0]])


def test_cyclic_and_reverse_parameters():
    s = poly([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], cyclic=True)
    rail = Rail(s)
    assert rail.length == pytest.approx(8.0)
    assert np.allclose(rail.position([9.0]), [[1, 0, 0]])
    assert rail.s_at(4.0) == pytest.approx(0.0)
    r = Rail(sp.reverse(poly([(0, 0, 0), (4, 0, 0), (4, 4, 0)])))
    assert r.s_at(2.0) == pytest.approx(0.0)
    assert np.allclose(r.position(r.s_at(1.0)), [[4, 0, 0]])


def test_nurbs_endpoint_ends_on_control_points():
    co = [(0, 0, 0), (1, 2, 0), (3, 2, 0), (4, 0, 0)]
    s = sp.sample(sp.SplineSpec("NURBS", co, order=4, endpoint=True), resolution=8)
    assert np.allclose(s.points[0], co[0]) and np.allclose(s.points[-1], co[-1])
    assert list(sp.knot_vector(4, 4, False, True, False)) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_minimum_twist_frames_orthonormal():
    pts = [(math.cos(a), math.sin(a), 0.3 * a) for a in np.linspace(0, 6, 40)]
    rail = Rail(poly(pts), twist="MINIMUM")
    _, f, l, u = rail.frames(np.linspace(0, rail.length, 20))
    assert np.allclose(np.einsum("ij,ij->i", f, l), 0, atol=1e-9)
    assert np.allclose(np.linalg.norm(u, axis=1), 1)


def test_expressions():
    f = compile_expression("index % 3 == 0 and distance > 2", ["index", "distance"])
    assert f({"index": 3, "distance": 5}) == 1.0
    assert f({"index": 4, "distance": 5}) == 0.0
    g = compile_expression("clamp(a * 2, 0, 1) + sin(pi / 2)", ["a"])
    assert g({"a": 0.25}) == pytest.approx(1.5)
    assert compile_expression("1 / 0", [])({}) == 0.0
    for bad in ("__import__('os')", "a.b", "[1, 2]", "open('x')", "unknown + 1"):
        with pytest.raises(ExpressionError):
            compile_expression(bad, ["a"])
