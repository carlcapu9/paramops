import math

import numpy as np
import pytest

from core import pathmath as pm


def poly(points, cyclic=False):
    sp = pm.SplineInput("POLY", points, cyclic=cyclic)
    return pm.sample_spline(sp)


def L_shape():
    return poly([(0, 0, 0), (4, 0, 0), (4, 4, 0)])


def test_poly_sampling_and_corner_angle():
    pl = L_shape()
    assert pl.pts.shape == (3, 3)
    assert list(pl.u) == [0.0, 1.0, 2.0]
    assert pl.corner_angle[1] == pytest.approx(math.pi / 2)
    assert pl.corner_angle[0] == 0.0 and pl.corner_angle[2] == 0.0


def test_straight_bezier_uses_single_step():
    p0, p3 = np.array((0.0, 0, 0)), np.array((3.0, 0, 0))
    assert pm.bezier_segment_steps(p0, np.array((1.0, 0, 0)), np.array((2.0, 0, 0)), p3) == 1
    assert pm.bezier_segment_steps(p0, np.array((1.0, 1, 0)), np.array((2.0, 1, 0)), p3) > 2


def test_bezier_corner_detection_uses_handles():
    # Smooth (aligned) handles at the middle point -> no corner even with coarse sampling.
    co = [(0, 0, 0), (2, 2, 0), (4, 0, 0)]
    hl = [(-1, -1, 0), (1, 2, 0), (3, 1, 0)]
    hr = [(1, 1, 0), (3, 2, 0), (5, -1, 0)]
    sp = pm.SplineInput("BEZIER", co, handle_left=hl, handle_right=hr)
    pl = pm.sample_spline(sp)
    assert pl.corner_angle[1] == pytest.approx(0.0, abs=1e-9)
    # Vector-like handles -> 90 degree corner.
    hl2 = [(0, 0, 0), (1, 1, 0), (4, 0, 0)]
    hr2 = [(0, 0, 0), (3, 1, 0), (4, 0, 0)]
    pl2 = pm.sample_spline(pm.SplineInput("BEZIER", co, handle_left=hl2, handle_right=hr2))
    assert pl2.corner_angle[1] == pytest.approx(math.pi / 2)


def test_path_straight_offsets():
    path = pm.Path(poly([(0, 0, 0), (10, 0, 0)]))
    assert path.L == pytest.approx(10)
    out = path.evaluate([2.5], [1.0], [0.5])
    assert np.allclose(out, [[2.5, 1.0, 0.5]])
    # Extrapolation past the ends keeps going straight.
    out = path.evaluate([-1.0, 11.0], [0.0, 0.0], [0.0, 0.0])
    assert np.allclose(out, [[-1, 0, 0], [11, 0, 0]])


def test_miter_join_is_watertight():
    path = pm.Path(L_shape())
    corner_in = path.evaluate([4.0], [1.0], [0.0])
    assert np.allclose(corner_in, [[3.0, 1.0, 0.0]])
    corner_out = path.evaluate([4.0], [-1.0], [0.0])
    assert np.allclose(corner_out, [[5.0, -1.0, 0.0]])


def test_miter_zone_keeps_far_geometry_unsheared():
    path = pm.Path(L_shape(), miter_extent=0.1)
    # Inside the zone the miter is exact...
    assert np.allclose(path.evaluate([4.0], [0.1], [0.0]), [[3.9, 0.1, 0.0]])
    # ...and away from it the offset is a plain perpendicular offset.
    assert np.allclose(path.evaluate([6.0], [0.1], [0.0]), [[3.9, 2.0, 0.0]])
    assert np.allclose(path.evaluate([2.0], [0.1], [0.0]), [[2.0, 0.1, 0.0]])


def test_vertical_mode_keeps_z_up_on_slopes():
    path = pm.Path(poly([(0, 0, 0), (4, 0, 3)]))  # slope 3/4, length 5
    a = path.evaluate([2.5], [0.0], [1.0], vertical=True)
    b = path.evaluate([2.5], [0.0], [1.0], vertical=False)
    assert np.allclose(a, [[2.0, 0.0, 2.5]])         # straight up from the path point
    assert np.allclose(b, [[2.0 - 0.6, 0.0, 1.5 + 0.8]])  # along the slope normal


def test_frames_bisector_at_corner():
    path = pm.Path(L_shape())
    P, T, L, N = path.frames([4.0])
    assert np.allclose(P, [[4, 0, 0]])
    assert np.allclose(T, [[math.sqrt(0.5), math.sqrt(0.5), 0]])
    _, Ti, _, _ = path.frames([4.0], orient="INCOMING")
    assert np.allclose(Ti, [[1, 0, 0]])
    _, To, _, _ = path.frames([4.0], orient="OUTGOING")
    assert np.allclose(To, [[0, 1, 0]])


def test_fillet_open_corner():
    pl = L_shape()
    out, rounded = pm.fillet_polyline(pl, [1], 1.0)
    assert rounded == {1}
    path = pm.Path(out)
    assert path.L == pytest.approx(3 + math.pi / 2 + 3, rel=1e-3)
    # The control point now sits on the arc (its middle).
    s1 = path.s_of_u(1.0)
    assert s1 == pytest.approx(3 + math.pi / 4, rel=1e-2)
    p = path.point(s1)[0]
    centre = np.array((3.0, 1.0, 0.0))
    assert np.linalg.norm(p - centre) == pytest.approx(1.0, rel=1e-3)
    assert out.corner_angle[1] == 0.0


def test_fillet_clamps_to_available_length():
    pl = poly([(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)])
    out, rounded = pm.fillet_polyline(pl, [1, 2], 10.0)
    assert rounded == {1, 2}
    u = out.u
    assert np.all(np.diff(u) > 0)


def test_fillet_cyclic_square_including_seam():
    sq = poly([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], cyclic=True)
    assert np.allclose(sq.pts[0], sq.pts[-1])
    out, rounded = pm.fillet_polyline(sq, [0, 1, 2, 3], 0.5)
    assert rounded == {0, 1, 2, 3}
    path = pm.Path(out)
    expected = 4 * (2 - 1.0) + 2 * math.pi * 0.5
    assert path.L == pytest.approx(expected, rel=1e-3)
    assert out.u[0] == pytest.approx(0.0)
    assert out.u[-1] == pytest.approx(4.0)
    assert np.all(np.diff(out.u) > 0)
    # Control point 0 sits on the arc around (0.5, 0.5).
    p0 = path.point(path.s_of_u(0.0))[0]
    assert np.linalg.norm(p0 - np.array((0.5, 0.5, 0))) == pytest.approx(0.5, rel=1e-2)


def test_reverse_keeps_control_mapping():
    pl = pm.reverse_polyline(L_shape())
    path = pm.Path(pl)
    assert path.s_of_u(2.0) == pytest.approx(0.0)
    assert path.s_of_u(0.0) == pytest.approx(8.0)
    assert np.allclose(path.point(path.s_of_u(1.0)), [[4, 0, 0]])


def test_nurbs_knots_endpoint_bezier_like():
    k = pm.nurbs_knots(4, 4, cyclic=False, endpoint=True, bezier=False)
    assert list(k) == [0, 0, 0, 0, 1, 1, 1, 1]
    k = pm.nurbs_knots(5, 3, cyclic=False, endpoint=False, bezier=False)
    assert list(k) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_nurbs_endpoint_interpolates_ends():
    co = [(0, 0, 0), (1, 2, 0), (3, 2, 0), (4, 0, 0)]
    sp = pm.SplineInput("NURBS", co, order=4, use_endpoint=True)
    pl = pm.sample_spline(sp, resolution=8)
    assert np.allclose(pl.pts[0], co[0])
    assert np.allclose(pl.pts[-1], co[-1])
    assert pl.u[0] == 0.0 and pl.u[-1] == 3.0


def test_cyclic_poly_wraps_arclength():
    sq = poly([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)], cyclic=True)
    path = pm.Path(sq)
    assert path.L == pytest.approx(8.0)
    assert np.allclose(path.point([9.0]), [[1, 0, 0]])
    assert np.allclose(path.point([3.0]), [[2, 1, 0]])
    assert path.s_of_u(4.0) == pytest.approx(0.0)


def test_minimum_twist_frames_are_orthonormal():
    pts = [(math.cos(a), math.sin(a), 0.3 * a) for a in np.linspace(0, 6, 40)]
    path = pm.Path(poly(pts), twist="MINIMUM")
    P, T, L, N = path.frames(np.linspace(0, path.L, 25))
    assert np.allclose(np.einsum("ij,ij->i", T, L), 0, atol=1e-9)
    assert np.allclose(np.linalg.norm(L, axis=1), 1)
    assert np.allclose(np.linalg.norm(N, axis=1), 1)


def test_densify():
    pl = pm.densify_polyline(L_shape(), 0.5)
    assert len(pl.pts) == 17
    assert np.all(np.diff(pl.u) > 0)
