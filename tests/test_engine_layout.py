import math

import numpy as np
import pytest

from engine import graph as g
from engine import spline as sp
from engine.generate import SplineJob, clip_intervals, run
from engine.linear import ANCHOR, SPAN, LinearSettings, layout, marker_positions
from engine.mesh import Assembler, MeshBuf, merge
from engine.place import PlaceOptions, place
from engine.rail import Rail


def box_buf(x0, x1, y0=-0.1, y1=0.1, z0=0.0, z1=1.0, nx=1):
    co, polys = [], []
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        co += [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    for i in range(nx):
        a, b = 4 * i, 4 * (i + 1)
        polys += [(a, b, b + 1, a + 1), (a + 1, b + 1, b + 2, a + 2), (a + 2, b + 2, b + 3, a + 3),
                  (a + 3, b + 3, b, a)]
    polys += [(0, 1, 2, 3), (4 * nx, 4 * nx + 3, 4 * nx + 2, 4 * nx + 1)]
    return MeshBuf.from_polygons(co, polys)


def seg(length=2.0, name="s", centered=False, **flags):
    x0 = -length / 2 if centered else 0.0
    geo = g.Geo(name, box_buf(x0, x0 + length))
    s = g.Seg([(geo, np.eye(4))], name=name)
    for k, v in flags.items():
        setattr(s, k, v)
    return s


def fixed(*a, **k):
    return g.Fixed(seg(*a, **k))


def rail_of(points, cyclic=False):
    return Rail(sp.sample(sp.SplineSpec("POLY", points, cyclic=cyclic)))


def lay(rail, sources, **kw):
    placed, trunc = layout(rail, sources, LinearSettings(**kw), {"spline": 0, "spline_length": rail.length})
    return placed


def of(placed, inp):
    return [p for p in placed if p.inp == inp]


STRAIGHT = [(0, 0, 0), (10.4, 0, 0)]


def test_adaptive_fit_modes():
    r = rail_of(STRAIGHT)
    d = {g.INPUT_DEFAULT: fixed(2.0)}
    p = lay(r, d)
    assert len(p) == 5 and p[0].s0 == pytest.approx(0) and p[-1].s1 == pytest.approx(10.4)
    assert p[0].k == pytest.approx(1.04)
    assert len(lay(r, d, fit="CEIL")) == 6
    assert len(lay(r, d, fit="FLOOR")) == 5
    assert len(lay(r, d, fit="COUNT", count=3)) == 3
    sp_ = lay(r, d, fit="COUNT", count=2, spacing=0.4)
    assert sp_[1].s0 - sp_[0].s1 == pytest.approx(0.4)


def test_real_size_alignment_and_remainder():
    r = rail_of([(0, 0, 0), (5.5, 0, 0)])
    d = {g.INPUT_DEFAULT: fixed(2.0)}
    p = lay(r, d, adaptive=False)
    assert len(p) == 3 and p[-1].cut == pytest.approx((4.0, 5.5))
    p = lay(r, d, adaptive=False, remainder="EMPTY", align="CENTER")
    assert len(p) == 2 and p[0].s0 == pytest.approx(0.75)
    p = lay(r, d, adaptive=False, remainder="SCALE", align="END")
    assert p[0].s0 == pytest.approx(0) and p[0].k == pytest.approx(0.75)
    p = lay(r, d, adaptive=False, align="SPREAD")
    assert p[0].s0 == pytest.approx(0) and p[-1].s1 == pytest.approx(5.5)


def test_start_end_and_clip():
    r = rail_of([(0, 0, 0), (10, 0, 0)])
    src = {g.INPUT_DEFAULT: fixed(1.0), g.INPUT_START: fixed(0.5, "st"), g.INPUT_END: fixed(0.25, "en")}
    p = lay(r, src, clip_start=1.0, clip_end=1.0)
    st = of(p, g.INPUT_START)[0]
    en = of(p, g.INPUT_END)[0]
    assert (st.s0, st.s1) == pytest.approx((1.0, 1.5))
    assert (en.s0, en.s1) == pytest.approx((8.75, 9.0))
    p = lay(r, src, clip_start=10.0, clip_end=10.0, clip_percent=True)
    assert of(p, g.INPUT_START)[0].s0 == pytest.approx(1.0)
    p = lay(r, src, clip_start=-1.0, extend=True)
    assert of(p, g.INPUT_START)[0].s0 == pytest.approx(-1.0)
    p = lay(r, src, clip_start=-1.0)
    assert of(p, g.INPUT_START)[0].s0 == pytest.approx(0.0)


def test_corners_anchor_and_split():
    r = rail_of([(0, 0, 0), (4, 0, 0), (4, 4, 0)])
    src = {g.INPUT_DEFAULT: fixed(1.0), g.INPUT_CORNER: fixed(0.2, "post", centered=True)}
    p = lay(r, src)
    c = of(p, g.INPUT_CORNER)
    assert len(c) == 1 and c[0].kind == ANCHOR and c[0].anchor == pytest.approx(4.0)
    assert (c[0].s0, c[0].s1) == pytest.approx((3.9, 4.1))
    d = of(p, g.INPUT_DEFAULT)
    assert all(q.s1 <= 3.9 + 1e-9 or q.s0 >= 4.1 - 1e-9 for q in d)
    p = lay(r, {g.INPUT_DEFAULT: fixed(3.0)})
    assert any(q.s1 == pytest.approx(4.0) for q in p)
    p = lay(r, {g.INPUT_DEFAULT: fixed(3.0)}, corner_mode="NONE")
    assert len(p) == 3
    p = lay(r, {g.INPUT_DEFAULT: fixed(3.0)}, corner_angle=math.radians(95))
    assert len(p) == 3


def test_evenly_modes():
    r = rail_of([(0, 0, 0), (10, 0, 0)])
    src = {g.INPUT_DEFAULT: fixed(1.0), g.INPUT_EVENLY: fixed(0.1, "ev", centered=True)}
    ev = [q.anchor for q in of(lay(r, src, evenly_distance=3.0), g.INPUT_EVENLY)]
    assert ev == pytest.approx([3, 6, 9])
    ev = [q.anchor for q in of(lay(r, src, evenly_distance=3.0, evenly_offset=1.5), g.INPUT_EVENLY)]
    assert ev == pytest.approx([1.5, 4.5, 7.5])
    ev = [q.anchor for q in of(lay(r, src, evenly_mode="COUNT", evenly_count=1), g.INPUT_EVENLY)]
    assert ev == pytest.approx([5.0])
    r2 = rail_of([(0, 0, 0), (4, 0, 0), (4, 6, 0)])
    per_section = [q.anchor for q in of(lay(r2, src, evenly_distance=3.0), g.INPUT_EVENLY)]
    assert per_section == pytest.approx([3.0, 7.0])
    whole = [q.anchor for q in of(lay(r2, src, evenly_distance=3.0, evenly_scope="SPLINE"), g.INPUT_EVENLY)]
    assert whole == pytest.approx([3.0, 6.0, 9.0])


def test_markers():
    r = rail_of([(0, 0, 0), (4, 0, 0), (8, 0, 0), (12, 0, 0)])
    st = LinearSettings(marker_mode="VERTICES", marker_list="1, 2")
    assert [s for s, _ in marker_positions(st, r, r.length)] == pytest.approx([4, 8])
    st = LinearSettings(marker_mode="DISTANCES", marker_list="25%, -2, 3")
    assert [s for s, _ in marker_positions(st, r, r.length)] == pytest.approx([3, 10])
    st = LinearSettings(marker_mode="REPEAT", marker_step=5, marker_offset=1)
    assert [s for s, _ in marker_positions(st, r, r.length)] == pytest.approx([1, 6, 11])
    src = {g.INPUT_DEFAULT: fixed(1.0), g.INPUT_MARKER: fixed(1.0, "door", centered=True)}
    p = lay(r, src, marker_mode="DISTANCES", marker_list="6", corner_mode="NONE")
    m = of(p, g.INPUT_MARKER)
    assert len(m) == 1 and (m[0].s0, m[0].s1) == pytest.approx((5.5, 6.5))
    assert all(q.s1 <= 5.5 + 1e-9 or q.s0 >= 6.5 - 1e-9 for q in of(p, g.INPUT_DEFAULT))


def test_sequence_randomize_selector_conditional():
    a, b, c = seg(1.0, "A"), seg(1.0, "B"), seg(1.0, "C")
    r = rail_of([(0, 0, 0), (6, 0, 0)])
    seq = g.Sequence([g.Fixed(a), g.Fixed(b), g.Fixed(c)], [1, 2, 1])
    names = [q.seg.name for q in lay(r, {g.INPUT_DEFAULT: seq})]
    assert names == ["A", "B", "B", "C", "A", "B"]
    rnd = g.Randomize([g.Fixed(a), g.Fixed(b)], [1.0, 0.0])
    assert {q.seg.name for q in lay(rail_of([(0, 0, 0), (20, 0, 0)]), {g.INPUT_DEFAULT: rnd})} == {"A"}
    rnd2 = g.Randomize([g.Fixed(a), g.Fixed(b)], [1.0, 1.0], seed=1)
    run1 = [q.seg.name for q in lay(rail_of([(0, 0, 0), (40, 0, 0)]), {g.INPUT_DEFAULT: rnd2})]
    run2 = [q.seg.name for q in lay(rail_of([(0, 0, 0), (40, 0, 0)]), {g.INPUT_DEFAULT: rnd2})]
    assert run1 == run2 and set(run1) == {"A", "B"}
    cond = g.Conditional(g.Fixed(a), g.Fixed(b), g.MathNum("EVEN", g.Var("index")))
    assert [q.seg.name for q in lay(r, {g.INPUT_DEFAULT: cond})] == ["A", "B"] * 3
    sel = g.Selector([g.Fixed(a), g.Fixed(b), g.Fixed(c)], g.Var("index"), "WRAP")
    assert [q.seg.name for q in lay(r, {g.INPUT_DEFAULT: sel})] == ["A", "B", "C"] * 2
    last = g.Conditional(g.Fixed(c), g.Fixed(a), g.MathNum("EQUAL", g.Var("from_end"), g.Const(0)))
    assert [q.seg.name for q in lay(r, {g.INPUT_DEFAULT: last})] == ["A"] * 5 + ["C"]


def test_variable_lengths_and_non_adaptive_segments():
    post = seg(0.2, "post", adaptive=False)
    panel = seg(1.8, "panel")
    seq = g.Sequence([g.Fixed(post), g.Fixed(panel)], [1, 1])
    p = lay(rail_of([(0, 0, 0), (10.2, 0, 0)]), {g.INPUT_DEFAULT: seq})
    posts = [q for q in p if q.seg.name == "post"]
    assert all(q.k == pytest.approx(1.0) for q in posts)
    assert p[-1].s1 == pytest.approx(10.2)


def test_compose_gap_and_transforms():
    a, b = seg(1.0, "A"), seg(0.5, "B")
    comp = g.Compose([g.Fixed(a), g.Fixed(b)], gap=0.1)
    s = comp.get(g.Ctx({}))
    assert s.length() == pytest.approx(1.6)
    assert len(s.parts) == 2 and s.parts[1][1][0, 3] == pytest.approx(1.1)
    gap = g.Gap(g.Const(0.7)).get(g.Ctx({}))
    assert gap.empty and gap.length() == pytest.approx(0.7)
    mir = g.Mirror(g.Fixed(a), (False, True, False), mode="ALTERNATE")
    assert mir.get(g.Ctx({"index": 0})) is a
    assert mir.get(g.Ctx({"index": 1})).parts[0][1][1, 1] == -1.0
    xf = g.Transform(g.Fixed(a), [g.Const(0), g.Const(0), g.Const(1)], [g.Const(0)] * 3,
                     [g.Const(1)] * 3, rand_rotation=(0, 0, math.pi), rotation_step=math.pi / 2, seed=3)
    m1 = xf.get(g.Ctx({"index": 1})).parts[0][1]
    m2 = xf.get(g.Ctx({"index": 1})).parts[0][1]
    assert np.allclose(m1, m2) and m1[2, 3] == 1.0
    ang = math.atan2(m1[1, 0], m1[0, 0])
    assert abs(round(ang / (math.pi / 2)) * (math.pi / 2) - ang) < 1e-9
    mat = g.MaterialOp(g.Fixed(a), "RANDOM", ["m1", "m2"], seed=0)
    picks = {mat.get(g.Ctx({"index": i})).materials[0][2] for i in range(20)}
    assert picks == {"m1", "m2"}


def test_segment_ids_split_and_variable():
    spec = sp.SplineSpec("POLY", [(0, 0, 0), (3, 0, 0), (6, 0, 0), (9, 0, 0)], seg_ids=[0, 2, 0])
    a, b = seg(1.0, "A"), seg(1.0, "B")
    sel = g.Conditional(g.Fixed(b), g.Fixed(a), g.MathNum("EQUAL", g.Var("segment_id"), g.Const(2)))
    asm = Assembler()
    count, trunc, _ = run([SplineJob(spec)], {g.INPUT_DEFAULT: sel},
                          lambda ctx: LinearSettings(fit="CEIL", corner_mode="NONE"), PlaceOptions(), asm)
    assert count == 9
    mesh = asm.build()
    tall = mesh.co
    assert len(tall) == 9 * 8


def test_clip_area_slices_spans():
    r = rail_of([(0, 0, 0), (10, 0, 0)])
    square = [(2.5, -1), (7.5, -1), (7.5, 1), (2.5, 1)]
    inside = clip_intervals(r, [square], True)
    assert inside == [(pytest.approx(2.5), pytest.approx(7.5))]
    outside = clip_intervals(r, [square], False)
    assert len(outside) == 2
    p = layout(r, {g.INPUT_DEFAULT: fixed(2.0)}, LinearSettings(fit="FLOOR"), {}, clip_intervals=inside)[0]
    assert p[0].cut == pytest.approx((2.5, 4.0)) and p[-1].cut == pytest.approx((6.0, 7.5))


def check_topology(mesh):
    fs = list(mesh.face_start) + [len(mesh.corner_vert)]
    for f in range(len(mesh.face_start)):
        a, b = fs[f], fs[f + 1]
        for c in range(a, b):
            v0, v1 = mesh.corner_vert[c], mesh.corner_vert[c + 1 if c + 1 < b else a]
            assert {int(v0), int(v1)} == set(int(x) for x in mesh.edges[mesh.corner_edge[c]])


def test_assembler_flip_materials_uv():
    buf = box_buf(0, 1)
    buf.uvs = [("UVMap", np.tile([[0.25, 0.5]], (len(buf.corner_vert), 1)).astype(np.float32))]
    asm = Assembler()
    asm.add(buf, np.stack([buf.co, buf.co + 2]), np.array([False, True]), overrides=(("ALL", -1, "M"),))
    uvx = np.broadcast_to(np.eye(3), (1, 3, 3)).copy()
    uvx[0, 0, 2] = 1.0
    asm.add(buf, buf.co[None], uv=uvx)
    mesh = asm.build()
    check_topology(mesh)
    assert mesh.materials == ["M", None]
    assert set(mesh.face_mat[:12]) == {0} and set(mesh.face_mat[12:]) == {1}
    assert np.allclose(mesh.uvs[0][1][-1], [1.25, 0.5])
    m = merge([buf, buf])
    assert len(m.co) == 16 and m.n_faces == 12


def test_place_bend_miter_and_rigid_upright():
    r = Rail(sp.sample(sp.SplineSpec("POLY", [(0, 0, 0), (4, 0, 0), (4, 4, 0)])), joint_zone=0.2)
    long_seg = g.Seg([(g.Geo("long", box_buf(0, 8, nx=16)), np.eye(4))], name="long")
    pl = layout(r, {g.INPUT_DEFAULT: g.Fixed(long_seg)}, LinearSettings(corner_mode="NONE"), {})[0]
    asm = Assembler()
    place(r, pl, PlaceOptions(), asm)
    mesh = asm.build()
    mid = np.isclose(long_seg.parts[0][0].mesh.co[:, 0], 4.0)
    assert np.allclose(mesh.co[mid][:, 0] + mesh.co[mid][:, 1], 4.0)
    slope = Rail(sp.sample(sp.SplineSpec("POLY", [(0, 0, 0), (4, 0, 3)])))
    post = seg(0.2, "post", centered=True, bend=False, upright=True)
    pl = layout(slope, {g.INPUT_EVENLY: g.Fixed(post)}, LinearSettings(evenly_mode="COUNT", evenly_count=1), {})[0]
    asm = Assembler()
    place(slope, pl, PlaceOptions(), asm)
    co = asm.build().co
    top = co[np.isclose(co[:, 2] - co[:, 2].min(), 1.0)]
    assert np.ptp(top[:, 2]) == pytest.approx(0.0, abs=1e-9)


def test_place_slicing_with_custom_slicer():
    calls = []

    def slicer(buf, co, no):
        calls.append((tuple(np.round(co, 6)), tuple(np.round(no, 6))))
        lo, hi = buf.co[:, 0].min(), buf.co[:, 0].max()
        return box_buf(lo, min(hi, co[0])) if no[0] < 0 else box_buf(max(lo, co[0]), hi)

    r = rail_of([(0, 0, 0), (5.5, 0, 0)])
    pl = layout(r, {g.INPUT_DEFAULT: fixed(2.0)}, LinearSettings(adaptive=False), {})[0]
    asm = Assembler()
    place(r, pl, PlaceOptions(slicer=slicer), asm)
    co = asm.build().co
    assert co[:, 0].max() == pytest.approx(5.5)
    assert len(calls) == 1 and calls[0][1][0] == -1.0
