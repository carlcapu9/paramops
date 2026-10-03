import numpy as np
import pytest

from core import pathmath as pm
from core.layout import LayoutSettings, PathInfo, SlotInfo, VariantInfo, layout_path
from core.meshdata import MeshData, assemble, merge
from core.placement import SlotConfig, place_group


def grid_box(length=2.0, height=1.0, width=0.2, nx=8):
    """Box subdivided along X (so it can bend), pivot at its start / bottom / centre-y."""
    co, faces = [], []
    ys = (-width / 2, width / 2)
    zs = (0.0, height)
    for i in range(nx + 1):
        x = length * i / nx
        for y in ys:
            for z in zs:
                co.append((x, y, z))

    def vid(i, a, b):
        return i * 4 + a * 2 + b

    for i in range(nx):
        faces.append((vid(i, 0, 0), vid(i + 1, 0, 0), vid(i + 1, 0, 1), vid(i, 0, 1)))
        faces.append((vid(i, 1, 0), vid(i, 1, 1), vid(i + 1, 1, 1), vid(i + 1, 1, 0)))
        faces.append((vid(i, 0, 1), vid(i + 1, 0, 1), vid(i + 1, 1, 1), vid(i, 1, 1)))
        faces.append((vid(i, 0, 0), vid(i, 1, 0), vid(i + 1, 1, 0), vid(i + 1, 0, 0)))
    faces.append((vid(0, 0, 0), vid(0, 0, 1), vid(0, 1, 1), vid(0, 1, 0)))
    faces.append((vid(nx, 0, 0), vid(nx, 1, 0), vid(nx, 1, 1), vid(nx, 0, 1)))
    return MeshData.from_faces(co, faces)


def check_topology(asm):
    """Every corner edge must join the corner vertex and the next one."""
    fs = list(asm.face_start) + [len(asm.loop_vert)]
    for f in range(len(asm.face_start)):
        a, b = fs[f], fs[f + 1]
        for c in range(a, b):
            v0 = asm.loop_vert[c]
            v1 = asm.loop_vert[c + 1 if c + 1 < b else a]
            e = asm.edges[asm.loop_edge[c]]
            assert {int(v0), int(v1)} == {int(e[0]), int(e[1])}


def test_from_faces_and_flip_consistency():
    md = grid_box()
    assert md.nv == 36 and md.nf == 34
    flipped = md.flipped()
    asm = assemble([(md, md.co[None], np.array([False])), (flipped, md.co[None], np.array([True]))])
    check_topology(asm)


def test_mirror_transform_flips_winding():
    md = MeshData.from_faces([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [(0, 1, 2)])
    mirrored = md.transformed(np.diag((-1.0, 1.0, 1.0)))
    assert list(mirrored.loop_vert) == [0, 2, 1]
    a, b, c = mirrored.co[mirrored.loop_vert]
    assert np.cross(b - a, c - a)[2] > 0  # still faces +Z


def test_assemble_offsets_and_materials():
    m1, m2 = object(), object()
    a = MeshData.box((0, 0, 0), (1, 1, 1), material=m1)
    b = MeshData.box((0, 0, 0), (1, 1, 1), material=m2)
    pos = np.stack([a.co, a.co + 2.0])
    asm = assemble([(a, pos, np.array([False, True])), (b, b.co[None], np.array([False]))])
    assert len(asm.co) == 24 and len(asm.face_start) == 18 and len(asm.edges) == 36
    assert asm.materials == [m1, m2]
    assert set(asm.mat_index[:12]) == {0} and set(asm.mat_index[12:]) == {1}
    check_topology(asm)


def test_attributes_union_fill_zeros():
    a = MeshData.box((0, 0, 0), (1, 1, 1))
    b = MeshData.box((0, 0, 0), (1, 1, 1))
    a.attrs["sharp_face"] = ("FACE", "BOOLEAN", np.ones(6, dtype=bool))
    a.uvs = [("UVMap", np.ones((24, 2), np.float32))]
    asm = assemble([(a, a.co[None], np.array([False])), (b, b.co[None], np.array([False]))])
    sf = asm.attrs["sharp_face"][2]
    assert sf.tolist() == [True] * 6 + [False] * 6
    assert asm.uvs[0][0] == "UVMap" and asm.uvs[0][1].shape == (48, 2)


def test_merge_two_parts():
    a = MeshData.box((0, 0, 0), (1, 1, 1))
    b = MeshData.box((2, 0, 0), (3, 1, 1))
    m = merge([a, b])
    assert m.nv == 16 and m.nf == 12
    assert list(m.face_size) == [4] * 12


def straight_path(length=10.0):
    return pm.Path(pm.sample_spline(pm.SplineInput("POLY", [(0, 0, 0), (length, 0, 0)])))


def test_span_modules_follow_the_path():
    md = grid_box(2.0)
    path = straight_path(10.0)
    sl = SlotInfo("default", variants=[VariantInfo(0.0, 2.0)])
    res = layout_path(PathInfo(path.L), {"default": sl}, LayoutSettings())
    pos, flips = place_group(path, md, res.placements, SlotConfig())
    assert pos.shape == (5, 36, 3)
    assert pos[..., 0].min() == pytest.approx(0.0)
    assert pos[..., 0].max() == pytest.approx(10.0)
    assert not flips.any()


def test_bent_module_around_corner_is_mitered():
    md = grid_box(8.0, nx=16)
    pts = [(0, 0, 0), (4, 0, 0), (4, 4, 0)]
    path = pm.Path(pm.sample_spline(pm.SplineInput("POLY", pts)))
    sl = SlotInfo("default", variants=[VariantInfo(0.0, 8.0)])
    res = layout_path(PathInfo(path.L), {"default": sl}, LayoutSettings())
    pos, _ = place_group(path, md, res.placements, SlotConfig(deform=True))
    # The vertices that sat at x == 4 (middle of the box) land on the miter line x == 4 - y'.
    mid = np.isclose(md.co[:, 0], 4.0)
    p = pos[0][mid]
    assert np.allclose(p[:, 0] + p[:, 1], 4.0)


def test_point_modules_rigid_and_aligned():
    post = MeshData.box((-0.05, -0.05, 0.0), (0.05, 0.05, 1.0))
    path = straight_path()
    sl = SlotInfo("evenly", variants=[VariantInfo(-0.05, 0.05)])
    st = LayoutSettings(evenly_mode="COUNT", evenly_count=2)
    res = layout_path(PathInfo(path.L), {"evenly": sl, "default": SlotInfo("default")}, st)
    pts = [p for p in res.placements if p.slot == "evenly"]
    pos, _ = place_group(path, post, pts, SlotConfig(deform=False, align_z="BOTTOM"))
    centres = pos.mean(axis=1)
    assert np.allclose(centres[:, 0], [10 / 3, 20 / 3])
    assert np.allclose(pos[..., 2].min(), 0.0)


def test_vertical_flat_top_levels_the_cap():
    md = grid_box(1.0, height=1.0, nx=4)
    path = pm.Path(pm.sample_spline(pm.SplineInput("POLY", [(0, 0, 0), (4, 0, 2)])))
    sl = SlotInfo("default", variants=[VariantInfo(0.0, 1.0)])
    res = layout_path(PathInfo(path.L), {"default": sl}, LayoutSettings(fit_mode="COUNT", fit_count=1))
    cfg = SlotConfig(deform=True, vertical=True, flat_top=0.05)
    pos, _ = place_group(path, md, res.placements, cfg)
    top = np.isclose(md.co[:, 2], 1.0)
    assert np.ptp(pos[0][top][:, 2]) == pytest.approx(0.0, abs=1e-9)
    bottom = np.isclose(md.co[:, 2], 0.0)
    assert np.ptp(pos[0][bottom][:, 2]) > 1.0  # the rest still follows the slope


def test_random_flip_marks_winding():
    md = grid_box(1.0, nx=2)
    path = straight_path()
    sl = SlotInfo("default", variants=[VariantInfo(0.0, 1.0)])
    res = layout_path(PathInfo(path.L), {"default": sl}, LayoutSettings())
    pos, flips = place_group(path, md, res.placements, SlotConfig(random_flip_y=0.5, seed=1))
    assert flips.any() and not flips.all()
    # Flipped modules still span their slot along X.
    assert pos[..., 0].min() == pytest.approx(0.0) and pos[..., 0].max() == pytest.approx(10.0)
