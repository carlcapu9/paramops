# SPDX-License-Identifier: GPL-3.0-or-later
"""From placed segments to vertex positions: bending, rigid placement and batching."""

import numpy as np

from .graph import INPUT_CORNER
from .linear import SPAN


class PlaceOptions:
    """Generator-level defaults and output options."""

    def __init__(self, **kw):
        self.bend = True
        self.upright = False
        self.slice = True
        self.offset_y = 0.0
        self.offset_z = 0.0
        self.uv_mode = "KEEP"     # KEEP RAIL
        self.uv_scale = 1.0
        self.instance = False     # rigid segments become instances (segments can override)
        self.slicer = None        # callable(buf, plane_co, plane_no, keep_positive) -> buf
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)


def _x_ref(seg, kind, inp):
    """(reference x in segment space, shift in segment units) for the X alignment."""
    mode = seg.align[0]
    lo, hi = float(seg.bmin[0]), float(seg.bmax[0])
    core = seg.size if seg.size else hi - lo
    if kind == SPAN:
        if mode in ("AUTO", "MIN"):
            return lo, 0.0
        if mode == "MAX":
            return hi, core
        if mode == "CENTER":
            return (lo + hi) * 0.5, core * 0.5
        return 0.0, 0.0
    if mode == "AUTO":
        mode = "PIVOT" if inp == INPUT_CORNER else "CENTER"
    if mode == "MIN":
        return lo, 0.0
    if mode == "MAX":
        return hi, 0.0
    if mode == "CENTER":
        return (lo + hi) * 0.5, 0.0
    return 0.0, 0.0


def _yz_ref(seg):
    ay, az = seg.align[1], seg.align[2]
    lo, hi = seg.bmin, seg.bmax
    ry = {"MIN": lo[1], "CENTER": (lo[1] + hi[1]) * 0.5, "MAX": hi[1]}.get(ay, 0.0)
    rz = {"MIN": lo[2], "CENTER": (lo[2] + hi[2]) * 0.5, "MAX": hi[2]}.get(az, 0.0)
    return float(ry), float(rz)


def _cut_plane(m, c, keep_above):
    """Plane (co, normal) in part space for segment-space x == c."""
    n = np.array(m[0, :3], dtype=np.float64)
    nn = float(n @ n)
    if nn < 1e-18:
        return None
    co = n * (c - m[0, 3]) / nn
    return co, (n if keep_above else -n)


class _Item:
    __slots__ = ("geo", "m", "pl", "seg", "bend", "upright", "xref", "xshift", "yref", "zref")


def place(rail, placed, opts, asm, inv_matrix=None, slice_cache=None):
    """Add the geometry of ``placed`` (one rail) to the :class:`Assembler` ``asm``."""
    if slice_cache is None:
        slice_cache = {}
    batches = {}
    inst_batches = {}
    for pl in placed:
        seg = pl.seg
        if not seg.parts:
            continue
        bend = opts.bend if seg.bend is None else seg.bend
        upright = opts.upright if seg.upright is None else seg.upright
        xref, xshift = _x_ref(seg, pl.kind, pl.inp)
        yref, zref = _yz_ref(seg)
        instanced = ((opts.instance if seg.instance is None else seg.instance) and not bend
                     and pl.cut is None and not seg.materials and seg.uv is None and opts.uv_mode != "RAIL")
        for geo, m in seg.parts:
            it = _Item()
            it.geo, it.m, it.pl, it.seg = geo, m, pl, seg
            it.bend, it.upright = bend, upright
            it.xref, it.xshift, it.yref, it.zref = xref, xshift, yref, zref
            if instanced and geo.inst is not None:
                inst_batches.setdefault((geo.inst[0], id(geo), pl.kind, upright, seg.orient), []).append(it)
                continue
            buf = geo.mesh
            cut_key = None
            if pl.cut is not None and pl.kind == SPAN and (seg.slice if seg.slice is not None else opts.slice):
                buf, cut_key = _sliced(it, pl, opts, slice_cache)
                if buf is None:
                    continue
            key = (id(buf), pl.kind, bend, upright, seg.materials and tuple(
                (k, s, id(mm)) for k, s, mm in seg.materials), cut_key)
            entry = batches.get(key)
            if entry is None:
                batches[key] = entry = (buf, [])
            entry[1].append(it)
    for buf, items in batches.values():
        _emit(rail, buf, items, opts, asm, inv_matrix)
    for items in inst_batches.values():
        _emit_instances(rail, items, opts, asm, inv_matrix)


def _sliced(it, pl, opts, cache):
    """Geometry of one part cut to the rail range ``pl.cut``."""
    lo_s, hi_s = pl.cut
    k = pl.k if pl.k > 1e-12 else 1.0
    seg_lo = it.xref + (lo_s - pl.x0) / k - it.xshift - it.seg.offset[0] / k
    seg_hi = it.xref + (hi_s - pl.x0) / k - it.xshift - it.seg.offset[0] / k
    m = it.m
    buf = it.geo.mesh
    # Segment-space x range of this part.
    corners = np.array([(a, b, c, 1.0) for a in (it.geo.bmin[0], it.geo.bmax[0])
                        for b in (it.geo.bmin[1], it.geo.bmax[1])
                        for c in (it.geo.bmin[2], it.geo.bmax[2])]) @ m.T
    pmin, pmax = corners[:, 0].min(), corners[:, 0].max()
    if pmax <= seg_lo + 1e-9 or pmin >= seg_hi - 1e-9:
        return None, None
    if pmin >= seg_lo - 1e-9 and pmax <= seg_hi + 1e-9:
        return buf, None
    key = (id(buf), tuple(np.round(m[0], 6)), round(seg_lo, 6), round(seg_hi, 6))
    if key in cache:
        return cache[key], key
    out = buf
    if opts.slicer is not None:
        if pmin < seg_lo - 1e-9:
            plane = _cut_plane(m, seg_lo, True)
            if plane is not None:
                out = opts.slicer(out, plane[0], plane[1])
        if pmax > seg_hi + 1e-9 and out is not None and len(out.co):
            plane = _cut_plane(m, seg_hi, False)
            if plane is not None:
                out = opts.slicer(out, plane[0], plane[1])
    if out is not None and not len(out.co):
        out = None
    cache[key] = out
    return out, key


def _emit(rail, buf, items, opts, asm, inv_matrix):
    n = len(items)
    V = len(buf.co)
    mats = np.stack([it.m for it in items])
    co = np.einsum("mij,vj->mvi", mats[:, :3, :3], buf.co) + mats[:, None, :3, 3]
    flip = np.linalg.det(mats[:, :3, :3]) < 0.0

    pls = [it.pl for it in items]
    offs = np.array([it.seg.offset for it in items], dtype=np.float64)
    yref = np.array([it.yref for it in items])
    zref = np.array([it.zref for it in items])
    xref = np.array([it.xref for it in items])
    x = co[:, :, 0]
    y = co[:, :, 1] - yref[:, None] + (offs[:, 1] + opts.offset_y)[:, None]
    z = co[:, :, 2] - zref[:, None] + (offs[:, 2] + opts.offset_z)[:, None]
    kind = pls[0].kind
    bend, upright = items[0].bend, items[0].upright
    if kind == SPAN:
        x0 = np.array([p.x0 for p in pls]) + offs[:, 0]
        k = np.array([p.k for p in pls])
        shift = np.array([it.xshift for it in items])
        s = x0[:, None] + (x - xref[:, None] + shift[:, None]) * k[:, None]
        if bend:
            pos = rail.deform(s.ravel(), y.ravel(), z.ravel(), upright).reshape(n, V, 3)
        else:
            core = np.array([(it.seg.size or float(it.seg.bmax[0] - it.seg.bmin[0])) for it in items])
            sa = x0 + (np.array([float(it.seg.bmin[0]) for it in items]) - xref + shift) * k
            sb = sa + np.maximum(core, 1e-9) * k
            p0 = rail.position(sa)
            p1 = rail.position(sb)
            chord = p1 - p0
            t = (s - sa[:, None]) / np.maximum(sb - sa, 1e-12)[:, None]
            if upright:
                h = chord.copy()
                h[:, 2] = 0.0
                left = np.stack([-h[:, 1], h[:, 0], np.zeros(n)], axis=1)
                ln = np.linalg.norm(left, axis=1)
                bad = ln < 1e-12
                if bad.any():
                    left[bad] = rail.frames((sa + sb)[bad] * 0.5, upright=True)[2]
                    ln[bad] = 1.0
                left /= ln[:, None]
                up = np.broadcast_to((0.0, 0.0, 1.0), (n, 3))
            else:
                _o, fwd, left, _u = rail.frames((sa + sb) * 0.5)
                cl = np.linalg.norm(chord, axis=1)
                d = np.where(cl[:, None] > 1e-12, chord / np.maximum(cl, 1e-12)[:, None], fwd)
                left = left - np.einsum("ij,ij->i", left, d)[:, None] * d
                left /= np.maximum(np.linalg.norm(left, axis=1), 1e-12)[:, None]
                up = np.cross(d, left)
            pos = (p0[:, None] + chord[:, None] * t[..., None] + left[:, None] * y[..., None]
                   + up[:, None] * z[..., None])
    else:
        anchor = np.array([p.anchor for p in pls]) + offs[:, 0]
        lx = x - xref[:, None]
        s = anchor[:, None] + lx
        if bend:
            pos = rail.deform(s.ravel(), y.ravel(), z.ravel(), upright).reshape(n, V, 3)
        else:
            orient = items[0].seg.orient
            o, fwd, left, up = rail.frames(anchor, upright=upright, orient=orient)
            pos = (o[:, None] + fwd[:, None] * lx[..., None] + left[:, None] * y[..., None]
                   + up[:, None] * z[..., None])

    if inv_matrix is not None:
        im = np.asarray(inv_matrix, dtype=np.float64)
        pos = pos @ im[:3, :3].T + im[:3, 3]
        if np.linalg.det(im[:3, :3]) < 0.0:
            flip = ~flip

    uvx = None
    if any(it.seg.uv is not None for it in items) or (opts.uv_mode == "RAIL" and buf.uvs):
        uvx = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for i, it in enumerate(items):
            if it.seg.uv is not None:
                uvx[i] = it.seg.uv
    u_rail = s * opts.uv_scale if opts.uv_mode == "RAIL" else None
    asm.add(buf, pos, flip, overrides=items[0].seg.materials, uv=uvx, u_override=u_rail)


def _rigid_frames(rail, items, opts):
    """Affine maps (n, 4, 4) from segment space to world for rigid placements."""
    n = len(items)
    pls = [it.pl for it in items]
    offs = np.array([it.seg.offset for it in items], dtype=np.float64)
    yref = np.array([it.yref for it in items])
    zref = np.array([it.zref for it in items])
    xref = np.array([it.xref for it in items])
    oy = offs[:, 1] + opts.offset_y - yref
    oz = offs[:, 2] + opts.offset_z - zref
    upright = items[0].upright
    A = np.zeros((n, 4, 4))
    A[:, 3, 3] = 1.0
    if pls[0].kind == SPAN:
        x0 = np.array([p.x0 for p in pls]) + offs[:, 0]
        k = np.array([p.k for p in pls])
        shift = np.array([it.xshift for it in items])
        core = np.array([(it.seg.size or float(it.seg.bmax[0] - it.seg.bmin[0])) for it in items])
        sa = x0 + (np.array([float(it.seg.bmin[0]) for it in items]) - xref + shift) * k
        sb = sa + np.maximum(core, 1e-9) * k
        p0 = rail.position(sa)
        chord = rail.position(sb) - p0
        span = np.maximum(sb - sa, 1e-12)
        a = k / span
        b = (x0 + (shift - xref) * k - sa) / span
        if upright:
            h = chord.copy()
            h[:, 2] = 0.0
            left = np.stack([-h[:, 1], h[:, 0], np.zeros(n)], axis=1)
            ln = np.linalg.norm(left, axis=1)
            bad = ln < 1e-12
            if bad.any():
                left[bad] = rail.frames((sa + sb)[bad] * 0.5, upright=True)[2]
                ln[bad] = 1.0
            left /= ln[:, None]
            up = np.broadcast_to((0.0, 0.0, 1.0), (n, 3))
        else:
            _o, fwd, left, _u = rail.frames((sa + sb) * 0.5)
            cl = np.linalg.norm(chord, axis=1)
            d = np.where(cl[:, None] > 1e-12, chord / np.maximum(cl, 1e-12)[:, None], fwd)
            left = left - np.einsum("ij,ij->i", left, d)[:, None] * d
            left /= np.maximum(np.linalg.norm(left, axis=1), 1e-12)[:, None]
            up = np.cross(d, left)
        A[:, :3, 0] = chord * a[:, None]
        A[:, :3, 1] = left
        A[:, :3, 2] = up
        A[:, :3, 3] = p0 + chord * b[:, None] + left * oy[:, None] + up * oz[:, None]
    else:
        anchor = np.array([p.anchor for p in pls]) + offs[:, 0]
        o, fwd, left, up = rail.frames(anchor, upright=upright, orient=items[0].seg.orient)
        A[:, :3, 0] = fwd
        A[:, :3, 1] = left
        A[:, :3, 2] = up
        A[:, :3, 3] = o - fwd * xref[:, None] + left * oy[:, None] + up * oz[:, None]
    return A


def _emit_instances(rail, items, opts, asm, inv_matrix):
    """Rigid copies as instances of the source object (no geometry is generated)."""
    A = _rigid_frames(rail, items, opts)
    parts = np.stack([it.m for it in items])
    name, local = items[0].geo.inst
    mats = A @ parts @ np.asarray(local, dtype=np.float64)
    if inv_matrix is not None:
        mats = np.asarray(inv_matrix, dtype=np.float64) @ mats
    asm.add_instances(name, mats)
