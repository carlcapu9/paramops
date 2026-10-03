# SPDX-License-Identifier: GPL-3.0-or-later
"""Turns layout placements into world-space vertex positions (pure numpy)."""

import math
import random

import numpy as np

from .layout import SPAN, stable_seed


class SlotConfig:
    """Per-slot placement options (resolved from the Blender properties)."""

    def __init__(self, **kw):
        self.deform = True
        self.vertical = False
        self.align_x = "CENTER"     # POINT modules: CENTER PIVOT START END
        self.align_y = "PIVOT"      # PIVOT CENTER LEFT RIGHT
        self.align_z = "PIVOT"      # PIVOT BOTTOM CENTER TOP
        self.offset = (0.0, 0.0, 0.0)
        self.random_offset = (0.0, 0.0, 0.0)
        self.random_rotation = (0.0, 0.0, 0.0)   # radians (+/-)
        self.random_scale = 0.0                  # +/- fraction (uniform)
        self.random_flip_x = 0.0                 # probability
        self.random_flip_y = 0.0
        self.random_uv = (0.0, 0.0)              # +/- UV offset per module
        self.flat_top = 0.0
        self.flat_bottom = 0.0
        self.flat_center = 0.0
        self.flat_reference = "CENTER"           # CENTER LOW HIGH
        self.orient = "AUTO"                     # AUTO INCOMING OUTGOING (anchored samples)
        self.seed = 0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)

    @property
    def has_random(self):
        return (any(abs(v) > 0 for v in self.random_rotation) or self.random_scale > 0
                or self.random_flip_x > 0 or self.random_flip_y > 0)

    @property
    def has_random_offset(self):
        return any(abs(v) > 0 for v in self.random_offset)


def euler_matrix(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    my = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    mz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return mz @ my @ mx


def random_transforms(cfg, placements, global_seed, path_index):
    """Per-module random matrices (M, 3, 3), offsets (M, 3) and flip flags (M,)."""
    M = len(placements)
    mats = np.broadcast_to(np.eye(3), (M, 3, 3)).copy()
    offs = np.zeros((M, 3))
    flips = np.zeros(M, dtype=bool)
    if not (cfg.has_random or cfg.has_random_offset):
        return mats, offs, flips
    rr = cfg.random_rotation
    ro = cfg.random_offset
    for m, p in enumerate(placements):
        rng = random.Random(stable_seed(global_seed, cfg.seed, path_index, p.slot, p.ordinal))
        if cfg.has_random_offset:
            offs[m] = [rng.uniform(-ro[0], ro[0]), rng.uniform(-ro[1], ro[1]), rng.uniform(-ro[2], ro[2])]
        if not cfg.has_random or p.sliced:
            continue
        sc = 1.0 + rng.uniform(-cfg.random_scale, cfg.random_scale) if cfg.random_scale > 0 else 1.0
        fx = -1.0 if rng.random() < cfg.random_flip_x else 1.0
        fy = -1.0 if rng.random() < cfg.random_flip_y else 1.0
        rot = euler_matrix(rng.uniform(-rr[0], rr[0]), rng.uniform(-rr[1], rr[1]),
                            rng.uniform(-rr[2], rr[2]))
        mats[m] = rot @ np.diag((sc * fx, sc * fy, sc))
        flips[m] = (fx * fy) < 0
    return mats, offs, flips


def random_uv_offsets(cfg, placements, global_seed, path_index):
    """Per-module UV offsets (M, 2) or ``None`` when UV randomisation is off."""
    ru, rv = cfg.random_uv
    if ru <= 0.0 and rv <= 0.0:
        return None
    out = np.zeros((len(placements), 2))
    for m, p in enumerate(placements):
        rng = random.Random(stable_seed(global_seed, cfg.seed, path_index, p.slot, p.ordinal, "uv"))
        out[m] = (rng.uniform(-ru, ru), rng.uniform(-rv, rv))
    return out


def place_group(path, md, placements, cfg, global_seed=0, path_index=0, inv_matrix=None,
                ref_bbox=None, vertical_up=None):
    """Positions of every module of one group.

    ``md`` is the (slot-transformed, possibly sliced) sample geometry and all
    placements share it. ``ref_bbox`` is the bounding box used for mapping
    (the unsliced box for partial modules). Returns ``(positions (M, V, 3),
    flip (M,))``.
    """
    M = len(placements)
    V = md.nv
    if M == 0 or V == 0:
        return np.zeros((M, V, 3)), np.zeros(M, dtype=bool)
    base = md.co
    mats, roffs, flips = random_transforms(cfg, placements, global_seed, path_index)
    if cfg.has_random:
        co = np.einsum("mij,vj->mvi", mats, base)
    else:
        co = np.broadcast_to(base[None], (M, V, 3)).copy()

    if ref_bbox is not None and not cfg.has_random:
        mn = np.broadcast_to(np.asarray(ref_bbox[0], dtype=np.float64), (M, 3))
        mx = np.broadcast_to(np.asarray(ref_bbox[1], dtype=np.float64), (M, 3))
    else:
        mn = co.min(axis=1)
        mx = co.max(axis=1)
        if ref_bbox is not None:
            # Sliced modules: keep the unsliced reference along X.
            mn = mn.copy()
            mx = mx.copy()
            mn[:, 0] = ref_bbox[0][0]
            mx[:, 0] = ref_bbox[1][0]

    ay, az = cfg.align_y, cfg.align_z
    if ay == "CENTER":
        refy = (mn[:, 1] + mx[:, 1]) * 0.5
    elif ay == "LEFT":
        refy = mn[:, 1]
    elif ay == "RIGHT":
        refy = mx[:, 1]
    else:
        refy = np.zeros(M)
    if az == "BOTTOM":
        refz = mn[:, 2]
    elif az == "CENTER":
        refz = (mn[:, 2] + mx[:, 2]) * 0.5
    elif az == "TOP":
        refz = mx[:, 2]
    else:
        refz = np.zeros(M)

    off = np.asarray(cfg.offset, dtype=np.float64)
    x = co[:, :, 0]
    y = co[:, :, 1] - refy[:, None] + (off[1] + roffs[:, 1])[:, None]
    z = co[:, :, 2] - refz[:, None] + (off[2] + roffs[:, 2])[:, None]
    shift = off[0] + roffs[:, 0]
    vertical = cfg.vertical
    kind = placements[0].kind

    flat = vertical and (cfg.flat_top > 0 or cfg.flat_bottom > 0 or cfg.flat_center > 0)
    if kind == SPAN:
        x0 = np.array([p.x0 for p in placements]) + shift
        k = np.array([p.k for p in placements])
        lenx = np.maximum(mx[:, 0] - mn[:, 0], 1e-12)
        if cfg.deform:
            s = x0[:, None] + (x - mn[:, 0][:, None]) * k[:, None]
            pos = path.evaluate(s.ravel(), y.ravel(), z.ravel(), vertical).reshape(M, V, 3)
            if flat:
                s_mid = x0 + lenx * k * 0.5
                s_ends = (x0, x0 + lenx * k)
                h = path.elevation(s.ravel()).reshape(M, V)
                href = _reference_height(path, cfg.flat_reference, s_mid, s_ends)
                _flatten(pos, z, h, href, cfg)
        else:
            sa = x0
            sb = x0 + lenx * k
            p0 = path.point(sa)
            p1 = path.point(sb)
            C = p1 - p0
            t = (x - mn[:, 0][:, None]) / lenx[:, None]
            if vertical:
                Ch = C.copy()
                Ch[:, 2] = 0.0
                lat = np.stack([-Ch[:, 1], Ch[:, 0], np.zeros(M)], axis=1)
                nrm = np.linalg.norm(lat, axis=1)
                bad = nrm < 1e-12
                if bad.any():
                    _, _, lf, _ = path.frames((sa + sb) * 0.5, vertical=True)
                    lat[bad] = lf[bad]
                    nrm[bad] = 1.0
                lat /= nrm[:, None]
                up = np.broadcast_to((0.0, 0.0, 1.0), (M, 3))
            else:
                _, _, lf, _ = path.frames((sa + sb) * 0.5)
                cl = np.linalg.norm(C, axis=1)
                cd = C / np.maximum(cl, 1e-12)[:, None]
                cd[cl < 1e-12] = path.frames((sa + sb) * 0.5)[1][cl < 1e-12]
                lat = lf - np.einsum("ij,ij->i", lf, cd)[:, None] * cd
                lat /= np.maximum(np.linalg.norm(lat, axis=1), 1e-12)[:, None]
                up = np.cross(cd, lat)
            pos = (p0[:, None, :] + C[:, None, :] * t[..., None]
                   + lat[:, None, :] * y[..., None] + up[:, None, :] * z[..., None])
            if flat:
                h = p0[:, 2][:, None] + C[:, 2][:, None] * t
                href = _reference_height_chord(cfg.flat_reference, p0[:, 2], p1[:, 2])
                _flatten(pos, z, h, href, cfg)
    else:
        anchor = np.array([p.anchor for p in placements]) + shift
        ax = cfg.align_x
        if ax == "PIVOT":
            refx = np.zeros(M)
        elif ax == "START":
            refx = mn[:, 0]
        elif ax == "END":
            refx = mx[:, 0]
        else:
            refx = (mn[:, 0] + mx[:, 0]) * 0.5
        lx = x - refx[:, None]
        if cfg.deform:
            s = anchor[:, None] + lx
            pos = path.evaluate(s.ravel(), y.ravel(), z.ravel(), vertical).reshape(M, V, 3)
            if flat:
                h = path.elevation(s.ravel()).reshape(M, V)
                ends = (anchor + mn[:, 0] - refx, anchor + mx[:, 0] - refx)
                href = _reference_height(path, cfg.flat_reference, anchor, ends)
                _flatten(pos, z, h, href, cfg)
        else:
            P, T, Lt, N = path.frames(anchor, vertical=vertical, orient=cfg.orient)
            pos = (P[:, None, :] + T[:, None, :] * lx[..., None]
                   + Lt[:, None, :] * y[..., None] + N[:, None, :] * z[..., None])

    if inv_matrix is not None:
        im = np.asarray(inv_matrix, dtype=np.float64)
        pos = pos @ im[:3, :3].T + im[:3, 3]
        if np.linalg.det(im[:3, :3]) < 0:
            flips = ~flips
    return pos, flips


def _reference_height(path, mode, s_mid, ends):
    if mode == "LOW":
        return np.minimum(path.elevation(ends[0]), path.elevation(ends[1]))
    if mode == "HIGH":
        return np.maximum(path.elevation(ends[0]), path.elevation(ends[1]))
    return path.elevation(s_mid)


def _reference_height_chord(mode, z0, z1):
    if mode == "LOW":
        return np.minimum(z0, z1)
    if mode == "HIGH":
        return np.maximum(z0, z1)
    return (z0 + z1) * 0.5


def _flatten(pos, z, h, href, cfg):
    """Level the top / bottom bands (and optionally the middle) of sheared modules."""
    zmin = z.min(axis=1)[:, None]
    zmax = z.max(axis=1)[:, None]
    w = np.full(z.shape, float(cfg.flat_center))
    if cfg.flat_top > 0:
        w[z >= zmax - cfg.flat_top] = 1.0
    if cfg.flat_bottom > 0:
        w[z <= zmin + cfg.flat_bottom] = 1.0
    pos[:, :, 2] += w * (href[:, None] - h)
