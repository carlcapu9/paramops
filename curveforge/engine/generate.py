# SPDX-License-Identifier: GPL-3.0-or-later
"""Runs one Linear generator over its splines and feeds the assembler."""

import math

import numpy as np

from .graph import Ctx
from .linear import layout
from .place import place
from .rail import Rail
from .spline import reverse, sample


def point_in_polygon(px, py, poly):
    """Even-odd test of points (arrays) against a 2D polygon (k, 2)."""
    inside = np.zeros(len(px), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    return inside


def clip_intervals(rail, polygons, keep_inside=True):
    """Rail ranges [(s0, s1), ...] inside (or outside) the union of closed 2D polygons."""
    P = rail.P
    S = rail.S
    cuts = [0.0, rail.length]
    for poly in polygons:
        poly = np.asarray(poly, dtype=np.float64)[:, :2]
        q0 = poly
        q1 = np.roll(poly, -1, axis=0)
        for i in range(len(P) - 1):
            a, b = P[i, :2], P[i + 1, :2]
            d = b - a
            e = q1 - q0
            den = d[0] * e[:, 1] - d[1] * e[:, 0]
            ok = np.abs(den) > 1e-12
            if not ok.any():
                continue
            w = q0 - a
            t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1.0)
            u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1.0)
            hit = ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
            for tt in t[hit]:
                cuts.append(S[i] + tt * (S[i + 1] - S[i]))
    cuts = sorted(set(round(c, 9) for c in cuts))
    out = []
    for c0, c1 in zip(cuts, cuts[1:]):
        if c1 - c0 <= 1e-7:
            continue
        mid = rail.position((c0 + c1) * 0.5)[0]
        inside = any(point_in_polygon(np.array([mid[0]]), np.array([mid[1]]), np.asarray(p)[:, :2])[0]
                     for p in polygons)
        if inside == keep_inside:
            if out and abs(out[-1][1] - c0) < 1e-7:
                out[-1] = (out[-1][0], c1)
            else:
                out.append((c0, c1))
    return out


def _seg_id_tools(rail, spec):
    ids = spec.seg_ids
    n_seg = spec.n_segments
    if not len(ids) or not np.any(ids):
        return (lambda s: 0), []

    def id_at(s):
        k = int(math.floor(float(rail.param_at(s)) + 1e-9))
        if spec.cyclic:
            k %= max(n_seg, 1)
        k = min(max(k, 0), n_seg - 1)
        return int(ids[k])

    breaks = []
    for k in range(1, n_seg):
        if ids[k] != ids[k - 1]:
            breaks.append(float(rail.s_at(k)))
    if spec.cyclic and n_seg > 1 and ids[0] != ids[-1]:
        breaks.append(float(rail.s_at(0)) % rail.length)
    return id_at, sorted(breaks)


class SplineJob:
    """A spline as seen by one generator (with the Spline node options)."""

    def __init__(self, spec, reverse=False, resolution=0, twist="Z_UP", use_tilt=True):
        self.spec = spec
        self.reverse = reverse
        self.resolution = resolution
        self.twist = twist
        self.use_tilt = use_tilt


def _extent(placed, opts):
    ext = 0.0
    for pl in placed:
        seg = pl.seg
        if not seg.parts:
            continue
        ay, az = seg.align[1], seg.align[2]
        lo, hi = seg.bmin, seg.bmax
        ry = {"MIN": lo[1], "CENTER": (lo[1] + hi[1]) * 0.5, "MAX": hi[1]}.get(ay, 0.0)
        rz = {"MIN": lo[2], "CENTER": (lo[2] + hi[2]) * 0.5, "MAX": hi[2]}.get(az, 0.0)
        oy = seg.offset[1] + opts.offset_y
        oz = seg.offset[2] + opts.offset_z
        ext = max(ext, abs(lo[1] - ry + oy), abs(hi[1] - ry + oy), abs(lo[2] - rz + oz),
                  abs(hi[2] - rz + oz))
    return float(ext)


def run(jobs, sources, settings_for, place_opts, asm, inv_matrix=None, first_spline=0,
        clip=None, seed=0, slice_cache=None):
    """Run a generator.

    ``settings_for(ctx) -> LinearSettings`` evaluates the generator parameters
    for one spline (they may depend on numeric nodes). ``clip`` is ``None`` or
    ``(polygons, keep_inside)``. Returns ``(segment_count, truncated, splines)``.
    """
    count = 0
    truncated = False
    spline_no = first_spline
    for job in jobs:
        sampled = sample(job.spec, job.resolution)
        if sampled is None:
            continue
        if job.reverse:
            sampled = reverse(sampled)
        rail = Rail(sampled, twist=job.twist, use_tilt=job.use_tilt)
        if rail.length <= 1e-6:
            continue
        base = {"spline": spline_no, "spline_length": rail.length,
                "spline_material": job.spec.material}
        spline_no += 1
        st = settings_for(Ctx(dict(base), {}, seed))
        st.seed = seed
        id_at, id_breaks = _seg_id_tools(rail, job.spec)
        intervals = clip_intervals(rail, clip[0], clip[1]) if clip else None
        placed, trunc = layout(rail, sources, st, base, id_at, intervals, id_breaks)
        truncated = truncated or trunc
        count += len(placed)
        ext = _extent(placed, place_opts)
        deform_rail = Rail(sampled, twist=job.twist, use_tilt=job.use_tilt, joint_zone=ext) if ext > 0 else rail
        place(deform_rail, placed, place_opts, asm, inv_matrix, slice_cache)
    return count, truncated, spline_no
