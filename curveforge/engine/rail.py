# SPDX-License-Identifier: GPL-3.0-or-later
"""The rail: an arc-length parameterised path that segments are bent along.

Segment space: +X runs along the rail, +Y is the left side, +Z is up.

Two orientation modes:

* *follow*: the cross section follows the path normal (it tilts on slopes and
  banks with the curve tilt);
* *upright* ("vertical" in RailClone terms): the cross section stays vertical
  and is sheared along slopes.

Offsets are interpolated between per-vertex axes scaled by the miter factor,
so geometry wrapped around a sharp vertex closes exactly on its bisector
plane. Extra straight vertices next to sharp corners keep that miter blend
local, so long straight pieces are not sheared.
"""

import math

import numpy as np

WORLD_UP = np.array((0.0, 0.0, 1.0))
MITER_LIMIT = 8.0


def _rows_norm(v):
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def _unit_rows(v):
    n = _rows_norm(v)
    ok = n > 1e-12
    out = np.zeros_like(v)
    out[ok] = v[ok] / n[ok, None]
    return out, ok


def _propagate(rows, ok):
    """Fill invalid rows with the closest previous valid row (or the first valid one)."""
    if ok.all() or not ok.any():
        return rows
    idx = np.where(ok, np.arange(len(ok)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = int(np.argmax(ok))
    return rows[idx]


def _rotate_about(v, axis, angle):
    c = np.cos(angle)[:, None]
    s = np.sin(angle)[:, None]
    dot = np.einsum("ij,ij->i", axis, v)[:, None]
    return v * c + np.cross(axis, v) * s + axis * dot * (1.0 - c)


def _joint_zones(P, param, tilt, cyclic, zone):
    """Insert vertices at distance Z on both sides of sharp vertices (see module doc)."""
    n = len(P)
    if n < 3 or zone <= 0.0:
        return P, param, tilt
    d = P[1:] - P[:-1]
    ln = np.maximum(_rows_norm(d), 1e-12)
    u = d / ln[:, None]
    if cyclic:
        verts = np.arange(n - 1)
        seg_in = np.where(verts == 0, n - 2, verts - 1)
    else:
        verts = np.arange(1, n - 1)
        seg_in = verts - 1
    seg_out = verts
    a, b = u[seg_in], u[seg_out]
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0))
    ah, bh = a.copy(), b.copy()
    ah[:, 2] = 0.0
    bh[:, 2] = 0.0
    ah, ok_a = _unit_rows(ah)
    bh, ok_b = _unit_rows(bh)
    ang_h = np.where(ok_a & ok_b, np.arccos(np.clip(np.einsum("ij,ij->i", ah, bh), -1.0, 1.0)), 0.0)
    ang = np.maximum(ang, ang_h)
    sharp = ang > math.radians(2.0)
    width = np.maximum(4.0 * zone * np.tan(np.minimum(ang, math.radians(170.0)) * 0.5), 1e-4)
    seg_idx, frac = [], []
    for si, so, w in zip(seg_in[sharp], seg_out[sharp], width[sharp]):
        if w < 0.45 * ln[si]:
            seg_idx.append(si)
            frac.append(1.0 - w / ln[si])
        if w < 0.45 * ln[so]:
            seg_idx.append(so)
            frac.append(w / ln[so])
    if not seg_idx:
        return P, param, tilt
    si = np.asarray(seg_idx, dtype=np.int64)
    f = np.asarray(frac)
    key = np.concatenate([np.arange(n, dtype=np.float64), si + f])
    order = np.argsort(key, kind="stable")
    P2 = np.vstack([P, P[si] + d[si] * f[:, None]])[order]
    k2 = np.concatenate([param, param[si] + (param[si + 1] - param[si]) * f])[order]
    t2 = np.concatenate([tilt, tilt[si] + (tilt[si + 1] - tilt[si]) * f])[order]
    return P2, k2, t2


class Rail:
    """See module docstring."""

    def __init__(self, sampled, twist="Z_UP", use_tilt=True, joint_zone=0.0):
        P = np.asarray(sampled.points, dtype=np.float64)
        param = np.asarray(sampled.param, dtype=np.float64)
        tilt = np.asarray(sampled.tilt, dtype=np.float64)
        self.cyclic = bool(sampled.cyclic)
        self.n_ctrl = sampled.n_ctrl
        self.vertex_angle = sampled.vertex_angle
        P, param, tilt = _joint_zones(P, param, tilt, self.cyclic, joint_zone)
        if not use_tilt:
            tilt = np.zeros(len(P))
        self.P = P
        self.param = param
        self.tilt = tilt
        d = P[1:] - P[:-1]
        self.seg_len = np.maximum(_rows_norm(d), 1e-12)
        self.D = d / self.seg_len[:, None]
        self.S = np.concatenate(([0.0], np.cumsum(self.seg_len)))
        self.length = float(self.S[-1])
        self._frames(twist)

    # -- frames -----------------------------------------------------------------

    def _frames(self, twist):
        D = self.D
        n = len(self.P)
        into = np.empty((n, 3))
        out = np.empty((n, 3))
        into[1:] = D
        out[:-1] = D
        if self.cyclic:
            into[0] = D[-1]
            out[-1] = D[0]
        else:
            into[0] = D[0]
            out[-1] = D[-1]
        self.into, self.out = into, out
        T, ok = _unit_rows(into + out)
        T[~ok] = out[~ok]
        self.T = T

        if twist == "MINIMUM":
            side = self._parallel_transport(T)
        else:
            side, oks = _unit_rows(np.cross(WORLD_UP, T))
            if not oks.any():
                side[:] = (0.0, 1.0, 0.0)
                oks[:] = True
            side = _propagate(side, oks)
            side = _unit_rows(side - np.einsum("ij,ij->i", side, T)[:, None] * T)[0]
        up = np.cross(T, side)
        if np.any(self.tilt):
            c = np.cos(self.tilt)[:, None]
            s = np.sin(self.tilt)[:, None]
            side, up = side * c + up * s, up * c - side * s
        self.side, self.up = side, up

        # Miter-scaled axes (follow mode).
        cos_half = np.einsum("ij,ij->i", T, into)
        miter = 1.0 / np.clip(cos_half, 1.0 / MITER_LIMIT, 1.0)
        bend, okb = _unit_rows(out - into)
        gy = np.einsum("ij,ij->i", side, bend) * (miter - 1.0) * okb
        gz = np.einsum("ij,ij->i", up, bend) * (miter - 1.0) * okb
        self.Ay = side + gy[:, None] * bend
        self.Az = up + gz[:, None] * bend

        # Upright axes: horizontal miter only.
        def flat(v):
            h = v.copy()
            h[:, 2] = 0.0
            return _unit_rows(h)
        hin, ok_in = flat(into)
        hout, ok_out = flat(out)
        hin[~ok_in] = hout[~ok_in]
        hout[~ok_out] = hin[~ok_out]
        okh = ok_in | ok_out
        if not okh.any():
            hin[:] = hout[:] = (1.0, 0.0, 0.0)
            okh[:] = True
        hin = _propagate(hin, okh)
        hout = _propagate(hout, okh)
        Th, okt = _unit_rows(hin + hout)
        Th[~okt] = hout[~okt]
        self.Th = Th
        self.Sh = np.stack([-Th[:, 1], Th[:, 0], np.zeros(n)], axis=1)
        mh = 1.0 / np.clip(np.einsum("ij,ij->i", Th, hin), 1.0 / MITER_LIMIT, 1.0)
        self.Hy = self.Sh * mh[:, None]

    def _parallel_transport(self, T):
        n = len(T)
        side = np.empty((n, 3))
        first = np.cross(WORLD_UP, T[0])
        if np.linalg.norm(first) < 1e-6:
            first = np.cross(T[0], np.cross((0.0, 1.0, 0.0), T[0]))
        side[0] = first / np.linalg.norm(first)
        for i in range(1, n):
            a, b = T[i - 1], T[i]
            v = side[i - 1]
            axis = np.cross(a, b)
            sn = np.linalg.norm(axis)
            if sn > 1e-12:
                v = _rotate_about(v[None], (axis / sn)[None], np.array([math.atan2(sn, float(a @ b))]))[0]
            v = v - float(v @ b) * b
            nv = np.linalg.norm(v)
            side[i] = v / nv if nv > 1e-12 else side[i - 1]
        if self.cyclic and n > 2:
            mismatch = math.atan2(float(np.cross(side[-1], side[0]) @ T[0]), float(side[-1] @ side[0]))
            side = _rotate_about(side, T, mismatch * self.S / max(self.length, 1e-12))
        return side

    # -- evaluation -------------------------------------------------------------

    def locate(self, s):
        s = np.asarray(s, dtype=np.float64)
        if self.cyclic:
            s = np.mod(s, self.length)
        i = np.clip(np.searchsorted(self.S, s, side="right") - 1, 0, len(self.P) - 2)
        t = (s - self.S[i]) / self.seg_len[i]
        if self.cyclic:
            t = np.clip(t, 0.0, 1.0)
        return s, i, t

    def position(self, s):
        """Centre line (straight extrapolation past the ends of open rails)."""
        _, i, t = self.locate(np.atleast_1d(s))
        return self.P[i] + (self.P[i + 1] - self.P[i]) * t[:, None]

    def height(self, s):
        return self.position(s)[:, 2]

    def deform(self, s, y, z, upright=False):
        """World positions of segment-space points (s along the rail, y left, z up)."""
        _, i, t = self.locate(np.atleast_1d(s))
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        w1 = np.clip(t, 0.0, 1.0)[:, None]
        w0 = 1.0 - w1
        base = self.P[i] + (self.P[i + 1] - self.P[i]) * t[:, None]
        if upright:
            out = base + (w0 * self.Hy[i] + w1 * self.Hy[i + 1]) * y[:, None]
            out[:, 2] += z
            return out
        return (base + (w0 * self.Ay[i] + w1 * self.Ay[i + 1]) * y[:, None]
                + (w0 * self.Az[i] + w1 * self.Az[i + 1]) * z[:, None])

    def frames(self, s, upright=False, orient="BISECTOR"):
        """Orthonormal frames (origin, forward, left, up) at arc lengths ``s``."""
        sw, i, t = self.locate(np.atleast_1d(s))
        origin = self.P[i] + (self.P[i + 1] - self.P[i]) * t[:, None]
        fwd = self.D[i].copy()
        w1 = np.clip(t, 0.0, 1.0)[:, None]
        left = self.side[i] * (1.0 - w1) + self.side[i + 1] * w1
        v0 = np.abs(sw - self.S[i]) < 1e-6
        v1 = np.abs(sw - self.S[i + 1]) < 1e-6
        on = v0 | v1
        if on.any():
            vi = np.where(v1, i + 1, i)[on]
            src = {"INCOMING": self.into, "OUTGOING": self.out}.get(orient, self.T)
            fwd[on] = src[vi]
            left[on] = self.side[vi]
        if upright:
            fh = fwd.copy()
            fh[:, 2] = 0.0
            fh, ok = _unit_rows(fh)
            if (~ok).any():
                alt = self.Th[i] * (1.0 - w1) + self.Th[i + 1] * w1
                fh[~ok] = _unit_rows(alt[~ok])[0]
            lh = np.stack([-fh[:, 1], fh[:, 0], np.zeros(len(fh))], axis=1)
            return origin, fh, lh, np.broadcast_to(WORLD_UP, fh.shape).copy()
        left = left - np.einsum("ij,ij->i", left, fwd)[:, None] * fwd
        left, ok = _unit_rows(left)
        if (~ok).any():
            left[~ok] = _unit_rows(np.cross(WORLD_UP, fwd[~ok]))[0]
        return origin, fwd, left, np.cross(fwd, left)

    def slope(self, s):
        """Slope angle (radians, positive uphill) of the rail at ``s``."""
        _, i, _ = self.locate(np.atleast_1d(s))
        d = self.D[i]
        return np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))

    # -- control parameter --------------------------------------------------------

    def s_at(self, k):
        """Arc length of control parameter ``k`` (control point k sits at k)."""
        p, S = self.param, self.S
        k = np.asarray(k, dtype=np.float64)
        if self.cyclic:
            lo = min(p[0], p[-1])
            k = lo + np.mod(k - lo, self.n_ctrl)
        if p[-1] >= p[0]:
            return np.interp(k, p, S)
        return np.interp(k, p[::-1], S[::-1])

    def param_at(self, s):
        s = np.mod(s, self.length) if self.cyclic else np.asarray(s, dtype=np.float64)
        return np.interp(s, self.S, self.param)
