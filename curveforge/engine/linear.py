# SPDX-License-Identifier: GPL-3.0-or-later
"""The Linear generator: lays segments out along one rail.

Order of work for every spline (as in a "linear one spline" generator):

1. trim / extend the rail ends (Clip Start / End);
2. Start and End segments at the ends of open splines;
3. hard breaks: corners (with or without a Corner segment), markers and
   Segment ID changes split the rail into sections;
4. Evenly segments inside every section (or along the whole spline);
5. Default segments fill what is left, fitted (adaptive) or at real size;
6. an optional clipping area keeps only the parts of the rail inside / outside
   a closed curve, slicing segments at its border.

Every segment request goes through the operator graph with a context that
describes where the segment will be (index, distance, section...). Default
segments are requested twice: a first pass finds how many fit, a second pass
asks again with the final positions and section counts.
"""

import math

import numpy as np

from .graph import (Ctx, INPUT_CORNER, INPUT_DEFAULT, INPUT_END, INPUT_EVENLY, INPUT_MARKER,
                    INPUT_START, hash01)

EPS = 1e-6
SPAN, ANCHOR = "SPAN", "ANCHOR"


class LinearSettings:
    def __init__(self, **kw):
        self.clip_start = 0.0
        self.clip_end = 0.0
        self.clip_percent = False
        self.extend = False
        self.spacing = 0.0
        self.adaptive = True
        self.fit = "ROUND"            # ROUND CEIL FLOOR COUNT
        self.count = 1
        self.align = "START"          # START CENTER END SPREAD (real size)
        self.remainder = "SLICE"      # SLICE SCALE EMPTY (real size)
        self.corner_mode = "SHARP"    # SHARP ALL NONE
        self.corner_angle = math.radians(2.0)
        self.corner_split = True
        self.split_ids = True
        self.evenly_mode = "DISTANCE"  # DISTANCE COUNT
        self.evenly_distance = 2.0
        self.evenly_count = 1
        self.evenly_offset = 0.0
        self.evenly_scope = "SECTION"  # SECTION SPLINE
        self.marker_mode = "NONE"      # NONE VERTICES DISTANCES REPEAT
        self.marker_list = ""
        self.marker_step = 5.0
        self.marker_offset = 0.0
        self.max_segments = 50000
        self.seed = 0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)


class Placed:
    """One segment along the rail.

    SPAN: occupies [s0, s1]; the geometry starts at ``x0`` and is stretched by
    ``k``. ANCHOR: placed at ``anchor`` (corners, evenly, markers). ``cut``
    holds (lo, hi) rail positions to slice at, or None.
    """

    __slots__ = ("seg", "kind", "s0", "s1", "x0", "k", "anchor", "cut", "inp", "index")

    def __init__(self, seg, kind, s0, s1, x0=0.0, k=1.0, anchor=0.0, inp=0, index=0):
        self.seg = seg
        self.kind = kind
        self.s0 = s0
        self.s1 = s1
        self.x0 = x0
        self.k = k
        self.anchor = anchor
        self.cut = None
        self.inp = inp
        self.index = index

    def __repr__(self):
        return "Placed(%s %s [%.3f, %.3f] k=%.3f)" % (self.seg.name, self.kind, self.s0, self.s1, self.k)


def _parse_list(text):
    out = []
    for tok in (text or "").replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def marker_positions(st, rail, length):
    """Marker (s, number) pairs for one rail."""
    mode = st.marker_mode
    res = []
    if mode == "VERTICES":
        n = rail.n_ctrl
        for tok in _parse_list(st.marker_list):
            if tok.lower() == "all":
                ks = range(n)
            elif "-" in tok[1:]:
                a, b = tok.split("-", 1) if not tok.startswith("-") else (tok, tok)
                try:
                    ks = range(int(a), int(b) + 1)
                except ValueError:
                    continue
            else:
                try:
                    ks = [int(tok)]
                except ValueError:
                    continue
            for k in ks:
                if 0 <= k < n:
                    res.append(float(rail.s_at(k)))
    elif mode == "DISTANCES":
        for tok in _parse_list(st.marker_list):
            pct = tok.endswith("%")
            try:
                v = float(tok[:-1] if pct else tok)
            except ValueError:
                continue
            s = length * v / 100.0 if pct else v
            if v < 0:
                s = length + s
            if -EPS <= s <= length + EPS:
                res.append(min(max(s, 0.0), length))
    elif mode == "REPEAT" and st.marker_step > 1e-4:
        s = st.marker_offset
        while s <= length + EPS and len(res) < st.max_segments:
            if s >= -EPS:
                res.append(min(max(s, 0.0), length))
            s += st.marker_step
    res = sorted(set(round(s, 9) for s in res))
    return [(s, j) for j, s in enumerate(res)]


class _Run:
    def __init__(self, rail, sources, st, base, seg_id_at, clip_intervals=None, id_breaks=()):
        self.rail = rail
        self.id_breaks = id_breaks
        self.sources = sources
        self.st = st
        self.base = base            # spline-level variables
        self.seg_id_at = seg_id_at  # function(s) -> segment id
        self.clip = clip_intervals
        self.state = {}
        self.counters = [0] * 6
        self.global_index = 0
        self.out = []
        self.truncated = False
        self.L = rail.length
        self.section = 0

    # -- requests ------------------------------------------------------------

    def values(self, inp, index, pos, extra=None):
        rail = self.rail
        p = rail.position(pos)[0]
        v = dict(self.base)
        v.update(input=inp, index=index, global_index=self.global_index, distance=pos,
                 distance_pct=pos / self.L if self.L > 0 else 0.0, x=p[0], y=p[1], z=p[2],
                 slope=math.degrees(float(rail.slope(pos)[0])), segment_id=self.seg_id_at(pos),
                 section=self.section)
        v["random"] = hash01(self.st.seed, int(v.get("spline", 0)), inp, index, "random")
        if extra:
            v.update(extra)
        return v

    def request(self, inp, pos, extra=None, index=None):
        src = self.sources.get(inp)
        if src is None:
            return None
        if index is None:
            index = self.counters[inp]
            self.counters[inp] += 1
        ctx = Ctx(self.values(inp, index, pos, extra), self.state, self.st.seed)
        self.global_index += 1
        return src.get(ctx)

    def emit(self, placed):
        if len(self.out) >= self.st.max_segments:
            self.truncated = True
            return False
        self.out.append(placed)
        return True

    # -- anchors -----------------------------------------------------------------

    @staticmethod
    def extents(seg, inp):
        """Distance a segment occupies before / after its anchor point."""
        mode = seg.align[0]
        if mode == "AUTO":
            mode = "PIVOT" if inp == INPUT_CORNER else "CENTER"
        lo, hi = float(seg.bmin[0]), float(seg.bmax[0])
        if seg.size:
            if mode == "CENTER":
                before = after = seg.size * 0.5
            elif mode == "MIN":
                before, after = 0.0, seg.size
            elif mode == "MAX":
                before, after = seg.size, 0.0
            else:
                before, after = -lo, seg.size + lo
        elif mode == "CENTER":
            before = after = (hi - lo) * 0.5
        elif mode == "MIN":
            before, after = 0.0, hi - lo
        elif mode == "MAX":
            before, after = hi - lo, 0.0
        else:
            before, after = -lo, hi
        return max(before + seg.pad[0], 0.0), max(after + seg.pad[1], 0.0)

    def anchor(self, inp, pos, extra):
        seg = self.request(inp, pos, extra)
        if seg is None:
            return None
        before, after = self.extents(seg, inp)
        return Placed(seg, ANCHOR, pos - before, pos + after, anchor=pos, inp=inp,
                      index=self.counters[inp] - 1)

    # -- main --------------------------------------------------------------------

    def run(self):
        st, rail, L = self.st, self.rail, self.L
        if L <= EPS:
            return self.out
        cyclic = rail.cyclic
        if cyclic:
            a, b = 0.0, L
        else:
            cs, ce = st.clip_start, st.clip_end
            if st.clip_percent:
                cs, ce = L * cs / 100.0, L * ce / 100.0
            a, b = cs, L - ce
            if not st.extend:
                a, b = max(a, 0.0), min(b, L)
            if b - a <= EPS:
                return self.out
        cur_a, cur_b = a, b
        if not cyclic:
            seg = self.request(INPUT_START, a)
            if seg is not None:
                ln = seg.length()
                self.emit(Placed(seg, SPAN, a, a + ln, x0=a + seg.pad[0], k=1.0, inp=INPUT_START))
                cur_a = a + ln
            seg = self.request(INPUT_END, b)
            if seg is not None:
                ln = seg.length()
                self.emit(Placed(seg, SPAN, b - ln, b, x0=b - ln + seg.pad[0], k=1.0, inp=INPUT_END))
                cur_b = b - ln

        breaks = self.breaks(cur_a, cur_b)
        # Sections: free range (g0, g1) and the anchor positions (m0, m1) that Evenly
        # distances are measured from (spline ends and corner / marker points).
        sections = []
        if cyclic:
            if not breaks:
                sections.append((0.0, L, 0.0, L))
            for j, (pos, before, after) in enumerate(breaks):
                npos, nbefore, _ = breaks[(j + 1) % len(breaks)]
                wrap = L if j == len(breaks) - 1 else 0.0
                sections.append((pos + after, npos - nbefore + wrap, pos, npos + wrap))
        else:
            prev, mark = cur_a, a
            for pos, before, after in breaks:
                sections.append((prev, pos - before, mark, pos))
                prev, mark = pos + after, pos
            sections.append((prev, cur_b, mark, b))

        evenly_all = None
        if st.evenly_scope == "SPLINE" and self.sources.get(INPUT_EVENLY) is not None:
            evenly_all = self.evenly_positions(a, b) if not cyclic else self.evenly_positions(0.0, L)
        for g0, g1, m0, m1 in sections:
            self.fill_section(g0, g1, evenly_all, m0, m1)
            self.section += 1
        if self.clip is not None:
            self.apply_clip()
        self.out.sort(key=lambda p: p.s0)
        return self.out

    def breaks(self, cur_a, cur_b):
        """Corners, markers and ID changes as (position, before, after) sorted by position."""
        st, rail, L = self.st, self.rail, self.L
        cyclic = rail.cyclic
        items = []

        def inside(pos):
            return cyclic or (cur_a + EPS < pos < cur_b - EPS)

        markers = marker_positions(st, rail, L) if st.marker_mode != "NONE" else []
        marker_at = [s for s, _ in markers]
        if st.corner_mode != "NONE":
            n = rail.n_ctrl
            for k in range(n):
                if not cyclic and k in (0, n - 1):
                    continue
                ang = float(rail.vertex_angle[k]) if k < len(rail.vertex_angle) else 0.0
                if st.corner_mode == "SHARP" and ang <= st.corner_angle:
                    continue
                pos = float(rail.s_at(k)) % L if cyclic else float(rail.s_at(k))
                if not inside(pos) or any(abs(pos - m) < 1e-6 for m in marker_at):
                    continue
                pl = self.anchor(INPUT_CORNER, pos, {"corner_angle": math.degrees(ang), "vertex": k})
                if pl is not None:
                    self.emit(pl)
                    items.append((pos, pos - pl.s0, pl.s1 - pos))
                elif st.corner_split:
                    items.append((pos, 0.0, 0.0))
        for s, j in markers:
            pos = s % L if cyclic else s
            if not cyclic and not (cur_a - EPS <= pos <= cur_b + EPS):
                continue
            pl = self.anchor(INPUT_MARKER, pos, {"marker": j})
            if pl is not None:
                self.emit(pl)
                items.append((pos, pos - pl.s0, pl.s1 - pos))
        if st.split_ids:
            for pos in self.id_changes():
                if inside(pos) and not any(abs(pos - it[0]) < 1e-6 for it in items):
                    items.append((pos, 0.0, 0.0))
        items.sort(key=lambda it: it[0])
        return items

    def id_changes(self):
        return list(self.id_breaks)

    def evenly_positions(self, g0, g1):
        st = self.st
        G = g1 - g0
        if G <= EPS:
            return []
        if st.evenly_mode == "COUNT":
            n = max(0, int(st.evenly_count))
            return [g0 + G * (j + 1) / (n + 1) for j in range(n)]
        d = max(st.evenly_distance, 1e-3)
        out = []
        pos = g0 + math.fmod(st.evenly_offset, d)
        if pos < g0:
            pos += d
        while pos < g1 - EPS and len(out) <= st.max_segments:
            if pos > g0 + EPS:
                out.append(pos)
            pos += d
        return out

    def fill_section(self, g0, g1, evenly_all, m0=None, m1=None):
        evenly = self.sources.get(INPUT_EVENLY)
        spans = []
        if evenly is not None:
            if evenly_all is not None:
                period = self.L if self.rail.cyclic else 0.0
                candidates = []
                for p in evenly_all:
                    candidates.extend((p, p + period) if period else (p,))
            else:
                candidates = self.evenly_positions(g0 if m0 is None else m0, g1 if m1 is None else m1)
            positions = [q for q in candidates if g0 + EPS < q < g1 - EPS]
            for pos in positions:
                pl = self.anchor(INPUT_EVENLY, pos, None)
                if pl is None:
                    continue
                self.emit(pl)
                spans.append((pl.s0, pl.s1))
        prev = g0
        for b0, b1 in spans:
            self.fill(prev, b0)
            prev = b1
        self.fill(prev, g1)

    # -- default fill -----------------------------------------------------------------

    def _ask(self, index, pos, count, local):
        extra = {"section_index": local, "section_count": count,
                 "from_end": (count - 1 - local) if count > 0 else -1}
        return self.request(INPUT_DEFAULT, pos, extra, index=index)

    def fill(self, g0, g1):
        src = self.sources.get(INPUT_DEFAULT)
        G = g1 - g0
        if src is None or G <= 1e-5:
            return
        st = self.st
        sp = st.spacing
        base_index = self.counters[INPUT_DEFAULT]
        snapshot = dict(self.state)
        g_snapshot = self.global_index
        limit = max(1, st.max_segments - len(self.out))
        adaptive = st.adaptive

        # Pass 1: how many segments?
        lens = []
        fallback = None
        total = 0.0
        n = 0
        while n < limit:
            if adaptive and st.fit == "COUNT" and n >= max(1, st.count):
                break
            seg = self._ask(base_index + n, g0 + total, -1, n)
            ln = seg.length() if seg is not None else (fallback or 1.0)
            if seg is not None:
                fallback = ln
            new_total = total + ln + (sp if n else 0.0)
            if adaptive and st.fit == "ROUND" and new_total >= G:
                if n == 0 or (new_total - G) <= (G - total):
                    lens.append(ln)
                    n += 1
                break
            if (not adaptive or st.fit == "FLOOR") and new_total > G + 1e-9:
                if adaptive and n == 0:
                    lens.append(ln)
                    n += 1
                break
            lens.append(ln)
            total = new_total
            n += 1
            if adaptive and st.fit == "CEIL" and total >= G - 1e-9:
                break
        if n >= limit:
            self.truncated = True
        if n == 0:
            self.state.clear()
            self.state.update(snapshot)
            self.global_index = g_snapshot
            if not adaptive:
                self._remainder_only(g0, g1, base_index)
            return

        # Pass 2: ask again with the final positions.
        self.state.clear()
        self.state.update(snapshot)
        self.global_index = g_snapshot
        scale_guess = 1.0
        if adaptive and lens:
            fixed_total = sum(lens) + sp * (n - 1)
            scale_guess = G / fixed_total if fixed_total > 0 else 1.0
        segs = []
        pos = g0
        for j in range(n):
            seg = self._ask(base_index + j, pos, n, j)
            segs.append(seg)
            ln = seg.length() if seg is not None else lens[j]
            pos += ln * scale_guess + sp
        self.counters[INPUT_DEFAULT] = base_index + n
        lens = [s.length() if s is not None else l for s, l in zip(segs, lens)]

        if adaptive:
            self.place_adaptive(g0, g1, segs, lens)
        else:
            self.place_fixed(g0, g1, segs, lens, base_index + n)

    def place_adaptive(self, g0, g1, segs, lens):
        sp = self.st.spacing
        n = len(segs)
        G = g1 - g0
        stretch = [s is None or s.adaptive is not False for s in segs]
        fixed = sum(l for l, st_ in zip(lens, stretch) if not st_)
        flexible = sum(l for l, st_ in zip(lens, stretch) if st_)
        avail = G - sp * (n - 1) - fixed
        if flexible <= 0 or avail <= 1e-6:
            k = 1.0
        else:
            k = avail / flexible
        pos = g0
        for seg, ln, flex in zip(segs, lens, stretch):
            kk = k if flex else 1.0
            occ = ln * kk
            if seg is not None and not seg.empty:
                if not self.emit(Placed(seg, SPAN, pos, pos + occ, x0=pos + seg.pad[0] * kk, k=kk,
                                        inp=INPUT_DEFAULT)):
                    return
            pos += occ + sp

    def place_fixed(self, g0, g1, segs, lens, next_index):
        st = self.st
        sp = st.spacing
        G = g1 - g0
        n = len(segs)
        used = sum(lens) + sp * (n - 1)
        r = max(G - used, 0.0)
        align = st.align if not (st.align == "SPREAD" and n < 2) else "CENTER"
        gap = sp
        lead = 0.0
        remainder = st.remainder
        if align == "END":
            lead = r
        elif align == "CENTER":
            lead = r * 0.5
        elif align == "SPREAD":
            gap = sp + r / (n - 1)
            remainder = "EMPTY"
        pos = g0 + lead
        first = pos
        for seg, ln in zip(segs, lens):
            if seg is not None and not seg.empty:
                if not self.emit(Placed(seg, SPAN, pos, pos + ln, x0=pos + seg.pad[0], k=1.0, inp=INPUT_DEFAULT)):
                    return
            pos += ln + gap
        last = pos - gap
        if remainder == "EMPTY" or r <= 1e-5:
            return
        pieces = []
        if align == "START":
            pieces.append(("after", last + sp, g1))
        elif align == "END":
            pieces.append(("before", g0, first - sp))
        else:
            pieces.append(("before", g0, first - sp))
            pieces.append(("after", last + sp, g1))
        for side, p0, p1 in pieces:
            if p1 - p0 <= 1e-5:
                continue
            seg = self._ask(next_index, p0, n, n)
            self.counters[INPUT_DEFAULT] = next_index + 1
            next_index += 1
            if seg is None or seg.empty:
                continue
            self.place_piece(seg, side, p0, p1, remainder)

    def place_piece(self, seg, side, p0, p1, remainder):
        ln = seg.length()
        piece = p1 - p0
        if remainder == "SCALE":
            k = piece / ln
            self.emit(Placed(seg, SPAN, p0, p1, x0=p0 + seg.pad[0] * k, k=k, inp=INPUT_DEFAULT))
            return
        start = p0 if side == "after" else p1 - ln
        pl = Placed(seg, SPAN, p0, p1, x0=start + seg.pad[0], k=1.0, inp=INPUT_DEFAULT)
        pl.cut = (p0, p1)
        self.emit(pl)

    def _remainder_only(self, g0, g1, index):
        st = self.st
        if st.remainder == "EMPTY":
            return
        seg = self._ask(index, g0, 1, 0)
        self.counters[INPUT_DEFAULT] = index + 1
        if seg is None or seg.empty:
            return
        if st.remainder == "SCALE":
            self.place_piece(seg, "after", g0, g1, "SCALE")
            return
        ln = seg.length()
        if st.align == "END":
            start = g1 - ln
        elif st.align in ("CENTER", "SPREAD"):
            start = g0 - (ln - (g1 - g0)) * 0.5
        else:
            start = g0
        pl = Placed(seg, SPAN, g0, g1, x0=start + seg.pad[0], k=1.0, inp=INPUT_DEFAULT)
        pl.cut = (g0, g1)
        self.emit(pl)

    # -- clipping area -------------------------------------------------------------

    def apply_clip(self):
        keep = []
        for pl in self.out:
            if pl.kind == ANCHOR:
                if any(a - EPS <= pl.anchor <= b + EPS for a, b in self.clip):
                    keep.append(pl)
                continue
            lo, hi = (pl.cut if pl.cut else (pl.s0, pl.s1))
            for a, b in self.clip:
                c0, c1 = max(lo, a), min(hi, b)
                if c1 - c0 <= 1e-6:
                    continue
                if c0 <= lo + 1e-9 and c1 >= hi - 1e-9:
                    keep.append(pl)
                    break
                cp = Placed(pl.seg, SPAN, c0, c1, x0=pl.x0, k=pl.k, inp=pl.inp, index=pl.index)
                cp.cut = (c0, c1)
                keep.append(cp)
        self.out = keep


def layout(rail, sources, settings, base_values, seg_id_at=lambda s: 0, clip_intervals=None,
           id_breaks=()):
    """Lay out one rail. ``sources`` maps generator inputs to sources.

    ``id_breaks`` are rail positions where the Segment ID changes. Returns
    ``(placements, truncated)``.
    """
    run = _Run(rail, sources, settings, base_values, seg_id_at, clip_intervals, id_breaks)
    out = run.run()
    return out, run.truncated
