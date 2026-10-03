# SPDX-License-Identifier: GPL-3.0-or-later
"""Layout engine: decides which sample goes where along one path.

The layout works purely with arc-length positions; it knows nothing about
meshes or Blender. For every path it produces a list of :class:`Placement`
records that the geometry stage turns into deformed copies of the samples.

Order of operations (mirrors the classic "linear one spline" generator):

1. Start / End samples take the ends of open paths.
2. *Hard anchors* split the remaining range into sections: corners (always
   breaks, with or without a corner sample) and markers.
3. Evenly samples are distributed inside every section.
4. The remaining gaps are split where the segment ID changes and filled with
   the Default sample (or the sample of that segment ID), stretched to fit or
   at real size depending on the fit mode.
"""

import bisect
import math
import random
import zlib

SPAN = "SPAN"     # module occupies [s_begin, s_end] and may be stretched
POINT = "POINT"   # module anchored at a position (corners, evenly, markers)

EPS = 1e-6


def stable_seed(*parts):
    """Deterministic seed (does not depend on Python's hash randomisation)."""
    return zlib.crc32(repr(parts).encode("utf8")) & 0x7FFFFFFF


class VariantInfo:
    """Bounding box of one sample variant along local X (after slot transforms)."""

    __slots__ = ("minx", "maxx")

    def __init__(self, minx, maxx):
        self.minx = float(minx)
        self.maxx = float(maxx)

    @property
    def lenx(self):
        return max(self.maxx - self.minx, 0.0)


class SlotInfo:
    """What the layout needs to know about one sample slot."""

    def __init__(self, key, active=True, variants=(), pick="SINGLE", weights=None,
                 pad_before=0.0, pad_after=0.0, align_x="CENTER", seed=0):
        self.key = key
        self.active = bool(active)
        self.variants = list(variants)
        self.pick = pick
        self.weights = list(weights) if weights else [1.0] * len(self.variants)
        self.pad_before = float(pad_before)
        self.pad_after = float(pad_after)
        self.align_x = align_x
        self.seed = int(seed)

    @property
    def has_geometry(self):
        return bool(self.variants)

    def full_length(self, v):
        """Length along the path including paddings (never zero)."""
        return max(self.variants[v].lenx + self.pad_before + self.pad_after, 1e-4)

    def extents(self, v):
        """Occupied distance (before, after) around an anchor position."""
        pb, pa = self.pad_before, self.pad_after
        if v is None:
            return max(pb, 0.0), max(pa, 0.0)
        var = self.variants[v]
        if self.align_x == "PIVOT":
            before, after = -var.minx, var.maxx
        elif self.align_x == "START":
            before, after = 0.0, var.lenx
        elif self.align_x == "END":
            before, after = var.lenx, 0.0
        else:
            before = after = var.lenx * 0.5
        return max(before + pb, 0.0), max(after + pa, 0.0)


class PathInfo:
    """Arc-length description of one path for the layout.

    ``seg_bounds``: list of ``(s_start, s_end, seg_id)`` for every control
    segment in path order (``seg_id`` is an index into the segment-ID list or
    ``-1`` for Default).
    ``markers``: list of ``(s, slot_key)``.
    """

    def __init__(self, length, cyclic=False, corners=(), seg_bounds=(), markers=(), index=0):
        self.length = float(length)
        self.cyclic = bool(cyclic)
        self.corners = sorted(float(c) for c in corners)
        self.seg_bounds = sorted(seg_bounds, key=lambda b: b[0])
        self.markers = list(markers)
        self.index = int(index)


class LayoutSettings:
    def __init__(self, **kw):
        self.clip_start = 0.0
        self.clip_end = 0.0
        self.fit_mode = "ROUND"          # ROUND CEIL FLOOR COUNT FIXED
        self.fit_count = 1
        self.spacing = 0.0
        self.fixed_align = "START"       # START CENTER END DISTRIBUTE
        self.fixed_remainder = "NONE"    # NONE SCALE SLICE
        self.evenly_mode = "SPACING"     # SPACING FIT COUNT
        self.evenly_spacing = 2.0
        self.evenly_count = 1
        self.evenly_min_gap = 0.0
        self.evenly_align = "START"      # START CENTER END
        self.corner_slide = 0.0
        self.seed = 0
        self.max_modules = 20000
        for k, v in kw.items():
            if not hasattr(self, k):
                raise AttributeError(k)
            setattr(self, k, v)


class Placement:
    """One module along a path.

    SPAN modules: geometry X maps to ``x0 + (x - minx) * k`` (``minx`` of the
    module bounding box). POINT modules: geometry X maps to
    ``anchor + (x - ref_x)`` where ``ref_x`` depends on the slot's X alignment.
    ``slice_lo/slice_hi`` (fractions of the bounding box) cut the geometry for
    partial modules.
    """

    __slots__ = ("slot", "variant", "kind", "x0", "k", "anchor", "slice_lo", "slice_hi",
                 "ordinal", "s_begin", "s_end", "role")

    def __init__(self, slot, variant, kind, s_begin, s_end, x0=0.0, k=1.0, anchor=0.0,
                 slice_lo=None, slice_hi=None, ordinal=0, role=""):
        self.slot = slot
        self.variant = variant
        self.kind = kind
        self.x0 = x0
        self.k = k
        self.anchor = anchor
        self.slice_lo = slice_lo
        self.slice_hi = slice_hi
        self.ordinal = ordinal
        self.s_begin = s_begin
        self.s_end = s_end
        self.role = role

    @property
    def sliced(self):
        return self.slice_lo is not None or self.slice_hi is not None

    def __repr__(self):
        return ("Placement(%s v%d %s [%.4f, %.4f] k=%.4f)"
                % (self.slot, self.variant, self.kind, self.s_begin, self.s_end, self.k))


class _Anchor:
    __slots__ = ("pos", "before", "after", "slot", "variant", "role")

    def __init__(self, pos, before, after, slot=None, variant=None, role=""):
        self.pos = pos
        self.before = before
        self.after = after
        self.slot = slot
        self.variant = variant
        self.role = role


class _Picker:
    """Chooses variants (single, weighted random or sequence) with look-ahead."""

    def __init__(self, slot, seed, start=0):
        self.slot = slot
        self.n = len(slot.variants)
        self.mode = slot.pick if self.n > 1 else "SINGLE"
        self.rng = random.Random(seed)
        self.start = start
        self.buf = []

    def peek(self, j):
        while len(self.buf) <= j:
            self.buf.append(self._gen(len(self.buf)))
        return self.buf[j]

    def _gen(self, j):
        if self.mode == "SINGLE":
            return 0
        if self.mode == "SEQUENCE":
            return (self.start + j) % self.n
        w = [max(0.0, x) for x in self.slot.weights[:self.n]]
        w += [1.0] * (self.n - len(w))
        total = sum(w)
        if total <= 0.0:
            return self.rng.randrange(self.n)
        r = self.rng.random() * total
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if r < acc:
                return i
        return self.n - 1


class LayoutResult:
    def __init__(self):
        self.placements = []
        self.truncated = False


class _Layout:
    def __init__(self, path, slots, st):
        self.path = path
        self.slots = slots
        self.st = st
        self.res = LayoutResult()
        self.seq_state = {}
        self.ordinals = {}
        self.gap_counter = 0
        L = path.length
        self.L = L
        # Segment-ID lookup tables.
        self.bound_starts = [b[0] for b in path.seg_bounds]
        self.id_changes = []
        sb = path.seg_bounds
        for j in range(1, len(sb)):
            if sb[j][2] != sb[j - 1][2]:
                self.id_changes.append(sb[j][0])
        if path.cyclic and len(sb) > 1 and sb[0][2] != sb[-1][2]:
            self.id_changes.append(sb[0][0])
        self.has_ids = any(b[2] >= 0 for b in sb)

    # -- helpers -------------------------------------------------------------

    def slot(self, key):
        s = self.slots.get(key)
        if s is None or not s.active:
            return None
        return s

    def next_ordinal(self, key):
        n = self.ordinals.get(key, 0)
        self.ordinals[key] = n + 1
        return n

    def emit(self, p):
        if len(self.res.placements) >= self.st.max_modules:
            self.res.truncated = True
            return False
        self.res.placements.append(p)
        return True

    def pick_point_variant(self, slot, ordinal):
        if slot.pick == "SEQUENCE" and len(slot.variants) > 1:
            return ordinal % len(slot.variants)
        picker = _Picker(slot, stable_seed(self.st.seed, slot.seed, self.path.index, slot.key, ordinal))
        return picker.peek(0)

    def id_at(self, s):
        if not self.has_ids:
            return -1
        if self.path.cyclic:
            s = s % self.L
        j = bisect.bisect_right(self.bound_starts, s + 1e-9) - 1
        j = min(max(j, 0), len(self.path.seg_bounds) - 1)
        return self.path.seg_bounds[j][2]

    # -- main ------------------------------------------------------------------

    def run(self):
        path, st = self.path, self.st
        L = self.L
        if L <= EPS:
            return self.res
        if path.cyclic:
            a, b = 0.0, L
        else:
            a, b = st.clip_start, L - st.clip_end
            if b - a <= EPS:
                return self.res

        cur_a, cur_b = a, b
        if not path.cyclic:
            start = self.slot("start")
            if start and start.has_geometry:
                v = self.pick_point_variant(start, 0)
                ln = start.full_length(v)
                self.emit(Placement("start", v, SPAN, a, a + ln, x0=a + start.pad_before, k=1.0,
                                    ordinal=self.next_ordinal("start"), role="start"))
                cur_a = a + ln
            end = self.slot("end")
            if end and end.has_geometry:
                v = self.pick_point_variant(end, 1)
                ln = end.full_length(v)
                self.emit(Placement("end", v, SPAN, b - ln, b, x0=b - ln + end.pad_before, k=1.0,
                                    ordinal=self.next_ordinal("end"), role="end"))
                cur_b = b - ln

        anchors = self.hard_anchors(cur_a, cur_b)
        for anc in anchors:
            self.emit_anchor(anc)

        # Sections between hard anchors.
        sections = []
        if path.cyclic:
            if not anchors:
                sections.append((0.0, L))
            else:
                for j, anc in enumerate(anchors):
                    nxt = anchors[(j + 1) % len(anchors)]
                    g0 = anc.pos + anc.after
                    g1 = nxt.pos - nxt.before
                    if j == len(anchors) - 1:
                        g1 += L
                    sections.append((g0, g1))
        else:
            prev_end = cur_a
            for anc in anchors:
                sections.append((prev_end, anc.pos - anc.before))
                prev_end = anc.pos + anc.after
            sections.append((prev_end, cur_b))

        for g0, g1 in sections:
            self.fill_section(g0, g1)
        self.res.placements.sort(key=lambda p: p.s_begin)
        return self.res

    def hard_anchors(self, cur_a, cur_b):
        path, st = self.path, self.st
        L = self.L
        anchors = []
        corner = self.slots.get("corner")
        for c in path.corners:
            pos = c + st.corner_slide
            if path.cyclic:
                pos %= L
            elif pos <= cur_a + EPS or pos >= cur_b - EPS:
                continue
            if corner is not None and corner.active:
                if corner.has_geometry:
                    v = self.pick_point_variant(corner, len(anchors))
                    before, after = corner.extents(v)
                    anchors.append(_Anchor(pos, before, after, corner, v, "corner"))
                else:
                    before, after = corner.extents(None)
                    anchors.append(_Anchor(pos, before, after, None, None, "corner"))
            else:
                anchors.append(_Anchor(pos, 0.0, 0.0, None, None, "corner"))
        for s, key in path.markers:
            slot = self.slots.get(key)
            if slot is None or not slot.active:
                continue
            pos = s % L if path.cyclic else s
            if not path.cyclic and (pos < cur_a - EPS or pos > cur_b + EPS):
                continue
            if slot.has_geometry:
                v = self.pick_point_variant(slot, self.ordinals.get(key, 0) + len(anchors))
                before, after = slot.extents(v)
                anchors.append(_Anchor(pos, before, after, slot, v, "marker"))
            else:
                before, after = slot.extents(None)
                anchors.append(_Anchor(pos, before, after, None, None, "marker"))
        marker_pos = [an.pos for an in anchors if an.role == "marker"]
        if marker_pos:
            anchors = [an for an in anchors if an.role != "corner"
                       or all(abs(an.pos - m) > 1e-6 for m in marker_pos)]
        anchors.sort(key=lambda an: an.pos)
        return anchors

    def emit_anchor(self, anc):
        if anc.slot is None or anc.variant is None:
            return
        self.emit(Placement(anc.slot.key, anc.variant, POINT, anc.pos - anc.before,
                            anc.pos + anc.after, anchor=anc.pos,
                            ordinal=self.next_ordinal(anc.slot.key), role=anc.role))

    def evenly_positions(self, g0, g1):
        st = self.st
        G = g1 - g0
        if G <= EPS:
            return []
        mode = st.evenly_mode
        if mode == "COUNT":
            n = max(0, int(st.evenly_count))
            return [g0 + G * (j + 1) / (n + 1) for j in range(n)]
        sp = max(st.evenly_spacing, 1e-3)
        if mode == "FIT":
            n_div = max(1, int(math.ceil(G / sp - 1e-9)))
            return [g0 + G * j / n_div for j in range(1, n_div)]
        gap = max(st.evenly_min_gap, 0.0)
        if st.evenly_align == "CENTER":
            usable = G - 2.0 * gap
            if usable < -EPS:
                return []
            n = int(math.floor(max(usable, 0.0) / sp + 1e-9)) + 1
            mid = (g0 + g1) * 0.5
            return [mid + (j - (n - 1) * 0.5) * sp for j in range(n)]
        out = []
        j = 1
        while True:
            off = j * sp
            if off > G - gap + 1e-9 or off >= G - EPS:
                break
            out.append(g0 + off if st.evenly_align != "END" else g1 - off)
            j += 1
            if len(out) > st.max_modules:
                break
        out.sort()
        return out

    def fill_section(self, g0, g1):
        evenly = self.slot("evenly")
        bounds = []  # (begin, end) of evenly anchors
        if evenly is not None and evenly.has_geometry:
            for pos in self.evenly_positions(g0, g1):
                ordinal = self.ordinals.get("evenly", 0)
                v = self.pick_point_variant(evenly, ordinal)
                before, after = evenly.extents(v)
                p = Placement("evenly", v, POINT, pos - before, pos + after, anchor=pos,
                              ordinal=self.next_ordinal("evenly"), role="evenly")
                self.emit(p)
                bounds.append((pos - before, pos + after))
        prev = g0
        for b0, b1 in bounds:
            self.fill_ids(prev, b0)
            prev = b1
        self.fill_ids(prev, g1)

    def fill_ids(self, g0, g1):
        if g1 - g0 <= EPS:
            return
        if not self.has_ids:
            self.fill_gap(g0, g1, self.slot("default"))
            return
        cuts = []
        shifts = (0.0, self.L, -self.L, 2 * self.L) if self.path.cyclic else (0.0,)
        for c in self.id_changes:
            for sh in shifts:
                x = c + sh
                if g0 + EPS < x < g1 - EPS:
                    cuts.append(x)
        cuts.sort()
        edges = [g0] + cuts + [g1]
        for p0, p1 in zip(edges, edges[1:]):
            sid = self.id_at((p0 + p1) * 0.5)
            seg = self.slots.get("seg:%d" % sid) if sid >= 0 else None
            # Enabled IDs without geometry leave a hole; disabled IDs use the Default sample.
            slot = seg if seg is not None and seg.active else self.slot("default")
            self.fill_gap(p0, p1, slot)

    def fill_gap(self, g0, g1, slot):
        G = g1 - g0
        self.gap_counter += 1
        if G <= 1e-5 or slot is None or not slot.has_geometry:
            return
        st = self.st
        key = slot.key
        picker = _Picker(slot, stable_seed(st.seed, slot.seed, self.path.index, key, self.gap_counter),
                         self.seq_state.get(key, 0))
        min_len = min(slot.full_length(v) for v in range(len(slot.variants)))
        sp = max(st.spacing, -0.9 * min_len)
        limit = max(1, st.max_modules - len(self.res.placements))
        mode = st.fit_mode
        if mode == "FIXED":
            used = self.fill_fixed(g0, g1, slot, picker, sp, limit)
        else:
            used = self.fill_adaptive(g0, g1, slot, picker, sp, limit, mode)
        self.seq_state[key] = self.seq_state.get(key, 0) + used

    def _span(self, slot, v, s0, s1, k):
        return Placement(slot.key, v, SPAN, s0, s1, x0=s0 + slot.pad_before * k, k=k,
                         ordinal=self.next_ordinal(slot.key), role="fill")

    def fill_adaptive(self, g0, g1, slot, picker, sp, limit, mode):
        G = g1 - g0
        chosen = []
        if mode == "COUNT":
            chosen = [picker.peek(j) for j in range(max(1, int(self.st.fit_count)))]
        else:
            total = 0.0
            j = 0
            while len(chosen) < limit:
                v = picker.peek(j)
                ln = slot.full_length(v)
                new_total = total + ln + (sp if chosen else 0.0)
                if mode == "CEIL":
                    chosen.append(v)
                    total = new_total
                    j += 1
                    if total >= G - 1e-9:
                        break
                elif mode == "FLOOR":
                    if chosen and new_total > G + 1e-9:
                        break
                    chosen.append(v)
                    total = new_total
                    j += 1
                else:  # ROUND
                    if new_total >= G:
                        if not chosen or (new_total - G) <= (G - total):
                            chosen.append(v)
                            j += 1
                        break
                    chosen.append(v)
                    total = new_total
                    j += 1
            if len(chosen) >= limit:
                self.res.truncated = True
        n = len(chosen)
        if n == 0:
            return 0
        lens = [slot.full_length(v) for v in chosen]
        avail = G - (n - 1) * sp
        if avail <= 1e-6:
            return n
        k = avail / sum(lens)
        cur = g0
        for v, ln in zip(chosen, lens):
            seg = ln * k
            if not self.emit(self._span(slot, v, cur, cur + seg, k)):
                break
            cur += seg + sp
        return n

    def fill_fixed(self, g0, g1, slot, picker, sp, limit):
        st = self.st
        G = g1 - g0
        chosen = []
        total = 0.0
        j = 0
        while len(chosen) < limit:
            v = picker.peek(j)
            new_total = total + slot.full_length(v) + (sp if chosen else 0.0)
            if new_total > G + 1e-9:
                break
            chosen.append(v)
            total = new_total
            j += 1
        if len(chosen) >= limit:
            self.res.truncated = True
        n = len(chosen)
        r = max(G - total, 0.0)
        align = st.fixed_align
        rem = st.fixed_remainder
        if align == "DISTRIBUTE" and n < 2:
            align = "CENTER"
        gap = sp
        lead = 0.0
        if align == "END":
            lead = r
        elif align == "CENTER":
            lead = r * 0.5
        elif align == "DISTRIBUTE":
            gap = sp + r / (n - 1)
            rem = "NONE"
        used = n

        if n == 0:
            if rem == "NONE":
                return 0
            v = picker.peek(0)
            full = slot.full_length(v)
            if rem == "SCALE":
                k = G / full
                self.emit(self._span(slot, v, g0, g1, k))
            else:
                if align == "END":
                    self.emit_slice(slot, v, g1 - full, g0, g1)
                elif align == "CENTER":
                    off = (full - G) * 0.5
                    self.emit_slice(slot, v, g0 - off, g0, g1)
                else:
                    self.emit_slice(slot, v, g0, g0, g1)
            return 1

        cur = g0 + lead
        first_begin = cur
        for v in chosen:
            full = slot.full_length(v)
            if not self.emit(self._span(slot, v, cur, cur + full, 1.0)):
                return used
            cur += full + gap
        last_end = cur - gap

        if rem == "NONE" or r <= 1e-5:
            return used
        pieces = []
        if align == "START":
            pieces.append(("after", last_end + sp, g1))
        elif align == "END":
            pieces.append(("before", g0, first_begin - sp))
        else:
            pieces.append(("before", g0, first_begin - sp))
            pieces.append(("after", last_end + sp, g1))
        for side, p0, p1 in pieces:
            piece = p1 - p0
            if piece <= 1e-5:
                continue
            v = picker.peek(used)
            used += 1
            full = slot.full_length(v)
            if rem == "SCALE":
                self.emit(self._span(slot, v, p0, p1, piece / full))
            elif side == "after":
                self.emit_slice(slot, v, p0, p0, p1)
            else:
                self.emit_slice(slot, v, p1 - full, p0, p1)
        return used

    def emit_slice(self, slot, v, module_begin, keep0, keep1):
        """Full-size module starting at ``module_begin``, cut to [keep0, keep1]."""
        var = slot.variants[v]
        lenx = max(var.lenx, 1e-9)
        x0 = module_begin + slot.pad_before
        lo = (keep0 - x0) / lenx
        hi = (keep1 - x0) / lenx
        lo = max(lo, 0.0)
        hi = min(hi, 1.0)
        if hi - lo <= 1e-6:
            return
        p = Placement(slot.key, v, SPAN, keep0, keep1, x0=x0, k=1.0,
                      slice_lo=lo if lo > 1e-6 else None,
                      slice_hi=hi if hi < 1.0 - 1e-6 else None,
                      ordinal=self.next_ordinal(slot.key), role="slice")
        self.emit(p)


def layout_path(path, slots, settings):
    """Compute placements for one path. ``slots`` maps slot keys to :class:`SlotInfo`."""
    return _Layout(path, slots, settings).run()
