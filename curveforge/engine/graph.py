# SPDX-License-Identifier: GPL-3.0-or-later
"""Segments, segment sources (operators) and numeric values.

The node graph is compiled into these objects. A generator asks a source for
a segment every time it needs one, passing a :class:`Ctx` with the variables
of that request (index, distance, spline, corner angle...). Operators wrap
other sources: they choose between them (Sequence, Randomize, Conditional,
Selector), combine them (Compose) or modify the segment they return (Mirror,
Transform, Material, UV Transform).
"""

import math

import numpy as np

from .expr import compile_expression

INPUT_DEFAULT, INPUT_START, INPUT_END, INPUT_CORNER, INPUT_EVENLY, INPUT_MARKER = range(6)
INPUT_NAMES = ("Default", "Start", "End", "Corner", "Evenly", "Marker")

VARIABLES = (
    ("index", "Index of the segment among those of the same input on the spline"),
    ("global_index", "Index of the request among all inputs of the spline"),
    ("input", "Generator input: 0 Default, 1 Start, 2 End, 3 Corner, 4 Evenly, 5 Marker"),
    ("section", "Section number (sections are split by corners, markers and Segment ID changes)"),
    ("section_index", "Index of the Default segment inside its section"),
    ("section_count", "Number of Default segments in the section"),
    ("from_end", "Default segments left until the end of the section"),
    ("distance", "Distance from the start of the spline"),
    ("distance_pct", "Distance from the start as a fraction of the spline length (0 to 1)"),
    ("spline_length", "Length of the spline"),
    ("section_length", "Length of the section"),
    ("spline", "Spline index (all splines of all Spline nodes, in order)"),
    ("spline_material", "Material index of the spline"),
    ("segment_id", "Segment ID of the curve under the segment (set in Edit Mode)"),
    ("marker", "Marker number (Marker input)"),
    ("corner_angle", "Turning angle of the corner in degrees (Corner input)"),
    ("slope", "Slope of the curve in degrees (positive uphill)"),
    ("x", "World X of the segment start"),
    ("y", "World Y of the segment start"),
    ("z", "World Z of the segment start"),
    ("random", "Random value between 0 and 1, different for every segment"),
)
VARIABLE_NAMES = tuple(v[0] for v in VARIABLES)

_MASK = (1 << 64) - 1


def hash01(*parts):
    """Deterministic pseudo random value in [0, 1) from integers/strings."""
    h = 0x9E3779B97F4A7C15
    for p in parts:
        if isinstance(p, str):
            v = 0
            for ch in p.encode("utf8"):
                v = (v * 131 + ch) & _MASK
        elif isinstance(p, float):
            v = int(p * 1000003.0) & _MASK
        else:
            v = int(p) & _MASK
        h = (h ^ v) & _MASK
        h = (h * 0xBF58476D1CE4E5B9) & _MASK
        h ^= h >> 31
        h = (h * 0x94D049BB133111EB) & _MASK
        h ^= h >> 29
    return (h >> 11) / float(1 << 53)


class Ctx:
    """Variables of one segment request plus state shared by the operators."""

    __slots__ = ("values", "state", "seed")

    def __init__(self, values, state=None, seed=0):
        self.values = values
        self.state = {} if state is None else state
        self.seed = seed

    def __getitem__(self, name):
        return self.values.get(name, 0.0)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

class Num:
    def __call__(self, ctx):
        return 0.0

    constant = False


class Const(Num):
    constant = True

    def __init__(self, value):
        self.value = float(value)

    def __call__(self, ctx):
        return self.value


class Var(Num):
    def __init__(self, name):
        self.name = name

    def __call__(self, ctx):
        return float(ctx.values.get(self.name, 0.0))


MATH_OPS = {
    "ADD": lambda a, b, c: a + b,
    "SUBTRACT": lambda a, b, c: a - b,
    "MULTIPLY": lambda a, b, c: a * b,
    "DIVIDE": lambda a, b, c: a / b if b else 0.0,
    "POWER": lambda a, b, c: math.pow(a, b) if a >= 0 or float(b).is_integer() else 0.0,
    "MINIMUM": lambda a, b, c: min(a, b),
    "MAXIMUM": lambda a, b, c: max(a, b),
    "MODULO": lambda a, b, c: a - b * math.floor(a / b) if b else 0.0,
    "SNAP": lambda a, b, c: round(a / b) * b if b else a,
    "ABSOLUTE": lambda a, b, c: abs(a),
    "NEGATE": lambda a, b, c: -a,
    "FLOOR": lambda a, b, c: float(math.floor(a)),
    "CEIL": lambda a, b, c: float(math.ceil(a)),
    "ROUND": lambda a, b, c: float(round(a)),
    "SINE": lambda a, b, c: math.sin(a),
    "COSINE": lambda a, b, c: math.cos(a),
    "CLAMP": lambda a, b, c: max(b, min(c, a)),
    "MULTIPLY_ADD": lambda a, b, c: a * b + c,
    "LERP": lambda a, b, c: a + (b - a) * c,
    "EQUAL": lambda a, b, c: 1.0 if abs(a - b) <= max(c, 1e-9) else 0.0,
    "NOT_EQUAL": lambda a, b, c: 0.0 if abs(a - b) <= max(c, 1e-9) else 1.0,
    "LESS": lambda a, b, c: 1.0 if a < b else 0.0,
    "LESS_EQUAL": lambda a, b, c: 1.0 if a <= b else 0.0,
    "GREATER": lambda a, b, c: 1.0 if a > b else 0.0,
    "GREATER_EQUAL": lambda a, b, c: 1.0 if a >= b else 0.0,
    "BETWEEN": lambda a, b, c: 1.0 if b <= a <= c else 0.0,
    "AND": lambda a, b, c: 1.0 if (a > 0.5 and b > 0.5) else 0.0,
    "OR": lambda a, b, c: 1.0 if (a > 0.5 or b > 0.5) else 0.0,
    "NOT": lambda a, b, c: 0.0 if a > 0.5 else 1.0,
    "EVEN": lambda a, b, c: 1.0 if int(round(a)) % 2 == 0 else 0.0,
    "ODD": lambda a, b, c: 1.0 if int(round(a)) % 2 == 1 else 0.0,
    "EVERY": lambda a, b, c: 1.0 if b >= 1 and (int(round(a)) - int(round(c))) % int(round(b)) == 0 else 0.0,
}


class MathNum(Num):
    def __init__(self, op, a, b=None, c=None):
        self.fn = MATH_OPS[op]
        self.a = a
        self.b = b or Const(0.0)
        self.c = c or Const(0.0)

    def __call__(self, ctx):
        try:
            return float(self.fn(self.a(ctx), self.b(ctx), self.c(ctx)))
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0


class RandomNum(Num):
    """Random number; ``per`` = SEGMENT (every request), SPLINE or INDEX (same index, same value)."""

    def __init__(self, lo, hi, seed=0, per="SEGMENT", integer=False, key=""):
        self.lo, self.hi = lo, hi
        self.seed = int(seed)
        self.per = per
        self.integer = integer
        self.key = key

    def __call__(self, ctx):
        v = ctx.values
        if self.per == "SPLINE":
            r = hash01(ctx.seed, self.seed, self.key, int(v.get("spline", 0)))
        elif self.per == "INDEX":
            r = hash01(ctx.seed, self.seed, self.key, int(v.get("index", 0)))
        else:
            r = hash01(ctx.seed, self.seed, self.key, int(v.get("spline", 0)), int(v.get("input", 0)),
                       int(v.get("index", 0)), int(v.get("global_index", 0)))
        lo, hi = self.lo(ctx), self.hi(ctx)
        if self.integer:
            a, b = int(math.floor(min(lo, hi))), int(math.floor(max(lo, hi)))
            return float(a + min(int(r * (b - a + 1)), b - a))
        return lo + (hi - lo) * r


class ExprNum(Num):
    INPUTS = ("a", "b", "c", "d")

    def __init__(self, text, inputs):
        self.fn = compile_expression(text, VARIABLE_NAMES + self.INPUTS)
        self.inputs = inputs

    def __call__(self, ctx):
        values = dict(ctx.values)
        for name, num in self.inputs.items():
            values[name] = num(ctx)
        return self.fn(values)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

class Geo:
    """Geometry of one source object (local space) and its bounding box."""

    __slots__ = ("key", "mesh", "bmin", "bmax")

    def __init__(self, key, mesh):
        self.key = key
        self.mesh = mesh
        self.bmin, self.bmax = mesh.bbox()


def _box_corners(bmin, bmax):
    x = (bmin[0], bmax[0])
    y = (bmin[1], bmax[1])
    z = (bmin[2], bmax[2])
    return np.array([(a, b, c, 1.0) for a in x for b in y for c in z])


def euler_matrix(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    m = np.eye(4)
    m[:3, :3] = (np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
                 @ np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
                 @ np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx))))
    return m


def translation(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scaling(x, y, z):
    return np.diag((x, y, z, 1.0))


class Seg:
    """A concrete segment: geometry parts in segment space plus placement options.

    Tri-state options (``bend``, ``upright``, ``slice``, ``adaptive``,
    ``instance``) use ``None`` for "generator default".
    """

    __slots__ = ("parts", "bmin", "bmax", "size", "pad", "bend", "upright", "slice", "adaptive",
                 "instance", "align", "offset", "orient", "materials", "uv", "name")

    def __init__(self, parts=(), size=None, name=""):
        self.parts = list(parts)
        self.size = size
        self.pad = (0.0, 0.0)
        self.bend = None
        self.upright = None
        self.slice = None
        self.adaptive = None
        self.instance = None
        self.align = ("AUTO", "PIVOT", "PIVOT")
        self.offset = (0.0, 0.0, 0.0)
        self.orient = "BISECTOR"
        self.materials = ()
        self.uv = None
        self.name = name
        self.update_bounds()

    def update_bounds(self):
        if not self.parts:
            ln = self.size or 0.0
            self.bmin = np.array((0.0, 0.0, 0.0))
            self.bmax = np.array((ln, 0.0, 0.0))
            return
        pts = []
        for geo, m in self.parts:
            pts.append((_box_corners(geo.bmin, geo.bmax) @ m.T)[:, :3])
        pts = np.concatenate(pts)
        self.bmin = pts.min(axis=0)
        self.bmax = pts.max(axis=0)

    def copy(self):
        s = Seg.__new__(Seg)
        for name in Seg.__slots__:
            setattr(s, name, getattr(self, name))
        s.parts = list(self.parts)
        return s

    def length(self):
        """Distance the segment occupies along the rail."""
        core = self.size if self.size else float(self.bmax[0] - self.bmin[0])
        return max(core + self.pad[0] + self.pad[1], 1e-4)

    def transformed(self, m):
        out = self.copy()
        out.parts = [(geo, m @ pm) for geo, pm in self.parts]
        out.update_bounds()
        return out

    @property
    def empty(self):
        return not self.parts


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class Source:
    def get(self, ctx):
        return None


class Fixed(Source):
    """A Segment node: always the same segment."""

    def __init__(self, seg):
        self.seg = seg

    def get(self, ctx):
        return self.seg


def _first_flags(dst, src):
    for name in ("bend", "upright", "slice", "adaptive", "instance", "align", "offset", "orient", "pad"):
        setattr(dst, name, getattr(src, name))


class Compose(Source):
    """Joins the segments of its inputs into one segment."""

    def __init__(self, inputs, mode="SEQUENCE", gap=0.0, overrides=None):
        self.inputs = inputs
        self.mode = mode
        self.gap = gap
        self.overrides = overrides or {}
        self._memo = None

    def get(self, ctx):
        segs = [src.get(ctx) for src in self.inputs if src is not None]
        segs = [s for s in segs if s is not None]
        if not segs:
            return None
        ident = tuple(id(s) for s in segs)
        if self._memo is not None and self._memo[0] == ident:
            return self._memo[1]
        parts = []
        cursor = 0.0
        for s in segs:
            if self.mode == "OVERLAP":
                parts.extend(s.parts)
                continue
            shift = cursor + s.pad[0] - float(s.bmin[0])
            m = translation(shift, 0.0, 0.0)
            parts.extend((geo, m @ pm) for geo, pm in s.parts)
            cursor += s.length() + self.gap
        out = Seg(parts, name="compose")
        if self.mode != "OVERLAP":
            out.size = max(cursor - self.gap, 1e-4)
            out.bmin = out.bmin.copy()
            out.bmax = out.bmax.copy()
            out.bmin[0] = min(out.bmin[0], 0.0)
        _first_flags(out, segs[0])
        out.pad = (0.0, 0.0)
        for k, v in self.overrides.items():
            setattr(out, k, v)
        self._memo = (ident, out)
        return out


class Sequence(Source):
    """Uses its inputs one after the other, each ``count`` times."""

    def __init__(self, inputs, counts, scope="SPLINE", offset=0, key="seq"):
        self.inputs = inputs
        self.counts = [max(0, int(c)) for c in counts]
        self.total = sum(self.counts)
        self.scope = scope
        self.offset = int(offset)
        self.key = key

    def _scope_key(self, ctx):
        v = ctx.values
        if self.scope == "SECTION":
            return (self.key, int(v.get("spline", 0)), int(v.get("section", 0)))
        if self.scope == "INPUT":
            return (self.key, int(v.get("spline", 0)), int(v.get("input", 0)))
        if self.scope == "STYLE":
            return (self.key,)
        return (self.key, int(v.get("spline", 0)))

    def get(self, ctx):
        if self.total <= 0:
            return None
        k = self._scope_key(ctx)
        n = ctx.state.get(k, 0)
        ctx.state[k] = n + 1
        pos = (n + self.offset) % self.total
        for src, c in zip(self.inputs, self.counts):
            if pos < c:
                return src.get(ctx) if src is not None else None
            pos -= c
        return None


class Randomize(Source):
    """Picks one input at random (weighted)."""

    def __init__(self, inputs, weights, seed=0, per="SEGMENT", key="rnd"):
        self.inputs = inputs
        self.weights = [max(0.0, float(w)) for w in weights]
        self.total = sum(self.weights)
        self.seed = int(seed)
        self.per = per
        self.key = key

    def get(self, ctx):
        if not self.inputs or self.total <= 0.0:
            return None
        v = ctx.values
        if self.per == "SPLINE":
            r = hash01(ctx.seed, self.seed, self.key, int(v.get("spline", 0)))
        else:
            r = hash01(ctx.seed, self.seed, self.key, int(v.get("spline", 0)), int(v.get("input", 0)),
                       int(v.get("index", 0)), int(v.get("global_index", 0)))
        r *= self.total
        for src, w in zip(self.inputs, self.weights):
            if r < w:
                return src.get(ctx) if src is not None else None
            r -= w
        last = self.inputs[-1]
        return last.get(ctx) if last is not None else None


class Conditional(Source):
    """Input A when the condition is true (> 0.5), otherwise input B."""

    def __init__(self, a, b, condition):
        self.a = a
        self.b = b
        self.condition = condition

    def get(self, ctx):
        src = self.a if self.condition(ctx) > 0.5 else self.b
        return src.get(ctx) if src is not None else None


class Selector(Source):
    """Input number ``index`` (wrap around, clamp or nothing when out of range)."""

    def __init__(self, inputs, index, mode="WRAP"):
        self.inputs = inputs
        self.index = index
        self.mode = mode

    def get(self, ctx):
        n = len(self.inputs)
        if n == 0:
            return None
        i = int(math.floor(self.index(ctx) + 1e-9))
        if self.mode == "WRAP":
            i %= n
        elif self.mode == "CLAMP":
            i = min(max(i, 0), n - 1)
        elif not 0 <= i < n:
            return None
        src = self.inputs[i]
        return src.get(ctx) if src is not None else None


class Mirror(Source):
    """Mirrors segments; ALTERNATE mirrors every other one, RANDOM uses a probability."""

    def __init__(self, src, axes=(True, False, False), mode="ALWAYS", probability=0.5, seed=0, key="mir"):
        self.src = src
        self.axes = axes
        self.mode = mode
        self.probability = probability
        self.seed = seed
        self.key = key
        sx, sy, sz = (-1.0 if a else 1.0 for a in axes)
        self.matrix = scaling(sx, sy, sz)

    def get(self, ctx):
        seg = self.src.get(ctx) if self.src is not None else None
        if seg is None or not any(self.axes):
            return seg
        v = ctx.values
        if self.mode == "ALTERNATE" and int(v.get("index", 0)) % 2 == 0:
            return seg
        if self.mode == "RANDOM":
            r = hash01(ctx.seed, self.seed, self.key, int(v.get("spline", 0)), int(v.get("input", 0)),
                       int(v.get("index", 0)))
            if r >= self.probability:
                return seg
        return seg.transformed(self.matrix)


class Transform(Source):
    """Offset / rotation / scale (each optionally randomised) applied to the segment."""

    def __init__(self, src, offset, rotation, scale, rand_offset=(0.0, 0.0, 0.0),
                 rand_rotation=(0.0, 0.0, 0.0), rand_scale=(0.0, 0.0, 0.0), uniform=True,
                 rotation_step=0.0, seed=0, key="xf"):
        self.src = src
        self.offset = offset
        self.rotation = rotation
        self.scale = scale
        self.rand_offset = rand_offset
        self.rand_rotation = rand_rotation
        self.rand_scale = rand_scale
        self.uniform = uniform
        self.rotation_step = rotation_step
        self.seed = seed
        self.key = key
        self.static = None
        if all(n.constant for n in (*offset, *rotation, *scale)) and not any(rand_offset) \
                and not any(rand_rotation) and not any(rand_scale):
            self.static = self._matrix(None)

    def _matrix(self, ctx):
        def val(n):
            return n(ctx) if ctx is not None else n.value
        o = [val(n) for n in self.offset]
        r = [val(n) for n in self.rotation]
        s = [val(n) for n in self.scale]
        if ctx is not None:
            v = ctx.values
            base = (ctx.seed, self.seed, self.key, int(v.get("spline", 0)), int(v.get("input", 0)),
                    int(v.get("index", 0)), int(v.get("global_index", 0)))
            for i in range(3):
                if self.rand_offset[i]:
                    o[i] += (hash01(*base, "o", i) * 2.0 - 1.0) * self.rand_offset[i]
                if self.rand_rotation[i]:
                    a = (hash01(*base, "r", i) * 2.0 - 1.0) * self.rand_rotation[i]
                    if self.rotation_step > 1e-6:
                        a = round(a / self.rotation_step) * self.rotation_step
                    r[i] += a
            if self.uniform:
                if self.rand_scale[0]:
                    f = 1.0 + (hash01(*base, "s") * 2.0 - 1.0) * self.rand_scale[0]
                    s = [c * f for c in s]
            else:
                for i in range(3):
                    if self.rand_scale[i]:
                        s[i] *= 1.0 + (hash01(*base, "s", i) * 2.0 - 1.0) * self.rand_scale[i]
        return translation(*o) @ euler_matrix(*r) @ scaling(*s)

    def get(self, ctx):
        seg = self.src.get(ctx) if self.src is not None else None
        if seg is None:
            return None
        m = self.static if self.static is not None else self._matrix(ctx)
        return seg.transformed(m)


class MaterialOp(Source):
    """Material overrides: ALL faces, one SLOT, or pick from a list (RANDOM / SEQUENCE / INDEX)."""

    def __init__(self, src, mode, materials, slot=0, index=None, seed=0, key="mat"):
        self.src = src
        self.mode = mode
        self.materials = [m for m in materials]
        self.slot = int(slot)
        self.index = index or Const(0.0)
        self.seed = seed
        self.key = key
        self.only_slot = mode.endswith("_SLOT")

    def _pick(self, ctx):
        mats = [m for m in self.materials if m is not None]
        if not mats:
            return None
        mode = self.mode.replace("_SLOT", "")
        v = ctx.values
        if mode == "RANDOM":
            r = hash01(ctx.seed, self.seed, self.key, int(v.get("spline", 0)), int(v.get("input", 0)),
                       int(v.get("index", 0)), int(v.get("global_index", 0)))
            return mats[min(int(r * len(mats)), len(mats) - 1)]
        if mode == "SEQUENCE":
            k = (self.key, int(v.get("spline", 0)))
            n = ctx.state.get(k, 0)
            ctx.state[k] = n + 1
            return mats[n % len(mats)]
        if mode == "INDEX":
            return mats[int(math.floor(self.index(ctx) + 1e-9)) % len(mats)]
        return mats[0]

    def get(self, ctx):
        seg = self.src.get(ctx) if self.src is not None else None
        if seg is None:
            return None
        mat = self._pick(ctx)
        if mat is None:
            return seg
        out = seg.copy()
        rule = ("SLOT", self.slot, mat) if self.only_slot else ("ALL", -1, mat)
        out.materials = seg.materials + (rule,)
        return out


class UVTransform(Source):
    """Offset / scale / rotate the UVs of the segment (with random offsets)."""

    def __init__(self, src, offset, scale, rotation, rand_offset=(0.0, 0.0), seed=0, key="uv"):
        self.src = src
        self.offset = offset
        self.scale = scale
        self.rotation = rotation
        self.rand_offset = rand_offset
        self.seed = seed
        self.key = key

    def get(self, ctx):
        seg = self.src.get(ctx) if self.src is not None else None
        if seg is None:
            return None
        v = ctx.values
        base = (ctx.seed, self.seed, self.key, int(v.get("spline", 0)), int(v.get("input", 0)),
                int(v.get("index", 0)), int(v.get("global_index", 0)))
        ou = self.offset[0](ctx) + (hash01(*base, "u") * 2.0 - 1.0) * self.rand_offset[0]
        ov = self.offset[1](ctx) + (hash01(*base, "v") * 2.0 - 1.0) * self.rand_offset[1]
        su, sv = self.scale[0](ctx), self.scale[1](ctx)
        a = self.rotation(ctx)
        c, s = math.cos(a), math.sin(a)
        m = np.array(((c * su, -s * sv, ou), (s * su, c * sv, ov), (0.0, 0.0, 1.0)))
        out = seg.copy()
        out.uv = m if seg.uv is None else m @ seg.uv
        return out


class Gap(Source):
    """An empty segment of a given length (leaves a hole)."""

    def __init__(self, length):
        self.length = length

    def get(self, ctx):
        return Seg((), size=max(float(self.length(ctx)), 1e-4), name="gap")
