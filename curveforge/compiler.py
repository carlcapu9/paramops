# SPDX-License-Identifier: GPL-3.0-or-later
"""Turns a CurveForge style (node tree) into engine sources and generator jobs."""

import math

import numpy as np

from . import curvedata, meshio
from .engine import graph as g
from .engine.expr import ExpressionError
from .engine.generate import SplineJob
from .engine.linear import LinearSettings
from .engine.mesh import MeshBuf
from .engine.place import PlaceOptions

GEN_INPUTS = {"Start": g.INPUT_START, "End": g.INPUT_END, "Corner": g.INPUT_CORNER,
              "Evenly": g.INPUT_EVENLY, "Default": g.INPUT_DEFAULT, "Marker": g.INPUT_MARKER}
TRI = {"DEFAULT": None, "ON": True, "OFF": False}
MESH_LIKE = {"MESH", "CURVE", "SURFACE", "FONT", "META"}


def parse_indices(text):
    """'0, 2, 4-6' -> {0, 2, 4, 5, 6}; empty -> None (all)."""
    text = (text or "").strip()
    if not text:
        return None
    out = set()
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            try:
                out.update(range(int(a), int(b) + 1))
            except ValueError:
                continue
        else:
            try:
                out.add(int(tok))
            except ValueError:
                continue
    return out


class GeneratorJob:
    __slots__ = ("node", "jobs", "sources", "settings_for", "place", "clip", "weld", "seed", "instance")


class Compiler:
    def __init__(self, tree, depsgraph, display="FULL"):
        self.tree = tree
        self.dg = depsgraph
        self.display = display
        self.sources = {}
        self.visiting = set()
        self.deps = set()
        self.warnings = []

    # -- links -----------------------------------------------------------------

    @staticmethod
    def _follow(link):
        # ``is_valid`` is not checked: new links are only validated after the tree
        # update callback that triggers the rebuild.
        if link is None or getattr(link, "is_muted", False):
            return None
        node, out = link.from_node, link.from_socket
        seen = 0
        while node.bl_idname == "NodeReroute":
            inp = node.inputs[0]
            links = inp.links
            if not links or seen > 256:
                return None
            link = links[0]
            if getattr(link, "is_muted", False):
                return None
            node, out = link.from_node, link.from_socket
            seen += 1
        return node, out

    # ``socket.links`` is read instead of the ``is_linked`` flag, which is only
    # refreshed after the tree update callback (the link list is always current).
    def upstream(self, socket):
        links = socket.links if socket is not None else ()
        if not links:
            return None
        return self._follow(links[0])

    def upstream_all(self, socket):
        links = socket.links if socket is not None else ()
        if not links:
            return []
        links = sorted(links, key=lambda lk: getattr(lk, "multi_input_sort_id", 0))
        return [r for r in (self._follow(lk) for lk in links) if r is not None]

    # -- numbers -----------------------------------------------------------------

    def num(self, socket, default=0.0):
        if socket is None:
            return g.Const(default)
        up = self.upstream(socket)
        if up is None:
            val = getattr(socket, "default_value", default)
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            try:
                return g.Const(float(val))
            except TypeError:
                return g.Const(float(val[0]))
        node, out = up
        return self.node_num(node, out)

    def node_num(self, node, out):
        key = ("num", node.name, out.identifier)
        if key in self.sources:
            return self.sources[key]
        if key in self.visiting:
            self.warnings.append("Loop at node '%s'" % node.name)
            return g.Const(0.0)
        self.visiting.add(key)
        try:
            num = self._build_num(node, out)
        finally:
            self.visiting.discard(key)
        self.sources[key] = num
        return num

    def _build_num(self, node, out):
        bid = node.bl_idname
        if node.mute:
            first = next((s for s in node.inputs if s.bl_idname != "CF_SocketSegment"), None)
            return self.num(first) if first is not None else g.Const(0.0)
        if bid == "CF_NodeValue":
            return g.Const(node.value)
        if bid == "CF_NodeInteger":
            return g.Const(float(node.value))
        if bid == "CF_NodeInfo":
            return g.Var(node.variable)
        if bid == "CF_NodeMath":
            return g.MathNum(node.operation, self.num(node.inputs["A"]), self.num(node.inputs["B"]),
                             self.num(node.inputs["C"]))
        if bid == "CF_NodeRandom":
            return g.RandomNum(self.num(node.inputs["Min"]), self.num(node.inputs["Max"], 1.0), node.seed,
                               node.per, node.integer, key=node.name)
        if bid == "CF_NodeExpression":
            try:
                return g.ExprNum(node.expression, {n: self.num(node.inputs[n]) for n in ("a", "b", "c", "d")})
            except ExpressionError as exc:
                self.warnings.append("Expression '%s': %s" % (node.name, exc))
                return g.Const(0.0)
        if bid == "CF_NodeCombine":
            return self.num(node.inputs["X"])
        self.warnings.append("'%s' does not give a number" % node.name)
        return g.Const(0.0)

    def vec(self, socket, default=(0.0, 0.0, 0.0)):
        up = self.upstream(socket)
        if up is None:
            val = tuple(socket.default_value) if socket is not None else default
            return tuple(g.Const(v) for v in val)
        node, out = up
        if node.bl_idname == "CF_NodeCombine" and not node.mute:
            return tuple(self.num(node.inputs[n]) for n in ("X", "Y", "Z"))
        n = self.node_num(node, out)
        return (n, n, n)

    # -- segments --------------------------------------------------------------------

    def source(self, socket):
        up = self.upstream(socket)
        if up is None:
            return None
        node, out = up
        key = ("src", node.name, out.identifier)
        if key in self.sources:
            return self.sources[key]
        if key in self.visiting:
            self.warnings.append("Loop at node '%s'" % node.name)
            return None
        self.visiting.add(key)
        try:
            src = self._build_source(node)
        finally:
            self.visiting.discard(key)
        self.sources[key] = src
        return src

    def segment_inputs(self, node):
        socks = [s for s in node.inputs if s.bl_idname == "CF_SocketSegment"]
        while socks and not socks[-1].links:
            socks.pop()
        return socks

    def _build_source(self, node):
        bid = node.bl_idname
        if node.mute:
            first = next((s for s in node.inputs if s.bl_idname == "CF_SocketSegment"), None)
            return self.source(first)
        if bid == "CF_NodeSegment":
            return self.segment_node(node)
        if bid == "CF_NodeGap":
            return g.Gap(self.num(node.inputs["Length"], 1.0))
        ins = self.segment_inputs(node) if bid in {"CF_NodeCompose", "CF_NodeSequence", "CF_NodeRandomize",
                                                    "CF_NodeSelector"} else None
        if bid == "CF_NodeCompose":
            return g.Compose([self.source(s) for s in ins], node.mode, node.gap)
        if bid == "CF_NodeSequence":
            return g.Sequence([self.source(s) for s in ins], [s.count for s in ins], node.scope, node.offset,
                              key=node.name)
        if bid == "CF_NodeRandomize":
            return g.Randomize([self.source(s) for s in ins], [s.weight for s in ins], node.seed, node.per,
                               key=node.name)
        if bid == "CF_NodeSelector":
            idx_sock = node.inputs.get("Index")
            index = self.num(idx_sock) if idx_sock is not None and idx_sock.links else g.Var(node.variable)
            return g.Selector([self.source(s) for s in ins], index, node.mode)
        if bid == "CF_NodeConditional":
            cond_sock = node.inputs.get("Condition")
            if cond_sock is not None and cond_sock.links:
                cond = self.num(cond_sock)
            else:
                var = g.Var(node.variable)
                op = node.compare
                if op == "BETWEEN":
                    cond = g.MathNum(op, var, g.Const(node.value), g.Const(node.value2))
                elif op == "EVERY":
                    cond = g.MathNum(op, var, g.Const(node.value), g.Const(node.value2))
                elif op in {"EQUAL", "NOT_EQUAL"}:
                    cond = g.MathNum(op, var, g.Const(node.value), g.Const(1e-6))
                else:
                    cond = g.MathNum(op, var, g.Const(node.value))
            return g.Conditional(self.source(node.inputs["True"]), self.source(node.inputs["False"]), cond)
        if bid == "CF_NodeMirror":
            return g.Mirror(self.source(node.inputs["Segment"]), (node.x, node.y, node.z), node.mode,
                            node.probability, node.seed, key=node.name)
        if bid == "CF_NodeTransform":
            rot = tuple(g.MathNum("MULTIPLY", n, g.Const(math.pi / 180.0))
                        for n in self.vec(node.inputs["Rotation"]))
            rs = tuple(node.rand_scale)
            rand_scale = (rs[0], rs[0], rs[0]) if node.uniform else rs
            return g.Transform(self.source(node.inputs["Segment"]), self.vec(node.inputs["Offset"]), rot,
                               self.vec(node.inputs["Scale"], (1.0, 1.0, 1.0)), tuple(node.rand_offset),
                               tuple(node.rand_rotation), rand_scale, node.uniform, node.rotation_step,
                               node.seed, key=node.name)
        if bid == "CF_NodeMaterial":
            mode = node.pick if node.target == "ALL" else node.pick + "_SLOT"
            return g.MaterialOp(self.source(node.inputs["Segment"]), mode, node.materials(), node.slot,
                                self.num(node.inputs["Index"]), node.seed, key=node.name)
        if bid == "CF_NodeUVTransform":
            return g.UVTransform(self.source(node.inputs["Segment"]),
                                 (g.Const(node.offset[0]), g.Const(node.offset[1])),
                                 (g.Const(node.scale[0]), g.Const(node.scale[1])), g.Const(node.rotation),
                                 tuple(node.rand_offset), node.seed, key=node.name)
        self.warnings.append("'%s' does not give segments" % node.name)
        return None

    def _geo(self, obj, use_transform, world=False):
        self.deps.add(obj)
        buf = meshio.object_buf(obj, self.dg, use_transform, world)
        if buf is None or not len(buf.co):
            return None, None
        return buf, (obj.session_uid, use_transform, world)

    def _box(self, buf):
        bmin, bmax = buf.bbox()
        return MeshBuf.box(bmin, bmax, buf.materials[0] if buf.materials else None)

    def segment_node(self, node):
        static = np.eye(4)
        static[:3, :3] = g.euler_matrix(*node.rotation)[:3, :3] @ np.diag(
            [s * (-1.0 if m else 1.0) for s, m in zip(node.scale, (node.mirror_x, node.mirror_y, node.mirror_z))])

        def make(parts, name):
            if not parts:
                return None
            seg = g.Seg([(geo, static @ m) for geo, m in parts], name=name)
            seg.bend = TRI[node.bend]
            seg.upright = TRI[node.upright]
            seg.slice = TRI[node.slice]
            seg.adaptive = None if node.adaptive == "ON" else False
            seg.instance = TRI[node.instance]
            seg.align = (node.align_x, node.align_y, node.align_z)
            seg.offset = tuple(node.offset)
            seg.orient = node.orient
            seg.size = node.size if node.use_size else None
            seg.pad = (node.padding[0], node.padding[1])
            return seg

        def geo_of(obj, world=False):
            buf, key = self._geo(obj, node.use_transform, world)
            if buf is None:
                return None
            if self.display == "BOX":
                return g.Geo(key + ("box",), self._box(buf))
            m = np.array(obj.matrix_world, dtype=np.float64)
            if not world:
                rot = np.eye(4)
                if node.use_transform:
                    rot[:3, :3] = m[:3, :3]
                m = rot
            return g.Geo(key, buf, inst=(obj.name, m))

        if node.source == "OBJECT":
            if node.object is None:
                return None
            geo = geo_of(node.object)
            seg = make([(geo, np.eye(4))] if geo is not None else [], node.object.name)
            return g.Fixed(seg) if seg is not None else None
        col = node.collection
        if col is None:
            return None
        objs = sorted((o for o in col.all_objects if o.type in MESH_LIKE), key=lambda o: o.name)
        if node.collection_mode == "COMBINE":
            parts = []
            off = np.eye(4)
            off[:3, 3] = -np.array(col.instance_offset)
            for o in objs:
                geo = geo_of(o, world=True)
                if geo is not None:
                    parts.append((geo, off))
            seg = make(parts, col.name)
            return g.Fixed(seg) if seg is not None else None
        fixed = []
        for o in objs:
            geo = geo_of(o)
            seg = make([(geo, np.eye(4))] if geo is not None else [], o.name)
            if seg is not None:
                fixed.append(g.Fixed(seg))
        if not fixed:
            return None
        if node.collection_mode == "SEQUENCE":
            return g.Sequence(fixed, [1] * len(fixed), "SPLINE", 0, key=node.name)
        weights = [getattr(o, "cf_weight", 1.0) for o in objs[:len(fixed)]]
        return g.Randomize(fixed, weights, node.seed, "SEGMENT", key=node.name)

    # -- generators -------------------------------------------------------------------

    def generators(self):
        out = []
        for node in self.tree.nodes:
            if node.bl_idname == "CF_NodeLinear" and not node.mute and node.enabled:
                job = self.generator(node)
                if job is not None:
                    out.append(job)
        return out

    def generator(self, node):
        jobs = []
        for up_node, _out in self.upstream_all(node.inputs.get("Spline")):
            if up_node.bl_idname != "CF_NodeSpline" or up_node.mute or up_node.curve is None:
                continue
            curve = up_node.curve
            self.deps.add(curve)
            for spec in curvedata.read_splines(curve, parse_indices(up_node.splines)):
                jobs.append(SplineJob(spec, up_node.reverse, up_node.resolution, up_node.twist, up_node.use_tilt))
        sources = {}
        for name, inp in GEN_INPUTS.items():
            src = self.source(node.inputs.get(name))
            if src is not None:
                sources[inp] = src
        spacing = self.num(node.inputs.get("Spacing"))
        ev_dist = self.num(node.inputs.get("Evenly Distance"), 2.0)
        ev_count = self.num(node.inputs.get("Evenly Count"), 1.0)
        clip_s = self.num(node.inputs.get("Clip Start"))
        clip_e = self.num(node.inputs.get("Clip End"))
        props = dict(clip_percent=node.clip_percent, extend=node.extend, adaptive=node.adaptive, fit=node.fit,
                     count=node.count, align=node.align, remainder=node.remainder,
                     corner_mode=node.corner_mode, corner_angle=node.corner_angle,
                     corner_split=node.corner_split, split_ids=node.split_ids, evenly_mode=node.evenly_mode,
                     evenly_offset=node.evenly_offset, evenly_scope=node.evenly_scope,
                     marker_mode=node.marker_mode, marker_list=node.marker_list, marker_step=node.marker_step,
                     marker_offset=node.marker_offset, max_segments=node.max_segments)

        def settings_for(ctx):
            return LinearSettings(spacing=spacing(ctx), evenly_distance=max(ev_dist(ctx), 1e-3),
                                  evenly_count=int(round(ev_count(ctx))), clip_start=clip_s(ctx),
                                  clip_end=clip_e(ctx), **props)

        neutral = g.Ctx({})
        job = GeneratorJob()
        job.node = node
        job.jobs = jobs
        job.sources = sources
        job.settings_for = settings_for
        job.place = PlaceOptions(bend=node.bend, upright=node.upright, slice=node.slice,
                                 offset_y=self.num(node.inputs.get("Offset Y"))(neutral),
                                 offset_z=self.num(node.inputs.get("Offset Z"))(neutral),
                                 uv_mode=node.uv_mode, uv_scale=node.uv_scale, slicer=meshio.slicer,
                                 instance=node.instance)
        job.clip = None
        if node.clip_curve is not None:
            self.deps.add(node.clip_curve)
            polys = curvedata.polygons(node.clip_curve)
            if polys:
                job.clip = (polys, node.clip_inside)
        job.weld = node.weld_distance if node.weld else 0.0
        job.seed = node.seed
        job.instance = node.instance
        return job
