# SPDX-License-Identifier: GPL-3.0-or-later
"""Ready-made examples: procedural segment meshes, a curve and a style graph."""

import math

import bpy
import numpy as np

from . import curvedata, live, meshio
from .engine.mesh import Assembler, MeshBuf, merge
from .nodes import TREE_ID
from .ops import add_node, link, new_output

# ---------------------------------------------------------------------------
# Procedural meshes
# ---------------------------------------------------------------------------


def _flat(buf, value=True):
    buf.attrs["sharp_face"] = ("FACE", "BOOLEAN", np.full(buf.n_faces, value, dtype=bool))
    return buf


def box(x0, x1, y0, y1, z0, z1, nx=1, mat=None):
    co, polys = [], []
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        co += [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    for i in range(nx):
        a, b = 4 * i, 4 * (i + 1)
        polys += [(a, b, b + 1, a + 1), (a + 1, b + 1, b + 2, a + 2), (a + 2, b + 2, b + 3, a + 3),
                  (a + 3, b + 3, b, a)]
    e = 4 * nx
    polys += [(0, 1, 2, 3), (e, e + 3, e + 2, e + 1)]
    return _flat(MeshBuf.from_polygons(co, polys, materials=[mat]))


def tube_x(x0, x1, radius, y=0.0, z=0.0, segs=12, rings=6, mat=None):
    co, polys = [], []
    for i in range(rings + 1):
        x = x0 + (x1 - x0) * i / rings
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            co.append((x, y + radius * math.cos(a), z + radius * math.sin(a)))
    for i in range(rings):
        for j in range(segs):
            a, b = i * segs + j, i * segs + (j + 1) % segs
            polys.append((a, b, b + segs, a + segs))
    polys.append(tuple(reversed(range(segs))))
    polys.append(tuple(rings * segs + j for j in range(segs)))
    buf = MeshBuf.from_polygons(co, polys, materials=[mat])
    sharp = np.zeros(buf.n_faces, dtype=bool)
    sharp[-2:] = True
    buf.attrs["sharp_face"] = ("FACE", "BOOLEAN", sharp)
    # Sharp cap outlines, otherwise the smooth sides blend with the flat caps.
    last = rings * segs
    buf.attrs["sharp_edge"] = ("EDGE", "BOOLEAN", (buf.edges < segs).all(axis=1) | (buf.edges >= last).all(axis=1))
    return buf


def tube_z(z0, z1, radius, x=0.0, y=0.0, segs=10, mat=None):
    """Vertical cylinder (the X-tube rotated so X becomes Z)."""
    buf = tube_x(z0, z1, radius, segs=segs, rings=1, mat=mat)
    c = buf.co
    return buf.with_co(np.stack([x - c[:, 2], y + c[:, 1], c[:, 0]], axis=1))


def _material(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1.0)
        mat.metallic = metallic
        mat.roughness = roughness
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True
        tree = mat.node_tree
        bsdf = tree.nodes.get("Principled BSDF") if tree else None
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            bsdf.inputs["Metallic"].default_value = metallic
            bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _collection(name, parent):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in parent.children:
        parent.children.link(col)
    return col


def _object(name, buf, collection, location):
    asm = Assembler()
    asm.add(buf, buf.co[None])
    me = bpy.data.meshes.new(name)
    meshio.write_mesh(me, asm.build())
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    ob.location = location
    ob.hide_render = True
    return ob


def _curve(name, points, collection, location, kind="POLY", cyclic=False, handles="AUTO"):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    if kind == "BEZIER":
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(len(points) - 1)
        for bp, co in zip(sp.bezier_points, points):
            bp.co = co
            bp.handle_left_type = bp.handle_right_type = handles
    else:
        sp = cu.splines.new("POLY")
        sp.points.add(len(points) - 1)
        for p, co in zip(sp.points, points):
            p.co = (*co, 1.0)
    sp.use_cyclic_u = cyclic
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    ob.location = location
    return ob


class _Env:
    def __init__(self, context, name, origin):
        root = context.scene.collection
        self.context = context
        self.col = _collection("CurveForge Examples", root)
        self.samples = _collection("CurveForge Samples", self.col)
        self.origin = origin
        self.name = name
        self.count = 0

    def sample(self, name, buf):
        loc = (self.origin[0] - 4.0, self.origin[1] + 2.5 * self.count, self.origin[2])
        self.count += 1
        ob = _object("%s %s" % (self.name, name), buf, self.samples, loc)
        return ob

    def curve(self, points, **kw):
        return _curve("%s Curve" % self.name, points, self.col, self.origin, **kw)

    def tree(self):
        return bpy.data.node_groups.new("%s Style" % self.name, TREE_ID)

    def finish(self, tree):
        obj = new_output(self.context, self.name, tree)
        if obj.name not in self.col.objects:
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            self.col.objects.link(obj)
        for ob in self.samples.objects:
            ob.hide_set(True)
        return obj


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def fence(env):
    metal = _material("CF Metal", (0.06, 0.07, 0.08), 0.8, 0.35)
    white = _material("CF White", (0.85, 0.85, 0.82), 0.0, 0.45)
    parts = [box(0.0, 2.0, -0.02, 0.02, 0.10, 0.14, nx=16, mat=metal),
             box(0.0, 2.0, -0.02, 0.02, 0.86, 0.90, nx=16, mat=metal)]
    parts += [box(x - 0.012, x + 0.012, -0.012, 0.012, 0.03, 1.0, mat=metal)
              for x in np.linspace(0.0625, 1.9375, 16)]
    panel = env.sample("Panel", merge(parts))
    post = env.sample("Post", merge([box(-0.06, 0.06, -0.06, 0.06, 0.0, 1.1, mat=white),
                                     box(-0.075, 0.075, -0.075, 0.075, 1.1, 1.16, mat=white)]))
    curve = env.curve([(0, 0, 0), (8, 0, 0), (8, 6, 1.2), (2, 9, 1.2)])
    t = env.tree()
    with live.suspended():
        spl = add_node(t, "CF_NodeSpline", (-560, 220), curve=curve)
        n_panel = add_node(t, "CF_NodeSegment", (-560, 20), label="Panel", object=panel, align_z="MIN")
        n_post = add_node(t, "CF_NodeSegment", (-560, -230), label="Post", object=post, align_z="MIN",
                          bend="OFF", upright="ON", instance="ON")
        gen = add_node(t, "CF_NodeLinear", (-60, 160), upright=True, cf_panel=True)
        gen.inputs["Evenly Distance"].default_value = 4.0
        link(t, spl, "Spline", gen, "Spline")
        link(t, n_panel, "Segment", gen, "Default")
        for name in ("Start", "End", "Corner", "Evenly"):
            link(t, n_post, "Segment", gen, name)
    return t


def railing(env):
    steel = _material("CF Steel", (0.75, 0.76, 0.78), 1.0, 0.25)
    dark = _material("CF Dark Steel", (0.12, 0.12, 0.13), 0.9, 0.4)
    hand = env.sample("Handrail", tube_x(0.0, 0.3, 0.03, z=1.0, rings=4, mat=steel))
    low = env.sample("Bottom Rail", box(0.0, 0.3, -0.015, 0.015, 0.08, 0.11, nx=3, mat=dark))
    bal_a = env.sample("Baluster A", box(0.135, 0.165, -0.015, 0.015, 0.11, 0.97, mat=dark))
    bal_b = env.sample("Baluster B", merge([box(0.14, 0.16, -0.01, 0.01, 0.11, 0.97, mat=dark),
                                            box(0.11, 0.19, -0.03, 0.03, 0.45, 0.6, mat=steel)]))
    post = env.sample("Post", merge([box(-0.04, 0.04, -0.04, 0.04, 0.0, 1.0, mat=dark),
                                     box(-0.06, 0.06, -0.06, 0.06, 0.0, 0.02, mat=dark)]))
    curve = env.curve([(0, 0, 0), (4, 0, 0), (7, 3, 0), (11, 3, 0)], kind="BEZIER")
    t = env.tree()
    with live.suspended():
        spl = add_node(t, "CF_NodeSpline", (-760, 300), curve=curve)
        n_hand = add_node(t, "CF_NodeSegment", (-760, 110), label="Handrail", object=hand)
        n_low = add_node(t, "CF_NodeSegment", (-760, -80), label="Bottom Rail", object=low)
        n_a = add_node(t, "CF_NodeSegment", (-760, -270), label="Baluster A", object=bal_a, bend="OFF")
        n_b = add_node(t, "CF_NodeSegment", (-760, -460), label="Baluster B", object=bal_b, bend="OFF")
        rnd = add_node(t, "CF_NodeRandomize", (-500, -330), seed=4)
        link(t, n_a, "Segment", rnd, 0)
        link(t, n_b, "Segment", rnd, 1)
        rnd.cf_sync_inputs()
        rnd.inputs[0].weight = 3.0
        comp = add_node(t, "CF_NodeCompose", (-300, -40), mode="OVERLAP")
        link(t, n_hand, "Segment", comp, 0)
        comp.cf_sync_inputs()
        link(t, n_low, "Segment", comp, 1)
        comp.cf_sync_inputs()
        link(t, rnd, "Segment", comp, 2)
        comp.cf_sync_inputs()
        n_post = add_node(t, "CF_NodeSegment", (-300, -320), label="Post", object=post, bend="OFF")
        gen = add_node(t, "CF_NodeLinear", (0, 200), cf_panel=True)
        gen.inputs["Evenly Distance"].default_value = 2.5
        link(t, spl, "Spline", gen, "Spline")
        link(t, comp, "Segment", gen, "Default")
        for name in ("Start", "End", "Evenly"):
            link(t, n_post, "Segment", gen, name)
    return t


def wall(env):
    plaster = _material("CF Plaster", (0.43, 0.56, 0.48), 0.0, 0.7)
    frame = _material("CF Frame", (0.2, 0.21, 0.22), 0.4, 0.4)
    glass = _material("CF Glass", (0.55, 0.7, 0.8), 0.0, 0.05)
    wood = _material("CF Wood", (0.36, 0.2, 0.09), 0.0, 0.5)
    panel = env.sample("Wall", merge([box(0.0, 1.2, -0.12, 0.12, 0.0, 2.8, nx=4, mat=plaster),
                                      box(1.18, 1.2, -0.125, 0.125, 0.0, 2.8, mat=frame)]))
    window = env.sample("Window", merge([
        box(0.0, 1.2, -0.12, 0.12, 0.0, 0.9, nx=4, mat=plaster),
        box(0.0, 1.2, -0.12, 0.12, 2.1, 2.8, nx=4, mat=plaster),
        box(0.0, 0.2, -0.12, 0.12, 0.9, 2.1, mat=plaster),
        box(1.0, 1.2, -0.12, 0.12, 0.9, 2.1, mat=plaster),
        box(0.2, 1.0, -0.04, 0.04, 0.9, 0.96, mat=frame),
        box(0.2, 1.0, -0.04, 0.04, 2.04, 2.1, mat=frame),
        box(0.2, 1.0, -0.01, 0.01, 0.96, 2.04, mat=glass),
        box(1.18, 1.2, -0.125, 0.125, 0.0, 2.8, mat=frame)]))
    pillar = env.sample("Pillar", box(-0.2, 0.2, -0.2, 0.2, 0.0, 3.0, mat=frame))
    door = env.sample("Door", merge([box(-0.8, 0.8, -0.12, 0.12, 2.3, 2.8, nx=2, mat=plaster),
                                     box(-0.8, -0.65, -0.12, 0.12, 0.0, 2.3, mat=plaster),
                                     box(0.65, 0.8, -0.12, 0.12, 0.0, 2.3, mat=plaster),
                                     box(-0.65, 0.65, -0.03, 0.03, 0.0, 2.25, mat=wood)]))
    curve = env.curve([(0, 0, 0), (12, 0, 0), (12, 7, 0), (0, 7, 0)], cyclic=True)
    t = env.tree()
    with live.suspended():
        spl = add_node(t, "CF_NodeSpline", (-780, 360), curve=curve)
        n_wall = add_node(t, "CF_NodeSegment", (-780, 160), label="Wall", object=panel)
        n_win = add_node(t, "CF_NodeSegment", (-780, -40), label="Window", object=window)
        expr = add_node(t, "CF_NodeExpression", (-780, -250), expression="index % 3 == 1 and from_end > 0",
                        cf_panel=True)
        cond = add_node(t, "CF_NodeConditional", (-450, 60))
        link(t, n_win, "Segment", cond, "True")
        link(t, n_wall, "Segment", cond, "False")
        link(t, expr, "Value", cond, "Condition")
        n_pillar = add_node(t, "CF_NodeSegment", (-450, -200), label="Pillar", object=pillar, bend="OFF")
        n_door = add_node(t, "CF_NodeSegment", (-450, -420), label="Door", object=door, bend="OFF")
        gen = add_node(t, "CF_NodeLinear", (-60, 240), marker_mode="DISTANCES", marker_list="25%", cf_panel=True)
        link(t, spl, "Spline", gen, "Spline")
        link(t, cond, "Segment", gen, "Default")
        link(t, n_pillar, "Segment", gen, "Corner")
        link(t, n_door, "Segment", gen, "Marker")
    return t


def kerb(env):
    mats = [_material("CF Stone %d" % i, c, 0.0, r) for i, (c, r) in enumerate(
        (((0.55, 0.55, 0.52), 0.8), ((0.42, 0.41, 0.4), 0.85), ((0.66, 0.62, 0.55), 0.75)))]
    stone = env.sample("Stone", merge([box(0.0, 0.5, -0.15, 0.15, 0.0, 0.22, nx=3, mat=mats[0]),
                                       box(0.02, 0.48, -0.13, 0.13, 0.22, 0.25, nx=3, mat=mats[0])]))
    curve = env.curve([(0, 0, 0), (4, 2, 0.2), (8, -1, 0.5), (12, 1, 0.3)], kind="BEZIER")
    t = env.tree()
    with live.suspended():
        spl = add_node(t, "CF_NodeSpline", (-800, 260), curve=curve)
        n_stone = add_node(t, "CF_NodeSegment", (-800, 40), label="Stone", object=stone)
        xf = add_node(t, "CF_NodeTransform", (-560, 40), rand_rotation=(0.0, 0.0, math.pi),
                      rotation_step=math.pi, rand_offset=(0.0, 0.01, 0.015), seed=2, cf_panel=True)
        link(t, n_stone, "Segment", xf, "Segment")
        mat = add_node(t, "CF_NodeMaterial", (-320, 40), pick="RANDOM", count=3, seed=5,
                       material_0=mats[0], material_1=mats[1], material_2=mats[2])
        link(t, xf, "Segment", mat, "Segment")
        rnd = add_node(t, "CF_NodeRandom", (-800, -180), seed=1)
        rnd.inputs["Min"].default_value = 0.004
        rnd.inputs["Max"].default_value = 0.02
        gen = add_node(t, "CF_NodeLinear", (-40, 200), upright=True)
        link(t, spl, "Spline", gen, "Spline")
        link(t, mat, "Segment", gen, "Default")
        link(t, rnd, "Value", gen, "Spacing")
    return t


def pattern(env):
    red = _material("CF Red", (0.7, 0.15, 0.1), 0.0, 0.5)
    blue = _material("CF Blue", (0.1, 0.25, 0.7), 0.0, 0.5)
    grey = _material("CF Grey", (0.3, 0.3, 0.3), 0.0, 0.5)
    a = env.sample("Block A", box(0.0, 1.0, -0.2, 0.2, 0.0, 0.6, nx=4, mat=red))
    b = env.sample("Block B", box(0.0, 0.5, -0.2, 0.2, 0.0, 1.0, nx=2, mat=blue))
    cap = env.sample("Cap", box(0.0, 0.3, -0.3, 0.3, 0.0, 1.2, mat=grey))
    curve = env.curve([(0, 0, 0), (5, 3, 0), (10, 0, 0), (14, 2, 0)], kind="BEZIER")
    t = env.tree()
    with live.suspended():
        spl = add_node(t, "CF_NodeSpline", (-760, 300), curve=curve)
        n_a = add_node(t, "CF_NodeSegment", (-760, 100), label="Block A", object=a)
        n_b = add_node(t, "CF_NodeSegment", (-760, -100), label="Block B", object=b)
        gap = add_node(t, "CF_NodeGap", (-760, -300))
        gap.inputs["Length"].default_value = 0.3
        seq = add_node(t, "CF_NodeSequence", (-460, 40))
        link(t, n_a, "Segment", seq, 0)
        seq.cf_sync_inputs()
        link(t, n_b, "Segment", seq, 1)
        seq.cf_sync_inputs()
        link(t, gap, "Segment", seq, 2)
        seq.cf_sync_inputs()
        seq.inputs[0].count = 2
        mir = add_node(t, "CF_NodeMirror", (-460, -260), x=False, y=True, mode="ALTERNATE")
        n_cap = add_node(t, "CF_NodeSegment", (-760, -470), label="Cap", object=cap, bend="OFF")
        link(t, n_cap, "Segment", mir, "Segment")
        gen = add_node(t, "CF_NodeLinear", (-60, 220), cf_panel=True)
        link(t, spl, "Spline", gen, "Spline")
        link(t, seq, "Segment", gen, "Default")
        link(t, mir, "Segment", gen, "Start")
        link(t, mir, "Segment", gen, "End")
    return t


def garden(env):
    leaf = _material("CF Hedge", (0.1, 0.28, 0.08), 0.0, 0.9)
    paint = _material("CF Picket", (0.92, 0.9, 0.84), 0.0, 0.5)
    stone = _material("CF Stone Wall", (0.52, 0.48, 0.42), 0.0, 0.85)
    hedge = env.sample("Hedge", merge([box(0.0, 1.0, -0.32, 0.32, 0.0, 0.9, nx=4, mat=leaf),
                                       box(0.0, 1.0, -0.27, 0.27, 0.9, 1.0, nx=4, mat=leaf)]))
    parts = [box(0.0, 1.0, -0.03, 0.0, 0.22, 0.3, nx=4, mat=paint),
             box(0.0, 1.0, -0.03, 0.0, 0.72, 0.8, nx=4, mat=paint)]
    parts += [box(x - 0.04, x + 0.04, 0.0, 0.025, 0.04, 1.0, mat=paint) for x in (0.1, 0.3, 0.5, 0.7, 0.9)]
    picket = env.sample("Picket Fence", merge(parts))
    wall = env.sample("Low Wall", merge([box(0.0, 1.0, -0.2, 0.2, 0.0, 0.55, nx=4, mat=stone),
                                         box(0.0, 1.0, -0.25, 0.25, 0.55, 0.65, nx=4, mat=stone)]))
    post = env.sample("Post", box(-0.1, 0.1, -0.1, 0.1, 0.0, 1.15, mat=stone))
    curve = env.curve([(0, 0, 0), (6, 0, 0), (6, 5, 0), (0, 5, 0), (0, 1.5, 0)])
    curvedata.set_ids(curve, [(0, 1)], 1)
    curvedata.set_ids(curve, [(0, 2)], 2)
    t = env.tree()
    with live.suspended():
        spl = add_node(t, "CF_NodeSpline", (-760, 320), curve=curve)
        n_hedge = add_node(t, "CF_NodeSegment", (-760, 120), label="Hedge (ID 0)", object=hedge, align_z="MIN")
        n_picket = add_node(t, "CF_NodeSegment", (-760, -80), label="Picket Fence (ID 1)", object=picket,
                            align_z="MIN")
        n_wall = add_node(t, "CF_NodeSegment", (-760, -280), label="Low Wall (ID 2)", object=wall, align_z="MIN")
        sel = add_node(t, "CF_NodeSelector", (-460, 40), variable="segment_id", mode="CLAMP")
        for i, node in enumerate((n_hedge, n_picket, n_wall)):
            link(t, node, "Segment", sel, str(i))
            sel.cf_sync_inputs()
        n_post = add_node(t, "CF_NodeSegment", (-460, -240), label="Post", object=post, align_z="MIN",
                          bend="OFF", instance="ON")
        gen = add_node(t, "CF_NodeLinear", (-60, 220), cf_panel=True)
        link(t, spl, "Spline", gen, "Spline")
        link(t, sel, "Segment", gen, "Default")
        for name in ("Start", "End", "Corner"):
            link(t, n_post, "Segment", gen, name)
    return t


BUILDERS = {"FENCE": ("Fence", fence), "RAILING": ("Railing", railing), "WALL": ("Wall", wall),
            "KERB": ("Kerb", kerb), "PATTERN": ("Pattern", pattern), "GARDEN": ("Garden", garden)}


def build(context, key, origin=(0.0, 0.0, 0.0)):
    name, fn = BUILDERS[key]
    env = _Env(context, "CF " + name, tuple(origin))
    tree = fn(env)
    return env.finish(tree)
