# SPDX-License-Identifier: GPL-3.0-or-later
"""Procedural demo scene: a fence on a slope, a wall with round corners and a stair rail."""

import math

import bpy
import numpy as np

from . import handlers, writer
from .core.meshdata import MeshData, assemble, merge


def _material(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1.0)
        mat.metallic = metallic
        mat.roughness = roughness
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            bsdf.inputs["Metallic"].default_value = metallic
            bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _flat(md):
    md.attrs["sharp_face"] = ("FACE", "BOOLEAN", np.ones(md.nf, dtype=bool))
    return md


def box(x0, x1, y0, y1, z0, z1, nx=1, material=None):
    """Axis aligned box, subdivided ``nx`` times along X (so it can bend)."""
    co, faces = [], []
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        co += [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
    for i in range(nx):
        a, b = 4 * i, 4 * (i + 1)
        faces += [(a, b, b + 1, a + 1), (a + 1, b + 1, b + 2, a + 2),
                  (a + 2, b + 2, b + 3, a + 3), (a + 3, b + 3, b, a)]
    e = 4 * nx
    faces += [(0, 1, 2, 3), (e, e + 3, e + 2, e + 1)]
    return _flat(MeshData.from_faces(co, faces, materials=[material]))


def tube_x(x0, x1, radius, segs=12, rings=8, material=None, caps=True):
    """Cylinder along X with ``rings`` subdivisions."""
    co, faces = [], []
    for i in range(rings + 1):
        x = x0 + (x1 - x0) * i / rings
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            co.append((x, radius * math.cos(a), radius * math.sin(a)))
    for i in range(rings):
        for j in range(segs):
            a = i * segs + j
            b = i * segs + (j + 1) % segs
            faces.append((a, b, b + segs, a + segs))
    if caps:
        faces.append(tuple(reversed(range(segs))))
        faces.append(tuple(rings * segs + j for j in range(segs)))
    md = MeshData.from_faces(co, faces, materials=[material])
    sharp = np.zeros(md.nf, dtype=bool)
    if caps:
        sharp[-2:] = True
    md.attrs["sharp_face"] = ("FACE", "BOOLEAN", sharp)
    return md


def tube_z(z0, z1, radius, segs=10, material=None):
    md = tube_x(z0, z1, radius, segs=segs, rings=1, material=material)
    return md.transformed(np.array(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))))  # X -> Z


def sphere(radius, segs=12, rings=6, material=None):
    co = [(0.0, 0.0, -radius)]
    for i in range(1, rings):
        t = math.pi * i / rings - math.pi / 2.0
        for j in range(segs):
            a = 2.0 * math.pi * j / segs
            co.append((radius * math.cos(t) * math.cos(a), radius * math.cos(t) * math.sin(a),
                       radius * math.sin(t)))
    co.append((0.0, 0.0, radius))
    top = len(co) - 1
    faces = []
    for j in range(segs):
        faces.append((0, 1 + (j + 1) % segs, 1 + j))
    for i in range(rings - 2):
        for j in range(segs):
            a = 1 + i * segs + j
            b = 1 + i * segs + (j + 1) % segs
            faces.append((a, b, b + segs, a + segs))
    base = 1 + (rings - 2) * segs
    for j in range(segs):
        faces.append((base + j, base + (j + 1) % segs, top))
    md = MeshData.from_faces(co, faces, materials=[material])
    md.attrs["sharp_face"] = ("FACE", "BOOLEAN", np.zeros(md.nf, dtype=bool))
    return md


def _object(name, md, collection):
    me = bpy.data.meshes.get(name)
    if me is None:
        me = bpy.data.meshes.new(name)
    writer.write(me, assemble([(md, md.co[None], np.zeros(1, dtype=bool))]))
    ob = bpy.data.objects.get(name)
    if ob is None:
        ob = bpy.data.objects.new(name, me)
    ob.data = me
    if ob.name not in collection.objects:
        collection.objects.link(ob)
    return ob


def _collection(name, parent):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in parent.children:
        parent.children.link(col)
    return col


def _curve(name, points, collection, cyclic=False, location=(0.0, 0.0, 0.0)):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    for p, co in zip(sp.points, points):
        p.co = (*co, 1.0)
    sp.use_cyclic_u = cyclic
    ob = bpy.data.objects.new(name, cu)
    ob.location = location
    collection.objects.link(ob)
    return ob


def build_samples(collection):
    metal = _material("PO_Metal", (0.08, 0.09, 0.1), metallic=0.8, roughness=0.35)
    white = _material("PO_White", (0.85, 0.85, 0.82), roughness=0.4)
    wall = _material("PO_Wall", (0.43, 0.56, 0.48), roughness=0.6)
    frame = _material("PO_Frame", (0.25, 0.26, 0.27), metallic=0.5, roughness=0.4)
    wood = _material("PO_Wood", (0.35, 0.18, 0.08), roughness=0.55)
    steel = _material("PO_Steel", (0.75, 0.76, 0.78), metallic=1.0, roughness=0.25)

    # Fence panel: two rails + vertical bars, 2 m long, built along +X from x = 0.
    parts = [box(0.0, 2.0, -0.02, 0.02, 0.10, 0.14, nx=16, material=metal),
             box(0.0, 2.0, -0.02, 0.02, 0.86, 0.90, nx=16, material=metal)]
    for i in range(1, 16):
        x = 2.0 * i / 16
        parts.append(box(x - 0.01, x + 0.01, -0.01, 0.01, 0.02, 1.0, material=metal))
    panel = merge(parts)
    post = merge([box(-0.06, 0.06, -0.06, 0.06, 0.0, 1.1, material=white),
                  box(-0.075, 0.075, -0.075, 0.075, 1.1, 1.16, material=white)])
    gate = merge([box(-0.6, 0.6, -0.025, 0.025, 0.05, 0.11, nx=6, material=wood),
                  box(-0.6, 0.6, -0.025, 0.025, 0.92, 0.98, nx=6, material=wood),
                  box(-0.6, -0.54, -0.025, 0.025, 0.05, 0.98, material=wood),
                  box(0.54, 0.6, -0.025, 0.025, 0.05, 0.98, material=wood)]
                 + [box(x - 0.03, x + 0.03, -0.02, 0.02, 0.11, 0.92, material=wood)
                    for x in np.linspace(-0.42, 0.42, 6)])

    # Wall module: 1 m panel with a dark joint, subdivided so it bends around round corners.
    wall_md = merge([box(0.0, 0.98, -0.1, 0.1, 0.0, 2.6, nx=10, material=wall),
                     box(0.98, 1.0, -0.09, 0.09, 0.0, 2.6, material=frame),
                     box(0.0, 1.0, -0.11, 0.11, 2.6, 2.7, nx=10, material=frame)])

    # Handrail: tube along X (the curve runs through the rail axis), ball ends, hanging posts.
    rail = tube_x(0.0, 1.0, 0.025, segs=16, rings=10, material=steel)
    ball = sphere(0.04, material=steel)
    support = merge([tube_z(-0.95, 0.0, 0.018, material=steel),
                     box(-0.06, 0.06, -0.06, 0.06, -0.97, -0.95, material=steel)])

    objs = {
        "PO_FencePanel": panel, "PO_Post": post, "PO_Gate": gate, "PO_WallPanel": wall_md,
        "PO_Rail": rail, "PO_RailEnd": ball, "PO_RailSupport": support,
    }
    out = {}
    for i, (name, md) in enumerate(objs.items()):
        ob = _object(name, md, collection)
        ob.location = (-6.0, 3.0 * i, 0.0)
        ob.hide_render = True
        out[name] = ob
    return out


def build_demo(context):
    from .ops import create_scatter

    root = context.scene.collection
    demo = _collection("ParamOps Demo", root)
    sample_col = _collection("ParamOps Samples", demo)
    smp = build_samples(sample_col)
    created = []
    with handlers.suspended():
        # 1. Fence with a slope, corner posts, evenly posts and a gate marker.
        fence_curve = _curve("PO_FencePath", [(0, 0, 0), (9, 0, 0), (9, 7, 1.6), (2, 10, 1.6)], demo)
        fence = create_scatter(context, fence_curve, name="PO_Fence")
        st = fence.paramops
        st.default.object = smp["PO_FencePanel"]
        st.default.keep_vertical = True
        st.default.align_z = "BOTTOM"
        for slot in (st.start, st.end, st.corner, st.evenly):
            slot.object = smp["PO_Post"]
            slot.align_z = "BOTTOM"
            slot.flat_top = 0.07
        st.evenly_mode = "SPACING"
        st.evenly_spacing = 4.0
        st.evenly_min_gap = 1.0
        mk = st.markers.add()
        mk.name = "Gate"
        mk.slot.object = smp["PO_Gate"]
        mk.slot.deform = False
        mk.slot.keep_vertical = True
        mk.slot.align_z = "BOTTOM"
        mk.positions = "4.5"
        created.append(fence)

        # 2. Closed wall with round corners.
        wall_curve = _curve("PO_WallPath", [(0, 0, 0), (12, 0, 0), (12, 7, 0), (0, 7, 0)], demo,
                            cyclic=True, location=(16.0, 0.0, 0.0))
        wall = create_scatter(context, wall_curve, name="PO_Wall")
        st = wall.paramops
        st.default.object = smp["PO_WallPanel"]
        st.corner_radius = 1.5
        created.append(wall)

        # 3. Stair handrail: ball ends, supports every metre.
        rail_curve = _curve("PO_RailPath", [(0, 0, 1.0), (3, 0, 1.0), (7, 0, 3.2), (10, 0, 3.2)], demo,
                            location=(16.0, -6.0, 0.0))
        rail = create_scatter(context, rail_curve, name="PO_Handrail")
        st = rail.paramops
        st.default.object = smp["PO_Rail"]
        st.corner_radius = 0.3
        st.start.object = smp["PO_RailEnd"]
        st.end.object = smp["PO_RailEnd"]
        st.evenly.object = smp["PO_RailSupport"]
        st.evenly_mode = "FIT"
        st.evenly_spacing = 1.2
        created.append(rail)
    for ob in created:
        handlers.rebuild(ob)
    for ob in smp.values():
        ob.hide_set(True)
    return created
