# SPDX-License-Identifier: GPL-3.0-or-later
"""Writes assembled arrays into a Blender mesh datablock."""

import numpy as np


def write(me, asm):
    """Replace the geometry of ``me`` with the :class:`Assembled` arrays."""
    from .samples import attr_value_key

    me.clear_geometry()
    mats = me.materials
    mats.clear()
    if any(mat is not None for mat in asm.materials):
        for mat in asm.materials:
            mats.append(mat)

    nv, ne, nl, nf = len(asm.co), len(asm.edges), len(asm.loop_vert), len(asm.face_start)
    if nv == 0:
        me.update()
        return
    me.vertices.add(nv)
    me.vertices.foreach_set("co", np.ascontiguousarray(asm.co, dtype=np.float32).ravel())
    if ne:
        me.edges.add(ne)
        me.edges.foreach_set("vertices", np.ascontiguousarray(asm.edges, dtype=np.int32).ravel())
    if nl:
        me.loops.add(nl)
        me.loops.foreach_set("vertex_index", np.ascontiguousarray(asm.loop_vert, dtype=np.int32))
        me.loops.foreach_set("edge_index", np.ascontiguousarray(asm.loop_edge, dtype=np.int32))
    if nf:
        me.polygons.add(nf)
        me.polygons.foreach_set("loop_start", np.ascontiguousarray(asm.face_start, dtype=np.int32))
        me.polygons.foreach_set("material_index", np.ascontiguousarray(asm.mat_index, dtype=np.int32))

    for name, arr in asm.uvs:
        layer = me.uv_layers.new(name=name, do_init=False)
        if layer is not None and nl:
            layer.data.foreach_set("uv", np.ascontiguousarray(arr, dtype=np.float32).ravel())

    for name, (dom, dt, arr) in asm.attrs.items():
        if name in me.attributes:
            continue
        try:
            at = me.attributes.new(name, dt, dom)
        except (RuntimeError, TypeError):
            continue
        if at is None or len(at.data) == 0:
            continue
        try:
            at.data.foreach_set(attr_value_key(dt), np.ascontiguousarray(arr).ravel())
        except (RuntimeError, TypeError):
            pass

    if asm.seams is not None and ne:
        me.edges.foreach_set("use_seam", np.ascontiguousarray(asm.seams, dtype=bool))
    me.update()
