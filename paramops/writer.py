# SPDX-License-Identifier: GPL-3.0-or-later
"""Writes assembled arrays into a Blender mesh datablock.

Geometry goes through the generic attribute API (``position``,
``.edge_verts``, ``.corner_vert`` ...), which is many times faster than the
legacy ``vertices`` / ``edges`` / ``loops`` collections and keeps live editing
smooth on large scatters.
"""

import bpy
import numpy as np

SEAM_ATTR = "uv_seam" if bpy.app.version >= (5, 0, 0) else ".uv_seam"


def _set_attr(me, name, key, data, fallback=None):
    attr = me.attributes.get(name)
    if attr is not None:
        try:
            attr.data.foreach_set(key, data)
            return
        except (RuntimeError, TypeError):
            pass
    if fallback is not None:
        fallback(data)


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
    co = np.ascontiguousarray(asm.co, dtype=np.float32).ravel()
    _set_attr(me, "position", "vector", co, lambda d: me.vertices.foreach_set("co", d))
    if ne:
        me.edges.add(ne)
        ed = np.ascontiguousarray(asm.edges, dtype=np.int32).ravel()
        _set_attr(me, ".edge_verts", "value", ed, lambda d: me.edges.foreach_set("vertices", d))
    if nl:
        me.loops.add(nl)
        lv = np.ascontiguousarray(asm.loop_vert, dtype=np.int32)
        le = np.ascontiguousarray(asm.loop_edge, dtype=np.int32)
        _set_attr(me, ".corner_vert", "value", lv, lambda d: me.loops.foreach_set("vertex_index", d))
        _set_attr(me, ".corner_edge", "value", le, lambda d: me.loops.foreach_set("edge_index", d))
    if nf:
        me.polygons.add(nf)
        me.polygons.foreach_set("loop_start", np.ascontiguousarray(asm.face_start, dtype=np.int32))
        mi = np.ascontiguousarray(asm.mat_index, dtype=np.int32)
        if mi.any():
            if me.attributes.get("material_index") is None:
                me.attributes.new("material_index", "INT", "FACE")
            _set_attr(me, "material_index", "value", mi,
                      lambda d: me.polygons.foreach_set("material_index", d))

    for name, arr in asm.uvs:
        layer = me.uv_layers.new(name=name, do_init=False)
        if layer is not None and nl:
            data = np.ascontiguousarray(arr, dtype=np.float32).ravel()
            _set_attr(me, layer.name, "vector", data, lambda d, lay=layer: lay.data.foreach_set("uv", d))

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
        seams = np.ascontiguousarray(asm.seams, dtype=bool)
        if me.attributes.get(SEAM_ATTR) is None:
            try:
                me.attributes.new(SEAM_ATTR, "BOOLEAN", "EDGE")
            except (RuntimeError, TypeError):
                pass
        _set_attr(me, SEAM_ATTR, "value", seams, lambda d: me.edges.foreach_set("use_seam", d))
    me.update()
