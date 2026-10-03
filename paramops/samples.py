# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading sample geometry from Blender objects (cached as numpy MeshData)."""

import bmesh
import bpy
import numpy as np

from .core.meshdata import ATTR_TYPES, MeshData

MESH_LIKE = {"MESH", "CURVE", "SURFACE", "FONT", "META"}

_VALUE_KEY = {
    "FLOAT": "value", "INT": "value", "INT8": "value", "BOOLEAN": "value",
    "INT32_2D": "value", "QUATERNION": "value",
    "FLOAT_VECTOR": "vector", "FLOAT2": "vector",
    "FLOAT_COLOR": "color", "BYTE_COLOR": "color",
}

_cache = {}


def object_key(obj):
    return obj.session_uid


def clear_cache():
    _cache.clear()


def invalidate(keys):
    if not keys:
        return
    for ck in list(_cache):
        if ck[0] in keys:
            del _cache[ck]


def attr_value_key(data_type):
    return _VALUE_KEY[data_type]


def _get_attr(me, name, key, out, fallback):
    attr = me.attributes.get(name)
    if attr is not None and len(attr.data) * (out.size // max(len(attr.data), 1)) == out.size:
        try:
            attr.data.foreach_get(key, out)
            return out
        except (RuntimeError, TypeError):
            pass
    fallback(out)
    return out


SEAM_NAMES = ("uv_seam", ".uv_seam")


def mesh_to_data(me, materials=None):
    """Convert a ``bpy.types.Mesh`` into :class:`MeshData` (fast attribute reads)."""
    nv, ne, nl, nf = len(me.vertices), len(me.edges), len(me.loops), len(me.polygons)
    co = np.empty(nv * 3, np.float32)
    if nv:
        _get_attr(me, "position", "vector", co, lambda o: me.vertices.foreach_get("co", o))
    edges = np.empty(ne * 2, np.int32)
    if ne:
        _get_attr(me, ".edge_verts", "value", edges, lambda o: me.edges.foreach_get("vertices", o))
    lv = np.empty(nl, np.int32)
    le = np.empty(nl, np.int32)
    if nl:
        _get_attr(me, ".corner_vert", "value", lv, lambda o: me.loops.foreach_get("vertex_index", o))
        _get_attr(me, ".corner_edge", "value", le, lambda o: me.loops.foreach_get("edge_index", o))
    fs = np.empty(nf, np.int32)
    if nf:
        me.polygons.foreach_get("loop_start", fs)
    fz = np.diff(np.append(fs, nl)).astype(np.int32) if nf else np.empty(0, np.int32)
    if nf and (fz <= 0).any():
        me.polygons.foreach_get("loop_total", fz)
    mi = np.zeros(nf, np.int32)
    if nf and me.attributes.get("material_index") is not None:
        _get_attr(me, "material_index", "value", mi, lambda o: me.polygons.foreach_get("material_index", o))

    uvs = []
    uv_names = set()
    for uv in me.uv_layers:
        arr = np.empty(nl * 2, np.float32)
        if nl:
            _get_attr(me, uv.name, "vector", arr, lambda o, lay=uv: lay.data.foreach_get("uv", o))
        uvs.append((uv.name, arr.reshape(-1, 2)))
        uv_names.add(uv.name)

    attrs = {}
    seams = None
    for at in me.attributes:
        name = at.name
        if name in SEAM_NAMES and at.domain == "EDGE" and ne:
            s = np.empty(ne, bool)
            at.data.foreach_get("value", s)
            if s.any():
                seams = s
            continue
        if name.startswith(".") or name in {"position", "material_index"} or name in uv_names:
            continue
        dt, dom = at.data_type, at.domain
        if dt not in ATTR_TYPES or dom not in {"POINT", "EDGE", "FACE", "CORNER"}:
            continue
        npdt, comps = ATTR_TYPES[dt]
        count = len(at.data)
        arr = np.empty(count * comps, npdt)
        try:
            at.data.foreach_get(_VALUE_KEY[dt], arr)
        except (TypeError, RuntimeError, AttributeError):
            continue
        attrs[name] = (dom, dt, arr.reshape(-1, comps) if comps > 1 else arr)
    return MeshData(co.reshape(-1, 3), lv, le, fs, fz, edges.reshape(-1, 2), mat_index=mi,
                    materials=materials or [None], uvs=uvs, attrs=attrs, seams=seams)


def _extract(obj, depsgraph):
    ev = obj
    if depsgraph is not None:
        try:
            ev = obj.evaluated_get(depsgraph)
        except (ReferenceError, RuntimeError):
            ev = obj
    try:
        if depsgraph is not None:
            me = ev.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        else:
            me = ev.to_mesh()
    except RuntimeError:
        me = None
    if me is None:
        return None
    try:
        # Evaluated objects hold evaluated copies of materials: keep the originals.
        mats = [slot.material.original if slot.material is not None else None
                for slot in ev.material_slots] or [None]
        return mesh_to_data(me, mats)
    finally:
        ev.to_mesh_clear()


def object_mesh(obj, depsgraph, apply_transform=True):
    """Evaluated geometry of ``obj`` in its local space (rotation/scale optionally applied)."""
    if obj is None or obj.type not in MESH_LIKE:
        return None
    key = (object_key(obj), bool(apply_transform))
    md = _cache.get(key)
    if md is not None:
        return md
    md = _extract(obj, depsgraph)
    if md is None:
        return None
    if apply_transform:
        m = np.array(obj.matrix_world.to_3x3(), dtype=np.float64)
        if not np.allclose(m, np.eye(3)):
            md = md.transformed(m)
    _cache[key] = md
    return md


def object_mesh_world(obj, depsgraph):
    """Evaluated geometry in world space (used to combine collections)."""
    md = object_mesh(obj, depsgraph, apply_transform=False)
    if md is None:
        return None
    return md.transformed(np.array(obj.matrix_world, dtype=np.float64))


def data_to_mesh(md, me):
    """Write MeshData into an existing (empty) Blender mesh."""
    from . import writer
    from .core.meshdata import assemble
    writer.write(me, assemble([(md, md.co[None], np.zeros(1, dtype=bool))]))


def slice_data(md, lo=None, hi=None):
    """Cut ``md`` along X at fractions ``lo`` / ``hi`` of its bounding box."""
    bmin, bmax = md.bbox()
    lenx = float(bmax[0] - bmin[0])
    if lenx <= 0.0:
        return md
    tmp = bpy.data.meshes.new(".paramops_slice")
    try:
        data_to_mesh(md, tmp)
        bm = bmesh.new()
        try:
            bm.from_mesh(tmp)
            if hi is not None:
                x = float(bmin[0]) + lenx * hi
                geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
                bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(x, 0.0, 0.0), plane_no=(1.0, 0.0, 0.0),
                                       clear_outer=True)
            if lo is not None:
                x = float(bmin[0]) + lenx * lo
                geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
                bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(x, 0.0, 0.0), plane_no=(1.0, 0.0, 0.0),
                                       clear_inner=True)
            bm.to_mesh(tmp)
        finally:
            bm.free()
        out = mesh_to_data(tmp, md.materials)
    finally:
        bpy.data.meshes.remove(tmp)
    return out
