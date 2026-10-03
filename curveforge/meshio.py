# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender mesh <-> MeshBuf conversion, slicing and welding."""

import bmesh
import bpy
import numpy as np

from .engine.mesh import ATTRIBUTE_TYPES, MeshBuf

VALUE_KEY = {"FLOAT": "value", "INT": "value", "INT8": "value", "BOOLEAN": "value", "INT32_2D": "value",
             "QUATERNION": "value", "FLOAT_VECTOR": "vector", "FLOAT2": "vector", "FLOAT_COLOR": "color",
             "BYTE_COLOR": "color", "FLOAT4X4": "value"}
SEAM_NAMES = ("uv_seam", ".uv_seam")
SEAM_WRITE = "uv_seam" if bpy.app.version >= (5, 0, 0) else ".uv_seam"


def _read(me, name, key, out, fallback):
    at = me.attributes.get(name)
    if at is not None and len(at.data) and out.size % len(at.data) == 0:
        try:
            at.data.foreach_get(key, out)
            return out
        except (RuntimeError, TypeError):
            pass
    fallback(out)
    return out


def _write(me, name, key, data, fallback):
    at = me.attributes.get(name)
    if at is not None:
        try:
            at.data.foreach_set(key, data)
            return
        except (RuntimeError, TypeError):
            pass
    fallback(data)


def read_mesh(me, materials=None):
    """``bpy.types.Mesh`` -> :class:`MeshBuf`."""
    nv, ne, nc, nf = len(me.vertices), len(me.edges), len(me.loops), len(me.polygons)
    co = np.zeros(nv * 3, np.float32)
    if nv:
        _read(me, "position", "vector", co, lambda o: me.vertices.foreach_get("co", o))
    ed = np.zeros(ne * 2, np.int32)
    if ne:
        _read(me, ".edge_verts", "value", ed, lambda o: me.edges.foreach_get("vertices", o))
    cv = np.zeros(nc, np.int32)
    ce = np.zeros(nc, np.int32)
    if nc:
        _read(me, ".corner_vert", "value", cv, lambda o: me.loops.foreach_get("vertex_index", o))
        _read(me, ".corner_edge", "value", ce, lambda o: me.loops.foreach_get("edge_index", o))
    starts = np.zeros(nf, np.int32)
    if nf:
        me.polygons.foreach_get("loop_start", starts)
    offsets = np.append(starts, nc).astype(np.int32)
    fm = np.zeros(nf, np.int32)
    if nf and me.attributes.get("material_index") is not None:
        _read(me, "material_index", "value", fm, lambda o: me.polygons.foreach_get("material_index", o))
    uvs, uv_names = [], set()
    for layer in me.uv_layers:
        arr = np.zeros(nc * 2, np.float32)
        if nc:
            _read(me, layer.name, "vector", arr, lambda o, lay=layer: lay.data.foreach_get("uv", o))
        uvs.append((layer.name, arr.reshape(-1, 2)))
        uv_names.add(layer.name)
    attrs, seams = {}, None
    for at in me.attributes:
        name = at.name
        if name in SEAM_NAMES and at.domain == "EDGE" and ne:
            s = np.zeros(ne, bool)
            at.data.foreach_get("value", s)
            seams = s if s.any() else None
            continue
        if name.startswith(".") or name in {"position", "material_index"} or name in uv_names:
            continue
        if name in {"cf_is_instance", "cf_instance", "cf_matrix"}:
            continue
        if at.data_type not in ATTRIBUTE_TYPES or at.domain not in {"POINT", "EDGE", "FACE", "CORNER"}:
            continue
        npdt, comps = ATTRIBUTE_TYPES[at.data_type]
        arr = np.zeros(len(at.data) * comps, npdt)
        try:
            at.data.foreach_get(VALUE_KEY[at.data_type], arr)
        except (RuntimeError, TypeError, AttributeError):
            continue
        attrs[name] = (at.domain, at.data_type, arr.reshape(-1, comps) if comps > 1 else arr)
    return MeshBuf(co.reshape(-1, 3), cv, ce, offsets, ed.reshape(-1, 2), fm, materials or [None], uvs,
                   attrs, seams)


def write_mesh(me, mesh):
    """Replace the geometry of ``me`` with an assembled :class:`engine.mesh.Mesh`."""
    me.clear_geometry()
    me.materials.clear()
    if any(m is not None for m in mesh.materials):
        for m in mesh.materials:
            me.materials.append(m)
    nv, ne, nc, nf = len(mesh.co), len(mesh.edges), len(mesh.corner_vert), len(mesh.face_start)
    if nv == 0:
        me.update()
        return
    me.vertices.add(nv)
    _write(me, "position", "vector", np.ascontiguousarray(mesh.co, np.float32).ravel(),
           lambda d: me.vertices.foreach_set("co", d))
    if ne:
        me.edges.add(ne)
        _write(me, ".edge_verts", "value", np.ascontiguousarray(mesh.edges, np.int32).ravel(),
               lambda d: me.edges.foreach_set("vertices", d))
    if nc:
        me.loops.add(nc)
        _write(me, ".corner_vert", "value", np.ascontiguousarray(mesh.corner_vert, np.int32),
               lambda d: me.loops.foreach_set("vertex_index", d))
        _write(me, ".corner_edge", "value", np.ascontiguousarray(mesh.corner_edge, np.int32),
               lambda d: me.loops.foreach_set("edge_index", d))
    if nf:
        me.polygons.add(nf)
        me.polygons.foreach_set("loop_start", np.ascontiguousarray(mesh.face_start, np.int32))
        fm = np.ascontiguousarray(mesh.face_mat, np.int32)
        if fm.any():
            if me.attributes.get("material_index") is None:
                me.attributes.new("material_index", "INT", "FACE")
            _write(me, "material_index", "value", fm, lambda d: me.polygons.foreach_set("material_index", d))
    for name, arr in mesh.uvs:
        layer = me.uv_layers.new(name=name, do_init=False)
        if layer is not None and nc:
            _write(me, layer.name, "vector", np.ascontiguousarray(arr, np.float32).ravel(),
                   lambda d, lay=layer: lay.data.foreach_set("uv", d))
    for name, (dom, dt, arr) in mesh.attrs.items():
        if name in me.attributes:
            continue
        try:
            at = me.attributes.new(name, dt, dom)
        except (RuntimeError, TypeError):
            continue
        if at is not None and len(at.data):
            try:
                at.data.foreach_set(VALUE_KEY[dt], np.ascontiguousarray(arr).ravel())
            except (RuntimeError, TypeError):
                pass
    if mesh.seams is not None and ne:
        if me.attributes.get(SEAM_WRITE) is None:
            try:
                me.attributes.new(SEAM_WRITE, "BOOLEAN", "EDGE")
            except (RuntimeError, TypeError):
                pass
        _write(me, SEAM_WRITE, "value", np.ascontiguousarray(mesh.seams, bool),
               lambda d: me.edges.foreach_set("use_seam", d))
    me.update()


def _buf_to_temp_mesh(buf):
    from .engine.mesh import Assembler
    asm = Assembler()
    asm.add(buf, buf.co[None])
    me = bpy.data.meshes.new(".cf_temp")
    write_mesh(me, asm.build())
    return me


def slicer(buf, plane_co, plane_no):
    """Keep the part of ``buf`` on the positive side of the plane (bmesh bisect)."""
    me = _buf_to_temp_mesh(buf)
    try:
        bm = bmesh.new()
        try:
            bm.from_mesh(me)
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                                   plane_co=tuple(plane_co), plane_no=tuple(plane_no), clear_inner=True)
            bm.to_mesh(me)
        finally:
            bm.free()
        return read_mesh(me, buf.materials)
    finally:
        bpy.data.meshes.remove(me)


def weld(me, distance):
    """Merge vertices of ``me`` closer than ``distance``."""
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=distance)
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()


_cache = {}


def clear_cache():
    _cache.clear()


def invalidate(keys):
    for k in list(_cache):
        if k[0] in keys:
            del _cache[k]


def object_buf(obj, depsgraph, use_transform=True, world=False):
    """Evaluated geometry of ``obj`` (modifiers applied) as a cached MeshBuf."""
    key = (obj.session_uid, bool(use_transform), bool(world))
    buf = _cache.get(key)
    if buf is not None:
        return buf
    ev = obj
    if depsgraph is not None:
        try:
            ev = obj.evaluated_get(depsgraph)
        except (ReferenceError, RuntimeError):
            ev = obj
    try:
        me = ev.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph) if depsgraph else ev.to_mesh()
    except RuntimeError:
        me = None
    if me is None:
        return None
    try:
        mats = [s.material.original if s.material is not None else None for s in ev.material_slots] or [None]
        buf = read_mesh(me, mats)
    finally:
        ev.to_mesh_clear()
    m = np.array(obj.matrix_world, dtype=np.float64)
    if world:
        buf = _transform(buf, m)
    elif use_transform:
        m3 = m[:3, :3]
        if not np.allclose(m3, np.eye(3)):
            m4 = np.eye(4)
            m4[:3, :3] = m3
            buf = _transform(buf, m4)
    _cache[key] = buf
    return buf


def _transform(buf, m):
    co = buf.co @ m[:3, :3].T + m[:3, 3]
    out = buf.with_co(co)
    if np.linalg.det(m[:3, :3]) < 0:
        pv, pe = buf.flip_perms()
        out.corner_vert = buf.corner_vert[pv]
        out.corner_edge = buf.corner_edge[pe]
        out.uvs = [(n, a[pv]) for n, a in buf.uvs]
        out.attrs = {k: (d, t, a[pv] if d == "CORNER" else a) for k, (d, t, a) in buf.attrs.items()}
        out._flip = None
    return out
