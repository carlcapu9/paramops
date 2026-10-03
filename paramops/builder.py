# SPDX-License-Identifier: GPL-3.0-or-later
"""Builds the scatter mesh of one scatter object."""

import re
import time
import traceback

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import curves, samples, writer
from .core import pathmath as pm
from .core.layout import LayoutSettings, PathInfo, SlotInfo, VariantInfo, layout_path
from .core.meshdata import MeshData, assemble, merge
from .core.placement import SlotConfig, euler_matrix, place_group, random_uv_offsets
from .props import iter_slots, slot_objects

POINT_SLOTS = {"evenly", "corner"}

_bvh_cache = {}


def clear_caches():
    _bvh_cache.clear()


def invalidate(keys):
    for k in list(_bvh_cache):
        if k in keys:
            del _bvh_cache[k]


class ResolvedSlot:
    __slots__ = ("info", "cfg", "variants")

    def __init__(self, info, cfg, variants):
        self.info = info
        self.cfg = cfg
        self.variants = variants


def _static_matrix(slot):
    rx, ry, rz = slot.rotation
    sx, sy, sz = slot.scale
    if slot.mirror_x:
        sx = -sx
    if slot.mirror_y:
        sy = -sy
    return euler_matrix(rx, ry, rz) @ np.diag((sx, sy, sz))


def resolve_slot(key, slot, st, depsgraph):
    variants = []
    weights = []
    pick = "SINGLE"
    if slot.enabled:
        objs = slot_objects(slot)
        if slot.source == "COLLECTION" and slot.collection is not None and slot.collection_mode == "COMBINE":
            parts = [samples.object_mesh_world(o, depsgraph) for o in objs]
            parts = [p for p in parts if p is not None and p.nv]
            if parts:
                md = merge(parts)
                off = np.array(slot.collection.instance_offset, dtype=np.float64)
                md = md.copy_with(co=md.co - off)
                variants.append(md)
                weights.append(1.0)
        else:
            if slot.source == "COLLECTION":
                pick = slot.collection_mode
            for o in objs:
                md = samples.object_mesh(o, depsgraph, st.apply_sample_transform)
                if md is not None and md.nv:
                    variants.append(md)
                    weights.append(o.paramops_weight)
    m = _static_matrix(slot)
    if not np.allclose(m, np.eye(3)):
        variants = [md.transformed(m) for md in variants]
    if st.preview == "BOXES":
        variants = [MeshData.box(*md.bbox(), material=md.materials[0]) for md in variants]
    infos = [VariantInfo(md.bbox()[0][0], md.bbox()[1][0]) for md in variants]
    info = SlotInfo(key, active=slot.enabled, variants=infos, pick=pick, weights=weights,
                    pad_before=slot.padding_before, pad_after=slot.padding_after,
                    align_x=slot.align_x, seed=slot.seed)
    off = tuple(slot.offset)
    cfg = SlotConfig(
        deform=slot.deform, vertical=slot.keep_vertical, align_x=slot.align_x,
        align_y=slot.align_y, align_z=slot.align_z,
        offset=(off[0], off[1] + st.offset_y, off[2] + st.offset_z),
        random_offset=tuple(slot.random_offset), random_rotation=tuple(slot.random_rotation),
        random_scale=slot.random_scale, random_flip_x=slot.random_flip_x,
        random_flip_y=slot.random_flip_y, random_uv=tuple(slot.random_uv),
        flat_top=slot.flat_top, flat_bottom=slot.flat_bottom,
        flat_center=slot.flat_center, flat_reference=slot.flat_reference, orient=slot.orient,
        seed=slot.seed)
    return ResolvedSlot(info, cfg, variants)


def _lateral_extent(rs):
    """Largest distance of deformed geometry from the curve (for miter zones)."""
    ext = 0.0
    for md in rs.variants:
        if not rs.cfg.deform:
            continue
        bmin, bmax = md.bbox()
        cfg = rs.cfg
        refy = {"CENTER": (bmin[1] + bmax[1]) * 0.5, "LEFT": bmin[1], "RIGHT": bmax[1]}.get(cfg.align_y, 0.0)
        refz = {"BOTTOM": bmin[2], "CENTER": (bmin[2] + bmax[2]) * 0.5, "TOP": bmax[2]}.get(cfg.align_z, 0.0)
        ys = (abs(bmin[1] - refy + cfg.offset[1]), abs(bmax[1] - refy + cfg.offset[1]))
        zs = (abs(bmin[2] - refz + cfg.offset[2]), abs(bmax[2] - refz + cfg.offset[2]))
        ext = max(ext, max(ys), max(zs))
    return float(ext)


_NUM = re.compile(r"^\s*(-?\d+(?:\.\d*)?|-?\.\d+)\s*(%?)\s*$")


def parse_positions(text, length):
    out = []
    for tok in re.split(r"[,;\s]+", text or ""):
        if not tok:
            continue
        m = _NUM.match(tok)
        if not m:
            continue
        v = float(m.group(1))
        if m.group(2):
            s = length * v / 100.0
            if v < 0:
                s = length + s
        else:
            s = v if v >= 0 else length + v
        if -1e-9 <= s <= length + 1e-9:
            out.append(min(max(s, 0.0), length))
    return out


def _get_bvh(target, depsgraph):
    key = samples.object_key(target)
    entry = _bvh_cache.get(key)
    if entry is None:
        entry = BVHTree.FromObject(target, depsgraph)
        _bvh_cache[key] = entry
    return entry


def project_polyline(poly, target, depsgraph, step, height):
    poly = pm.densify_polyline(poly, step)
    bvh = _get_bvh(target, depsgraph)
    if bvh is None:
        return poly
    mw = target.matrix_world
    inv = mw.inverted()
    corners = [mw @ Vector(c) for c in target.bound_box]
    top = max(c.z for c in corners) + 1.0
    d_local = (inv.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    pts = poly.pts.copy()
    for i, p in enumerate(pts):
        origin = inv @ Vector((p[0], p[1], max(top, p[2] + 1.0)))
        hit, _n, _idx, _d = bvh.ray_cast(origin, d_local)
        if hit is not None:
            pts[i, 2] = (mw @ hit).z + height
    return pm.Polyline(pts, poly.tilt, poly.u, poly.cyclic, poly.n_ctrl, poly.corner_angle)


def _segment_bounds(path, n_seg, cyclic, ids):
    L = path.L
    bounds = []
    for k in range(n_seg):
        sid = ids[k] if k < len(ids) else -1
        sa = float(path.s_of_u(k))
        sb = float(path.s_of_u(k + 1))
        sm = float(path.s_of_u(k + 0.5))
        lo, hi = min(sa, sb), max(sa, sb)
        if cyclic and not (lo - 1e-9 <= sm <= hi + 1e-9):
            if L - hi > 1e-9:
                bounds.append((hi, L, sid))
            if lo > 1e-9:
                bounds.append((0.0, lo, sid))
        elif hi - lo > 1e-9:
            bounds.append((lo, hi, sid))
        elif cyclic and n_seg == 1:
            bounds.append((0.0, L, sid))
    return bounds


def _layout_settings(st, length):
    if st.clip_mode == "PERCENT":
        cs = length * st.clip_start_pct / 100.0
        ce = length * st.clip_end_pct / 100.0
    else:
        cs, ce = st.clip_start, st.clip_end
    return LayoutSettings(
        clip_start=cs, clip_end=ce, fit_mode=st.fit_mode, fit_count=st.fit_count, spacing=st.spacing,
        fixed_align=st.fixed_align, fixed_remainder=st.fixed_remainder,
        evenly_mode=st.evenly_mode, evenly_spacing=st.evenly_spacing, evenly_count=st.evenly_count,
        evenly_min_gap=st.evenly_min_gap, evenly_align=st.evenly_align,
        corner_slide=st.corner_slide, seed=st.seed, max_modules=st.max_modules)


def build(obj, depsgraph=None):
    """Rebuild the mesh of scatter ``obj``. Returns True on success."""
    st = obj.paramops
    t0 = time.perf_counter()
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
    try:
        try:
            groups, n_modules, truncated = _compute(obj, depsgraph)
        except _NoPath as exc:
            groups, n_modules, truncated = [], 0, False
            st.last_error = str(exc)
        asm = assemble(groups)
        if obj.data.users > 1 and not obj.data.library:
            obj.data = obj.data.copy()  # linked duplicates must not overwrite each other
        writer.write(obj.data, asm)
        st.stat_modules = n_modules
        st.stat_verts = len(asm.co)
        st.stat_faces = len(asm.face_start)
        st.stat_truncated = truncated
        if st.last_error and st.path is not None:
            st.last_error = ""
        ok = True
    except Exception as exc:  # keep Blender responsive; report in the panel
        traceback.print_exc()
        st.last_error = "%s: %s" % (type(exc).__name__, exc)
        ok = False
    st.stat_time = time.perf_counter() - t0
    return ok


def sync_refs(st, curve_obj):
    """Keep Segment ID / Marker assignments attached to their points after edits."""
    if not st.segment_ids and not st.markers:
        st.topo_signature = curves.topology_signature(curve_obj)
        return
    pts = curves.local_points(curve_obj)
    cyc = [sp.use_cyclic_u for sp in curve_obj.data.splines]
    sig = curves.topology_signature(curve_obj)
    topo_changed = sig != st.topo_signature
    tol = 1e-5

    def find(co, prefer):
        co = np.asarray(co, dtype=np.float64)
        order = [prefer] + [i for i in range(len(pts)) if i != prefer] if 0 <= prefer < len(pts) else range(len(pts))
        for si in order:
            arr = pts[si]
            if not len(arr):
                continue
            d = np.linalg.norm(arr - co, axis=1)
            j = int(np.argmin(d))
            if d[j] <= tol:
                return si, j
        return None

    def valid(si, pi):
        return 0 <= si < len(pts) and 0 <= pi < len(pts[si])

    for marker in st.markers:
        drop = []
        for r_i, ref in enumerate(marker.refs):
            si, pi = ref.spline, ref.index
            if valid(si, pi) and np.linalg.norm(pts[si][pi] - np.array(ref.co)) <= tol:
                continue
            hit = find(ref.co, si)
            if hit is not None:
                ref.spline, ref.index = hit
                continue
            if not topo_changed and valid(si, pi):
                ref.co = pts[si][pi]  # the point was moved
                continue
            drop.append(r_i)
        for r_i in reversed(drop):
            marker.refs.remove(r_i)

    for item in st.segment_ids:
        drop = []
        add = []
        for r_i, ref in enumerate(item.refs):
            si, i = ref.spline, ref.index
            if valid(si, i):
                n = len(pts[si])
                j = (i + 1) % n
                if (np.linalg.norm(pts[si][i] - np.array(ref.co_a)) <= tol
                        and np.linalg.norm(pts[si][j] - np.array(ref.co_b)) <= tol):
                    continue
            ha, hb = find(ref.co_a, si), find(ref.co_b, si)
            if ha is not None and hb is not None and ha[0] == hb[0]:
                s2, a = ha
                b = hb[1]
                n = len(pts[s2])
                if b == (a + 1) % n:
                    ref.spline, ref.index = s2, a
                    continue
                lo, hi = (a, b) if a < b else (b, a)
                segs = list(range(lo, hi))
                if segs and len(segs) < n:
                    ref.spline, ref.index = s2, segs[0]
                    ref.co_a, ref.co_b = pts[s2][segs[0]], pts[s2][(segs[0] + 1) % n]
                    add.extend((s2, k) for k in segs[1:])
                    continue
            if not topo_changed and valid(si, i):
                n = len(pts[si])
                if i < n - 1 or cyc[si]:
                    ref.co_a, ref.co_b = pts[si][i], pts[si][(i + 1) % n]
                    continue
            drop.append(r_i)
        for r_i in reversed(drop):
            item.refs.remove(r_i)
        for s2, k in add:
            n = len(pts[s2])
            ref = item.refs.add()
            ref.spline, ref.index = s2, k
            ref.co_a, ref.co_b = pts[s2][k], pts[s2][(k + 1) % n]
    if st.topo_signature != sig:
        st.topo_signature = sig


def _compute(obj, depsgraph):
    st = obj.paramops
    curve_obj = st.path
    if curve_obj is None or curve_obj.type != "CURVE":
        raise _NoPath()
    sync_refs(st, curve_obj)
    splines = curves.read_splines(curve_obj)

    resolved = {key: resolve_slot(key, slot, st, depsgraph) for key, slot in iter_slots(st)}
    slot_infos = {key: rs.info for key, rs in resolved.items()}
    extent = max([_lateral_extent(rs) for rs in resolved.values()] + [0.0])
    inv = np.array(obj.matrix_world.inverted(), dtype=np.float64)

    seg_ids = {}
    for j, item in enumerate(st.segment_ids):
        if not item.slot.enabled:
            continue  # a disabled ID falls back to the Default sample
        for ref in item.refs:
            seg_ids.setdefault(ref.spline, {})[ref.index] = j
    point_markers = {}
    for j, item in enumerate(st.markers):
        for ref in item.refs:
            point_markers.setdefault(ref.spline, []).append((ref.index, "marker:%d" % j))

    groups = []
    n_modules = 0
    truncated = False
    slice_cache = {}
    for si, spin in enumerate(splines):
        if spin is None:
            continue
        poly = pm.sample_spline(spin, st.resolution)
        if poly is None:
            continue
        if st.reverse:
            poly = pm.reverse_polyline(poly)
        corner_ks = pm.corner_indices(poly, st.corner_threshold)
        if st.corner_radius > 0.0 and corner_ks:
            poly, rounded = pm.fillet_polyline(poly, corner_ks, st.corner_radius)
            corner_ks = [k for k in corner_ks if k not in rounded]
        if st.project and st.project_target is not None:
            poly = project_polyline(poly, st.project_target, depsgraph, st.project_step, st.project_offset)
        path = pm.Path(poly, twist=st.twist_mode, use_tilt=st.use_tilt, miter_extent=extent)
        L = path.L
        if L <= 1e-6:
            continue

        ids_map = seg_ids.get(si, {})
        n_seg = poly.n_segments
        ids = [ids_map.get(k, -1) for k in range(n_seg)]
        bounds = _segment_bounds(path, n_seg, poly.cyclic, ids) if ids_map else []
        markers = [(float(path.s_of_u(idx)), key) for idx, key in point_markers.get(si, [])
                   if 0 <= idx < poly.n_ctrl]
        for j, item in enumerate(st.markers):
            for s in parse_positions(item.positions, L):
                markers.append((s, "marker:%d" % j))
        info = PathInfo(L, poly.cyclic, corners=[float(path.s_of_u(k)) for k in corner_ks],
                        seg_bounds=bounds, markers=markers, index=si)
        res = layout_path(info, slot_infos, _layout_settings(st, L))
        truncated = truncated or res.truncated
        n_modules += len(res.placements)

        grouped = {}
        for p in res.placements:
            gk = (p.slot, p.variant, p.slice_lo, p.slice_hi)
            grouped.setdefault(gk, []).append(p)
        for (key, variant, lo, hi), plist in grouped.items():
            rs = resolved[key]
            md = rs.variants[variant]
            ref_bbox = None
            if lo is not None or hi is not None:
                ck = (key, variant, round(lo or 0.0, 5), round(hi or 1.0, 5))
                sliced = slice_cache.get(ck)
                if sliced is None:
                    sliced = samples.slice_data(md, lo, hi)
                    slice_cache[ck] = sliced
                ref_bbox = md.bbox()
                md = sliced
            if md.nv == 0:
                continue
            pos, flips = place_group(path, md, plist, rs.cfg, st.seed, si, inv, ref_bbox)
            groups.append((md, pos, flips, random_uv_offsets(rs.cfg, plist, st.seed, si)))
        if n_modules >= st.max_modules:
            truncated = True
            break
    return groups, n_modules, truncated


class _NoPath(Exception):
    def __str__(self):
        return "Assign a curve to scatter along"
