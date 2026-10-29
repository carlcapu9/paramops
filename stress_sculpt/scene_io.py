"""Blender scene -> numpy: voxelization and gathering of the elastic problem.

Everything here reads Blender data, so it runs on the main thread only. The
conventions of ``core/grid.py`` apply (element centres at
``origin + (ijk + 0.5) * h``, ``dof = 3 * node + component``).

Object roles (``obj.stress_sculpt.role``; objects with ``enabled == False`` are
ignored, the domain object never has a role):

* ``SUPPORT`` -- active nodes inside the object's region are fixed along the
  enabled world axes (``fix_x`` / ``fix_y`` / ``fix_z``).
* ``LOAD`` -- ``magnitude`` newtons along the object's local +Z axis (in world
  space), shared equally by the active nodes within ``load_radius`` (0 means
  1.5 voxels) of the object's origin. Several loads add up.
* ``KEEP_SOLID`` / ``KEEP_VOID`` / ``THICKEN`` / ``HOLLOW`` / ``BRANCH`` -- zones:
  element centres inside the region are members.

Region of an empty: ``CUBE`` display -> local coordinates within
``+-empty_display_size`` on every axis; any other display -> local sphere of
radius ``empty_display_size`` (the full ``matrix_world`` is used, so scale and
rotation apply). Objects with geometry (meshes, curves, text, ...) use their
evaluated interior; other objects their local bounding box. When a support or
load touches no active node, the nearest active nodes are used (with a
warning).
"""

import hashlib
import importlib

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .core.grid import Grid, Problem
from .props import ZONE_ROLES

# Object types whose evaluated geometry converts to a triangle mesh.
GEOMETRY_TYPES = frozenset({"MESH", "CURVE", "SURFACE", "META", "FONT"})

_F32_EPS = float(np.finfo(np.float32).eps)
_MAX_HITS_PER_RAY = 100000
# Offsets (fractions of the jitter length) used to re-cast rays whose number of
# surface crossings came out odd (grazing an edge / vertex, open meshes).
_JITTER = ((0.0137, 0.0071), (-0.0093, 0.0119), (0.0061, -0.0157), (-0.0113, -0.0047))


# ------------------------------------------------------------------ core modules
def core_module(name: str):
    """Import ``stress_sculpt.core.<name>`` lazily, with a readable error if missing."""
    package = __package__ or "stress_sculpt"
    full_name = f"{package}.core.{name}"
    try:
        return importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        if exc.name == full_name:
            raise RuntimeError(
                f"Stress Sculpt is incomplete: the module 'core/{name}.py' is missing"
            ) from exc
        raise


# ------------------------------------------------------------------------ meshes
def evaluated_world_mesh(obj, depsgraph) -> tuple:
    """World-space triangles of ``obj`` with modifiers applied.

    Returns ``verts`` (V, 3) float64 and ``tris`` (T, 3) int64 built from the
    evaluated mesh's loop triangles. Objects without geometry give empty arrays.
    """
    empty = (np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64))
    if obj is None or obj.type not in GEOMETRY_TYPES:
        return empty
    obj_eval = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    try:
        mesh = obj_eval.to_mesh()
    except RuntimeError:
        return empty
    try:
        if mesh is None:
            return empty
        mesh.calc_loop_triangles()
        n_verts = len(mesh.vertices)
        n_tris = len(mesh.loop_triangles)
        co = np.empty(n_verts * 3, dtype=np.float32)
        tris = np.empty(n_tris * 3, dtype=np.int32)
        if n_verts:
            mesh.vertices.foreach_get("co", co)
        if n_tris:
            mesh.loop_triangles.foreach_get("vertices", tris)
        matrix = np.array(obj_eval.matrix_world, dtype=np.float64)
    finally:
        obj_eval.to_mesh_clear()
    verts = co.reshape(-1, 3).astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    return verts, tris.reshape(-1, 3).astype(np.int64)


def _mesh_bounds(verts, tris) -> tuple:
    """Bounding box of the vertices referenced by ``tris`` (all vertices if none)."""
    if tris.size:
        referenced = np.zeros(verts.shape[0], dtype=bool)
        referenced[tris.ravel()] = True
        used = verts[referenced] if not referenced.all() else verts
    else:
        used = verts
    return used.min(axis=0), used.max(axis=0)


def _bvh(verts, tris) -> BVHTree:
    return BVHTree.FromPolygons(verts.tolist(), tris.tolist(), all_triangles=True)


def _ray_span(bmin, bmax, length) -> tuple:
    """``(z_start, z_stop, step)`` for vertical rays crossing a mesh's bounding box.

    ``step`` (how far a ray is advanced past each hit) stays well above the
    float32 resolution of the BVH coordinates.
    """
    scale = float(max(np.abs(bmin).max(), np.abs(bmax).max(), (bmax - bmin).max(), 1e-12))
    step = max(1e-6 * length, 16.0 * _F32_EPS * scale)
    margin = max(length, 64.0 * step)
    return float(bmin[2] - margin), float(bmax[2] + margin), step


def _cast_columns(bvh, xy, z_start, z_stop, step, dedupe) -> tuple:
    """Every crossing of vertical (+Z) rays at the ``(x, y)`` columns.

    One BVH ray per crossing: the ray is re-cast from ``step`` beyond each hit,
    so a crossing through an edge or vertex shared by several triangles is
    reported once (hits closer than ``dedupe`` to the previous one are merged).
    Returns ``(z, column)`` arrays in column order, ascending z per column.
    """
    ray_cast = bvh.ray_cast
    up = Vector((0.0, 0.0, 1.0))
    zs, cols = [], []
    for column, (x, y) in enumerate(xy.tolist()):
        z = z_start
        last = -np.inf
        for _ in range(_MAX_HITS_PER_RAY):
            location, _normal, _index, dist = ray_cast(Vector((x, y, z)), up, z_stop - z)
            if location is None:
                break
            z_hit = z + dist
            if z_hit - last > dedupe:
                zs.append(z_hit)
                cols.append(column)
                last = z_hit
            z = max(z_hit, z) + step
            if z >= z_stop:
                break
    return np.asarray(zs, dtype=np.float64), np.asarray(cols, dtype=np.int64)


def _column_hits(bvh, xy, z_start, z_stop, step, jitter, dedupe) -> tuple:
    """Crossings of all columns with even counts enforced.

    Columns with an odd number of crossings are re-cast at a few tiny ``(x, y)``
    offsets (``jitter`` is the length the offsets are relative to); if none
    gives an even count the topmost crossing is dropped (a ray always ends
    outside a closed surface). Returns ``(z, column, n_ambiguous)`` sorted by
    column then z; ``n_ambiguous`` counts the columns that needed the fallback.
    """
    n_cols = xy.shape[0]
    zs, cols = _cast_columns(bvh, xy, z_start, z_stop, step, dedupe)
    odd = np.nonzero(np.bincount(cols, minlength=n_cols) & 1)[0]
    for dx, dy in _JITTER:
        if odd.size == 0:
            break
        shifted = xy[odd] + np.array([dx, dy]) * jitter
        z2, c2 = _cast_columns(bvh, shifted, z_start, z_stop, step, dedupe)
        even = (np.bincount(c2, minlength=odd.size) & 1) == 0
        if np.any(even):
            fixed_cols = odd[even]
            keep = ~np.isin(cols, fixed_cols)
            take = even[c2]
            zs = np.concatenate([zs[keep], z2[take]])
            cols = np.concatenate([cols[keep], odd[c2[take]]])
            odd = odd[~even]
    order = np.lexsort((zs, cols))
    zs, cols = zs[order], cols[order]
    n_ambiguous = int(odd.size)
    if n_ambiguous:
        last = np.searchsorted(cols, odd, side="right") - 1
        keep = np.ones(zs.shape[0], dtype=bool)
        keep[last] = False
        zs, cols = zs[keep], cols[keep]
    return zs, cols, n_ambiguous


def _voxelize(verts, tris, grid) -> tuple:
    """``voxelize_mesh`` plus the number of ambiguous ray columns."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    nx, ny, nz = grid.shape
    inside = np.zeros(grid.shape, dtype=bool)
    if tris.shape[0] == 0:
        return inside, 0
    bmin, bmax = _mesh_bounds(verts, tris)
    h = grid.h
    xc, yc, zc = (grid.origin[a] + (np.arange(grid.shape[a]) + 0.5) * h for a in range(3))
    ii = np.nonzero((xc >= bmin[0]) & (xc <= bmax[0]))[0]
    jj = np.nonzero((yc >= bmin[1]) & (yc <= bmax[1]))[0]
    if ii.size == 0 or jj.size == 0 or zc[-1] < bmin[2] or zc[0] > bmax[2]:
        return inside, 0

    xy = np.stack(np.meshgrid(xc[ii], yc[jj], indexing="ij"), axis=-1).reshape(-1, 2)
    z_start, z_stop, step = _ray_span(bmin, bmax, h)
    zs, cols, n_ambiguous = _column_hits(
        _bvh(verts, tris), xy, z_start, z_stop, step, jitter=h, dedupe=1e-7 * h
    )
    # A crossing at height z toggles every centre at or above it: centre k is
    # inside iff an odd number of crossings lie at or below it.
    first_above = np.searchsorted(zc, zs, side="left")
    toggles = np.bincount(cols * (nz + 1) + first_above, minlength=xy.shape[0] * (nz + 1))
    toggles = (toggles & 1).astype(np.uint8).reshape(xy.shape[0], nz + 1)
    parity = np.bitwise_xor.accumulate(toggles, axis=1)[:, :nz]
    inside[ii[:, None], jj[None, :], :] = parity.reshape(ii.size, jj.size, nz).astype(bool)
    return inside, n_ambiguous


def voxelize_mesh(verts, tris, grid) -> np.ndarray:
    """Boolean ``grid.shape`` mask: element centre inside a closed triangle mesh.

    Ray parity along +Z per (i, j) column with ``mathutils.bvhtree.BVHTree``
    (one ray per crossing, re-cast past each hit). A centre is inside when an
    odd number of crossings lie at or below it, so a centre exactly on an
    entering face is inside and one exactly on a leaving face is outside.
    Crossings through edges / vertices shared by several triangles count once;
    columns with an odd crossing count are re-cast slightly offset, and as a
    last resort lose their topmost crossing.
    """
    return _voxelize(verts, tris, grid)[0]


def points_in_mesh(verts, tris, points) -> np.ndarray:
    """``(P,)`` bool: which world points lie inside a closed triangle mesh.

    Same parity rule as ``voxelize_mesh``; points sharing an ``(x, y)`` (e.g. the
    nodes of a grid) share one ray column.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    out = np.zeros(pts.shape[0], dtype=bool)
    if tris.shape[0] == 0 or pts.shape[0] == 0:
        return out
    bmin, bmax = _mesh_bounds(verts, tris)
    candidates = np.nonzero(np.all((pts >= bmin) & (pts <= bmax), axis=1))[0]
    if candidates.size == 0:
        return out
    p = pts[candidates]
    xy, inverse = np.unique(p[:, :2], axis=0, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    size = float((bmax - bmin).max()) or 1.0
    z_start, z_stop, step = _ray_span(bmin, bmax, 1e-3 * size)
    zs, cols, _ = _column_hits(
        _bvh(verts, tris), xy, z_start, z_stop, step, jitter=1e-2 * size, dedupe=1e-10 * size
    )
    # Composite keys "column + fraction of the ray" make one sorted array for all columns.
    span = z_stop - z_start
    keys = cols + 0.5 * (zs - z_start) / span
    point_keys = inverse + 0.5 * (p[:, 2] - z_start) / span
    below = np.searchsorted(keys, point_keys, side="right") - np.searchsorted(
        keys, inverse.astype(np.float64), side="left"
    )
    out[candidates] = (below & 1).astype(bool)
    return out


def make_grid(domain_obj, depsgraph, resolution) -> Grid:
    """Grid over the world bounding box of the evaluated domain (``pad=0``)."""
    verts, tris = evaluated_world_mesh(domain_obj, depsgraph)
    if verts.shape[0] == 0:
        raise ValueError(f"The domain '{domain_obj.name}' has no geometry")
    bmin, bmax = _mesh_bounds(verts, tris)
    return Grid.from_bounds(bmin, bmax, int(resolution), pad=0)


# ----------------------------------------------------------------------- regions
def region_mask_points(obj, points) -> np.ndarray:
    """``(P,)`` bool: which world points lie in the region of an empty.

    ``CUBE`` display: local coordinates within ``+-empty_display_size`` on every
    axis; any other display: local sphere of radius ``empty_display_size``.
    Other objects use their local bounding box (``obj.bound_box``). Only the
    points inside the region's world-space bounding box are transformed.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(pts.shape[0], dtype=bool)
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    linear = matrix[:3, :3]
    if pts.shape[0] == 0 or abs(np.linalg.det(linear)) < 1e-30:  # flattened: no volume
        return out
    if obj.type == "EMPTY":
        size = float(obj.empty_display_size)
        if size <= 0.0:
            return out
        lo = np.full(3, -size)
        hi = np.full(3, size)
    else:
        corners = np.array([tuple(corner) for corner in obj.bound_box], dtype=np.float64)
        lo, hi = corners.min(axis=0), corners.max(axis=0)
    tol = 1e-6 * max(float((hi - lo).max()), 1e-12)  # matrix_world is float32
    lo, hi = lo - tol, hi + tol
    # World bounding box of the local box: centre +- |linear| @ half extents.
    centre = matrix[:3, 3] + linear @ (0.5 * (lo + hi))
    half = np.abs(linear) @ (0.5 * (hi - lo))
    candidates = np.nonzero(np.all(np.abs(pts - centre) <= half, axis=1))[0]
    if candidates.size == 0:
        return out
    inverse = np.linalg.inv(matrix)
    local = pts[candidates] @ inverse[:3, :3].T + inverse[:3, 3]
    if obj.type == "EMPTY" and obj.empty_display_type != "CUBE":
        out[candidates] = np.einsum("ij,ij->i", local, local) <= hi[0] * hi[0]
    else:
        out[candidates] = np.all((local >= lo) & (local <= hi), axis=1)
    return out


def _object_points_mask(obj, points, depsgraph) -> np.ndarray:
    """Region test for supports: evaluated interior for geometry, else the empty rule."""
    if obj.type in GEOMETRY_TYPES:
        verts, tris = evaluated_world_mesh(obj, depsgraph)
        if tris.shape[0]:
            return points_in_mesh(verts, tris, points)
    return region_mask_points(obj, points)


def _zone_mask(obj, grid, depsgraph, method, centres, warnings) -> np.ndarray:
    """Elements of ``grid`` whose centre lies in the zone ``obj``."""
    if obj.type in GEOMETRY_TYPES:
        verts, tris = evaluated_world_mesh(obj, depsgraph)
        if tris.shape[0]:
            if method == "BOUNDS":
                bmin, bmax = _mesh_bounds(verts, tris)
                return np.all((centres() >= bmin) & (centres() <= bmax), axis=-1)
            mask, n_ambiguous = _voxelize(verts, tris, grid)
            if n_ambiguous:
                warnings.append(
                    f"Zone '{obj.name}' may not be a closed mesh ({n_ambiguous} ambiguous rays)"
                )
            return mask
    return region_mask_points(obj, centres().reshape(-1, 3)).reshape(grid.shape)


def _node_coords(grid, node_ids) -> np.ndarray:
    ijk = np.stack(np.unravel_index(node_ids, grid.nodes_shape), axis=1)
    return grid.origin + ijk * grid.h


def _nearest_nodes(coords, node_ids, target, h) -> np.ndarray:
    """Nodes within (nearest distance + h/2) of ``target``."""
    dist = np.linalg.norm(coords - np.asarray(target, dtype=np.float64), axis=1)
    return node_ids[dist <= dist.min() + 0.5 * h]


# ----------------------------------------------------------------------- objects
def view_layer_objects(context) -> list:
    """Objects of the context's view layer (all objects of the scene without one).

    Blender syncs ``view_layer.objects`` lazily: objects linked by a script are
    missing until the next sync, so the evaluated depsgraph is requested first
    (which syncs the view layer and is needed by the callers anyway).
    """
    view_layer = getattr(context, "view_layer", None)
    if view_layer is None:
        return list(context.scene.objects)
    context.evaluated_depsgraph_get()
    return [obj for obj in view_layer.objects if obj is not None]


def role_objects(context, include_disabled=False, exclude=()) -> list:
    """Objects with a role (sorted by name), excluding the domain (and ``exclude``).

    Objects of collections excluded from the view layer (e.g. saved variants)
    are not part of the setup.
    """
    domain = context.scene.stress_sculpt.domain
    skipped = {obj.name for obj in exclude if obj is not None}
    found = []
    for obj in view_layer_objects(context):
        settings = getattr(obj, "stress_sculpt", None)
        if settings is None or settings.role == "NONE" or obj == domain:
            continue
        if obj.name in skipped:
            continue
        if settings.enabled or include_disabled:
            found.append(obj)
    return sorted(found, key=lambda o: o.name)


def validated_mesh_object(context, obj, what="domain"):
    """``obj`` checked to be a mesh of the current view layer (ValueError otherwise)."""
    if obj.type != "MESH":
        raise ValueError(f"The {what} '{obj.name}' is not a mesh")
    if obj.name not in {o.name for o in view_layer_objects(context)}:
        raise ValueError(f"The {what} '{obj.name}' is not in the current scene / view layer")
    return obj


def validated_domain(context, settings):
    """The scene's domain object, checked (raises ValueError with an actionable message)."""
    domain = settings.domain
    if domain is None:
        raise ValueError(
            "No design domain: pick a closed mesh in Setup > Domain, or start from a demo scene"
        )
    return validated_mesh_object(context, domain, "domain")


def validated_target(context, settings):
    """The Structure panel's object (``settings.target``), checked."""
    target = settings.target
    if target is None:
        raise ValueError(
            "No object: pick the mesh to work with in the Structure panel (or start from an "
            "example)"
        )
    return validated_mesh_object(context, target, "object")


# --------------------------------------------------------------- structure inputs
def structure_loads(context, exclude=()) -> list:
    """Enabled LOAD objects as dicts ``name, position (3,), force (3,), magnitude, radius``.

    The force is ``magnitude`` newtons along the object's world +Z axis (the
    arrow). Loads without force or direction are skipped.
    """
    loads = []
    for obj in role_objects(context, exclude=exclude):
        settings = obj.stress_sculpt
        if settings.role != "LOAD":
            continue
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        direction = matrix[:3, 2].copy()
        norm = float(np.linalg.norm(direction))
        magnitude = float(settings.magnitude)
        if magnitude <= 0.0 or norm < 1e-12:
            continue
        direction /= norm
        loads.append(
            {
                "name": obj.name,
                "position": matrix[:3, 3].copy(),
                "direction": direction,
                "force": magnitude * direction,
                "magnitude": magnitude,
                "radius": float(settings.load_radius),
            }
        )
    return loads


def structure_supports(context, exclude=()) -> list:
    """Enabled SUPPORT objects as dicts ``name, position (3,), fix (3,), object name``."""
    supports = []
    for obj in role_objects(context, exclude=exclude):
        settings = obj.stress_sculpt
        if settings.role != "SUPPORT":
            continue
        fix = (bool(settings.fix_x), bool(settings.fix_y), bool(settings.fix_z))
        if not any(fix):
            continue
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        supports.append(
            {
                "name": obj.name,
                "position": matrix[:3, 3].copy(),
                "fix": fix,
                "parent": obj.parent.name if obj.parent is not None else "",
            }
        )
    return supports


def underside_contacts(verts, tris, points, directions) -> tuple:
    """Where the structure carrying ``points`` meets the mesh ``verts`` / ``tris``.

    For every point, the line through it along its force ``direction`` is
    followed to the last surface it crosses (seen from far along ``direction``,
    the first hit): a weight on top of an object is carried from its underside.
    Points not over the mesh, or below it, keep their position. Returns
    ``(contacts (P, 3), moved (P,) bool)``.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    out = pts.copy()
    moved = np.zeros(pts.shape[0], dtype=bool)
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if tris.shape[0] == 0 or pts.shape[0] == 0:
        return out, moved
    bmin, bmax = _mesh_bounds(verts, tris)
    size = float(np.linalg.norm(bmax - bmin)) or 1.0
    bvh = _bvh(verts, tris)
    for i, (p, d) in enumerate(zip(pts, dirs)):
        norm = float(np.linalg.norm(d))
        if norm < 1e-12:
            continue
        d = d / norm
        reach = size + float(np.linalg.norm(p - 0.5 * (bmin + bmax))) + 1.0
        start = p + d * reach
        location, _normal, _index, _dist = bvh.ray_cast(
            Vector(start.tolist()), Vector((-d).tolist()), 2.0 * reach
        )
        if location is None:
            continue
        hit = np.array(location, dtype=np.float64)
        along = float(np.dot(hit - p, d))
        if along > 1e-6 * size:
            out[i] = hit
            moved[i] = True
        elif along > -1e-6 * size:
            out[i] = hit
    return out, moved


def voxelize_object(obj, depsgraph, grid) -> np.ndarray:
    """``voxelize_mesh`` of an object's evaluated world mesh on ``grid``."""
    verts, tris = evaluated_world_mesh(obj, depsgraph)
    return voxelize_mesh(verts, tris, grid)


def zone_fields(context, grid, domain_mask, exclude=(), method="RAY") -> tuple:
    """Zone masks of the enabled zone objects on ``grid``.

    Returns ``(passive_solid, passive_void, bias, branch, warnings)`` (``None``
    for absent fields), restricted to ``domain_mask``.
    """
    depsgraph = context.evaluated_depsgraph_get()
    warnings = []
    centres_cache = []

    def centres():
        if not centres_cache:
            centres_cache.append(grid.element_centers())
        return centres_cache[0]

    passive_solid = passive_void = bias = branch = None
    for obj in role_objects(context, exclude=exclude):
        role = obj.stress_sculpt.role
        if role not in ZONE_ROLES:
            continue
        member = _zone_mask(obj, grid, depsgraph, method, centres, warnings) & domain_mask
        if not member.any():
            continue
        strength = float(obj.stress_sculpt.strength)
        if role == "KEEP_SOLID":
            passive_solid = member if passive_solid is None else passive_solid | member
        elif role == "KEEP_VOID":
            passive_void = member if passive_void is None else passive_void | member
        elif role in {"THICKEN", "HOLLOW"}:
            if bias is None:
                bias = np.zeros(grid.shape, dtype=np.float64)
            bias[member] += strength if role == "THICKEN" else -strength
        else:
            if branch is None:
                branch = np.zeros(grid.shape, dtype=np.float64)
            branch[member] = np.maximum(branch[member], strength)
    return passive_solid, passive_void, bias, branch, warnings


# ------------------------------------------------------------------------ gather
def gather_problem(context, settings=None, domain=None) -> tuple:
    """Build the ``Problem`` of the current scene (main thread).

    ``domain`` overrides ``settings.domain`` (the Structure panel carves its
    target this way without touching the analysis domain); the override object
    never acts as a support / load / zone itself.

    Returns ``(problem, info)`` with ``info`` = ``{"warnings": [str],
    "loads": [(position, direction, magnitude)], "supports": [position],
    "seed_points": (k, 3), "domain_name": str, "n_elems": int,
    "bounds": (bmin, bmax), "ambiguous_rays": int}``. Seed points are domain
    element centres next to every load and support (streamline seeds).

    Raises ``ValueError`` with an actionable message when the setup cannot be
    solved (no domain, not a mesh, empty voxelization, no supports, no loads).
    Parts of the domain touching no support are removed (warning) and the
    warnings of ``fem.check_problem`` are appended.
    """
    scene = context.scene
    if settings is None:
        settings = scene.stress_sculpt
    fem = core_module("fem")
    warnings = []

    if domain is None:
        domain = validated_domain(context, settings)
    else:
        domain = validated_mesh_object(context, domain, "object")
    depsgraph = context.evaluated_depsgraph_get()
    verts, tris = evaluated_world_mesh(domain, depsgraph)
    if tris.shape[0] == 0:
        raise ValueError(f"The domain '{domain.name}' has no faces")
    bmin, bmax = _mesh_bounds(verts, tris)
    try:
        grid = Grid.from_bounds(bmin, bmax, int(settings.resolution), pad=0)
    except ValueError as exc:
        raise ValueError(f"The domain '{domain.name}' is degenerate: {exc}") from exc

    n_ambiguous = 0
    if settings.voxelize_method == "BOUNDS":
        domain_mask = np.ones(grid.shape, dtype=bool)
    else:
        domain_mask, n_ambiguous = _voxelize(verts, tris, grid)
        if not domain_mask.any():
            raise ValueError(
                f"No voxel lies inside '{domain.name}': the mesh must be closed (watertight) "
                "and thicker than one voxel. Raise the resolution or use Voxelize > Bounding Box"
            )
        if n_ambiguous:
            warnings.append(
                f"The domain mesh may not be closed: {n_ambiguous} ray(s) gave an ambiguous "
                "inside test"
            )

    h = grid.h
    active_nodes = np.nonzero(grid.nodes_of_elements(domain_mask))[0]
    node_xyz = _node_coords(grid, active_nodes)
    centres_cache = []

    def centres():
        if not centres_cache:
            centres_cache.append(grid.element_centers())
        return centres_cache[0]

    fixed = []
    forces = np.zeros(grid.n_dofs, dtype=np.float64)
    forces_xyz = forces.reshape(-1, 3)
    passive_solid = passive_void = bias = branch = None
    loads_info, supports_info, seeds = [], [], []

    if getattr(domain.stress_sculpt, "role", "NONE") != "NONE":
        warnings.append(f"The domain '{domain.name}' has a role: the role is ignored")

    for obj in role_objects(context, exclude=(domain,)):
        obj_settings = obj.stress_sculpt
        role = obj_settings.role
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        location = matrix[:3, 3].copy()

        if role == "SUPPORT":
            fix = (obj_settings.fix_x, obj_settings.fix_y, obj_settings.fix_z)
            axes = [axis for axis in range(3) if fix[axis]]
            if not axes:
                warnings.append(f"Support '{obj.name}' blocks no axis: ignored")
                continue
            inside = _object_points_mask(obj, node_xyz, depsgraph)
            nodes = active_nodes[inside]
            if nodes.size == 0:
                nodes = _nearest_nodes(node_xyz, active_nodes, location, h)
                warnings.append(
                    f"Support '{obj.name}' does not touch the domain: attached to the "
                    f"{nodes.size} nearest node(s)"
                )
            fixed.append((3 * nodes[:, None] + np.asarray(axes, dtype=np.int64)[None, :]).ravel())
            supports_info.append(location)
            seeds.append(_node_coords(grid, nodes).mean(axis=0))

        elif role == "LOAD":
            magnitude = float(obj_settings.magnitude)
            direction = matrix[:3, 2].copy()
            norm = float(np.linalg.norm(direction))
            if magnitude <= 0.0:
                warnings.append(f"Load '{obj.name}' has no force: ignored")
                continue
            if norm < 1e-12:
                warnings.append(f"Load '{obj.name}' has no direction (zero scale): ignored")
                continue
            direction /= norm
            radius = float(obj_settings.load_radius) if obj_settings.load_radius > 0.0 else 1.5 * h
            dist2 = np.einsum("ij,ij->i", node_xyz - location, node_xyz - location)
            nodes = active_nodes[dist2 <= radius * radius]
            if nodes.size == 0:
                nodes = _nearest_nodes(node_xyz, active_nodes, location, h)
                warnings.append(
                    f"Load '{obj.name}' is outside the domain: applied to the {nodes.size} "
                    "nearest node(s)"
                )
            forces_xyz[nodes] += magnitude * direction / nodes.size
            loads_info.append((location, direction, magnitude))
            seeds.append(_node_coords(grid, nodes).mean(axis=0))

        elif role in ZONE_ROLES:
            member = _zone_mask(obj, grid, depsgraph, settings.voxelize_method, centres, warnings)
            member &= domain_mask
            if not member.any():
                warnings.append(f"Zone '{obj.name}' does not overlap the domain: ignored")
                continue
            strength = float(obj_settings.strength)
            if role == "KEEP_SOLID":
                passive_solid = member if passive_solid is None else passive_solid | member
            elif role == "KEEP_VOID":
                passive_void = member if passive_void is None else passive_void | member
            elif role in {"THICKEN", "HOLLOW"}:
                if bias is None:
                    bias = np.zeros(grid.shape, dtype=np.float64)
                bias[member] += strength if role == "THICKEN" else -strength
            else:  # BRANCH
                if branch is None:
                    branch = np.zeros(grid.shape, dtype=np.float64)
                branch[member] = np.maximum(branch[member], strength)

    fixed_dofs = np.unique(np.concatenate(fixed)) if fixed else np.zeros(0, dtype=np.int64)
    if fixed_dofs.size == 0:
        raise ValueError(
            "No supports: add a Support (an empty touching the domain) so the shape cannot fly away"
        )
    if not np.any(forces):
        raise ValueError(
            "No loads: add a Load empty (its arrow is the force direction) with a force > 0"
        )

    problem = Problem(
        grid=grid,
        domain=domain_mask,
        fixed_dofs=fixed_dofs,
        forces=forces,
        passive_solid=passive_solid,
        passive_void=passive_void,
        bias=bias,
        branch=branch,
        E=float(settings.youngs_modulus) * 1e9,
        nu=float(settings.poisson_ratio),
    )

    total_force = float(np.abs(forces).sum())
    problem, n_removed = fem.remove_unsupported_components(problem)
    if n_removed:
        warnings.append(f"{n_removed} part(s) of the domain touch no support and were ignored")
    if not problem.domain.any():
        raise ValueError("No part of the domain touches a support")
    # Forces on nodes of removed parts carry nothing (fem already zeroes them).
    problem.forces.reshape(-1, 3)[~grid.nodes_of_elements(problem.domain)] = 0.0
    if not np.any(problem.forces):
        raise ValueError("All loads act on parts of the domain that touch no support")
    if float(np.abs(problem.forces).sum()) < total_force * (1.0 - 1e-9):
        warnings.append("Some loads act on parts without supports and were ignored")
    warnings.extend(fem.check_problem(problem))

    # Streamline seeds: the domain element centre closest to each load / support.
    seed_points = np.zeros((0, 3), dtype=np.float64)
    if seeds:
        element_xyz = grid.element_centers()[problem.domain]
        nearest = [
            int(np.argmin(np.einsum("ij,ij->i", element_xyz - target, element_xyz - target)))
            for target in np.asarray(seeds, dtype=np.float64)
        ]
        seed_points = element_xyz[nearest]

    info = {
        "warnings": warnings,
        "loads": loads_info,
        "supports": supports_info,
        "seed_points": seed_points,
        "domain_name": domain.name,
        "n_elems": int(problem.domain.sum()),
        "bounds": (bmin, bmax),
        "ambiguous_rays": n_ambiguous,
    }
    return problem, info


# --------------------------------------------------------------------- signature
def setup_signature(context, settings=None) -> str:
    """Hash of everything that affects the ``Problem`` (cache invalidation).

    Covers the evaluated domain mesh in world space, resolution, voxelization
    method, material (E, nu) and every enabled role object (transform, role,
    region shape, role settings and the evaluated geometry of mesh regions).
    """
    scene = context.scene
    if settings is None:
        settings = scene.stress_sculpt
    digest = hashlib.blake2b(digest_size=16)

    def feed(*values):
        digest.update(repr(values).encode("utf-8"))

    feed(
        "stress-sculpt-setup-1",
        int(settings.resolution),
        str(settings.voxelize_method),
        float(settings.youngs_modulus),
        float(settings.poisson_ratio),
    )
    depsgraph = context.evaluated_depsgraph_get()
    domain = settings.domain
    if domain is None:
        feed("no-domain")
    else:
        verts, tris = evaluated_world_mesh(domain, depsgraph)
        feed("domain", domain.name, verts.shape, tris.shape)
        digest.update(np.ascontiguousarray(verts).tobytes())
        digest.update(np.ascontiguousarray(tris).tobytes())
    for obj in role_objects(context):
        obj_settings = obj.stress_sculpt
        role = obj_settings.role
        feed("object", obj.name, obj.type, role)
        digest.update(np.array(obj.matrix_world, dtype=np.float64).tobytes())
        if obj.type == "EMPTY":
            feed(str(obj.empty_display_type), float(obj.empty_display_size))
        if role == "LOAD":
            feed(float(obj_settings.magnitude), float(obj_settings.load_radius))
        elif role == "SUPPORT":
            feed(bool(obj_settings.fix_x), bool(obj_settings.fix_y), bool(obj_settings.fix_z))
        else:
            feed(float(obj_settings.strength))
        if role != "LOAD" and obj.type in GEOMETRY_TYPES:
            verts, tris = evaluated_world_mesh(obj, depsgraph)
            digest.update(np.ascontiguousarray(verts).tobytes())
            digest.update(np.ascontiguousarray(tris).tobytes())
    return digest.hexdigest()
