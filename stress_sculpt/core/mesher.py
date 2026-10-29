"""Isosurfaces of voxel fields and small mesh utilities (pure numpy).

Conventions (shared with ``core/grid.py``):

* scalar / vector fields are sampled at element centres: value ``field[i, j, k]``
  lives at ``origin + (ijk + 0.5) * h``; extra trailing axes are channels,
* "inside" means ``field >= iso``; surface normals (right-hand rule on the
  polygon order) point outwards, towards ``field < iso``,
* meshes are ``verts (V, 3) float64`` + ``faces (F, k) int64`` (quads ``k = 4``
  or triangles ``k = 3``), indices compact and zero-based.

Surface nets in short (:func:`surface_nets`):

* the field is padded with one layer of an "outside" value, so the surface is
  always closed; every grid edge whose ends lie on different sides of ``iso`` is
  crossed by the surface and yields one quad, built from vertices of the four
  dual cubes (cells of the sample lattice) around that edge,
* a cube usually holds one vertex; when the surface passes through it several
  times (ambiguous configurations) it holds one vertex per *patch*: the crossed
  edges of a cube are linked face by face into closed cycles and every cycle gets
  its own vertex, placed at the mean of its edge crossings (linear interpolation),
* a face with four crossed edges (diagonal "checkerboard" corners) is resolved
  consistently for both cubes sharing it, with the asymptotic decider (the inside
  corners connect through the face when the bilinear saddle value is above
  ``iso``; ties keep them apart). A face resolution that would make two patches
  meet twice through the same face (a sub-voxel tunnel, which would collapse into
  an edge used by four quads) is flipped; this never creates a new conflict, so
  the result is a closed, consistently oriented 2-manifold (every edge in exactly
  two faces, every vertex umbrella a single disk) for any input field.

Everything is vectorized over cells, edges or vertices; the only Python loops run
over the 3 axes, 6 cube faces, 12 cube edges or smoothing iterations. No global
mutable state: the lookup tables are built once at import and are read-only.
"""

import numpy as np


__all__ = [
    "surface_nets",
    "triangulate",
    "vertex_normals",
    "mesh_edges",
    "taubin_smooth",
    "laplacian_smooth_scalar",
    "extrapolate",
    "sample_trilinear",
    "colormap",
    "srgb_to_linear",
    "remove_small_components",
    "COLORMAPS",
]


# ============================================================================ cube tables
# Cube corner n (0..7) sits at offset (n & 1, (n >> 1) & 1, (n >> 2) & 1) from the cube's
# lowest sample; bit n of a cube "config" is set when that corner is inside.
_CORNER_OFFSETS = np.array([[n & 1, (n >> 1) & 1, (n >> 2) & 1] for n in range(8)], dtype=np.int64)


def _other_axes(axis: int) -> tuple:
    return tuple(b for b in range(3) if b != axis)


def _corner_id(offset) -> int:
    return int(offset[0]) + 2 * int(offset[1]) + 4 * int(offset[2])


def _build_cube_edges() -> np.ndarray:
    """Cube edge ``e = 4 * axis + l`` joins corners ``(n0, n1)``, ``n1 = n0 + 2**axis``.

    ``l = ob + 2 * oc`` encodes the offsets along the two other axes ``(b, c)``
    (increasing order), so the grid edge from sample ``s`` along ``axis`` is local
    edge ``l`` of the cube whose lowest sample is ``s - ob * e_b - oc * e_c``.
    """
    edges = []
    for axis in range(3):
        b, c = _other_axes(axis)
        for local in range(4):
            offset = [0, 0, 0]
            offset[b], offset[c] = local & 1, local >> 1
            n0 = _corner_id(offset)
            edges.append((n0, n0 + (1 << axis)))
    return np.array(edges, dtype=np.int64)


_CUBE_EDGES = _build_cube_edges()


def _build_cube_faces() -> tuple:
    """Face ``f = 2 * axis + side``: 4 corners in cyclic order and the 4 edges between them.

    Edge ``i`` of a face joins its corners ``i`` and ``i + 1 (mod 4)``.
    """
    lookup = {frozenset((int(a), int(b))): e for e, (a, b) in enumerate(_CUBE_EDGES)}
    corners, edges = [], []
    for axis in range(3):
        b, c = _other_axes(axis)
        for side in (0, 1):
            ring = []
            for ob, oc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                offset = [0, 0, 0]
                offset[axis], offset[b], offset[c] = side, ob, oc
                ring.append(_corner_id(offset))
            corners.append(ring)
            edges.append([lookup[frozenset((ring[i], ring[(i + 1) % 4]))] for i in range(4)])
    return np.array(corners, dtype=np.int64), np.array(edges, dtype=np.int64)


_FACE_CORNERS, _FACE_EDGES = _build_cube_faces()


def _face_pairs(inside, face: int, connect: int) -> list:
    """Pairs of crossed cube edges joined by the surface on one cube face."""
    ring = _FACE_CORNERS[face]
    fedges = _FACE_EDGES[face]
    state = [inside[int(n)] for n in ring]
    crossed = [i for i in range(4) if state[i] != state[(i + 1) % 4]]
    if len(crossed) == 2:
        return [(int(fedges[crossed[0]]), int(fedges[crossed[1]]))]
    if len(crossed) == 4:
        # The surface cuts off two diagonal corners: the inside ones (inside corners
        # stay apart) or, when ``connect``, the outside ones (inside joins across).
        cut_state = 0 if connect else 1
        return [
            (int(fedges[(i - 1) % 4]), int(fedges[i])) for i in range(4) if state[i] == cut_state
        ]
    return []


def _build_patch_tables() -> tuple:
    """Lookup tables of the surface patches inside a cube.

    Indexed by ``(config, bits)`` where ``bits`` holds one "connect inside" flag per
    ambiguous face (bit ``f``); only flags of faces that are ambiguous for the
    config are ever set, so only those entries are filled:

    * ``patch (256, 64, 12) int8`` - patch id of every crossed cube edge, -1 if the
      edge is not crossed,
    * ``count (256, 64) int8`` - number of patches (0..4),
    * ``double (256, 64, 6) bool`` - the two surface segments on ambiguous face
      ``f`` belong to the same patch (the patch visits that face twice).

    Also returns ``ambiguous (256, 6) bool``.
    """
    patch = np.full((256, 64, 12), -1, dtype=np.int8)
    count = np.zeros((256, 64), dtype=np.int8)
    double = np.zeros((256, 64, 6), dtype=bool)
    ambiguous = np.zeros((256, 6), dtype=bool)
    for config in range(256):
        inside = [(config >> n) & 1 for n in range(8)]
        amb_faces = []
        for face in range(6):
            state = [inside[int(n)] for n in _FACE_CORNERS[face]]
            if state[0] == state[2] and state[1] == state[3] and state[0] != state[1]:
                amb_faces.append(face)
                ambiguous[config, face] = True
        for subset in range(1 << len(amb_faces)):
            bits = 0
            for k, face in enumerate(amb_faces):
                if (subset >> k) & 1:
                    bits |= 1 << face
            partners = {}
            face_pairs = {}
            for face in range(6):
                pairs = _face_pairs(inside, face, (bits >> face) & 1)
                face_pairs[face] = pairs
                for e0, e1 in pairs:
                    partners.setdefault(e0, []).append(e1)
                    partners.setdefault(e1, []).append(e0)
            label = {}
            n_patches = 0
            for start in sorted(partners):
                if start in label:
                    continue
                stack = [start]
                while stack:
                    e = stack.pop()
                    if e in label:
                        continue
                    label[e] = n_patches
                    stack.extend(p for p in partners[e] if p not in label)
                n_patches += 1
            for e, lab in label.items():
                patch[config, bits, e] = lab
            count[config, bits] = n_patches
            for face in amb_faces:
                (a0, _a1), (b0, _b1) = face_pairs[face]
                double[config, bits, face] = label[a0] == label[b0]
    for table in (patch, count, double, ambiguous):
        table.setflags(write=False)
    return patch, count, double, ambiguous


_PATCH, _PATCH_COUNT, _PATCH_DOUBLE, _FACE_AMBIGUOUS = _build_patch_tables()
for _table in (_CORNER_OFFSETS, _CUBE_EDGES, _FACE_CORNERS, _FACE_EDGES):
    _table.setflags(write=False)
del _table

# For a grid edge along ``axis`` from sample ``s``: offsets ``(ob, oc)`` (along the two other
# axes) of the four cubes around it, in the cyclic order whose right-hand normal is +axis.
_QUAD_ORDER = (
    ((1, 1), (0, 1), (0, 0), (1, 0)),
    ((1, 1), (1, 0), (0, 0), (0, 1)),
    ((1, 1), (0, 1), (0, 0), (1, 0)),
)
# Face-resolution flips needed in practice: at most one round (see module docstring).
_MAX_FLIP_ROUNDS = 8


# ============================================================================ surface nets
def _empty_mesh(corners: int = 4) -> tuple:
    return np.zeros((0, 3), dtype=np.float64), np.zeros((0, corners), dtype=np.int64)


def _default_pad(field: np.ndarray, iso: float) -> float:
    """Outside value for the padding layer.

    ``min(field.min(), 2 * iso - field.max())``: the lowest value of the field (so the
    grid border behaves like its emptiest cell, e.g. 0 for densities), lowered if
    needed so that a cell holding the maximum is cut half-way to the border, like
    an interior cell next to a 0 (binary 0 / 1 fields mesh at the voxel faces).
    Always strictly below ``iso``.
    """
    valid = ~np.isnan(field)
    lo = float(np.min(field, initial=np.inf, where=valid))
    hi = float(np.max(field, initial=-np.inf, where=valid))
    pad = min(lo, 2.0 * iso - hi)
    if not (np.isfinite(pad) and pad < iso):
        pad = iso - max(1.0, abs(iso))
    return pad


def _padded_values(flat_field, shape, pad, i, j, k) -> np.ndarray:
    """Values at padded-lattice samples ``(i, j, k)`` (original index + 1); NaN -> pad."""
    nx, ny, nz = shape
    i = i - 1
    j = j - 1
    k = k - 1
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
    flat = (np.clip(i, 0, nx - 1) * ny + np.clip(j, 0, ny - 1)) * nz + np.clip(k, 0, nz - 1)
    values = flat_field[flat]
    ok &= ~np.isnan(values)
    return np.where(ok, values, pad)


def _crossing(v0, v1, iso) -> np.ndarray:
    """Parameter in [0, 1] where the segment v0 -> v1 crosses ``iso`` (linear)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = (iso - v0) / (v1 - v0)
    t = np.where(np.isfinite(t), t, 0.5)
    return np.clip(t, 0.0, 1.0)


def surface_nets(field, iso, origin, h, pad_value=None) -> tuple:
    """Closed quad mesh of the iso-surface ``field == iso`` (manifold surface nets).

    ``field`` ``(nx, ny, nz)`` is sampled at element centres ``origin + (ijk + 0.5) * h``.
    The field is padded with one layer of ``pad_value`` so the surface is closed;
    the default ``min(field.min(), 2 * iso - field.max())`` (always below ``iso``)
    makes the border behave like the emptiest cell, and cuts a cell holding the
    maximum half-way to the border (binary 0 / 1 fields mesh at the voxel faces,
    also along the grid border). Returns
    ``verts (V, 3) float64`` (world) and ``quads (F, 4) int64``; quad normals
    (right-hand rule) point outwards, towards ``field < iso``. Every vertex is the
    mean of the edge crossings (linear interpolation) of its patch of the dual cube;
    see the module docstring for the treatment of ambiguous configurations: the
    mesh is a closed 2-manifold for any input. NaN samples count as outside.
    With ``pad_value >= iso`` the border counts as inside and the faces that would
    close the surface at the grid border are omitted (open surface). Empty result:
    ``(zeros((0, 3)), zeros((0, 4), int64))``.
    """
    verts, quads, _info = _surface_nets(field, iso, origin, h, pad_value)
    return verts, quads


def _surface_nets(field, iso, origin, h, pad_value=None) -> tuple:
    """:func:`surface_nets` plus an ``info`` dict of statistics (used by the tests)."""
    f = np.ascontiguousarray(field, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError("surface_nets: field must have shape (nx, ny, nz)")
    iso = float(iso)
    h = float(h)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    if not np.isfinite(iso):
        raise ValueError("surface_nets: iso must be finite")
    if not h > 0.0:
        raise ValueError("surface_nets: h must be positive")
    info = {"ambiguous_faces": 0, "flipped_faces": 0, "residual_conflicts": 0, "split_cubes": 0}
    if f.size == 0:
        return (*_empty_mesh(), info)
    pad = float(pad_value) if pad_value is not None else _default_pad(f, iso)
    if np.isnan(pad):
        pad = _default_pad(f, iso)
    pad_inside = pad >= iso
    nx, ny, nz = f.shape
    flat_field = f.reshape(-1)

    inside = np.empty((nx + 2, ny + 2, nz + 2), dtype=bool)
    inside.fill(pad_inside)
    np.greater_equal(f, iso, out=inside[1:-1, 1:-1, 1:-1])

    # ---- cube configurations (cube = cell of the padded sample lattice)
    cshape = (nx + 1, ny + 1, nz + 1)
    cfg = np.zeros(cshape, dtype=np.uint8)
    tmp = np.empty(cshape, dtype=np.uint8)
    for n in range(8):
        ox, oy, oz = (int(o) for o in _CORNER_OFFSETS[n])
        np.copyto(tmp, inside[ox : ox + nx + 1, oy : oy + ny + 1, oz : oz + nz + 1])
        np.left_shift(tmp, n, out=tmp)
        np.bitwise_or(cfg, tmp, out=cfg)
    del tmp
    act = np.flatnonzero((cfg != 0) & (cfg != 255))  # sorted flat ids of mixed cubes
    if act.size == 0:
        return (*_empty_mesh(), info)
    c_act = cfg.reshape(-1)[act].astype(np.int64)
    del cfg
    ci, cj, ck = np.unravel_index(act, cshape)
    strides = (cshape[1] * cshape[2], cshape[2], 1)

    # ---- ambiguous faces: asymptotic decider, identical for both cubes of a face
    bits = np.zeros(act.size, dtype=np.int64)
    ambiguous = []
    for face in range(6):
        sel = np.flatnonzero(_FACE_AMBIGUOUS[c_act, face])
        if sel.size == 0:
            continue
        ring = _FACE_CORNERS[face]
        rel = []
        for n in ring:
            ox, oy, oz = (int(o) for o in _CORNER_OFFSETS[n])
            rel.append(
                _padded_values(flat_field, f.shape, pad, ci[sel] + ox, cj[sel] + oy, ck[sel] + oz)
                - iso
            )
        with np.errstate(over="ignore", invalid="ignore"):
            p02 = rel[0] * rel[2]  # products of the two diagonal pairs (both >= 0)
            p13 = rel[1] * rel[3]
        first_pair_inside = ((c_act[sel] >> int(ring[0])) & 1).astype(bool)
        connect = np.where(first_pair_inside, p02 > p13, p13 > p02)
        bits[sel] |= connect.astype(np.int64) << face
        axis, side = divmod(face, 2)
        nb_flat = act[sel] + (1 if side else -1) * strides[axis]
        nb = np.minimum(np.searchsorted(act, nb_flat), act.size - 1)
        if np.any(act[nb] != nb_flat):  # cannot happen: both cubes of the face are mixed
            raise RuntimeError("surface_nets: inconsistent ambiguous face")
        ambiguous.append((face, sel, nb))
        info["ambiguous_faces"] += int(sel.size)
    info["ambiguous_faces"] //= 2  # every face was seen from its two cubes

    # ---- flip resolutions that would make two patches meet twice through one face
    for _round in range(_MAX_FLIP_ROUNDS):
        double = _PATCH_DOUBLE[c_act, bits]
        flips = []
        for face, sel, nb in ambiguous:
            conflict = double[sel, face] & double[nb, face ^ 1]
            if conflict.any():
                flips.append((face, sel[conflict]))
        if not flips:
            break
        for face, cubes in flips:
            bits[cubes] ^= 1 << face
            info["flipped_faces"] += int(cubes.size)
    else:
        double = _PATCH_DOUBLE[c_act, bits]
        for face, sel, nb in ambiguous:
            info["residual_conflicts"] += int(np.count_nonzero(double[sel, face] & double[nb, face ^ 1]))
    info["flipped_faces"] //= 2
    info["residual_conflicts"] //= 2

    # ---- one vertex per patch: mean of its edge crossings
    edge_patch = _PATCH[c_act, bits].astype(np.int64)  # (A, 12), -1 = edge not crossed
    n_patch = _PATCH_COUNT[c_act, bits].astype(np.int64)
    info["split_cubes"] = int(np.count_nonzero(n_patch > 1))
    first = np.cumsum(n_patch) - n_patch
    n_verts = int(n_patch.sum())
    ids, coords = [], [[], [], []]
    for e in range(12):
        sel = np.flatnonzero(edge_patch[:, e] >= 0)
        if sel.size == 0:
            continue
        axis = e // 4
        o0 = _CORNER_OFFSETS[_CUBE_EDGES[e, 0]]
        s = [ci[sel] + o0[0], cj[sel] + o0[1], ck[sel] + o0[2]]
        v0 = _padded_values(flat_field, f.shape, pad, *s)
        s1 = list(s)
        s1[axis] = s1[axis] + 1
        v1 = _padded_values(flat_field, f.shape, pad, *s1)
        t = _crossing(v0, v1, iso)
        for a in range(3):
            pos = origin[a] + (s[a] - 0.5) * h
            if a == axis:
                pos = pos + t * h
            coords[a].append(pos)
        ids.append(first[sel] + edge_patch[sel, e])
    ids = np.concatenate(ids)
    counts = np.bincount(ids, minlength=n_verts).astype(np.float64)
    verts = np.empty((n_verts, 3), dtype=np.float64)
    for a in range(3):
        verts[:, a] = np.bincount(ids, weights=np.concatenate(coords[a]), minlength=n_verts)
    verts /= counts[:, None]
    del ids, coords, counts

    # ---- one quad per crossed grid edge, oriented inside -> outside
    quads = []
    shape_p = inside.shape
    for axis in range(3):
        b, c = _other_axes(axis)
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, shape_p[axis] - 1)
        hi[axis] = slice(1, shape_p[axis])
        lower = inside[tuple(lo)]
        s = np.nonzero(lower != inside[tuple(hi)])
        if s[0].size == 0:
            continue
        lower_inside = lower[s]
        valid = np.ones(s[0].size, dtype=bool)
        corners = []
        for ob, oc in _QUAD_ORDER[axis]:
            cube = [s[0], s[1], s[2]]
            cube[b] = cube[b] - ob
            cube[c] = cube[c] - oc
            for a in (b, c):
                valid &= (cube[a] >= 0) & (cube[a] < cshape[a])
            flat = (
                np.clip(cube[0], 0, cshape[0] - 1) * strides[0]
                + np.clip(cube[1], 0, cshape[1] - 1) * strides[1]
                + np.clip(cube[2], 0, cshape[2] - 1) * strides[2]
            )
            idx = np.minimum(np.searchsorted(act, flat), act.size - 1)
            corners.append(first[idx] + edge_patch[idx, 4 * axis + ob + 2 * oc])
        quad = np.stack(corners, axis=1)
        quad[~lower_inside] = quad[~lower_inside][:, ::-1]
        quads.append(quad[valid])
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), dtype=np.int64)

    # ---- compact (vertices can only be unused when the border counts as inside)
    if quads.shape[0] == 0:
        return (*_empty_mesh(), info)
    used = np.zeros(n_verts, dtype=bool)
    used[quads.ravel()] = True
    if not used.all():
        remap = np.cumsum(used) - 1
        verts = verts[used]
        quads = remap[quads]
    return verts, quads.astype(np.int64, copy=False), info


# ============================================================================ mesh utilities
def _as_faces(faces) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim == 2:
        return faces
    if faces.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    raise ValueError("faces must be a 2-D array (F, k)")


def triangulate(quads) -> np.ndarray:
    """Split quads ``(F, 4)`` into triangles ``(2F, 3)``, keeping the orientation.

    Quad ``q = (a, b, c, d)`` becomes triangles ``2q = (a, b, c)`` and
    ``2q + 1 = (a, c, d)``.
    """
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    tris = np.empty((2 * quads.shape[0], 3), dtype=np.int64)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    return tris


def _face_vector_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(F, 3)`` vector area x 2 of every polygon (normal times twice the area).

    Fan around the first corner, relative coordinates (exact for triangles and for
    any quad: it equals ``(v2 - v0) x (v3 - v1)``).
    """
    base = verts[faces[:, 0]]
    total = np.zeros((faces.shape[0], 3), dtype=np.float64)
    for c in range(1, faces.shape[1] - 1):
        total += np.cross(verts[faces[:, c]] - base, verts[faces[:, c + 1]] - base)
    return total


def vertex_normals(verts, faces) -> np.ndarray:
    """Area-weighted unit vertex normals ``(V, 3)`` of a triangle or quad mesh.

    Vertices without (non-degenerate) faces get ``(0, 0, 1)``.
    """
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = _as_faces(faces)
    n_verts = verts.shape[0]
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    if faces.size and n_verts:
        area = _face_vector_areas(verts, faces)
        idx = faces.ravel()
        for a in range(3):
            normals[:, a] = np.bincount(
                idx, weights=np.repeat(area[:, a], faces.shape[1]), minlength=n_verts
            )
    length = np.linalg.norm(normals, axis=1)
    ok = length > 0.0
    normals[ok] /= length[ok, None]
    normals[~ok] = (0.0, 0.0, 1.0)
    return normals


def mesh_edges(faces) -> np.ndarray:
    """Unique undirected edges ``(E, 2)`` (``edge[:, 0] < edge[:, 1]``) of polygons ``(F, k)``."""
    faces = _as_faces(faces)
    if faces.size == 0 or faces.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.int64)
    a = faces.ravel()
    b = np.roll(faces, -1, axis=1).ravel()
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keep = lo != hi
    n = int(hi.max()) + 1
    key = np.unique(lo[keep] * n + hi[keep])
    return np.stack([key // n, key % n], axis=1)


class _UmbrellaOperator:
    """Uniform ("umbrella") Laplacian ``L x = mean(neighbours) - x`` over mesh edges."""

    def __init__(self, faces, n_verts: int):
        edges = mesh_edges(faces)
        if edges.size and int(edges.max()) >= n_verts:
            raise ValueError("faces reference vertices beyond n_verts")
        self.n = int(n_verts)
        self.i = edges[:, 0]
        self.j = edges[:, 1]
        degree = np.bincount(edges.ravel(), minlength=self.n).astype(np.float64)
        self.has_neighbours = degree > 0.0
        self.inv_degree = np.zeros(self.n, dtype=np.float64)
        self.inv_degree[self.has_neighbours] = 1.0 / degree[self.has_neighbours]

    def neighbour_mean(self, x: np.ndarray) -> np.ndarray:
        """Mean of the neighbours (rows without neighbours: 0); ``x`` is ``(V,)`` or ``(V, C)``."""
        flat = x.reshape(self.n, -1)
        out = np.empty_like(flat)
        for c in range(flat.shape[1]):
            col = flat[:, c]
            out[:, c] = np.bincount(self.i, weights=col[self.j], minlength=self.n)
            out[:, c] += np.bincount(self.j, weights=col[self.i], minlength=self.n)
        out *= self.inv_degree[:, None]
        return out.reshape(x.shape)

    def laplacian(self, x: np.ndarray) -> np.ndarray:
        mean = self.neighbour_mean(x)
        mask = self.has_neighbours.reshape((-1,) + (1,) * (x.ndim - 1))
        return np.where(mask, mean - x, 0.0)


def _pinned_mask(pinned, n_verts: int) -> np.ndarray:
    if pinned is None:
        return np.zeros(n_verts, dtype=bool)
    pinned = np.asarray(pinned)
    if pinned.dtype == bool:
        return pinned.reshape(n_verts)
    mask = np.zeros(n_verts, dtype=bool)
    mask[pinned.astype(np.int64).ravel()] = True
    return mask


def taubin_smooth(verts, faces, iterations=10, lam=0.5, mu=-0.53, pinned=None) -> np.ndarray:
    """Taubin lambda|mu smoothing (low-pass, practically volume preserving).

    Every iteration applies ``x += lam * L x`` then ``x += mu * L x`` with the uniform
    umbrella Laplacian ``L`` over the mesh edges (``mu < -lam < 0``; the default pass
    band ``1/lam + 1/mu`` is ~0.11). ``pinned`` (bool mask ``(V,)`` or vertex indices)
    keeps vertices in place. Returns new vertices ``(V, 3)``; the input is untouched.
    """
    out = np.array(verts, dtype=np.float64).reshape(-1, 3)
    faces = _as_faces(faces)
    iterations = int(iterations)
    if iterations <= 0 or faces.size == 0 or out.shape[0] == 0:
        return out
    op = _UmbrellaOperator(faces, out.shape[0])
    free = ~_pinned_mask(pinned, out.shape[0])
    if not free.any():
        return out
    for _ in range(iterations):
        for factor in (float(lam), float(mu)):
            step = op.laplacian(out)
            step *= factor
            step[~free] = 0.0
            out += step
    return out


def laplacian_smooth_scalar(values, faces, n_verts, iterations) -> np.ndarray:
    """Smooth per-vertex values ``(V,)`` or ``(V, C)`` over mesh edges.

    Every iteration replaces each value by the average of itself and the mean of
    its edge neighbours (damped Jacobi, never overshoots); isolated vertices keep
    their value. Returns a new float64 array.
    """
    out = np.array(values, dtype=np.float64)
    n_verts = int(n_verts)
    if out.shape[0] != n_verts:
        raise ValueError("values must have n_verts rows")
    faces = _as_faces(faces)
    iterations = int(iterations)
    if iterations <= 0 or faces.size == 0 or n_verts == 0:
        return out
    op = _UmbrellaOperator(faces, n_verts)
    for _ in range(iterations):
        out += 0.5 * op.laplacian(out)
    return out


def _component_labels(n: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component root (smallest vertex id) of every vertex.

    Vectorized union-find: hook every root onto the smallest root it is linked to,
    then pointer jumping, until no edge joins two different roots.
    """
    parent = np.arange(n, dtype=np.int64)
    if edges.size == 0:
        return parent
    u = edges[:, 0]
    v = edges[:, 1]
    while True:
        pu = parent[u]
        pv = parent[v]
        diff = pu != pv
        if not diff.any():
            return parent
        np.minimum.at(parent, np.maximum(pu[diff], pv[diff]), np.minimum(pu[diff], pv[diff]))
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped


def remove_small_components(verts, faces, min_faces) -> tuple:
    """Drop floating islands: connected components (by shared vertices) with fewer
    than ``min_faces`` faces. The largest component is always kept.

    Returns ``(verts, faces)`` with unused vertices removed and indices compacted.
    """
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = _as_faces(faces)
    if faces.shape[0] == 0:
        return verts[:0].copy(), faces.copy()
    n_verts = verts.shape[0]
    links = [np.stack([faces[:, 0], faces[:, c]], axis=1) for c in range(1, faces.shape[1])]
    labels = _component_labels(n_verts, np.concatenate(links))
    face_label = labels[faces[:, 0]]
    sizes = np.bincount(face_label, minlength=n_verts)
    keep_label = sizes >= int(min_faces)
    keep_label[int(np.argmax(sizes))] = True
    keep = keep_label[face_label]
    faces = faces[keep]
    used = np.zeros(n_verts, dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    return verts[used], remap[faces]


# ============================================================================ field sampling
_SAMPLE_CHUNK = 1 << 18  # points per block in sample_trilinear (bounds temporaries)


def _dilate6(mask: np.ndarray) -> np.ndarray:
    """Boolean dilation by the 6 face neighbours (zero-padded, no wrap-around)."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _extrapolate(field, mask, iterations) -> tuple:
    """:func:`extrapolate` that also returns the final validity mask ``(nx, ny, nz)``."""
    out = np.array(field, dtype=np.float64, order="C")
    if out.ndim < 3:
        raise ValueError("field must have shape (nx, ny, nz) or (nx, ny, nz, C...)")
    shape = out.shape[:3]
    valid = np.array(mask, dtype=bool, order="C").reshape(shape)
    flat = out.reshape(int(np.prod(shape)), -1)
    vflat = valid.reshape(-1)
    strides = (shape[1] * shape[2], shape[2], 1)
    for _ in range(int(iterations)):
        if not vflat.any():
            break
        frontier = _dilate6(valid)
        frontier &= ~valid
        idx = np.flatnonzero(frontier)
        if idx.size == 0:
            break
        coords = np.unravel_index(idx, shape)
        acc = np.zeros((idx.size, flat.shape[1]), dtype=np.float64)
        cnt = np.zeros(idx.size, dtype=np.float64)
        for axis in range(3):
            for step in (-1, 1):
                inb = np.flatnonzero((coords[axis] + step >= 0) & (coords[axis] + step < shape[axis]))
                nidx = idx[inb] + step * strides[axis]
                ok = vflat[nidx]
                rows = inb[ok]
                acc[rows] += flat[nidx[ok]]
                cnt[rows] += 1.0
        flat[idx] = acc / cnt[:, None]  # every frontier cell has a valid neighbour
        vflat[idx] = True
    return out, valid


def extrapolate(field, mask, iterations=3) -> np.ndarray:
    """Extend a field past the cells of ``mask``.

    ``iterations`` times, every cell outside the current valid set that has valid
    6-neighbours gets the mean of their values and becomes valid (so values reach
    ``iterations`` cells beyond the mask). Values in ``mask`` and cells never reached
    are returned unchanged. ``field`` is ``(nx, ny, nz)`` or has trailing channel
    axes ``(nx, ny, nz, C...)``; ``mask`` is ``(nx, ny, nz)`` bool. Returns a new array.
    """
    return _extrapolate(field, mask, iterations)[0]


def _trilinear_block(flat, shape, origin, h, pts, mask_flat) -> tuple:
    """Weighted corner sums ``(acc (P, C), weight (P,))`` of a block of points."""
    u = (pts - origin) / h - 0.5
    finite = np.isfinite(u).all(axis=1)
    u[~finite] = 0.0
    lo, hi, frac = [], [], []
    for a in range(3):
        n = shape[a]
        ua = np.clip(u[:, a], 0.0, n - 1.0)
        i0 = np.minimum(np.floor(ua).astype(np.int64), max(n - 2, 0))
        lo.append(i0)
        hi.append(np.minimum(i0 + 1, n - 1))
        frac.append(ua - i0)
    acc = np.zeros((pts.shape[0], flat.shape[1]), dtype=np.float64)
    weight = np.zeros(pts.shape[0], dtype=np.float64)
    for n in range(8):
        dx, dy, dz = (int(o) for o in _CORNER_OFFSETS[n])
        idx = (
            ((hi[0] if dx else lo[0]) * shape[1] + (hi[1] if dy else lo[1])) * shape[2]
            + (hi[2] if dz else lo[2])
        )
        w = (frac[0] if dx else 1.0 - frac[0]) * (frac[1] if dy else 1.0 - frac[1])
        w = w * (frac[2] if dz else 1.0 - frac[2])
        if mask_flat is not None:
            w = w * mask_flat[idx]
        acc += w[:, None] * flat[idx]
        weight += w
    weight[~finite] = 0.0
    return acc, weight, finite


def sample_trilinear(field, origin, h, points, mask=None) -> np.ndarray:
    """Trilinear interpolation of an element-centred field at world points.

    ``field`` is ``(nx, ny, nz)`` or ``(nx, ny, nz, C...)`` with samples at
    ``origin + (ijk + 0.5) * h``; ``points`` ``(P, 3)``. Points outside the grid are
    clamped to it (constant extension). With ``mask`` ``(nx, ny, nz)`` bool, only
    mask cells contribute and their weights are renormalized; points without any
    valid neighbour use the field extended by 3 :func:`extrapolate` passes, and get
    0 if that does not reach them either. Non-finite points get 0. Returns ``(P,)``
    or ``(P, C...)`` float64.
    """
    f = np.ascontiguousarray(field, dtype=np.float64)
    if f.ndim < 3:
        raise ValueError("field must have shape (nx, ny, nz) or (nx, ny, nz, C...)")
    shape = f.shape[:3]
    channels = f.shape[3:]
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    h = float(h)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    flat = f.reshape(int(np.prod(shape)), -1)
    mask_flat = None
    if mask is not None:
        mask_flat = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_flat.size != flat.shape[0]:
            raise ValueError("mask must have the grid shape")
    out = np.zeros((pts.shape[0], flat.shape[1]), dtype=np.float64)
    missing = []
    for start in range(0, pts.shape[0], _SAMPLE_CHUNK):
        block = pts[start : start + _SAMPLE_CHUNK]
        acc, weight, finite = _trilinear_block(flat, shape, origin, h, block, mask_flat)
        good = weight > 1e-9 if mask_flat is not None else finite
        rows = np.flatnonzero(good)
        out[start + rows] = acc[rows] / weight[rows, None] if mask_flat is not None else acc[rows]
        if mask_flat is not None:
            missing.append(start + np.flatnonzero(~good & finite))
    if missing:
        missing = np.concatenate(missing)
        if missing.size and mask_flat.any():
            ext, ext_valid = _extrapolate(
                np.where(mask_flat[:, None], flat, 0.0).reshape(f.shape), mask_flat.reshape(shape), 3
            )
            ext_flat = ext.reshape(flat.shape)
            acc, weight, _finite = _trilinear_block(
                ext_flat, shape, origin, h, pts[missing], ext_valid.reshape(-1)
            )
            good = weight > 1e-9
            out[missing[good]] = acc[good] / weight[good, None]
    return out.reshape((pts.shape[0],) + channels)


# ============================================================================ colormaps
COLORMAPS = ("HEAT", "COOLWARM", "VIRIDIS")

# Control points (position, r, g, b) in display sRGB; interpolated in sRGB (perceptual
# steps) and converted to linear RGB, the space of Blender color attributes.
_COLORMAP_POINTS = {
    "HEAT": (  # black - red - yellow - white
        (0.00, 0.000, 0.000, 0.000),
        (0.30, 0.620, 0.040, 0.020),
        (0.55, 0.950, 0.300, 0.020),
        (0.78, 1.000, 0.800, 0.100),
        (1.00, 1.000, 1.000, 1.000),
    ),
    "COOLWARM": (  # Moreland's diverging blue - white - red
        (0.00, 0.230, 0.299, 0.754),
        (0.25, 0.552, 0.690, 0.996),
        (0.50, 0.865, 0.865, 0.865),
        (0.75, 0.958, 0.604, 0.481),
        (1.00, 0.706, 0.016, 0.150),
    ),
    "VIRIDIS": (
        (0.000, 0.267, 0.005, 0.329),
        (0.125, 0.283, 0.141, 0.458),
        (0.250, 0.230, 0.322, 0.546),
        (0.375, 0.173, 0.449, 0.558),
        (0.500, 0.128, 0.567, 0.551),
        (0.625, 0.158, 0.684, 0.502),
        (0.750, 0.369, 0.789, 0.383),
        (0.875, 0.678, 0.864, 0.190),
        (1.000, 0.993, 0.906, 0.144),
    ),
}


def srgb_to_linear(c) -> np.ndarray:
    """sRGB transfer function inverse (display value -> linear light)."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def colormap(values, vmin, vmax, name="HEAT") -> np.ndarray:
    """Linear-RGB colors ``(..., 4)`` (alpha 1) of ``values`` mapped from ``[vmin, vmax]``.

    ``name``: ``"HEAT"`` (black - red - yellow - white), ``"COOLWARM"`` (diverging;
    when ``vmin < 0 < vmax`` zero maps to the white centre and each side is scaled
    separately) or ``"VIRIDIS"``. Values outside the range saturate; NaN maps to the
    low end; a degenerate range maps everything to the middle color.
    """
    key = str(name).upper()
    if key not in _COLORMAP_POINTS:
        raise ValueError(f"Unknown colormap {name!r}; expected one of {', '.join(COLORMAPS)}")
    v = np.asarray(values, dtype=np.float64)
    vmin = float(vmin)
    vmax = float(vmax)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if key == "COOLWARM" and vmin < 0.0 < vmax:
            t = np.where(v < 0.0, 0.5 - 0.5 * v / vmin, 0.5 + 0.5 * v / vmax)
        elif vmax > vmin:
            t = (v - vmin) / (vmax - vmin)
        else:
            t = np.full(v.shape, 0.5)
    t = np.where(np.isnan(t), 0.0, np.clip(t, 0.0, 1.0))
    table = np.asarray(_COLORMAP_POINTS[key], dtype=np.float64)
    out = np.ones(v.shape + (4,), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(t, table[:, 0], table[:, c + 1])
    out[..., :3] = srgb_to_linear(out[..., :3])
    return out
