# SPDX-License-Identifier: GPL-3.0-or-later
"""Numpy mesh container used for samples and for the generated result."""

import numpy as np

# Attribute types we know how to copy: data_type -> (numpy dtype, components)
ATTR_TYPES = {
    "FLOAT": (np.float32, 1),
    "INT": (np.int32, 1),
    "INT8": (np.int32, 1),
    "BOOLEAN": (np.bool_, 1),
    "FLOAT_VECTOR": (np.float32, 3),
    "FLOAT2": (np.float32, 2),
    "FLOAT_COLOR": (np.float32, 4),
    "BYTE_COLOR": (np.float32, 4),
    "INT32_2D": (np.int32, 2),
    "QUATERNION": (np.float32, 4),
}

DOMAINS = ("POINT", "EDGE", "FACE", "CORNER")


class MeshData:
    """Plain arrays describing a polygon mesh.

    ``loop_vert`` / ``loop_edge`` are the per-corner vertex and edge indices,
    faces are described by ``face_start`` / ``face_size``. ``attrs`` maps an
    attribute name to ``(domain, data_type, array)``; ``uvs`` is a list of
    ``(name, (n_corners, 2) array)``.
    """

    __slots__ = ("co", "loop_vert", "loop_edge", "face_start", "face_size", "edges",
                 "mat_index", "materials", "uvs", "attrs", "seams", "_rev", "_bbox")

    def __init__(self, co, loop_vert, loop_edge, face_start, face_size, edges,
                 mat_index=None, materials=None, uvs=None, attrs=None, seams=None):
        self.co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
        self.loop_vert = np.asarray(loop_vert, dtype=np.int32).reshape(-1)
        self.loop_edge = np.asarray(loop_edge, dtype=np.int32).reshape(-1)
        self.face_start = np.asarray(face_start, dtype=np.int32).reshape(-1)
        self.face_size = np.asarray(face_size, dtype=np.int32).reshape(-1)
        self.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        nf = len(self.face_start)
        self.mat_index = (np.zeros(nf, np.int32) if mat_index is None
                          else np.asarray(mat_index, dtype=np.int32).reshape(-1))
        self.materials = list(materials) if materials else [None]
        self.uvs = list(uvs) if uvs else []
        self.attrs = dict(attrs) if attrs else {}
        self.seams = None if seams is None else np.asarray(seams, dtype=np.bool_).reshape(-1)
        self._rev = None
        self._bbox = None

    # -- basic info ------------------------------------------------------------

    @property
    def nv(self):
        return len(self.co)

    @property
    def ne(self):
        return len(self.edges)

    @property
    def nl(self):
        return len(self.loop_vert)

    @property
    def nf(self):
        return len(self.face_start)

    def domain_size(self, domain):
        return {"POINT": self.nv, "EDGE": self.ne, "FACE": self.nf, "CORNER": self.nl}[domain]

    def bbox(self):
        if self._bbox is None:
            if self.nv:
                self._bbox = (self.co.min(axis=0), self.co.max(axis=0))
            else:
                self._bbox = (np.zeros(3), np.zeros(3))
        return self._bbox

    def copy_with(self, **kw):
        args = dict(co=self.co, loop_vert=self.loop_vert, loop_edge=self.loop_edge,
                    face_start=self.face_start, face_size=self.face_size, edges=self.edges,
                    mat_index=self.mat_index, materials=self.materials, uvs=self.uvs,
                    attrs=self.attrs, seams=self.seams)
        args.update(kw)
        return MeshData(**args)

    # -- winding -----------------------------------------------------------------

    def reversed_perms(self):
        """Corner permutations that flip every face (vertex order, edge order)."""
        if self._rev is None:
            starts = np.repeat(self.face_start, self.face_size)
            sizes = np.repeat(self.face_size, self.face_size)
            local = np.arange(self.nl, dtype=np.int32) - starts
            pv = starts + np.where(local == 0, 0, sizes - local)
            pe = starts + (sizes - 1 - local)
            self._rev = (pv.astype(np.int32), pe.astype(np.int32))
        return self._rev

    def flipped(self):
        """Same mesh with reversed face winding."""
        pv, pe = self.reversed_perms()
        attrs = {}
        for name, (dom, dt, arr) in self.attrs.items():
            attrs[name] = (dom, dt, arr[pv] if dom == "CORNER" else arr)
        return self.copy_with(loop_vert=self.loop_vert[pv], loop_edge=self.loop_edge[pe],
                              uvs=[(n, a[pv]) for n, a in self.uvs], attrs=attrs)

    def transformed(self, matrix):
        """Apply a 3x3 or 4x4 matrix; mirrored transforms also flip the winding."""
        m = np.asarray(matrix, dtype=np.float64)
        co = self.co @ m[:3, :3].T
        if m.shape == (4, 4):
            co = co + m[:3, 3]
        out = self.copy_with(co=co)
        if np.linalg.det(m[:3, :3]) < 0.0:
            out = out.flipped()
        return out

    # -- constructors ----------------------------------------------------------

    @staticmethod
    def empty():
        z = np.zeros(0)
        return MeshData(np.zeros((0, 3)), z, z, z, z, np.zeros((0, 2)))

    @staticmethod
    def box(bmin, bmax, material=None):
        x0, y0, z0 = (float(v) for v in bmin)
        x1, y1, z1 = (float(v) for v in bmax)
        co = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
              (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return MeshData.from_faces(co, faces, materials=[material])

    @staticmethod
    def from_faces(co, faces, materials=None):
        """Build from vertex positions and polygon index lists (edges derived)."""
        loop_vert = []
        face_start = []
        face_size = []
        edge_map = {}
        edges = []
        loop_edge = []
        for f in faces:
            face_start.append(len(loop_vert))
            face_size.append(len(f))
            for j, v in enumerate(f):
                loop_vert.append(v)
                w = f[(j + 1) % len(f)]
                key = (min(v, w), max(v, w))
                e = edge_map.get(key)
                if e is None:
                    e = len(edges)
                    edge_map[key] = e
                    edges.append(key)
                loop_edge.append(e)
        return MeshData(co, loop_vert, loop_edge, face_start, face_size,
                        np.array(edges, dtype=np.int32).reshape(-1, 2), materials=materials)


def merge(parts):
    """Merge several :class:`MeshData` into one (unifying materials and attributes)."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return MeshData.empty()
    if len(parts) == 1:
        return parts[0]
    groups = []
    for md in parts:
        groups.append((md, md.co[None, :, :], np.zeros(1, dtype=bool)))
    res = assemble(groups)
    return MeshData(res.co, res.loop_vert, res.loop_edge, res.face_start,
                    np.diff(np.append(res.face_start, len(res.loop_vert))).astype(np.int32),
                    res.edges, mat_index=res.mat_index, materials=res.materials,
                    uvs=res.uvs, attrs=res.attrs, seams=res.seams)


class Assembled:
    """Final arrays ready to be written into a Blender mesh."""

    __slots__ = ("co", "loop_vert", "loop_edge", "face_start", "edges", "mat_index",
                 "materials", "uvs", "attrs", "seams")


def _material_key(mat):
    if mat is None:
        return None
    as_pointer = getattr(mat, "as_pointer", None)
    # Blender returns a new Python wrapper on every access: compare pointers.
    return ("ptr", as_pointer()) if as_pointer else ("id", id(mat))


def assemble(groups):
    """Concatenate module copies.

    ``groups``: list of ``(mesh_data, positions (M, V, 3), flip (M,))``. Each
    module copy uses the topology of ``mesh_data`` with its own positions;
    flipped modules get reversed face winding.
    """
    # Global material list.
    materials = []
    mat_lookup = {}
    luts = []
    for md, _, _ in groups:
        lut = []
        for mat in md.materials:
            key = _material_key(mat)
            if key not in mat_lookup:
                mat_lookup[key] = len(materials)
                materials.append(mat)
            lut.append(mat_lookup[key])
        luts.append(np.asarray(lut if lut else [0], dtype=np.int32))
    if not materials:
        materials = [None]

    # Attribute and UV layer union.
    spec = {}
    for md, _, _ in groups:
        for name, (dom, dt, _) in md.attrs.items():
            if name not in spec:
                spec[name] = (dom, dt)
    n_uv = max((len(md.uvs) for md, _, _ in groups), default=0)
    uv_names = []
    for j in range(n_uv):
        nm = None
        for md, _, _ in groups:
            if j < len(md.uvs):
                nm = md.uvs[j][0]
                break
        uv_names.append(nm or "UVMap.%03d" % j)
    any_seams = any(md.seams is not None and md.seams.any() for md, _, _ in groups)

    co_parts, lv_parts, le_parts, fs_parts, ed_parts, mi_parts, seam_parts = [], [], [], [], [], [], []
    uv_parts = [[] for _ in range(n_uv)]
    attr_parts = {name: [] for name in spec}
    v_base = e_base = l_base = 0
    for gi, (md, pos, flip) in enumerate(groups):
        M = pos.shape[0]
        if M == 0 or md.nf == 0 and md.ne == 0 and md.nv == 0:
            continue
        flip = np.asarray(flip, dtype=bool).reshape(-1)
        V, E, Lc, F = md.nv, md.ne, md.nl, md.nf
        co_parts.append(np.asarray(pos, dtype=np.float64).reshape(-1, 3))
        mods = np.arange(M, dtype=np.int64)
        voff = (v_base + mods * V)[:, None]
        eoff = (e_base + mods * E)[:, None]
        loff = (l_base + mods * Lc)[:, None]
        any_flip = flip.any()
        if any_flip:
            pv, pe = md.reversed_perms()
            lv = np.where(flip[:, None], md.loop_vert[pv][None, :], md.loop_vert[None, :])
            le = np.where(flip[:, None], md.loop_edge[pe][None, :], md.loop_edge[None, :])
        else:
            lv = np.broadcast_to(md.loop_vert[None, :], (M, Lc))
            le = np.broadcast_to(md.loop_edge[None, :], (M, Lc))
        lv_parts.append((lv + voff).reshape(-1))
        le_parts.append((le + eoff).reshape(-1))
        fs_parts.append((md.face_start[None, :] + loff).reshape(-1))
        ed_parts.append((md.edges[None, :, :] + voff[:, :, None]).reshape(-1, 2))
        lut = luts[gi]
        mi = lut[np.clip(md.mat_index, 0, len(lut) - 1)] if F else md.mat_index
        mi_parts.append(np.tile(mi, M))
        if any_seams:
            s = md.seams if md.seams is not None else np.zeros(E, dtype=bool)
            seam_parts.append(np.tile(s, M))

        def corner_tile(arr):
            if any_flip:
                rev = arr[md.reversed_perms()[0]]
                tiled = np.where(flip.reshape((M,) + (1,) * arr.ndim), rev[None], arr[None])
            else:
                tiled = np.broadcast_to(arr[None], (M,) + arr.shape)
            return tiled.reshape((M * arr.shape[0],) + arr.shape[1:])

        for j in range(n_uv):
            if j < len(md.uvs):
                uv_parts[j].append(corner_tile(md.uvs[j][1]))
            else:
                uv_parts[j].append(np.zeros((M * Lc, 2), np.float32))
        for name, (dom, dt) in spec.items():
            npdt, comps = ATTR_TYPES[dt]
            count = md.domain_size(dom)
            entry = md.attrs.get(name)
            if entry is None or entry[0] != dom or entry[1] != dt:
                shape = (M * count,) if comps == 1 else (M * count, comps)
                attr_parts[name].append(np.zeros(shape, dtype=npdt))
                continue
            arr = entry[2]
            if dom == "CORNER":
                attr_parts[name].append(corner_tile(arr))
            else:
                attr_parts[name].append(np.tile(arr, (M,) + (1,) * (arr.ndim - 1)))
        v_base += M * V
        e_base += M * E
        l_base += M * Lc

    out = Assembled()

    def cat(parts, shape_tail=(), dtype=np.int32):
        if parts:
            return np.concatenate(parts).astype(dtype, copy=False)
        return np.zeros((0,) + shape_tail, dtype=dtype)

    out.co = cat(co_parts, (3,), np.float64)
    out.loop_vert = cat(lv_parts)
    out.loop_edge = cat(le_parts)
    out.face_start = cat(fs_parts)
    out.edges = cat(ed_parts, (2,))
    out.mat_index = cat(mi_parts)
    out.materials = materials
    out.uvs = [(uv_names[j], cat(uv_parts[j], (2,), np.float32)) for j in range(n_uv)]
    out.attrs = {}
    for name, (dom, dt) in spec.items():
        npdt, comps = ATTR_TYPES[dt]
        tail = () if comps == 1 else (comps,)
        out.attrs[name] = (dom, dt, cat(attr_parts[name], tail, npdt))
    out.seams = cat(seam_parts, (), np.bool_) if any_seams else None
    return out
