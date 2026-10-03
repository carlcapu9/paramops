# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh buffers (numpy) and the assembly of many placed copies into one mesh."""

import numpy as np

# data_type -> (numpy dtype, components)
ATTRIBUTE_TYPES = {
    "FLOAT": (np.float32, 1), "INT": (np.int32, 1), "INT8": (np.int32, 1),
    "BOOLEAN": (np.bool_, 1), "FLOAT_VECTOR": (np.float32, 3), "FLOAT2": (np.float32, 2),
    "FLOAT_COLOR": (np.float32, 4), "BYTE_COLOR": (np.float32, 4), "INT32_2D": (np.int32, 2),
    "QUATERNION": (np.float32, 4),
}


class MeshBuf:
    """Polygon mesh as arrays. Faces use CSR offsets: face f owns corners
    ``offsets[f]:offsets[f + 1]``."""

    __slots__ = ("co", "corner_vert", "corner_edge", "offsets", "edges", "face_mat", "materials",
                 "uvs", "attrs", "seams", "_flip", "_bbox")

    def __init__(self, co, corner_vert, corner_edge, offsets, edges, face_mat=None, materials=None,
                 uvs=None, attrs=None, seams=None):
        self.co = np.asarray(co, dtype=np.float64).reshape(-1, 3)
        self.corner_vert = np.asarray(corner_vert, dtype=np.int32).reshape(-1)
        self.corner_edge = np.asarray(corner_edge, dtype=np.int32).reshape(-1)
        self.offsets = np.asarray(offsets, dtype=np.int32).reshape(-1)
        self.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        nf = max(len(self.offsets) - 1, 0)
        self.face_mat = np.zeros(nf, np.int32) if face_mat is None else np.asarray(face_mat, np.int32)
        self.materials = list(materials) if materials else [None]
        self.uvs = list(uvs or [])
        self.attrs = dict(attrs or {})
        self.seams = None if seams is None else np.asarray(seams, dtype=bool)
        self._flip = None
        self._bbox = None

    @property
    def n_faces(self):
        return max(len(self.offsets) - 1, 0)

    def count(self, domain):
        return {"POINT": len(self.co), "EDGE": len(self.edges), "FACE": self.n_faces,
                "CORNER": len(self.corner_vert)}[domain]

    def bbox(self):
        if self._bbox is None:
            if len(self.co):
                self._bbox = (self.co.min(axis=0), self.co.max(axis=0))
            else:
                self._bbox = (np.zeros(3), np.zeros(3))
        return self._bbox

    def flip_perms(self):
        """Corner permutations reversing every face: (vertex order, edge order)."""
        if self._flip is None:
            sizes = np.diff(self.offsets)
            start = np.repeat(self.offsets[:-1], sizes)
            size = np.repeat(sizes, sizes)
            local = np.arange(len(self.corner_vert), dtype=np.int64) - start
            pv = start + np.where(local == 0, 0, size - local)
            pe = start + size - 1 - local
            self._flip = (pv.astype(np.int64), pe.astype(np.int64))
        return self._flip

    def with_co(self, co):
        out = MeshBuf.__new__(MeshBuf)
        for name in MeshBuf.__slots__:
            setattr(out, name, getattr(self, name))
        out.co = co
        out._bbox = None
        return out

    @staticmethod
    def from_polygons(co, polygons, materials=None, face_mat=None):
        """Build from vertex positions and lists of vertex indices (edges are derived)."""
        corner_vert, corner_edge, offsets, edges = [], [], [0], []
        lookup = {}
        for poly in polygons:
            for j, v in enumerate(poly):
                w = poly[(j + 1) % len(poly)]
                key = (v, w) if v < w else (w, v)
                e = lookup.get(key)
                if e is None:
                    e = lookup[key] = len(edges)
                    edges.append(key)
                corner_vert.append(v)
                corner_edge.append(e)
            offsets.append(len(corner_vert))
        return MeshBuf(co, corner_vert, corner_edge, offsets, np.array(edges, np.int32).reshape(-1, 2),
                       face_mat=face_mat, materials=materials)

    @staticmethod
    def box(bmin, bmax, material=None):
        x0, y0, z0 = (float(v) for v in bmin)
        x1, y1, z1 = (float(v) for v in bmax)
        co = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
              (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return MeshBuf.from_polygons(co, quads, materials=[material])


class Mesh:
    """Assembled result."""

    __slots__ = ("co", "corner_vert", "corner_edge", "face_start", "edges", "face_mat", "materials",
                 "uvs", "attrs", "seams")


def _material_key(mat):
    if mat is None:
        return None
    ptr = getattr(mat, "as_pointer", None)
    return ("p", ptr()) if ptr else ("i", id(mat))


class Assembler:
    """Collects batches of copies and concatenates them."""

    def __init__(self):
        self.batches = []
        self.materials = []
        self._mat_index = {}

    def material_index(self, mat):
        key = _material_key(mat)
        idx = self._mat_index.get(key)
        if idx is None:
            idx = self._mat_index[key] = len(self.materials)
            self.materials.append(mat)
        return idx

    def add(self, buf, positions, flip=None, overrides=(), uv=None, u_override=None):
        """Add ``len(positions)`` copies of ``buf``.

        ``overrides``: tuple of ("ALL" | "SLOT", slot, material) rules.
        ``uv``: (M, 3, 3) affine UV transforms or None.
        ``u_override``: (M, V) values replacing U of the first UV map (rail mapping).
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 2:
            positions = positions[None]
        if positions.shape[0] == 0 or buf.n_faces == 0 and len(buf.edges) == 0:
            return
        mats = list(buf.materials)
        for kind, slot, mat in overrides:
            if kind == "ALL":
                mats = [mat] * len(mats)
            elif 0 <= slot < len(mats):
                mats[slot] = mat
        lut = np.array([self.material_index(m) for m in mats] or [self.material_index(None)], np.int32)
        if flip is None:
            flip = np.zeros(positions.shape[0], dtype=bool)
        self.batches.append((buf, positions, np.asarray(flip, dtype=bool), lut, uv, u_override))

    def build(self):
        batches = self.batches
        spec = {}
        for buf, *_ in batches:
            for name, (dom, dt, _a) in buf.attrs.items():
                spec.setdefault(name, (dom, dt))
        n_uv = max((len(b[0].uvs) for b in batches), default=0)
        if any(b[5] is not None for b in batches):
            n_uv = max(n_uv, 1)
        uv_names = []
        for j in range(n_uv):
            uv_names.append(next((b[0].uvs[j][0] for b in batches if j < len(b[0].uvs)), "UVMap.%03d" % j))
        any_seam = any(b[0].seams is not None for b in batches)

        co, cv, ce, fs, ed, fm, seams = [], [], [], [], [], [], []
        uvs = [[] for _ in range(n_uv)]
        attrs = {name: [] for name in spec}
        nv = ne = nc = 0
        for buf, pos, flip, lut, uvx, u_over in batches:
            M = pos.shape[0]
            V, E, C, F = len(buf.co), len(buf.edges), len(buf.corner_vert), buf.n_faces
            co.append(pos.reshape(-1, 3))
            r = np.arange(M, dtype=np.int64)
            voff = (nv + r * V)[:, None]
            eoff = (ne + r * E)[:, None]
            coff = (nc + r * C)[:, None]
            if flip.any():
                pv, pe = buf.flip_perms()
                vert = np.where(flip[:, None], buf.corner_vert[pv][None], buf.corner_vert[None])
                edge = np.where(flip[:, None], buf.corner_edge[pe][None], buf.corner_edge[None])
            else:
                vert = np.broadcast_to(buf.corner_vert[None], (M, C))
                edge = np.broadcast_to(buf.corner_edge[None], (M, C))
            cv.append((vert + voff).ravel())
            ce.append((edge + eoff).ravel())
            fs.append((buf.offsets[:-1][None] + coff).ravel())
            ed.append((buf.edges[None] + voff[:, :, None]).reshape(-1, 2))
            fm.append(np.tile(lut[np.clip(buf.face_mat, 0, len(lut) - 1)], M))
            if any_seam:
                seams.append(np.tile(buf.seams if buf.seams is not None else np.zeros(E, bool), M))

            def corners(arr, flip=flip, buf=buf, M=M):
                if flip.any():
                    rev = arr[buf.flip_perms()[0]]
                    shape = (M,) + (1,) * arr.ndim
                    out = np.where(flip.reshape(shape), rev[None], arr[None])
                else:
                    out = np.broadcast_to(arr[None], (M,) + arr.shape)
                return out.reshape((M * arr.shape[0],) + arr.shape[1:])

            for j in range(n_uv):
                if j < len(buf.uvs):
                    uv = corners(buf.uvs[j][1])
                else:
                    uv = np.zeros((M * C, 2), np.float32)
                if uvx is not None and j < len(buf.uvs):
                    u3 = uv.reshape(M, C, 2).astype(np.float64)
                    uv = (np.einsum("mij,mcj->mci", uvx[:, :2, :2], u3) + uvx[:, None, :2, 2]).reshape(-1, 2)
                if j == 0 and u_over is not None:
                    uv = np.array(uv, dtype=np.float32).reshape(M, C, 2)
                    uv[:, :, 0] = np.take_along_axis(np.asarray(u_over, np.float64), vert.astype(np.int64), axis=1)
                    uv = uv.reshape(-1, 2)
                uvs[j].append(np.asarray(uv, np.float32))
            for name, (dom, dt) in spec.items():
                npdt, comps = ATTRIBUTE_TYPES[dt]
                have = buf.attrs.get(name)
                count = buf.count(dom)
                if have is None or have[0] != dom or have[1] != dt:
                    attrs[name].append(np.zeros((M * count,) if comps == 1 else (M * count, comps), npdt))
                elif dom == "CORNER":
                    attrs[name].append(corners(have[2]))
                else:
                    attrs[name].append(np.tile(have[2], (M,) + (1,) * (have[2].ndim - 1)))
            nv += M * V
            ne += M * E
            nc += M * C

        def cat(parts, tail=(), dtype=np.int32):
            return np.concatenate(parts).astype(dtype, copy=False) if parts else np.zeros((0,) + tail, dtype)

        out = Mesh()
        out.co = cat(co, (3,), np.float64)
        out.corner_vert = cat(cv)
        out.corner_edge = cat(ce)
        out.face_start = cat(fs)
        out.edges = cat(ed, (2,))
        out.face_mat = cat(fm)
        out.materials = self.materials or [None]
        out.uvs = [(uv_names[j], cat(uvs[j], (2,), np.float32)) for j in range(n_uv)]
        out.attrs = {}
        for name, (dom, dt) in spec.items():
            npdt, comps = ATTRIBUTE_TYPES[dt]
            out.attrs[name] = (dom, dt, cat(attrs[name], () if comps == 1 else (comps,), npdt))
        out.seams = cat(seams, (), np.bool_) if any_seam else None
        return out


def merge(bufs):
    """Merge several buffers into one (used to cut geometry and for previews)."""
    asm = Assembler()
    for b in bufs:
        asm.add(b, b.co[None])
    m = asm.build()
    offsets = np.append(m.face_start, len(m.corner_vert))
    return MeshBuf(m.co, m.corner_vert, m.corner_edge, offsets, m.edges, m.face_mat, m.materials,
                   m.uvs, m.attrs, m.seams)
