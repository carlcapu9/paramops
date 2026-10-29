"""Voxel finite elements: trilinear 8-node hexahedra on the grid of ``core.grid``.

Linear elasticity on the active voxels of a :class:`~.grid.Problem`, solved
matrix-free with a preconditioned conjugate gradient (PCG). Two preconditioners:

* ``"JACOBI"`` - diagonal scaling (simple baseline),
* ``"MG"`` - geometric multigrid V-cycle (default, ~10x fewer iterations).

Conventions (see ``core/grid.py``):

* element dofs are node-major ``[n0x, n0y, n0z, n1x, ..., n7z]`` with the local
  node order ``HEX_NODE_OFFSETS``; global ``dof = 3 * node + component``,
* stresses / strains use Voigt order ``[xx, yy, zz, xy, yz, zx]`` (engineering
  shear strains), stresses are recovered at the element centroid,
* the stiffness of an element is ``scale_e * KE`` with
  ``KE = E * h * hex8_ke(nu)`` (``scale`` is the SIMP stiffness factor),
* the cube element stiffness ``hex8_ke`` is integrated with 2x2x2 Gauss points.

Multigrid in short (all matrix-free, see :class:`FEModel` and :class:`MGOptions`):

* coarse level = the element grid halved along every axis that still has at least
  two elements (coarse element ``(I, J, K)`` has the children ``(2I+a, 2J+b, 2K+c)``;
  axes that are one element thick are kept: semi-coarsening, so thin plates work),
* re-discretized: ``KE`` of the (box) coarse element, stiffness scale = sum of the
  children scales / number of children slots (missing or inactive children count
  0), coarse element active iff any child is,
* trilinear prolongation ``P`` (per component) and restriction ``P^T`` (exact
  transpose); a coarse dof is fixed when its column of ``P`` touches a fine fixed
  dof of the same component,
* damped Jacobi smoothing, the same number of sweeps before and after the coarse
  correction (symmetric V-cycle); the damping is chosen per level and per solve
  from a Lanczos estimate of ``lambda_max(D^-1 K)`` capped by a rigorous element
  bound, which keeps the V-cycle positive definite also for high-contrast SIMP
  densities (where the classic fixed ``omega = 0.6`` diverges),
* coarsest level (at most ``MGOptions.coarse_dofs`` free dofs) solved exactly with
  a precomputed dense inverse Cholesky factor ``G`` (``K^-1 = G^T G``).

The V-cycle is therefore a symmetric positive definite operator and plain PCG stays
valid; positive definiteness is also checked at run time, and the solver falls back
to Jacobi (``info["fallback"]``) if the hierarchy cannot be built or ever fails.

Speed (see :mod:`.backend` for the choice of device):

* element loops are "structured": the elements are cut into slabs, each slab into
  boxes of up to ~16k elements whose 24 element dofs are gathered with one strided
  view of the sub-grid (vectors are stored component-major inside the solver),
  multiplied by ``KE`` with small batched BLAS calls (single-threaded inside BLAS,
  so there is no oversubscription) and scattered back with 8 strided additions;
  slabs of equal parity touch disjoint nodes and run on a thread pool (numpy
  releases the GIL). Very sparse domains use index chunks instead of boxes. The
  decomposition depends on the geometry only: results are bit-for-bit identical for
  any number of threads,
* vector operations (dot products, updates) are threaded the same way and avoid
  BLAS level-1 calls, whose internal threads are very slow on busy machines; the
  coarsest level is factored with a blocked Cholesky made of small BLAS products
  (LAPACK's threaded routines took seconds per call on a busy machine),
* ``precision="mixed"`` (backend): the V-cycle runs in float32 inside the float64
  conjugate gradient, which then uses the flexible Polak-Ribiere update; the
  stopping test (true relative residual) is unchanged,
* a PyTorch backend runs the same algorithm on a GPU (:mod:`.fem_torch`) and falls
  back to numpy if the device fails.

Measured on one core (shared 4-core container, min of repeats): K u for 250k
elements 90 ms with the former index/bincount loop, 30-47 ms structured float64,
18-26 ms float32; a 250k-element SIMP solve to 1e-6 took 18.8 s before, 9.3 s now
in float64 and 6.5 s mixed (the former code was 2-5x slower still with OpenBLAS's
own threads on a busy machine). At 4 threads ~3.3 slab tasks run concurrently (no
GIL / lock serialization); there the cores were shared with other jobs, so the
wall-clock gain was at most 1.5x; on an idle machine memory bandwidth is the limit.

Everything here keeps no global mutable state: independent :class:`FEModel`
instances may be used from different threads (a single instance must not be shared
by concurrent calls).
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
import threading
import time
from typing import Optional

import numpy as np

from .backend import Backend, cpu_backend, current_backend, using_backend  # noqa: F401
from .grid import HEX_NODE_OFFSETS, Cancelled, Grid, Problem, ProgressFn, StopFn, check_stop


SOLVERS = ("MG", "JACOBI")

# 1-D Gauss-Legendre points of the 2-point rule mapped to [0, 1] (weights 1/2 each).
_GAUSS_01 = 0.5 + np.array([-0.5, 0.5]) / np.sqrt(3.0)

# Names of the six rigid-body modes used by check_problem (same order as the columns).
_RIGID_MODE_NAMES = (
    "translation along X",
    "translation along Y",
    "translation along Z",
    "rotation about X",
    "rotation about Y",
    "rotation about Z",
)


# --------------------------------------------------------------------------- material
def elasticity_matrix(nu: float) -> np.ndarray:
    """Isotropic constitutive matrix ``D`` (6, 6) for a unit Young's modulus.

    ``stress = E * D @ strain`` in Voigt order ``[xx, yy, zz, xy, yz, zx]`` with
    engineering shear strains (``gamma_xy = 2 eps_xy``), so the shear diagonal is
    the shear modulus ``1 / (2 (1 + nu))``.
    """
    nu = float(nu)
    if not (-1.0 < nu < 0.5):
        raise ValueError("Poisson ratio must be in (-1, 0.5)")
    c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.zeros((6, 6), dtype=np.float64)
    D[:3, :3] = c * nu
    D[[0, 1, 2], [0, 1, 2]] = c * (1.0 - nu)
    D[[3, 4, 5], [3, 4, 5]] = 0.5 / (1.0 + nu)
    return D


def _shape_gradients(point, sizes) -> np.ndarray:
    """Cartesian gradients ``dN_a/dx`` (8, 3) of the trilinear shape functions.

    ``point`` are the local coordinates in ``[0, 1]^3`` (fractions of the element
    edges), ``sizes`` the element edge lengths along x, y, z.
    """
    p = np.asarray(point, dtype=np.float64).reshape(1, 3)
    off = HEX_NODE_OFFSETS.astype(np.float64)
    n1 = np.where(off == 1.0, p, 1.0 - p)  # 1-D factors of N_a along each axis
    d1 = (2.0 * off - 1.0) / np.asarray(sizes, dtype=np.float64).reshape(1, 3)
    grad = np.empty((8, 3), dtype=np.float64)
    grad[:, 0] = d1[:, 0] * n1[:, 1] * n1[:, 2]
    grad[:, 1] = n1[:, 0] * d1[:, 1] * n1[:, 2]
    grad[:, 2] = n1[:, 0] * n1[:, 1] * d1[:, 2]
    return grad


def _b_from_gradients(grad: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix (6, 24) from shape-function gradients (8, 3)."""
    B = np.zeros((6, 24), dtype=np.float64)
    gx, gy, gz = grad[:, 0], grad[:, 1], grad[:, 2]
    B[0, 0::3] = gx
    B[1, 1::3] = gy
    B[2, 2::3] = gz
    B[3, 0::3] = gy
    B[3, 1::3] = gx
    B[4, 1::3] = gz
    B[4, 2::3] = gy
    B[5, 0::3] = gz
    B[5, 2::3] = gx
    return B


def _hex8_ke_box(nu: float, sizes) -> np.ndarray:
    """Stiffness (24, 24) of a box element with edge lengths ``sizes`` (E = 1)."""
    sizes = np.asarray(sizes, dtype=np.float64).reshape(3)
    D = elasticity_matrix(nu)
    ke = np.zeros((24, 24), dtype=np.float64)
    for x in _GAUSS_01:
        for y in _GAUSS_01:
            for z in _GAUSS_01:
                B = _b_from_gradients(_shape_gradients((x, y, z), sizes))
                ke += B.T @ D @ B
    ke *= float(np.prod(sizes)) / 8.0  # 8 points, each weight 1/8 of the volume
    return 0.5 * (ke + ke.T)


def hex8_ke(nu: float) -> np.ndarray:
    """Stiffness matrix (24, 24) of the unit cube element (``h = 1``, ``E = 1``).

    2x2x2 Gauss integration, local node order ``HEX_NODE_OFFSETS``, node-major dofs.
    For an element of size ``h`` and modulus ``E`` the stiffness is ``E * h * KE``.
    Symmetric positive semi-definite with exactly six zero eigenvalues (rigid modes).
    """
    return _hex8_ke_box(nu, (1.0, 1.0, 1.0))


def hex8_b_center() -> np.ndarray:
    """Strain-displacement matrix (6, 24) at the centroid of the unit cube.

    Divide by ``h`` for an element of size ``h``: ``strain = (B / h) @ u_e``.
    """
    return _b_from_gradients(_shape_gradients((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)))


# ----------------------------------------------------------------- stress measures
def voigt_to_tensor(s6) -> np.ndarray:
    """Symmetric tensors (..., 3, 3) from stress-like Voigt vectors (..., 6).

    Order ``[xx, yy, zz, xy, yz, zx]``; the shear entries are copied as they are
    (for engineering *strains* halve them first).
    """
    s = np.asarray(s6, dtype=np.float64)
    if s.shape[-1:] != (6,):
        raise ValueError("Voigt arrays must have a trailing dimension of 6")
    t = np.empty(s.shape[:-1] + (3, 3), dtype=np.float64)
    t[..., 0, 0] = s[..., 0]
    t[..., 1, 1] = s[..., 1]
    t[..., 2, 2] = s[..., 2]
    t[..., 0, 1] = t[..., 1, 0] = s[..., 3]
    t[..., 1, 2] = t[..., 2, 1] = s[..., 4]
    t[..., 0, 2] = t[..., 2, 0] = s[..., 5]
    return t


def von_mises(s6) -> np.ndarray:
    """Von Mises equivalent stress (...) of Voigt stresses (..., 6)."""
    s = np.asarray(s6, dtype=np.float64)
    if s.shape[-1:] != (6,):
        raise ValueError("Voigt arrays must have a trailing dimension of 6")
    sxx, syy, szz = s[..., 0], s[..., 1], s[..., 2]
    sxy, syz, szx = s[..., 3], s[..., 4], s[..., 5]
    j2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    j2 += 3.0 * (sxy * sxy + syz * syz + szx * szx)
    return np.sqrt(j2)


def principal(s6) -> tuple:
    """Principal stresses and directions of Voigt stresses (..., 6).

    Returns ``(values, vectors)``: eigenvalues (..., 3) in ascending order (so
    ``values[..., 2]`` is the max principal = tension, ``values[..., 0]`` the min
    principal = compression) and unit eigenvectors (..., 3, 3) stored as columns
    (``vectors[..., :, k]`` belongs to ``values[..., k]``). Tensors containing
    non-finite numbers give NaN values and vectors.
    """
    t = voigt_to_tensor(s6)
    bad = ~np.isfinite(t).all(axis=(-2, -1))
    any_bad = bool(np.any(bad))
    if any_bad:
        t[bad] = 0.0
    values, vectors = np.linalg.eigh(t)
    if any_bad:
        values[bad] = np.nan
        vectors[bad] = np.nan
    return values, vectors


# ------------------------------------------------------------ connected components
def _neighbour_min(a: np.ndarray, connectivity: int) -> np.ndarray:
    """Minimum of ``a`` over each cell and its neighbours (windows clipped at borders)."""

    def axis_slices(axis):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        return tuple(lo), tuple(hi)

    def box_min(src, axes):
        # Separable 3-wide minimum along every axis in ``axes`` (= box window min).
        out = src
        for axis in axes:
            if out.shape[axis] < 2:
                continue
            lo, hi = axis_slices(axis)
            nxt = out.copy()
            np.minimum(nxt[lo], out[hi], out=nxt[lo])
            np.minimum(nxt[hi], out[lo], out=nxt[hi])
            out = nxt
        return out

    if connectivity == 26:
        return box_min(a, (0, 1, 2))
    if connectivity == 18:
        # 3x3x3 window without its 8 corners = union of the three axis-plane 3x3 windows.
        out = box_min(a, (0, 1))
        np.minimum(out, box_min(a, (1, 2)), out=out)
        np.minimum(out, box_min(a, (0, 2)), out=out)
        return out
    out = a.copy()  # 6-connectivity: face neighbours only
    for axis in range(3):
        if a.shape[axis] < 2:
            continue
        lo, hi = axis_slices(axis)
        np.minimum(out[lo], a[hi], out=out[lo])
        np.minimum(out[hi], a[lo], out=out[hi])
    return out


def connected_components(mask, connectivity: int = 26) -> tuple:
    """Label the connected components of a boolean voxel mask (nx, ny, nz).

    Returns ``(labels, n)``: int64 labels shaped like ``mask`` with 0 for
    background and ``1..n`` for the components, numbered in C-order of their
    first voxel. ``connectivity`` is 26 (default: faces, edges and corners, i.e.
    elements sharing at least one node), 18 (faces and edges) or 6 (faces).

    Vectorized label propagation: every voxel takes the minimum label of its
    neighbourhood, the roots of the label trees are hooked to that minimum too
    (union-find style) and pointer jumping flattens the trees. Thanks to the root
    hooking a handful of rounds suffice even for long winding components (measured
    on 100^3 grids: 0.1 - 0.3 s for solid, random and serpentine masks).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask must be a 3-D array")
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18 or 26")
    labels = np.zeros(mask.shape, dtype=np.int64)
    idx = np.flatnonzero(mask)
    m = int(idx.size)
    if m == 0:
        return labels, 0
    itype = np.int32 if m < np.iinfo(np.int32).max else np.int64
    # Labels are positions in the compact list ``idx`` (sorted, so the order of the
    # labels is the C-order of the voxels); ``m`` is the background sentinel.
    lab = np.arange(m, dtype=itype)
    work = np.full(mask.size, m, dtype=itype)
    while True:
        work[idx] = lab
        nmin = _neighbour_min(work.reshape(mask.shape), connectivity).ravel()[idx]
        changed = nmin < lab
        if not changed.any():
            break
        roots = lab[changed]  # ``lab`` is flat here: these are tree roots
        lab = np.minimum(lab, nmin)
        # Hook each root below the smallest neighbouring label of its members
        # (any of the candidate values is valid: all are smaller labels of the
        # same component, so the trees stay acyclic).
        lab[roots] = np.minimum(lab[roots], nmin[changed])
        while True:  # pointer jumping until every voxel points at its root
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt
    is_root = lab == np.arange(m, dtype=itype)
    comp_id = np.cumsum(is_root, dtype=np.int64)
    labels.ravel()[idx] = comp_id[lab]
    return labels, int(comp_id[-1])


def _node_component_labels(grid: Grid, elem_labels: np.ndarray) -> np.ndarray:
    """Component label (n_nodes,) of every node from element labels (0 = no element).

    Elements of different components never share a node (26-connectivity), so the
    maximum over the touching elements is the component of the node.
    """
    nx, ny, nz = grid.shape
    out = np.zeros(grid.nodes_shape, dtype=np.int64)
    for di, dj, dk in HEX_NODE_OFFSETS:
        view = out[di : di + nx, dj : dj + ny, dk : dk + nz]
        np.maximum(view, elem_labels, out=view)
    return out.ravel()


def _valid_fixed_dofs(problem: Problem) -> np.ndarray:
    fixed = problem.fixed_dofs
    if fixed.size and (fixed[0] < 0 or fixed[-1] >= problem.grid.n_dofs):
        raise ValueError("fixed_dofs contains dof ids outside the grid")
    return fixed


def remove_unsupported_components(problem: Problem) -> tuple:
    """Drop the parts of the domain that touch no fixed dof.

    Parts are 26-connected components of ``problem.domain`` (elements sharing a
    node). A part without any fixed dof would make the stiffness matrix singular.
    Returns ``(new_problem, n_removed)``; the forces on the nodes of removed parts
    are set to zero, everything else is kept. When nothing is removed - or when no
    part is supported at all (nothing sensible to keep; ``check_problem`` reports
    it) - the input ``problem`` itself is returned with ``n_removed == 0``.
    """
    labels, n = connected_components(problem.domain, 26)
    if n <= 1:
        return problem, 0
    fixed = _valid_fixed_dofs(problem)
    node_labels = _node_component_labels(problem.grid, labels)
    supported = np.zeros(n + 1, dtype=bool)
    supported[node_labels[fixed // 3]] = True
    supported[0] = False
    n_supported = int(np.count_nonzero(supported))
    if n_supported == 0 or n_supported == n:
        return problem, 0
    keep = supported[labels]
    removed_nodes = ~supported[node_labels] & (node_labels > 0)
    forces = problem.forces.copy()
    forces.reshape(-1, 3)[removed_nodes] = 0.0
    return replace(problem, domain=keep, forces=forces), n - n_supported


# ------------------------------------------------------------------- problem check
def _rigid_mode_matrix(coords: np.ndarray, comps: np.ndarray, centre, length: float):
    """Rows (n, 6) of the six rigid modes evaluated at the given dofs.

    ``coords`` (n, 3) are the node positions of the dofs, ``comps`` (n,) their
    components; rotations are about ``centre`` with lever arms divided by ``length``
    so that every column has a comparable magnitude.
    """
    r = (coords - centre) / length
    n = r.shape[0]
    A = np.zeros((n, 6), dtype=np.float64)
    rows = np.arange(n)
    A[rows, comps] = 1.0
    # columns 3..5: (e_x x r), (e_y x r), (e_z x r) evaluated at component ``comp``
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    rot = np.stack(
        [
            np.stack([np.zeros(n), -rz, ry], axis=1),  # e_x x r
            np.stack([rz, np.zeros(n), -rx], axis=1),  # e_y x r
            np.stack([-ry, rx, np.zeros(n)], axis=1),  # e_z x r
        ],
        axis=1,
    )  # (n, 3 modes, 3 components)
    A[:, 3:] = rot[rows, :, comps]
    return A


def _free_rigid_modes(A: np.ndarray, rtol: float = 1e-6) -> list:
    """Describe the null space of the restricted rigid-mode matrix (empty = restrained)."""
    if A.shape[0] == 0:
        return list(_RIGID_MODE_NAMES)
    # ``full_matrices`` only for fewer rows than modes (U stays small, vt is 6 x 6).
    _, s, vt = np.linalg.svd(A, full_matrices=A.shape[0] < 6)
    smax = float(s[0]) if s.size else 0.0
    rank = int(np.count_nonzero(s > rtol * smax)) if smax > 0.0 else 0
    if rank >= 6:
        return []
    null = vt[rank:]  # (k, 6) orthonormal basis of the free rigid motions
    names = []
    # Translations that are completely free are named directly.
    trans_free = [a for a in range(3) if not np.any(A[:, a] != 0.0)]
    names.extend(_RIGID_MODE_NAMES[a] for a in trans_free)
    # The remaining free motions are rotations (possibly combined with a shift):
    # name them after the axis of their rotation part.
    if trans_free:
        q = np.eye(6)[trans_free]  # remove the pure translations from the basis
        null = null - (null @ q.T) @ q
        u, sv, wt = np.linalg.svd(null, full_matrices=False)
        null = wt[sv > 1e-8]
    for v in null:
        w = v[3:]
        nw = float(np.linalg.norm(w))
        if nw < 1e-8:
            names.append("translation")
            continue
        w = w / nw
        axis = int(np.argmax(np.abs(w)))
        if abs(w[axis]) > 0.95:
            names.append(_RIGID_MODE_NAMES[3 + axis])
        else:
            names.append("rotation about an inclined axis")
    return names


def check_problem(problem: Problem) -> list:
    """Human-readable warnings about a problem (empty list = looks solvable).

    Checks: empty domain, missing supports, missing / zero / invalid loads, loads
    acting only on fixed (or outside the domain) nodes, rigid-body mechanisms (for
    each supported part: the six rigid modes - 3 translations, 3 rotations about
    the part's centroid - restricted to its fixed dofs must have rank 6) and
    disconnected parts that touch no support.
    """
    warnings = []
    grid = problem.grid
    domain = problem.domain
    n_vox = int(np.count_nonzero(domain))
    if n_vox == 0:
        return ["The design domain is empty: no voxel lies inside the domain mesh."]
    fixed = _valid_fixed_dofs(problem)
    labels, n_parts = connected_components(domain, 26)
    node_labels = _node_component_labels(grid, labels)
    dom_nodes = node_labels > 0
    fixed_in = fixed[dom_nodes[fixed // 3]]

    # ---------------------------------------------------------------- loads
    # Only loaded nodes matter (usually a handful): keep the bookkeeping small.
    f3 = problem.forces.reshape(-1, 3)
    loaded = np.flatnonzero(np.any(f3 != 0.0, axis=1))  # NaN rows count as loaded
    fl = f3[loaded]
    finite = np.isfinite(fl).all(axis=1)
    if not finite.all():
        warnings.append(
            "Some loads are not valid numbers (NaN or infinite); check the load magnitudes."
        )
    fl = np.where(finite[:, None], fl, 0.0)
    fnorm = np.sqrt(np.einsum("ij,ij->i", fl, fl))
    on_dom = dom_nodes[loaded]
    total = float(fnorm.sum())
    on_domain = float(fnorm[on_dom].sum())
    if total <= 0.0:
        warnings.append("No loads: the applied force is zero, nothing will deform.")
    elif on_domain <= 1e-12 * total:
        warnings.append("All loads lie outside the domain and are ignored.")
    else:
        if on_domain < total * (1.0 - 1e-9):
            pct = 100.0 * (1.0 - on_domain / total)
            warnings.append(f"{pct:.0f}% of the load lies outside the domain and is ignored.")
        fixed_mask = np.zeros(grid.n_dofs, dtype=bool)
        fixed_mask[fixed_in] = True
        fixed_l = fixed_mask.reshape(-1, 3)[loaded]
        on_fixed_part = np.where(fixed_l, fl, 0.0)
        on_fixed = float(np.sqrt(np.einsum("ij,ij->i", on_fixed_part, on_fixed_part))[on_dom].sum())
        free_part = np.where(fixed_l, 0.0, fl)
        if not np.any(free_part[on_dom] != 0.0):
            warnings.append(
                "All loads act on supported (fixed) nodes, so nothing will deform: "
                "move the loads away from the supports."
            )
        elif on_fixed > 0.01 * on_domain:
            pct = 100.0 * on_fixed / on_domain
            warnings.append(
                f"{pct:.0f}% of the load acts directly on supported nodes and is absorbed there."
            )

    # ------------------------------------------------------------- supports
    if fixed_in.size == 0:
        warnings.append("No supports: at least one support must touch the domain.")
        return warnings
    part_of_dof = node_labels[fixed_in // 3]
    supported = np.zeros(n_parts + 1, dtype=bool)
    supported[part_of_dof] = True
    supported[0] = False
    part_sizes = np.bincount(labels.ravel(), minlength=n_parts + 1)

    order = np.argsort(part_of_dof, kind="stable")
    bounds = np.searchsorted(part_of_dof[order], np.arange(1, n_parts + 2))
    many = int(np.count_nonzero(supported)) > 1
    # Centroids of the parts (mean element centre), from the voxel indices only.
    vox = np.flatnonzero(domain.ravel())
    vox_ijk = np.unravel_index(vox, grid.shape)
    vox_lab = labels.ravel()[vox]
    centroids = np.stack(
        [np.bincount(vox_lab, vox_ijk[a], minlength=n_parts + 1) for a in range(3)], axis=1
    )
    centroids = grid.origin + (centroids / np.maximum(part_sizes, 1)[:, None] + 0.5) * grid.h
    length = float(np.max(grid.size))
    n_mech_reported = 0
    for part in np.flatnonzero(supported):
        dofs = fixed_in[order[bounds[part - 1] : bounds[part]]]
        nodes = np.stack(np.unravel_index(dofs // 3, grid.nodes_shape), axis=1)
        coords = grid.origin + nodes * grid.h
        A = _rigid_mode_matrix(coords, dofs % 3, centroids[part], length)
        free_modes = _free_rigid_modes(A)
        if not free_modes:
            continue
        n_mech_reported += 1
        if n_mech_reported > 5:
            continue
        where = f"Part {int(part)} ({int(part_sizes[part])} voxels): s" if many else "S"
        warnings.append(
            f"{where}upports do not prevent rigid motion (free: {', '.join(free_modes)}); "
            "add supports or fix more axes."
        )
    if n_mech_reported > 5:
        warnings.append(f"... and {n_mech_reported - 5} more parts that can move freely.")

    unsupported = np.flatnonzero(~supported[1:]) + 1
    if unsupported.size:
        n_un = int(unsupported.size)
        vox = int(part_sizes[unsupported].sum())
        un_mask = np.zeros(n_parts + 1, dtype=bool)
        un_mask[unsupported] = True
        load_un = float(fnorm[un_mask[node_labels[loaded]]].sum())
        msg = (
            f"{n_un} disconnected part{'s' if n_un > 1 else ''} of the domain ({vox} voxels) "
            f"{'touch' if n_un > 1 else 'touches'} no support and will be ignored"
        )
        if load_un > 0.0 and on_domain > 0.0:
            msg += f" (carrying {100.0 * load_un / on_domain:.0f}% of the load)"
        warnings.append(msg + ".")
    return warnings



# ----------------------------------------------------------------- matrix free core
# Elements per block in the indexed element loops (``_apply_elements`` and the
# diagonal): bounds the (block, 24) temporaries to ~13 MB whatever the model size.
_ELEM_BLOCK = 1 << 16


def _blocks(n: int):
    for start in range(0, n, _ELEM_BLOCK):
        yield start, min(n, start + _ELEM_BLOCK)


def _scatter_block(y: np.ndarray, edof_blk: np.ndarray, values: np.ndarray) -> None:
    """``y[dofs] += values`` for a block of consecutive (sorted) elements.

    Element dofs grow with the C-order element id and inside an element the
    smallest / largest dof are columns 0 / 20 (local nodes (0,0,0) / (1,1,1)), so a
    block only touches the band ``[edof_blk[0, 0], edof_blk[-1, 20]]``.
    """
    lo = int(edof_blk[0, 0])
    hi = int(edof_blk[-1, 20]) + 1
    y[lo:hi] += np.bincount((edof_blk - lo).ravel(), weights=values.ravel(), minlength=hi - lo)


def _apply_elements(x, edof, edof_flat, ke, scale, n_dofs):
    """Assembled ``sum_e scale_e * KE`` times ``x`` (node-major, no boundary masking).

    Reference implementation of the element loop (``ue = x[edof]`` -> ``fe = (ue @
    KE) * scale`` -> ``bincount``); the solver uses the faster :class:`_ElementLoops`.
    """
    n = edof.shape[0]
    if n <= _ELEM_BLOCK:
        fe = x[edof] @ ke
        fe *= scale[:, None]
        return np.bincount(edof_flat, weights=fe.ravel(), minlength=n_dofs)
    y = np.zeros(n_dofs, dtype=np.float64)
    for s, t in _blocks(n):
        blk = edof[s:t]
        fe = x[blk] @ ke
        fe *= scale[s:t, None]
        _scatter_block(y, blk, fe)
    return y


def _assembled_diagonal(edof, ke_diag, scale, n_dofs):
    """Diagonal of the assembled ``sum_e scale_e * KE`` (n_dofs,), node-major."""
    y = np.zeros(n_dofs, dtype=np.float64)
    for s, t in _blocks(edof.shape[0]):
        _scatter_block(y, edof[s:t], scale[s:t, None] * ke_diag[None, :])
    return y


# -------------------------------------------------------------- threads and layout
# The work decomposition of the threaded kernels depends on the model geometry only,
# never on the number of threads: results are bit-for-bit identical for any thread
# count (and deterministic from run to run).
_SLAB_TARGET = 16384  # elements per box of the structured kernel
_SERIAL_ELEMS = 32768  # levels with at most this many grid elements run in one task
_MIN_SLABS = 16  # large levels are cut into at least this many slabs (8 per colour) ...
_SLAB_MIN_ELEMS = 8192  # ... unless slabs would get smaller than this (thread overhead)
_GEMM_COLS = 128  # columns per BLAS call: 24 * 24 * 128 keeps OpenBLAS single-threaded
_MIN_BOX_FILL = 0.35  # structured kernel when active / computed elements >= this
_VEC_CHUNK = 1 << 17  # elements per chunk of the threaded vector operations
_PARALLEL_TRANSFER_NODES = 1 << 16  # restriction / prolongation threaded from this size
# The kernels number the 8 local nodes lexicographically, (di, dj, dk) = (L>>2, L>>1&1,
# L&1), so that one strided view gathers all of them; ``_LEX_DOFS`` maps those element
# dofs to the HEX_NODE_OFFSETS order of ``KE`` (``KE_lex = KE[_LEX_DOFS][:, _LEX_DOFS]``).
_LOCAL_NODES = tuple((di, dj, dk) for di in (0, 1) for dj in (0, 1) for dk in (0, 1))
_LEX_DOFS = np.array(
    [
        3 * [tuple(int(v) for v in o) for o in HEX_NODE_OFFSETS].index(off) + c
        for off in _LOCAL_NODES
        for c in range(3)
    ],
    dtype=np.int64,
)


def _lex_ke(ke: np.ndarray) -> np.ndarray:
    """Element stiffness with rows / columns in the kernels' local node order."""
    return np.ascontiguousarray(ke[np.ix_(_LEX_DOFS, _LEX_DOFS)])


class _Workers:
    """Runs independent tasks on a small thread pool (serially for one thread).

    numpy releases the GIL inside its loops, so tasks working on disjoint memory run
    in parallel. The calling thread takes part in the work and tasks are handed out
    dynamically (robust against busy cores). Threads are created on first use and
    end with the owner (or :meth:`shutdown`).
    """

    def __init__(self, threads: int):
        self.threads = max(1, int(threads))
        self._pool = None
        self._lock = threading.Lock()

    def run(self, fn, items) -> None:
        n = len(items)
        if n == 0:
            return
        if self.threads == 1 or n == 1:
            for item in items:
                fn(item)
            return
        lock = threading.Lock()
        state = {"next": 0}
        errors = []

        def drain():
            while True:
                with lock:
                    i = state["next"]
                    if i >= n or errors:
                        return
                    state["next"] = i + 1
                try:
                    fn(items[i])
                except BaseException as exc:  # re-raised by the caller
                    errors.append(exc)
                    return

        pool = self._executor()
        futures = [pool.submit(drain) for _ in range(min(self.threads, n) - 1)]
        drain()
        for future in futures:
            future.result()
        if errors:
            raise errors[0]

    def _executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.threads - 1, thread_name_prefix="stress_sculpt_fem"
                )
            return self._pool

    def shutdown(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False)


class _VecOps:
    """Deterministic threaded vector operations on fixed chunks of ``_VEC_CHUNK``.

    Dot products avoid BLAS (whose own threads oversubscribe the cores and are very
    slow on busy machines) and sum per-chunk partials in a fixed order.
    """

    def __init__(self, workers: _Workers):
        self.workers = workers

    @staticmethod
    def _bounds(n: int) -> list:
        return [(s, min(n, s + _VEC_CHUNK)) for s in range(0, n, _VEC_CHUNK)]

    def dot(self, a: np.ndarray, b: np.ndarray) -> float:
        n = a.shape[0]
        if n <= _VEC_CHUNK:
            return float(np.einsum("i,i->", a, b))
        bounds = self._bounds(n)
        part = np.empty(len(bounds), dtype=np.float64)

        def task(i):
            s, t = bounds[i]
            part[i] = np.einsum("i,i->", a[s:t], b[s:t])

        self.workers.run(task, range(len(bounds)))
        return float(part.sum())

    def norm(self, a: np.ndarray) -> float:
        return float(np.sqrt(max(self.dot(a, a), 0.0)))

    def each(self, fn, *arrays) -> None:
        """``fn(*chunks)`` on aligned chunks of 1-D arrays (elementwise work only)."""
        n = arrays[0].shape[0]
        if self.workers.threads == 1 or n <= 2 * _VEC_CHUNK:
            fn(*arrays)
            return
        self.workers.run(lambda st: fn(*[a[st[0] : st[1]] for a in arrays]), self._bounds(n))


def _ew_mul(y, f):  # y *= f
    np.multiply(y, f, out=y)


def _ew_mul_to(out, a, b):  # out = a * b
    np.multiply(a, b, out=out)


def _ew_jacobi(x, d, r, ax):  # x += d * (r - ax)
    t = r - ax
    t *= d
    x += t


def _ew_residual(ax, r, f):  # ax = (r - ax) * f
    np.subtract(r, ax, out=ax)
    ax *= f


def _ew_add_masked(x, c, f):  # x += c * f
    c *= f
    x += c


def _ew_copy(dst, src):
    np.copyto(dst, src)


def _ew_scaled_cast(dst, src, factor):  # dst = src * factor (float64 math, dst's precision)
    dt = np.promote_types(dst.dtype, src.dtype)
    np.multiply(src, factor, out=dst, dtype=dt, casting="same_kind")


class _Buffers:
    """Scratch ``(24, width)`` array pairs of the element kernels (one per running task)."""

    def __init__(self):
        self._free = {}
        self._lock = threading.Lock()

    def get(self, dtype, width: int) -> tuple:
        key = np.dtype(dtype).str
        with self._lock:
            stack = self._free.setdefault(key, [])
            while stack:
                pair = stack.pop()
                if pair[0].shape[1] >= width:
                    return pair
        width = max(int(width), _GEMM_COLS)
        # zero-filled: padding columns stay finite (they are multiplied, never used)
        return np.zeros((24, width), dtype=dtype), np.zeros((24, width), dtype=dtype)

    def put(self, pair: tuple) -> None:
        with self._lock:
            self._free.setdefault(pair[0].dtype.str, []).append(pair)


def _soa(v, dtype=None) -> np.ndarray:
    """Node-major dof vector ``[n0x, n0y, n0z, n1x, ...]`` -> component-major copy."""
    a = np.asarray(v).reshape(-1, 3).T
    return np.ascontiguousarray(a, dtype=dtype).reshape(-1)


def _aos(v, dtype=None) -> np.ndarray:
    """Component-major dof vector -> node-major copy (inverse of :func:`_soa`)."""
    a = np.asarray(v).reshape(3, -1).T
    return np.ascontiguousarray(a, dtype=dtype).reshape(-1)


def _soa_index(dofs, n_nodes: int) -> np.ndarray:
    """Component-major positions of node-major dof ids."""
    d = np.asarray(dofs, dtype=np.int64)
    return (d % 3) * int(n_nodes) + d // 3


def _gemm(mat: np.ndarray, ue: np.ndarray, fe: np.ndarray) -> None:
    """``fe[:k] = mat @ ue`` for ``(24, cols)`` blocks (``cols`` multiple of _GEMM_COLS).

    Issued as a batch of small BLAS calls: each stays single-threaded inside OpenBLAS
    (no oversubscription with our own threads), works without BLAS too.
    """
    k = mat.shape[0]
    nb = ue.shape[1] // _GEMM_COLS
    np.matmul(
        mat,
        ue.reshape(24, nb, _GEMM_COLS).transpose(1, 0, 2),
        out=fe[:k].reshape(k, nb, _GEMM_COLS).transpose(1, 0, 2),
    )


def _padded(m: int) -> int:
    return max(_GEMM_COLS, -(-int(m) // _GEMM_COLS) * _GEMM_COLS)


class _Box:
    """Sub-grid of elements handled at once by the structured kernel."""

    __slots__ = ("shape", "m", "cols", "elem", "origin", "nodes", "ue_shape")

    def __init__(self, lo, hi):
        self.shape = tuple(int(h - l) for l, h in zip(lo, hi))
        self.m = int(np.prod(self.shape))
        self.cols = _padded(self.m)
        self.elem = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        self.origin = (slice(None),) + tuple(slice(int(l), None) for l in lo)
        self.nodes = tuple(
            (slice(None),) + tuple(slice(int(l) + o, int(h) + o) for l, h, o in zip(lo, hi, off))
            for off in _LOCAL_NODES
        )
        self.ue_shape = (2, 2, 2, 3) + self.shape

    def node_view(self, x4: np.ndarray) -> np.ndarray:
        """Read-only view ``(2, 2, 2, 3, t0, t1, t2)``: [di, dj, dk, c, i, j, k] =
        ``x4[c, i0 + i + di, j0 + j + dj, k0 + k + dk]`` (all 8 local nodes at once)."""
        base = x4[self.origin]
        sc, sx, sy, sz = base.strides
        return np.lib.stride_tricks.as_strided(
            base, shape=self.ue_shape, strides=(sx, sy, sz, sc, sx, sy, sz), writeable=False
        )


class _Slab:
    """Elements of one slab: boxes (structured) or index chunks (indexed)."""

    __slots__ = ("boxes", "chunks", "cols")

    def __init__(self):
        self.boxes = []
        self.chunks = []  # (positions in the active list, (8, n) local-node ids, cols)
        self.cols = _GEMM_COLS


class _ElementLoops:
    """Element loops over the active elements of one grid level.

    Vectors are component-major ("SoA"): ``v.reshape(3, nx + 1, ny + 1, nz + 1)``.
    The elements are cut into slabs along the longest axis; slabs of equal parity
    touch disjoint nodes, so each colour runs in parallel without write conflicts and
    the summation order is fixed. Inside a slab the work is done per box (structured
    kernel: strided gathers / scatters of whole sub-grids, no index arrays; inactive
    elements of a box have scale 0) or, when the boxes would be mostly empty (sparse
    domains), per chunk of element indices.
    """

    def __init__(self, shape, elems, workers: _Workers, buffers: _Buffers):
        self.shape = tuple(int(s) for s in shape)
        self.nodes_shape = tuple(s + 1 for s in self.shape)
        self.n_nodes = int(np.prod(self.nodes_shape))
        self.n_elems = int(np.prod(self.shape))
        self.elems = np.asarray(elems, dtype=np.int64)
        self.n_act = int(self.elems.size)
        self.workers = workers
        self.buffers = buffers
        self.structured = True
        self.fill = 1.0
        self.colors = ([], [])
        self.slabs = []
        if self.n_act:
            self._plan()

    # ---------------------------------------------------------------- planning
    def _plan(self) -> None:
        shape = self.shape
        # Slabs along x or y, whichever is longer, so that boxes keep the whole z extent
        # as the contiguous inner loop; z only for thin columns.
        ax = 0 if shape[0] >= shape[1] else 1
        if shape[ax] < 8 and shape[2] > 2 * shape[ax]:
            ax = 2
        b1, b2 = [a for a in range(3) if a != ax]
        n_ax = shape[ax]
        cross = shape[b1] * shape[b2]
        if self.n_elems <= _SERIAL_ELEMS:
            t = n_ax  # small level: one slab, no threading overhead
        else:
            n_slabs = max(2, min(_MIN_SLABS, self.n_elems // _SLAB_MIN_ELEMS))
            t = max(1, min(_SLAB_TARGET // max(1, cross), -(-n_ax // n_slabs)))
        act = np.zeros(self.n_elems, dtype=bool)
        act[self.elems] = True
        act = act.reshape(shape)
        # rows of the b1 axis per box so that a box has about _SLAB_TARGET elements
        rows = max(1, _SLAB_TARGET // max(1, t * shape[b2]))
        slabs = []
        computed = 0
        for lo in range(0, n_ax, t):
            hi = min(n_ax, lo + t)
            slab = _Slab()
            for r0 in range(0, shape[b1], rows):
                r1 = min(shape[b1], r0 + rows)
                region = [slice(None)] * 3
                region[ax] = slice(lo, hi)
                region[b1] = slice(r0, r1)
                sub = act[tuple(region)]
                if not sub.any():
                    continue
                blo, bhi = [0, 0, 0], [0, 0, 0]
                for a in range(3):
                    proj = np.flatnonzero(sub.any(axis=tuple(b for b in range(3) if b != a)))
                    base = (lo, r0, 0)[[ax, b1, b2].index(a)]
                    blo[a] = base + int(proj[0])
                    bhi[a] = base + int(proj[-1]) + 1
                box = _Box(blo, bhi)
                slab.boxes.append(box)
                computed += box.m
            if slab.boxes:
                slab.cols = max(b.cols for b in slab.boxes)
            slabs.append(slab)  # every slab, also empty ones: the parity is its position
        self.fill = self.n_act / float(max(1, computed))
        if self.fill < _MIN_BOX_FILL:
            self.structured = False
            self._plan_indexed(ax, t, slabs)
        for si, slab in enumerate(slabs):
            if slab.boxes or slab.chunks:
                self.colors[si % 2].append(slab)
                self.slabs.append(slab)

    def _plan_indexed(self, ax: int, t: int, slabs: list) -> None:
        ijk = np.unravel_index(self.elems, self.shape)
        slab_of = ijk[ax] // t
        order = np.argsort(slab_of, kind="stable")
        bounds = np.searchsorted(slab_of[order], np.arange(len(slabs) + 1))
        base = np.ravel_multi_index(ijk, self.nodes_shape)
        strides = np.array(
            [self.nodes_shape[1] * self.nodes_shape[2], self.nodes_shape[2], 1], dtype=np.int64
        )
        local = np.array([int(np.dot(off, strides)) for off in _LOCAL_NODES], dtype=np.int64)
        for k, slab in enumerate(slabs):  # slab k holds the elements with slab_of == k
            slab.boxes = []
            pos_all = order[bounds[k] : bounds[k + 1]]
            for s in range(0, pos_all.size, _SLAB_TARGET):
                pos = np.ascontiguousarray(pos_all[s : s + _SLAB_TARGET])
                enode = base[pos][None, :] + local[:, None]  # (8, n) node ids
                slab.chunks.append((pos, enode, _padded(pos.size)))
            if slab.chunks:
                slab.cols = max(c[2] for c in slab.chunks)

    # ----------------------------------------------------------------- kernels
    def scale_context(self, scale_act, dtype) -> np.ndarray:
        """Per-solve element scales in the layout :meth:`apply` expects."""
        if self.structured:
            g = np.zeros(self.n_elems, dtype=dtype)
            g[self.elems] = scale_act
            return g.reshape(self.shape)
        return np.ascontiguousarray(scale_act, dtype=dtype)

    def apply(self, x: np.ndarray, sc: np.ndarray, ke: np.ndarray, out: np.ndarray) -> np.ndarray:
        """``out = sum_e s_e KE x_e`` (SoA vectors of ``ke``'s dtype, nothing masked).

        ``ke`` must be in the kernels' local node order (:func:`_lex_ke`).
        """
        out.fill(0.0)
        if not self.n_act:
            return out
        x4 = x.reshape((3,) + self.nodes_shape)
        y4 = out.reshape((3,) + self.nodes_shape)
        task = self._apply_boxes if self.structured else self._apply_chunks
        for color in self.colors:
            self.workers.run(lambda slab: task(slab, x4, y4, sc, ke), color)
        return out

    def _apply_boxes(self, slab, x4, y4, sgrid, ke) -> None:
        ue_b, fe_b = self.buffers.get(ke.dtype, slab.cols)
        try:
            for box in slab.boxes:
                ue = ue_b[:, : box.cols]
                fe = fe_b[:, : box.cols]
                gathered = ue[:, : box.m].reshape(box.ue_shape)
                np.multiply(box.node_view(x4), sgrid[box.elem], out=gathered)
                _gemm(ke, ue, fe)
                fe8 = fe[:, : box.m].reshape((8, 3) + box.shape)
                for a in range(8):
                    yv = y4[box.nodes[a]]
                    np.add(yv, fe8[a], out=yv)
        finally:
            self.buffers.put((ue_b, fe_b))

    def _apply_chunks(self, slab, x4, y4, scale, ke) -> None:
        x3 = x4.reshape(3, -1)
        y3 = y4.reshape(3, -1)
        ue_b, fe_b = self.buffers.get(ke.dtype, slab.cols)
        try:
            for pos, enode, cols in slab.chunks:
                n = pos.size
                ue = ue_b[:, :cols]
                fe = fe_b[:, :cols]
                for a in range(8):
                    ue[3 * a : 3 * a + 3, :n] = x3[:, enode[a]]
                ue[:, :n] *= scale[pos]
                _gemm(ke, ue, fe)
                for a in range(8):
                    y3[:, enode[a]] += fe[3 * a : 3 * a + 3, :n]  # unique ids per local node
        finally:
            self.buffers.put((ue_b, fe_b))

    def element_values(self, x: np.ndarray, mat: np.ndarray, energy: bool = False) -> np.ndarray:
        """Per active element ``mat @ x_e`` -> (n_act, k), or ``x_e^T mat x_e`` -> (n_act,).

        ``x`` is a float64 SoA vector, ``mat`` (k, 24) (``(24, 24)`` for energies).
        """
        k = 1 if energy else mat.shape[0]
        if not self.n_act:
            return np.zeros(0) if energy else np.zeros((0, k))
        x4 = x.reshape((3,) + self.nodes_shape)
        mat = np.asarray(mat, dtype=np.float64)
        mat = _lex_ke(mat) if energy else np.ascontiguousarray(mat[:, _LEX_DOFS])
        if self.structured:
            grid = np.zeros((k,) + self.shape, dtype=np.float64)

            def task(slab):
                ue_b, fe_b = self.buffers.get(np.float64, slab.cols)
                try:
                    for box in slab.boxes:
                        ue = ue_b[:, : box.cols]
                        fe = fe_b[:, : box.cols]
                        np.copyto(ue[:, : box.m].reshape(box.ue_shape), box.node_view(x4))
                        _gemm(mat, ue, fe)
                        dst = grid[(slice(None),) + box.elem]
                        if energy:
                            vals = np.einsum("ij,ij->j", ue[:, : box.m], fe[:, : box.m])
                            dst[0] = vals.reshape(box.shape)
                        else:
                            dst[...] = fe[:k, : box.m].reshape((k,) + box.shape)
                finally:
                    self.buffers.put((ue_b, fe_b))

            self.workers.run(task, self.slabs)
            out = grid.reshape(k, -1)[:, self.elems]
        else:
            out = np.zeros((k, self.n_act), dtype=np.float64)
            x3 = x4.reshape(3, -1)

            def task(slab):
                ue_b, fe_b = self.buffers.get(np.float64, slab.cols)
                try:
                    for pos, enode, cols in slab.chunks:
                        n = pos.size
                        ue = ue_b[:, :cols]
                        fe = fe_b[:, :cols]
                        for a in range(8):
                            ue[3 * a : 3 * a + 3, :n] = x3[:, enode[a]]
                        _gemm(mat, ue, fe)
                        if energy:
                            out[0, pos] = np.einsum("ij,ij->j", ue[:, :n], fe[:, :n])
                        else:
                            out[:, pos] = fe[:k, :n]
                finally:
                    self.buffers.put((ue_b, fe_b))

            self.workers.run(task, self.slabs)
        return out[0] if energy else np.ascontiguousarray(out.T)


# ------------------------------------------------------------------- multigrid
@dataclass(frozen=True)
class MGOptions:
    """Tuning knobs of the multigrid preconditioner (defaults are well tested).

    * ``sweeps`` - damped-Jacobi sweeps before and after each coarse correction,
    * ``omega`` - fixed Jacobi damping for every level, or ``None`` (automatic,
      recommended): ``omega = omega_scale / lam`` per level and per solve, with
      ``lam = min(lam_bound, lambda_margin * lam_lanczos)``. ``lam_lanczos`` is a
      ``lanczos_steps`` Lanczos estimate of ``lambda_max(D^-1 K)`` (always from
      below) and ``lam_bound`` the element bound (always from above, see
      ``_element_lambda_bound``). The V-cycle is positive definite iff
      ``omega * lambda_max < 2``: guaranteed when the bound is active, and otherwise
      as long as the estimate is less than ~25 % low (measured: < 8 %); the PCG
      verifies it at run time and falls back to Jacobi if it ever fails.
      (A fixed ``omega = 0.6`` is NOT safe for SIMP densities: there
      ``lambda_max`` reaches ~4.7 for nu = 0.3, versus ~3.1 for a solid body.)
    * ``coarse_dofs`` - a level (other than the finest) with at most this many free
      dofs is the coarsest one, solved with a dense Cholesky inverse,
    * ``max_levels`` - hierarchy depth cap (including the fine level),
    * ``min_axis_elements`` - an axis is halved only while it has more elements than
      this (default 1: thin plates are coarsened through their thickness down to
      one element, which measured 3x faster than keeping 2 elements; axes that are
      already one element thick keep their size: semi-coarsening).
    """

    sweeps: int = 2
    omega: Optional[float] = None
    omega_scale: float = 1.75
    lambda_margin: float = 1.1
    lanczos_steps: int = 10
    coarse_dofs: int = 1000
    max_levels: int = 10
    min_axis_elements: int = 1


class _MGSetupError(RuntimeError):
    """The multigrid hierarchy cannot be used (the solver falls back to Jacobi)."""


@dataclass
class _Level:
    """Geometry of one multigrid level (independent of the stiffness scales)."""

    shape: tuple
    sizes: np.ndarray  # element edge lengths (3,)
    elems: np.ndarray  # active element ids (C-order of ``shape``)
    edof: np.ndarray
    edof_flat: np.ndarray
    ke: np.ndarray  # stiffness at scale 1 (includes E)
    fixed: np.ndarray  # (n_dofs,) bool, node-major, Dirichlet dofs (inherited on coarse levels)
    free: np.ndarray  # (n_dofs,) bool, node-major, touched by an active element and not fixed
    lambda_bound: float  # upper bound of lambda_max(D^-1 K) for any element scales
    factors: tuple = (1, 1, 1)  # coarsening factors towards the next level
    loops: Optional[_ElementLoops] = None  # element kernels (component-major vectors)

    @property
    def nodes_shape(self) -> tuple:
        return tuple(s + 1 for s in self.shape)

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.nodes_shape))

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def n_elems(self) -> int:
        return int(np.prod(self.shape))


@dataclass
class _LevelOp:
    """Scale-dependent part of a level for one solve (vectors component-major)."""

    scale: np.ndarray  # (n_act,) float64
    free: np.ndarray  # (n_dofs,) bool: geometric free dofs with a positive diagonal
    freef: np.ndarray  # same as ``dtype`` (multiplicative mask)
    dinv: np.ndarray  # omega / diagonal on free dofs, 0 elsewhere (``dtype``)
    omega: float
    dtype: type = np.float64  # precision of the smoother / transfers on this level
    ke: Optional[np.ndarray] = None  # level stiffness in ``dtype``
    sc: Optional[np.ndarray] = None  # element scales in the kernel layout (``dtype``)
    coarse: Optional["_DenseCholesky"] = None  # coarsest level: Cholesky factor of K
    coarse_idx: Optional[np.ndarray] = None  # coarsest level: free dof ids (component-major)

    @property
    def coarse_inv(self) -> Optional[np.ndarray]:
        """Inverse Cholesky factor ``G`` (``K^-1 = G^T G``) of the coarsest level (tests)."""
        return None if self.coarse is None else self.coarse.inverse_factor()


_CHOL_BLOCK = 48  # block size: 48^3 multiply-adds per BLAS call stay single-threaded


class _DenseCholesky:
    """Blocked Cholesky factor ``K = L L^T`` of a dense SPD matrix, blocked solves.

    Written with batches of small BLAS products on the model's threads instead of
    LAPACK: LAPACK's internal threads stall on busy machines (seconds for a 400 x 400
    matrix) and builds without an optimized LAPACK are slow. Raises
    ``np.linalg.LinAlgError`` when ``K`` is not positive definite. The matrix is
    padded with an identity block to a multiple of ``_CHOL_BLOCK``.
    """

    def __init__(self, K: np.ndarray, workers: _Workers):
        n = int(K.shape[0])
        B = _CHOL_BLOCK
        nb = max(1, -(-n // B))
        full = np.eye(nb * B)
        full[:n, :n] = K
        L = np.ascontiguousarray(full.reshape(nb, B, nb, B).transpose(0, 2, 1, 3))
        inv_diag = np.empty((nb, B, B))
        for k in range(nb):
            lkk = np.linalg.cholesky(L[k, k])
            ikk = np.linalg.inv(lkk)
            if not np.all(np.isfinite(ikk)):
                raise np.linalg.LinAlgError("non-finite Cholesky factor")
            L[k, k] = lkk
            inv_diag[k] = ikk
            if k + 1 == nb:
                break
            panel = np.matmul(L[k + 1 :, k], ikk.T)  # L_ik = K_ik L_kk^-T
            L[k + 1 :, k] = panel

            def update(cols, k=k, panel=panel):  # trailing block columns: K_ij -= L_ik L_jk^T
                for j in cols:
                    L[j:, j] -= np.matmul(panel[j - k - 1 :], panel[j - k - 1].T)

            cols = range(k + 1, nb)
            step = len(cols) if len(cols) < 16 else 4  # small updates: no thread overhead
            workers.run(update, [cols[i : i + step] for i in range(0, len(cols), step)])
        self.n = n
        self.nb = nb
        self.blocks = L  # (nb, nb, B, B), lower block triangle valid
        self.inv_diag = inv_diag

    def solve(self, b: np.ndarray) -> np.ndarray:
        """``K^-1 b`` (float64) by blocked forward / backward substitution."""
        nb, L = self.nb, self.blocks
        y = np.zeros((nb, _CHOL_BLOCK))
        y.reshape(-1)[: self.n] = b
        for k in range(nb):
            if k:
                y[k] -= np.einsum("jab,jb->a", L[k, :k], y[:k])
            y[k] = np.einsum("ab,b->a", self.inv_diag[k], y[k])
        for k in range(nb - 1, -1, -1):
            if k + 1 < nb:
                y[k] -= np.einsum("jba,jb->a", L[k + 1 :, k], y[k + 1 :])
            y[k] = np.einsum("ba,b->a", self.inv_diag[k], y[k])
        return y.reshape(-1)[: self.n].copy()

    def inverse_factor(self) -> np.ndarray:
        """``G = L^-1`` (n, n) as a dense array (diagnostics; O(n^3))."""
        nb, B = self.nb, _CHOL_BLOCK
        full = np.zeros((nb * B, nb * B))
        for i in range(nb):
            for j in range(i + 1):
                full[i * B : (i + 1) * B, j * B : (j + 1) * B] = self.blocks[i, j]
        full = np.tril(full)
        return np.tril(np.linalg.inv(full)[: self.n, : self.n])


def _element_lambda_bound(ke: np.ndarray) -> float:
    """Upper bound of ``lambda_max(D^-1 K)`` valid for ANY non-negative element scales.

    For ``K = sum_e s_e KE`` and ``D = diag(K) = sum_e s_e diag(KE)``:
    ``x^T K x <= lam_e x^T D x`` with ``lam_e = lambda_max(diag(KE)^-1 KE)``
    (element by element), also for any principal sub-matrix (free dofs). So damped
    Jacobi with ``omega < 2 / lam_e`` always makes ``2 D / omega - K`` positive
    definite, i.e. the V-cycle SPD. The bound is tight for strongly varying scales
    (SIMP) but pessimistic for uniform ones (5.32 against 3.1 for nu = 0.3).
    """
    s = 1.0 / np.sqrt(np.diag(ke))
    return float(np.linalg.eigvalsh(s[:, None] * ke * s[None, :])[-1])


def _coarsen_sum(values: np.ndarray, factors: tuple, coarse_shape: tuple) -> np.ndarray:
    """Sum of the children of every coarse element (missing children count 0)."""
    padded = np.zeros(tuple(c * f for c, f in zip(coarse_shape, factors)), dtype=values.dtype)
    nx, ny, nz = values.shape
    padded[:nx, :ny, :nz] = values
    cx, cy, cz = coarse_shape
    fx, fy, fz = factors
    return padded.reshape(cx, fx, cy, fy, cz, fz).sum(axis=(1, 3, 5))


def _prolong_axis(a: np.ndarray, axis: int, n_fine: int) -> np.ndarray:
    """1-D linear interpolation along ``axis`` from coarse to ``n_fine`` nodes."""
    src = np.moveaxis(a, axis, 0)
    n_even = (n_fine + 1) // 2
    n_odd = n_fine // 2
    out = np.empty((n_fine,) + src.shape[1:], dtype=a.dtype)
    out[0::2] = src[:n_even]
    odd = out[1::2]
    np.add(src[:n_odd], src[1 : n_odd + 1], out=odd)
    odd *= 0.5
    return np.moveaxis(out, 0, axis)


def _restrict_axis(a: np.ndarray, axis: int, n_coarse: int) -> np.ndarray:
    """Exact transpose of :func:`_prolong_axis`."""
    src = np.moveaxis(a, axis, 0)
    n_fine = src.shape[0]
    n_even = (n_fine + 1) // 2
    n_odd = n_fine // 2
    out = np.zeros((n_coarse,) + src.shape[1:], dtype=a.dtype)
    out[:n_even] += src[0::2]
    half = src[1::2] * 0.5
    out[:n_odd] += half
    out[1 : n_odd + 1] += half
    return np.moveaxis(out, 0, axis)


def _prolong(xc: np.ndarray, fine: _Level, coarse: _Level) -> np.ndarray:
    """Trilinear prolongation of a node-major dof vector (reference / tests)."""
    a = xc.reshape(coarse.nodes_shape + (3,))
    for axis in range(3):
        if fine.factors[axis] == 2:
            a = _prolong_axis(a, axis, fine.nodes_shape[axis])
    return a.reshape(-1)


def _restrict(rf: np.ndarray, fine: _Level, coarse: _Level) -> np.ndarray:
    """Restriction (exact transpose of :func:`_prolong`) of a node-major vector."""
    a = rf.reshape(fine.nodes_shape + (3,))
    for axis in range(3):
        if fine.factors[axis] == 2:
            a = _restrict_axis(a, axis, coarse.nodes_shape[axis])
    return a.reshape(-1)


def _prolong_soa(xc: np.ndarray, fine: _Level, coarse: _Level, workers: _Workers) -> np.ndarray:
    """:func:`_prolong` for component-major vectors (the 3 components in parallel)."""
    src = xc.reshape((3,) + coarse.nodes_shape)
    out = np.empty((3,) + fine.nodes_shape, dtype=xc.dtype)

    def component(c):
        a = src[c]
        for axis in range(3):
            if fine.factors[axis] == 2:
                a = _prolong_axis(a, axis, fine.nodes_shape[axis])
        out[c] = a

    _run_components(workers, component, fine.n_nodes)
    return out.reshape(-1)


def _run_components(workers: _Workers, fn, n_nodes: int) -> None:
    """The 3 components in parallel on big levels, serially on small ones (dispatching
    to the pool costs more than the work there)."""
    if n_nodes >= _PARALLEL_TRANSFER_NODES:
        workers.run(fn, (0, 1, 2))
    else:
        for c in range(3):
            fn(c)


def _restrict_soa(rf: np.ndarray, fine: _Level, coarse: _Level, workers: _Workers) -> np.ndarray:
    """:func:`_restrict` for component-major vectors (the 3 components in parallel)."""
    src = rf.reshape((3,) + fine.nodes_shape)
    out = np.empty((3,) + coarse.nodes_shape, dtype=rf.dtype)

    def component(c):
        a = src[c]
        for axis in range(3):
            if fine.factors[axis] == 2:
                a = _restrict_axis(a, axis, coarse.nodes_shape[axis])
        out[c] = a

    _run_components(workers, component, fine.n_nodes)
    return out.reshape(-1)


def _coarsen_level(fine: _Level, factors: tuple, E: float, nu: float) -> _Level:
    """Geometry of the next coarser level (``fine.factors`` must be set to ``factors``)."""
    shape_c = tuple(-(-n // f) for n, f in zip(fine.shape, factors))
    active_f = np.zeros(fine.n_elems, dtype=np.int32)
    active_f[fine.elems] = 1
    active_c = _coarsen_sum(active_f.reshape(fine.shape), factors, shape_c) > 0
    elems_c = np.flatnonzero(active_c).astype(np.int64)
    connectivity = Grid(shape_c, (0.0, 0.0, 0.0), 1.0)
    edof_c = connectivity.element_dofs(elems_c)
    edof_flat_c = edof_c.ravel()
    level_c = _Level(
        shape=shape_c,
        sizes=fine.sizes * np.asarray(factors, dtype=np.float64),
        elems=elems_c,
        edof=edof_c,
        edof_flat=edof_flat_c,
        ke=E * _hex8_ke_box(nu, fine.sizes * np.asarray(factors, dtype=np.float64)),
        fixed=np.zeros(0, dtype=bool),
        free=np.zeros(0, dtype=bool),
        lambda_bound=0.0,
    )
    # A coarse dof is fixed when its prolongation column touches a fine fixed dof
    # of the same component: restriction of the fixed indicator is then positive.
    fixed_c = _restrict(fine.fixed.astype(np.float64), fine, level_c) > 0.0
    touched_c = np.zeros(level_c.n_dofs, dtype=bool)
    touched_c[edof_flat_c] = True
    level_c.fixed = fixed_c
    level_c.free = touched_c & ~fixed_c
    level_c.lambda_bound = _element_lambda_bound(level_c.ke)
    return level_c


def _lanczos_lambda_max(apply, v0, d_isqrt, steps: int, should_stop=None, dot=None) -> float:
    """Lanczos estimate (from below) of ``lambda_max(D^-1 A)``.

    ``v0`` is the start vector (zero on the dofs to ignore), ``d_isqrt`` = ``D^-1/2``
    (zero there too), ``dot`` the inner product (default: plain einsum).
    """
    if dot is None:
        dot = lambda a, b: float(np.einsum("i,i->", a, b))  # noqa: E731
    v = np.array(v0, copy=True)
    nv = float(np.sqrt(max(dot(v, v), 0.0)))
    if nv == 0.0:
        return 0.0
    v /= nv
    v_prev = np.zeros_like(v)
    beta = 0.0
    alphas, betas = [], []
    for _ in range(max(1, steps)):
        check_stop(should_stop)
        w = d_isqrt * apply(d_isqrt * v)
        alpha = dot(w, v)
        w -= alpha * v + beta * v_prev
        alphas.append(alpha)
        beta = float(np.sqrt(max(dot(w, w), 0.0)))
        if not np.isfinite(beta) or beta <= 1e-12 * max(abs(alpha), 1e-300):
            break
        betas.append(beta)
        v_prev, v = v, w / beta
    k = len(alphas)
    T = np.diag(alphas) + np.diag(betas[: k - 1], 1) + np.diag(betas[: k - 1], -1)
    return float(np.linalg.eigvalsh(T)[-1])


_COARSE_REGULARIZATION = (1e-12, 1e-10, 1e-8, 1e-6)  # relative to mean(diag(K))


def _assemble_dense(level: _Level, scale: np.ndarray, free_nm: np.ndarray) -> tuple:
    """Dense stiffness of the free dofs of a (coarsest) level.

    Returns ``(idx, K)``: node-major free dof ids and the symmetric matrix (``None``
    when there is no free dof); raises ``_MGSetupError`` for a degenerate matrix.
    """
    idx = np.flatnonzero(free_nm)
    nf = int(idx.size)
    if nf == 0:
        return idx, None
    g2f = np.full(level.n_dofs, -1, dtype=np.int64)
    g2f[idx] = np.arange(nf)
    loc = g2f[level.edof]  # (n_act, 24) free index or -1
    rows = np.broadcast_to(loc[:, :, None], (loc.shape[0], 24, 24))
    cols = np.broadcast_to(loc[:, None, :], (loc.shape[0], 24, 24))
    ok = (rows >= 0) & (cols >= 0)
    vals = scale[:, None, None] * level.ke[None, :, :]
    K = np.bincount((rows[ok] * nf + cols[ok]), weights=vals[ok], minlength=nf * nf)
    K = K.reshape(nf, nf)
    K = 0.5 * (K + K.T)
    mean_diag = float(np.mean(np.diag(K)))
    if not (mean_diag > 0.0 and np.isfinite(mean_diag)):
        raise _MGSetupError("coarsest multigrid matrix is degenerate")
    return idx, K


def _torch_failure_message(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    detail = text[0][:300] if text else ""
    kind = type(exc).__name__
    if "out of memory" in detail.lower():
        return f"the GPU ran out of memory ({detail}); solved on the CPU instead"
    return f"the GPU solve failed ({kind}: {detail}); solved on the CPU instead"


# --------------------------------------------------------------------------- model
class FEModel:
    """Matrix-free voxel FE model of a :class:`~.grid.Problem`.

    Public attributes (read-only by convention):

    * ``problem``, ``grid``
    * ``elems`` (n_act,) int64 flat ids of the active elements (``problem.domain``)
    * ``edof`` (n_act, 24) int64 global dofs of the active elements
    * ``KE`` (24, 24) ``problem.E * grid.h * hex8_ke(nu)``
    * ``free`` (n_dofs,) bool, dofs of nodes touched by an active element and not
      fixed
    * ``fixed`` (n_dofs,) bool, the fixed dofs of the problem
    * ``solver`` ``"MG"`` or ``"JACOBI"``, ``emin`` minimum stiffness scale
    * ``backend`` the :class:`~.backend.Backend` (numpy threads / PyTorch device and
      precision); ``threads`` its CPU thread count

    ``backend=None`` uses the backend set with :func:`~.backend.using_backend` in the
    calling thread, else numpy on all cores. A PyTorch backend solves on its device
    and falls back to numpy (sticky for this model, reported in ``info``) when the
    device fails, e.g. out of memory.

    The multigrid geometry is built lazily on the first MG solve and reused; the
    scale dependent parts are rebuilt when the scales change. A model owns a few
    worker threads (created on first use); a single instance must not be used by
    concurrent calls.
    """

    def __init__(
        self,
        problem: Problem,
        solver: str = "MG",
        emin: float = 1e-3,
        *,
        mg_options: Optional[MGOptions] = None,
        backend: Optional[Backend] = None,
    ):
        solver = str(solver).upper()
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver {solver!r} (expected one of {SOLVERS})")
        if backend is None:
            backend = current_backend()
        if backend is None:
            backend = cpu_backend()
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a core.backend.Backend")
        self.backend = backend
        self.problem = problem
        self.grid = problem.grid
        self.solver = solver
        self.emin = float(emin)
        self.mg_options = mg_options if mg_options is not None else MGOptions()
        g = self.grid
        self.n_dofs = g.n_dofs
        self.elems = np.flatnonzero(problem.domain.ravel()).astype(np.int64)
        self.edof = g.element_dofs(self.elems)
        self._edof_flat = self.edof.ravel()
        self.KE = problem.E * g.h * hex8_ke(problem.nu)
        self._ke_lex = _lex_ke(self.KE)  # local node order of the element kernels
        self._ke_diag = np.diag(self.KE).copy()
        fixed = np.zeros(self.n_dofs, dtype=bool)
        fixed[_valid_fixed_dofs(problem)] = True
        self.fixed = fixed
        touched = np.zeros(self.n_dofs, dtype=bool)
        touched[self._edof_flat] = True
        self._touched = touched
        self.free = touched & ~fixed
        self._freef = self.free.astype(np.float64)
        # stress = scale * E * D @ (B_center / h) @ u_e
        self._stress_op = (problem.E / g.h) * (elasticity_matrix(problem.nu) @ hex8_b_center())
        self._workers = _Workers(backend.threads)
        self._vec = _VecOps(self._workers)
        self._buffers = _Buffers()
        self._loops = None  # fine-level element kernels (lazy)
        self._levels = None  # multigrid geometry (lazy)
        self._mg_error = None  # reason why multigrid is unavailable
        self._ops_scale = None  # scale the cached level operators were built for
        self._ops_precision = None
        self._ops = None
        self._torch = None  # fem_torch solver state (lazy)
        self._torch_error = None  # set when the torch backend failed (sticky fallback)

    @property
    def n_act(self) -> int:
        return int(self.elems.size)

    @property
    def threads(self) -> int:
        return self._workers.threads

    def release_device(self) -> None:
        """Free device memory held for the PyTorch backend (it is rebuilt on demand)."""
        state, self._torch = self._torch, None
        if state is not None:
            try:
                state.release()
            except Exception:
                pass

    def _fine_loops(self) -> _ElementLoops:
        if self._loops is None:
            self._loops = _ElementLoops(self.grid.shape, self.elems, self._workers, self._buffers)
        return self._loops

    # ------------------------------------------------------------------ basics
    def stiffness_scale(self, rho_act, penal: float) -> np.ndarray:
        """SIMP stiffness factors ``emin + (1 - emin) * rho**penal`` (n_act,).

        ``rho`` is clipped to [0, 1] (avoids NaN for tiny negative round-off with
        non-integer penalties).
        """
        rho = np.clip(np.asarray(rho_act, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if rho.shape[0] != self.n_act:
            raise ValueError("rho_act must have one value per active element")
        return self.emin + (1.0 - self.emin) * rho ** float(penal)

    def _check_scale(self, scale) -> np.ndarray:
        if scale is None:
            return np.ones(self.n_act, dtype=np.float64)
        s = np.asarray(scale, dtype=np.float64).reshape(-1)
        if s.shape[0] != self.n_act:
            raise ValueError("scale must have one value per active element")
        if s.size and not (np.all(np.isfinite(s)) and s.min() >= 0.0):
            raise ValueError("stiffness scales must be finite and non-negative")
        return s

    def _check_vector(self, u, name="u") -> np.ndarray:
        v = np.asarray(u, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.n_dofs:
            raise ValueError(f"{name} must have n_dofs = {self.n_dofs} entries")
        return v

    def _soa_input(self, u) -> np.ndarray:
        """Component-major copy of ``u`` with untouched dofs zeroed (the structured
        kernel reads the nodes of inactive elements, multiplied by 0)."""
        return _soa(np.where(self._touched, u, 0.0))

    def apply_K(self, u, scale) -> np.ndarray:
        """Matrix-free product ``K(scale) @ u`` (n_dofs,), zero on non-free dofs."""
        u = self._check_vector(u)
        scale = self._check_scale(scale)
        loops = self._fine_loops()
        y = loops.apply(
            self._soa_input(u),
            loops.scale_context(scale, np.float64),
            self._ke_lex,
            np.empty(self.n_dofs, dtype=np.float64),
        )
        y = _aos(y)
        y *= self._freef
        return y

    def diagonal(self, scale) -> np.ndarray:
        """Diagonal of the assembled stiffness ``K(scale)`` (n_dofs,), all dofs."""
        scale = self._check_scale(scale)
        return _assembled_diagonal(self.edof, self._ke_diag, scale, self.n_dofs)

    def element_energy(self, u) -> np.ndarray:
        """Element strain energies ``u_e^T KE u_e`` (n_act,), without the scales.

        The compliance of a solution is ``sum(scale * element_energy(u))``.
        """
        u = self._check_vector(u)
        return self._fine_loops().element_values(self._soa_input(u), self.KE, energy=True)

    def element_stress(self, u, scale) -> np.ndarray:
        """Physical stress (n_act, 6) at the element centroids, Voigt order.

        ``scale_e * E * D(nu) @ (B_center / h) @ u_e``.
        """
        u = self._check_vector(u)
        scale = self._check_scale(scale)
        out = self._fine_loops().element_values(self._soa_input(u), self._stress_op)
        out *= scale[:, None]
        return out

    def to_grid(self, values_act, fill=0.0) -> np.ndarray:
        """Scatter per-element values (n_act, ...) onto the grid (nx, ny, nz, ...)."""
        v = np.asarray(values_act)
        if v.ndim == 0 or v.shape[0] != self.n_act:
            raise ValueError("values_act must have one row per active element")
        dtype = np.result_type(v.dtype, np.asarray(fill).dtype)
        out = np.full((self.grid.n_elems,) + v.shape[1:], fill, dtype=dtype)
        out[self.elems] = v
        return out.reshape(self.grid.shape + v.shape[1:])

    # --------------------------------------------------------------- multigrid
    def _mg_geometry(self) -> list:
        """Build (once) the geometric multigrid levels; raises ``_MGSetupError``."""
        if self._levels is not None:
            return self._levels
        if self._mg_error is not None:
            raise _MGSetupError(self._mg_error)
        opts = self.mg_options
        g = self.grid
        fine = _Level(
            shape=g.shape,
            sizes=np.full(3, g.h, dtype=np.float64),
            elems=self.elems,
            edof=self.edof,
            edof_flat=self._edof_flat,
            ke=self.KE,
            fixed=self.fixed,
            free=self.free,
            lambda_bound=_element_lambda_bound(self.KE),
            loops=self._fine_loops(),
        )
        levels = [fine]
        E, nu = self.problem.E, self.problem.nu
        while len(levels) < max(1, int(opts.max_levels)):
            cur = levels[-1]
            n_free = int(np.count_nonzero(cur.free))
            factors = tuple(2 if n > opts.min_axis_elements else 1 for n in cur.shape)
            if factors == (1, 1, 1) or n_free == 0:
                break
            if len(levels) > 1 and n_free <= opts.coarse_dofs:
                break
            cur.factors = factors
            levels.append(_coarsen_level(cur, factors, E, nu))
        n_coarse = int(np.count_nonzero(levels[-1].free))
        if n_coarse > max(3 * opts.coarse_dofs, 3000):  # dense inverse would be too big
            self._mg_error = f"coarsest multigrid level too large ({n_coarse} dofs)"
            raise _MGSetupError(self._mg_error)
        levels[-1].factors = (1, 1, 1)
        for level in levels[1:-1]:  # the coarsest level is solved densely, no kernel
            level.loops = _ElementLoops(level.shape, level.elems, self._workers, self._buffers)
        self._levels = levels
        return levels

    def _level_apply(self, level: _Level, op: _LevelOp, x: np.ndarray) -> np.ndarray:
        y = level.loops.apply(x, op.sc, op.ke, np.empty(level.n_dofs, dtype=op.dtype))
        self._vec.each(_ew_mul, y, op.freef)
        return y

    def _level_scales(self, levels: list, scale: np.ndarray) -> list:
        """Element scales of every level (coarse = children sum / children slots)."""
        scales = [scale]
        for li in range(1, len(levels)):
            prev, level = levels[li - 1], levels[li]
            full = np.zeros(prev.n_elems, dtype=np.float64)
            full[prev.elems] = scales[-1]
            summed = _coarsen_sum(full.reshape(prev.shape), prev.factors, level.shape)
            scales.append(summed.ravel()[level.elems] / float(np.prod(prev.factors)))
        return scales

    def _mg_operators(self, scale, diag, precision: str = "float64", should_stop=None) -> list:
        """Scale-dependent level operators (cached for identical scales / precision)."""
        if (
            self._ops is not None
            and self._ops_precision == precision
            and np.array_equal(self._ops_scale, scale)
        ):
            return self._ops
        levels = self._mg_geometry()
        opts = self.mg_options
        dt = np.float32 if precision == "mixed" else np.float64
        scales = self._level_scales(levels, scale)
        ops = []
        for li, level in enumerate(levels):
            check_stop(should_stop)
            scale_l = scales[li]
            if li == 0:
                diag_l = diag
            else:
                diag_l = _assembled_diagonal(level.edof, np.diag(level.ke), scale_l, level.n_dofs)
            free_l = level.free & (diag_l > 0.0)  # node-major
            free_s = _soa(free_l)
            d_inv = free_s / np.where(free_s, _soa(diag_l), 1.0)
            op = _LevelOp(
                scale=scale_l,
                free=free_s,
                freef=free_s.astype(dt),
                dinv=d_inv,
                omega=0.0,
                dtype=dt,
            )
            if li == len(levels) - 1:
                self._factor_coarsest(level, op, free_l)
            else:
                op.ke = _lex_ke(level.ke).astype(dt)
                op.sc = level.loops.scale_context(scale_l, dt)
                if opts.omega is not None:
                    omega = float(opts.omega)
                else:
                    rng = np.random.default_rng(1234567)  # local: deterministic, thread-safe
                    v0 = (_soa(rng.standard_normal(level.n_dofs)) * free_s).astype(dt)
                    lam = _lanczos_lambda_max(
                        lambda v, lv=level, o=op: self._level_apply(lv, o, v),
                        v0,
                        np.sqrt(d_inv).astype(dt),
                        opts.lanczos_steps,
                        should_stop,
                        dot=self._vec.dot,
                    )
                    lam_used = level.lambda_bound
                    if np.isfinite(lam) and lam > 0.0:
                        # lambda_max(D^-1 K) >= 1 always (unit Rayleigh quotients).
                        lam_used = min(lam_used, opts.lambda_margin * max(lam, 1.0))
                    omega = opts.omega_scale / lam_used
                op.omega = omega
                op.dinv = (omega * d_inv).astype(dt)
            ops.append(op)
        self._ops = ops
        self._ops_scale = scale.copy()
        self._ops_precision = precision
        return ops

    def _factor_coarsest(self, level: _Level, op: _LevelOp, free_nm: np.ndarray) -> None:
        idx, K = _assemble_dense(level, op.scale, free_nm)
        op.coarse_idx = _soa_index(idx, level.n_nodes)
        op.coarse = None
        if K is None:
            return
        mean_diag = float(np.mean(np.diag(K)))
        eye = np.eye(K.shape[0])
        for reg in _COARSE_REGULARIZATION:
            try:
                op.coarse = _DenseCholesky(K + (reg * mean_diag) * eye, self._workers)
            except np.linalg.LinAlgError:
                continue
            return
        raise _MGSetupError("coarsest multigrid matrix is not positive definite")

    def _mg_preconditioner(self, scale, precision: str = "float64"):
        """The V-cycle ``r -> M r`` (node-major vectors) for the given scales (tests)."""
        scale = self._check_scale(scale)
        diag = self.diagonal(scale)
        levels = self._mg_geometry()
        ops = self._mg_operators(scale, diag, precision)
        freef = ops[0].freef.astype(np.float64)
        dt = ops[0].dtype

        def M(r):
            rs = (_soa(np.asarray(r, dtype=np.float64)) * freef).astype(dt)
            return _aos(self._vcycle(levels, ops, 0, rs), dtype=np.float64)

        return M

    def _vcycle(self, levels, ops, li: int, r: np.ndarray, should_stop=None) -> np.ndarray:
        """Symmetric V-cycle on level ``li`` for the residual ``r`` (zero initial guess).

        Component-major vectors in the level's precision (``ops[li].dtype``).
        """
        level, op = levels[li], ops[li]
        if li == len(levels) - 1:
            x = np.zeros(level.n_dofs, dtype=op.dtype)
            if op.coarse is not None:
                x[op.coarse_idx] = op.coarse.solve(r[op.coarse_idx].astype(np.float64))
            return x
        vec = self._vec
        sweeps = max(1, int(self.mg_options.sweeps))
        n = level.n_dofs
        x = np.empty(n, dtype=op.dtype)
        vec.each(_ew_mul_to, x, op.dinv, r)  # first sweep from a zero initial guess
        ax = np.empty(n, dtype=op.dtype)
        for _ in range(sweeps - 1):
            level.loops.apply(x, op.sc, op.ke, ax)
            vec.each(_ew_jacobi, x, op.dinv, r, ax)
        if li == 0:
            # Poll between the fine-level phases (a fine V-cycle is 4 fine products).
            check_stop(should_stop)
        level.loops.apply(x, op.sc, op.ke, ax)
        vec.each(_ew_residual, ax, r, op.freef)  # ax <- masked residual
        coarse, cop = levels[li + 1], ops[li + 1]
        rc = _restrict_soa(ax, level, coarse, self._workers)
        vec.each(_ew_mul, rc, cop.freef)
        xc = self._vcycle(levels, ops, li + 1, rc)
        corr = _prolong_soa(xc, level, coarse, self._workers)
        vec.each(_ew_add_masked, x, corr, op.freef)
        if li == 0:
            check_stop(should_stop)
        for _ in range(sweeps):
            level.loops.apply(x, op.sc, op.ke, ax)
            vec.each(_ew_jacobi, x, op.dinv, r, ax)
        return x

    # ------------------------------------------------------------------- solve
    def solve(
        self,
        scale,
        x0=None,
        tol: float = 1e-6,
        maxiter: int = 4000,
        should_stop: Optional[StopFn] = None,
        progress: Optional[ProgressFn] = None,
    ) -> tuple:
        """Solve ``K(scale) u = f`` with PCG on the free dofs.

        Returns ``(u, info)``: ``u`` (n_dofs,) with zeros on non-free dofs and
        ``info = {"iterations", "residual" (true relative residual |f - K u| / |f|
        on the free dofs), "converged", "solver" ("MG" | "JACOBI": the
        preconditioner that produced the result), "time" (s), "backend"
        (description of the device), "precision" ("float64" | "mixed"), "threads"}``
        plus ``"fallback"`` (reason) when multigrid was abandoned for Jacobi,
        ``"backend_fallback"`` (reason) when the PyTorch device failed and numpy
        solved instead, and ``"breakdown"`` when the iteration stopped on a
        non-positive curvature / non-finite value.

        ``x0`` (n_dofs,) is a warm start (ignored if it is not finite or worse than
        zero). Zero load or no free dof return zeros at once; dofs whose stiffness is
        zero (all touching elements have scale 0) are held at zero. Never raises on
        non-convergence (``converged`` is False; the iterate with the smallest
        residual is returned, always finite). Raises ``ValueError`` for invalid
        input (wrong sizes, negative / non-finite scales, non-finite forces) and
        ``Cancelled`` when ``should_stop()`` returns True (polled every iteration);
        ``progress(fraction, message)`` is called about every 0.2 s.

        With ``precision == "mixed"`` the multigrid V-cycle runs in float32 and the
        float64 conjugate gradient uses the flexible (Polak-Ribiere) update, which
        stays robust when the preconditioner is not exactly symmetric; the stopping
        test is the same float64 true residual.
        """
        t_start = time.perf_counter()
        scale = self._check_scale(scale)
        forces = self.problem.forces
        if not np.all(np.isfinite(forces)):
            raise ValueError("forces contain non-finite values")
        if x0 is not None:
            x0 = self._check_vector(x0, "x0")
        backend_fallback = None
        be = self.backend
        if be.is_torch and self._torch_error is None and self.n_act >= be.min_elements:
            try:
                from . import fem_torch

                u, info = fem_torch.solve(self, scale, x0, tol, maxiter, should_stop, progress)
            except Cancelled:
                raise
            except Exception as exc:  # torch missing / CUDA error / out of memory ...
                backend_fallback = _torch_failure_message(exc)
                self._torch_error = backend_fallback
                self.release_device()
            else:
                info["time"] = time.perf_counter() - t_start
                return u, info
        u, info = self._solve_numpy(scale, x0, tol, maxiter, should_stop, progress)
        if backend_fallback is not None:
            info["backend_fallback"] = backend_fallback
        elif self._torch_error is not None:
            info["backend_fallback"] = self._torch_error
        info["time"] = time.perf_counter() - t_start
        return u, info

    def _solve_numpy(self, scale, x0, tol, maxiter, should_stop, progress) -> tuple:
        tol = float(tol)
        maxiter = max(0, int(maxiter))
        precision = self.backend.precision
        info = {
            "iterations": 0,
            "residual": 0.0,
            "converged": True,
            "solver": self.solver,
            "backend": f"CPU ({self.threads} thread{'s' if self.threads != 1 else ''})",
            "threads": self.threads,
            "precision": "float64",
        }
        u = np.zeros(self.n_dofs, dtype=np.float64)

        diag = self.diagonal(scale)
        free = self.free & (diag > 0.0)
        b_nm = np.where(free, self.problem.forces, 0.0)
        bnorm = float(np.sqrt(np.einsum("i,i->", b_nm, b_nm)))
        if bnorm == 0.0 or not free.any():
            return u, info
        vec = self._vec
        n = self.n_dofs
        loops = self._fine_loops()
        free_s = _soa(free)
        freef = free_s.astype(np.float64)
        b = _soa(b_nm)
        jacobi_dinv = freef / np.where(free_s, _soa(diag), 1.0)
        sc64 = loops.scale_context(scale, np.float64)
        ke64 = self._ke_lex

        def apply(v):
            y = loops.apply(v, sc64, ke64, np.empty(n, dtype=np.float64))
            vec.each(_ew_mul, y, freef)
            return y

        def jacobi(v):
            z = np.empty(n, dtype=np.float64)
            vec.each(_ew_mul_to, z, jacobi_dinv, v)
            return z

        precond = jacobi
        flexible = False
        info["solver"] = "JACOBI"
        if self.solver == "MG":
            if progress is not None:
                progress(0.0, "Preparing the multigrid solver")
            try:
                levels = self._mg_geometry()
                ops = self._mg_operators(scale, diag, precision, should_stop)
            except _MGSetupError as exc:
                info["fallback"] = str(exc)
            else:
                if precision == "mixed":
                    r32 = np.empty(n, dtype=np.float32)

                    def precond(v):  # float32 V-cycle of the normalized residual
                        nrm = vec.norm(v)
                        z = np.empty(n, dtype=np.float64)
                        if not (nrm > 0.0 and np.isfinite(nrm)):
                            z.fill(0.0 if nrm == 0.0 else np.nan)
                            return z
                        vec.each(partial(_ew_scaled_cast, factor=1.0 / nrm), r32, v)
                        z32 = self._vcycle(levels, ops, 0, r32, should_stop)
                        vec.each(partial(_ew_scaled_cast, factor=nrm), z, z32)
                        return z

                    flexible = True
                    info["precision"] = "mixed"
                else:
                    precond = lambda v: self._vcycle(levels, ops, 0, v, should_stop)  # noqa: E731
                info["solver"] = "MG"

        # ------------------------------------------------------------ initial guess
        x = np.zeros(n, dtype=np.float64)
        r = b.copy()
        if x0 is not None and np.all(np.isfinite(x0)):
            xw = _soa(x0) * freef
            rw = b - apply(xw)
            if vec.norm(rw) < bnorm:
                x, r = xw, rw
        rel = vec.norm(r) / bnorm
        best_x, best_rel = x.copy(), rel  # returned when the iteration fails
        rel_true = rel  # true relative residual of ``x`` when known, else None
        it = 0
        restarts = 0
        breakdown = False
        last_report = time.perf_counter()
        log_tol = np.log(tol) if 0.0 < tol < 1.0 else None

        while rel > tol and it < maxiter:
            # (Re)start the PCG recursion from the current iterate.
            z = precond(r)
            rz = vec.dot(r, z)
            if not (np.isfinite(rz) and rz > 0.0):
                if info["solver"] != "MG":
                    breakdown = True
                    break
                precond = jacobi
                flexible = False
                info["solver"] = "JACOBI"
                info["precision"] = "float64"
                info["fallback"] = "multigrid preconditioner was not positive definite"
                continue
            p = z.copy()
            switch = False
            rel_true = None
            while True:
                check_stop(should_stop)
                q = apply(p)
                pq = vec.dot(p, q)
                if not (np.isfinite(pq) and pq > 0.0):
                    breakdown = True  # K is singular along p (mechanism) or not finite
                    break
                alpha = rz / pq
                vec.each(partial(_cg_update, alpha=alpha), x, r, p, q)
                it += 1
                rel = vec.norm(r) / bnorm
                if not np.isfinite(rel):
                    breakdown = True
                    break
                if rel < best_rel:
                    best_rel = rel
                    vec.each(_ew_copy, best_x, x)
                if rel <= tol or it >= maxiter:
                    break
                z = precond(r)
                rz_new = vec.dot(r, z)
                if not (np.isfinite(rz_new) and rz_new > 0.0):
                    if info["solver"] == "MG":
                        precond = jacobi
                        flexible = False
                        info["solver"] = "JACOBI"
                        info["precision"] = "float64"
                        info["fallback"] = "multigrid preconditioner was not positive definite"
                        switch = True
                    else:
                        breakdown = True
                    break
                if flexible:
                    # Polak-Ribiere: beta = z_new . (r_new - r_old) / rz = -alpha z_new . q / rz
                    beta = -alpha * vec.dot(z, q) / rz
                else:
                    beta = rz_new / rz
                vec.each(partial(_xpby, beta=beta), p, z)
                rz = rz_new
                if progress is not None:
                    now = time.perf_counter()
                    if now - last_report > 0.2:
                        last_report = now
                        frac = 0.0
                        if log_tol is not None and rel > 0.0:
                            frac = float(np.clip(np.log(rel) / log_tol, 0.0, 1.0))
                        progress(frac, f"Solving: residual {rel:.1e} (iteration {it})")
            if breakdown:
                break
            # Converged (recursive residual), out of iterations or switching to
            # Jacobi: continue from the true residual, which may have drifted.
            r = b - apply(x)
            rel = rel_true = vec.norm(r) / bnorm
            if not switch:
                if rel <= tol or it >= maxiter or restarts >= 3:
                    break
                restarts += 1

        if breakdown or rel_true is None or rel_true > tol:
            # Not converged: return the iterate with the smallest residual seen.
            if breakdown or best_rel < rel or not np.all(np.isfinite(x)):
                x = best_x
            rel_true = vec.norm(b - apply(x)) / bnorm
        info["iterations"] = it
        info["residual"] = rel_true
        info["converged"] = bool(rel_true <= tol)
        if breakdown:
            info["breakdown"] = True
        x *= freef
        u[:] = _aos(x)
        return u, info


def _axpy(y, a, x):  # y += a * x
    t = x * a
    y += t


def _cg_update(x, r, p, q, alpha):  # x += alpha p, r -= alpha q
    _axpy(x, alpha, p)
    _axpy(r, -alpha, q)


def _xpby(p, z, beta):  # p = z + beta * p
    p *= beta
    p += z


# ------------------------------------------------------------------------ analysis
@dataclass
class AnalysisResult:
    """Result of :func:`analyze` (arrays are float64, grid-shaped ones are zero outside)."""

    u: np.ndarray  # (n_dofs,) nodal displacements
    stress: np.ndarray  # (nx, ny, nz, 6) element-centre stress, zeros outside the domain
    von_mises: np.ndarray  # (nx, ny, nz)
    displacement: np.ndarray  # (nx, ny, nz) mean |u| of the 8 element nodes, zeros outside
    compliance: float  # forces . u
    max_displacement: float  # max nodal |u|
    info: dict  # solver info + "time": seconds of the whole analysis


def analyze(
    problem: Problem,
    solver: str = "MG",
    tol: float = 1e-6,
    maxiter: int = 4000,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
    backend: Optional[Backend] = None,
) -> AnalysisResult:
    """Linear static analysis of the solid domain (all active elements at full stiffness).

    ``backend`` as for :class:`FEModel` (default: the thread's backend, else numpy).
    """
    t_start = time.perf_counter()
    if progress is not None:
        progress(0.0, "Building the finite-element model")
    model = FEModel(problem, solver=solver, backend=backend)
    check_stop(should_stop)

    def solve_progress(frac, message):
        if progress is not None:
            progress(0.05 + 0.85 * frac, message)

    scale = np.ones(model.n_act, dtype=np.float64)
    u, info = model.solve(
        scale, tol=tol, maxiter=maxiter, should_stop=should_stop, progress=solve_progress
    )
    check_stop(should_stop)
    if progress is not None:
        progress(0.9, "Computing stresses")
    stress = model.to_grid(model.element_stress(u, scale))
    vm = von_mises(stress)
    unode = np.linalg.norm(u.reshape(-1, 3), axis=1)
    enodes = model.edof[:, 0::3] // 3
    disp = model.to_grid(unode[enodes].mean(axis=1) if model.n_act else np.zeros(0))
    info = dict(info)
    info["solve_time"] = info.get("time", 0.0)
    info["time"] = time.perf_counter() - t_start
    model.release_device()
    if progress is not None:
        progress(1.0, "Analysis done")
    return AnalysisResult(
        u=u,
        stress=stress,
        von_mises=vm,
        displacement=disp,
        compliance=float(problem.forces @ u),
        max_displacement=float(unode.max()) if unode.size else 0.0,
        info=info,
    )
