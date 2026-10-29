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

Everything here is pure numpy, float64, and keeps no global mutable state:
independent :class:`FEModel` instances may be used from different threads (a single
instance must not be shared by concurrent ``solve`` calls).
"""

from dataclasses import dataclass, replace
import time
from typing import Optional

import numpy as np

from .grid import HEX_NODE_OFFSETS, Grid, Problem, ProgressFn, StopFn, check_stop


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
# Elements per block in element loops: bounds the (block, 24) temporaries to ~13 MB
# whatever the model size (and is slightly faster than one huge block).
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
    """Assembled ``sum_e scale_e * KE`` times ``x`` (no boundary masking).

    ``ue = x[edof]`` -> ``fe = (ue @ KE) * scale`` -> ``bincount`` (KE is symmetric);
    large models are processed in blocks of sorted elements.
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
    """Diagonal of the assembled ``sum_e scale_e * KE`` (n_dofs,)."""
    y = np.zeros(n_dofs, dtype=np.float64)
    for s, t in _blocks(edof.shape[0]):
        _scatter_block(y, edof[s:t], scale[s:t, None] * ke_diag[None, :])
    return y


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
    fixed: np.ndarray  # (n_dofs,) bool, Dirichlet dofs (inherited on coarse levels)
    free: np.ndarray  # (n_dofs,) bool, touched by an active element and not fixed
    lambda_bound: float  # upper bound of lambda_max(D^-1 K) for any element scales
    factors: tuple = (1, 1, 1)  # coarsening factors towards the next level

    @property
    def nodes_shape(self) -> tuple:
        return tuple(s + 1 for s in self.shape)

    @property
    def n_dofs(self) -> int:
        return 3 * int(np.prod(self.nodes_shape))

    @property
    def n_elems(self) -> int:
        return int(np.prod(self.shape))


@dataclass
class _LevelOp:
    """Scale-dependent part of a level for one solve."""

    scale: np.ndarray  # (n_act,)
    free: np.ndarray  # (n_dofs,) bool: geometric free dofs with a positive diagonal
    freef: np.ndarray  # same as float (multiplicative mask)
    dinv: np.ndarray  # omega / diagonal on free dofs, 0 elsewhere
    omega: float
    coarse_inv: Optional[np.ndarray] = None  # coarsest level: inverse Cholesky factor
    coarse_idx: Optional[np.ndarray] = None  # coarsest level: free dof ids


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
    out = np.empty((n_fine,) + src.shape[1:], dtype=np.float64)
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
    out = np.zeros((n_coarse,) + src.shape[1:], dtype=np.float64)
    out[:n_even] += src[0::2]
    half = 0.5 * src[1::2]
    out[:n_odd] += half
    out[1 : n_odd + 1] += half
    return np.moveaxis(out, 0, axis)


def _prolong(xc: np.ndarray, fine: _Level, coarse: _Level) -> np.ndarray:
    a = xc.reshape(coarse.nodes_shape + (3,))
    for axis in range(3):
        if fine.factors[axis] == 2:
            a = _prolong_axis(a, axis, fine.nodes_shape[axis])
    return a.reshape(-1)


def _restrict(rf: np.ndarray, fine: _Level, coarse: _Level) -> np.ndarray:
    a = rf.reshape(fine.nodes_shape + (3,))
    for axis in range(3):
        if fine.factors[axis] == 2:
            a = _restrict_axis(a, axis, coarse.nodes_shape[axis])
    return a.reshape(-1)


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


def _lanczos_lambda_max(apply, free, d_isqrt, steps: int, should_stop=None) -> float:
    """Lanczos estimate (from below) of ``lambda_max(D^-1 A)`` on the free dofs."""
    n = free.size
    rng = np.random.default_rng(1234567)  # local generator: deterministic, thread-safe
    v = rng.standard_normal(n) * free
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        return 0.0
    v /= nv
    v_prev = np.zeros(n)
    beta = 0.0
    alphas, betas = [], []
    for _ in range(max(1, steps)):
        check_stop(should_stop)
        w = d_isqrt * apply(d_isqrt * v)
        alpha = float(w @ v)
        w -= alpha * v + beta * v_prev
        alphas.append(alpha)
        beta = float(np.linalg.norm(w))
        if not np.isfinite(beta) or beta <= 1e-12 * max(abs(alpha), 1e-300):
            break
        betas.append(beta)
        v_prev, v = v, w / beta
    k = len(alphas)
    T = np.diag(alphas) + np.diag(betas[: k - 1], 1) + np.diag(betas[: k - 1], -1)
    return float(np.linalg.eigvalsh(T)[-1])


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

    The multigrid geometry is built lazily on the first MG solve and reused; the
    scale dependent parts are rebuilt when the scales change.
    """

    def __init__(
        self,
        problem: Problem,
        solver: str = "MG",
        emin: float = 1e-3,
        *,
        mg_options: Optional[MGOptions] = None,
    ):
        solver = str(solver).upper()
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver {solver!r} (expected one of {SOLVERS})")
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
        self._ke_diag = np.diag(self.KE).copy()
        fixed = np.zeros(self.n_dofs, dtype=bool)
        fixed[_valid_fixed_dofs(problem)] = True
        self.fixed = fixed
        touched = np.zeros(self.n_dofs, dtype=bool)
        touched[self._edof_flat] = True
        self.free = touched & ~fixed
        self._freef = self.free.astype(np.float64)
        # stress = scale * E * D @ (B_center / h) @ u_e
        self._stress_op = (problem.E / g.h) * (elasticity_matrix(problem.nu) @ hex8_b_center())
        self._levels = None  # multigrid geometry (lazy)
        self._mg_error = None  # reason why multigrid is unavailable
        self._ops_scale = None  # scale the cached level operators were built for
        self._ops = None

    @property
    def n_act(self) -> int:
        return int(self.elems.size)

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

    def apply_K(self, u, scale) -> np.ndarray:
        """Matrix-free product ``K(scale) @ u`` (n_dofs,), zero on non-free dofs."""
        u = self._check_vector(u)
        scale = self._check_scale(scale)
        y = _apply_elements(u, self.edof, self._edof_flat, self.KE, scale, self.n_dofs)
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
        out = np.empty(self.n_act, dtype=np.float64)
        for s, t in _blocks(self.n_act):
            ue = u[self.edof[s:t]]
            out[s:t] = np.einsum("ij,ij->i", ue @ self.KE, ue)
        return out

    def element_stress(self, u, scale) -> np.ndarray:
        """Physical stress (n_act, 6) at the element centroids, Voigt order.

        ``scale_e * E * D(nu) @ (B_center / h) @ u_e``.
        """
        u = self._check_vector(u)
        scale = self._check_scale(scale)
        out = np.empty((self.n_act, 6), dtype=np.float64)
        for s, t in _blocks(self.n_act):
            out[s:t] = u[self.edof[s:t]] @ self._stress_op.T
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
        self._levels = levels
        return levels

    def _level_apply(self, level: _Level, op: _LevelOp, x: np.ndarray) -> np.ndarray:
        y = _apply_elements(x, level.edof, level.edof_flat, level.ke, op.scale, level.n_dofs)
        y *= op.freef
        return y

    def _mg_operators(self, scale, free_eff, diag, should_stop=None) -> list:
        """Scale-dependent level operators (cached for identical scales)."""
        if self._ops is not None and np.array_equal(self._ops_scale, scale):
            return self._ops
        levels = self._mg_geometry()
        opts = self.mg_options
        ops = []
        scale_l = scale
        for li, level in enumerate(levels):
            check_stop(should_stop)
            if li == 0:
                diag_l = diag
                free_l = free_eff
            else:
                prev = levels[li - 1]
                full = np.zeros(prev.n_elems, dtype=np.float64)
                full[prev.elems] = scale_l
                summed = _coarsen_sum(full.reshape(prev.shape), prev.factors, level.shape)
                scale_l = summed.ravel()[level.elems] / float(np.prod(prev.factors))
                diag_l = _assembled_diagonal(level.edof, np.diag(level.ke), scale_l, level.n_dofs)
                free_l = level.free & (diag_l > 0.0)
            freef = free_l.astype(np.float64)
            safe_diag = np.where(free_l, diag_l, 1.0)
            d_inv = freef / safe_diag
            op = _LevelOp(scale=scale_l, free=free_l, freef=freef, dinv=d_inv, omega=0.0)
            if li == len(levels) - 1:
                self._factor_coarsest(level, op)
            else:
                if opts.omega is not None:
                    omega = float(opts.omega)
                else:
                    lam = _lanczos_lambda_max(
                        lambda v, lv=level, o=op: self._level_apply(lv, o, v),
                        free_l,
                        np.sqrt(d_inv),
                        opts.lanczos_steps,
                        should_stop,
                    )
                    lam_used = level.lambda_bound
                    if np.isfinite(lam) and lam > 0.0:
                        # lambda_max(D^-1 K) >= 1 always (unit Rayleigh quotients).
                        lam_used = min(lam_used, opts.lambda_margin * max(lam, 1.0))
                    omega = opts.omega_scale / lam_used
                op.omega = omega
                op.dinv = omega * d_inv
            ops.append(op)
        self._ops = ops
        self._ops_scale = scale.copy()
        return ops

    def _factor_coarsest(self, level: _Level, op: _LevelOp) -> None:
        idx = np.flatnonzero(op.free)
        op.coarse_idx = idx
        nf = int(idx.size)
        if nf == 0:
            op.coarse_inv = np.zeros((0, 0))
            return
        g2f = np.full(level.n_dofs, -1, dtype=np.int64)
        g2f[idx] = np.arange(nf)
        loc = g2f[level.edof]  # (n_act, 24) free index or -1
        rows = np.broadcast_to(loc[:, :, None], (loc.shape[0], 24, 24))
        cols = np.broadcast_to(loc[:, None, :], (loc.shape[0], 24, 24))
        ok = (rows >= 0) & (cols >= 0)
        vals = op.scale[:, None, None] * level.ke[None, :, :]
        K = np.bincount(
            (rows[ok] * nf + cols[ok]), weights=vals[ok], minlength=nf * nf
        ).reshape(nf, nf)
        K = 0.5 * (K + K.T)
        mean_diag = float(np.mean(np.diag(K)))
        if not (mean_diag > 0.0 and np.isfinite(mean_diag)):
            raise _MGSetupError("coarsest multigrid matrix is degenerate")
        eye = np.eye(nf)
        for reg in (1e-12, 1e-10, 1e-8, 1e-6):
            try:
                L = np.linalg.cholesky(K + (reg * mean_diag) * eye)
            except np.linalg.LinAlgError:
                continue
            linv = np.linalg.inv(L)
            if np.all(np.isfinite(linv)):
                op.coarse_inv = np.tril(linv)
                return
        raise _MGSetupError("coarsest multigrid matrix is not positive definite")

    def _mg_preconditioner(self, scale):
        """The V-cycle ``r -> M r`` for the given scales (testing / diagnostics)."""
        scale = self._check_scale(scale)
        diag = self.diagonal(scale)
        free = self.free & (diag > 0.0)
        levels = self._mg_geometry()
        ops = self._mg_operators(scale, free, diag)
        freef = free.astype(np.float64)
        return lambda r: self._vcycle(levels, ops, 0, np.asarray(r, dtype=np.float64) * freef)

    def _vcycle(self, levels, ops, li: int, r: np.ndarray, should_stop=None) -> np.ndarray:
        """Symmetric V-cycle on level ``li`` for the residual ``r`` (zero initial guess)."""
        level, op = levels[li], ops[li]
        if li == len(levels) - 1:
            x = np.zeros(level.n_dofs)
            if op.coarse_idx.size:
                G = op.coarse_inv
                x[op.coarse_idx] = G.T @ (G @ r[op.coarse_idx])
            return x
        sweeps = max(1, int(self.mg_options.sweeps))
        x = op.dinv * r  # first sweep from a zero initial guess
        for _ in range(sweeps - 1):
            x += op.dinv * (r - self._level_apply(level, op, x))
        if li == 0:
            # Poll between the fine-level phases: one fine V-cycle (4 fine products)
            # takes ~0.5 s on models with ~250k elements.
            check_stop(should_stop)
        res = r - self._level_apply(level, op, x)
        coarse, cop = levels[li + 1], ops[li + 1]
        rc = _restrict(res, level, coarse)
        rc *= cop.freef
        xc = self._vcycle(levels, ops, li + 1, rc)
        corr = _prolong(xc, level, coarse)
        corr *= op.freef
        x += corr
        if li == 0:
            check_stop(should_stop)
        for _ in range(sweeps):
            x += op.dinv * (r - self._level_apply(level, op, x))
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
        preconditioner that produced the result), "time" (s)}`` plus ``"fallback"``
        (reason) when multigrid was abandoned for Jacobi and ``"breakdown"`` when
        the iteration stopped on a non-positive curvature / non-finite value.

        ``x0`` (n_dofs,) is a warm start (ignored if it is not finite or worse than
        zero). Zero load or no free dof return zeros at once; dofs whose stiffness is
        zero (all touching elements have scale 0) are held at zero. Never raises on
        non-convergence (``converged`` is False; the iterate with the smallest
        residual is returned, always finite). Raises ``ValueError`` for invalid
        input (wrong sizes, negative / non-finite scales, non-finite forces) and
        ``Cancelled`` when ``should_stop()`` returns True (polled every iteration);
        ``progress(fraction, message)`` is called about every 0.2 s.
        """
        t_start = time.perf_counter()
        scale = self._check_scale(scale)
        tol = float(tol)
        maxiter = max(0, int(maxiter))
        info = {"iterations": 0, "residual": 0.0, "converged": True, "solver": self.solver}
        u = np.zeros(self.n_dofs, dtype=np.float64)

        forces = self.problem.forces
        if not np.all(np.isfinite(forces)):
            raise ValueError("forces contain non-finite values")
        diag = self.diagonal(scale)
        free = self.free & (diag > 0.0)
        b = np.where(free, forces, 0.0)
        bnorm = float(np.linalg.norm(b))
        if bnorm == 0.0 or not free.any():
            info["time"] = time.perf_counter() - t_start
            return u, info
        freef = free.astype(np.float64)
        jacobi_dinv = freef / np.where(free, diag, 1.0)

        def apply(v):
            y = _apply_elements(v, self.edof, self._edof_flat, self.KE, scale, self.n_dofs)
            y *= freef
            return y

        def jacobi(v):
            return jacobi_dinv * v

        precond = jacobi
        info["solver"] = "JACOBI"
        if self.solver == "MG":
            if progress is not None:
                progress(0.0, "Preparing the multigrid solver")
            try:
                levels = self._mg_geometry()
                ops = self._mg_operators(scale, free, diag, should_stop)
            except _MGSetupError as exc:
                info["fallback"] = str(exc)
            else:
                precond = lambda v: self._vcycle(levels, ops, 0, v, should_stop)  # noqa: E731
                info["solver"] = "MG"

        # ------------------------------------------------------------ initial guess
        x = np.zeros(self.n_dofs, dtype=np.float64)
        r = b.copy()
        if x0 is not None:
            x0 = self._check_vector(x0, "x0")
            if np.all(np.isfinite(x0)):
                xw = x0 * freef
                rw = b - apply(xw)
                if float(np.linalg.norm(rw)) < bnorm:
                    x, r = xw, rw
        rel = float(np.linalg.norm(r)) / bnorm
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
            rz = float(r @ z)
            if not (np.isfinite(rz) and rz > 0.0):
                if info["solver"] != "MG":
                    breakdown = True
                    break
                precond = jacobi
                info["solver"] = "JACOBI"
                info["fallback"] = "multigrid preconditioner was not positive definite"
                continue
            p = z.copy()
            switch = False
            rel_true = None
            while True:
                check_stop(should_stop)
                q = apply(p)
                pq = float(p @ q)
                if not (np.isfinite(pq) and pq > 0.0):
                    breakdown = True  # K is singular along p (mechanism) or not finite
                    break
                alpha = rz / pq
                x += alpha * p
                r -= alpha * q
                it += 1
                rel = float(np.linalg.norm(r)) / bnorm
                if not np.isfinite(rel):
                    breakdown = True
                    break
                if rel < best_rel:
                    best_rel = rel
                    best_x[:] = x
                if rel <= tol or it >= maxiter:
                    break
                z = precond(r)
                rz_new = float(r @ z)
                if not (np.isfinite(rz_new) and rz_new > 0.0):
                    if info["solver"] == "MG":
                        precond = jacobi
                        info["solver"] = "JACOBI"
                        info["fallback"] = "multigrid preconditioner was not positive definite"
                        switch = True
                    else:
                        breakdown = True
                    break
                p *= rz_new / rz
                p += z
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
            rel = rel_true = float(np.linalg.norm(r)) / bnorm
            if not switch:
                if rel <= tol or it >= maxiter or restarts >= 3:
                    break
                restarts += 1

        if breakdown or rel_true is None or rel_true > tol:
            # Not converged: return the iterate with the smallest residual seen.
            if breakdown or best_rel < rel or not np.all(np.isfinite(x)):
                x = best_x
            rel_true = float(np.linalg.norm(b - apply(x))) / bnorm
        info["iterations"] = it
        info["residual"] = rel_true
        info["converged"] = bool(rel_true <= tol)
        if breakdown:
            info["breakdown"] = True
        info["time"] = time.perf_counter() - t_start
        u[:] = x * freef
        return u, info


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
) -> AnalysisResult:
    """Linear static analysis of the solid domain (all active elements at full stiffness)."""
    t_start = time.perf_counter()
    if progress is not None:
        progress(0.0, "Building the finite-element model")
    model = FEModel(problem, solver=solver)
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
