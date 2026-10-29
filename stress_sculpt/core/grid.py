"""Structured voxel grid and problem definition shared by every solver.

Conventions (used everywhere in the add-on):

* Elements (voxels) are indexed ``(i, j, k)`` along world ``(x, y, z)``.
  Flat element ids are C-order ravels of ``Grid.shape``.
* Nodes are the voxel corners, grid ``Grid.nodes_shape = shape + 1``,
  flat node ids are C-order ravels of ``nodes_shape``.
* Degrees of freedom: ``dof = 3 * node + component`` (component 0=x, 1=y, 2=z).
* Element ``(i, j, k)`` spans ``origin + [i, i+1] * h`` (same for y, z);
  its centre is ``origin + (i + 0.5) * h``.
* Hex8 local node order (offsets added to the element's (i, j, k)) is
  ``HEX_NODE_OFFSETS``; element dof order is node-major
  ``[n0x, n0y, n0z, n1x, ..., n7z]``.
* Stress / strain use Voigt order ``[xx, yy, zz, xy, yz, zx]`` (engineering
  shear strains in strain vectors).
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


HEX_NODE_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int64,
)

# Signature of progress callbacks: progress(fraction_0_to_1, message)
ProgressFn = Callable[[float, str], None]
# Signature of cancellation callbacks: should_stop() -> True to abort
StopFn = Callable[[], bool]


class Cancelled(Exception):
    """Raised by long computations when ``should_stop()`` returns True."""


def check_stop(should_stop: Optional[StopFn]) -> None:
    if should_stop is not None and should_stop():
        raise Cancelled()


@dataclass
class Grid:
    shape: tuple
    origin: np.ndarray
    h: float

    def __post_init__(self):
        self.shape = tuple(int(s) for s in self.shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError("Grid.shape must be three positive integers")
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.h = float(self.h)
        if not self.h > 0.0:
            raise ValueError("Grid.h must be positive")

    # ------------------------------------------------------------------ sizes
    @property
    def nodes_shape(self) -> tuple:
        return tuple(s + 1 for s in self.shape)

    @property
    def n_elems(self) -> int:
        return int(np.prod(self.shape))

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.nodes_shape))

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    @property
    def size(self) -> np.ndarray:
        """World-space extent of the whole grid."""
        return np.asarray(self.shape, dtype=np.float64) * self.h

    # ------------------------------------------------------------- geometry
    def element_centers(self) -> np.ndarray:
        """World coordinates of element centres, shape ``(nx, ny, nz, 3)``."""
        axes = [self.origin[a] + (np.arange(self.shape[a]) + 0.5) * self.h for a in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        return np.stack([X, Y, Z], axis=-1)

    def node_coords(self) -> np.ndarray:
        """World coordinates of all nodes, shape ``(n_nodes, 3)`` (C-order)."""
        axes = [self.origin[a] + np.arange(self.nodes_shape[a]) * self.h for a in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    def world_to_index(self, points) -> np.ndarray:
        """Continuous element-index coordinates (element centres are integers)."""
        p = np.asarray(points, dtype=np.float64)
        return (p - self.origin) / self.h - 0.5

    def world_to_node_index(self, points) -> np.ndarray:
        """Continuous node-index coordinates (nodes are integers)."""
        p = np.asarray(points, dtype=np.float64)
        return (p - self.origin) / self.h

    # --------------------------------------------------------- connectivity
    def element_nodes(self, elems=None) -> np.ndarray:
        """Node ids of elements, shape ``(n, 8)`` in ``HEX_NODE_OFFSETS`` order.

        ``elems`` are flat element ids (default: all elements).
        """
        if elems is None:
            elems = np.arange(self.n_elems, dtype=np.int64)
        elems = np.asarray(elems, dtype=np.int64).ravel()
        i, j, k = np.unravel_index(elems, self.shape)
        ijk = np.stack([i, j, k], axis=-1)[:, None, :] + HEX_NODE_OFFSETS[None, :, :]
        return np.ravel_multi_index(
            (ijk[..., 0], ijk[..., 1], ijk[..., 2]), self.nodes_shape
        ).astype(np.int64)

    def element_dofs(self, elems=None) -> np.ndarray:
        """Global dof ids of elements, shape ``(n, 24)`` (node-major)."""
        nodes = self.element_nodes(elems)
        return (3 * nodes[:, :, None] + np.arange(3, dtype=np.int64)[None, None, :]).reshape(
            nodes.shape[0], 24
        )

    def nodes_of_elements(self, elem_mask) -> np.ndarray:
        """Boolean mask ``(n_nodes,)`` of nodes touched by elements in ``elem_mask``."""
        elem_mask = np.asarray(elem_mask, dtype=bool).reshape(self.shape)
        out = np.zeros(self.nodes_shape, dtype=bool)
        nx, ny, nz = self.shape
        for di, dj, dk in HEX_NODE_OFFSETS:
            out[di : di + nx, dj : dj + ny, dk : dk + nz] |= elem_mask
        return out.ravel()

    # ---------------------------------------------------------- factories
    @classmethod
    def from_bounds(cls, bmin, bmax, resolution: int, pad: int = 0) -> "Grid":
        """Grid with ``resolution`` voxels along the longest side of a box.

        The grid is centred on the box; ``pad`` extra voxels are added on every
        side.
        """
        bmin = np.asarray(bmin, dtype=np.float64).reshape(3)
        bmax = np.asarray(bmax, dtype=np.float64).reshape(3)
        extent = np.maximum(bmax - bmin, 0.0)
        longest = float(extent.max())
        if longest <= 0.0:
            raise ValueError("Bounding box is degenerate")
        resolution = max(1, int(resolution))
        h = longest / resolution
        counts = np.maximum(1, np.ceil(extent / h - 1e-9).astype(np.int64)) + 2 * int(pad)
        center = 0.5 * (bmin + bmax)
        origin = center - 0.5 * counts * h
        return cls(tuple(int(c) for c in counts), origin, h)


@dataclass
class Problem:
    """A linear-elastic voxel problem.

    Arrays shaped like the grid are ``(nx, ny, nz)``. Optional arrays may be
    ``None`` (meaning "nothing").
    """

    grid: Grid
    domain: np.ndarray  # bool: element exists (inside the design domain)
    fixed_dofs: np.ndarray  # int64 global dof ids with zero displacement
    forces: np.ndarray  # float64 (n_dofs,) nodal forces [N]
    passive_solid: Optional[np.ndarray] = None  # bool: forced density 1
    passive_void: Optional[np.ndarray] = None  # bool: forced density 0
    bias: Optional[np.ndarray] = None  # float in [-1, 1]: + thicken, - hollow
    branch: Optional[np.ndarray] = None  # float in [0, 1]: branching zones
    E: float = 1.0  # Young's modulus [Pa]
    nu: float = 0.3  # Poisson ratio

    def __post_init__(self):
        g = self.grid
        self.domain = np.asarray(self.domain, dtype=bool).reshape(g.shape)
        self.fixed_dofs = np.unique(np.asarray(self.fixed_dofs, dtype=np.int64).ravel())
        self.forces = np.asarray(self.forces, dtype=np.float64).ravel()
        if self.forces.shape[0] != g.n_dofs:
            raise ValueError("forces must have n_dofs entries")
        for name in ("passive_solid", "passive_void"):
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr, dtype=bool).reshape(g.shape))
        for name in ("bias", "branch"):
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, np.asarray(arr, dtype=np.float64).reshape(g.shape))
        self.E = float(self.E)
        self.nu = float(self.nu)
        if not (-1.0 < self.nu < 0.5):
            raise ValueError("Poisson ratio must be in (-1, 0.5)")

    def passive_solid_or_empty(self) -> np.ndarray:
        if self.passive_solid is None:
            return np.zeros(self.grid.shape, dtype=bool)
        return self.passive_solid & self.domain

    def passive_void_or_empty(self) -> np.ndarray:
        if self.passive_void is None:
            return np.zeros(self.grid.shape, dtype=bool)
        return self.passive_void & self.domain & ~self.passive_solid_or_empty()

    def bias_or_zero(self) -> np.ndarray:
        if self.bias is None:
            return np.zeros(self.grid.shape, dtype=np.float64)
        return np.clip(self.bias, -1.0, 1.0) * self.domain

    def branch_or_zero(self) -> np.ndarray:
        if self.branch is None:
            return np.zeros(self.grid.shape, dtype=np.float64)
        return np.clip(self.branch, 0.0, 1.0) * self.domain
