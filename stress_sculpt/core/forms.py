"""Form-found branching structures ("tree columns") that carry point loads to supports.

The tree is a funicular network: every member works in pure axial compression
under the given loads, like Frei Otto's branching columns or an inverted Gaudí
hanging model. Construction:

1. **Frame** — the tree grows against the resultant load (``up = -sum(F)``), so
   gravity loads give upright trees; sideways loads give sideways trees.
2. **Topology** — leaves are the load points; they are grouped per support
   (nearest support, or automatic footings found by weighted k-means), then split
   recursively into ``branching`` sub-groups (k-means in the horizontal plane).
3. **Heights** — designer controlled: the trunk splits at ``split_height`` of the
   way up and every deeper level shares the remaining height (``level_curve``
   shapes the spacing).
4. **Equilibrium** — with prescribed heights the vertical force carried by member
   ``e`` is the weight ``V_e`` of all leaves above it, so its force density is
   ``q_e = V_e / dz_e``. Horizontal positions of the inner nodes follow from the
   force-density equations ``sum_j q_ij (x_j - x_i) = 0`` (Schek's FDM, linear).
   Automatic footings are placed under the trunk so the trunk stays vertical and
   the carried object receives no net horizontal thrust.
5. **Sizing** — axial force ``F_e = q_e * L_e``; radius ``∝ sqrt(F_e)`` (constant
   stress design), clamped to ``[radius_min, radius_max]``.
6. **Shape** — members become cubic Hermite curves leaving each node tangent to
   the force flow (which is exactly continuous through a funicular node), arriving
   at the loads along ``up``; footings flare, load points get capitals.

Pure numpy, deterministic, thread-safe.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .grid import StopFn, check_stop

_EPS = 1e-12


@dataclass
class TreeSettings:
    n_trunks: int = 0  # automatic footings: 0 = choose from the footprint
    branching: int = 2  # children per split (2..4)
    split_height: float = 0.35  # where the trunk first splits (fraction of its height)
    level_curve: float = 1.0  # >1: splits crowd towards the loads, <1: towards the ground
    organic: float = 0.6  # 0 = straight members, 1 = strongly flowing curves
    radius_min: float = 0.02  # world units
    radius_max: float = 0.12  # world units (the most loaded member)
    footing_flare: float = 1.5  # base radius multiplier
    capital_flare: float = 1.6  # radius multiplier at the load points
    samples: int = 10  # points per member curve
    max_leaves: int = 96  # more load points are clustered


@dataclass
class Tree:
    nodes: np.ndarray  # (N, 3) world positions
    edges: np.ndarray  # (E, 2) parent, child node ids
    forces: np.ndarray  # (E,) axial compression [N]
    radii: np.ndarray  # (E,) member radius [world units]
    kind: np.ndarray  # (N,) 0 = base, 1 = inner node, 2 = leaf (load point)
    depth: np.ndarray  # (N,) edges from the base
    up: np.ndarray  # (3,) growth direction
    leaf_weights: np.ndarray  # (N,) load carried at leaves (0 elsewhere)
    curves: List[np.ndarray] = field(default_factory=list)  # per edge (samples, 3)
    curve_radii: List[np.ndarray] = field(default_factory=list)  # per edge (samples,)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_load(self) -> float:
        return float(self.leaf_weights.sum())

    def base_reactions(self) -> np.ndarray:
        """Force each base receives from its trunk(s), ``(n_bases, 3)`` (compression)."""
        bases = np.flatnonzero(self.kind == 0)
        out = np.zeros((bases.size, 3))
        for b_i, b in enumerate(bases):
            for e in np.flatnonzero(self.edges[:, 0] == b):
                c = self.edges[e, 1]
                d = self.nodes[b] - self.nodes[c]
                out[b_i] += self.forces[e] * d / max(np.linalg.norm(d), _EPS)
        return out

    def equilibrium_residual(self) -> float:
        """Max norm of the unbalanced force at inner nodes, relative to the total load."""
        res = np.zeros_like(self.nodes)
        for e, (p, c) in enumerate(self.edges):
            d = self.nodes[c] - self.nodes[p]
            f = self.forces[e] * d / max(np.linalg.norm(d), _EPS)
            # compression member pushes its two end nodes apart
            res[c] += f
            res[p] -= f
        inner = self.kind == 1
        if not inner.any():
            return 0.0
        return float(np.linalg.norm(res[inner], axis=1).max() / max(self.total_load, _EPS))

    def tubes(self):
        """Members as ``sculpt.Tube`` objects (value = axial force)."""
        from .sculpt import Tube

        out = []
        for e, (curve, radius) in enumerate(zip(self.curves, self.curve_radii)):
            out.append(Tube(curve, radius, np.full(curve.shape[0], self.forces[e])))
        return out


# --------------------------------------------------------------------------- helpers
def _frame(up) -> np.ndarray:
    """Rows e1, e2, up of an orthonormal frame."""
    up = np.asarray(up, dtype=np.float64)
    up = up / max(np.linalg.norm(up), _EPS)
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(helper, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return np.stack([e1, e2, up])


def _kmeans_once(points, weights, k, first, iterations):
    centers = [points[first]]
    dist = np.linalg.norm(points - centers[0], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dist))
        centers.append(points[nxt])
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    centers = np.array(centers)
    labels = np.full(points.shape[0], -1, dtype=np.int64)
    for _ in range(iterations):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = np.argmin(d2, axis=1)
        # keep every cluster non-empty: steal the farthest point of the biggest cluster
        for c in range(k):
            if not np.any(new == c):
                big = np.bincount(new, minlength=k).argmax()
                members = np.flatnonzero(new == big)
                new[members[np.argmax(d2[members, big])]] = c
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            centers[c] = (points[m] * weights[m, None]).sum(0) / weights[m].sum()
    inertia = float((weights * ((points - centers[labels]) ** 2).sum(-1)).sum())
    return labels, inertia


def weighted_kmeans(points, weights, k: int, iterations: int = 30, restarts: int = 8) -> np.ndarray:
    """Deterministic weighted k-means. Returns labels ``(n,)``.

    Farthest-point seeding started from the ``restarts`` points farthest from the
    weighted centroid; the clustering with the lowest weighted inertia wins.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), _EPS)
    centroid = (points * weights[:, None]).sum(0) / weights.sum()
    order = np.argsort(-np.linalg.norm(points - centroid, axis=1), kind="stable")
    best, best_inertia = None, np.inf
    for first in order[: max(1, int(restarts))]:
        labels, inertia = _kmeans_once(points, weights, k, int(first), iterations)
        if best is None or inertia < best_inertia - 1e-12 * max(best_inertia, 1.0):
            best, best_inertia = labels, inertia
    # relabel in order of first appearance (stable output)
    _, first_seen = np.unique(best, return_index=True)
    remap = np.empty(k, dtype=np.int64)
    remap[best[np.sort(first_seen)]] = np.arange(k)
    return remap[best]


def cluster_loads(points, forces, max_points: int):
    """Merge load points into at most ``max_points`` representatives (force-weighted)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    forces = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] <= max_points:
        return points, forces
    weights = np.linalg.norm(forces, axis=1)
    labels = weighted_kmeans(points, weights, max_points)
    out_p, out_f = [], []
    for c in range(labels.max() + 1):
        m = labels == c
        w = np.maximum(weights[m], _EPS)
        out_p.append((points[m] * w[:, None]).sum(0) / w.sum())
        out_f.append(forces[m].sum(0))
    return np.array(out_p), np.array(out_f)


def _hermite(p0, p1, t0, t1, n) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def _smoothstep(x) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _unit(v) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), _EPS)


# --------------------------------------------------------------------------- footings
def _default_ground(local, ground) -> float:
    """Ground height along ``up``: given, or below the lowest load by its height spread."""
    if ground is not None:
        return float(ground)
    z_low = float(local[:, 2].min())
    return z_low - max(float(local[:, 2].max()) - z_low, 1.0)


def _auto_bases_local(local, weights, ground: float, n: int) -> np.ndarray:
    """Footings in the local frame: weighted k-means of the loads in plan, on the ground."""
    height = max(float(local[:, 2].min()) - ground, _EPS)
    k = int(n)
    if k <= 0:
        c = (local[:, :2] * weights[:, None]).sum(0) / weights.sum()
        r_foot = float(np.linalg.norm(local[:, :2] - c, axis=1).max())
        k = int(np.clip(round((r_foot / (0.9 * height)) ** 2), 1, 8))
    labels = weighted_kmeans(local[:, :2], weights, k)
    bases = []
    for c in range(labels.max() + 1):
        m = labels == c
        xy = (local[m, :2] * weights[m, None]).sum(0) / weights[m].sum()
        bases.append([xy[0], xy[1], ground])
    return np.array(bases)


def _load_frame(load_points, load_forces):
    pts = np.asarray(load_points, dtype=np.float64).reshape(-1, 3)
    frc = np.asarray(load_forces, dtype=np.float64).reshape(-1, 3)
    total = frc.sum(0)
    up = -total if np.linalg.norm(total) > _EPS else np.array([0.0, 0.0, 1.0])
    R = _frame(up)
    weights = -(frc @ R[2])
    weights = np.maximum(weights, 1e-3 * max(float(weights.max()), 1.0))
    return pts, frc, R, weights


def auto_footings(load_points, load_forces, ground=None, n: int = 0) -> np.ndarray:
    """World positions of automatic footings for these loads (same rule as ``grow_tree``).

    ``ground`` is the ground height along the growth direction (``-sum(forces)``; world Z for
    gravity loads); ``n`` = 0 picks the count from the footprint / height ratio (1..8).
    """
    pts, _frc, R, weights = _load_frame(load_points, load_forces)
    if pts.shape[0] == 0:
        return np.zeros((0, 3))
    local = pts @ R.T
    return _auto_bases_local(local, weights, _default_ground(local, ground), n) @ R


# --------------------------------------------------------------------------- main
def grow_tree(
    load_points,
    load_forces,
    settings: Optional[TreeSettings] = None,
    supports=None,
    ground: Optional[float] = None,
    constrain: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    should_stop: Optional[StopFn] = None,
) -> Tree:
    """Branching funicular structure carrying ``load_forces`` at ``load_points``.

    ``supports``: optional ``(S, 3)`` fixed footing positions. Without supports,
    footings are placed automatically on the plane ``up . x = ground`` (default:
    below the lowest load by its height spread, at least 1 unit).
    ``constrain``: optional callable mapping inner node positions ``(M, 3)`` to
    allowed positions (e.g. nearest point inside a design domain); constrained
    nodes are then held fixed and the equilibrium is solved again.
    """
    st = settings or TreeSettings()
    pts = np.asarray(load_points, dtype=np.float64).reshape(-1, 3)
    frc = np.asarray(load_forces, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("No load points")
    if frc.shape != pts.shape:
        raise ValueError("load_forces must match load_points")
    total = frc.sum(0)
    warnings = []
    up = -total if np.linalg.norm(total) > _EPS else np.array([0.0, 0.0, 1.0])
    R = _frame(up)
    up = R[2]

    pts, frc = cluster_loads(pts, frc, max(1, int(st.max_leaves)))
    if pts.shape[0] < len(load_points):
        warnings.append(f"{len(load_points)} load points merged into {pts.shape[0]} branch tips")
    local = pts @ R.T  # (K, 3): e1, e2, height
    weights = -(frc @ up)  # component along gravity
    if np.any(weights <= 0):
        warnings.append("Some loads do not push along the main load direction: carried as light loads")
    weights = np.maximum(weights, 1e-3 * max(float(weights.max()), 1.0))

    # ----------------------------------------------------------- footings
    if supports is not None and len(supports):
        bases = np.asarray(supports, dtype=np.float64).reshape(-1, 3) @ R.T
        fixed_bases = True
    else:
        fixed_bases = False
        bases = _auto_bases_local(local, weights, _default_ground(local, ground), st.n_trunks)

    # leaves -> footing (horizontal distance)
    d2 = ((local[:, None, :2] - bases[None, :, :2]) ** 2).sum(-1)
    owner = np.argmin(d2, axis=1)

    nodes, kind, depth, parent = [], [], [], []
    leaf_w = []
    children = {}

    def add(pos, k, dep, par, w=0.0):
        nodes.append(np.array(pos, dtype=np.float64))
        kind.append(k)
        depth.append(dep)
        parent.append(par)
        leaf_w.append(w)
        nid = len(nodes) - 1
        children[nid] = []
        if par >= 0:
            children[par].append(nid)
        return nid

    b_branch = int(np.clip(st.branching, 2, 6))

    def build(idx, par, dep):
        """Subtree over leaf indices ``idx`` hanging from node ``par``."""
        check_stop(should_stop)
        if idx.size == 1:
            i = int(idx[0])
            return add(local[i], 2, dep, par, weights[i])
        w = weights[idx]
        centroid = (local[idx] * w[:, None]).sum(0) / w.sum()
        nid = add(centroid, 1, dep, par)  # height fixed below
        labels = weighted_kmeans(local[idx, :2], w, min(b_branch, idx.size))
        for c in range(labels.max() + 1):
            build(idx[labels == c], nid, dep + 1)
        return nid

    base_ids = []
    for b in range(bases.shape[0]):
        idx = np.flatnonzero(owner == b)
        if idx.size == 0:
            continue
        bid = add(bases[b], 0, 0, -1)
        base_ids.append(bid)
        if np.any(local[idx, 2] <= bases[b, 2] + 1e-9):
            raise ValueError("Every load point must lie above its support (along the load direction)")
        build(idx, bid, 1)
    if not base_ids:
        raise ValueError("No load reaches a support")

    nodes = np.array(nodes)
    kind = np.array(kind, dtype=np.int64)
    depth = np.array(depth, dtype=np.int64)
    parent = np.array(parent, dtype=np.int64)
    leaf_w = np.array(leaf_w, dtype=np.float64)
    n = nodes.shape[0]

    # subtree data: carried weight, lowest leaf height, height of subtree (edges)
    order = np.argsort(-depth, kind="stable")  # deepest first
    carried = leaf_w.copy()
    z_hi = np.where(kind == 2, nodes[:, 2], np.inf)
    sub_h = np.zeros(n, dtype=np.int64)
    for i in order:
        p = parent[i]
        if p >= 0:
            carried[p] += carried[i]
            z_hi[p] = min(z_hi[p], z_hi[i])
            sub_h[p] = max(sub_h[p], sub_h[i] + 1)

    # ----------------------------------------------------------- heights
    gamma = max(float(st.level_curve), 0.05)
    split = float(np.clip(st.split_height, 0.02, 0.95))
    for i in np.argsort(depth, kind="stable"):  # parents first
        if kind[i] != 1:
            continue
        p = parent[i]
        z_p = nodes[p, 2]
        span = z_hi[i] - z_p
        if kind[p] == 0:  # trunk top
            t = split
        else:
            t = (1.0 / (sub_h[i] + 1.0)) ** (1.0 / gamma)
        nodes[i, 2] = z_p + max(t * span, 1e-6 * max(span, 1.0))

    # ----------------------------------------------------------- equilibrium (x, y)
    edges = np.array([[parent[i], i] for i in range(n) if parent[i] >= 0], dtype=np.int64)
    dz = nodes[edges[:, 1], 2] - nodes[edges[:, 0], 2]
    if np.any(dz <= 0):
        raise ValueError("Degenerate tree heights")
    q = carried[edges[:, 1]] / dz

    vertical_trunk = np.zeros(n, dtype=bool)  # trunk tops whose base follows them
    if not fixed_bases:
        for b in base_ids:
            for c in children[b]:
                if kind[c] == 1:
                    vertical_trunk[c] = True

    held = kind != 1  # bases and leaves are fixed

    def solve_xy(held_mask):
        free = np.flatnonzero(~held_mask)
        if free.size == 0:
            return
        index = -np.ones(n, dtype=np.int64)
        index[free] = np.arange(free.size)
        A = np.zeros((free.size, free.size))
        rhs = np.zeros((free.size, 2))
        for e, (a, b) in enumerate(edges):
            qe = q[e]
            # vertical trunks transmit no horizontal force: their base follows the top
            if vertical_trunk[b] and kind[a] == 0:
                continue
            for u, v in ((a, b), (b, a)):
                iu = index[u]
                if iu < 0:
                    continue
                A[iu, iu] += qe
                iv = index[v]
                if iv >= 0:
                    A[iu, iv] -= qe
                else:
                    rhs[iu] += qe * nodes[v, :2]
        sol = np.linalg.solve(A + np.eye(free.size) * 1e-12 * max(np.abs(A).max(), 1.0), rhs)
        nodes[free, :2] = sol

    solve_xy(held)
    if not fixed_bases:
        for b in base_ids:
            tops = [c for c in children[b] if kind[c] == 1]
            if tops:
                nodes[b, :2] = nodes[tops[0], :2]

    if constrain is not None:
        inner = np.flatnonzero(kind == 1)
        if inner.size:
            world = nodes[inner] @ R
            moved = np.asarray(constrain(world), dtype=np.float64).reshape(-1, 3)
            shift = np.linalg.norm(moved - world, axis=1)
            if np.any(shift > 1e-9):
                nodes[inner] = moved @ R.T
                held = held.copy()
                held[inner[shift > 1e-9]] = True
                # heights may have changed: refresh force densities, then re-balance
                dz = nodes[edges[:, 1], 2] - nodes[edges[:, 0], 2]
                if np.any(dz <= 0):
                    warnings.append("The domain forces some branches to run downwards")
                    dz = np.maximum(dz, 1e-6)
                q = carried[edges[:, 1]] / dz
                solve_xy(held)
                warnings.append(f"{int((shift > 1e-9).sum())} branch point(s) moved inside the domain")

    # ----------------------------------------------------------- forces & radii
    vec = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    length = np.linalg.norm(vec, axis=1)
    forces = q * length
    fmax = max(float(forces.max()), _EPS)
    r_lo, r_hi = float(st.radius_min), max(float(st.radius_max), float(st.radius_min))
    radii = np.clip(r_hi * np.sqrt(forces / fmax), r_lo, r_hi)

    # ----------------------------------------------------------- member curves
    world_nodes = nodes @ R
    in_dir = np.zeros((n, 3))
    in_dir[:] = up
    for e, (a, b) in enumerate(edges):
        in_dir[b] = _unit(world_nodes[b] - world_nodes[a])
    organic = float(np.clip(st.organic, 0.0, 1.0))
    samples = max(2, int(st.samples))
    curves, curve_r = [], []
    for e, (a, b) in enumerate(edges):
        p0, p1 = world_nodes[a], world_nodes[b]
        chord = p1 - p0
        L = float(np.linalg.norm(chord))
        cdir = _unit(chord)
        t0 = _unit((1 - organic) * cdir + organic * (up if kind[a] == 0 else in_dir[a]))
        t1 = _unit((1 - 0.6 * organic) * cdir + 0.6 * organic * up) if kind[b] == 2 else cdir
        curve = _hermite(p0, p1, t0 * L, t1 * L, samples)
        s = np.linspace(0.0, 1.0, samples)
        r = np.full(samples, radii[e])
        if kind[a] == 0:  # footing flare over the first ~25% of the trunk
            r *= 1.0 + (float(st.footing_flare) - 1.0) * _smoothstep(1.0 - s / 0.25)
        if kind[b] == 2:  # trumpet-shaped capital over the last ~35% of the branch
            r *= 1.0 + (float(st.capital_flare) - 1.0) * _smoothstep((s - 0.6) / 0.4) ** 1.3
            # run into the carried object so the tube's round end stays hidden inside it
            tip = curve[-1] + up * r[-1]
            curve = np.vstack([curve, tip])
            r = np.append(r, r[-1])
        curves.append(curve)
        curve_r.append(r)

    return Tree(
        nodes=world_nodes,
        edges=edges,
        forces=forces,
        radii=radii,
        kind=kind,
        depth=depth,
        up=up,
        leaf_weights=leaf_w,
        curves=curves,
        curve_radii=curve_r,
        warnings=warnings,
    )
