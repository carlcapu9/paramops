# Stress Sculpt — Blender add-on: implementation spec (shared contract)

## 0. What we are building

The user (Italian, writes in Italian) asked for a Blender add-on with this idea:

> Turn the forces applied to an object into a sculptural language. Instead of only
> showing colors on a surface, the add-on uses stress information to generate new
> geometry.
> * Place supports and loads.
> * The system estimates the stress distribution.
> * Selected zones thicken, hollow out or branch.
> * The geometry evolves according to the rules you set.
> * You can compare different configurations.
> Killer feature: turn a force distribution into a 3D sculpture that looks as if it
> formed itself following the tension lines.
> Result: organic structures, biomimetic objects, architecture inspired by
> structural optimization.

Repository: `/home/user/paramops` (git). Add-on package: `/home/user/paramops/stress_sculpt/`.
Tests: `/home/user/paramops/tests/`.

Pipeline:
1. **Setup** — a closed mesh is the *design domain*; empties are *supports*, *loads* and
   *zones* (keep solid / keep void / thicken / hollow / branch). Empties or meshes can be zones.
2. **Analyze** — voxelize the domain, solve 3D linear elasticity (8-node hex FEM on a
   voxel grid, matrix-free PCG with multigrid preconditioner, pure numpy), show a heatmap.
3. **Evolve** — iterative shape evolution, live in the viewport: SIMP topology
   optimization ("Optimize") or stress-driven growth/erosion rules ("Growth Rules"),
   with zones biasing thicken/hollow, symmetry, passive solid/void; result meshed with
   surface nets + Taubin smoothing, colored by stress.
4. **Tension lines** — principal stress trajectories (tension = max principal,
   compression = min principal) traced through the stress field, with branching
   (secondary lines sprout from primary ones where the crossing principal stress is
   significant, recursively) → Blender curves with per-point radius ∝ stress.
5. **Sculpt** (killer feature) — fuse the lines (optionally + evolved solid) into one
   watertight organic mesh: metaball-like implicit field of tubes with smooth union,
   isosurface, smoothing, stress colors.
6. **Relief** — copy of the domain surface pushed outwards where stress is high and
   inwards where it is low.
7. **Variants** — save results as variants (copies in their own collection + metrics +
   settings JSON), list them, compare them side by side (collection instances in a row
   with text labels), restore settings, delete.

## 1. Hard constraints

* `stress_sculpt/core/**` is **pure numpy** — never import `bpy`, `mathutils`, `bmesh`,
  `scipy`. Must run with plain CPython ≥3.10 and numpy ≥1.23 (Blender 4.2 ships numpy 1.26,
  Blender 5.x ships numpy 2.x). Avoid numpy-2-only APIs (`np.astype`, `copy=` in `np.asarray`,
  `np.trapezoid`, `np.unstack`, `np.vecdot`, …) and APIs removed in numpy 2 (`np.float_`,
  `np.product`, `np.in1d` is ok-ish but prefer `np.isin`, …). No Python 3.12+ syntax.
* Core code must be thread-safe (no global mutable state) – it runs in a worker thread.
* Long core functions accept optional `progress(fraction, message)` and
  `should_stop() -> bool` callables (see `core/grid.py`: `ProgressFn`, `StopFn`,
  `Cancelled`, `check_stop`). Call `check_stop(should_stop)` at least every ~0.2 s of work.
  Topopt step loops / PCG loops / tracing loops must poll it.
* Blender target: **4.2 LTS … 5.2**. Test environment is Blender **5.2.3** as a Python module
  (`bpy`), see §8. Never access registered properties dict-style (`obj["prop"]`), always
  attribute access. Guard APIs that differ across versions with `hasattr`/try-except.
* All user-facing strings in the Blender layer are English (the user explicitly wants an
  English-only add-on: no translations module).
* Float64 in core; convert to float32/int32 only when writing to Blender with `foreach_set`.
* Match the code style of `core/grid.py` and `props.py` (docstrings, naming, 4-space, ~100 cols).
* Do not modify files owned by other agents (see §9). If you find a bug in someone else's file,
  report it in your final answer instead of editing (unless you are the integration agent).

## 2. Shared, already written (read them first, do not change their API)

* `stress_sculpt/core/grid.py` — `Grid`, `Problem`, `HEX_NODE_OFFSETS`, `Cancelled`,
  `check_stop`, conventions (element/node/dof numbering, Voigt order). READ IT.
* `stress_sculpt/props.py` — every Blender property (scene + object + variant). READ IT.
  Property names there are the contract for the UI and the operators.

## 3. `core/fem.py` — voxel FEM

```python
def elasticity_matrix(nu: float) -> np.ndarray            # (6,6) isotropic D for E=1, Voigt [xx,yy,zz,xy,yz,zx], engineering shear
def hex8_ke(nu: float) -> np.ndarray                      # (24,24) stiffness of a unit cube (h=1, E=1), 2x2x2 Gauss, symmetric
def hex8_b_center() -> np.ndarray                         # (6,24) strain-displacement matrix at the centroid of a unit cube
def voigt_to_tensor(s6) -> np.ndarray                     # (...,6) -> (...,3,3) symmetric
def von_mises(s6) -> np.ndarray                           # (...,6) -> (...)
def principal(s6) -> tuple[np.ndarray, np.ndarray]        # eigvals (...,3) ascending, eigvecs (...,3,3) (columns)
def connected_components(mask) -> tuple[np.ndarray, int]  # 26-connectivity labels (0 = background, 1..n), n
def remove_unsupported_components(problem) -> tuple[Problem, int]  # drop domain parts touching no fixed dof; returns new Problem, n removed
def check_problem(problem) -> list[str]                   # human-readable warnings, see below

class FEModel:
    def __init__(self, problem: Problem, solver: str = "MG", emin: float = 1e-3): ...
    # attributes (public, read-only by convention):
    problem, grid
    elems     # (n_act,) int64 flat ids of active elements (= problem.domain, C-order)
    edof      # (n_act, 24) int64
    KE        # (24,24) = problem.E * grid.h * hex8_ke(nu)
    free      # (n_dofs,) bool — dof of a node touched by an active element and not fixed
    def stiffness_scale(self, rho_act, penal: float) -> np.ndarray   # emin + (1-emin) * rho**penal, (n_act,)
    def apply_K(self, u, scale) -> np.ndarray                         # K(scale) @ u, result zero on non-free dofs
    def solve(self, scale, x0=None, tol=1e-6, maxiter=4000, should_stop=None) -> tuple[np.ndarray, dict]
        # PCG on free dofs; returns u (n_dofs,) with zeros on non-free dofs and
        # info = {"iterations": int, "residual": float (relative), "converged": bool, "solver": "MG"|"JACOBI"}
        # x0 = warm start. MG falls back to JACOBI automatically if the hierarchy can't be built.
    def element_energy(self, u) -> np.ndarray      # (n_act,) ue^T KE ue (unscaled) — compliance = sum(scale*energy)
    def element_stress(self, u, scale) -> np.ndarray   # (n_act,6) physical stress = scale * E * D(nu) @ (B_center/h) @ ue
    def to_grid(self, values_act, fill=0.0) -> np.ndarray  # scatter (n_act, ...) onto (nx,ny,nz, ...)

@dataclass
class AnalysisResult:
    u: np.ndarray            # (n_dofs,)
    stress: np.ndarray       # (nx,ny,nz,6), zeros outside the domain
    von_mises: np.ndarray    # (nx,ny,nz)
    displacement: np.ndarray # (nx,ny,nz) |u| averaged over the 8 element nodes, zeros outside
    compliance: float        # forces . u
    max_displacement: float  # max nodal |u|
    info: dict               # solver info + "time": seconds

def analyze(problem, solver="MG", tol=1e-6, maxiter=4000, progress=None, should_stop=None) -> AnalysisResult
```

Details:
* Matrix-free product: `ue = u[edof]` → `fe = (ue @ KE) * scale[:, None]` → `np.bincount(edof.ravel(), fe.ravel(), minlength=n_dofs)`; zero non-free.
* Jacobi-PCG baseline. **Multigrid preconditioner** (default): geometric, matrix-free, levels
  built by 2× coarsening of the element grid (coarse element (I,J,K) has children
  (2I+a,2J+b,2K+c), a,b,c∈{0,1}); coarse stiffness scale = sum(children scales)/8 (missing /
  inactive children count 0); coarse element active iff scale>0; coarse KE = KE(h_c) (KE ∝ h).
  Prolongation = trilinear (tensor product of 1-D: even fine node ← coarse i/2, odd ← average of
  neighbours), applied per component; restriction = exact transpose of prolongation. Coarse fixed
  dofs: a coarse dof is fixed if its prolongation column touches any fine fixed dof of the same
  component (over-constraining is fine – it is only a preconditioner). Coarse free dofs = touched
  by an active coarse element and not fixed. Damped Jacobi smoother ω≈0.6, ν_pre = ν_post (≥2),
  symmetric V-cycle (so PCG stays valid). Coarsest level when free dofs ≤ ~3000 or any dimension
  ≤ 2: assemble the dense matrix of free dofs and use a precomputed inverse (add 1e-12·mean(diag)
  regularization). Cap levels sensibly. The preconditioner must be SPD.
* Must be correct for arbitrary (non-box) domains, holes, non-power-of-two shapes, odd sizes.
* Stress recovery at element centroid (`B_center/h`), Voigt order of grid.py.
* `check_problem` warnings (strings): empty domain; no fixed dofs; no loads / zero force; loads
  entirely on fixed dofs; **mechanism**: rigid-body modes not restrained — build the 6 rigid modes
  (3 translations, 3 rotations about the domain centroid) restricted to the fixed dofs, if rank < 6
  (SVD with relative tolerance) → warning "supports do not prevent rigid motion (…)";
  disconnected parts without supports.
* Performance target (4 CPU cores, numpy): 40×20×20 elements solid box cantilever, MG solve
  to 1e-6 in < 3 s; Jacobi must still work. Report measured numbers.
* Tests `tests/test_fem.py` (pytest, plain python): KE symmetric PSD with exactly 6 zero
  eigenvalues; apply_K equals an explicitly assembled dense K on a small irregular domain;
  PCG (both preconditioners) equals dense direct solve; **patch test** (uniaxial tension block with
  symmetry BCs and consistent nodal tractions → uniform σxx = F/A, others ≈ 0, ux linear);
  cantilever tip deflection within ~25% of Timoshenko beam theory (hex8 locking is ok);
  MG iteration count clearly below Jacobi on a 32×16×16 problem and with SIMP-like random
  densities in [emin,1]; mechanism detection; components removal; von Mises / principal sanity.

## 4. `core/topopt.py` — evolution

```python
@dataclass
class OptSettings:
    mode: str = "SIMP"          # "SIMP" | "RULES"
    volfrac: float = 0.3
    penal: float = 3.0
    penal_start: float = 1.0    # continuation: penal ramps linearly from penal_start to penal
    continuation_iters: int = 15
    rmin: float = 1.5           # density filter radius in voxels
    move: float = 0.2
    emin: float = 1e-3
    max_iter: int = 40
    tol_change: float = 0.01
    grow_threshold: float = 0.6     # RULES
    hollow_threshold: float = 0.25  # RULES
    rate: float = 0.15              # RULES
    keep_volume: bool = True        # RULES
    symmetry: tuple = (False, False, False)
    solver: str = "MG"
    solver_tol: float = 1e-5
    solver_maxiter: int = 4000
    initial_density: Optional[float] = None   # None → volfrac (SIMP) / 1.0 (RULES)

@dataclass
class IterationState:
    iteration: int              # 1-based count of completed iterations
    density: np.ndarray         # (nx,ny,nz) physical (filtered, passive-applied) density, 0 outside domain
    stress: np.ndarray          # (nx,ny,nz,6) physical stress of the current design
    von_mises: np.ndarray       # (nx,ny,nz)
    compliance: float
    volume: float               # mean density over domain elements
    change: float               # max |Δx| of design variables this iteration
    converged: bool
    solver_iterations: int
    penal: float

class DensityFilter:
    def __init__(self, domain: np.ndarray, rmin: float): ...   # cone weights max(0, rmin - dist), only domain neighbours
    def forward(self, x_grid) -> np.ndarray      # filtered field, 0 outside domain
    def backward(self, dx_grid) -> np.ndarray    # exact chain rule (adjoint) of forward
    # zero-padded shifts, NEVER np.roll (no wrap-around)

class Optimizer:
    def __init__(self, problem: Problem, settings: OptSettings): ...
    def step(self, should_stop=None) -> IterationState
    def run(self, callback=None, should_stop=None) -> IterationState
        # callback(state) is called after every iteration; returning False stops early.
    @property
    def done(self) -> bool      # converged or max_iter reached
    model: FEModel              # public
```

* SIMP: compliance minimization, density filter (forward on x, adjoint on sensitivities),
  OC update with bisection on the Lagrange multiplier honouring move limit, bounds [0,1],
  passive solid=1 / void=0, volume constraint on mean physical density over domain elements.
  Zone **bias** b∈[-1,1] (from `problem.bias_or_zero()`): multiply the compliance sensitivity
  magnitude by `(1 + b)` (thicken zones attract material, hollow zones repel it; b=-1 → never).
  Warm-start the solver with the previous displacement. Penalty continuation.
* RULES ("growth rules", BESO-flavoured but continuous): start from `initial_density` (default 1);
  each iteration solve, compute solid-equivalent von Mises s = vm / scale (stress the material
  would carry, undefined for void → use energy-based estimate or vm/max(scale, emin)), filter it
  with the density filter, normalize by its 90th percentile over material elements (x>0.5, fallback
  all domain), apply bias `s *= (1 + b)`, then
  `dx = rate * (2*clip((s - hollow)/(grow - hollow), 0, 1) - 1)` (≤hollow → −rate, ≥grow → +rate).
  If `keep_volume`: add a uniform shift found by bisection so the new mean density approaches
  volfrac (limited to ±rate per iteration). x = clip(x + dx, 0, 1); filter; passives.
  Converges when change < tol_change for 3 consecutive iterations or max_iter.
* Symmetry: after each update average x with its mirror along each enabled axis (mirror across the
  grid mid-plane, only where both cells are in the domain); reapply passives.
* Must be robust: empty bias ok, all-passive ok, volfrac ≥ 1 → full domain, domain with holes.
* Tests `tests/test_topopt.py`: filter forward/backward adjointness (<Fx,y> = <x,Fᵀy>), filter
  preserves constant fields inside the domain, no wrap-around; finite-difference check of the
  filtered compliance sensitivity on a tiny problem; MBB / cantilever 3D: compliance decreases
  overall and volume constraint satisfied within 1e-3 after a few iterations; passive regions
  respected; symmetry exact; RULES mode reaches ~volfrac with keep_volume and keeps material near
  the load path; should_stop raises Cancelled.

## 5. `core/mesher.py`, `core/sculpt.py` — geometry

```python
# mesher.py
def surface_nets(field, iso, origin, h, pad_value=None) -> tuple[np.ndarray, np.ndarray]
    # field (nx,ny,nz) sampled at ELEMENT CENTRES origin + (ijk+0.5)*h; returns verts (V,3) float64 world,
    # quads (F,4) int64. The field is padded with one layer of pad_value (default: min(field.min(), iso) - 1)
    # so the surface is closed. Normals (right-hand rule on quad order) point OUTWARDS, i.e. towards
    # field < iso. Vertex = mean of the edge crossings of its dual cube (linear interpolation).
    # Empty result → (zeros((0,3)), zeros((0,4), int)). Unused verts removed, indices compact.
def triangulate(quads) -> np.ndarray                       # (2F,3), consistent orientation
def vertex_normals(verts, faces) -> np.ndarray             # area-weighted, unit length (works for tris & quads)
def mesh_edges(faces) -> np.ndarray                        # (E,2) unique undirected edges
def taubin_smooth(verts, faces, iterations=10, lam=0.5, mu=-0.53, pinned=None) -> np.ndarray
def laplacian_smooth_scalar(values, faces, n_verts, iterations) -> np.ndarray   # smooth per-vertex scalar over mesh edges
def extrapolate(field, mask, iterations=3) -> np.ndarray   # fill cells outside mask near it with the mean of valid 6-neighbours, iteratively (supports trailing channel dims)
def sample_trilinear(field, origin, h, points, mask=None) -> np.ndarray
    # field at element centres, (nx,ny,nz) or (nx,ny,nz,C); points (P,3) world; clamps to the grid;
    # if mask given: weights only from mask cells, renormalized; points with no valid neighbour → nearest valid via a few extrapolate passes or 0.
def colormap(values, vmin, vmax, name="HEAT") -> np.ndarray   # (P,4) float RGBA in [0,1] (linear-ish), names HEAT|COOLWARM|VIRIDIS; COOLWARM is diverging around 0 if vmin<0<vmax
def remove_small_components(verts, faces, min_faces) -> tuple   # drop tiny floating islands (connected components by shared verts)

# sculpt.py
@dataclass
class Tube:
    points: np.ndarray   # (n,3) world
    radius: np.ndarray   # (n,) world
    value: np.ndarray    # (n,) scalar carried to the surface (stress magnitude), any units
def tubes_field(tubes, voxel_size, blend=0.5, pad=2, progress=None, should_stop=None) -> tuple[field, origin, h, value_field]
    # Implicit field on a regular grid (element-centred like everything else, shape fits the tubes' bbox + pad).
    # Per tube: exact capsule distance d to its polyline (per-segment, radius linearly interpolated along the segment),
    # f_tube = phi(d / (R / t_iso)) with Wyvill phi(t) = (1 - t^2)^3 for t<1 else 0, chosen so f_tube == ISO (=0.5 for one tube alone… use a documented ISO)
    # exactly at d == R. Tubes are combined with a smooth union controlled by blend ∈ [0,1]
    # (0 → max (hard union), 1 → sum (metaball fillets)); e.g. f = max + blend*(sum - max) .
    # value_field: field-weighted average of tube values (for coloring).
    # Must be vectorized and chunked (memory bounded: never allocate > ~200 MB);
    # only evaluate grid points within each segment's bbox + R.
TUBE_ISO: float   # iso level to mesh tubes_field with
def resample_to(field, src_origin, src_h, dst_shape, dst_origin, dst_h) -> np.ndarray  # trilinear resample of element-centred fields
def union_fields(a, b, blend) -> np.ndarray      # same smooth-union rule
```
* Tests `tests/test_mesher.py`, `tests/test_sculpt.py`: sphere field → closed 2-manifold
  (every edge shared by exactly two faces), Euler characteristic 2, outward normals (signed
  volume > 0 and close to 4/3πr³ within ~5% at moderate resolution), area close; torus → χ = 0;
  box-shaped binary field; empty and full fields; taubin keeps volume within a few %;
  sample_trilinear exact for linear fields; colormap ranges; single straight tube → mesh radius ≈
  R within 10%, two crossing tubes → one connected component with blend>0; memory/time on 200 tubes.

## 6. `core/streamlines.py` — principal stress trajectories

```python
@dataclass
class StressField:
    grid: Grid
    stress: np.ndarray        # (nx,ny,nz,6) Voigt
    solid: np.ndarray         # (nx,ny,nz) bool — where lines may travel (domain, or density>0.5)
    def sample_tensor(self, points) -> np.ndarray  # (P,3,3) masked trilinear interpolation of the TENSOR (not of eigenvectors)
    def inside(self, points) -> np.ndarray         # (P,) bool: the element containing the point is solid (and point inside grid)

@dataclass
class Polyline:
    points: np.ndarray        # (n,3) world
    magnitude: np.ndarray     # (n,) |principal stress of the family| at each point [Pa]
    family: str               # "TENSION" (max principal) | "COMPRESSION" (min principal)
    generation: int           # 0 = primary, 1.. = branches
    sign: np.ndarray          # (n,) sign of the family's principal stress (+1 tension, -1 compression)

@dataclass
class TraceSettings:
    families: tuple = ("TENSION", "COMPRESSION")
    n_seeds: int = 40
    seed_points: Optional[np.ndarray] = None   # (k,3) extra seeds (loads, supports) – traced first
    step: float = 0.5              # in voxels
    max_steps: int = 4000          # per direction
    min_stress: float = 0.02       # relative to the 99th percentile of |family stress| in the solid
    separation: float = 2.0        # voxels; a line stops when it gets closer than this to an accepted line of the same family
    branch_depth: int = 1
    branch_spacing: float = 4.0    # voxels between candidate branch points along a line
    branch_ratio: float = 0.3      # spawn a branch of the other family where |σ_other| >= ratio * |σ_this|
    branch_weight: Optional[np.ndarray] = None  # (nx,ny,nz) in [0,1]: BRANCH zones; increase branching probability and seeding density there
    min_points: int = 5
    rng_seed: int = 0
    max_lines: int = 2000

def trace(field: StressField, settings: TraceSettings, progress=None, should_stop=None) -> list[Polyline]
```
* Integration: RK2 (midpoint) or RK4 on the eigenvector field of the interpolated tensor,
  family = eigenvector of the largest (TENSION) or smallest (COMPRESSION) eigenvalue; keep
  orientation continuity (flip the eigenvector if its dot with the previous direction < 0);
  trace both directions from each seed and join them into one polyline.
  Stop on: leaving the solid / grid, |σ_family| < min_stress·ref, max_steps, closing a loop
  (back near its own start after ≥ some length), and separation from previously accepted lines
  of the same family (spatial hash / occupancy grid with cell = separation; the seed's own
  vicinity is exempt for the first few steps).
* **Batch/vectorize**: advance all active streamlines of a round simultaneously with batched
  `np.linalg.eigh`; acceptance + separation clipping can then be done line by line (vectorized per
  line). Target: 300 lines on a 60×30×30 field in < 10 s.
* Seeding: `seed_points` first; then `n_seeds` samples drawn from solid element centres with
  probability ∝ |σ_family|·(1 + 3·branch_weight), jittered within the voxel, rejecting seeds within
  `separation` of accepted lines. Deterministic for a fixed rng_seed.
* Branching: for generation g < branch_depth, walk every accepted line of generation g at
  `branch_spacing`; where |σ_other|/|σ_this| ≥ branch_ratio·(1 − 0.5·branch_weight) seed the
  other family (generation g+1). The same separation rules apply.
* Tests `tests/test_streamlines.py`: uniaxial σxx field → straight lines along x, compression
  family along… (pick a field with σ1 along x, σ3 along y); pure shear rotated 45°; a ring
  (radial/hoop) field → closed-ish circles; lines stay inside the solid; separation respected;
  determinism; branching produces generation-1 lines of the other family; stopping in
  low-stress regions; should_stop.

## 7. Blender layer

### 7.1 Files
* `__init__.py` — `bl_info` (name "Stress Sculpt", blender (4, 2, 0), category "Object",
  location "View3D > Sidebar > Stress Sculpt", version (1, 0, 0), author "ParamOps",
  description …), `register()/unregister()` calling `props`, `operators`, `ui`
  in order (and reverse). Use relative imports. `__init__.py` must stay importable WITHOUT bpy
  (so `stress_sculpt.core` can be tested with plain python):
  `try: import bpy` / `except ImportError: bpy = None`, and import the Blender submodules only
  when bpy is available. Support module reload (`importlib.reload` when re-registering).
* `blender_manifest.toml` — Blender ≥4.2 extension manifest (schema_version "1.0.0",
  id "stress_sculpt", version "1.0.0", name "Stress Sculpt", tagline ≤64 chars without trailing
  punctuation, maintainer "ParamOps", type "add-on", tags ["Modeling", "Object", "Physics"],
  blender_version_min "4.2.0", license ["SPDX:GPL-3.0-or-later"], permissions none needed).
* `props.py` — written (contract).
* `scene_io.py` — bpy → numpy (`gather_problem`), voxelization.
* `build.py` — numpy → bpy (meshes, curves, materials, collections, text).
* `jobs.py` — background-thread job runner + runtime state + result cache.
* `operators.py` — all operators.
* `ui.py` — sidebar panels + UIList.

### 7.2 Object conventions
* Roles live in `obj.stress_sculpt.role`; disabled objects (`enabled == False`) are ignored.
* **Support** empty: region = the empty's display shape: `CUBE` → points whose local coords are
  within ±`empty_display_size` on every axis (uses full matrix_world, so scaling/rotation work);
  anything else (SPHERE default) → ellipsoid of radius `empty_display_size` in local space.
  Fixed axes fix_x/y/z (world axes). If no active node lies inside, use the nearest active node(s)
  (all nodes within nearest distance + 0.5 h) and add a warning.
* **Load** empty (display `SINGLE_ARROW`): direction = normalized world-space local +Z,
  magnitude `magnitude` N, spread over active nodes within `load_radius` (0 → 1.5 h) of its
  location, equally; if none, the nearest active node(s). Several loads add up.
* **Zone**: empties (same region rule as supports) or MESH objects (voxelized interior, same
  method as the domain) with roles KEEP_SOLID → `passive_solid`, KEEP_VOID → `passive_void`,
  THICKEN → `bias += strength`, HOLLOW → `bias -= strength`, BRANCH → `branch = max(branch, strength)`.
  Element centres decide membership.
* A mesh object with a role other than NONE is never the domain; the domain is
  `scene.stress_sculpt.domain`.
* Objects created by the add-on: setup objects go to collection **"StressSculpt Setup"**,
  results to **"StressSculpt Results"** (child collections of the scene collection, created on
  demand). Result object names: `SS_Stress` (heatmap), `SS_Evolved`, `SS_Lines_Tension`,
  `SS_Lines_Compression`, `SS_Sculpture`, `SS_Relief`. Re-running replaces the mesh/curve data
  of the existing object (keep the object, its transform = identity world, its materials),
  removing the old datablock if it has no other users.
* Materials (created once, reused, node-based, Blender 4.2–5.x compatible):
  `SS_Heatmap` (Color Attribute node "Stress" → Principled Base Color, roughness 0.5),
  `SS_Tension` (warm red/orange), `SS_Compression` (cool blue), `SS_Sculpture` (attribute color
  "Stress" blended with a warm stone/clay look). Meshes with stress get a FLOAT_COLOR POINT color
  attribute named "Stress" (set as active/render color) and a FLOAT POINT attribute "stress"
  (raw values). Smooth shading on generated meshes (`mesh.shade_smooth()` if available, else
  polygon `use_smooth`).
* Variants: parent collection **"StressSculpt Variants"** linked to the scene but excluded in
  the view layer (so it persists in the file but is hidden); each variant = child collection
  "Variant N – <kind>" with copies (data copies) of the current result objects. Compare = create
  collection **"StressSculpt Compare"** with one collection-instance empty per variant placed in a
  row along +X (spacing = max bbox width × `compare_spacing`, starting right of the domain), plus a
  FONT text object under each with name and metrics. Clear compare deletes that collection's
  objects. Remove variant deletes its collection + objects + datablocks without users.

### 7.3 `scene_io.py`
```python
def evaluated_world_mesh(obj, depsgraph) -> tuple[verts (V,3) float64, tris (T,3) int64]   # world space, modifiers applied, triangulated (loop_triangles)
def voxelize_mesh(verts, tris, grid) -> np.ndarray   # bool (nx,ny,nz): element centre inside (ray parity along +Z per (i,j) column using mathutils.bvhtree.BVHTree; robust to hits exactly on centres; dedupe double hits on shared edges)
def make_grid(domain_obj, depsgraph, resolution) -> Grid    # world bbox of evaluated mesh → Grid.from_bounds(..., pad=0)
def region_mask_points(obj, points) -> np.ndarray          # for empties: sphere/cube rule above; points (P,3) world
def gather_problem(context, settings=None) -> tuple[Problem, dict]   # info dict: {"warnings": [...], "loads": [(pos, dir, mag)], "supports": [pos...], "seed_points": (k,3), "domain_name": str, "n_elems": int}
def setup_signature(context) -> str     # hash of everything that affects the Problem (domain mesh eval, transforms, roles, props, resolution, material) for cache invalidation
```
Gathering happens on the main thread; raise `ValueError` with a clear message for fatal problems
(no domain, domain not a mesh, empty voxelization, no supports, no loads).
`remove_unsupported_components` is applied (warn). `check_problem` warnings are appended.
E = `youngs_modulus * 1e9` Pa.

### 7.4 `jobs.py`
* `RUNTIME` dict-like singleton: `running: bool`, `name`, `progress` (0..1), `message`, `cancel`
  (threading.Event). UI reads it.
* `CACHE` per scene name: `analysis` (problem, result, signature, grid), `evolution`
  (problem, final IterationState, settings), `lines` (list[Polyline], grid, h). Helpers
  `get_cache(scene)`, `clear_cache(scene)`.
* `class ThreadedOperatorMixin` used by long operators:
  - subclass implements `prepare(self, context) -> payload` (main thread, may raise ValueError →
    report ERROR and CANCELLED), `compute(self, payload, progress, should_stop, emit) -> result`
    (worker thread; pure numpy; `emit(obj)` queues partial results e.g. density snapshots),
    `on_partial(self, context, obj)` (main thread) and `finish(self, context, result)` (main thread).
  - `execute()` runs synchronously (used by scripts/tests, `bpy.app.background`, or when invoked
    with EXEC_DEFAULT). `invoke()` starts a `threading.Thread(daemon=True)`, a window-manager timer
    (0.1 s) and a modal handler; `modal()` drains the queue (only the latest partial per tick),
    updates progress + area header/status text, redraws VIEW_3D areas, handles ESC/right-click
    → cancel (the worker sees should_stop() True; after it stops, a cancelled job still calls
    `finish` with the last good partial if the subclass supports it, otherwise reports
    "Cancelled"); worker exceptions are reported with the message, traceback printed to console.
  - Only one job at a time (poll returns False while RUNTIME.running). Always clean up (timer,
    header text, RUNTIME) in every exit path, including errors in finish.

### 7.5 `operators.py` (bl_idname → behaviour; all have `bl_options = {"REGISTER", "UNDO"}` unless noted)
* `stress_sculpt.set_domain` — active mesh object becomes `settings.domain` (poll: active mesh).
* `stress_sculpt.add_support` — empty SPHERE at 3D cursor, display size = 5% of the domain
  diagonal (or 0.25 if no domain), role SUPPORT, name "SS_Support", color blue, into
  "StressSculpt Setup", selected + active. Property `shape` enum SPHERE/CUBE.
* `stress_sculpt.add_load` — empty SINGLE_ARROW at cursor pointing down (rotation x = π),
  display size = 15% of domain diagonal, role LOAD, name "SS_Load", color red.
* `stress_sculpt.add_zone` — property `role` (KEEP_SOLID, KEEP_VOID, THICKEN, HOLLOW, BRANCH),
  empty SPHERE at cursor, name "SS_<Role>".
* `stress_sculpt.set_role` — property `role` (all ROLE_ITEMS) → applied to all selected objects
  (never to the domain; warn).
* `stress_sculpt.demo_scene` — builds a ready-to-run example: if the scene has no domain,
  create a box mesh domain 4×1×1.5 (m) centred above the origin named "SS_Domain", two supports
  (CUBE empties covering the bottom ends; one fully fixed, one fixing only Z+Y — a roller still
  stable together), one load pressing down at the top centre with 5000 N; sets settings.domain.
  Property `preset` enum: BRIDGE (as described), CANTILEVER (wall support on one end, load at the
  free end tip), TOWER (tall box, base support, lateral + vertical load at the top), CHAIR? (optional).
* `stress_sculpt.analyze` (threaded) — gather → `fem.analyze` → build heatmap object `SS_Stress`
  (copy of evaluated domain mesh in world space colored by the chosen `stress_measure` sampled with
  `sample_trilinear` on `extrapolate`d fields, or the voxel surface when `heatmap_on_original` is
  False), store cache, update `last_*` props, set domain wire, switch VIEW_3D solid shading
  color_type to 'VERTEX' (attribute) where in SOLID mode. Report summary (max von Mises MPa, max
  displacement mm, warnings).
* `stress_sculpt.evolve` (threaded, live) — gather → `Optimizer` with OptSettings from props →
  emits a density snapshot each iteration (when live_update) → `on_partial` rebuilds `SS_Evolved`
  with surface_nets (no smoothing, or 2 iterations) colored by von Mises; `finish` builds final
  mesh with `smooth_iterations` Taubin, stress colors, cache, last_* props. Cancelling keeps the
  last state as result.
* `stress_sculpt.trace_lines` (threaded) — stress field from cache (`line_source`), recomputing the
  analysis if missing/outdated (signature mismatch); seeds from loads & supports if
  `seed_from_loads`; `branch_weight` from BRANCH zones; builds curve objects
  `SS_Lines_Tension` / `SS_Lines_Compression` (POLY splines, 3D, bevel_depth = 1 voxel·radius_max
  scaled via per-point radius: point.radius = r_i / bevel_depth where r_i maps magnitude^0.5 into
  [radius_min, radius_max] voxels × branch_taper^generation; bevel_resolution 2, fill caps). Stores
  lines in cache.
* `stress_sculpt.sculpt` (threaded) — needs lines in cache (else trace first, synchronously in the
  worker); tubes from lines (same radius mapping) → `sculpt.tubes_field(voxel_size = h /
  sculpt_detail, blend = sculpt_blend)`; if `LINES_AND_SOLID` and an evolution is cached, union with
  the evolved density resampled; surface_nets at TUBE_ISO, remove tiny components, Taubin
  `sculpt_smooth`, color by value field → `SS_Sculpture` with `SS_Sculpture` material.
* `stress_sculpt.relief` (threaded) — analysis from cache or recompute; copy of the evaluated
  domain mesh (optionally subdivided `relief_subdivisions` times — use bmesh `subdivide_edges` or
  a temporary Subsurf modifier evaluated via depsgraph), sample von Mises at vertices, normalize by
  the `color_clip` percentile, s^gamma, offset = relief_amount·diag·(s − threshold)/(1 − threshold)
  for s>threshold; if relief_hollow: negative offsets below threshold scaled by threshold;
  smooth the offset field (laplacian_smooth_scalar) then move along vertex normals → `SS_Relief`.
* `stress_sculpt.cancel` — sets RUNTIME.cancel (poll: running).
* `stress_sculpt.clear_results` — delete objects in "StressSculpt Results" (+ orphan data), clear
  cache, restore domain display to TEXTURED/SOLID.
* `stress_sculpt.variant_save` — copy current visible result objects into a new variant
  collection; metrics from `last_*`; `settings_json` = JSON of all scene-setting values (exclude
  variants, pointers → names); `setup_json` = list of role objects (name, role, location,
  rotation, magnitude, enabled). Name "Variant N" (+ kind).
* `stress_sculpt.variant_remove`, `stress_sculpt.variant_restore` (apply settings_json back,
  skipping unknown keys), `stress_sculpt.variant_compare`, `stress_sculpt.variant_clear_compare`,
  `stress_sculpt.variant_move` (direction UP/DOWN).
* `stress_sculpt.select_role` — select all objects with a given role (handy).
Every operator reports errors via `self.report({"ERROR"}, msg)` with actionable messages.

### 7.6 `ui.py`
View3D › Sidebar tab **"Stress Sculpt"**, panels (bl_idname prefix `STRESS_SCULPT_PT_`):
1. **Setup** — domain picker + "Use Active" button, resolution, voxelize method, material preset
   (E, ν when CUSTOM), buttons Add Support / Add Load / Add Zone (menu with zone roles), Demo
   (menu of presets), count of supports/loads/zones; sub-panel **Active Object** showing the role
   settings of the active object (role, enabled, magnitude+radius for loads, fix axes for supports,
   strength for zones) — hide fields that don't apply.
2. **Analyze** — stress measure, colormap, clip, heatmap_on_original, big Analyze button, last
   results (max stress in MPa, max displacement in mm, compliance) — formatted nicely.
3. **Evolve** — method, material fraction, iterations, smoothness (filter radius), mode-specific
   settings (penalty/move only under an "Advanced" toggle; rules thresholds/rate/keep_volume for
   RULES), symmetry X/Y/Z toggles in a row, surface level, smoothing, live preview, Evolve button.
4. **Tension Lines** — source, families, seeds, seed at loads, separation, min stress, branch
   levels/spacing/threshold, radius min/max, taper, random seed, Trace button.
5. **Sculpture** — fuse source, detail, blend, smoothing, Sculpt button (+ hint "traces lines if
   needed").
6. **Relief** — thickness, carve low stress, neutral stress, contrast, subdivisions, smoothing,
   Relief button.
7. **Variants** — UIList (name, kind, volume %, compliance) + side column add/remove/up/down,
   details of the active variant, Restore Settings, Compare / Clear Compare, spacing.
8. **Solver** (closed by default) — solver, tolerance, max iterations, domain_as_wire, Clear Results.
While a job runs: every panel header area shows a progress row (`layout.progress` when available
(Blender ≥4.0) else a label with percentage) and a Cancel button; action buttons disabled.

## 8. Environment & how to test

* Plain python (core tests): `cd /home/user/paramops && python3 -m pytest tests -q` — system
  python3 has numpy but **no pytest**: use the venv python which has numpy + pytest + bpy:
  `PY=/tmp/claude-0/-home-user-paramops/64513836-2a10-5b5e-b175-3cbab71e423e/scratchpad/bvenv/bin/python`
  then `$PY -m pytest tests -q`. (`import bpy` works in that interpreter: Blender 5.2.3 as module,
  headless, background mode; GPU/viewport drawing not available, Cycles CPU renders work.)
* The machine has 4 cores, 16 GB RAM. Keep any single test < 60 s; whole suite < 5 min.
* Tests import the package as `stress_sculpt.core...`; `tests/conftest.py` (already written)
  puts the repo root on `sys.path`. Note: `stress_sculpt/__init__.py` imports bpy, so core tests
  run with the venv python (bpy importable) — core modules themselves never import bpy.
* Blender-layer tests: `tests/blender/test_*.py`, run with the venv python; they
  `import bpy`, register the add-on (`import stress_sculpt; stress_sculpt.register()`), build a
  scene, call operators with `bpy.ops.stress_sculpt.<op>()` (EXEC → synchronous), assert results,
  unregister in teardown. bpy in module mode keeps one global state: reset with
  `bpy.ops.wm.read_factory_settings(use_empty=True)` between tests.
* Do NOT commit or push — the orchestrator does that.

## 9. File ownership
| Agent | Files |
|---|---|
| fem | `core/fem.py`, `tests/test_fem.py` |
| topopt | `core/topopt.py`, `tests/test_topopt.py` |
| geometry | `core/mesher.py`, `core/sculpt.py`, `tests/test_mesher.py`, `tests/test_sculpt.py` |
| lines | `core/streamlines.py`, `tests/test_streamlines.py` |
| blender-io | `scene_io.py`, `build.py`, `jobs.py`, `operators.py`, `tests/blender/*` |
| orchestrator | `__init__.py`, `blender_manifest.toml`, `ui.py` |
