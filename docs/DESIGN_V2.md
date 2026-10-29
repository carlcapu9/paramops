# Stress Sculpt v2 — "Generate Structure" (spec addendum; SPEC.md still applies)

## Why
The user (Italian architect-minded Blender user) saw the first results (stress lines filling a
solid box = a hairball; fused hairball "sculpture"; displaced box) and said, verbatim:
"it doesn't look at all like an architectural structure". What they want:

> I want to put the weight on specific points of a mesh, e.g. a cube, and get a beautiful
> architectural structure that respects the weight where it was placed — like those cool
> parametric architectural structures.

Asked to choose, they want BOTH modes and ALL THREE styles:
* **Mode SUPPORT ("Support Object")** — the mesh (e.g. a floating cube) is carried; a structure
  grows from the ground (or from footing points) up to the loaded points.
* **Mode CARVE ("Carve Inside")** — the mesh is the block of material; loads and supports are
  points on it; the structure forms inside it.
* **Style ORGANIC** — topology optimization (SIMP): fluid generative-design shapes.
* **Style BRANCHING** — form-found funicular tree columns (Frei Otto / Gaudí), members sized
  by force. Core already written: `stress_sculpt/core/forms.py` (`grow_tree`, `TreeSettings`,
  `Tree.tubes()`), equilibrium exact (residual ~1e-12).
* **Style RIBS** — Nervi-like isostatic ribs: a lattice of ribs following the principal stress
  lines *on the surface* of the optimized shape (or of the object itself in CARVE mode).

Everything must make results that look like architecture, not noise: few, clean, continuous
members; smooth junctions; respect for the given load points and supports.

## UI (already written by the orchestrator: `stress_sculpt/ui.py`, read it)
Main panel "Structure": target object, mode, style, "Put Weight on Selection" (Edit Mode),
supports/ground, "Generate Structure". Sub-panels: style settings, active object. Other panels:
Stress Analysis, Variants, More Tools (zones, raw stress lines, fuse lines, relief), Material &
Solver. New props are in `props.py` (section "structure (main flow)": `target`,
`structure_mode`, `structure_style`, `ground_height`, `ground_contact`, `footings`,
`support_material`, `load_pad`, `tree_*`, `rib_*`). Existing props are reused
(`resolution`, `filter_radius`, `max_iterations`, `volume_fraction` (CARVE organic),
`smooth_iterations`, `live_update`, symmetry, `evo_mode`…, `line_families` (rib families),
`sculpt_blend` (rib junction blend)).

Operators the UI calls that do not exist yet (bl_idname → behaviour):
* `stress_sculpt.set_target` — active mesh → `settings.target` (poll: active object is a mesh
  and not a role object).
* `stress_sculpt.add_load_at_selection` — works in Edit Mode on any mesh (poll: edit mesh) and
  also in Object Mode on the active mesh using its stored selection. Reads the selected
  vertices (vertex/edge select mode) or the selected face centres (face select mode — when
  faces are selected use face centres, area-weighted for TOTAL). Properties: `force` (N,
  default 1000), `distribute` (`PER_POINT` | `TOTAL`, default PER_POINT), `direction`
  (`GRAVITY` world −Z, `INTO_SURFACE` along −normal, default GRAVITY), `max_points` (64:
  more points are merged by force-weighted k-means — reuse `forms.cluster_loads`).
  Creates one LOAD empty per point (SINGLE_ARROW, arrow along the force, display size ≈ 8% of
  the object's bounding diagonal, red object color), named `SS_Load.###`, PARENTED to the mesh
  (keep world transform: `matrix_parent_inverse`), linked into "StressSculpt Setup".
  Sets `settings.target` to the mesh if unset. Stays in Edit Mode. Reports how many weights
  were added and the total. Must work when called repeatedly (adds more).
* `stress_sculpt.add_support_at_selection` — same selection logic; SUPPORT empties (SPHERE,
  radius ≈ 1.5 voxels of the current resolution or 3% of the object diagonal, all axes fixed,
  blue), parented to the mesh. In SUPPORT mode, support empties act as explicit footings (their
  position is projected to the ground height if they are above it? NO — use them as given; the
  user places footings where they like; if they are on the carried object, warn).
* `stress_sculpt.remove_role` — property `role`: delete all objects with that role (only
  empties; meshes just get role NONE), purge data.
* `stress_sculpt.generate` — threaded (ThreadedOperatorMixin), live preview for ORGANIC; see
  below. Result object **`SS_Structure`** (mesh) in "StressSculpt Results", replacing the
  previous one; other result objects hidden. `last_kind` = "ORGANIC" | "BRANCHING" | "RIBS".
* `stress_sculpt.verify` — FEM check of `SS_Structure`: voxelize it (resolution setting), loads
  = the LOAD empties (spread radius ≥ 1.5 voxels, snapped to the nearest structure nodes),
  supports = footings / ground contact as used by generate (SUPPORT) or the SUPPORT empties
  (CARVE) → `fem.analyze` → color SS_Structure's vertices by von Mises (Stress color
  attribute, heatmap material) and report max stress / displacement; update `last_*`.
* `stress_sculpt.demo_scene` — replace the presets with architectural ones (keep the operator
  and the `preset` enum property name; ui uses `operator_enum`):
  - `FLOATING_CUBE` (default): 2 m cube floating with its bottom at 3 m, 8 weights of 2000 N on
    its bottom corners + bottom edge mid-points, mode SUPPORT, style BRANCHING, ground 0.
  - `CANOPY`: 5×5×0.25 m roof slab at 3.2 m, 5×5 grid of 800 N weights on top, 4 automatic
    footings, SUPPORT + ORGANIC.
  - `PAVILION`: an irregular floating volume (e.g. a 4×2.5×1 m box rotated 15° about Z) with
    weights on its bottom face centres (a 4×3 grid), footings: 3 automatic, SUPPORT + RIBS.
  - `BRIDGE`: 8×1.2×2.5 m block, weights along the top (9 × 3000 N), supports at the two
    bottom ends, CARVE + ORGANIC (arch bridge expected).
  - `CARVED_CUBE`: 2 m cube on the ground, weights on the 4 top corners, supports at the
    4 bottom corners, CARVE + RIBS.
  Each preset sets target, mode, style and suitable settings, removes objects of an earlier
  demo (existing DEMO_TAG mechanism), frames nothing (no view ops needed).

## Generate: algorithms per mode × style
Gathering (main thread, in `prepare`): loads = enabled LOAD objects (world position, force
vector = magnitude × world +Z axis of the empty); supports = enabled SUPPORT objects.
Validate with actionable errors: no target; no loads; CARVE without supports; SUPPORT with all
loads below the ground.

### SUPPORT mode
* Footings: explicit SUPPORT empties (any count) or, when none and `ground_contact == FOOTINGS`,
  automatic: same rule as `forms.grow_tree` (weighted k-means of the load points in plan,
  `footings` = 0 → automatic count) at `ground_height`. Expose a helper in forms (e.g.
  `forms.auto_footings(points, weights, height, n)`) and use it for ORGANIC/RIBS too, so all
  styles stand on the same footings.
* **BRANCHING**: `forms.grow_tree(points, forces, TreeSettings(...), supports=explicit or None,
  ground=ground_height)`. Radii: `radius_max = tree_thickness × H` where H = mean load height
  above ground; `radius_min = tree_min_thickness × radius_max`. Field:
  `sculpt.tubes_field(tree.tubes(), voxel = radius_min / tree_detail, blend = tree_blend)`;
  cap the grid (SCULPT_MAX_CELLS) by enlarging the voxel with a warning. Mesh: surface_nets at
  TUBE_ISO, remove tiny components, Taubin `smooth_iterations`. Color attribute "Stress" =
  member force mapped with the colormap (value field) — material: a light stone/concrete look
  that shows the force colors subtly (e.g. mix 25% heat colors into an off-white base).
  `ground_contact == FULL` with BRANCHING: treat as automatic footings.
* **ORGANIC**: automatic design domain: box from `ground_height` up to the highest load point
  + 1 voxel; in plan the bounding box of loads + footings grown by `0.35 × H` (min 2 voxels).
  Grid: `Grid.from_bounds(.., resolution)`. Domain = whole box; KEEP_VOID = the target's interior
  (voxelized on the main thread with scene_io.voxelize_mesh) and anything above its bottom where
  inside it; KEEP_SOLID pads: spheres of `load_pad` voxels around every load point and footing
  (footing pads are half-discs on the ground); loads spread over the nodes of their pad;
  supports: nodes of the bottom layer inside the footing pads (all axes fixed), or the whole
  bottom node layer when `ground_contact == FULL`. Make sure the load pads are not inside the
  KEEP_VOID (pads win). Then SIMP (topopt.Optimizer, OptSettings from the props with
  `volfrac = support_material`) live, exactly like the evolve operator (reuse its code:
  `_opt_settings`, `evolved_geometry`, `_copy_state`…), result → SS_Structure.
* **RIBS**: run ORGANIC (or reuse the cached evolution when the problem signature matches), then
  ribs on its surface (see "Ribs" below).

### CARVE mode
* Problem = target interior (reuse `scene_io.gather_problem` with the target as the domain —
  add an optional `domain` override argument instead of mutating settings.domain), loads and
  supports from the empties (supports on the surface of the block).
* **ORGANIC**: SIMP with `volume_fraction`, live (like evolve) → SS_Structure.
* **BRANCHING**: `forms.grow_tree(points, forces, settings, supports=support positions)` (every
  load joins the nearest support), with `constrain=` mapping inner nodes to the nearest domain
  voxel centre when outside. Field clipped to the block: `field = min(field, domain_field)` where
  domain_field = smoothed domain indicator resampled to the tube grid (sculpt.resample_to), so
  nothing leaves the block. Then as SUPPORT/BRANCHING.
* **RIBS**: `rib_source == EVOLVED` → ORGANIC first, ribs on its surface; `SURFACE` → full
  analysis of the block (fem.analyze), ribs on the block's own surface (surface field = the
  domain indicator filtered with the density filter, iso 0.5).

### Ribs (both modes)
Surface field `f` (element-centred) + iso: the evolved physical density with `iso_level`, or
the filtered domain indicator at 0.5. Stress tensor field: the evolved state's stress
(IterationState.stress) or the analysis stress. Trace with the NEW surface mode of
`core/streamlines.py` (being added by another engineer, API below), families from
`line_families`, `separation = rib_spacing`, seeds stress-weighted on the surface + at loads and
footings, branching off (`branch_depth = 0`). Tubes: radius from |σ| along the line,
sqrt-mapped into `[rib_min_thickness × rib_thickness, rib_thickness]` voxels, tapering to 40% over
the last 10% of open ends. Field `sculpt.tubes_field(voxel = h / sculpt_detail, blend =
sculpt_blend)`; if `rib_skin > 0` union with a shell: `skin = 1 − |f_resampled − iso| /
(rib_skin_scaled)` clipped ≥0 mapped so the shell has the requested thickness in voxels (keep it
simple and documented). Also always add the load pads and footings as small spheres so the
lattice actually reaches the loads/supports. Mesh as above. Stress colors from the line values.

Streamlines surface API (expected; code against it, adapt if the other engineer reports
deviations — check the file):
```python
@dataclass
class SurfaceConstraint:
    field: np.ndarray        # (nx,ny,nz) element-centred scalar (same grid as the StressField)
    iso: float
    tolerance: float = 0.75  # voxels: a point further than this from the surface stops the line
TraceSettings(..., surface: Optional[SurfaceConstraint] = None)
```
With `surface` set, lines are traced along the principal directions of the stress tensor
projected onto the tangent plane (the eigenvector closest to the surface normal is discarded;
TENSION = larger remaining eigenvalue, COMPRESSION = smaller), every step is projected back
onto the iso-surface (Newton on the trilinear field), seeds are placed on the surface.

## Everything else
* Keep all existing operators working (analyze, evolve, trace_lines, sculpt, relief, variants…);
  they use `settings.domain` as before (the Stress Analysis panel shows it). Variants must also
  capture SS_Structure (kind = style).
* `clear_results` also removes SS_Structure.
* Threading rules of SPEC §7.4; never touch bpy in workers; robust error messages; undo.
* Performance targets (4 cores): FLOATING_CUBE BRANCHING < 10 s end to end at default settings;
  CANOPY ORGANIC at resolution 48 with 40 iterations < 3 min (live preview every iteration);
  RIBS < 4 min.
* Tests (`tests/blender/`): every demo preset generates its default style successfully (use
  small resolution / few iterations in tests via settings); add_load_at_selection on a cube in
  edit mode (select 4 vertices → 4 parented LOAD empties, totals for TOTAL mode, face centres in
  face mode, clustering above max_points); generate for all 6 mode×style combos on a tiny setup;
  verify; remove_role; variants with SS_Structure. Keep the suite fast.
* Render check: `/tmp/claude-0/-home-user-paramops/64513836-2a10-5b5e-b175-3cbab71e423e/scratchpad/render_util.py`
  has helpers to render a preview with Cycles CPU in the bpy module (see `proto_tree.py` next
  to it for a usage example). Render each demo's result at least once, LOOK at the images
  (Read tool shows PNGs) and fix anything that does not look clean/architectural (floating
  bits, lumps, broken members, hairballs). Save the final renders in
  `/tmp/claude-0/-home-user-paramops/64513836-2a10-5b5e-b175-3cbab71e423e/scratchpad/renders_v2/`.
