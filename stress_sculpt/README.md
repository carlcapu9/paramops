# Stress Sculpt

**Put weight on a mesh, get a structure that carries it.**

Stress Sculpt is a Blender add-on that turns loads into architecture. Select points or
faces of a mesh, tell it how heavy they are, and it grows a structure that carries that
weight to the ground. You can get organic generative-design shapes, branching tree
columns, or rib lattices that follow the stress lines. The solver is real: a 3D
finite-element model of the forces, not a look-alike effect.

* **Two modes**
  * **Support Object**: the mesh is carried (a floating cube, a roof, a pavilion). The
    structure grows from the ground up to the weighted points.
  * **Carve Inside**: the mesh is the block of material. You put weights and supports on it,
    and the structure forms inside it (a bridge out of a beam, a sculpture out of a cube).
* **Three styles**
  * **Organic**: topology optimization (SIMP). Material goes only where the forces flow,
    and the shape evolves live in the viewport.
  * **Branching**: form-found tree columns in the spirit of Frei Otto and Gaudí's inverted
    hanging models. Every member works in pure compression, and its thickness follows the
    force it carries.
  * **Ribs**: a lattice of ribs that follows the principal stress lines over the optimized
    shape or the object's surface, in the spirit of Pier Luigi Nervi's isostatic ribs.
* **Compare**: save results as variants and lay them out side by side with their metrics.
* **Analyze**: von Mises / principal stress heatmaps, and a check of any generated structure
  under the same loads.

## Install

Blender 4.2 or newer.

1. Download `stress_sculpt-<version>.zip` (do not unzip it).
2. In Blender, go to *Edit › Preferences › Get Extensions*, open the ▾ menu at the top
   right, choose *Install from Disk…* and pick the zip.
3. The panel is in the 3D Viewport sidebar (`N`), tab **Stress Sculpt**.

## Quick start

1. Add a cube and move it up (for example to Z = 3 m).
2. In the **Structure** panel, click the eyedropper to use the cube as the *Object*.
   Choose **Support Object** and **Branching**.
3. Press `Tab` to enter Edit Mode and select the vertices (or faces) where the weight rests.
   Click **Put Weight on Selection** (it is also in the right-click menu). An arrow appears
   on every point. In the redo panel you can set the force per point, or a total force to
   share between the points.
4. Leave Edit Mode and press **Generate Structure**.

To try it without setting anything up, open the presets menu (the icon in the panel header)
and pick an example: *Floating Cube*, *Canopy*, *Pavilion*, *Bridge* or *Carved Cube*.

## The Structure panel

| Setting | Meaning |
|---|---|
| Object | The mesh you work with. |
| Mode | *Support Object* (grows from the ground) or *Carve Inside* (forms inside the mesh). |
| Style | *Organic*, *Branching* or *Ribs*. |
| Weights | Arrows (Load empties) parented to the mesh. Edit their force in the *Active Object* sub-panel. |
| Ground / Footings | In Support mode the structure stands on the ground height. It lands on *Footings*: Support empties you place, or automatic ones (0 = choose from the footprint). *Whole Ground* lets it touch down anywhere. |
| Supports | In Carve mode, anchor the block where it rests with **Support at Selection**. |

### Style settings

* **Organic**: *Material* (fraction of the free space to fill), *Resolution* (voxels along
  the longest side), *Smoothness* (minimum member thickness), *Iterations*, *Symmetry*,
  *Contact Size* (solid pads kept at every load and footing).
* **Branching**: *Branches per Split*, *Trunk Height* (where the trunk first splits),
  *Split Rhythm*, *Flow* (straight ↔ curved members), *Thickness* of the most loaded member
  and *Thinnest* member, footing and capital flares, *Junction Blend*, *Detail*.
* **Ribs**: the organic settings, plus rib *Spacing*, *Thickness*, *Thinnest*, families
  (tension, compression or both) and an optional *Skin* under the ribs.

## Other panels

* **Stress Analysis**: solves a mesh as a solid block and colors it by stress. *Check
  Generated Structure* runs the same analysis on the last generated structure.
* **Variants**: `+` saves the current result with its metrics (material, compliance, max
  stress). *Compare Side by Side* lines the variants up with labels, and *Restore Settings*
  brings back the settings of a variant.
* **More Tools**: zones (Keep Solid, Keep Void, Thicken, Hollow, Branch) that bias the
  optimization, raw stress lines, fusing lines into a sculpture, and a stress relief that
  thickens a surface where it is most stressed.
* **Material & Solver**: material presets (PLA, steel, aluminium, concrete, wood, bone),
  solver tolerance, and clearing all results.

## How it works

* **Finite elements**: the space is split into voxels (8-node hexahedra) and linear
  elasticity is solved matrix-free with conjugate gradients and a geometric multigrid
  preconditioner, in pure numpy. A solve with 50k elements takes well under a second.
* **Organic**: SIMP topology optimization (density filter, optimality-criteria update,
  penalty continuation) minimizes compliance for the given amount of material.
* **Branching**: a funicular tree. The load points are grouped recursively towards the
  footings. With the heights fixed by the design settings, the force-density method gives
  the horizontal positions for which every node is in exact equilibrium. Each member's
  radius is ∝ √force (uniform stress).
* **Ribs**: principal stress trajectories are integrated on the surface of the structure,
  using the stress tensor projected onto the tangent plane. They become tubes whose radius
  follows the stress.
* **Meshing**: every style ends as an implicit field (densities or tubes with smooth
  unions), meshed with surface nets and smoothed with a Taubin filter. The result is one
  watertight mesh colored by stress or force.

Long computations run in a background thread with a progress bar. `Esc` cancels; for
*Organic* it keeps the current shape.

## Tips

* Start at resolution 32–48 and raise it for the final result.
* Loads are applied along the arrow. Rotate an arrow to push sideways; the Branching style
  grows against the resultant load.
* Units are SI: Blender meters, newtons, pascals.

## Development

```bash
python -m pytest tests -q
```

`stress_sculpt/core` is plain numpy and needs no Blender; the tests in `tests/blender` need
the `bpy` module (`pip install bpy`).
