"""Operators of Stress Sculpt (``bpy.ops.stress_sculpt.*``).

* Structure (main flow): ``set_target``, ``add_load_at_selection``,
  ``add_support_at_selection``, ``remove_role``, ``generate`` (threaded, live for
  the organic style), ``verify``, ``demo_scene`` (architectural examples).
* Setup: ``set_domain``, ``add_support``, ``add_load``, ``add_zone``, ``set_role``,
  ``select_role``.
* Computation (threaded with live progress, Esc cancels): ``analyze``,
  ``evolve``, ``trace_lines``, ``sculpt``, ``relief``.
* Housekeeping: ``cancel``, ``clear_results`` and the variants
  (``variant_save`` / ``_remove`` / ``_restore`` / ``_compare`` /
  ``_clear_compare`` / ``_move``).

Long operators follow ``jobs.ThreadedOperatorMixin``: ``prepare`` reads the
scene on the main thread, ``compute`` runs pure numpy in a worker thread and
``finish`` builds the Blender objects. Run from scripts (EXEC_DEFAULT) or in
background mode they execute synchronously. Results are looked up by name in
``finish`` so a scene edited (or deleted) during a job cannot be corrupted.
"""

import copy
import dataclasses
import json
import math

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator
from mathutils import Vector

from . import build, jobs, scene_io
from .core.grid import Grid, Problem, check_stop
from .jobs import RUNTIME, ThreadedOperatorMixin
from .props import ROLE_ITEMS, ZONE_ROLES
from .scene_io import core_module

KIND_ANALYSIS = "ANALYSIS"
KIND_EVOLVE = "EVOLVE"
KIND_LINES = "LINES"
KIND_SCULPT = "SCULPT"
KIND_RELIEF = "RELIEF"
KIND_LABELS = {
    KIND_ANALYSIS: "Analysis",
    KIND_EVOLVE: "Evolved",
    KIND_LINES: "Lines",
    KIND_SCULPT: "Sculpture",
    KIND_RELIEF: "Relief",
}

# Results that enclose the others (hidden when an inner result is shown).
ENVELOPES = (build.OBJ_STRESS, build.OBJ_RELIEF)
LINE_OBJECTS = (build.OBJ_LINES_TENSION, build.OBJ_LINES_COMPRESSION)

ZONE_ITEMS = [item for item in ROLE_ITEMS if item[0] in ZONE_ROLES]
ZONE_COLORS = {
    "KEEP_SOLID": (0.2, 0.85, 0.3, 1.0),
    "KEEP_VOID": (0.55, 0.55, 0.55, 1.0),
    "THICKEN": (1.0, 0.6, 0.1, 1.0),
    "HOLLOW": (0.6, 0.2, 0.9, 1.0),
    "BRANCH": (0.1, 0.8, 0.8, 1.0),
}
DEMO_TAG = "stress_sculpt_demo"  # ID property marking objects created by demo_scene
SCULPT_MAX_CELLS = 12_000_000  # bound on the sculpture grid (~100 MB per float64 field)
MESH_MAX_FACES = 400_000  # bound on refined copies of the domain mesh

# Scene settings that are not part of a variant's saved settings.
_SETTINGS_SKIP = {"rna_type", "name", "variants", "active_variant_index", "show_advanced"}


# =========================================================================== helpers
def _settings(context):
    return context.scene.stress_sculpt


def _job_scene(name):
    """The scene a job was started in (looked up by name: it may have been deleted)."""
    scene = bpy.data.scenes.get(name)
    if scene is None:
        raise ValueError(f"The scene '{name}' was deleted while computing")
    return scene


def _view_layer(context, scene):
    if context.scene == scene and getattr(context, "view_layer", None) is not None:
        return context.view_layer
    return scene.view_layers[0] if len(scene.view_layers) else None


def _depsgraph(context, scene):
    if context.scene == scene:
        return context.evaluated_depsgraph_get()
    layer = _view_layer(context, scene)
    return layer.depsgraph if layer is not None else context.evaluated_depsgraph_get()


def _modules(*names) -> dict:
    """Core modules resolved on the main thread (the worker never imports)."""
    return {name: core_module(name) for name in names}


def _sub_progress(progress, start, end):
    """``progress`` callback mapping [0, 1] into [start, end]."""

    def report(fraction, message=None):
        try:
            fraction = min(max(float(fraction), 0.0), 1.0)
        except (TypeError, ValueError):
            fraction = 0.0
        progress(start + (end - start) * fraction, message)

    return report


def _report_warnings(operator, warnings, limit=3) -> None:
    unique = list(dict.fromkeys(w for w in warnings if w))
    for warning in unique[:limit]:
        operator.report({"WARNING"}, warning)
    if len(unique) > limit:
        for warning in unique[limit:]:
            print(f"Stress Sculpt warning: {warning}")
        operator.report(
            {"WARNING"}, f"... and {len(unique) - limit} more warning(s), see the console"
        )


def _format_stress(pascal) -> str:
    value = abs(pascal)
    for scale, unit in ((1e9, "GPa"), (1e6, "MPa"), (1e3, "kPa")):
        if value >= scale:
            return f"{pascal / scale:.3g} {unit}"
    return f"{pascal:.3g} Pa"


def _format_force(newton) -> str:
    value = abs(newton)
    if value >= 1e6:
        return f"{newton / 1e6:.3g} MN"
    if value >= 1e3:
        return f"{newton / 1e3:.3g} kN"
    return f"{newton:.3g} N"


def _format_length(meters) -> str:
    value = abs(meters)
    if value == 0.0:
        return "0 mm"
    if value < 1e-3:
        return f"{meters * 1e6:.3g} um"
    if value < 1.0:
        return f"{meters * 1e3:.3g} mm"
    return f"{meters:.3g} m"


def _show_domain_as_wire(scene) -> None:
    settings = scene.stress_sculpt
    domain = settings.domain
    if settings.domain_as_wire and domain is not None:
        domain.display_type = "WIRE"


def _set_last(
    scene, kind, max_stress=0.0, max_displacement=0.0, compliance=0.0, volume=0.0, iterations=0
) -> None:
    settings = scene.stress_sculpt
    settings.last_kind = kind
    settings.last_max_stress = float(max_stress)
    settings.last_max_displacement = float(max_displacement)
    settings.last_compliance = float(compliance)
    settings.last_volume = float(max(volume, 0.0))  # may exceed 1 (relief, sculpture)
    settings.last_iterations = int(iterations)


def _ensure_object_mode(context) -> None:
    """Leave Edit / Sculpt / Paint modes: result meshes are replaced as whole datablocks."""
    if getattr(context, "mode", "OBJECT") == "OBJECT":
        return
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError as exc:
        raise ValueError("Switch to Object Mode first") from exc


def _deselect_all(context) -> None:
    view_layer = context.view_layer
    for obj in scene_io.view_layer_objects(context):
        if obj.select_get(view_layer=view_layer):
            obj.select_set(False, view_layer=view_layer)


def _select_only(context, obj) -> None:
    view_layer = context.view_layer
    _deselect_all(context)
    try:
        obj.select_set(True, view_layer=view_layer)
        view_layer.objects.active = obj
    except RuntimeError:  # the setup collection is excluded from the view layer
        pass


def _world_bounds(context, obj):
    """World bounding box of an object's evaluated geometry (``None`` if it has none)."""
    verts, _tris = scene_io.evaluated_world_mesh(obj, context.evaluated_depsgraph_get())
    if verts.shape[0]:
        return verts.min(axis=0), verts.max(axis=0)
    return build.data_bounds(obj)


def _domain_diagonal(context):
    domain = _settings(context).domain
    if domain is None:
        return None
    bounds = _world_bounds(context, domain)
    if bounds is None:
        return None
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    return diagonal if diagonal > 0.0 else None


def new_role_empty(
    scene,
    name,
    role,
    display_type,
    size,
    location,
    rotation=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    color=None,
):
    """Empty with a Stress Sculpt role, linked to the "StressSculpt Setup" collection."""
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display_type
    obj.empty_display_size = float(size)
    obj.location = tuple(float(c) for c in location)
    obj.rotation_euler = tuple(float(c) for c in rotation)
    obj.scale = tuple(float(c) for c in scale)
    obj.show_in_front = True
    if color is not None:
        obj.color = color
    obj.stress_sculpt.role = role
    build.link_to_collection(obj, build.ensure_collection(scene, build.SETUP_COLLECTION))
    return obj


# ------------------------------------------------------------ numeric (worker-safe)
def measure_field(problem, result, measure, fem):
    """Scalar field (nx, ny, nz) of ``measure`` and whether its values are signed."""
    if measure == "DISPLACEMENT":
        return np.asarray(result.displacement, dtype=np.float64), False
    if measure in {"PRINCIPAL_MAX", "PRINCIPAL_MIN"}:
        field = np.zeros(problem.grid.shape, dtype=np.float64)
        mask = problem.domain
        if mask.any():
            values, _vectors = fem.principal(np.asarray(result.stress)[mask])
            field[mask] = values[:, 2] if measure == "PRINCIPAL_MAX" else values[:, 0]
        return field, True
    return np.asarray(result.von_mises, dtype=np.float64), False


def color_range(values, signed, colormap, clip) -> tuple:
    """``(vmin, vmax)`` for the colors: percentile clipping, diverging around 0 for COOLWARM."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    clip = min(max(float(clip), 50.0), 100.0)
    if not signed:
        lo, hi = 0.0, float(np.percentile(values, clip))
    elif colormap == "COOLWARM":
        hi = float(np.percentile(np.abs(values), clip))
        lo = -hi
    else:
        lo, hi = float(np.percentile(values, 100.0 - clip)), float(np.percentile(values, clip))
    if not hi > lo:
        hi = lo + max(abs(lo), 1e-12)
    return lo, hi


def sample_field(field, mask, grid, points, mesher, iterations=3):
    """Field sampled at world points: extrapolated past ``mask`` then trilinear."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(np.asarray(points).reshape(-1, 3).shape[0], dtype=np.float64)
    extended = mesher.extrapolate(np.where(mask, field, 0.0), mask, iterations=iterations)
    return np.asarray(
        mesher.sample_trilinear(extended, grid.origin, grid.h, points), dtype=np.float64
    )


def line_radii(lines, h, radius_min, radius_max, taper, magnitude_ref=None) -> tuple:
    """World tube radius at every point of every polyline.

    ``sqrt(|magnitude| / ref)`` (clipped to [0, 1]) maps into
    ``[radius_min, radius_max]`` voxels, times ``taper ** generation``. ``ref`` is
    the 99th percentile of all magnitudes unless given. Returns ``(radii, ref)``.
    """
    lo, hi = sorted((float(radius_min), float(radius_max)))
    if magnitude_ref is None or not magnitude_ref > 0.0:
        magnitudes = [np.abs(np.asarray(line.magnitude, dtype=np.float64)) for line in lines]
        magnitudes = np.concatenate(magnitudes) if magnitudes else np.zeros(0)
        magnitudes = magnitudes[np.isfinite(magnitudes)]
        magnitude_ref = float(np.percentile(magnitudes, 99.0)) if magnitudes.size else 1.0
        if not magnitude_ref > 0.0:
            magnitude_ref = (
                float(magnitudes.max()) if magnitudes.size and magnitudes.max() > 0 else 1.0
            )
    radii = []
    for line in lines:
        t = np.sqrt(
            np.clip(np.abs(np.asarray(line.magnitude, dtype=np.float64)) / magnitude_ref, 0.0, 1.0)
        )
        radius = (lo + (hi - lo) * t) * (float(taper) ** int(line.generation)) * h
        radii.append(np.maximum(radius, 1e-6 * h))
    return radii, magnitude_ref


def tubes_volume(lines, radii) -> float:
    """Approximate volume of the tubes (sum of pi r^2 ds)."""
    total = 0.0
    for line, radius in zip(lines, radii):
        points = np.asarray(line.points, dtype=np.float64)
        if points.shape[0] < 2:
            continue
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        mean_r = 0.5 * (radius[1:] + radius[:-1])
        total += float(np.sum(math.pi * mean_r * mean_r * lengths))
    return total


def mesh_volume(verts, faces) -> float:
    """Signed volume of a closed polygon mesh (fan triangulation, outward normals > 0)."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return 0.0
    total = 0.0
    for corner in range(1, faces.shape[1] - 1):
        a, b, c = verts[faces[:, 0]], verts[faces[:, corner]], verts[faces[:, corner + 1]]
        total += float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())
    return total / 6.0


def polygon_volume(verts, loop_verts, loop_starts) -> float:
    """Signed volume of a closed mesh given as flat polygon arrays."""
    verts = np.asarray(verts, dtype=np.float64)
    loop_verts = np.asarray(loop_verts, dtype=np.int64)
    loop_starts = np.asarray(loop_starts, dtype=np.int64)
    if loop_starts.size == 0:
        return 0.0
    totals = np.diff(np.append(loop_starts, loop_verts.size))
    first = np.repeat(loop_starts, totals - 2)
    offsets = np.arange(first.size) - np.repeat(np.cumsum(totals - 2) - (totals - 2), totals - 2)
    a = verts[loop_verts[first]]
    b = verts[loop_verts[first + 1 + offsets]]
    c = verts[loop_verts[first + 2 + offsets]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _copy_state(state):
    """Copy of an IterationState with private arrays (the optimizer may reuse its buffers)."""
    arrays = {}
    for name in ("density", "stress", "von_mises"):
        value = getattr(state, name, None)
        if isinstance(value, np.ndarray):
            arrays[name] = value.copy()
    if dataclasses.is_dataclass(state):
        return dataclasses.replace(state, **arrays)
    clone = copy.copy(state)
    for name, value in arrays.items():
        setattr(clone, name, value)
    return clone


def _drop_small_parts(mesher, verts, faces, min_faces):
    """``mesher.remove_small_components`` (first two items of its result: verts, quads)."""
    result = mesher.remove_small_components(verts, faces, int(min_faces))
    verts, faces = result[0], result[1]
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64).reshape(-1, 4)


def evolved_geometry(state, problem, params, mesher, smooth_iterations, clean=True):
    """Mesh of an evolved density: surface nets at the surface level, Taubin, von Mises colors.

    Returns a dict ``verts``, ``faces`` (quads), ``colors``, ``values`` (von Mises
    at the vertices); ``verts`` is empty when no element exceeds the level.
    """
    grid = problem.grid
    density = np.where(problem.domain, np.asarray(state.density, dtype=np.float64), 0.0)
    iso = float(params["iso"])
    verts, faces = mesher.surface_nets(density, iso, grid.origin, grid.h)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    if faces.shape[0] and clean:
        verts, faces = _drop_small_parts(mesher, verts, faces, max(12, 0.005 * faces.shape[0]))
    if faces.shape[0] == 0:
        return {
            "verts": np.zeros((0, 3)),
            "faces": np.zeros((0, 4), dtype=np.int64),
            "colors": np.zeros((0, 4)),
            "values": np.zeros(0),
        }
    if smooth_iterations > 0:
        verts = mesher.taubin_smooth(verts, faces, iterations=int(smooth_iterations))
    von_mises = np.asarray(state.von_mises, dtype=np.float64)
    solid = (density >= iso) & problem.domain
    if not solid.any():
        solid = problem.domain
    values = sample_field(von_mises, solid, grid, verts, mesher)
    lo, hi = color_range(von_mises[solid], False, params["colormap"], params["clip"])
    colors = mesher.colormap(values, lo, hi, name=params["colormap"])
    return {"verts": np.asarray(verts), "faces": faces, "colors": colors, "values": values}


def _mesh_from_geometry(name, geometry, smooth=True):
    mesh = build.mesh_from_faces(name, geometry["verts"], geometry["faces"], smooth=smooth)
    try:
        if len(mesh.vertices):
            build.set_mesh_stress(mesh, geometry["colors"], geometry["values"])
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    return mesh


# --------------------------------------------------------------- shared requests
def _analysis_request(context, settings) -> dict:
    """The cached analysis if it still matches the setup, else a problem to analyze."""
    scene = context.scene
    signature = scene_io.setup_signature(context)
    cached = jobs.get_cache(scene).analysis
    if cached is not None and cached.get("signature") == signature:
        return {
            "problem": cached["problem"],
            "result": cached["result"],
            "info": cached["info"],
            "signature": signature,
            "computed": False,
        }
    problem, info = scene_io.gather_problem(context)
    return {
        "problem": problem,
        "result": None,
        "info": info,
        "signature": signature,
        "computed": False,
        "solver": settings.solver,
        "tol": float(settings.solver_tolerance),
        "maxiter": int(settings.max_solver_iterations),
        "fem": core_module("fem"),
    }


def _run_analysis(request, progress, should_stop):
    """Worker side of ``_analysis_request``: solve if no valid result was cached."""
    if request["result"] is None:
        request["result"] = request["fem"].analyze(
            request["problem"],
            solver=request["solver"],
            tol=request["tol"],
            maxiter=request["maxiter"],
            progress=progress,
            should_stop=should_stop,
        )
        request["computed"] = True
    return request["result"]


def _store_analysis(scene, request) -> None:
    if request is not None and request.get("computed"):
        jobs.get_cache(scene).analysis = {
            "problem": request["problem"],
            "result": request["result"],
            "signature": request["signature"],
            "grid": request["problem"].grid,
            "info": request["info"],
        }


def _trace_request(context, settings) -> dict:
    """Everything the line tracer needs (main thread)."""
    streamlines = core_module("streamlines")
    cache = jobs.get_cache(context.scene)
    request = {"source": settings.line_source, "streamlines": streamlines, "analysis": None}
    if settings.line_source == "EVOLVED":
        evolution = cache.evolution
        if evolution is None:
            raise ValueError(
                "No evolved shape yet: run Evolve first, or set 'Stress From' to 'Full Domain'"
            )
        problem, state = evolution["problem"], evolution["state"]
        solid = (np.asarray(state.density) > 0.5) & problem.domain
        if not solid.any():
            raise ValueError("The evolved shape is empty: evolve again with more material")
        request.update(
            problem=problem,
            stress=np.asarray(state.stress),
            solid=solid,
            info=evolution["info"],
            signature=evolution["signature"],
            metrics={
                "compliance": float(state.compliance),
                "max_stress": float(np.max(state.von_mises)),
                "max_displacement": float(evolution.get("max_displacement") or 0.0),
            },
        )
    else:
        analysis = _analysis_request(context, settings)
        request.update(
            problem=analysis["problem"],
            info=analysis["info"],
            signature=analysis["signature"],
            analysis=analysis,
        )
    families = {
        "BOTH": ("TENSION", "COMPRESSION"),
        "TENSION": ("TENSION",),
        "COMPRESSION": ("COMPRESSION",),
    }[settings.line_families]
    seeds = request["info"]["seed_points"] if settings.seed_from_loads else None
    if seeds is not None and len(seeds) == 0:
        seeds = None
    branch = request["problem"].branch_or_zero()
    request["trace_settings"] = streamlines.TraceSettings(
        families=families,
        n_seeds=int(settings.n_seeds),
        seed_points=None if seeds is None else np.asarray(seeds, dtype=np.float64),
        step=float(settings.line_step),
        min_stress=float(settings.line_min_stress),
        separation=float(settings.line_separation),
        branch_depth=int(settings.branch_depth),
        branch_spacing=float(settings.branch_spacing),
        branch_ratio=float(settings.branch_ratio),
        branch_weight=branch if branch.any() else None,
        rng_seed=int(settings.random_seed),
    )
    request["radius"] = (
        float(settings.radius_min),
        float(settings.radius_max),
        float(settings.branch_taper),
    )
    return request


def _run_trace(request, progress, should_stop) -> list:
    """Worker side of ``_trace_request`` (analyzes first when needed)."""
    trace_progress = progress
    analysis = request["analysis"]
    if analysis is not None:
        if analysis["result"] is None:
            _run_analysis(analysis, _sub_progress(progress, 0.0, 0.6), should_stop)
            trace_progress = _sub_progress(progress, 0.6, 1.0)
        result = analysis["result"]
        request["stress"] = np.asarray(result.stress)
        request["solid"] = request["problem"].domain
        request["metrics"] = {
            "compliance": float(result.compliance),
            "max_stress": float(np.max(result.von_mises)),
            "max_displacement": float(result.max_displacement),
        }
    streamlines = request["streamlines"]
    field = streamlines.StressField(
        grid=request["problem"].grid, stress=request["stress"], solid=request["solid"]
    )
    lines = streamlines.trace(
        field, request["trace_settings"], progress=trace_progress, should_stop=should_stop
    )
    return list(lines)


def _lines_entry(request, lines, magnitude_ref) -> dict:
    """Cache entry of traced lines (with the metrics of the stress they come from)."""
    problem = request["problem"]
    return {
        "lines": lines,
        "grid": problem.grid,
        "h": problem.grid.h,
        "source": request["source"],
        "signature": request["signature"],
        "magnitude_ref": magnitude_ref,
        "radius": request["radius"],
        "problem": problem,
        "metrics": dict(request.get("metrics") or {}),
    }


def _fresh_warnings(analysis_request) -> list:
    """Setup warnings of an analysis solved in this job (cached ones were reported already)."""
    if analysis_request is None or not analysis_request.get("computed"):
        return []
    return list(analysis_request["info"]["warnings"])


def _build_lines(scene, lines, h, radius, view_layer, magnitude_ref=None) -> tuple:
    """Curve objects of both families; stale family objects are removed."""
    radii, magnitude_ref = line_radii(lines, h, *radius, magnitude_ref=magnitude_ref)
    bevel = max(radius[0], radius[1], 1e-3) * h
    counts = {}
    for family, name, material, color in (
        ("TENSION", build.OBJ_LINES_TENSION, build.tension_material, build.TENSION_COLOR),
        (
            "COMPRESSION",
            build.OBJ_LINES_COMPRESSION,
            build.compression_material,
            build.COMPRESSION_COLOR,
        ),
    ):
        members = [
            i for i, line in enumerate(lines) if line.family == family and len(line.points) >= 2
        ]
        counts[family] = len(members)
        if not members:
            build.remove_result(scene, name)
            continue
        with build.NewData() as new:
            curve = new.add(
                build.curve_from_polylines(
                    name,
                    [lines[i].points for i in members],
                    [radii[i] / bevel for i in members],
                    bevel_depth=bevel,
                    bevel_resolution=2,
                )
            )
            obj = build.assign_result(scene, name, curve, material(), view_layer)
        obj.color = color  # solid-mode "Attribute" coloring falls back to the object color
    return counts, radii, magnitude_ref


# ========================================================================= setup ops
class STRESS_SCULPT_OT_set_domain(Operator):
    bl_idname = "stress_sculpt.set_domain"
    bl_label = "Use Active as Domain"
    bl_description = "Use the active mesh as the design domain (the space where material may exist)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and not RUNTIME.busy()

    def execute(self, context):
        settings = _settings(context)
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a closed mesh to use as the domain")
            return {"CANCELLED"}
        previous = settings.domain
        if (
            previous is not None
            and previous != obj
            and settings.domain_as_wire
            and previous.display_type == "WIRE"
        ):
            previous.display_type = "TEXTURED"
        if obj.stress_sculpt.role != "NONE":
            self.report(
                {"WARNING"},
                f"'{obj.name}' was a {obj.stress_sculpt.role.title()}: its role was cleared",
            )
            obj.stress_sculpt.role = "NONE"
        settings.domain = obj
        self.report({"INFO"}, f"Domain: {obj.name}")
        return {"FINISHED"}


class STRESS_SCULPT_OT_add_support(Operator):
    bl_idname = "stress_sculpt.add_support"
    bl_label = "Add Support"
    bl_description = "Add a support at the 3D cursor: the domain cannot move inside it"
    bl_options = {"REGISTER", "UNDO"}

    shape: EnumProperty(
        name="Shape",
        items=[
            ("SPHERE", "Sphere", "Nodes inside a sphere are fixed", "SPHERE", 0),
            ("CUBE", "Cube", "Nodes inside a box are fixed (scale it to cover a face)", "CUBE", 1),
        ],
        default="SPHERE",
    )

    def execute(self, context):
        diagonal = _domain_diagonal(context)
        size = 0.05 * diagonal if diagonal else 0.25
        obj = new_role_empty(
            context.scene,
            "SS_Support",
            "SUPPORT",
            self.shape,
            size,
            context.scene.cursor.location,
            color=build.SUPPORT_COLOR,
        )
        _select_only(context, obj)
        return {"FINISHED"}


class STRESS_SCULPT_OT_add_load(Operator):
    bl_idname = "stress_sculpt.add_load"
    bl_label = "Add Load"
    bl_description = "Add a load at the 3D cursor: a force along the arrow (rotate it to aim)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        diagonal = _domain_diagonal(context)
        size = 0.15 * diagonal if diagonal else 0.75
        obj = new_role_empty(
            context.scene,
            "SS_Load",
            "LOAD",
            "SINGLE_ARROW",
            size,
            context.scene.cursor.location,
            rotation=(math.pi, 0.0, 0.0),
            color=build.LOAD_COLOR,
        )
        _select_only(context, obj)
        return {"FINISHED"}


def _zone_object_name(role) -> str:
    return "SS_" + "".join(part.title() for part in role.split("_"))


class STRESS_SCULPT_OT_add_zone(Operator):
    bl_idname = "stress_sculpt.add_zone"
    bl_label = "Add Zone"
    bl_description = (
        "Add a zone at the 3D cursor that keeps, removes, thickens, hollows or branches material"
    )
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(name="Role", items=ZONE_ITEMS, default="THICKEN")

    def execute(self, context):
        diagonal = _domain_diagonal(context)
        size = 0.1 * diagonal if diagonal else 0.5
        obj = new_role_empty(
            context.scene,
            _zone_object_name(self.role),
            self.role,
            "SPHERE",
            size,
            context.scene.cursor.location,
            color=ZONE_COLORS.get(self.role),
        )
        _select_only(context, obj)
        return {"FINISHED"}


class STRESS_SCULPT_OT_set_role(Operator):
    bl_idname = "stress_sculpt.set_role"
    bl_label = "Set Role"
    bl_description = "Give the selected objects a Stress Sculpt role (the domain is never changed)"
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(name="Role", items=ROLE_ITEMS, default="SUPPORT")

    def execute(self, context):
        domain = _settings(context).domain
        selected = [obj for obj in scene_io.view_layer_objects(context) if obj.select_get()]
        if not selected:
            self.report({"ERROR"}, "Select the objects that should get the role")
            return {"CANCELLED"}
        changed = 0
        for obj in selected:
            if obj == domain:
                self.report({"WARNING"}, f"'{obj.name}' is the domain: its role was not changed")
                continue
            obj.stress_sculpt.role = self.role
            if obj.type == "EMPTY":
                if self.role == "LOAD" and obj.empty_display_type != "SINGLE_ARROW":
                    obj.empty_display_type = "SINGLE_ARROW"
                elif self.role not in {"LOAD", "NONE"} and obj.empty_display_type not in {
                    "CUBE",
                    "SPHERE",
                }:
                    obj.empty_display_type = "SPHERE"
            changed += 1
        if not changed:
            return {"CANCELLED"}
        self.report({"INFO"}, f"{changed} object(s) set to {self.role.replace('_', ' ').title()}")
        return {"FINISHED"}


class STRESS_SCULPT_OT_select_role(Operator):
    bl_idname = "stress_sculpt.select_role"
    bl_label = "Select by Role"
    bl_description = "Select every object with the given role"
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(name="Role", items=ROLE_ITEMS, default="SUPPORT")
    extend: BoolProperty(name="Extend", default=False, description="Keep the current selection")

    def execute(self, context):
        view_layer = context.view_layer
        if not self.extend:
            _deselect_all(context)
        found = None
        count = 0
        for obj in scene_io.view_layer_objects(context):
            if obj.stress_sculpt.role == self.role and obj.visible_get(view_layer=view_layer):
                obj.select_set(True, view_layer=view_layer)
                found = obj
                count += 1
        if found is not None and (
            view_layer.objects.active is None or not view_layer.objects.active.select_get()
        ):
            view_layer.objects.active = found
        self.report({"INFO"}, f"{count} object(s) selected")
        return {"FINISHED"}


# ----------------------------------------------------------- structure setup
def _is_role_object(obj) -> bool:
    settings = getattr(obj, "stress_sculpt", None)
    return settings is not None and settings.role != "NONE"


class STRESS_SCULPT_OT_set_target(Operator):
    bl_idname = "stress_sculpt.set_target"
    bl_label = "Use Active Object"
    bl_description = (
        "Work with the active mesh: the object to hold up (Support Object) or the block to carve "
        "the structure from (Carve Inside)"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == "MESH"
            and not _is_role_object(obj)
            and not RUNTIME.busy()
        )

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select the mesh to work with")
            return {"CANCELLED"}
        if _is_role_object(obj):
            self.report(
                {"ERROR"},
                f"'{obj.name}' is a {obj.stress_sculpt.role.replace('_', ' ').title()} helper: "
                "set its role to None first",
            )
            return {"CANCELLED"}
        _settings(context).target = obj
        self.report({"INFO"}, f"Object: {obj.name}")
        return {"FINISHED"}


def _selection_objects(context) -> list:
    """Meshes whose selection the "at selection" operators read (active first)."""
    active = context.active_object
    if context.mode == "EDIT_MESH":
        objects = [
            obj
            for obj in (getattr(context, "objects_in_mode_unique_data", None) or [])
            if obj is not None and obj.type == "MESH"
        ]
        edit_object = getattr(context, "edit_object", None)
        if not objects and edit_object is not None and edit_object.type == "MESH":
            objects = [edit_object]
        objects.sort(key=lambda o: o != active)
        return objects
    if active is not None and active.type == "MESH":
        return [active]
    return []


def _world_linear(obj):
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    linear = matrix[:3, :3]
    try:
        normal_matrix = np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        normal_matrix = np.eye(3)
    return matrix, linear, normal_matrix


def _unit_rows(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 1e-30)


def selection_points(obj, face_mode) -> dict:
    """Selected vertices (or face centres) of a mesh in world space.

    Reads the live Edit Mode selection (bmesh) or, in Object Mode, the selection
    stored in the mesh. With ``face_mode`` and at least one selected face the
    face centres are used (weights = world face areas); otherwise the selected
    vertices (weights 1). Returns ``points (P, 3)``, ``normals (P, 3)`` (unit,
    world), ``weights (P,)`` and ``faces`` (True when face centres were used).
    """
    matrix, linear, normal_matrix = _world_linear(obj)
    det = abs(float(np.linalg.det(linear)))
    points = normals = weights = None
    use_faces = False
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        if face_mode:
            faces = [face for face in bm.faces if face.select]
            if faces:
                use_faces = True
                points = np.array([tuple(f.calc_center_median()) for f in faces], dtype=np.float64)
                normals = np.array([tuple(f.normal) for f in faces], dtype=np.float64)
                weights = np.array([f.calc_area() for f in faces], dtype=np.float64)
        if not use_faces:
            verts = [vert for vert in bm.verts if vert.select]
            points = np.array([tuple(v.co) for v in verts], dtype=np.float64).reshape(-1, 3)
            normals = np.array([tuple(v.normal) for v in verts], dtype=np.float64).reshape(-1, 3)
            weights = np.ones(points.shape[0], dtype=np.float64)
    else:
        mesh = obj.data
        if face_mode and len(mesh.polygons):
            n_polys = len(mesh.polygons)
            selected = np.zeros(n_polys, dtype=bool)
            mesh.polygons.foreach_get("select", selected)
            if selected.any():
                use_faces = True
                centres = np.empty(n_polys * 3, dtype=np.float32)
                face_normals = np.empty(n_polys * 3, dtype=np.float32)
                areas = np.empty(n_polys, dtype=np.float32)
                mesh.polygons.foreach_get("center", centres)
                mesh.polygons.foreach_get("normal", face_normals)
                mesh.polygons.foreach_get("area", areas)
                points = centres.reshape(-1, 3)[selected].astype(np.float64)
                normals = face_normals.reshape(-1, 3)[selected].astype(np.float64)
                weights = areas[selected].astype(np.float64)
        if not use_faces:
            n_verts = len(mesh.vertices)
            selected = np.zeros(n_verts, dtype=bool)
            if n_verts:
                mesh.vertices.foreach_get("select", selected)
            co = build.mesh_vertices(mesh)
            vertex_normals = np.zeros(n_verts * 3, dtype=np.float32)
            if n_verts:
                if hasattr(mesh, "vertex_normals"):  # Blender 4.1+
                    mesh.vertex_normals.foreach_get("vector", vertex_normals)
                else:
                    mesh.vertices.foreach_get("normal", vertex_normals)
            points = co[selected]
            normals = vertex_normals.reshape(-1, 3)[selected].astype(np.float64)
            weights = np.ones(points.shape[0], dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    world_normals = _unit_rows(np.asarray(normals).reshape(-1, 3) @ normal_matrix.T)
    if use_faces:
        # area of a planar face under a linear map: det(A) * |A^-T n| * area
        scale = det * np.linalg.norm(_unit_rows(normals) @ normal_matrix.T, axis=1)
        weights = np.asarray(weights, dtype=np.float64) * scale
    return {
        "points": points @ linear.T + matrix[:3, 3],
        "normals": world_normals,
        "weights": np.maximum(np.asarray(weights, dtype=np.float64), 0.0),
        "faces": use_faces,
    }


def _object_diagonal(context, obj) -> float:
    bounds = _world_bounds(context, obj)
    if bounds is None:
        return 1.0
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    return diagonal if diagonal > 0.0 else 1.0


def _rotation_to(direction):
    """Euler rotation turning local +Z onto ``direction``."""
    target = Vector(tuple(float(c) for c in direction))
    if target.length < 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(Vector((0.0, 0.0, 1.0)).rotation_difference(target.normalized()).to_euler("XYZ"))


def add_role_empties(context, parent, role, points, directions=None, magnitudes=None, size=1.0):
    """LOAD (arrows along ``directions``) or SUPPORT empties at world ``points``.

    Named ``SS_Load`` / ``SS_Support`` (Blender numbers them), colored, linked to
    "StressSculpt Setup" and parented to ``parent`` (world transforms kept).
    Returns the new objects.
    """
    scene = context.scene
    created = []
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    for index, point in enumerate(points):
        if role == "LOAD":
            direction = directions[index]
            obj = new_role_empty(
                scene,
                "SS_Load",
                "LOAD",
                "SINGLE_ARROW",
                size,
                point,
                rotation=_rotation_to(direction),
                color=build.LOAD_COLOR,
            )
            obj.stress_sculpt.magnitude = float(magnitudes[index])
        else:
            obj = new_role_empty(
                scene, "SS_Support", "SUPPORT", "SPHERE", size, point, color=build.SUPPORT_COLOR
            )
            obj.stress_sculpt.fix_x = obj.stress_sculpt.fix_y = obj.stress_sculpt.fix_z = True
        if parent is not None:
            obj.parent = parent
            obj.matrix_parent_inverse = parent.matrix_world.inverted()
        created.append(obj)
    return created


def _selection_or_error(context, max_points=None) -> list:
    """``[(object, selection_points dict)]`` for the selected points (ValueError if none)."""
    tool = getattr(context, "tool_settings", None)
    mode = tuple(tool.mesh_select_mode) if tool is not None else (True, False, False)
    face_mode = bool(mode[2])
    found = []
    for obj in _selection_objects(context):
        selection = selection_points(obj, face_mode)
        if selection["points"].shape[0]:
            found.append((obj, selection))
    if not found:
        raise ValueError(
            "Nothing selected: in Edit Mode select the vertices or faces where the weight goes"
        )
    return found


class _AtSelectionMixin:
    @classmethod
    def poll(cls, context):
        if RUNTIME.busy():
            return False
        if context.mode == "EDIT_MESH":
            return True
        obj = context.active_object
        return context.mode == "OBJECT" and obj is not None and obj.type == "MESH"


class STRESS_SCULPT_OT_add_load_at_selection(_AtSelectionMixin, Operator):
    bl_idname = "stress_sculpt.add_load_at_selection"
    bl_label = "Put Weight on Selection"
    bl_description = (
        "Put a weight on every selected vertex (or selected face centre): the structure will "
        "carry these points"
    )
    bl_options = {"REGISTER", "UNDO"}

    force: FloatProperty(
        name="Force",
        default=1000.0,
        min=0.0,
        soft_max=1.0e6,
        description="Weight in newtons, per point or in total",
    )
    distribute: EnumProperty(
        name="Weight Is",
        items=[
            ("PER_POINT", "Per Point", "Every selected point carries the full force"),
            ("TOTAL", "Total", "The force is shared by the points (by face area for faces)"),
        ],
        default="PER_POINT",
    )
    direction: EnumProperty(
        name="Direction",
        items=[
            ("GRAVITY", "Gravity", "Straight down (world -Z)"),
            ("INTO_SURFACE", "Into Surface", "Pressing into the surface (against the normal)"),
        ],
        default="GRAVITY",
    )
    max_points: IntProperty(
        name="Max Points",
        default=64,
        min=1,
        max=1024,
        description="More selected points are merged into this many weights (force-weighted)",
    )

    def execute(self, context):
        try:
            found = _selection_or_error(context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        forms = core_module("forms")
        settings = _settings(context)
        added = 0
        total = 0.0
        merged = 0
        for obj, selection in found:
            points, normals, weights = (
                selection["points"],
                selection["normals"],
                selection["weights"],
            )
            n = points.shape[0]
            if self.distribute == "TOTAL":
                share = weights / weights.sum() if weights.sum() > 0.0 else np.full(n, 1.0 / n)
                magnitudes = float(self.force) * share
            else:
                magnitudes = np.full(n, float(self.force))
            if self.direction == "INTO_SURFACE":
                directions = -normals
                bad = np.linalg.norm(directions, axis=1) < 0.5
                directions[bad] = (0.0, 0.0, -1.0)
            else:
                directions = np.tile([0.0, 0.0, -1.0], (n, 1))
            forces = directions * magnitudes[:, None]
            if n > int(self.max_points):
                points, forces = forms.cluster_loads(points, forces, int(self.max_points))
                merged += n - points.shape[0]
            magnitudes = np.linalg.norm(forces, axis=1)
            keep = magnitudes > 0.0
            directions = forces[keep] / magnitudes[keep, None]
            size = 0.08 * _object_diagonal(context, obj)
            created = add_role_empties(
                context, obj, "LOAD", points[keep], directions, magnitudes[keep], size
            )
            added += len(created)
            total += float(magnitudes[keep].sum())
        if settings.target is None:
            settings.target = found[0][0]
        if not added:
            self.report({"WARNING"}, "No weight added: the force is 0")
            return {"CANCELLED"}
        message = f"Added {added} weight(s), total {_format_force(total)}"
        if merged:
            message += f" ({merged} nearby point(s) merged)"
        self.report({"INFO"}, message)
        return {"FINISHED"}


class STRESS_SCULPT_OT_add_support_at_selection(_AtSelectionMixin, Operator):
    bl_idname = "stress_sculpt.add_support_at_selection"
    bl_label = "Support at Selection"
    bl_description = (
        "Anchor the selected vertices (or face centres): where the block rests (Carve Inside) "
        "or where the structure lands (Support Object)"
    )
    bl_options = {"REGISTER", "UNDO"}

    max_points: IntProperty(
        name="Max Points",
        default=64,
        min=1,
        max=1024,
        description="More selected points are merged into this many supports",
    )

    def execute(self, context):
        try:
            found = _selection_or_error(context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        forms = core_module("forms")
        settings = _settings(context)
        added = 0
        on_target = False
        for obj, selection in found:
            points = selection["points"]
            if points.shape[0] > int(self.max_points):
                weights = np.maximum(selection["weights"], 1e-12)
                points, _w = forms.cluster_loads(
                    points, np.column_stack([np.zeros((points.shape[0], 2)), weights]),
                    int(self.max_points),
                )
            bounds = _world_bounds(context, obj)
            longest = float(np.max(bounds[1] - bounds[0])) if bounds is not None else 1.0
            voxel = max(longest, 1e-6) / max(int(settings.resolution), 1)
            size = max(1.5 * voxel, 0.03 * _object_diagonal(context, obj))
            added += len(add_role_empties(context, obj, "SUPPORT", points, size=size))
            on_target = on_target or (settings.target is not None and obj == settings.target)
        if settings.target is None and settings.structure_mode == "CARVE":
            settings.target = found[0][0]
        if on_target and settings.structure_mode == "SUPPORT":
            self.report(
                {"WARNING"},
                "These supports sit on the carried object: in Support Object mode footings go on "
                "the ground (or another mesh) below the weights; use Carve Inside to anchor the "
                "object itself",
            )
        self.report({"INFO"}, f"Added {added} support(s)")
        return {"FINISHED"}


class STRESS_SCULPT_OT_remove_role(Operator):
    bl_idname = "stress_sculpt.remove_role"
    bl_label = "Remove Helpers"
    bl_description = (
        "Delete every helper with this role (meshes are kept and just lose the role)"
    )
    bl_options = {"REGISTER", "UNDO"}

    role: EnumProperty(name="Role", items=ROLE_ITEMS, default="LOAD")

    @classmethod
    def poll(cls, context):
        return not RUNTIME.busy()

    def execute(self, context):
        if self.role == "NONE":
            self.report({"ERROR"}, "Pick a role to remove")
            return {"CANCELLED"}
        scene = context.scene
        empties, cleared = [], 0
        for obj in list(scene.objects):
            settings = getattr(obj, "stress_sculpt", None)
            if settings is None or settings.role != self.role:
                continue
            if obj.type == "EMPTY":
                empties.append(obj)
            else:
                settings.role = "NONE"
                cleared += 1
        removed = build.remove_objects(empties)
        label = self.role.replace("_", " ").lower()
        message = f"Removed {removed} {label} helper(s)"
        if cleared:
            message += f", {cleared} mesh(es) no longer {label}"
        self.report({"INFO"}, message)
        return {"FINISHED"}


# ------------------------------------------------------------------ demo scenes
def grid_box_mesh(name, size, divisions):
    """Closed quad mesh of a box centred on the origin, every side split into a grid.

    ``size`` (sx, sy, sz); ``divisions`` (nx, ny, nz) cells along each axis.
    Outward normals; the sides share their border vertices (watertight).
    """
    n = [max(1, int(d)) for d in divisions]
    axes = [np.linspace(-0.5 * s, 0.5 * s, d + 1) for s, d in zip(size, n)]
    I, J, K = np.meshgrid(*(np.arange(d + 1) for d in n), indexing="ij")
    surface = (I == 0) | (I == n[0]) | (J == 0) | (J == n[1]) | (K == 0) | (K == n[2])
    index = -np.ones(I.shape, dtype=np.int64)
    index[surface] = np.arange(int(surface.sum()))
    verts = np.stack(
        [axes[0][I[surface]], axes[1][J[surface]], axes[2][K[surface]]], axis=1
    )
    faces = []

    def quads(a, b, fixed_axis, fixed, flip):
        # grid of cells over axes (a, b) at index ``fixed`` of ``fixed_axis``
        for u in range(n[a]):
            for v in range(n[b]):
                corners = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    ijk = [0, 0, 0]
                    ijk[a], ijk[b], ijk[fixed_axis] = u + du, v + dv, fixed
                    corners.append(index[tuple(ijk)])
                faces.append(corners[::-1] if flip else corners)

    quads(0, 1, 2, n[2], False)  # top (+Z)
    quads(0, 1, 2, 0, True)  # bottom (-Z)
    quads(1, 2, 0, n[0], False)  # +X
    quads(1, 2, 0, 0, True)  # -X
    quads(2, 0, 1, n[1], False)  # +Y
    quads(2, 0, 1, 0, True)  # -Y
    return build.mesh_from_faces(name, verts, np.asarray(faces, dtype=np.int64), smooth=False)


def _face_centres(size, divisions, side) -> np.ndarray:
    """Local centres of the grid cells on the top (+1) or bottom (-1) side of a grid box."""
    nx, ny = int(divisions[0]), int(divisions[1])
    xs = (np.arange(nx) + 0.5) / nx * size[0] - 0.5 * size[0]
    ys = (np.arange(ny) + 0.5) / ny * size[1] - 0.5 * size[1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.5 * side * size[2])], axis=1)


# name, label, description, icon
DEMO_ITEMS = [
    (
        "FLOATING_CUBE",
        "Floating Cube",
        "A cube held up by a branching tree column (weights on its bottom corners and edges)",
        "MESH_CUBE",
        0,
    ),
    (
        "CANOPY",
        "Canopy",
        "A roof slab with 25 weights carried by organic columns on four footings",
        "MOD_FLUIDSIM",
        1,
    ),
    (
        "PAVILION",
        "Pavilion",
        "A rotated floating volume on three footings, its supports covered with stress ribs",
        "MOD_LATTICE",
        2,
    ),
    (
        "BRIDGE",
        "Arch Bridge",
        "A block resting on its two ends with weights along the top: an arch is carved out",
        "MOD_SIMPLEDEFORM",
        3,
    ),
    (
        "CARVED_CUBE",
        "Carved Cube",
        "A cube on its four bottom corners, weighted at the top corners, carved into ribs",
        "MOD_BOOLEAN",
        4,
    ),
]


def demo_layout(preset) -> dict:
    """Object, weights, supports and settings of an architectural demo preset.

    Returns ``size``, ``divisions``, ``location``, ``rotation_z``, ``loads``
    (local points), ``force`` (N per weight), ``supports`` (local points),
    ``settings`` (scene settings to apply).
    """
    if preset == "FLOATING_CUBE":
        size = (2.0, 2.0, 2.0)
        bottom = -1.0
        loads = [(x, y, bottom) for x in (-1.0, 1.0) for y in (-1.0, 1.0)]
        loads += [(0.0, -1.0, bottom), (0.0, 1.0, bottom), (-1.0, 0.0, bottom), (1.0, 0.0, bottom)]
        return {
            "name": "SS_Cube",
            "size": size,
            "divisions": (2, 2, 2),
            "location": (0.0, 0.0, 4.0),
            "rotation_z": 0.0,
            "loads": np.array(loads),
            "force": 2000.0,
            "supports": np.zeros((0, 3)),
            "settings": {
                "structure_mode": "SUPPORT",
                "structure_style": "BRANCHING",
                "ground_height": 0.0,
                "ground_contact": "FOOTINGS",
                "footings": 0,
                "resolution": 40,
            },
        }
    if preset == "CANOPY":
        size = (5.0, 5.0, 0.25)
        return {
            "name": "SS_Roof",
            "size": size,
            "divisions": (5, 5, 1),
            "location": (0.0, 0.0, 3.2 + 0.125),
            "rotation_z": 0.0,
            "loads": _face_centres(size, (5, 5), +1),
            "force": 800.0,
            "supports": np.zeros((0, 3)),
            "settings": {
                "structure_mode": "SUPPORT",
                "structure_style": "ORGANIC",
                "ground_height": 0.0,
                "ground_contact": "FOOTINGS",
                "footings": 4,
                "resolution": 48,
                "max_iterations": 40,
                "support_material": 0.06,
                "filter_radius": 1.5,
                "load_pad": 1.5,
                "symmetry_x": True,
                "symmetry_y": True,
                "symmetry_z": False,
            },
        }
    if preset == "PAVILION":
        size = (4.0, 2.5, 1.0)
        return {
            "name": "SS_Pavilion",
            "size": size,
            "divisions": (4, 3, 1),
            "location": (0.0, 0.0, 2.6 + 0.5),
            "rotation_z": math.radians(15.0),
            "loads": _face_centres(size, (4, 3), -1),
            "force": 1500.0,
            "supports": np.zeros((0, 3)),
            "settings": {
                "structure_mode": "SUPPORT",
                "structure_style": "RIBS",
                "ground_height": 0.0,
                "ground_contact": "FOOTINGS",
                "footings": 3,
                "resolution": 44,
                "max_iterations": 30,
                "support_material": 0.07,
                "filter_radius": 1.5,
                "load_pad": 1.5,
                "symmetry_x": False,
                "symmetry_y": False,
                "symmetry_z": False,
                "rib_spacing": 1.5,
                "rib_thickness": 0.45,
                "rib_min_thickness": 0.8,
            },
        }
    if preset == "BRIDGE":
        size = (8.0, 1.2, 2.5)
        xs = np.linspace(-3.6, 3.6, 9)
        return {
            "name": "SS_Block",
            "size": size,
            "divisions": (8, 1, 2),
            "location": (0.0, 0.0, 1.25),
            "rotation_z": 0.0,
            "loads": np.stack([xs, np.zeros(9), np.full(9, 1.25)], axis=1),
            "force": 3000.0,
            "supports": np.array(
                [[sx * 3.85, sy * 0.45, -1.25] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
            ),
            "settings": {
                "structure_mode": "CARVE",
                "structure_style": "ORGANIC",
                "resolution": 56,
                "max_iterations": 40,
                "volume_fraction": 0.18,
                "filter_radius": 1.5,
                "symmetry_x": True,
                "symmetry_y": True,
                "symmetry_z": False,
            },
            "load_radius": 0.5,
        }
    if preset == "CARVED_CUBE":
        size = (2.0, 2.0, 2.0)
        return {
            "name": "SS_Cube",
            "size": size,
            "divisions": (2, 2, 2),
            "location": (0.0, 0.0, 1.0),
            "rotation_z": 0.0,
            "loads": np.array([(x, y, 1.0) for x in (-1.0, 1.0) for y in (-1.0, 1.0)]),
            "force": 3000.0,
            "supports": np.array([(x, y, -1.0) for x in (-1.0, 1.0) for y in (-1.0, 1.0)]),
            "settings": {
                "structure_mode": "CARVE",
                "structure_style": "RIBS",
                "rib_source": "SURFACE",
                "resolution": 40,
                "max_iterations": 30,
                "volume_fraction": 0.25,
                "filter_radius": 1.5,
                "symmetry_x": True,
                "symmetry_y": True,
                "symmetry_z": False,
                "rib_spacing": 1.5,
                "rib_thickness": 0.45,
                "rib_min_thickness": 0.8,
            },
        }
    raise ValueError(f"Unknown demo preset '{preset}'")


# Scene settings every demo resets (so presets do not inherit tweaks of the previous one).
DEMO_DEFAULTS = (
    "tree_branching",
    "tree_split",
    "tree_levels",
    "tree_organic",
    "tree_thickness",
    "tree_min_thickness",
    "tree_footing_flare",
    "tree_capital_flare",
    "tree_blend",
    "tree_detail",
    "rib_source",
    "rib_spacing",
    "rib_thickness",
    "rib_min_thickness",
    "rib_skin",
    "line_families",
    "sculpt_blend",
    "smooth_iterations",
    "iso_level",
    "evo_mode",
    "penalty",
    "move_limit",
    "min_change",
    "live_update",
    "load_pad",
    "support_material",
    "volume_fraction",
    "filter_radius",
    "max_iterations",
    "symmetry_x",
    "symmetry_y",
    "symmetry_z",
    "ground_height",
    "ground_contact",
    "footings",
)


def _reset_to_defaults(settings, names) -> None:
    properties = settings.bl_rna.properties
    for name in names:
        prop = properties.get(name)
        if prop is None or prop.is_readonly:
            continue
        try:
            setattr(settings, name, prop.default)
        except (TypeError, AttributeError, ValueError):
            pass


class STRESS_SCULPT_OT_demo_scene(Operator):
    bl_idname = "stress_sculpt.demo_scene"
    bl_label = "Example"
    bl_description = (
        "Build a ready-to-run architectural example (object, weights, supports and settings): "
        "press Generate Structure"
    )
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(name="Example", items=DEMO_ITEMS, default="FLOATING_CUBE")

    def execute(self, context):
        if RUNTIME.busy():
            self.report({"ERROR"}, "Wait for the running job to finish (or cancel it)")
            return {"CANCELLED"}
        try:
            _ensure_object_mode(context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        scene = context.scene
        settings = scene.stress_sculpt
        layout = demo_layout(self.preset)

        # A fresh example: earlier demo objects and the results go away.
        build.remove_objects([obj for obj in scene.objects if obj.get(DEMO_TAG) is not None])
        results = build.results_collection(scene, create=False)
        if results is not None:
            build.remove_objects(list(results.objects))
        jobs.clear_cache(scene)
        disabled = 0
        for obj in scene_io.role_objects(context):
            obj.stress_sculpt.enabled = False
            disabled += 1

        setup = build.ensure_collection(scene, build.SETUP_COLLECTION)
        mesh = grid_box_mesh(layout["name"], layout["size"], layout["divisions"])
        target = bpy.data.objects.new(layout["name"], mesh)
        target.location = layout["location"]
        target.rotation_euler = (0.0, 0.0, layout["rotation_z"])
        target[DEMO_TAG] = self.preset
        setup.objects.link(target)
        context.view_layer.update()

        matrix = np.array(target.matrix_world, dtype=np.float64)
        to_world = lambda local: local @ matrix[:3, :3].T + matrix[:3, 3]  # noqa: E731
        loads = to_world(np.asarray(layout["loads"], dtype=np.float64).reshape(-1, 3))
        diagonal = float(np.linalg.norm(layout["size"]))
        created = add_role_empties(
            context,
            target,
            "LOAD",
            loads,
            np.tile([0.0, 0.0, -1.0], (loads.shape[0], 1)),
            np.full(loads.shape[0], layout["force"]),
            0.08 * diagonal,
        )
        for obj in created:
            obj.stress_sculpt.load_radius = float(layout.get("load_radius", 0.0))
        supports = np.asarray(layout["supports"], dtype=np.float64).reshape(-1, 3)
        if supports.shape[0]:
            longest = float(max(layout["size"]))
            voxel = longest / max(int(layout["settings"].get("resolution", 40)), 1)
            size = max(1.5 * voxel, 0.03 * diagonal)
            created += add_role_empties(context, target, "SUPPORT", to_world(supports), size=size)
        for obj in created:
            obj[DEMO_TAG] = self.preset

        _reset_to_defaults(settings, DEMO_DEFAULTS)
        for key, value in layout["settings"].items():
            setattr(settings, key, value)
        settings.target = target
        if settings.structure_mode == "CARVE":
            settings.domain = target
        elif settings.domain is not None and settings.domain.get(DEMO_TAG) is not None:
            settings.domain = None
        _set_last(scene, "")

        _deselect_all(context)
        try:
            target.select_set(True, view_layer=context.view_layer)
            context.view_layer.objects.active = target
        except RuntimeError:
            pass
        label = next(item[1] for item in DEMO_ITEMS if item[0] == self.preset)
        message = f"Example '{label}' ready: press Generate Structure"
        if disabled:
            message += f" ({disabled} earlier helper(s) disabled)"
        self.report({"INFO"}, message)
        return {"FINISHED"}


# ===================================================================== threaded ops
class STRESS_SCULPT_OT_analyze(ThreadedOperatorMixin, Operator):
    bl_idname = "stress_sculpt.analyze"
    bl_label = "Analyze Stress"
    bl_description = (
        "Voxelize the domain, solve for the stresses under the loads and color the shape"
    )
    bl_options = {"REGISTER", "UNDO"}
    job_name = "Analyze"

    def prepare(self, context):
        _ensure_object_mode(context)
        settings = _settings(context)
        problem, info = scene_io.gather_problem(context)
        return {
            "scene": context.scene.name,
            "problem": problem,
            "info": info,
            "signature": scene_io.setup_signature(context),
            "solver": settings.solver,
            "tol": float(settings.solver_tolerance),
            "maxiter": int(settings.max_solver_iterations),
            "fem": core_module("fem"),
            "mesher": core_module("mesher"),
        }

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        result = payload["fem"].analyze(
            payload["problem"],
            solver=payload["solver"],
            tol=payload["tol"],
            maxiter=payload["maxiter"],
            progress=progress,
            should_stop=should_stop,
        )
        return {"payload": payload, "result": result}

    def finish(self, context, output):
        payload, result = output["payload"], output["result"]
        scene = _job_scene(payload["scene"])
        problem, info = payload["problem"], payload["info"]
        jobs.get_cache(scene).analysis = {
            "problem": problem,
            "result": result,
            "signature": payload["signature"],
            "grid": problem.grid,
            "info": info,
        }
        view_layer = _view_layer(context, scene)
        self._build_heatmap(context, scene, payload, result, view_layer)
        build.hide_results(scene, (build.OBJ_RELIEF,), view_layer)
        _show_domain_as_wire(scene)
        build.use_attribute_colors(context)

        max_stress = float(np.max(result.von_mises)) if problem.domain.any() else 0.0
        _set_last(
            scene,
            KIND_ANALYSIS,
            max_stress=max_stress,
            max_displacement=result.max_displacement,
            compliance=result.compliance,
            volume=1.0,
        )
        _report_warnings(self, info["warnings"])
        converged = result.info.get("converged", True)
        if not converged:
            self.report(
                {"WARNING"}, "The solver did not fully converge: raise Max Solver Iterations"
            )
        self.report(
            {"INFO"},
            f"Max von Mises {max_stress / 1e6:.4g} MPa, max displacement "
            f"{result.max_displacement * 1e3:.4g} mm ({info['n_elems']} voxels)",
        )
        return {"FINISHED"}

    @staticmethod
    def _build_heatmap(context, scene, payload, result, view_layer):
        settings = scene.stress_sculpt
        problem = payload["problem"]
        grid = problem.grid
        fem, mesher = payload["fem"], payload["mesher"]
        field, signed = measure_field(problem, result, settings.stress_measure, fem)
        lo, hi = color_range(field[problem.domain], signed, settings.colormap, settings.color_clip)
        domain_obj = bpy.data.objects.get(payload["info"]["domain_name"])
        on_original = (
            settings.heatmap_on_original
            and domain_obj is not None
            and domain_obj.type == "MESH"
            and domain_obj.name in scene.objects
        )
        with build.NewData() as new:
            if on_original:
                mesh = new.add(
                    build.world_mesh_copy(
                        domain_obj,
                        _depsgraph(context, scene),
                        build.OBJ_STRESS,
                        target_edge=grid.h,
                        max_faces=MESH_MAX_FACES,
                    )
                )
                verts = build.mesh_vertices(mesh)
            else:
                verts, quads = mesher.surface_nets(
                    problem.domain.astype(np.float64), 0.5, grid.origin, grid.h
                )
                mesh = new.add(build.mesh_from_faces(build.OBJ_STRESS, verts, quads))
            values = sample_field(field, problem.domain, grid, verts, mesher)
            colors = mesher.colormap(values, lo, hi, name=settings.colormap)
            build.set_mesh_stress(mesh, colors, values)
            return build.assign_result(
                scene, build.OBJ_STRESS, mesh, build.heatmap_material(), view_layer
            )


def _opt_settings(settings, topopt):
    return topopt.OptSettings(
        mode=settings.evo_mode,
        volfrac=float(settings.volume_fraction),
        penal=float(settings.penalty),
        rmin=float(settings.filter_radius),
        move=float(settings.move_limit),
        max_iter=int(settings.max_iterations),
        tol_change=float(settings.min_change),
        grow_threshold=float(settings.grow_threshold),
        hollow_threshold=float(settings.hollow_threshold),
        rate=float(settings.growth_rate),
        keep_volume=bool(settings.keep_volume),
        symmetry=(bool(settings.symmetry_x), bool(settings.symmetry_y), bool(settings.symmetry_z)),
        solver=settings.solver,
        solver_tol=float(settings.solver_tolerance),
        solver_maxiter=int(settings.max_solver_iterations),
    )


def run_optimizer(topopt, problem, opt_settings, progress, should_stop, on_state):
    """Iterate ``topopt.Optimizer`` (worker side); ``on_state(state)`` after every iteration.

    ``progress`` maps the iterations onto [0, 1]. The loop is bounded by
    ``max_iter`` even if the optimizer never reports ``done``. Returns
    ``(optimizer, last_state)`` (states are private copies).
    """
    optimizer = topopt.Optimizer(problem, opt_settings)
    max_iter = max(1, int(opt_settings.max_iter))
    progress(0.0, "Iteration 1: solving")
    state = None
    for _ in range(max_iter):
        state = _copy_state(optimizer.step(should_stop=should_stop))
        on_state(state)
        progress(
            min(state.iteration / max_iter, 1.0),
            f"Iteration {state.iteration}/{max_iter}: material {state.volume * 100:.0f}%, "
            f"change {state.change:.3f}",
        )
        if optimizer.done or state.converged or state.iteration >= max_iter:
            break
        check_stop(should_stop)
    return optimizer, state


def _max_displacement(optimizer, state, problem, opt_settings, should_stop):
    """Largest nodal displacement of the final design (one extra solve if not exposed)."""
    for name in ("u", "displacement", "_u"):
        u = getattr(optimizer, name, None)
        if isinstance(u, np.ndarray) and u.size == problem.grid.n_dofs:
            return float(np.max(np.linalg.norm(u.reshape(-1, 3), axis=1)))
    model = getattr(optimizer, "model", None)
    if model is None:
        return 0.0
    rho = np.asarray(state.density, dtype=np.float64).ravel()[model.elems]
    scale = model.stiffness_scale(rho, float(state.penal))
    u, _info = model.solve(
        scale,
        tol=opt_settings.solver_tol,
        maxiter=opt_settings.solver_maxiter,
        should_stop=should_stop,
    )
    return float(np.max(np.linalg.norm(np.asarray(u).reshape(-1, 3), axis=1)))


class STRESS_SCULPT_OT_evolve(ThreadedOperatorMixin, Operator):
    bl_idname = "stress_sculpt.evolve"
    bl_label = "Evolve Shape"
    bl_description = (
        "Let the shape evolve under the loads: topology optimization or stress-driven growth "
        "rules, shown live (Esc keeps the current state)"
    )
    bl_options = {"REGISTER", "UNDO"}
    job_name = "Evolve"
    keep_partial_on_cancel = True

    def prepare(self, context):
        _ensure_object_mode(context)
        settings = _settings(context)
        modules = _modules("topopt", "mesher")
        problem, info = scene_io.gather_problem(context)
        return {
            "scene": context.scene.name,
            "problem": problem,
            "info": info,
            "signature": scene_io.setup_signature(context),
            "opt": _opt_settings(settings, modules["topopt"]),
            "params": {
                "iso": float(settings.iso_level),
                "smooth": int(settings.smooth_iterations),
                "colormap": settings.colormap,
                "clip": float(settings.color_clip),
            },
            "preview": self.is_async() and bool(settings.live_update),
            "modules": modules,
        }

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        topopt, mesher = payload["modules"]["topopt"], payload["modules"]["mesher"]
        problem, opt_settings, params = payload["problem"], payload["opt"], payload["params"]
        snapshots = []

        def on_state(state):
            snapshot = {"payload": payload, "state": state, "preview": None}
            if payload["preview"]:
                snapshot["preview"] = evolved_geometry(
                    state, problem, params, mesher, 2, clean=False
                )
            snapshots[:] = [snapshot]
            emit(snapshot)

        optimizer, state = run_optimizer(
            topopt, problem, opt_settings, progress, should_stop, on_state
        )
        snapshot = snapshots[0]
        final = dict(snapshot)
        final["geometry"] = evolved_geometry(state, problem, params, mesher, params["smooth"])
        final["max_displacement"] = _max_displacement(
            optimizer, state, problem, opt_settings, should_stop
        )
        return final

    def on_partial(self, context, snapshot):
        geometry = snapshot.get("preview")
        if geometry is None:
            return
        scene = bpy.data.scenes.get(snapshot["payload"]["scene"])
        if scene is None:
            return
        view_layer = _view_layer(context, scene)
        with build.NewData() as new:
            mesh = new.add(_mesh_from_geometry(build.OBJ_EVOLVED, geometry))
            build.assign_result(
                scene, build.OBJ_EVOLVED, mesh, build.heatmap_material(), view_layer
            )
        if not getattr(self, "_ss_preview_shown", False):
            self._ss_preview_shown = True
            build.hide_results(scene, ENVELOPES, view_layer)
            _show_domain_as_wire(scene)
            build.use_attribute_colors(context)

    def finish(self, context, snapshot):
        payload, state = snapshot["payload"], snapshot["state"]
        scene = _job_scene(payload["scene"])
        problem, params = payload["problem"], payload["params"]
        geometry = snapshot.get("geometry")
        if geometry is None:  # cancelled: mesh the last state here
            geometry = evolved_geometry(
                state, problem, params, payload["modules"]["mesher"], params["smooth"]
            )
        view_layer = _view_layer(context, scene)
        with build.NewData() as new:
            mesh = new.add(_mesh_from_geometry(build.OBJ_EVOLVED, geometry))
            build.assign_result(
                scene, build.OBJ_EVOLVED, mesh, build.heatmap_material(), view_layer
            )
        build.hide_results(scene, ENVELOPES, view_layer)
        _show_domain_as_wire(scene)
        build.use_attribute_colors(context)

        max_displacement = snapshot.get("max_displacement")
        jobs.get_cache(scene).evolution = {
            "problem": problem,
            "state": state,
            "settings": payload["opt"],
            "signature": payload["signature"],
            "info": payload["info"],
            "iso": params["iso"],
            "max_displacement": max_displacement,
        }
        _set_last(
            scene,
            KIND_EVOLVE,
            max_stress=float(np.max(state.von_mises)),
            max_displacement=max_displacement or 0.0,
            compliance=state.compliance,
            volume=state.volume,
            iterations=state.iteration,
        )
        _report_warnings(self, payload["info"]["warnings"])
        if len(geometry["faces"]) == 0:
            self.report(
                {"WARNING"},
                "No material above the Surface Level: lower it or raise the Material amount",
            )
        status = "converged" if state.converged else "stopped"
        self.report(
            {"INFO"},
            f"Evolved {state.iteration} iteration(s) ({status}): "
            f"material {state.volume * 100:.1f}%, "
            f"compliance {state.compliance:.4g} J",
        )
        return {"FINISHED"}


class STRESS_SCULPT_OT_trace_lines(ThreadedOperatorMixin, Operator):
    bl_idname = "stress_sculpt.trace_lines"
    bl_label = "Trace Lines"
    bl_description = (
        "Trace the tension and compression lines of the stress field (with branches) as curves "
        "whose thickness follows the stress"
    )
    bl_options = {"REGISTER", "UNDO"}
    job_name = "Tension Lines"

    def prepare(self, context):
        _ensure_object_mode(context)
        request = _trace_request(context, _settings(context))
        request["scene"] = context.scene.name
        return request

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        return {"request": payload, "lines": _run_trace(payload, progress, should_stop)}

    def finish(self, context, output):
        request, lines = output["request"], output["lines"]
        scene = _job_scene(request["scene"])
        _store_analysis(scene, request["analysis"])
        problem = request["problem"]
        view_layer = _view_layer(context, scene)
        counts, radii, magnitude_ref = _build_lines(
            scene, lines, problem.grid.h, request["radius"], view_layer
        )
        entry = _lines_entry(request, lines, magnitude_ref)
        jobs.get_cache(scene).lines = entry
        _report_warnings(self, _fresh_warnings(request["analysis"]))
        if not lines:
            self.report({"WARNING"}, "No lines found: lower Min Stress or add seeds")
            return {"FINISHED"}
        build.hide_results(scene, ENVELOPES, view_layer)
        _show_domain_as_wire(scene)
        domain_volume = float(problem.domain.sum()) * problem.grid.h**3
        metrics = entry["metrics"]
        _set_last(
            scene,
            KIND_LINES,
            max_stress=metrics.get("max_stress", 0.0),
            max_displacement=metrics.get("max_displacement", 0.0),
            compliance=metrics.get("compliance", 0.0),
            volume=tubes_volume(lines, radii) / domain_volume if domain_volume > 0 else 0.0,
        )
        branches = sum(1 for line in lines if line.generation > 0)
        self.report(
            {"INFO"},
            f"{counts['TENSION']} tension and {counts['COMPRESSION']} compression line(s), "
            f"{branches} branch(es)",
        )
        return {"FINISHED"}


def _union_with_solid(field, origin, voxel, values, solid, blend, sculpt):
    """Smooth union of the tube field with an evolved density (iso levels matched).

    ``solid`` is a dict ``density``, ``von_mises``, ``domain``, ``grid``, ``iso``.
    Both fields are padded with zeros and resampled onto a grid (spacing
    ``voxel``) covering both; the density is scaled so its surface level maps
    to ``sculpt.TUBE_ISO``. Returns ``(field, origin, values)``.
    """
    grid = solid["grid"]
    iso = max(float(solid["iso"]), 1e-6)
    density = np.where(solid["domain"], np.asarray(solid["density"], dtype=np.float64), 0.0)
    occupied = np.argwhere(density > 0.05 * iso)
    if occupied.size == 0:
        return field, origin, values
    pad = 2
    s_min = grid.origin + (occupied.min(axis=0) - pad) * grid.h
    s_max = grid.origin + (occupied.max(axis=0) + 1 + pad) * grid.h
    u_min = np.minimum(s_min, origin)
    u_max = np.maximum(s_max, origin + np.asarray(field.shape) * voxel)
    shape = tuple(int(n) for n in np.maximum(1, np.ceil((u_max - u_min) / voxel - 1e-9)))
    if int(np.prod(shape)) > SCULPT_MAX_CELLS:
        raise ValueError("The fused sculpture is too large: lower Detail")

    def resample(src, src_origin, src_h):
        padded = np.pad(np.asarray(src, dtype=np.float64), pad)
        return sculpt.resample_to(padded, src_origin - pad * src_h, src_h, shape, u_min, voxel)

    tubes = np.maximum(resample(field, origin, voxel), 0.0)
    tube_values = resample(values, origin, voxel)
    dens = np.maximum(resample(density, grid.origin, grid.h), 0.0) * (sculpt.TUBE_ISO / iso)
    von_mises = np.where(solid["domain"], np.asarray(solid["von_mises"], dtype=np.float64), 0.0)
    solid_values = resample(von_mises, grid.origin, grid.h)
    fused = sculpt.union_fields(tubes, dens, blend)
    weight = tubes + dens
    fused_values = (tubes * tube_values + dens * solid_values) / np.maximum(weight, 1e-30)
    return fused, u_min, np.where(weight > 0.0, fused_values, 0.0)


class STRESS_SCULPT_OT_sculpt(ThreadedOperatorMixin, Operator):
    bl_idname = "stress_sculpt.sculpt"
    bl_label = "Sculpt"
    bl_description = (
        "Fuse the tension lines (optionally with the evolved shape) into one organic sculpture "
        "(traces the lines first if needed)"
    )
    bl_options = {"REGISTER", "UNDO"}
    job_name = "Sculpt"

    def prepare(self, context):
        _ensure_object_mode(context)
        settings = _settings(context)
        cache = jobs.get_cache(context.scene)
        payload = {
            "scene": context.scene.name,
            "modules": _modules("sculpt", "mesher"),
            "warnings": [],
            "detail": float(settings.sculpt_detail),
            "blend": float(settings.sculpt_blend),
            "smooth": int(settings.sculpt_smooth),
            "colormap": settings.colormap,
            "clip": float(settings.color_clip),
            "radius": (
                float(settings.radius_min),
                float(settings.radius_max),
                float(settings.branch_taper),
            ),
            "solid": None,
            "trace": None,
            "lines": None,
            "magnitude_ref": None,
        }
        entry = cache.lines
        if entry is not None and entry.get("lines"):
            payload["lines"] = entry["lines"]
            payload["h"] = float(entry["h"])
            payload["magnitude_ref"] = entry.get("magnitude_ref")
            payload["problem"] = entry.get("problem")
            payload["metrics"] = dict(entry.get("metrics") or {})
        else:
            trace = _trace_request(context, settings)
            trace["scene"] = context.scene.name
            payload["trace"] = trace
            payload["h"] = float(trace["problem"].grid.h)
            payload["problem"] = trace["problem"]
            payload["metrics"] = {}
        if settings.sculpt_source == "LINES_AND_SOLID":
            evolution = cache.evolution
            if evolution is None:
                payload["warnings"].append(
                    "No evolved shape yet: the sculpture uses the lines only"
                )
            else:
                state = evolution["state"]
                payload["solid"] = {
                    "density": state.density,
                    "von_mises": state.von_mises,
                    "domain": evolution["problem"].domain,
                    "grid": evolution["problem"].grid,
                    "iso": float(evolution.get("iso", settings.iso_level)),
                }
        return payload

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        sculpt, mesher = payload["modules"]["sculpt"], payload["modules"]["mesher"]
        lines, start = payload["lines"], 0.0
        warnings = []
        if lines is None:
            lines = _run_trace(payload["trace"], _sub_progress(progress, 0.0, 0.4), should_stop)
            start = 0.4
            if not lines:
                raise ValueError("No tension lines were found: lower Min Stress or add seeds")
        span = 1.0 - start
        h = payload["h"]
        radii, magnitude_ref = line_radii(
            lines, h, *payload["radius"], magnitude_ref=payload["magnitude_ref"]
        )
        tubes = [
            sculpt.Tube(
                points=np.asarray(line.points, dtype=np.float64),
                radius=radius,
                value=np.abs(np.asarray(line.magnitude, dtype=np.float64)),
            )
            for line, radius in zip(lines, radii)
            if len(line.points) >= 2
        ]
        if not tubes:
            raise ValueError("No tension lines to sculpt: trace the lines first")

        # Voxel size from Detail, enlarged when the grid would not fit in memory.
        voxel = h / max(payload["detail"], 1e-3)
        points = np.concatenate([tube.points for tube in tubes])
        r_max = max(float(np.max(tube.radius)) for tube in tubes)
        extent = points.max(axis=0) - points.min(axis=0) + 2.0 * (3.0 * r_max + 4.0 * voxel)
        cells = float(np.prod(np.maximum(extent / voxel, 1.0)))
        if cells > SCULPT_MAX_CELLS:
            voxel *= (cells / SCULPT_MAX_CELLS) ** (1.0 / 3.0)
            warnings.append("Detail was reduced to keep the sculpture within memory")

        field, origin, voxel, values = sculpt.tubes_field(
            tubes,
            voxel,
            blend=payload["blend"],
            progress=_sub_progress(progress, start, start + 0.6 * span),
            should_stop=should_stop,
        )
        field = np.asarray(field, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if payload["solid"] is not None:
            field, origin, values = _union_with_solid(
                field, origin, voxel, values, payload["solid"], payload["blend"], sculpt
            )
        check_stop(should_stop)
        progress(start + 0.65 * span, "Building the surface")
        verts, faces = mesher.surface_nets(field, sculpt.TUBE_ISO, origin, voxel)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
        if faces.shape[0] == 0:
            raise ValueError("The sculpture came out empty: raise the line radius or the detail")
        verts, faces = _drop_small_parts(mesher, verts, faces, max(16, 0.002 * faces.shape[0]))
        check_stop(should_stop)
        if payload["smooth"] > 0:
            progress(start + 0.8 * span, "Smoothing")
            verts = mesher.taubin_smooth(verts, faces, iterations=payload["smooth"])
        vertex_values = sample_field(values, field > 0.0, _GridLike(origin, voxel), verts, mesher)
        lo, hi = color_range(vertex_values, False, payload["colormap"], payload["clip"])
        colors = mesher.colormap(vertex_values, lo, hi, name=payload["colormap"])
        progress(1.0, "Done")
        return {
            "payload": payload,
            "lines": lines,
            "magnitude_ref": magnitude_ref,
            "warnings": warnings,
            "geometry": {
                "verts": np.asarray(verts),
                "faces": faces,
                "colors": colors,
                "values": vertex_values,
            },
        }

    def finish(self, context, output):
        payload, geometry = output["payload"], output["geometry"]
        scene = _job_scene(payload["scene"])
        view_layer = _view_layer(context, scene)
        trace = payload["trace"]
        metrics = payload["metrics"]
        warnings = list(payload["warnings"]) + list(output["warnings"])
        if trace is not None:  # lines were traced in this job: show and cache them too
            _store_analysis(scene, trace["analysis"])
            _build_lines(
                scene,
                output["lines"],
                trace["problem"].grid.h,
                trace["radius"],
                view_layer,
                output["magnitude_ref"],
            )
            entry = _lines_entry(trace, output["lines"], output["magnitude_ref"])
            jobs.get_cache(scene).lines = entry
            metrics = entry["metrics"]
            warnings.extend(_fresh_warnings(trace["analysis"]))
        with build.NewData() as new:
            mesh = new.add(_mesh_from_geometry(build.OBJ_SCULPTURE, geometry))
            build.assign_result(
                scene, build.OBJ_SCULPTURE, mesh, build.sculpture_material(), view_layer
            )
        hidden = (
            ENVELOPES
            + LINE_OBJECTS
            + ((build.OBJ_EVOLVED,) if payload["solid"] is not None else ())
        )
        build.hide_results(scene, hidden, view_layer)
        _show_domain_as_wire(scene)
        build.use_attribute_colors(context)

        problem = payload["problem"]
        volume = 0.0
        if problem is not None and problem.domain.any():
            domain_volume = float(problem.domain.sum()) * problem.grid.h**3
            volume = mesh_volume(geometry["verts"], geometry["faces"]) / domain_volume
        _set_last(
            scene,
            KIND_SCULPT,
            max_stress=metrics.get("max_stress", 0.0),
            max_displacement=metrics.get("max_displacement", 0.0),
            compliance=metrics.get("compliance", 0.0),
            volume=volume,
        )
        _report_warnings(self, warnings)
        self.report(
            {"INFO"},
            f"Sculpture: {len(geometry['faces'])} faces from {len(output['lines'])} line(s)",
        )
        return {"FINISHED"}


class _GridLike:
    """Minimal grid (origin, h) for ``sample_field`` on the sculpture grid."""

    __slots__ = ("origin", "h")

    def __init__(self, origin, h):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.h = float(h)


class STRESS_SCULPT_OT_relief(ThreadedOperatorMixin, Operator):
    bl_idname = "stress_sculpt.relief"
    bl_label = "Relief"
    bl_description = (
        "Copy of the domain surface pushed outwards where stress is high "
        "and inwards where it is low"
    )
    bl_options = {"REGISTER", "UNDO"}
    job_name = "Relief"

    def prepare(self, context):
        _ensure_object_mode(context)
        settings = _settings(context)
        domain = scene_io.validated_domain(context, settings)
        analysis = _analysis_request(context, settings)
        h = analysis["problem"].grid.h
        mesh = build.world_mesh_copy(
            domain,
            context.evaluated_depsgraph_get(),
            "SS_Relief_source",
            target_edge=h,
            extra_levels=int(settings.relief_subdivisions),
            max_faces=MESH_MAX_FACES,
        )
        try:
            arrays = build.mesh_arrays(mesh)
        finally:
            bpy.data.meshes.remove(mesh)
        if arrays["loop_starts"].size == 0:
            raise ValueError(f"The domain '{domain.name}' has no faces")
        bmin, bmax = analysis["info"]["bounds"]
        return {
            "scene": context.scene.name,
            "analysis": analysis,
            "mesh": arrays,
            "diagonal": float(np.linalg.norm(np.asarray(bmax) - np.asarray(bmin))),
            "amount": float(settings.relief_amount),
            "hollow": bool(settings.relief_hollow),
            "threshold": float(settings.relief_threshold),
            "gamma": float(settings.relief_gamma),
            "smooth": int(settings.relief_smooth),
            "clip": float(settings.color_clip),
            "colormap": settings.colormap,
            "mesher": core_module("mesher"),
        }

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        analysis = payload["analysis"]
        needs_solve = analysis["result"] is None
        result = _run_analysis(
            analysis, _sub_progress(progress, 0.0, 0.85) if needs_solve else progress, should_stop
        )
        check_stop(should_stop)
        progress(0.9 if needs_solve else 0.5, "Offsetting the surface")
        mesher = payload["mesher"]
        problem = analysis["problem"]
        arrays = payload["mesh"]
        verts = arrays["verts"]
        von_mises = np.asarray(result.von_mises, dtype=np.float64)
        raw = sample_field(von_mises, problem.domain, problem.grid, verts, mesher)
        reference = (
            float(np.percentile(von_mises[problem.domain], payload["clip"]))
            if problem.domain.any()
            else 0.0
        )
        if not reference > 0.0:
            reference = float(np.max(raw)) if raw.size and np.max(raw) > 0.0 else 1.0
        s = np.clip(raw / reference, 0.0, 1.0) ** payload["gamma"]
        threshold = min(max(payload["threshold"], 0.0), 1.0)
        amplitude = payload["amount"] * payload["diagonal"]
        offset = np.zeros(verts.shape[0], dtype=np.float64)
        if threshold < 1.0:
            high = s > threshold
            offset[high] = amplitude * (s[high] - threshold) / (1.0 - threshold)
        if payload["hollow"] and threshold > 0.0:
            low = s < threshold
            offset[low] = amplitude * (s[low] - threshold) / threshold
        if payload["smooth"] > 0 and arrays["tris"].size:
            offset = np.asarray(
                mesher.laplacian_smooth_scalar(
                    offset, arrays["tris"], verts.shape[0], payload["smooth"]
                ),
                dtype=np.float64,
            )
        moved = verts + arrays["normals"] * offset[:, None]
        colors = mesher.colormap(s, 0.0, 1.0, name=payload["colormap"])
        progress(1.0, "Done")
        return {
            "payload": payload,
            "verts": moved,
            "colors": colors,
            "values": raw,
            "offset": offset,
        }

    def finish(self, context, output):
        payload = output["payload"]
        scene = _job_scene(payload["scene"])
        analysis = payload["analysis"]
        _store_analysis(scene, analysis)
        arrays = payload["mesh"]
        view_layer = _view_layer(context, scene)
        with build.NewData() as new:
            mesh = new.add(
                build.mesh_from_polygons(
                    build.OBJ_RELIEF, output["verts"], arrays["loop_verts"], arrays["loop_starts"]
                )
            )
            build.set_mesh_stress(mesh, output["colors"], output["values"])
            build.assign_result(scene, build.OBJ_RELIEF, mesh, build.heatmap_material(), view_layer)
        build.hide_results(scene, (build.OBJ_STRESS,), view_layer)
        _show_domain_as_wire(scene)
        build.use_attribute_colors(context)

        problem, result = analysis["problem"], analysis["result"]
        domain_volume = float(problem.domain.sum()) * problem.grid.h**3
        volume = polygon_volume(output["verts"], arrays["loop_verts"], arrays["loop_starts"])
        _set_last(
            scene,
            KIND_RELIEF,
            max_stress=float(np.max(result.von_mises)),
            max_displacement=result.max_displacement,
            compliance=result.compliance,
            volume=volume / domain_volume if domain_volume > 0 else 0.0,
        )
        _report_warnings(self, _fresh_warnings(analysis))
        offset = output["offset"]
        self.report(
            {"INFO"},
            f"Relief: {len(arrays['loop_starts'])} faces, "
            f"offset {_format_length(float(offset.min()))} "
            f"to {_format_length(float(offset.max()))}",
        )
        return {"FINISHED"}


# ===================================================================== structure
# "Generate Structure" (Structure panel): the target object is carried by a structure
# growing from the ground (SUPPORT) or carved out of it (CARVE), in three styles:
# ORGANIC (topology optimization), BRANCHING (form-found tree columns) and RIBS
# (stress-line ribs over the optimized shape / the object's surface).
KIND_ORGANIC = "ORGANIC"
KIND_BRANCHING = "BRANCHING"
KIND_RIBS = "RIBS"
STRUCTURE_KINDS = (KIND_ORGANIC, KIND_BRANCHING, KIND_RIBS)
KIND_LABELS.update({KIND_ORGANIC: "Organic", KIND_BRANCHING: "Branching", KIND_RIBS: "Ribs"})
OBJECT_MAX_CELLS = 4_000_000  # fine voxelization of the object (flush contacts)
OBJECT_MAX_SIDE = 160  # voxels along the longest side of that voxelization
VERIFY_MAX_CELLS = 3_000_000  # grid of the structural check
COMPONENT_KEEP = 0.03  # parts smaller than this share of the largest part are dropped
ORGANIC_UPSAMPLE = 2  # the optimized density is meshed on a grid this much finer
ORGANIC_BLUR = 2  # [1, 2, 1] passes over that finer grid (rounds the voxel facets)
ORGANIC_OPENING = 0  # grey opening radius in fine voxels: removes webs thinner than that
MEMBER_TAPER = 1.15  # a member ends at this times its thickest child's radius
RIB_END_TAPER = 0.4  # open rib ends taper to this share of their radius ...
RIB_END_SPAN = 0.1  # ... over this share of their length
# ID properties stored on SS_Structure (read back by "Check Generated Structure")
PROP_MODE = "ss_mode"
PROP_STYLE = "ss_style"
PROP_GROUND = "ss_ground"
PROP_CONTACT = "ss_contact"
PROP_FOOTINGS = "ss_footings"
PROP_THINNEST = "ss_thinnest"
PROP_PAD = "ss_pad"
PROP_CONTACTS = "ss_contacts"  # where the structure meets each weight (flat xyz list)
PROP_LOADS = "ss_loads"  # names of those weights, one per line


# ------------------------------------------------------------ numeric (worker-safe)
def _blur121(a) -> np.ndarray:
    """Separable [1, 2, 1] / 4 blur with replicated borders (softens voxel indicators)."""
    out = np.asarray(a, dtype=np.float64)
    for axis in range(3):
        n = out.shape[axis]
        if n < 2:
            continue
        lower = np.take(out, np.r_[0, np.arange(n - 1)], axis=axis)
        upper = np.take(out, np.r_[np.arange(1, n), n - 1], axis=axis)
        out = 0.5 * out + 0.25 * (lower + upper)
    return out


def _dilate6(mask) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    src = out.copy()
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        out[tuple(lo)] |= src[tuple(hi)]
        out[tuple(hi)] |= src[tuple(lo)]
    return out


def _axis_centres(origin, h, n, axis) -> np.ndarray:
    return float(origin[axis]) + (np.arange(n) + 0.5) * float(h)


def clip_below(field, origin, h, height, iso) -> np.ndarray:
    """Remove everything below the plane ``z = height`` (an exact flat cut).

    ``min(field, iso + (z - height) / h)``: between the cell centres around the
    plane the minimum crosses ``iso`` exactly at the plane wherever the field is
    solid there.
    """
    z = _axis_centres(origin, h, field.shape[2], 2)
    return np.minimum(field, (float(iso) + (z - float(height)) / float(h))[None, None, :])


def indicator_on_grid(indicator, shape, origin, h, sculpt) -> np.ndarray:
    """An indicator dict (``field`` 0..1, ``origin``, ``h``) resampled onto another grid."""
    src = np.pad(np.asarray(indicator["field"], dtype=np.float64), 1)
    src_h = float(indicator["h"])
    src_origin = np.asarray(indicator["origin"], dtype=np.float64) - src_h
    return sculpt.resample_to(src, src_origin, src_h, shape, origin, h)


def clip_with_indicator(field, origin, h, indicator, keep_inside, iso, sculpt) -> np.ndarray:
    """CSG with an object given by an indicator field (surface at 0.5).

    ``keep_inside`` True keeps the part of ``field`` inside the object
    (intersection: nothing leaves a carved block), False removes it
    (difference: the structure stops flush at the object it carries).
    """
    t = indicator_on_grid(indicator, field.shape, origin, h, sculpt)
    gain = 4.0
    g = float(iso) + gain * ((t - 0.5) if keep_inside else (0.5 - t))
    return np.minimum(field, g)


def end_cuts(field, origin, h, ends, up, iso) -> np.ndarray:
    """Flat member ends: cut by the plane through every end, perpendicular to ``up``.

    ``ends`` = ``[(point, radius, side)]``; ``side`` +1 removes what lies above
    the plane (load tips), -1 what lies below (footings). Only a cylinder of
    about 1.6 radii around the end's axis is affected.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / max(float(np.linalg.norm(up)), 1e-30)
    origin = np.asarray(origin, dtype=np.float64)
    shape = np.asarray(field.shape)
    for point, radius, side in ends:
        point = np.asarray(point, dtype=np.float64)
        reach = 1.6 * float(radius) + 1.5 * h
        lo = np.clip(np.floor((point - reach - origin) / h - 0.5).astype(np.int64), 0, shape)
        hi = np.clip(np.ceil((point + reach - origin) / h - 0.5).astype(np.int64) + 1, 0, shape)
        if np.any(hi <= lo):
            continue
        axes = [_axis_centres(origin, h, shape[a], a)[lo[a] : hi[a]] for a in range(3)]
        rel = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1) - point
        along = rel @ up
        radial = np.linalg.norm(rel - along[..., None] * up, axis=-1)
        cut = np.where(radial <= reach, float(iso) - float(side) * along / h, np.inf)
        block = tuple(slice(lo[a], hi[a]) for a in range(3))
        field[block] = np.minimum(field[block], cut)
    return field


def face_components(n_verts, faces) -> np.ndarray:
    """Connected component label (smallest vertex id of the part) of every face."""
    faces = np.asarray(faces, dtype=np.int64)
    parent = np.arange(int(n_verts), dtype=np.int64)
    if faces.size == 0:
        return np.zeros(0, dtype=np.int64)
    u = np.repeat(faces[:, 0], faces.shape[1] - 1)
    v = faces[:, 1:].ravel()
    while True:
        pu, pv = parent[u], parent[v]
        diff = pu != pv
        if not diff.any():
            break
        np.minimum.at(parent, np.maximum(pu[diff], pv[diff]), np.minimum(pu[diff], pv[diff]))
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    return parent[faces[:, 0]]


def drop_minor_parts(verts, faces, keep=COMPONENT_KEEP) -> tuple:
    """Remove parts smaller than ``keep`` x the largest part (floating crumbs)."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] == 0:
        return verts[:0], faces
    labels = face_components(verts.shape[0], faces)
    sizes = np.bincount(labels, minlength=verts.shape[0])
    kept = sizes[labels] >= max(1.0, float(keep) * float(sizes.max()))
    faces = faces[kept]
    used = np.zeros(verts.shape[0], dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    return verts[used], remap[faces]


def keep_grounded(verts, faces, anchors) -> tuple:
    """Keep the parts that touch a support: nothing floats.

    ``anchors`` = dict ``points`` (S, 3) with ``radius`` and/or ``floor`` (a part
    reaching ``z <= floor`` stands on the ground). When no part touches (a
    degenerate setup) the input is returned unchanged.
    """
    if anchors is None or faces.shape[0] == 0:
        return verts, faces
    labels = face_components(verts.shape[0], faces)
    vertex_label = np.full(verts.shape[0], -1, dtype=np.int64)
    vertex_label[faces.ravel()] = np.repeat(labels, faces.shape[1])
    touching = np.zeros(verts.shape[0], dtype=bool)
    floor = anchors.get("floor")
    if floor is not None:
        touching |= verts[:, 2] <= float(floor)
    points = np.asarray(anchors.get("points", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
    radius = float(anchors.get("radius", 0.0))
    for point in points:
        touching |= np.linalg.norm(verts - point, axis=1) <= radius
    grounded = np.unique(vertex_label[touching & (vertex_label >= 0)])
    if grounded.size == 0:
        return verts, faces
    kept = np.isin(labels, grounded)
    if kept.all():
        return verts, faces
    faces = faces[kept]
    used = np.zeros(verts.shape[0], dtype=bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    return verts[used], remap[faces]


def structure_surface(raw, clipped, origin, h, iso, smooth, mesher, anchors=None,
                      planes=()) -> tuple:
    """Mesh of ``clipped`` at ``iso``: crumbs and floating parts removed, smoothed.

    Vertices on cut faces (where ``clipped`` is below the unclipped ``raw``
    field: flush contacts, flat footings) are pinned so the cuts stay planar;
    the ones within a voxel of a horizontal cut in ``planes`` (heights) are
    snapped onto it. ``anchors``: see :func:`keep_grounded`.
    """
    verts, faces = mesher.surface_nets(clipped, iso, origin, h)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    if faces.shape[0] == 0:
        return np.zeros((0, 3)), faces
    verts, faces = drop_minor_parts(verts, faces)
    verts, faces = keep_grounded(np.asarray(verts, dtype=np.float64), faces, anchors)
    pinned = None
    if raw is not None and faces.shape[0]:
        cut = np.asarray(mesher.sample_trilinear(raw - clipped, origin, h, verts), dtype=np.float64)
        pinned = cut > 0.04
        verts = np.array(verts, dtype=np.float64)
        for height in planes:
            if height is None:
                continue
            snap = pinned & (np.abs(verts[:, 2] - float(height)) < 0.75 * h)
            verts[snap, 2] = float(height)
        if not pinned.any() or pinned.all():
            pinned = None
    if smooth > 0 and faces.shape[0]:
        verts = mesher.taubin_smooth(verts, faces, iterations=int(smooth), pinned=pinned)
    return np.asarray(verts, dtype=np.float64), faces


def _colored(verts, faces, values, colormap, clip, mesher, signed=False) -> dict:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = color_range(values, signed, colormap, clip)
    return {
        "verts": verts,
        "faces": faces,
        "colors": mesher.colormap(values, lo, hi, name=colormap),
        "values": values,
    }


def _empty_geometry() -> dict:
    return {
        "verts": np.zeros((0, 3)),
        "faces": np.zeros((0, 4), dtype=np.int64),
        "colors": np.zeros((0, 4)),
        "values": np.zeros(0),
    }


def unreached_points(points, verts, distance) -> np.ndarray:
    """Indices of ``points`` farther than ``distance`` from every mesh vertex."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    if verts.shape[0] == 0:
        return np.arange(points.shape[0])
    far = []
    for index, point in enumerate(points):
        if float(np.min(np.linalg.norm(verts - point, axis=1))) > distance:
            far.append(index)
    return np.asarray(far, dtype=np.int64)


def _array_digest(*items) -> str:
    """Hash of arrays / plain values (cache signatures)."""
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for item in items:
        if isinstance(item, np.ndarray):
            digest.update(repr((item.dtype.str, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif item is None:
            digest.update(b"none")
        else:
            digest.update(repr(item).encode())
    return digest.hexdigest()


def problem_digest(problem, opt_settings) -> str:
    grid = problem.grid
    return _array_digest(
        "organic-1",
        grid.shape,
        grid.origin,
        grid.h,
        problem.domain,
        problem.fixed_dofs,
        problem.forces,
        problem.passive_solid,
        problem.passive_void,
        problem.bias,
        problem.E,
        problem.nu,
        repr(opt_settings),
    )


# --------------------------------------------------------------------- footprint
def _bilinear_mask(mask, origin2, h2, points) -> np.ndarray:
    """Bilinear sample of a 2D 0/1 mask at ``points (P, 2)`` (0 outside the mask grid)."""
    padded = np.pad(np.asarray(mask, dtype=np.float64), 1)
    u = (np.asarray(points, dtype=np.float64) - origin2) / h2 + 0.5
    i0 = np.floor(u).astype(np.int64)
    frac = u - i0
    out = np.zeros(u.shape[0], dtype=np.float64)
    nx, ny = padded.shape
    for di in (0, 1):
        for dj in (0, 1):
            ii = np.clip(i0[:, 0] + di, 0, nx - 1)
            jj = np.clip(i0[:, 1] + dj, 0, ny - 1)
            w = (frac[:, 0] if di else 1.0 - frac[:, 0]) * (frac[:, 1] if dj else 1.0 - frac[:, 1])
            out += w * padded[ii, jj]
    return out


def inset_contacts(points, radii, footprint, margin=1.08, n_ring=24) -> np.ndarray:
    """Slide contact points towards the footprint centre until their disc fits under it.

    ``footprint`` = dict ``mask`` (nx, ny) bool, ``origin`` (2,), ``h``: the
    object's plan (optionally ``under`` (P,) bool: which points sit on the
    object's underside; the others are left alone). A member ending at a corner
    or an edge of the object would otherwise stick out beside it; now its
    capital (``radius`` x ``margin``) lies fully under the object. Points move by
    at most three radii; points whose disc never fits stay put.
    """
    pts = np.array(points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(footprint["mask"], dtype=bool)
    if not mask.any():
        return pts
    under = footprint.get("under")
    under = np.ones(pts.shape[0], dtype=bool) if under is None else np.asarray(under, bool)
    origin2 = np.asarray(footprint["origin"], dtype=np.float64)[:2]
    h2 = float(footprint["h"])
    ii, jj = np.nonzero(mask)
    centre = origin2 + (np.stack([ii, jj], axis=1).mean(axis=0) + 0.5) * h2
    theta = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    for index in range(pts.shape[0]):
        if not under[index]:
            continue
        start = pts[index, :2].copy()
        radius = float(radii[index]) * margin
        towards = centre - start
        length = float(np.linalg.norm(towards))
        if length > 1e-12:
            towards /= length
        length = min(length, 3.0 * radius)
        steps = np.arange(0.0, length + 1e-9, 0.25 * h2) if length > 1e-12 else np.zeros(1)
        for t in steps:
            q = start + towards * t
            probe = np.vstack([q[None, :], q + radius * ring])
            if np.all(_bilinear_mask(mask, origin2, h2, probe) >= 0.5 - 1e-9):
                pts[index, :2] = q
                break
    return pts


# ------------------------------------------------------------------------ trees
def tree_settings(forms, params, radius_max):
    return forms.TreeSettings(
        n_trunks=int(params["footings"]),
        branching=int(params["branching"]),
        split_height=float(params["split"]),
        level_curve=float(params["levels"]),
        organic=float(params["organic"]),
        radius_min=float(params["min_thickness"]) * radius_max,
        radius_max=radius_max,
        footing_flare=float(params["footing_flare"]),
        capital_flare=float(params["capital_flare"]),
        samples=16,
    )


def taper_members(tree, end_ratio=MEMBER_TAPER, start=0.55) -> list:
    """Member radii along their curves, tapering into the branches they split into.

    A member ending at a split narrows (smoothstep over its last 45%) to
    ``end_ratio`` x its thickest child's radius: the fork grows out of the trunk
    instead of sitting on a shoulder.
    """
    children = {}
    for e, (a, _b) in enumerate(tree.edges):
        children.setdefault(int(a), []).append(e)
    radii = [np.array(r, dtype=np.float64) for r in tree.curve_radii]
    for e, (_a, b) in enumerate(tree.edges):
        kids = children.get(int(b))
        if tree.kind[b] != 1 or not kids:
            continue
        r = radii[e]
        target = min(float(r[-1]), end_ratio * max(float(radii[k][0]) for k in kids))
        s = np.linspace(0.0, 1.0, r.size)
        w = np.clip((s - start) / (1.0 - start), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)
        radii[e] = r * (1.0 - w) + target * w
    return radii


def leaf_radii(tree, points) -> np.ndarray:
    """Capital radius of the member reaching each of ``points`` (nearest leaf)."""
    leaf_edge = {int(b): e for e, (_a, b) in enumerate(tree.edges) if tree.kind[b] == 2}
    leaves = np.array(sorted(leaf_edge), dtype=np.int64)
    out = np.zeros(len(points))
    for i, point in enumerate(np.asarray(points, dtype=np.float64)):
        nearest = int(leaves[np.argmin(np.linalg.norm(tree.nodes[leaves] - point, axis=1))])
        out[i] = float(np.max(tree.curve_radii[leaf_edge[nearest]]))
    return out


def tree_ends(tree, radii) -> list:
    """``(point, radius, side)`` of every load tip (+1) and footing (-1) of a tree."""
    ends = []
    for e, (a, b) in enumerate(tree.edges):
        if tree.kind[b] == 2:
            ends.append((tree.nodes[b], float(radii[e][-1]), 1.0))
        if tree.kind[a] == 0:
            ends.append((tree.nodes[a], float(radii[e][0]), -1.0))
    return ends


def nearest_inside(domain, grid):
    """``constrain`` callable for ``grow_tree``: points outside ``domain`` move to the
    nearest domain element centre."""
    centres = grid.element_centers()[domain]
    shape = np.asarray(grid.shape)

    def constrain(points):
        pts = np.array(points, dtype=np.float64).reshape(-1, 3)
        if centres.shape[0] == 0:
            return pts
        ijk = np.floor((pts - grid.origin) / grid.h).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
        inside = np.zeros(pts.shape[0], dtype=bool)
        inside[ok] = domain[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        for i in np.flatnonzero(~inside):
            pts[i] = centres[int(np.argmin(np.linalg.norm(centres - pts[i], axis=1)))]
        return pts

    return constrain


def _cap_voxel(voxel, bmin, bmax, max_cells, warnings, what) -> float:
    extent = np.maximum(np.asarray(bmax) - np.asarray(bmin), voxel)
    cells = float(np.prod(np.ceil(extent / voxel) + 4))
    if cells > max_cells:
        voxel *= (cells / max_cells) ** (1.0 / 3.0)
        warnings.append(f"{what} was coarsened to stay within memory: lower Detail")
    return voxel


def branching_geometry(payload, progress, should_stop) -> dict:
    """Worker: form-found tree(s) -> tube field -> clipped, smoothed mesh."""
    modules = payload["modules"]
    forms, sculpt, mesher = modules["forms"], modules["sculpt"], modules["mesher"]
    job = payload["branching"]
    params = payload["params"]
    warnings = []
    points = np.asarray(job["points"], dtype=np.float64)
    forces = np.asarray(job["forces"], dtype=np.float64)
    settings = tree_settings(forms, job["tree"], job["radius_max"])
    supports = job["supports"]
    constrain = job.get("constrain")
    progress(0.02, "Growing the tree")

    def grow(pts):
        return forms.grow_tree(
            pts,
            forces,
            settings,
            supports=supports,
            ground=job["ground"],
            constrain=constrain,
            should_stop=should_stop,
        )

    try:
        tree = grow(points)
        footprint = job.get("footprint")
        if footprint is not None and points.shape[0] <= settings.max_leaves:
            inset = inset_contacts(points, leaf_radii(tree, points), footprint)
            if np.any(np.abs(inset - points) > 1e-9):
                points = inset
                tree = grow(points)
    except ValueError as exc:
        raise ValueError(f"No tree fits these weights and supports: {exc}") from exc
    warnings.extend(tree.warnings)
    check_stop(should_stop)

    radii = taper_members(tree)
    tubes = [
        sculpt.Tube(points=curve, radius=r, value=np.full(curve.shape[0], float(force)))
        for curve, r, force in zip(tree.curves, radii, tree.forces)
    ]
    thinnest = min(float(np.min(r)) for r in radii)
    voxel = thinnest / max(float(job["detail"]), 1e-3)
    all_points = np.concatenate([np.asarray(t.points) for t in tubes])
    reach = 3.0 * max(float(np.max(r)) for r in radii)
    voxel = _cap_voxel(
        voxel,
        all_points.min(axis=0) - reach,
        all_points.max(axis=0) + reach,
        SCULPT_MAX_CELLS,
        warnings,
        "The structure surface",
    )
    progress(0.08, "Shaping the members")
    field, origin, h, values = sculpt.tubes_field(
        tubes,
        voxel,
        blend=float(job["blend"]),
        progress=_sub_progress(progress, 0.08, 0.6),
        should_stop=should_stop,
    )
    field = np.asarray(field, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    check_stop(should_stop)
    progress(0.62, "Cutting the contacts")
    iso = sculpt.TUBE_ISO
    clipped = field.copy()
    if payload["mode"] == "SUPPORT":
        bases = tree.nodes[tree.kind == 0]
        floor = min(float(job["ground"]), float(bases[:, 2].min()) if bases.size else 0.0)
        clipped = clip_below(clipped, origin, h, floor, iso)
        clipped = end_cuts(clipped, origin, h, tree_ends(tree, radii), tree.up, iso)
        if payload.get("object") is not None:
            clipped = clip_with_indicator(clipped, origin, h, payload["object"], False, iso, sculpt)
    elif payload.get("object") is not None:
        clipped = clip_with_indicator(clipped, origin, h, payload["object"], True, iso, sculpt)
    check_stop(should_stop)
    progress(0.7, "Building the surface")
    planes = ()
    if payload["mode"] == "SUPPORT":
        tips = tree.nodes[tree.kind == 2]
        planes = (floor,) + tuple(np.unique(np.round(tips[:, 2], 9))) if abs(
            float(tree.up[2])) > 0.999 else (floor,)
    bases = tree.nodes[tree.kind == 0]
    base_radius = max((float(r[0]) for r in radii), default=h)
    anchors = {"points": bases, "radius": 3.0 * base_radius + 2.0 * h}
    if payload["mode"] == "SUPPORT":
        anchors["floor"] = floor + 0.75 * h
    verts, faces = structure_surface(
        field, clipped, origin, h, iso, params["smooth"], mesher, anchors, planes
    )
    del field, clipped
    if faces.shape[0] == 0:
        raise ValueError("The structure came out empty: raise Thickness or Detail")
    progress(0.92, "Coloring")
    vertex_values = sample_field(values, values > 0.0, _GridLike(origin, h), verts, mesher)
    geometry = _colored(verts, faces, vertex_values, params["colormap"], params["clip"], mesher)
    stress = tree.forces / (math.pi * np.maximum(tree.radii, 1e-9) ** 2)
    progress(1.0, "Done")
    return {
        "geometry": geometry,
        "warnings": warnings,
        "contacts": points,
        "footings": tree.nodes[tree.kind == 0],
        "thinnest": thinnest,
        "metrics": {
            "max_stress": float(np.max(stress)) if stress.size else 0.0,
            "members": int(tree.edges.shape[0]),
            "footings": int((tree.kind == 0).sum()),
            "total_load": float(tree.total_load),
        },
    }


# ---------------------------------------------------------------------- organic
def structure_anchors(payload, h) -> dict:
    """Where the structure may touch its supports (see :func:`keep_grounded`)."""
    pad = float(payload.get("pad") or 0.0)
    footings = payload.get("footings")
    points = np.zeros((0, 3)) if footings is None else np.asarray(footings, dtype=np.float64)
    radius = max(2.0 * pad, 3.0 * float(h))
    if payload["mode"] == "SUPPORT":
        return {"points": points, "radius": radius, "floor": float(payload["floor"]) + 0.75 * h}
    return {"points": points, "radius": radius}


def support_planes(payload) -> tuple:
    """Heights of the flat cuts of a SUPPORT structure (floor and top)."""
    if payload["mode"] != "SUPPORT":
        return ()
    return (float(payload["floor"]), payload.get("top"))


def support_clips(field, origin, h, iso, payload, sculpt) -> np.ndarray:
    """Copy of ``field`` cut to the room the structure may use.

    SUPPORT: nothing below the floor, nothing above the highest contact and
    nothing inside the carried object (flush contacts). CARVE: nothing outside
    the block.
    """
    clipped = np.array(field, dtype=np.float64, copy=True)
    obj = payload.get("object")
    if payload["mode"] == "SUPPORT":
        clipped = clip_below(clipped, origin, h, float(payload["floor"]), iso)
        top = payload.get("top")
        if top is not None:  # mirror image of clip_below: cut above the plane
            z = _axis_centres(origin, h, clipped.shape[2], 2)
            np.minimum(clipped, (float(iso) - (z - float(top)) / float(h))[None, None, :],
                       out=clipped)
        if obj is not None:
            clipped = clip_with_indicator(clipped, origin, h, obj, False, iso, sculpt)
    elif obj is not None:
        clipped = clip_with_indicator(clipped, origin, h, obj, True, iso, sculpt)
    return clipped


def _box_filter(a, op, passes) -> np.ndarray:
    """``passes`` separable 3x3x3 min / max filters (grey erosion / dilation)."""
    out = np.asarray(a, dtype=np.float64)
    for _ in range(int(passes)):
        for axis in range(3):
            n = out.shape[axis]
            if n < 2:
                continue
            lower = np.take(out, np.r_[0, np.arange(n - 1)], axis=axis)
            upper = np.take(out, np.r_[np.arange(1, n), n - 1], axis=axis)
            out = op(out, op(lower, upper))
    return out


def organic_field(state, problem, mode, upsample=ORGANIC_UPSAMPLE, sculpt=None) -> tuple:
    """Optimized density on a finer grid, padded for flush cuts.

    SUPPORT: zeros around the design box except below the ground, where the
    bottom layer continues (cut flat later). CARVE: borders replicated (the block
    clip trims them flush with the block faces). Returns ``(field, origin, h)``.
    """
    grid = problem.grid
    density = np.where(problem.domain, np.asarray(state.density, dtype=np.float64), 0.0)
    if mode == "SUPPORT":
        padded = np.pad(density, 1)
        padded[1:-1, 1:-1, 0] = density[:, :, 0]
    else:
        padded = np.pad(density, 1, mode="edge")
    origin = grid.origin - grid.h
    if upsample <= 1 or sculpt is None:
        return padded, origin, grid.h
    fine_h = grid.h / float(upsample)
    shape = tuple(int(n) * int(upsample) for n in padded.shape)
    fine = sculpt.resample_to(padded, origin, grid.h, shape, origin, fine_h)
    if ORGANIC_OPENING > 0:
        fine = _box_filter(_box_filter(fine, np.minimum, ORGANIC_OPENING), np.maximum,
                           ORGANIC_OPENING)
    for _ in range(max(0, int(ORGANIC_BLUR))):
        fine = _blur121(fine)
    return fine, origin, fine_h


def organic_geometry(state, problem, payload, smooth, progress=None, should_stop=None) -> dict:
    """Final mesh of an optimized density: finer grid, flush cuts, smoothing, colors."""
    modules = payload["modules"]
    sculpt, mesher = modules["sculpt"], modules["mesher"]
    params = payload["params"]
    iso = float(params["iso"])
    mode = payload["mode"]
    field, origin, h = organic_field(state, problem, mode, ORGANIC_UPSAMPLE, sculpt)
    check_stop(should_stop)
    clipped = support_clips(field, origin, h, iso, payload, sculpt)
    check_stop(should_stop)
    verts, faces = structure_surface(
        field,
        clipped,
        origin,
        h,
        iso,
        smooth,
        mesher,
        structure_anchors(payload, h),
        support_planes(payload),
    )
    if faces.shape[0] == 0:
        return _empty_geometry()
    grid = problem.grid
    density = np.where(problem.domain, np.asarray(state.density, dtype=np.float64), 0.0)
    solid = (density >= iso) & problem.domain
    if not solid.any():
        solid = problem.domain
    von_mises = np.asarray(state.von_mises, dtype=np.float64)
    values = sample_field(von_mises, solid, grid, verts, mesher)
    lo, hi = color_range(von_mises[solid], False, params["colormap"], params["clip"])
    return {
        "verts": verts,
        "faces": faces,
        "colors": mesher.colormap(values, lo, hi, name=params["colormap"]),
        "values": values,
    }


def organic_smoothing(payload) -> int:
    """Taubin iterations for an organic mesh: the finer meshing grid needs ~1.5x."""
    return int(round(1.5 * int(payload["params"]["smooth"])))


def organic_preview(state, problem, payload) -> dict:
    """Quick live-preview mesh of an optimization state (coarse grid, light smoothing)."""
    return evolved_geometry(
        state, problem, payload["params"], payload["modules"]["mesher"], 2, clean=False
    )


# ------------------------------------------------------------------------- ribs
def ribs_supported(streamlines) -> bool:
    """Whether ``core/streamlines.py`` can trace lines over a surface."""
    try:
        names = {f.name for f in dataclasses.fields(streamlines.TraceSettings)}
    except TypeError:
        return False
    return "surface" in names and hasattr(streamlines, "SurfaceConstraint")


def trim_dangling(lines, anchors, connect, anchor_reach, rounds=3, join=0.0) -> list:
    """Clean rib lattice: every rib runs between two connections.

    A rib point is *connected* where another rib passes within ``connect`` or a
    pad (``anchors``) within ``anchor_reach``. Open ribs are trimmed to the span
    between their first and last connection, ribs (open or closed) with fewer
    than two connections are dropped; repeated ``rounds`` times because removing
    a rib can strand others. No spikes, no floating loops.
    """
    lines = [line for line in lines if len(line.points) >= 2]
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
    for _ in range(int(rounds)):
        if not lines:
            break
        index = _PointHash(connect)
        for k, line in enumerate(lines):
            index.add(line.points, [k] * len(line.points))
        changed = False
        kept = []
        for k, line in enumerate(lines):
            points = np.asarray(line.points)
            gap = np.full(points.shape[0], np.inf)  # distance to the connection, scaled
            partner = [None] * points.shape[0]
            for i, point in enumerate(points):
                distance, other = index.nearest(point, connect, skip=lambda tag, k=k: tag == k)
                gap[i] = distance / connect
                partner[i] = other
            if anchors.size:
                dist = np.min(
                    np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2), axis=1
                )
                on_pad = dist <= anchor_reach
                gap = np.where(on_pad, 0.0, gap)
                for i in np.flatnonzero(on_pad):
                    partner[i] = None
            hits = np.flatnonzero(np.isfinite(gap))
            # contiguous runs count once: a crossing touches a few consecutive points
            breaks = np.flatnonzero(np.diff(hits) > 1)
            runs = int(breaks.size + 1) if hits.size else 0
            if runs < 2:
                changed = True
                continue
            if getattr(line, "closed", False):
                kept.append(line)
                continue
            # end each rib where it is closest to its first / last connection (no stubs)
            first = hits[: breaks[0] + 1]
            last = hits[breaks[-1] + 1 :]
            a = int(first[np.argmin(gap[first])])
            b = int(last[np.argmin(gap[last])])
            if gap[0] == 0.0:  # starts on a pad
                a = 0
            if gap[-1] == 0.0:
                b = points.shape[0] - 1
            pts = points[a : b + 1]
            mag = np.asarray(line.magnitude)[a : b + 1]
            # frame into the rib it meets: the end reaches that rib's axis, the jog is
            # smoothed into a curve
            joined = False
            if partner[a] is not None and float(np.linalg.norm(partner[a] - pts[0])) > join:
                pts = np.vstack([partner[a][None, :], pts])
                mag = np.concatenate([mag[:1], mag])
                joined = True
            if partner[b] is not None and float(np.linalg.norm(partner[b] - pts[-1])) > join:
                pts = np.vstack([pts, partner[b][None, :]])
                mag = np.concatenate([mag, mag[-1:]])
                joined = True
            if joined:
                pts = _smooth_polyline(pts, False, passes=6)
            if pts.shape[0] != points.shape[0] or a > 0:
                changed = True
            line = dataclasses.replace(
                line, points=pts, magnitude=mag, sign=np.full(pts.shape[0], line.sign[0])
            )
            kept.append(line)
        lines = kept
        if not changed:
            break
    return lines


def connector_ribs(streamlines, lines, loads, footings, field, tensors, iso, step, connect,
                   reach, min_stress, floor=None, max_steps=800) -> list:
    """Ribs from every weight the lattice does not reach, down its stress flow.

    For each load pad without a rib within ``reach``, both families are traced
    from the pad (separation ignored; all pads at once, vectorized) and the half
    that first meets the lattice (a rib within ``connect``) or a footing wins: it
    runs from the pad to that meeting point and frames into it. Weights whose
    flow meets nothing stay unreached (reported by the operator).
    """
    loads = np.asarray(loads, dtype=np.float64).reshape(-1, 3)
    footings = np.asarray(footings, dtype=np.float64).reshape(-1, 3)
    index = _PointHash(connect)
    for line in lines:
        index.add(np.asarray(line.points), [0] * len(line.points))
    missing = [pad for pad in loads if not index.near(pad, reach)]
    if not missing:
        return []
    start, ok, normal = field.project(np.asarray(missing), iso, max_move=8.0)
    start, normal = start[ok], normal[ok]
    if start.shape[0] == 0:
        return []
    best = [None] * start.shape[0]
    for family in ("COMPRESSION", "TENSION"):
        _d, value, _g = tangent_principal(tensors.sample_tensor(start), normal, family)
        low = min_stress * max(float(np.max(np.abs(value))), 1e-30)
        halves = _trace_halves(field, tensors, start, family, iso, step, low, max_steps, floor)
        for i, half in enumerate(halves):
            if half is None:
                continue
            points, mags, centre = half
            for direction in (1, -1):
                idx = np.arange(centre, points.shape[0]) if direction == 1 else np.arange(
                    centre, -1, -1)
                for count, k in enumerate(idx):
                    if count * step < reach:
                        continue
                    point = points[k]
                    _distance, partner = index.nearest(point, connect)
                    near_foot = footings.size and float(
                        np.min(np.linalg.norm(footings - point, axis=1))) <= reach
                    if partner is None and not near_foot:
                        continue
                    span = idx[: count + 1]
                    pts = points[span]
                    mag = np.asarray(mags)[span]
                    if partner is not None:
                        pts = np.vstack([pts, partner[None, :]])
                        mag = np.append(mag, mag[-1])
                    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                    if best[i] is None or length < best[i][0]:
                        best[i] = (length, pts, mag, family)
                    break
    added = []
    for entry in best:
        if entry is None:
            continue
        _length, pts, mag, family = entry
        pts = _smooth_polyline(pts, False, passes=6)
        added.append(
            streamlines.Polyline(
                points=pts,
                magnitude=mag,
                family=family,
                generation=0,
                sign=np.full(pts.shape[0], 1.0 if family == "TENSION" else -1.0),
            )
        )
        index.add(pts, [0] * pts.shape[0])
    return added


def rib_tubes(lines, h, thickness, min_thickness, sculpt, anchors=None, anchor_radius=0.0) -> list:
    """Tubes along the rib lines: radius ~ sqrt(|sigma|), tapered open ends.

    Ends within ``anchor_radius`` of an ``anchors`` point (a load or support pad)
    keep their full radius so they land on the pad.
    """
    hi = float(thickness) * h
    lo = float(min_thickness) * hi
    magnitudes = [np.abs(np.asarray(line.magnitude, dtype=np.float64)) for line in lines]
    flat = np.concatenate(magnitudes) if magnitudes else np.zeros(0)
    flat = flat[np.isfinite(flat)]
    ref = float(np.percentile(flat, 99.0)) if flat.size else 1.0
    ref = ref if ref > 0.0 else 1.0
    tubes = []
    for line, magnitude in zip(lines, magnitudes):
        points = np.asarray(line.points, dtype=np.float64)
        if points.shape[0] < 2:
            continue
        radius = lo + (hi - lo) * np.sqrt(np.clip(magnitude / ref, 0.0, 1.0))
        if not getattr(line, "closed", False):
            seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            total = max(float(s[-1]), 1e-12)
            span = RIB_END_SPAN * total
            w0 = np.clip(s / max(span, 1e-12), 0.0, 1.0)
            w1 = np.clip((total - s) / max(span, 1e-12), 0.0, 1.0)
            if anchors is not None and len(anchors):
                pads = np.asarray(anchors, dtype=np.float64).reshape(-1, 3)
                if np.min(np.linalg.norm(pads - points[0], axis=1)) <= anchor_radius:
                    w0 = np.ones_like(w0)
                if np.min(np.linalg.norm(pads - points[-1], axis=1)) <= anchor_radius:
                    w1 = np.ones_like(w1)
            w = np.minimum(w0, w1)
            radius = radius * (RIB_END_TAPER + (1.0 - RIB_END_TAPER) * w * w * (3.0 - 2.0 * w))
        tubes.append(sculpt.Tube(points=points, radius=radius, value=magnitude))
    return tubes


def skin_field(surface, iso, thickness, shape, origin, h, sculpt) -> np.ndarray:
    """Shell of ``thickness`` (world) centred on the ``iso`` surface of ``surface``.

    ``surface`` is an indicator dict (``field``, ``origin``, ``h``). Near the
    surface ``|f - iso|`` grows like ``|grad f| d`` (distance ``d``), so
    ``skin = 2 ISO (1 - |f - iso| / (|grad f| thickness))`` reaches the tube iso
    level at ``d = thickness / 2``: a shell of the requested thickness, meshed
    with the tubes at ``sculpt.TUBE_ISO``. ``|grad f|`` is the median gradient on
    the surface.
    """
    f = np.asarray(surface["field"], dtype=np.float64)
    gx, gy, gz = np.gradient(f, float(surface["h"]))
    near = np.abs(f - iso) < 0.15
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)
    g = float(np.median(grad[near])) if near.any() else 1.0 / float(surface["h"])
    values = indicator_on_grid(
        {"field": f, "origin": surface["origin"], "h": surface["h"]}, shape, origin, h, sculpt
    )
    width = max(g * float(thickness), 1e-9)
    return np.maximum(2.0 * sculpt.TUBE_ISO * (1.0 - np.abs(values - iso) / width), 0.0)


class SurfaceField:
    """Trilinear scalar field (element-centred) with its analytic gradient (worker-safe)."""

    def __init__(self, field, origin, h):
        self.field = np.asarray(field, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.h = float(h)
        nx, ny, nz = self.field.shape
        # one replicated layer on the upper side: the 8 corners of a sample always exist
        self._table = np.pad(self.field, ((0, 1), (0, 1), (0, 1)), mode="edge").ravel()
        sy = nz + 1
        sx = (ny + 1) * sy
        self._strides = np.array([sx, sy, 1], dtype=np.int64)
        self._offsets = np.array(
            [a * sx + b * sy + c for a in (0, 1) for b in (0, 1) for c in (0, 1)],
            dtype=np.int64,
        )
        self._top = np.asarray(self.field.shape, dtype=np.float64) - 1.0

    def sample(self, points) -> tuple:
        """``(value (P,), gradient (P, 3))`` at world points (clamped to the grid)."""
        u = (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.origin) / self.h - 0.5
        u = np.clip(u, 0.0, self._top)
        i0 = np.floor(u)
        t = u - i0
        c = self._table[(i0.astype(np.int64) @ self._strides)[:, None] + self._offsets]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        sx, sy, sz = 1.0 - tx, 1.0 - ty, 1.0 - tz
        # c[:, 4a + 2b + d] is the corner (i + a, j + b, k + d)
        z00 = sz * c[:, 0] + tz * c[:, 1]
        z01 = sz * c[:, 2] + tz * c[:, 3]
        z10 = sz * c[:, 4] + tz * c[:, 5]
        z11 = sz * c[:, 6] + tz * c[:, 7]
        y0 = sy * z00 + ty * z01
        y1 = sy * z10 + ty * z11
        value = sx * y0 + tx * y1
        gx = y1 - y0
        gy = sx * (z01 - z00) + tx * (z11 - z10)
        dz00 = c[:, 1] - c[:, 0]
        dz01 = c[:, 3] - c[:, 2]
        dz10 = c[:, 5] - c[:, 4]
        dz11 = c[:, 7] - c[:, 6]
        gz = sx * (sy * dz00 + ty * dz01) + tx * (sy * dz10 + ty * dz11)
        return value, np.stack([gx, gy, gz], axis=1) / self.h

    def project(self, points, iso, iterations=3, max_move=2.0) -> tuple:
        """Newton steps onto the ``iso`` level: ``(points, ok, normals)``.

        ``ok``: converged within 5% of a voxel and moved less than ``max_move``
        voxels; ``normals`` unit gradients (pointing to growing field values).
        """
        start = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        p = start.copy()
        iterations = max(int(iterations), int(np.ceil(max_move / 0.75)))
        for _ in range(iterations):
            value, grad = self.sample(p)
            g2 = np.einsum("ij,ij->i", grad, grad)
            step = np.divide(value - iso, g2, out=np.zeros_like(g2), where=g2 > 1e-18)
            delta = step[:, None] * grad
            length = np.linalg.norm(delta, axis=1)
            scale = np.minimum(1.0, (0.75 * self.h) / np.maximum(length, 1e-30))
            p -= delta * scale[:, None]
        value, grad = self.sample(p)
        gnorm = np.linalg.norm(grad, axis=1)
        distance = np.abs(value - iso) / np.maximum(gnorm, 1e-30)
        ok = (gnorm > 1e-9) & (distance < 0.05 * self.h)
        ok &= np.linalg.norm(p - start, axis=1) < float(max_move) * self.h
        normals = grad / np.maximum(gnorm, 1e-30)[:, None]
        return p, ok, normals


def tangent_principal(tensors, normals, family) -> tuple:
    """Principal direction of ``family`` in the tangent plane: ``(dirs, magnitude, gap)``.

    The tensor is projected onto the plane ``P T P`` (``P = I - n n^T``); of its
    eigenvectors the one closest to the normal is discarded; TENSION takes the
    larger remaining eigenvalue, COMPRESSION the smaller. ``gap`` is their
    difference relative to the largest magnitude (small = isotropic, direction
    undefined).
    """
    n = np.asarray(normals, dtype=np.float64)
    proj = np.eye(3)[None, :, :] - n[:, :, None] * n[:, None, :]
    tt = proj @ np.asarray(tensors, dtype=np.float64) @ proj
    w, v = np.linalg.eigh(tt)
    along = np.abs(np.einsum("pi,pij->pj", n, v))
    drop = np.argmax(along, axis=1)
    order = np.argsort(np.where(np.arange(3)[None, :] == drop[:, None], np.inf, w), axis=1)
    lo, hi = order[:, 0], order[:, 1]
    rows = np.arange(w.shape[0])
    pick = hi if family == "TENSION" else lo
    dirs = v[rows, :, pick]
    values = w[rows, pick]
    scale = np.maximum(np.max(np.abs(w), axis=1), 1e-30)
    gap = np.abs(w[rows, hi] - w[rows, lo]) / scale
    return dirs, values, gap


class _PointHash:
    """Spatial hash of polyline points (cell = ``cell`` world units)."""

    def __init__(self, cell):
        self.cell = float(cell)
        self.cells = {}

    def _key(self, p):
        return tuple(np.floor(np.asarray(p) / self.cell).astype(np.int64).tolist())

    def add(self, points, tags):
        for point, tag in zip(np.asarray(points), tags):
            self.cells.setdefault(self._key(point), []).append((point, tag))

    def nearest(self, point, radius, skip=None) -> tuple:
        """``(distance, point)`` of the nearest stored point within ``radius``
        (``(inf, None)`` if none)."""
        key = self._key(point)
        best, where = np.inf, None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = self.cells.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    if not bucket:
                        continue
                    for other, tag in bucket:
                        if skip is not None and skip(tag):
                            continue
                        d = other - point
                        d2 = float(d @ d)
                        if d2 < best:
                            best, where = d2, other
        best = float(np.sqrt(best))
        return (best, where) if best <= radius else (np.inf, None)

    def near(self, point, radius, skip=None) -> bool:
        """Any stored point within ``radius`` (entries with tag ``skip(tag)`` true ignored)."""
        key = self._key(point)
        r2 = radius * radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = self.cells.get((key[0] + dx, key[1] + dy, key[2] + dz))
                    if not bucket:
                        continue
                    for other, tag in bucket:
                        if skip is not None and skip(tag):
                            continue
                        d = other - point
                        if float(d @ d) <= r2:
                            return True
        return False


def surface_stress_lines(streamlines, surface, surface_grid, iso, tensors, options, candidates,
                         progress=None, should_stop=None, max_lines=400, batch=96) -> list:
    """Principal stress lines over the ``iso`` surface of ``surface`` (fallback tracer).

    Used when ``core/streamlines.py`` has no surface mode. Same idea as that
    mode: lines follow the principal directions of the stress tensor (``tensors``,
    a ``streamlines.StressField``) projected onto the tangent plane
    (:func:`tangent_principal`); every RK2 step is projected back onto the
    surface (Newton on the trilinear field). A line stops where the stress drops
    below ``min_stress`` x the family's 99th percentile on the surface, where the
    direction is undefined or turns by more than 60 degrees, below
    ``options["floor"]``, and where it comes closer than ``separation`` voxels (of
    ``surface_grid``) to an accepted line of its family or to its own earlier part
    (no curls; a line meeting its own start closes into a hoop). Accepted lines
    are smoothed and projected again; fragments shorter than 2.5 separations are
    dropped. Seeds: ``seed_points`` (projected) first, then ``candidates``
    (surface points) drawn with probability ~ stress. Candidates are integrated in
    vectorized batches and accepted one by one. Returns ``streamlines.Polyline``.
    """
    field = SurfaceField(surface, surface_grid.origin, surface_grid.h)
    h = surface_grid.h
    step = min(float(options.get("step", 0.5)), 0.5 * float(options["separation"])) * h
    separation = float(options["separation"]) * h
    floor = options.get("floor")
    max_steps = int(options.get("max_steps", 1200))
    rng = np.random.default_rng(int(options.get("rng_seed", 0)))
    cand_p, cand_ok, cand_n = field.project(np.asarray(candidates).reshape(-1, 3), iso)
    candidates, cand_n = cand_p[cand_ok], cand_n[cand_ok]
    out = []
    families = tuple(options["families"])
    total = max(len(families), 1)
    for f_index, family in enumerate(families):
        if candidates.shape[0] == 0:
            break
        _dirs, values, _gap = tangent_principal(
            tensors.sample_tensor(candidates), cand_n, family
        )
        magnitude = np.abs(values)
        finite = magnitude[np.isfinite(magnitude)]
        ref = float(np.percentile(finite, 99.0)) if finite.size else 0.0
        if not ref > 0.0:
            continue
        low = float(options.get("min_stress", 0.02)) * ref
        weights = np.where(magnitude > low, magnitude, 0.0)
        if not weights.any():
            continue
        order = rng.choice(
            candidates.shape[0],
            size=min(int(options.get("n_seeds", 60)) * 8, int((weights > 0).sum())),
            replace=False,
            p=weights / weights.sum(),
        )
        seeds = candidates[order]
        explicit = options.get("seed_points")
        if explicit is not None and len(explicit):
            ep, eok, _n = field.project(
                np.asarray(explicit, dtype=np.float64), iso, max_move=8.0
            )
            seeds = np.vstack([ep[eok], seeds])
        accepted = _PointHash(separation)
        n_accepted = 0
        cursor = 0
        while cursor < seeds.shape[0] and n_accepted < max_lines:
            check_stop(should_stop)
            chunk = []
            while cursor < seeds.shape[0] and len(chunk) < batch:
                seed = seeds[cursor]
                cursor += 1
                if not accepted.near(seed, separation):
                    chunk.append(seed)
            if not chunk:
                break
            halves = _trace_halves(field, tensors, np.asarray(chunk), family, iso, step, low,
                                   max_steps, floor)
            for candidate in halves:
                if n_accepted >= max_lines:
                    break
                line = _accept_line(candidate, accepted, separation, field, iso)
                if line is None:
                    continue
                pts, mag, closed = line
                out.append(
                    streamlines.Polyline(
                        points=pts,
                        magnitude=mag,
                        family=family,
                        generation=0,
                        sign=np.full(pts.shape[0], 1.0 if family == "TENSION" else -1.0),
                        closed=closed,
                    )
                )
                n_accepted += 1
            if progress is not None:
                progress(
                    (f_index + min(1.0, cursor / max(seeds.shape[0], 1))) / total,
                    f"Tracing ribs: {len(out)} line(s)",
                )
    return out


def _trace_halves(field, tensors, seeds, family, iso, step, low, max_steps, floor=None) -> list:
    """Integrate both halves of a line from every seed (vectorized); per seed
    ``(points (n, 3), magnitudes (n,), seed index)`` or ``None``."""
    n = seeds.shape[0]
    lo = field.origin + 1.0 * field.h
    hi = field.origin + (np.asarray(field.field.shape) - 1.0) * field.h
    p0, ok0, normals = field.project(seeds, iso)
    d0, v0, _g = tangent_principal(tensors.sample_tensor(p0), normals, family)
    p = np.concatenate([p0, p0])
    d_prev = np.concatenate([d0, -d0])
    active = np.concatenate([ok0, ok0]) & np.concatenate([np.abs(v0) >= low] * 2)
    track = [p.copy()]
    mags = [np.concatenate([np.abs(v0), np.abs(v0)])]
    alive = active.copy()
    for _ in range(int(max_steps)):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        q = p[idx]
        prev = d_prev[idx]
        # RK2 (midpoint) on the surface: every point is projected back onto it
        _value, grad = field.sample(q)
        n1 = grad / np.maximum(np.linalg.norm(grad, axis=1), 1e-30)[:, None]
        d1, _v1, _g1 = tangent_principal(tensors.sample_tensor(q), n1, family)
        d1 *= np.where(np.einsum("ij,ij->i", d1, prev) < 0.0, -1.0, 1.0)[:, None]
        mid, okm, nm = field.project(q + 0.5 * step * d1, iso, iterations=2, max_move=1.0)
        d2, _v2, g2 = tangent_principal(tensors.sample_tensor(mid), nm, family)
        d2 *= np.where(np.einsum("ij,ij->i", d2, d1) < 0.0, -1.0, 1.0)[:, None]
        new, okn, nn = field.project(q + step * d2, iso, iterations=2, max_move=1.0)
        _d3, v3, g3 = tangent_principal(tensors.sample_tensor(new), nn, family)
        turn = np.einsum("ij,ij->i", d2, prev)
        good = okm & okn & (np.abs(v3) >= low) & (turn > 0.5) & (g2 > 1e-2) & (g3 > 1e-2)
        good &= np.linalg.norm(new - q, axis=1) > 0.25 * step
        good &= np.all((new >= lo) & (new <= hi), axis=1)  # the clamped border is no surface
        if floor is not None:
            good &= new[:, 2] >= float(floor)
        alive[idx[~good]] = False
        go = idx[good]
        p[go] = new[good]
        d_prev[go] = d2[good]
        record = np.full(p.shape, np.nan)
        record[go] = p[go]
        track.append(record)
        m = np.full(p.shape[0], np.nan)
        m[go] = np.abs(v3[good])
        mags.append(m)
    track = np.stack(track)  # (steps + 1, 2n, 3)
    mags = np.stack(mags)
    out = []
    for i in range(n):
        if not active[i]:
            out.append(None)
            continue
        fwd = track[:, i][np.isfinite(track[:, i, 0])]
        bwd = track[:, n + i][np.isfinite(track[:, n + i, 0])]
        fm = mags[:, i][np.isfinite(mags[:, i])]
        bm = mags[:, n + i][np.isfinite(mags[:, n + i])]
        points = np.vstack([bwd[::-1], fwd[1:]])
        values = np.concatenate([bm[::-1], fm[1:]])
        out.append((points, values, bwd.shape[0] - 1))
    return out


def _smooth_polyline(points, closed, passes=4) -> np.ndarray:
    """[1, 2, 1] smoothing of a polyline (ends fixed unless closed)."""
    pts = np.array(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts
    for _ in range(int(passes)):
        if closed:
            body = pts[:-1]
            body = 0.5 * body + 0.25 * (np.roll(body, 1, axis=0) + np.roll(body, -1, axis=0))
            pts = np.vstack([body, body[:1]])
        else:
            pts[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
    return pts


def _accept_line(candidate, accepted, separation, field, iso) -> tuple:
    """Clip a candidate where it comes too close to accepted lines or to itself,
    smooth it onto the surface and keep it when it is long enough.

    ``candidate`` = ``(points, magnitudes, seed index)``; scans from the seed
    outwards. Accepted points go into ``accepted``. Returns ``(points,
    magnitudes, closed)`` or ``None``.
    """
    if candidate is None:
        return None
    points, mags, centre = candidate
    if points.shape[0] < 3:
        return None
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    own = _PointHash(separation)
    window = 2.0 * separation
    keep_lo = keep_hi = centre
    closed = False
    for direction in (1, -1):
        k = centre if direction == 1 else centre - 1
        while 0 <= k < points.shape[0]:
            point = points[k]
            here = arc[k]
            if accepted.near(point, separation):
                break
            if own.near(point, separation, skip=lambda tag, here=here: abs(tag - here) < window):
                # back at its own start: a hoop (when long enough)
                if direction == 1 and here - arc[centre] > 1.5 * separation:
                    first = points[centre]
                    closed = float(np.linalg.norm(point - first)) < 1.2 * separation
                break
            own.add(point[None, :], [here])
            if direction == 1:
                keep_hi = k
            else:
                keep_lo = k
            k += direction
        if closed:
            break
    if closed:
        keep_lo = centre
    pts = points[keep_lo : keep_hi + 1]
    mag = mags[keep_lo : keep_hi + 1]
    if pts.shape[0] < 3:
        return None
    length = float(arc[keep_hi] - arc[keep_lo])
    if not closed and length < 2.0 * separation:
        return None
    if closed:
        pts = np.vstack([pts, pts[:1]])
        mag = np.append(mag, mag[0])
    smooth = _smooth_polyline(pts, closed, passes=8)
    projected, ok, _n = field.project(smooth, iso)
    pts = np.where(ok[:, None], projected, smooth)
    accepted.add(pts, [0] * pts.shape[0])
    return pts, mag, closed


def rib_surface(surface, grid, mode, sculpt, upsample=ORGANIC_UPSAMPLE) -> tuple:
    """Smooth, finer copy of a rib surface field (padded so the surface closes).

    Two empty layers around the grid (SUPPORT: the bottom continues below the
    ground instead), upsampled like the organic mesh and blurred. Returns
    ``(field, grid)``.
    """
    pad = 2
    f = np.pad(np.asarray(surface, dtype=np.float64), pad)
    if mode == "SUPPORT":
        f[pad:-pad, pad:-pad, :pad] = np.asarray(surface)[:, :, :1]
    origin = grid.origin - pad * grid.h
    fine_h = grid.h / float(upsample)
    shape = tuple(int(n) * int(upsample) for n in f.shape)
    fine = sculpt.resample_to(f, origin, grid.h, shape, origin, fine_h)
    for _ in range(max(0, int(ORGANIC_BLUR))):
        fine = _blur121(fine)
    return fine, Grid(shape, origin, fine_h)


def ribs_geometry(payload, surface, stress, grid, iso, progress, should_stop) -> dict:
    """Worker: principal stress lines traced over ``surface`` -> fused rib lattice.

    ``surface`` is the element-centred scalar field whose ``iso`` level is the
    shell the ribs follow (optimized density, or the filtered block indicator),
    ``stress`` the (nx, ny, nz, 6) stress on the same ``grid``.
    """
    modules = payload["modules"]
    streamlines, sculpt, mesher = modules["streamlines"], modules["sculpt"], modules["mesher"]
    job = payload["ribs"]
    params = payload["params"]
    warnings = []
    surface = np.asarray(surface, dtype=np.float64)
    seeds = job["seeds"]
    options = {
        "families": job["families"],
        "n_seeds": int(job["n_seeds"]),
        "seed_points": None if seeds is None or len(seeds) == 0 else np.asarray(seeds),
        "step": float(job["step"]),
        "min_stress": float(job["min_stress"]),
        "separation": float(job["spacing"]),
        "branch_depth": 0,
        "rng_seed": int(job["rng_seed"]),
    }
    trace_progress = _sub_progress(progress, 0.0, 0.45)
    fine, fine_grid = rib_surface(surface, grid, payload["mode"], sculpt)
    tensors = streamlines.StressField(
        grid=grid, stress=np.asarray(stress), solid=_dilate6(surface >= iso)
    )
    floor = float(payload["floor"]) + 0.5 * grid.h if payload["mode"] == "SUPPORT" else None
    if ribs_supported(streamlines):
        # two empty layers around the grid: the surface closes inside it (a block that
        # fills its grid has no iso crossing otherwise)
        padded = np.pad(surface, 2)
        padded_grid = Grid(padded.shape, grid.origin - 2.0 * grid.h, grid.h)
        field = streamlines.StressField(
            grid=padded_grid,
            stress=np.pad(np.asarray(stress), ((2, 2), (2, 2), (2, 2), (0, 0))),
            solid=_dilate6(padded >= iso),
        )
        trace_settings = streamlines.TraceSettings(
            surface=streamlines.SurfaceConstraint(field=padded, iso=float(iso)), **options
        )
        lines = list(
            streamlines.trace(
                field, trace_settings, progress=trace_progress, should_stop=should_stop
            )
        )
    else:  # this core has no surface tracing: the add-on's own surface tracer
        candidates, _quads = mesher.surface_nets(fine, iso, fine_grid.origin, fine_grid.h)
        options = dict(options, separation=options["separation"] * ORGANIC_UPSAMPLE)
        options["floor"] = floor
        lines = surface_stress_lines(
            streamlines,
            fine,
            fine_grid,
            iso,
            tensors,
            options,
            candidates,
            progress=trace_progress,
            should_stop=should_stop,
        )
    if not lines:
        raise ValueError("No stress lines found on the surface: lower Spacing or Min Stress")
    h = grid.h
    pad_radius = max(1.6 * float(job["thickness"]), 1.0 * float(job["pad"])) * h
    foot_radius = max(pad_radius, 1.5 * float(job["pad"]) * h)
    loads = np.asarray(job["anchors"], dtype=np.float64).reshape(-1, 3)
    footings = np.asarray(job.get("footings", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
    anchors = np.vstack([loads, footings]) if footings.size else loads
    rib_radius = float(job["thickness"]) * h
    connect = max(2.2 * rib_radius + 0.5 * h, 1.25 * float(job["spacing"]) * h)
    trimmed = trim_dangling(lines, anchors, connect, foot_radius + rib_radius, join=rib_radius)
    if not trimmed:
        warnings.append("Few ribs connect to each other: raise the Resolution or lower Spacing")
    lines = trimmed or lines
    check_stop(should_stop)
    field = SurfaceField(fine, fine_grid.origin, fine_grid.h)
    lines = lines + connector_ribs(
        streamlines,
        lines,
        loads,
        footings,
        field,
        tensors,
        iso,
        0.5 * fine_grid.h,
        connect,
        pad_radius + rib_radius,
        float(job["min_stress"]),
        floor,
    )
    if not lines:
        raise ValueError("The stress lines do not form a connected lattice: lower Spacing")
    tubes = rib_tubes(
        lines, h, job["thickness"], job["min_thickness"], sculpt, anchors, foot_radius + 2.0 * h
    )
    for point in loads:  # contact pads: the lattice reaches every load and support
        tubes.append(sculpt.Tube(points=point[None, :], radius=pad_radius, value=0.0))
    for point in footings:
        tubes.append(sculpt.Tube(points=point[None, :], radius=foot_radius, value=0.0))
    thinnest = min(float(np.min(t.radius)) for t in tubes)
    voxel = min(h / max(float(job["detail"]), 1e-3), max(thinnest / 1.25, 0.2 * h))
    pts = np.concatenate([np.asarray(t.points) for t in tubes])
    voxel = _cap_voxel(
        voxel,
        pts.min(axis=0) - 3.0 * pad_radius,
        pts.max(axis=0) + 3.0 * pad_radius,
        SCULPT_MAX_CELLS,
        warnings,
        "The rib surface",
    )
    progress(0.5, "Shaping the ribs")
    tube_field, origin, voxel, values = sculpt.tubes_field(
        tubes,
        voxel,
        blend=float(job["blend"]),
        progress=_sub_progress(progress, 0.5, 0.75),
        should_stop=should_stop,
    )
    tube_field = np.asarray(tube_field, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if float(job["skin"]) > 0.0:
        shell = skin_field(
            {"field": surface, "origin": grid.origin, "h": h},
            iso,
            float(job["skin"]) * h,
            tube_field.shape,
            origin,
            voxel,
            sculpt,
        )
        tube_field = sculpt.union_fields(tube_field, shell, float(job["blend"]))
    check_stop(should_stop)
    tiso = sculpt.TUBE_ISO
    clipped = support_clips(tube_field, origin, voxel, tiso, payload, sculpt)
    progress(0.8, "Building the surface")
    verts, faces = structure_surface(
        tube_field,
        clipped,
        origin,
        voxel,
        tiso,
        params["smooth"],
        mesher,
        structure_anchors(payload, voxel),
        support_planes(payload),
    )
    if faces.shape[0] == 0:
        raise ValueError("The ribs came out empty: raise their Thickness")
    vertex_values = sample_field(values, values > 0.0, _GridLike(origin, voxel), verts, mesher)
    geometry = _colored(verts, faces, vertex_values, params["colormap"], params["clip"], mesher)
    progress(1.0, "Done")
    return {"geometry": geometry, "warnings": warnings, "lines": len(lines)}


# ------------------------------------------------------------------ main thread
def _object_indicator(obj, depsgraph, voxel, pad=3) -> dict:
    """Fine voxelization of an object (for flush cuts), blurred: dict field/origin/h."""
    verts, tris = scene_io.evaluated_world_mesh(obj, depsgraph)
    if tris.shape[0] == 0:
        return None
    bmin, bmax = verts[np.unique(tris)].min(axis=0), verts[np.unique(tris)].max(axis=0)
    extent = np.maximum(bmax - bmin, 1e-9)
    voxel = max(float(voxel), float(extent.max()) / OBJECT_MAX_SIDE)
    cells = float(np.prod(np.ceil(extent / voxel) + 2 * pad))
    if cells > OBJECT_MAX_CELLS:
        voxel *= (cells / OBJECT_MAX_CELLS) ** (1.0 / 3.0)
    counts = np.maximum(1, np.ceil(extent / voxel - 1e-9).astype(np.int64)) + 2 * pad
    origin = 0.5 * (bmin + bmax) - 0.5 * counts * voxel
    grid = Grid(tuple(int(c) for c in counts), origin, voxel)
    inside = scene_io.voxelize_mesh(verts, tris, grid)
    if not inside.any():
        return None
    return {
        "field": _blur121(inside.astype(np.float64)),
        "origin": grid.origin,
        "h": grid.h,
        "inside": inside,
    }


def _footprint(indicator, contacts=None) -> dict:
    """The object's plan (for ``inset_contacts``); ``under`` flags the contacts that sit
    on its underside (the object fills the space just above them)."""
    if indicator is None:
        return None
    out = {
        "mask": indicator["inside"].any(axis=2),
        "origin": indicator["origin"][:2],
        "h": indicator["h"],
    }
    if contacts is not None:
        contacts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
        grid = Grid(indicator["inside"].shape, indicator["origin"], indicator["h"])
        probe = contacts + np.array([0.0, 0.0, 1.5 * grid.h])
        ijk = np.floor((probe - grid.origin) / grid.h).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.asarray(grid.shape)), axis=1)
        under = np.zeros(contacts.shape[0], dtype=bool)
        under[ok] = indicator["inside"][ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        out["under"] = under
    return out


def _nodes_near(grid, node_ids, points, radius) -> list:
    """Per point: node ids within ``radius`` (else the nearest ones)."""
    coords = grid.origin + np.stack(np.unravel_index(node_ids, grid.nodes_shape), axis=1) * grid.h
    found = []
    for point in np.asarray(points, dtype=np.float64).reshape(-1, 3):
        dist = np.linalg.norm(coords - point, axis=1)
        near = node_ids[dist <= radius]
        if near.size == 0:
            near = node_ids[dist <= dist.min() + 0.5 * grid.h]
        found.append(near)
    return found


def support_design_problem(context, settings, target, contacts, forces, footings, floor,
                           full_contact, warnings, footprint=None):
    """Problem of the SUPPORT / ORGANIC mode (main thread).

    Automatic design box from the ground (``floor``) to the highest contact + 1
    voxel, in plan the contacts and footings grown by ``0.35 H`` (at least 2
    voxels). The object and everything above its underside (inside its
    footprint) is void; solid pads of ``load_pad`` voxels sit at the contacts
    and footings (pads win over the void); every load is spread over the nodes
    of its pad; footings fix the bottom nodes of their pad (all axes), or the
    whole bottom layer with ``full_contact``. With a ``footprint`` (the object's
    plan) the contact pads are slid under the object (corners, edges). Returns
    ``(problem, info)``; ``info["contacts"]`` are the pad centres.
    """
    fem = core_module("fem")
    contacts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    footings = np.zeros((0, 3)) if footings is None else np.asarray(footings).reshape(-1, 3)
    height = max(float(contacts[:, 2].mean()) - floor, 1e-6)
    plan = np.vstack([contacts, footings]) if footings.shape[0] else contacts
    resolution = int(settings.resolution)
    grow = 0.35 * height
    h = 1.0
    for _ in range(2):  # the growth depends on the voxel size and vice versa
        lo = np.array([plan[:, 0].min() - grow, plan[:, 1].min() - grow, floor])
        hi = np.array([plan[:, 0].max() + grow, plan[:, 1].max() + grow, contacts[:, 2].max()])
        h = float((hi - lo).max()) / max(resolution, 1)
        grow = max(0.35 * height, 2.0 * h)
    hi[2] += h
    counts = np.maximum(1, np.ceil((hi - lo) / h - 1e-9).astype(np.int64))
    origin = np.array(
        [
            0.5 * (lo[0] + hi[0]) - 0.5 * counts[0] * h,
            0.5 * (lo[1] + hi[1]) - 0.5 * counts[1] * h,
            floor,
        ]
    )
    grid = Grid(tuple(int(c) for c in counts), origin, h)
    depsgraph = context.evaluated_depsgraph_get()
    verts, tris = scene_io.evaluated_world_mesh(target, depsgraph)
    inside = scene_io.voxelize_mesh(verts, tris, grid)
    shadow = np.logical_or.accumulate(inside, axis=2)  # the object and all above its underside
    centres = grid.element_centers()
    pad = max(float(settings.load_pad), 0.5) * h
    foot_pad = max(1.3 * pad, 1.5 * h)
    if footprint is not None:
        contacts = inset_contacts(contacts, np.full(contacts.shape[0], pad), footprint, 1.0)
    pads = np.zeros(grid.shape, dtype=bool)
    for point in contacts:
        pads |= np.einsum("...i,...i->...", centres - point, centres - point) <= pad * pad
    for point in footings:
        pads |= np.einsum("...i,...i->...", centres - point, centres - point) <= foot_pad * foot_pad
    domain = np.ones(grid.shape, dtype=bool)
    solid, void, bias, branch, zone_warnings = scene_io.zone_fields(
        context, grid, domain, exclude=(target,), method=settings.voxelize_method
    )
    warnings.extend(zone_warnings)
    passive_solid = pads if solid is None else (pads | solid)
    passive_void = shadow if void is None else (shadow | void)
    passive_void &= ~pads

    all_nodes = np.arange(grid.n_nodes, dtype=np.int64)
    forces_out = np.zeros(grid.n_dofs, dtype=np.float64)
    fxyz = forces_out.reshape(-1, 3)
    for nodes, force in zip(_nodes_near(grid, all_nodes, contacts, max(pad, 0.75 * h)), forces):
        fxyz[nodes] += np.asarray(force) / nodes.size
    if full_contact:
        k0 = np.zeros(grid.nodes_shape, dtype=bool)
        k0[:, :, 0] = True
        fixed_nodes = np.flatnonzero(k0.ravel())
    else:
        node_xyz = grid.node_coords()
        fixed_list = []
        for point in footings:
            plan_d = np.linalg.norm(node_xyz[:, :2] - point[:2], axis=1)
            low = node_xyz[:, 2] <= point[2] + 0.5 * h
            near = np.flatnonzero((plan_d <= foot_pad) & low & (node_xyz[:, 2] >= point[2] - h))
            if near.size == 0:
                dist = np.linalg.norm(node_xyz - point, axis=1)
                near = np.flatnonzero(dist <= dist.min() + 0.5 * h)
            fixed_list.append(near)
        fixed_nodes = np.unique(np.concatenate(fixed_list))
    fixed = (3 * fixed_nodes[:, None] + np.arange(3)[None, :]).ravel()
    problem = Problem(
        grid=grid,
        domain=domain,
        fixed_dofs=fixed,
        forces=forces_out,
        passive_solid=passive_solid,
        passive_void=passive_void,
        bias=bias,
        branch=branch,
        E=float(settings.youngs_modulus) * 1e9,
        nu=float(settings.poisson_ratio),
    )
    warnings.extend(fem.check_problem(problem))
    seeds_at = np.vstack([contacts, footings]) if footings.shape[0] else contacts
    flat_centres = centres.reshape(-1, 3)
    seed_points = flat_centres[
        [int(np.argmin(np.linalg.norm(flat_centres - p, axis=1))) for p in seeds_at]
    ]
    info = {
        "warnings": warnings,
        "loads": [(p, f / max(np.linalg.norm(f), 1e-30), float(np.linalg.norm(f)))
                  for p, f in zip(contacts, forces)],
        "supports": [p for p in footings],
        "seed_points": seed_points,
        "domain_name": target.name,
        "n_elems": int(domain.sum()),
        "bounds": (grid.origin, grid.origin + grid.size),
        "ambiguous_rays": 0,
        "pad": pad,
        "contacts": contacts,
    }
    return problem, info


def _support_box_voxel(contacts, footings, floor, resolution) -> float:
    """Voxel size of the SUPPORT design box (same rule as ``support_design_problem``)."""
    contacts = np.asarray(contacts, dtype=np.float64).reshape(-1, 3)
    footings = np.zeros((0, 3)) if footings is None else np.asarray(footings).reshape(-1, 3)
    height = max(float(contacts[:, 2].mean()) - floor, 1e-6)
    plan = np.vstack([contacts, footings]) if footings.shape[0] else contacts
    grow, h = 0.35 * height, 1.0
    for _ in range(2):
        extent = np.array(
            [np.ptp(plan[:, 0]) + 2 * grow, np.ptp(plan[:, 1]) + 2 * grow,
             float(contacts[:, 2].max()) - floor]
        )
        h = float(extent.max()) / max(int(resolution), 1)
        grow = max(0.35 * height, 2.0 * h)
    return h


def _structure_box_volume(contacts, footings, floor) -> float:
    pts = np.vstack([contacts, footings]) if len(footings) else np.asarray(contacts)
    height = max(float(np.asarray(contacts)[:, 2].mean()) - floor, 1e-6)
    grow = 0.35 * height
    size = np.array(
        [
            np.ptp(pts[:, 0]) + 2 * grow,
            np.ptp(pts[:, 1]) + 2 * grow,
            max(float(np.asarray(contacts)[:, 2].max()) - floor, 1e-6),
        ]
    )
    return float(np.prod(size))


def _opt_settings_for(settings, topopt, volume_fraction):
    opt = _opt_settings(settings, topopt)
    return dataclasses.replace(opt, volfrac=float(volume_fraction))


def _structure_params(settings) -> dict:
    return {
        "iso": float(settings.iso_level),
        "smooth": int(settings.smooth_iterations),
        "colormap": settings.colormap,
        "clip": float(settings.color_clip),
    }


def _families(settings) -> tuple:
    return {
        "BOTH": ("TENSION", "COMPRESSION"),
        "TENSION": ("TENSION",),
        "COMPRESSION": ("COMPRESSION",),
    }[settings.line_families]


def structure_request(context, settings, preview=False) -> dict:
    """Everything "Generate Structure" needs, gathered on the main thread.

    Raises ``ValueError`` with an actionable message for setups that cannot
    work (no object, no weights, Carve Inside without supports, every weight
    below the ground, ribs without surface tracing, ...).
    """
    scene = context.scene
    target = scene_io.validated_target(context, settings)
    mode, style = settings.structure_mode, settings.structure_style
    depsgraph = context.evaluated_depsgraph_get()
    warnings = []
    loads = scene_io.structure_loads(context, exclude=(target,))
    if not loads:
        raise ValueError(
            "No weights: in Edit Mode select points or faces of the object and use "
            "'Put Weight on Selection'"
        )
    supports = scene_io.structure_supports(context, exclude=(target,))
    modules = _modules("forms", "sculpt", "mesher", "fem")
    if style in {KIND_ORGANIC, KIND_RIBS} and not (style == KIND_RIBS and mode == "CARVE"
                                                    and settings.rib_source == "SURFACE"):
        try:
            modules["topopt"] = core_module("topopt")
        except RuntimeError as exc:
            raise ValueError(
                "The Organic style needs core/topopt.py, which is missing: use Branching"
            ) from exc
    if style == KIND_RIBS:
        if "topopt" not in modules:  # its density filter shapes the block's rib surface
            try:
                modules["topopt"] = core_module("topopt")
            except RuntimeError:
                pass
        modules["streamlines"] = core_module("streamlines")
        if not ribs_supported(modules["streamlines"]):
            print("Stress Sculpt: core/streamlines.py has no surface tracing: the ribs use "
                  "the add-on's own surface tracer")
    points = np.array([load["position"] for load in loads])
    forces = np.array([load["force"] for load in loads])
    params = _structure_params(settings)
    payload = {
        "scene": scene.name,
        "target": target.name,
        "mode": mode,
        "style": style,
        "modules": modules,
        "params": params,
        "warnings": warnings,
        "preview": bool(preview) and bool(settings.live_update),
        "loads": loads,
        "supports": supports,
        "object": None,
        "floor": float(settings.ground_height),
        "ground": float(settings.ground_height),
        "contact": settings.ground_contact,
    }
    t_verts, t_tris = scene_io.evaluated_world_mesh(target, depsgraph)
    if t_tris.shape[0] == 0:
        raise ValueError(f"The object '{target.name}' has no faces")
    t_min, t_max = t_verts.min(axis=0), t_verts.max(axis=0)
    longest = float(np.max(t_max - t_min))

    if mode == "SUPPORT":
        ground = float(settings.ground_height)
        contacts, moved = scene_io.underside_contacts(
            t_verts, t_tris, points, forces / np.linalg.norm(forces, axis=1)[:, None]
        )
        if moved.any():
            warnings.append(
                f"{int(moved.sum())} weight(s) on top of '{target.name}' are carried from its "
                "underside"
            )
        above = contacts[:, 2] > ground + 1e-6
        if not above.any():
            raise ValueError(
                f"Every weight is at or below the ground (Ground = {ground:.3g} m): raise the "
                "object or lower the Ground"
            )
        if not above.all():
            warnings.append(f"{int((~above).sum())} weight(s) below the ground were ignored")
        contacts, forces, points = contacts[above], forces[above], points[above]
        payload["load_names"] = [load["name"] for load, ok in zip(loads, above) if ok]
        explicit = np.array([s["position"] for s in supports]) if supports else None
        if explicit is not None:
            if np.all(explicit[:, 2] >= contacts[:, 2].min() - 1e-9):
                raise ValueError(
                    "The supports are not below the weights: in Support Object mode the "
                    "structure stands on footings below the object (or remove the supports to "
                    "place footings automatically; use Carve Inside to anchor the object itself)"
                )
            if any(s["parent"] == target.name for s in supports):
                warnings.append("Some supports sit on the carried object itself")
        full = settings.ground_contact == "FULL"
        if explicit is not None:
            footings = explicit
            payload["floor"] = min(ground, float(explicit[:, 2].min()))
        else:
            footings = modules["forms"].auto_footings(contacts, forces, ground, int(settings.footings))
        payload.update(contacts=contacts, forces=forces, footings=footings, explicit=explicit)
        height = max(float(contacts[:, 2].mean()) - ground, 1e-6)
        payload["height"] = height
        payload["box_volume"] = _structure_box_volume(contacts, footings, ground)
    else:
        if not supports:
            raise ValueError(
                "Carve Inside needs supports: select where the block rests (Edit Mode) and use "
                "'Support at Selection'"
            )
        contacts = points
        payload["load_names"] = [load["name"] for load in loads]
        payload.update(contacts=contacts, forces=forces, footings=np.array(
            [s["position"] for s in supports]), explicit=None)
        payload["box_volume"] = abs(mesh_volume(t_verts, t_tris)) or float(np.prod(t_max - t_min))

    # ------------------------------------------------------------------ per style
    voxel_hint = longest / max(int(settings.resolution), 1)
    if style == KIND_BRANCHING:
        tree_params = {
            "footings": int(settings.footings),
            "branching": int(settings.tree_branching),
            "split": float(settings.tree_split),
            "levels": float(settings.tree_levels),
            "organic": float(settings.tree_organic),
            "min_thickness": float(settings.tree_min_thickness),
            "footing_flare": float(settings.tree_footing_flare),
            "capital_flare": float(settings.tree_capital_flare),
        }
        if mode == "SUPPORT":
            height = payload["height"]
            supports_arg = payload["explicit"]
            ground_arg = float(settings.ground_height)
            constrain = None
        else:
            sup = payload["footings"]
            up = -forces.sum(axis=0)
            up = up / max(np.linalg.norm(up), 1e-30) if np.linalg.norm(up) > 1e-12 else np.array(
                [0.0, 0.0, 1.0])
            rise = [
                float(np.dot(p - sup[np.argmin(np.linalg.norm(sup - p, axis=1))], up))
                for p in contacts
            ]
            height = max(float(np.mean(rise)), 1e-3 * longest, 1e-6)
            supports_arg = sup
            ground_arg = None
            problem_grid = Grid.from_bounds(t_min, t_max, int(settings.resolution), pad=0)
            domain = scene_io.voxelize_mesh(t_verts, t_tris, problem_grid)
            if not domain.any():
                raise ValueError(
                    f"No voxel lies inside '{target.name}': the block must be a closed mesh"
                )
            constrain = nearest_inside(domain, problem_grid)
        radius_max = float(settings.tree_thickness) * height
        radius_min = float(settings.tree_min_thickness) * radius_max
        detail = float(settings.tree_detail)
        payload["branching"] = {
            "points": contacts,
            "forces": forces,
            "tree": tree_params,
            "radius_max": radius_max,
            "supports": supports_arg,
            "ground": ground_arg,
            "constrain": constrain,
            "detail": detail,
            "blend": float(settings.tree_blend),
            "footprint": None,
        }
        fine = radius_min / max(detail, 1e-3)
        payload["object"] = _object_indicator(target, depsgraph, max(fine, voxel_hint / 6.0))
        if mode == "SUPPORT":
            payload["branching"]["footprint"] = _footprint(payload["object"], contacts)
        payload["thinnest"] = radius_min
        payload["signature"] = _array_digest("branching", mode, contacts, forces,
                                             supports_arg, ground_arg, sorted(tree_params.items()),
                                             radius_max, detail, settings.tree_blend,
                                             params["smooth"], t_verts, t_tris)
        return payload

    # ORGANIC (and the optimized shape under RIBS)
    topopt = modules.get("topopt")
    if mode == "SUPPORT":
        full = settings.ground_contact == "FULL" and payload["explicit"] is None
        box_h = _support_box_voxel(payload["contacts"], payload["footings"], payload["floor"],
                                   int(settings.resolution))
        payload["object"] = _object_indicator(target, depsgraph, box_h / ORGANIC_UPSAMPLE)
        problem, info = support_design_problem(
            context,
            settings,
            target,
            payload["contacts"],
            payload["forces"],
            None if full else payload["footings"],
            payload["floor"],
            full,
            warnings,
            footprint=_footprint(payload["object"], payload["contacts"]),
        )
        fraction = float(settings.support_material)
        payload["pad"] = info["pad"]
        payload["contacts"] = info["contacts"]
    else:
        problem, info = scene_io.gather_problem(context, settings, domain=target)
        warnings.extend(info["warnings"])
        info = dict(info, warnings=[])
        fraction = float(settings.volume_fraction)
        payload["pad"] = max(float(settings.load_pad), 0.5) * problem.grid.h
    payload["problem"] = problem
    payload["info"] = info
    payload["thinnest"] = problem.grid.h
    payload["top"] = float(np.max(payload["contacts"][:, 2])) if mode == "SUPPORT" else None
    if payload["object"] is None:
        payload["object"] = _object_indicator(
            target, depsgraph, problem.grid.h / ORGANIC_UPSAMPLE
        )
    need_optimizer = not (style == KIND_RIBS and mode == "CARVE" and settings.rib_source == "SURFACE")
    if need_optimizer:
        payload["opt"] = _opt_settings_for(settings, topopt, fraction)
        payload["organic_signature"] = problem_digest(problem, payload["opt"])
        cached = jobs.get_cache(scene).structure
        payload["reuse"] = None
        if (
            style == KIND_RIBS
            and cached is not None
            and cached.get("organic_signature") == payload["organic_signature"]
            and cached.get("state") is not None
        ):
            payload["reuse"] = cached["state"]
    else:
        payload["opt"] = None
        payload["organic_signature"] = None
        payload["reuse"] = None
        payload["solver"] = {
            "solver": settings.solver,
            "tol": float(settings.solver_tolerance),
            "maxiter": int(settings.max_solver_iterations),
            "rmin": float(settings.filter_radius),
        }
    if style == KIND_RIBS:
        footings = np.asarray(payload["footings"], dtype=np.float64).reshape(-1, 3)
        anchors = np.vstack([payload["contacts"], footings]) if footings.size else payload[
            "contacts"]
        payload["ribs"] = {
            "families": _families(settings),
            "n_seeds": max(int(settings.n_seeds), 60),
            "seeds": anchors,
            "anchors": payload["contacts"],
            "footings": footings,
            "step": float(settings.line_step),
            "min_stress": 0.25 * float(settings.line_min_stress),
            "spacing": float(settings.rib_spacing),
            "thickness": float(settings.rib_thickness),
            "min_thickness": float(settings.rib_min_thickness),
            "skin": float(settings.rib_skin),
            "blend": float(settings.sculpt_blend),
            "detail": float(settings.sculpt_detail),
            "pad": float(settings.load_pad),
            "rng_seed": int(settings.random_seed),
            "source": "SURFACE" if mode == "CARVE" and settings.rib_source == "SURFACE"
            else "EVOLVED",
        }
    payload["signature"] = _array_digest(
        style, mode, payload["organic_signature"], sorted(payload.get("ribs", {}).items(),
                                                            key=lambda kv: kv[0]).__repr__()
    )
    return payload


def structure_compute(payload, progress, should_stop, emit) -> dict:
    """Worker side of "Generate Structure" (pure numpy)."""
    style, mode = payload["style"], payload["mode"]
    out = {"payload": payload, "warnings": [], "state": None, "kind": style}
    if style == KIND_BRANCHING:
        out.update(branching_geometry(payload, progress, should_stop))
        return out

    problem = payload["problem"]
    state = None
    max_displacement = 0.0
    ribs = style == KIND_RIBS
    organic_span = (0.0, 0.55) if ribs else (0.0, 0.9)
    if payload["opt"] is not None:
        state = payload.get("reuse")
        if state is None:
            topopt = payload["modules"]["topopt"]

            def on_state(current):
                snapshot = {"payload": payload, "state": current, "partial": True}
                if payload["preview"]:
                    snapshot["preview"] = organic_preview(current, problem, payload)
                emit(snapshot)

            optimizer, state = run_optimizer(
                topopt,
                problem,
                payload["opt"],
                _sub_progress(progress, *organic_span),
                should_stop,
                on_state,
            )
            max_displacement = _max_displacement(
                optimizer, state, problem, payload["opt"], should_stop
            )
            out["fresh_state"] = True
        out["state"] = state
        out["max_displacement"] = max_displacement
    if not ribs:
        progress(0.92, "Building the surface")
        out["geometry"] = organic_geometry(
            state, problem, payload, organic_smoothing(payload), should_stop=should_stop
        )
        progress(1.0, "Done")
        return out

    # ribs
    rib_progress = _sub_progress(progress, organic_span[1], 1.0)
    job = payload["ribs"]
    if job["source"] == "SURFACE":
        fem = payload["modules"]["fem"]
        solver = payload["solver"]
        result = fem.analyze(
            problem,
            solver=solver["solver"],
            tol=solver["tol"],
            maxiter=solver["maxiter"],
            progress=_sub_progress(progress, 0.0, 0.5),
            should_stop=should_stop,
        )
        surface = problem.domain.astype(np.float64)
        topopt = payload["modules"].get("topopt")
        if topopt is not None and hasattr(topopt, "DensityFilter"):
            surface = np.asarray(
                topopt.DensityFilter(problem.domain, max(solver["rmin"], 1.0)).forward(surface)
            )
        else:
            surface = np.where(problem.domain, _blur121(surface), 0.0)
        stress, iso = np.asarray(result.stress), 0.5
        out["analysis"] = {
            "compliance": float(result.compliance),
            "max_stress": float(np.max(result.von_mises)),
            "max_displacement": float(result.max_displacement),
        }
    else:
        surface = np.where(problem.domain, np.asarray(state.density, dtype=np.float64), 0.0)
        stress, iso = np.asarray(state.stress), float(payload["params"]["iso"])
    out.update(
        ribs_geometry(payload, surface, stress, problem.grid, iso, rib_progress, should_stop)
    )
    return out


def _structure_partial(snapshot) -> dict:
    """A cancelled organic optimization: finish with the last state."""
    return {
        "payload": snapshot["payload"],
        "warnings": [],
        "state": snapshot["state"],
        "kind": KIND_ORGANIC,
        "cancelled": True,
        "max_displacement": 0.0,
    }


def _set_structure_props(obj, payload, output) -> None:
    obj[PROP_MODE] = payload["mode"]
    obj[PROP_STYLE] = output.get("kind", payload["style"])
    obj[PROP_GROUND] = float(payload["ground"])
    obj[PROP_CONTACT] = payload["contact"]
    footings = output.get("footings")
    if footings is None:
        footings = payload.get("footings")
    footings = np.asarray(footings if footings is not None else np.zeros((0, 3)), dtype=np.float64)
    if footings.size:
        obj[PROP_FOOTINGS] = [float(v) for v in footings.ravel()]
    elif PROP_FOOTINGS in obj.keys():
        del obj[PROP_FOOTINGS]
    obj[PROP_THINNEST] = float(output.get("thinnest") or payload.get("thinnest") or 0.0)
    obj[PROP_PAD] = float(payload.get("pad") or 0.0)
    contacts = output.get("contacts")
    if contacts is None:
        contacts = payload.get("contacts")
    names = payload.get("load_names") or []
    contacts = np.asarray(contacts if contacts is not None else np.zeros((0, 3)), dtype=np.float64)
    if contacts.size and len(names) == contacts.reshape(-1, 3).shape[0]:
        obj[PROP_CONTACTS] = [float(v) for v in contacts.ravel()]
        obj[PROP_LOADS] = "\n".join(names)
    else:
        for key in (PROP_CONTACTS, PROP_LOADS):
            if key in obj.keys():
                del obj[key]


def _show_structure(context, scene, geometry, view_layer, payload):
    with build.NewData() as new:
        mesh = new.add(_mesh_from_geometry(build.OBJ_STRUCTURE, geometry))
        obj = build.assign_result(
            scene, build.OBJ_STRUCTURE, mesh, build.structure_material(), view_layer
        )
    build.set_result_material(obj, build.structure_material())
    others = tuple(name for name in build.RESULT_OBJECTS if name != build.OBJ_STRUCTURE)
    build.hide_results(scene, others, view_layer)
    if payload["mode"] == "CARVE":
        target = bpy.data.objects.get(payload["target"])
        if target is not None and scene.stress_sculpt.domain_as_wire:
            target.display_type = "WIRE"
    build.use_attribute_colors(context)
    return obj


class STRESS_SCULPT_OT_generate(ThreadedOperatorMixin, Operator):
    bl_idname = "stress_sculpt.generate"
    bl_label = "Generate Structure"
    bl_description = (
        "Grow a structure that carries the weights: from the ground up to the object (Support "
        "Object) or carved inside it (Carve Inside), in the chosen style. Esc stops (an organic "
        "shape keeps its current state)"
    )
    bl_options = {"REGISTER", "UNDO"}
    job_name = "Generate"
    keep_partial_on_cancel = True

    def prepare(self, context):
        _ensure_object_mode(context)
        return structure_request(context, _settings(context), preview=self.is_async())

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        return structure_compute(payload, progress, should_stop, emit)

    def on_partial(self, context, snapshot):
        geometry = snapshot.get("preview")
        if geometry is None or not len(geometry["faces"]):
            return
        payload = snapshot["payload"]
        scene = bpy.data.scenes.get(payload["scene"])
        if scene is None:
            return
        _show_structure(context, scene, geometry, _view_layer(context, scene), payload)

    def result_from_partial(self, partial):
        return _structure_partial(partial)

    def finish(self, context, output):
        payload = output["payload"]
        scene = _job_scene(payload["scene"])
        view_layer = _view_layer(context, scene)
        kind = output.get("kind", payload["style"])
        state = output.get("state")
        geometry = output.get("geometry")
        warnings = list(payload["warnings"]) + list(output.get("warnings") or [])
        if geometry is None:  # cancelled optimization: mesh the last state here
            geometry = organic_geometry(
                state, payload["problem"], payload, organic_smoothing(payload)
            )
        if len(geometry["faces"]) == 0:
            raise ValueError(
                "The structure came out empty: raise the Material amount (or lower Surface Level)"
            )
        obj = _show_structure(context, scene, geometry, view_layer, payload)
        _set_structure_props(obj, payload, output)

        cache = jobs.get_cache(scene)
        entry = {
            "kind": kind,
            "mode": payload["mode"],
            "signature": payload.get("signature"),
            "organic_signature": payload.get("organic_signature"),
            "problem": payload.get("problem"),
            "state": state,
            "footings": output.get("footings", payload.get("footings")),
            "contacts": output.get("contacts", payload.get("contacts")),
        }
        if state is None and cache.structure is not None:
            previous = cache.structure  # keep an earlier organic state for later ribs
            if previous.get("state") is not None:
                entry["state"] = previous["state"]
                entry["organic_signature"] = previous.get("organic_signature")
                entry["problem"] = previous.get("problem")
        cache.structure = entry
        if state is not None and output.get("fresh_state"):
            cache.evolution = {  # the line tools can trace inside the generated shape
                "problem": payload["problem"],
                "state": state,
                "settings": payload["opt"],
                "signature": payload["organic_signature"],
                "info": payload["info"],
                "iso": payload["params"]["iso"],
                "max_displacement": output.get("max_displacement"),
            }

        # metrics
        volume = abs(mesh_volume(geometry["verts"], geometry["faces"]))
        reference = max(float(payload.get("box_volume") or 0.0), 1e-30)
        reach = max(
            3.0 * float(payload.get("pad") or 0.0), 2.5 * float(payload.get("thinnest") or 0.0)
        )
        contacts = output.get("contacts", payload.get("contacts"))
        far = unreached_points(contacts, geometry["verts"], max(reach, 1e-9) * 2.0)
        if far.size:
            warnings.append(
                f"{far.size} weight(s) are not reached by the structure: raise the Material "
                "amount or the Contact Size"
            )
        metrics = output.get("metrics") or {}
        analysis = output.get("analysis") or {}
        if state is not None:
            max_stress = float(np.max(state.von_mises))
            compliance = float(state.compliance)
            iterations = int(state.iteration)
        else:
            max_stress = float(metrics.get("max_stress", analysis.get("max_stress", 0.0)))
            compliance = float(analysis.get("compliance", 0.0))
            iterations = 0
        _set_last(
            scene,
            kind,
            max_stress=max_stress,
            max_displacement=float(
                output.get("max_displacement") or analysis.get("max_displacement", 0.0)
            ),
            compliance=compliance,
            volume=volume / reference,
            iterations=iterations,
        )
        _report_warnings(self, warnings)
        if kind == KIND_BRANCHING:
            message = (
                f"Branching structure: {metrics.get('members', 0)} members on "
                f"{metrics.get('footings', 0)} footing(s), carrying "
                f"{_format_force(metrics.get('total_load', 0.0))}"
            )
        elif kind == KIND_RIBS:
            message = f"Ribs: {output.get('lines', 0)} stress line(s) over the shape"
        else:
            status = "stopped" if output.get("cancelled") else (
                "converged" if getattr(state, "converged", False) else "done"
            )
            message = (
                f"Organic structure ({iterations} iterations, {status}): "
                f"material {getattr(state, 'volume', 0.0) * 100:.1f}%"
            )
        self.report({"INFO"}, message + f", {len(geometry['faces'])} faces")
        return {"FINISHED"}


# ------------------------------------------------------------------------ verify
def verify_request(context, settings) -> dict:
    """FEM check of SS_Structure: voxelized, loaded by the weights, held by its supports."""
    scene = context.scene
    obj = build.find_result(scene, build.OBJ_STRUCTURE)
    if obj is None or obj.type != "MESH" or not len(obj.data.polygons):
        raise ValueError("No structure to check: press Generate Structure first")
    depsgraph = context.evaluated_depsgraph_get()
    verts, tris = scene_io.evaluated_world_mesh(obj, depsgraph)
    if tris.shape[0] == 0:
        raise ValueError("The structure has no faces")
    target = settings.target
    keys = set(obj.keys())
    mode = str(obj[PROP_MODE]) if PROP_MODE in keys else settings.structure_mode
    ground = float(obj[PROP_GROUND]) if PROP_GROUND in keys else float(settings.ground_height)
    contact = str(obj[PROP_CONTACT]) if PROP_CONTACT in keys else settings.ground_contact
    thinnest = float(obj[PROP_THINNEST]) if PROP_THINNEST in keys else 0.0
    warnings = []
    bmin, bmax = verts.min(axis=0), verts.max(axis=0)
    extent = np.maximum(bmax - bmin, 1e-9)
    h = float(extent.max()) / max(int(settings.resolution), 1)
    if thinnest > 0.0:
        h = min(h, thinnest / 1.25)
    cells = float(np.prod(np.ceil(extent / h) + 2))
    if cells > VERIFY_MAX_CELLS:
        h *= (cells / VERIFY_MAX_CELLS) ** (1.0 / 3.0)
        if thinnest > 0.0 and h > thinnest:
            warnings.append("The check grid is coarse for the thinnest members: results are rough")
    counts = np.maximum(1, np.ceil(extent / h - 1e-9).astype(np.int64)) + 2
    origin = 0.5 * (bmin + bmax) - 0.5 * counts * h
    grid = Grid(tuple(int(c) for c in counts), origin, h)
    domain = scene_io.voxelize_mesh(verts, tris, grid)
    if not domain.any():
        raise ValueError("The structure is thinner than one voxel: raise the Resolution")
    active = np.flatnonzero(grid.nodes_of_elements(domain))
    node_xyz = grid.origin + np.stack(np.unravel_index(active, grid.nodes_shape), axis=1) * h

    loads = scene_io.structure_loads(context, exclude=(target,) if target is not None else ())
    if not loads:
        raise ValueError("No weights to check the structure with")
    points = np.array([load["position"] for load in loads])
    forces = np.array([load["force"] for load in loads])
    stored_names = str(obj[PROP_LOADS]).split("\n") if PROP_LOADS in keys else []
    if stored_names and stored_names == [load["name"] for load in loads]:
        points = np.asarray(list(obj[PROP_CONTACTS]), dtype=np.float64).reshape(-1, 3)
    elif mode == "SUPPORT" and target is not None and target.type == "MESH":
        t_verts, t_tris = scene_io.evaluated_world_mesh(target, depsgraph)
        points, _moved = scene_io.underside_contacts(
            t_verts, t_tris, points, forces / np.linalg.norm(forces, axis=1)[:, None]
        )
    force_vec = np.zeros(grid.n_dofs, dtype=np.float64)
    fxyz = force_vec.reshape(-1, 3)
    for load, point, force in zip(loads, points, forces):
        radius = max(float(load["radius"]), 1.5 * h, thinnest)
        dist = np.linalg.norm(node_xyz - point, axis=1)
        near = active[dist <= radius]
        if near.size == 0:
            near = active[dist <= dist.min() + 0.5 * h]
            if dist.min() > 3.0 * h:
                warnings.append(f"'{load['name']}' does not touch the structure")
        fxyz[near] += force / near.size

    fixed_nodes = []
    supports = scene_io.structure_supports(context, exclude=(target,) if target else ())
    if mode == "CARVE":
        if not supports:
            raise ValueError("No supports to hold the structure: add supports to the block")
        for support in supports:
            sobj = bpy.data.objects.get(support["name"])
            inside = scene_io.region_mask_points(sobj, node_xyz) if sobj is not None else None
            nodes = active[inside] if inside is not None and inside.any() else np.zeros(0, int)
            if nodes.size == 0:
                dist = np.linalg.norm(node_xyz - support["position"], axis=1)
                nodes = active[dist <= max(dist.min() + 0.5 * h, 1.5 * h)]
            axes = [a for a in range(3) if support["fix"][a]]
            fixed_nodes.append((3 * nodes[:, None] + np.asarray(axes)[None, :]).ravel())
    else:
        if supports:
            footings = np.array([s["position"] for s in supports])
        elif PROP_FOOTINGS in keys and contact != "FULL":
            footings = np.asarray(list(obj[PROP_FOOTINGS]), dtype=np.float64).reshape(-1, 3)
        else:
            footings = None
        if footings is None or contact == "FULL" and not supports:
            low = node_xyz[:, 2] <= max(ground, float(node_xyz[:, 2].min())) + 0.75 * h
            nodes = active[low]
            fixed_nodes.append((3 * nodes[:, None] + np.arange(3)[None, :]).ravel())
        else:
            for footing in footings:
                dist = np.linalg.norm(node_xyz - footing, axis=1)
                radius = max(2.5 * h, float(obj[PROP_PAD]) if PROP_PAD in keys else 0.0)
                low = node_xyz[:, 2] <= footing[2] + 0.75 * h
                nodes = active[(dist <= radius) & low]
                if nodes.size == 0:
                    nodes = active[dist <= dist.min() + 0.5 * h]
                fixed_nodes.append((3 * nodes[:, None] + np.arange(3)[None, :]).ravel())
    fixed = np.unique(np.concatenate(fixed_nodes)) if fixed_nodes else np.zeros(0, np.int64)
    if fixed.size == 0:
        raise ValueError("The structure touches no support")
    fem = core_module("fem")
    problem = Problem(
        grid=grid,
        domain=domain,
        fixed_dofs=fixed,
        forces=force_vec,
        E=float(settings.youngs_modulus) * 1e9,
        nu=float(settings.poisson_ratio),
    )
    problem, n_removed = fem.remove_unsupported_components(problem)
    if n_removed:
        warnings.append(f"{n_removed} part(s) of the structure touch no support (floating)")
    if not problem.domain.any():
        raise ValueError("No part of the structure touches a support")
    problem.forces.reshape(-1, 3)[~grid.nodes_of_elements(problem.domain)] = 0.0
    if not np.any(problem.forces):
        raise ValueError("No weight acts on a supported part of the structure")
    warnings.extend(fem.check_problem(problem))
    mesh_verts = build.mesh_vertices(obj.data)
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    return {
        "scene": scene.name,
        "problem": problem,
        "warnings": warnings,
        "solver": settings.solver,
        "tol": float(settings.solver_tolerance),
        "maxiter": int(settings.max_solver_iterations),
        "points": mesh_verts @ matrix[:3, :3].T + matrix[:3, 3],
        "colormap": settings.colormap,
        "clip": float(settings.color_clip),
        "fem": fem,
        "mesher": core_module("mesher"),
        "kind": str(obj[PROP_STYLE]) if PROP_STYLE in keys else settings.last_kind,
    }


class STRESS_SCULPT_OT_verify(ThreadedOperatorMixin, Operator):
    bl_idname = "stress_sculpt.verify"
    bl_label = "Check Generated Structure"
    bl_description = (
        "Solve the stresses of the generated structure under the weights and color it by von "
        "Mises stress"
    )
    bl_options = {"REGISTER", "UNDO"}
    job_name = "Check"

    def prepare(self, context):
        _ensure_object_mode(context)
        return verify_request(context, _settings(context))

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        result = payload["fem"].analyze(
            payload["problem"],
            solver=payload["solver"],
            tol=payload["tol"],
            maxiter=payload["maxiter"],
            progress=progress,
            should_stop=should_stop,
        )
        problem = payload["problem"]
        von_mises = np.asarray(result.von_mises, dtype=np.float64)
        values = sample_field(
            von_mises, problem.domain, problem.grid, payload["points"], payload["mesher"]
        )
        lo, hi = color_range(von_mises[problem.domain], False, payload["colormap"], payload["clip"])
        colors = payload["mesher"].colormap(values, lo, hi, name=payload["colormap"])
        return {"payload": payload, "result": result, "values": values, "colors": colors}

    def finish(self, context, output):
        payload, result = output["payload"], output["result"]
        scene = _job_scene(payload["scene"])
        obj = build.find_result(scene, build.OBJ_STRUCTURE)
        if obj is None or len(obj.data.vertices) != len(output["values"]):
            raise ValueError("The structure changed while it was checked: check it again")
        build.set_mesh_stress(obj.data, output["colors"], output["values"])
        build.set_result_material(obj, build.heatmap_material())
        build.use_attribute_colors(context)
        problem = payload["problem"]
        max_stress = float(np.max(result.von_mises)) if problem.domain.any() else 0.0
        settings = scene.stress_sculpt
        kind = payload["kind"] if payload["kind"] in STRUCTURE_KINDS else settings.last_kind
        _set_last(
            scene,
            kind or KIND_BRANCHING,
            max_stress=max_stress,
            max_displacement=result.max_displacement,
            compliance=result.compliance,
            volume=settings.last_volume,
            iterations=settings.last_iterations,
        )
        _report_warnings(self, payload["warnings"])
        if not result.info.get("converged", True):
            self.report({"WARNING"}, "The solver did not fully converge: raise Max Solver Iterations")
        self.report(
            {"INFO"},
            f"Check: max von Mises {_format_stress(max_stress)}, max displacement "
            f"{_format_length(result.max_displacement)} ({int(problem.domain.sum())} voxels)",
        )
        return {"FINISHED"}


# ================================================================= housekeeping
class STRESS_SCULPT_OT_cancel(Operator):
    bl_idname = "stress_sculpt.cancel"
    bl_label = "Cancel Job"
    bl_description = "Stop the running Stress Sculpt computation"
    bl_options = {"REGISTER"}  # changes no data: nothing to undo

    @classmethod
    def poll(cls, context):
        return RUNTIME.running

    def execute(self, context):
        if not RUNTIME.busy():  # also clears a stale job state
            jobs.redraw_views(context)
            return {"FINISHED"}
        RUNTIME.request_cancel()
        jobs.redraw_views(context)
        self.report({"INFO"}, "Cancelling...")
        return {"FINISHED"}


class STRESS_SCULPT_OT_clear_results(Operator):
    bl_idname = "stress_sculpt.clear_results"
    bl_label = "Clear Results"
    bl_description = "Delete the generated result objects and forget the cached computations"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return not RUNTIME.busy()

    def execute(self, context):
        try:
            _ensure_object_mode(context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        scene = context.scene
        collection = build.results_collection(scene, create=False)
        removed = 0
        if collection is not None:
            removed = build.remove_objects(list(collection.objects))
            if not collection.objects and not collection.children:
                bpy.data.collections.remove(collection)
        jobs.clear_cache(scene)
        settings = scene.stress_sculpt
        for obj in (settings.domain, settings.target):
            if obj is not None and obj.display_type == "WIRE":
                obj.display_type = "TEXTURED"
        _set_last(scene, "")
        self.report({"INFO"}, f"Removed {removed} result object(s)")
        return {"FINISHED"}


# --------------------------------------------------------------------- variants
def settings_to_dict(settings) -> dict:
    """All scene setting values as JSON-able data (pointers -> names)."""
    data = {}
    for prop in settings.bl_rna.properties:
        key = prop.identifier
        if key in _SETTINGS_SKIP or key.startswith("last_") or prop.type == "COLLECTION":
            continue
        value = getattr(settings, key)
        if prop.type == "POINTER":
            data[key] = value.name if value is not None else None
        elif prop.type == "ENUM" and getattr(prop, "is_enum_flag", False):
            data[key] = sorted(value)
        elif getattr(prop, "is_array", False) or (
            hasattr(value, "__len__") and not isinstance(value, str)
        ):
            data[key] = list(value)
        else:
            data[key] = value
    return data


def apply_settings_dict(settings, data) -> list:
    """Write values back (unknown / invalid keys are skipped). Returns the skipped keys.

    Values are applied in declaration order, so the material preset (whose update
    callback sets E and nu) comes before the explicit modulus and Poisson ratio.
    """
    properties = settings.bl_rna.properties
    order = {prop.identifier: index for index, prop in enumerate(properties)}
    skipped = [key for key in data if key not in order]
    for key in sorted((k for k in data if k in order), key=order.__getitem__):
        value = data[key]
        prop = properties[key]
        if prop.is_readonly or key in _SETTINGS_SKIP or key.startswith("last_"):
            skipped.append(key)
            continue
        try:
            if prop.type == "POINTER":
                obj = bpy.data.objects.get(value) if value else None
                if value and (obj is None or obj.type != "MESH"):
                    skipped.append(key)
                    continue
                setattr(settings, key, obj)
            elif prop.type == "ENUM" and getattr(prop, "is_enum_flag", False):
                setattr(settings, key, set(value))
            else:
                setattr(settings, key, value)
        except (TypeError, ValueError, AttributeError):
            skipped.append(key)
    return skipped


def setup_to_list(context) -> list:
    """Role objects (name, role, transform, role settings) of the current setup."""
    entries = []
    for obj in scene_io.role_objects(context, include_disabled=True):
        s = obj.stress_sculpt
        entry = {
            "name": obj.name,
            "type": obj.type,
            "role": s.role,
            "enabled": bool(s.enabled),
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "rotation_mode": obj.rotation_mode,
            "scale": list(obj.scale),
            "magnitude": float(s.magnitude),
            "load_radius": float(s.load_radius),
            "fix": [bool(s.fix_x), bool(s.fix_y), bool(s.fix_z)],
            "strength": float(s.strength),
        }
        if obj.type == "EMPTY":
            entry["empty_display_type"] = obj.empty_display_type
            entry["empty_display_size"] = float(obj.empty_display_size)
        entries.append(entry)
    return entries


def apply_setup_list(context, entries) -> tuple:
    """Restore role objects by name (missing empties are recreated, others disabled).

    Returns ``(restored, recreated, disabled, missing)`` counts.
    """
    scene = context.scene
    restored = recreated = disabled = missing = 0
    names = set()
    for entry in entries:
        name = entry.get("name", "")
        names.add(name)
        obj = scene.objects.get(name)
        if obj is None:
            if entry.get("type") != "EMPTY":
                missing += 1
                continue
            obj = new_role_empty(
                scene,
                name,
                entry.get("role", "NONE"),
                entry.get("empty_display_type", "SPHERE"),
                entry.get("empty_display_size", 1.0),
                entry.get("location", (0, 0, 0)),
            )
            recreated += 1
        else:
            restored += 1
        s = obj.stress_sculpt
        try:
            s.role = entry.get("role", s.role)
            s.enabled = bool(entry.get("enabled", True))
            if "rotation_mode" in entry:
                obj.rotation_mode = entry["rotation_mode"]
            obj.location = entry.get("location", obj.location)
            obj.rotation_euler = entry.get("rotation", obj.rotation_euler)
            obj.scale = entry.get("scale", obj.scale)
            s.magnitude = float(entry.get("magnitude", s.magnitude))
            s.load_radius = float(entry.get("load_radius", s.load_radius))
            s.fix_x, s.fix_y, s.fix_z = (
                bool(v) for v in entry.get("fix", (s.fix_x, s.fix_y, s.fix_z))
            )
            s.strength = float(entry.get("strength", s.strength))
            if obj.type == "EMPTY":
                obj.empty_display_type = entry.get("empty_display_type", obj.empty_display_type)
                obj.empty_display_size = float(
                    entry.get("empty_display_size", obj.empty_display_size)
                )
        except (TypeError, ValueError, AttributeError):
            pass
    for obj in scene_io.role_objects(context):
        if obj.name not in names:
            obj.stress_sculpt.enabled = False
            disabled += 1
    return restored, recreated, disabled, missing


def _variant_number(settings) -> int:
    numbers = [0]
    for variant in settings.variants:
        parts = variant.name.split()
        if len(parts) >= 2 and parts[0] == "Variant" and parts[1].isdigit():
            numbers.append(int(parts[1]))
    return max(max(numbers), len(settings.variants)) + 1


def _compare_collection(scene, create=False):
    if create:
        return build.ensure_collection(scene, build.COMPARE_COLLECTION)
    return build.find_child_collection(scene.collection, build.COMPARE_COLLECTION)


def clear_compare(scene) -> int:
    collection = _compare_collection(scene)
    if collection is None:
        return 0
    count = build.remove_objects(list(collection.objects))
    bpy.data.collections.remove(collection)
    return count


def build_compare(context, scene) -> int:
    """Collection instances of all variants in a row along +X, right of the domain, with labels."""
    settings = scene.stress_sculpt
    variants = [
        v for v in settings.variants if v.collection is not None and len(v.collection.all_objects)
    ]
    clear_compare(scene)
    if not variants:
        return 0
    boxes = [build.union_bounds(v.collection.all_objects) for v in variants]
    widths = [float(b[1][0] - b[0][0]) for b in boxes if b is not None]
    domain_box = _world_bounds(context, settings.domain) if settings.domain is not None else None
    if domain_box is None:
        domain_box = (
            np.min([b[0] for b in boxes if b is not None], axis=0),
            np.max([b[1] for b in boxes if b is not None], axis=0),
        )
    width = max(widths + [float(domain_box[1][0] - domain_box[0][0]), 1e-3])
    step = width * float(settings.compare_spacing)
    gap = max(step - width, 0.05 * width)
    x_start = float(domain_box[1][0]) + gap
    collection = _compare_collection(scene, create=True)
    for index, (variant, box) in enumerate(zip(variants, boxes)):
        if box is None:
            continue
        centre_x = x_start + index * (width + gap) + 0.5 * width
        offset_x = centre_x - 0.5 * float(box[0][0] + box[1][0])
        instance = bpy.data.objects.new(f"SS_Compare {variant.name}", None)
        instance.instance_type = "COLLECTION"
        instance.instance_collection = variant.collection
        instance.empty_display_size = 0.05 * width
        instance.location = (offset_x, 0.0, 0.0)
        collection.objects.link(instance)

        height = float(box[1][2] - box[0][2])
        text_size = max(0.06 * width, 0.04 * height, 1e-3)
        kind = KIND_LABELS.get(variant.kind, variant.kind.title() if variant.kind else "Result")
        lines = [variant.name, kind]
        details = []
        if variant.kind and variant.kind != KIND_ANALYSIS:
            details.append(f"material {variant.volume * 100:.0f}%")
        if variant.compliance:
            details.append(f"compliance {variant.compliance:.3g} J")
        if details:
            lines.append(", ".join(details))
        if variant.max_stress:
            lines.append(f"max stress {_format_stress(variant.max_stress)}")
        build.text_object(
            f"SS_Label {variant.name}",
            "\n".join(lines),
            text_size,
            (centre_x, float(box[0][1]), float(box[0][2]) - 0.6 * text_size),
            collection,
        )
    return len(variants)


class STRESS_SCULPT_OT_variant_save(Operator):
    bl_idname = "stress_sculpt.variant_save"
    bl_label = "Save Variant"
    bl_description = (
        "Store copies of the visible results with their settings and metrics as a variant"
    )
    bl_options = {"REGISTER", "UNDO"}

    variant_name: StringProperty(
        name="Name",
        default="",
        description="Variant name (empty: automatic)",
        options={"SKIP_SAVE"},
    )

    @classmethod
    def poll(cls, context):
        return not RUNTIME.busy()

    def execute(self, context):
        scene = context.scene
        settings = scene.stress_sculpt
        view_layer = context.view_layer
        results = build.results_collection(scene, create=False)
        sources = (
            [obj for obj in results.objects if build.is_visible(obj, view_layer)] if results else []
        )
        if not sources:
            self.report(
                {"ERROR"},
                "Nothing to save: generate a structure first (or run Analyze, Evolve, Lines, "
                "Sculpt or Relief)",
            )
            return {"CANCELLED"}
        number = _variant_number(settings)
        kind = settings.last_kind or KIND_ANALYSIS
        label = f"Variant {number}"
        parent = build.ensure_collection(scene, build.VARIANTS_COLLECTION)
        build.set_collection_excluded(scene, parent, True)
        collection = bpy.data.collections.new(f"{label} – {KIND_LABELS.get(kind, kind.title())}")
        parent.children.link(collection)
        for source in sources:
            duplicate = source.copy()
            if source.data is not None:
                duplicate.data = source.data.copy()
            duplicate.name = f"V{number} {build.base_name(source.name)}"
            collection.objects.link(duplicate)
        build.set_collection_excluded(scene, parent, True)

        variant = settings.variants.add()
        variant.name = self.variant_name.strip() or label
        variant.collection = collection
        variant.kind = kind
        variant.compliance = settings.last_compliance
        variant.volume = settings.last_volume
        variant.max_stress = settings.last_max_stress
        variant.max_displacement = settings.last_max_displacement
        variant.iterations = settings.last_iterations
        variant.settings_json = json.dumps(settings_to_dict(settings), sort_keys=True)
        variant.setup_json = json.dumps(setup_to_list(context))
        settings.active_variant_index = len(settings.variants) - 1
        self.report({"INFO"}, f"Saved '{variant.name}' ({len(sources)} object(s))")
        return {"FINISHED"}


def _variant_index(settings, index) -> int:
    return settings.active_variant_index if index < 0 else index


class STRESS_SCULPT_OT_variant_remove(Operator):
    bl_idname = "stress_sculpt.variant_remove"
    bl_label = "Remove Variant"
    bl_description = "Delete the active variant and its objects"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(
        name="Index", default=-1, options={"SKIP_SAVE"}, description="-1: active variant"
    )

    @classmethod
    def poll(cls, context):
        return len(_settings(context).variants) > 0

    def execute(self, context):
        scene = context.scene
        settings = scene.stress_sculpt
        index = _variant_index(settings, self.index)
        if not 0 <= index < len(settings.variants):
            self.report({"ERROR"}, "No variant selected")
            return {"CANCELLED"}
        variant = settings.variants[index]
        name = variant.name
        if variant.collection is not None:
            build.remove_collection(variant.collection)
        settings.variants.remove(index)
        settings.active_variant_index = max(0, min(index, len(settings.variants) - 1))
        if _compare_collection(scene) is not None:
            build_compare(context, scene)
        self.report({"INFO"}, f"Removed '{name}'")
        return {"FINISHED"}


class STRESS_SCULPT_OT_variant_restore(Operator):
    bl_idname = "stress_sculpt.variant_restore"
    bl_label = "Restore Variant Settings"
    bl_description = (
        "Bring back the settings (and the supports / loads / zones) the variant was made with"
    )
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(
        name="Index", default=-1, options={"SKIP_SAVE"}, description="-1: active variant"
    )
    include_setup: BoolProperty(
        name="Restore Setup",
        default=True,
        description="Also restore the supports, loads and zones (by name); others are disabled",
    )

    @classmethod
    def poll(cls, context):
        return len(_settings(context).variants) > 0 and not RUNTIME.busy()

    def execute(self, context):
        settings = _settings(context)
        index = _variant_index(settings, self.index)
        if not 0 <= index < len(settings.variants):
            self.report({"ERROR"}, "No variant selected")
            return {"CANCELLED"}
        variant = settings.variants[index]
        name = variant.name
        try:
            data = json.loads(variant.settings_json) if variant.settings_json else {}
            setup = json.loads(variant.setup_json) if variant.setup_json else []
        except ValueError:
            self.report({"ERROR"}, f"The settings stored in '{name}' are damaged")
            return {"CANCELLED"}
        skipped = apply_settings_dict(settings, data)
        message = f"Restored the settings of '{name}'"
        if self.include_setup and setup:
            restored, recreated, disabled, missing = apply_setup_list(context, setup)
            message += f" and {restored + recreated} helper(s)"
            if disabled:
                message += f" ({disabled} newer helper(s) disabled)"
            if missing:
                self.report({"WARNING"}, f"{missing} helper mesh(es) of '{name}' no longer exist")
        if skipped:
            print(f"Stress Sculpt: settings not restored: {', '.join(skipped)}")
        self.report({"INFO"}, message)
        return {"FINISHED"}


class STRESS_SCULPT_OT_variant_compare(Operator):
    bl_idname = "stress_sculpt.variant_compare"
    bl_label = "Compare Variants"
    bl_description = (
        "Show all variants side by side (to the right of the domain) with their metrics"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(_settings(context).variants) > 0

    def execute(self, context):
        count = build_compare(context, context.scene)
        if not count:
            self.report({"ERROR"}, "The variants have no objects to compare")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Comparing {count} variant(s)")
        return {"FINISHED"}


class STRESS_SCULPT_OT_variant_clear_compare(Operator):
    bl_idname = "stress_sculpt.variant_clear_compare"
    bl_label = "Clear Comparison"
    bl_description = "Remove the side-by-side comparison"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        count = clear_compare(context.scene)
        self.report({"INFO"}, f"Removed {count} comparison object(s)")
        return {"FINISHED"}


class STRESS_SCULPT_OT_variant_move(Operator):
    bl_idname = "stress_sculpt.variant_move"
    bl_label = "Move Variant"
    bl_description = "Move the active variant up or down in the list"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(
        name="Direction",
        items=[("UP", "Up", "Move towards the top"), ("DOWN", "Down", "Move towards the bottom")],
        default="UP",
    )

    @classmethod
    def poll(cls, context):
        return len(_settings(context).variants) > 1

    def execute(self, context):
        settings = _settings(context)
        index = settings.active_variant_index
        target = index - 1 if self.direction == "UP" else index + 1
        if not (0 <= index < len(settings.variants) and 0 <= target < len(settings.variants)):
            return {"CANCELLED"}
        settings.variants.move(index, target)
        settings.active_variant_index = target
        return {"FINISHED"}


# ===================================================================== registration
classes = (
    STRESS_SCULPT_OT_set_domain,
    STRESS_SCULPT_OT_add_support,
    STRESS_SCULPT_OT_add_load,
    STRESS_SCULPT_OT_add_zone,
    STRESS_SCULPT_OT_set_role,
    STRESS_SCULPT_OT_select_role,
    STRESS_SCULPT_OT_set_target,
    STRESS_SCULPT_OT_add_load_at_selection,
    STRESS_SCULPT_OT_add_support_at_selection,
    STRESS_SCULPT_OT_remove_role,
    STRESS_SCULPT_OT_demo_scene,
    STRESS_SCULPT_OT_generate,
    STRESS_SCULPT_OT_verify,
    STRESS_SCULPT_OT_analyze,
    STRESS_SCULPT_OT_evolve,
    STRESS_SCULPT_OT_trace_lines,
    STRESS_SCULPT_OT_sculpt,
    STRESS_SCULPT_OT_relief,
    STRESS_SCULPT_OT_cancel,
    STRESS_SCULPT_OT_clear_results,
    STRESS_SCULPT_OT_variant_save,
    STRESS_SCULPT_OT_variant_remove,
    STRESS_SCULPT_OT_variant_restore,
    STRESS_SCULPT_OT_variant_compare,
    STRESS_SCULPT_OT_variant_clear_compare,
    STRESS_SCULPT_OT_variant_move,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    jobs.register()


def unregister():
    jobs.unregister()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
