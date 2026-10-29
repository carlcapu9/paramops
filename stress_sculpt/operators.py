"""Operators of Stress Sculpt (``bpy.ops.stress_sculpt.*``).

* Setup: ``set_domain``, ``add_support``, ``add_load``, ``add_zone``, ``set_role``,
  ``select_role``, ``demo_scene``.
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

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Operator

from . import build, jobs, scene_io
from .core.grid import check_stop
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


# ------------------------------------------------------------------ demo scenes
DEMO_DOMAINS = {  # world box (min, max) of the demo domain, centred above the origin
    "BRIDGE": ((-2.0, -0.5, 0.0), (2.0, 0.5, 1.5)),
    "CANTILEVER": ((-2.0, -0.5, 0.0), (2.0, 0.5, 2.0)),
    "TOWER": ((-0.6, -0.6, 0.0), (0.6, 0.6, 4.0)),
    "CHAIR": ((-0.5, -0.5, 0.0), (0.5, 0.5, 2.0)),
}


def _demo_specs(preset, bmin, bmax) -> list:
    """Supports / loads / zones of a demo preset fitted to the box ``bmin``..``bmax``.

    Every entry: ``(name, role, display, location, rotation, scale, settings)``;
    empties have display size 1, the region size comes from the scale.
    """
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    size = bmax - bmin
    centre = 0.5 * (bmin + bmax)
    longest = float(size.max())
    down = (math.pi, 0.0, 0.0)
    specs = []
    if preset == "BRIDGE":
        for side, fix_x in ((-1.0, True), (1.0, False)):
            x = (bmin[0] + 0.05 * size[0]) if side < 0 else (bmax[0] - 0.05 * size[0])
            specs.append(
                (
                    "SS_Support",
                    "SUPPORT",
                    "CUBE",
                    (x, centre[1], bmin[2]),
                    (0.0, 0.0, 0.0),
                    (0.0625 * size[0], 0.6 * size[1], 0.1 * size[2]),
                    {"fix_x": fix_x, "fix_y": True, "fix_z": True},
                )
            )
        specs.append(
            (
                "SS_Load",
                "LOAD",
                "SINGLE_ARROW",
                (centre[0], centre[1], bmax[2]),
                down,
                (0.15 * longest,) * 3,
                {"magnitude": 5000.0, "load_radius": 0.06 * longest},
            )
        )
    elif preset == "CANTILEVER":
        specs.append(
            (
                "SS_Support",
                "SUPPORT",
                "CUBE",
                (bmin[0], centre[1], centre[2]),
                (0.0, 0.0, 0.0),
                (0.04 * size[0], 0.6 * size[1], 0.6 * size[2]),
                {"fix_x": True, "fix_y": True, "fix_z": True},
            )
        )
        specs.append(
            (
                "SS_Load",
                "LOAD",
                "SINGLE_ARROW",
                (bmax[0], centre[1], centre[2]),
                down,
                (0.15 * longest,) * 3,
                {"magnitude": 5000.0, "load_radius": 0.06 * longest},
            )
        )
    elif preset == "TOWER":
        specs.append(
            (
                "SS_Support",
                "SUPPORT",
                "CUBE",
                (centre[0], centre[1], bmin[2]),
                (0.0, 0.0, 0.0),
                (0.6 * size[0], 0.6 * size[1], 0.04 * size[2]),
                {"fix_x": True, "fix_y": True, "fix_z": True},
            )
        )
        specs.append(
            (
                "SS_Load",
                "LOAD",
                "SINGLE_ARROW",
                (centre[0], centre[1], bmax[2]),
                down,
                (0.1 * longest,) * 3,
                {"magnitude": 5000.0, "load_radius": 0.1 * longest},
            )
        )
        specs.append(
            (
                "SS_Load",
                "LOAD",
                "SINGLE_ARROW",
                (centre[0], centre[1], bmax[2] - 0.05 * size[2]),
                (0.0, math.pi / 2.0, 0.0),
                (0.1 * longest,) * 3,
                {"magnitude": 2000.0, "load_radius": 0.1 * longest},
            )
        )
    elif preset == "CHAIR":
        for u in (0.08, 0.92):
            for v in (0.08, 0.92):
                specs.append(
                    (
                        "SS_Support",
                        "SUPPORT",
                        "CUBE",
                        (bmin[0] + u * size[0], bmin[1] + v * size[1], bmin[2]),
                        (0.0, 0.0, 0.0),
                        (0.08 * size[0], 0.08 * size[1], 0.03 * size[2]),
                        {"fix_x": True, "fix_y": True, "fix_z": True},
                    )
                )
        seat = bmin[2] + 0.5 * size[2]
        void_bottom = seat + 0.025 * size[2]
        void_y1 = bmin[1] + 0.8 * size[1]
        specs.append(
            (  # room for the person: everything in front of the backrest above the seat
                "SS_KeepVoid",
                "KEEP_VOID",
                "CUBE",
                (
                    centre[0],
                    0.5 * (bmin[1] - 0.1 * size[1] + void_y1),
                    0.5 * (void_bottom + bmax[2] + 0.05 * size[2]),
                ),
                (0.0, 0.0, 0.0),
                (
                    0.6 * size[0],
                    0.5 * (void_y1 - bmin[1] + 0.1 * size[1]),
                    0.5 * (bmax[2] + 0.05 * size[2] - void_bottom),
                ),
                {},
            )
        )
        plate_bottom = seat - 0.06 * size[2]
        specs.append(
            (  # the seat plate stays solid (at least one voxel thick from resolution 16)
                "SS_KeepSolid",
                "KEEP_SOLID",
                "CUBE",
                (centre[0], 0.5 * (bmin[1] + void_y1), 0.5 * (plate_bottom + void_bottom)),
                (0.0, 0.0, 0.0),
                (0.5 * size[0], 0.5 * (void_y1 - bmin[1]), 0.5 * (void_bottom - plate_bottom)),
                {},
            )
        )
        specs.append(
            (
                "SS_Load",
                "LOAD",
                "SINGLE_ARROW",
                (centre[0], bmin[1] + 0.4 * size[1], seat - 0.025 * size[2]),
                down,
                (0.15 * longest,) * 3,
                {"magnitude": 1000.0, "load_radius": 0.05 * longest},
            )
        )
        specs.append(
            (
                "SS_Load",
                "LOAD",
                "SINGLE_ARROW",
                (centre[0], bmin[1] + 0.9 * size[1], bmin[2] + 0.85 * size[2]),
                (-math.pi / 2.0, 0.0, 0.0),
                (0.1 * longest,) * 3,
                {"magnitude": 300.0, "load_radius": 0.05 * longest},
            )
        )
    else:
        raise ValueError(f"Unknown demo preset '{preset}'")
    return specs


def _box_mesh(name, half):
    """Closed box mesh centred on the origin with outward normals."""
    hx, hy, hz = half
    verts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5]],
        dtype=np.int64,
    )
    return build.mesh_from_faces(name, verts, faces, smooth=False)


class STRESS_SCULPT_OT_demo_scene(Operator):
    bl_idname = "stress_sculpt.demo_scene"
    bl_label = "Demo Scene"
    bl_description = (
        "Build a ready-to-run example (domain, supports and loads). An existing domain of yours is "
        "kept and the supports and loads are fitted to it"
    )
    bl_options = {"REGISTER", "UNDO"}

    preset: EnumProperty(
        name="Preset",
        items=[
            (
                "BRIDGE",
                "Bridge",
                "Block resting on two supports, pressed down in the middle",
                "MOD_SIMPLEDEFORM",
                0,
            ),
            (
                "CANTILEVER",
                "Cantilever",
                "Block fixed to a wall on one side, loaded at the free tip",
                "MOD_LATTICE",
                1,
            ),
            (
                "TOWER",
                "Tower",
                "Tall block fixed at the base, loaded from above and from the side",
                "MOD_BUILD",
                2,
            ),
            (
                "CHAIR",
                "Chair",
                "Seat and backrest loads on four feet, with a void zone for the sitter",
                "OUTLINER_OB_POINTCLOUD",
                3,
            ),
        ],
        default="BRIDGE",
    )

    def execute(self, context):
        if RUNTIME.busy():
            self.report({"ERROR"}, "Wait for the running job to finish (or cancel it)")
            return {"CANCELLED"}
        scene = context.scene
        settings = scene.stress_sculpt
        domain = settings.domain
        replace_domain = domain is None or domain.get(DEMO_TAG) is not None or domain.type != "MESH"

        # Remove what a previous demo created (keeping a user domain).
        previous = [obj for obj in scene.objects if obj.get(DEMO_TAG) is not None and obj != domain]
        if replace_domain and domain is not None and domain.get(DEMO_TAG) is not None:
            previous.append(domain)
        build.remove_objects(previous)

        setup = build.ensure_collection(scene, build.SETUP_COLLECTION)
        if replace_domain:
            bmin, bmax = (np.asarray(v, dtype=np.float64) for v in DEMO_DOMAINS[self.preset])
            mesh = _box_mesh("SS_Domain", 0.5 * (bmax - bmin))
            domain = bpy.data.objects.new("SS_Domain", mesh)
            domain.location = tuple(0.5 * (bmin + bmax))
            domain[DEMO_TAG] = self.preset
            setup.objects.link(domain)
            settings.domain = domain
        else:
            bounds = _world_bounds(context, domain)
            if bounds is None:
                self.report({"ERROR"}, f"The domain '{domain.name}' has no geometry")
                return {"CANCELLED"}
            bmin, bmax = bounds

        created = []
        for name, role, display, location, rotation, scale, values in _demo_specs(
            self.preset, bmin, bmax
        ):
            color = {"SUPPORT": build.SUPPORT_COLOR, "LOAD": build.LOAD_COLOR}.get(
                role, ZONE_COLORS.get(role)
            )
            obj = new_role_empty(scene, name, role, display, 1.0, location, rotation, scale, color)
            for key, value in values.items():
                setattr(obj.stress_sculpt, key, value)
            obj[DEMO_TAG] = self.preset
            created.append(obj)

        view_layer = context.view_layer
        _deselect_all(context)
        try:
            domain.select_set(True, view_layer=view_layer)
            view_layer.objects.active = domain
        except RuntimeError:
            pass
        label = self.preset.title()
        self.report(
            {"INFO"}, f"Demo '{label}' ready ({len(created)} helpers): press Analyze or Evolve"
        )
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
        optimizer = topopt.Optimizer(problem, opt_settings)
        max_iter = max(1, int(opt_settings.max_iter))
        progress(0.0, "Iteration 1: solving")
        snapshot = state = None
        for _ in range(max_iter):  # bounded even if the optimizer never reports done
            state = _copy_state(optimizer.step(should_stop=should_stop))
            snapshot = {"payload": payload, "state": state, "preview": None}
            if payload["preview"]:
                snapshot["preview"] = evolved_geometry(
                    state, problem, params, mesher, 2, clean=False
                )
            emit(snapshot)
            progress(
                min(state.iteration / max_iter, 1.0),
                f"Iteration {state.iteration}/{max_iter}: material {state.volume * 100:.0f}%, "
                f"change {state.change:.3f}",
            )
            if optimizer.done or state.converged or state.iteration >= max_iter:
                break
            check_stop(should_stop)
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
        if settings.domain is not None:
            settings.domain.display_type = "TEXTURED"
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
                {"ERROR"}, "Nothing to save: run Analyze, Evolve, Lines, Sculpt or Relief first"
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
    STRESS_SCULPT_OT_demo_scene,
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
