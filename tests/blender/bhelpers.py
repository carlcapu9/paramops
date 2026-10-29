"""Scene-building helpers shared by the Blender-layer tests."""

import math

import numpy as np
import pytest

import bpy
import bmesh
from mathutils import Vector


def require(*names):
    """Skip the test when a core module (``stress_sculpt.core.<name>``) is missing."""
    for name in names:
        pytest.importorskip(
            f"stress_sculpt.core.{name}", reason=f"core/{name}.py not available yet"
        )


# ------------------------------------------------------------------ scene helpers
def link(obj, scene=None):
    (scene or bpy.context.scene).collection.objects.link(obj)
    return obj


def mesh_object(name, bm, location=(0.0, 0.0, 0.0), scene=None):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    return link(obj, scene)


def uv_sphere(name="Sphere", radius=1.0, segments=48, rings=24, location=(0.0, 0.0, 0.0)):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    return mesh_object(name, bm, location)


def box(name="Box", size=(1.0, 1.0, 1.0), location=(0.0, 0.0, 0.0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    return mesh_object(name, bm, location)


def torus(name="Torus", major=1.0, minor=0.35, major_segments=48, minor_segments=24):
    """Torus around +Z built from a parametric quad grid (closed, outward normals)."""
    u = np.linspace(0.0, 2.0 * math.pi, major_segments, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, minor_segments, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    verts = np.stack(
        [
            (major + minor * np.cos(V)) * np.cos(U),
            (major + minor * np.cos(V)) * np.sin(U),
            minor * np.sin(V),
        ],
        axis=-1,
    ).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(major_segments), np.arange(minor_segments), indexing="ij")
    a = i * minor_segments + j
    b = ((i + 1) % major_segments) * minor_segments + j
    c = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
    d = i * minor_segments + (j + 1) % minor_segments
    faces = np.stack([a, b, c, d], axis=-1).reshape(-1, 4)
    from stress_sculpt import build

    mesh = build.mesh_from_faces(name, verts, faces, smooth=False)
    obj = bpy.data.objects.new(name, mesh)
    return link(obj)


def empty(
    name,
    role,
    display="SPHERE",
    size=1.0,
    location=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    **role_settings,
):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display
    obj.empty_display_size = size
    obj.location = location
    obj.rotation_euler = rotation
    obj.scale = scale
    obj.stress_sculpt.role = role
    for key, value in role_settings.items():
        setattr(obj.stress_sculpt, key, value)
    return link(obj)


# ------------------------------------------------------- classic analysis setups
# The original (v1) demo scenes: a box domain with supports / loads / zones, used by the
# analysis, evolution, lines, sculpture and relief tests.
CLASSIC_DOMAINS = {  # world box (min, max) of the domain, centred above the origin
    "BRIDGE": ((-2.0, -0.5, 0.0), (2.0, 0.5, 1.5)),
    "CANTILEVER": ((-2.0, -0.5, 0.0), (2.0, 0.5, 2.0)),
    "TOWER": ((-0.6, -0.6, 0.0), (0.6, 0.6, 4.0)),
    "CHAIR": ((-0.5, -0.5, 0.0), (0.5, 0.5, 2.0)),
}
CLASSIC_PRESETS = tuple(CLASSIC_DOMAINS)


def _classic_specs(preset, bmin, bmax):
    """``(name, role, display, location, rotation, scale, settings)`` of a classic setup."""
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
            specs.append(("SS_Support", "SUPPORT", "CUBE", (x, centre[1], bmin[2]), (0, 0, 0),
                          (0.0625 * size[0], 0.6 * size[1], 0.1 * size[2]),
                          {"fix_x": fix_x, "fix_y": True, "fix_z": True}))
        specs.append(("SS_Load", "LOAD", "SINGLE_ARROW", (centre[0], centre[1], bmax[2]), down,
                      (0.15 * longest,) * 3, {"magnitude": 5000.0, "load_radius": 0.06 * longest}))
    elif preset == "CANTILEVER":
        specs.append(("SS_Support", "SUPPORT", "CUBE", (bmin[0], centre[1], centre[2]), (0, 0, 0),
                      (0.04 * size[0], 0.6 * size[1], 0.6 * size[2]),
                      {"fix_x": True, "fix_y": True, "fix_z": True}))
        specs.append(("SS_Load", "LOAD", "SINGLE_ARROW", (bmax[0], centre[1], centre[2]), down,
                      (0.15 * longest,) * 3, {"magnitude": 5000.0, "load_radius": 0.06 * longest}))
    elif preset == "TOWER":
        specs.append(("SS_Support", "SUPPORT", "CUBE", (centre[0], centre[1], bmin[2]), (0, 0, 0),
                      (0.6 * size[0], 0.6 * size[1], 0.04 * size[2]),
                      {"fix_x": True, "fix_y": True, "fix_z": True}))
        specs.append(("SS_Load", "LOAD", "SINGLE_ARROW", (centre[0], centre[1], bmax[2]), down,
                      (0.1 * longest,) * 3, {"magnitude": 5000.0, "load_radius": 0.1 * longest}))
        specs.append(("SS_Load", "LOAD", "SINGLE_ARROW",
                      (centre[0], centre[1], bmax[2] - 0.05 * size[2]), (0.0, math.pi / 2.0, 0.0),
                      (0.1 * longest,) * 3, {"magnitude": 2000.0, "load_radius": 0.1 * longest}))
    elif preset == "CHAIR":
        for u in (0.08, 0.92):
            for v in (0.08, 0.92):
                specs.append(("SS_Support", "SUPPORT", "CUBE",
                              (bmin[0] + u * size[0], bmin[1] + v * size[1], bmin[2]), (0, 0, 0),
                              (0.08 * size[0], 0.08 * size[1], 0.03 * size[2]),
                              {"fix_x": True, "fix_y": True, "fix_z": True}))
        seat = bmin[2] + 0.5 * size[2]
        void_bottom = seat + 0.025 * size[2]
        void_y1 = bmin[1] + 0.8 * size[1]
        specs.append(("SS_KeepVoid", "KEEP_VOID", "CUBE",
                      (centre[0], 0.5 * (bmin[1] - 0.1 * size[1] + void_y1),
                       0.5 * (void_bottom + bmax[2] + 0.05 * size[2])), (0, 0, 0),
                      (0.6 * size[0], 0.5 * (void_y1 - bmin[1] + 0.1 * size[1]),
                       0.5 * (bmax[2] + 0.05 * size[2] - void_bottom)), {}))
        plate_bottom = seat - 0.06 * size[2]
        specs.append(("SS_KeepSolid", "KEEP_SOLID", "CUBE",
                      (centre[0], 0.5 * (bmin[1] + void_y1), 0.5 * (plate_bottom + void_bottom)),
                      (0, 0, 0),
                      (0.5 * size[0], 0.5 * (void_y1 - bmin[1]), 0.5 * (void_bottom - plate_bottom)),
                      {}))
        specs.append(("SS_Load", "LOAD", "SINGLE_ARROW",
                      (centre[0], bmin[1] + 0.4 * size[1], seat - 0.025 * size[2]), down,
                      (0.15 * longest,) * 3, {"magnitude": 1000.0, "load_radius": 0.05 * longest}))
        specs.append(("SS_Load", "LOAD", "SINGLE_ARROW",
                      (centre[0], bmin[1] + 0.9 * size[1], bmin[2] + 0.85 * size[2]),
                      (-math.pi / 2.0, 0.0, 0.0), (0.1 * longest,) * 3,
                      {"magnitude": 300.0, "load_radius": 0.05 * longest}))
    else:
        raise ValueError(preset)
    return specs


def classic_setup(preset="BRIDGE", resolution=16, domain=None):
    """A classic analysis setup (box domain + helpers); returns the scene settings.

    With ``domain`` the helpers are fitted to that mesh instead of a new box.
    """
    from stress_sculpt import build, operators

    scene = bpy.context.scene
    settings = scene.stress_sculpt
    settings.resolution = resolution
    if domain is None:
        bmin, bmax = (np.asarray(v, dtype=np.float64) for v in CLASSIC_DOMAINS[preset])
        half = 0.5 * (bmax - bmin)
        mesh = operators.grid_box_mesh("SS_Domain", 2.0 * half, (1, 1, 1))
        domain = bpy.data.objects.new("SS_Domain", mesh)
        domain.location = tuple(0.5 * (bmin + bmax))
        build.ensure_collection(scene, build.SETUP_COLLECTION).objects.link(domain)
    else:
        bpy.context.view_layer.update()
        corners = np.array([tuple(domain.matrix_world @ Vector(c)) for c in domain.bound_box])
        bmin, bmax = corners.min(axis=0), corners.max(axis=0)
    settings.domain = domain
    for name, role, display, location, rotation, scale, values in _classic_specs(preset, bmin, bmax):
        obj = empty(name, role, display, 1.0, location, rotation, scale, **values)
        if role == "SUPPORT":
            obj.color = build.SUPPORT_COLOR
        elif role == "LOAD":
            obj.color = build.LOAD_COLOR
    bpy.context.view_layer.update()
    return settings


def bridge(resolution=16, preset="BRIDGE"):
    """Classic analysis setup at a low resolution; returns the scene settings."""
    return classic_setup(preset, resolution)


def mesh_attribute_values(mesh, name, field, width=1):
    attribute = mesh.attributes[name]
    out = np.empty(len(attribute.data) * width, dtype=np.float32)
    attribute.data.foreach_get(field, out)
    return out.reshape(-1, width) if width > 1 else out
