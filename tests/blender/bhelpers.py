"""Scene-building helpers shared by the Blender-layer tests."""

import math

import numpy as np
import pytest

import bpy
import bmesh


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


def bridge(resolution=16, preset="BRIDGE"):
    """Demo scene at a low resolution; returns the scene settings."""
    settings = bpy.context.scene.stress_sculpt
    settings.resolution = resolution
    assert bpy.ops.stress_sculpt.demo_scene(preset=preset) == {"FINISHED"}
    return settings


def mesh_attribute_values(mesh, name, field, width=1):
    attribute = mesh.attributes[name]
    out = np.empty(len(attribute.data) * width, dtype=np.float32)
    attribute.data.foreach_get(field, out)
    return out.reshape(-1, width) if width > 1 else out
