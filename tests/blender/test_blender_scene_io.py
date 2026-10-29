"""scene_io: voxelization, region tests, gather_problem and setup_signature."""

import math

import numpy as np
import pytest

import bpy
from bhelpers import box, bridge, empty, link, require, torus, uv_sphere

from stress_sculpt import scene_io
from stress_sculpt.core.grid import Grid


def _depsgraph():
    return bpy.context.evaluated_depsgraph_get()


# ------------------------------------------------------------------ voxelization
def test_voxelize_uv_sphere_volume(scene):
    obj = uv_sphere(radius=1.0, segments=64, rings=32, location=(0.3, -0.2, 0.5))
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    assert verts.dtype == np.float64 and tris.dtype == np.int64
    assert np.allclose(verts.mean(axis=0), (0.3, -0.2, 0.5), atol=1e-3)
    grid = scene_io.make_grid(obj, _depsgraph(), 40)
    inside = scene_io.voxelize_mesh(verts, tris, grid)
    volume = inside.sum() * grid.h**3
    exact = 4.0 / 3.0 * math.pi
    assert abs(volume - exact) / exact < 0.05
    # every voxel centre inside is really inside (up to the polygon approximation)
    centres = grid.element_centers()[inside] - (0.3, -0.2, 0.5)
    assert np.linalg.norm(centres, axis=1).max() < 1.0 + 1e-9


def test_voxelize_torus_has_hole(scene):
    obj = torus(major=1.0, minor=0.35)
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    grid = Grid.from_bounds(verts.min(0), verts.max(0), 48)
    inside = scene_io.voxelize_mesh(verts, tris, grid)
    volume = inside.sum() * grid.h**3
    exact = 2.0 * math.pi**2 * 1.0 * 0.35**2
    assert abs(volume - exact) / exact < 0.06
    centre = grid.world_to_index(np.zeros(3)).round().astype(int)
    assert not inside[centre[0], centre[1], :].any()  # the hole is empty along the axis


def test_voxelize_box_rays_through_edges_and_vertices(scene):
    # Rays at x = y run exactly along the diagonals of the triangulated top / bottom
    # faces and through vertices: crossings must be counted once.
    obj = box(size=(2.0, 2.0, 2.0))
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    grid = Grid(shape=(4, 4, 4), origin=(-1.0, -1.0, -1.0), h=0.5)
    assert scene_io.voxelize_mesh(verts, tris, grid).all()
    # A grid whose columns pass exactly through the box's vertical edges / faces.
    grid = Grid(shape=(5, 5, 5), origin=(-1.25, -1.25, -1.25), h=0.5)
    inside = scene_io.voxelize_mesh(verts, tris, grid)
    assert inside[1:4, 1:4, 1:4].all()
    assert inside.sum() >= 27


def test_voxelize_centre_exactly_on_faces(scene):
    # Element centres lie exactly on the bottom and top faces: entering faces count
    # as inside, leaving faces as outside, consistently in every column.
    obj = box(size=(2.0, 2.0, 2.0))
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    grid = Grid(shape=(4, 4, 5), origin=(-1.0, -1.0, -1.25), h=0.5)  # z centres -1, -0.5, 0, 0.5, 1
    inside = scene_io.voxelize_mesh(verts, tris, grid)
    assert inside[:, :, 0].all() and inside[:, :, 1:4].all()
    assert not inside[:, :, 4].any()


def test_voxelize_empty_and_outside(scene):
    grid = Grid(shape=(3, 3, 3), origin=(10.0, 10.0, 10.0), h=1.0)
    obj = uv_sphere(radius=1.0)
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    assert not scene_io.voxelize_mesh(verts, tris, grid).any()
    assert not scene_io.voxelize_mesh(np.zeros((0, 3)), np.zeros((0, 3), int), grid).any()


def test_voxelize_open_mesh_does_not_leak(scene):
    # A sphere with its top cap removed: the inside test stays bounded in z.
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
    top = [f for f in bm.faces if f.calc_center_median().z > 0.9]
    bmesh.ops.delete(bm, geom=top, context="FACES")
    mesh = bpy.data.meshes.new("open")
    bm.to_mesh(mesh)
    bm.free()
    obj = link(bpy.data.objects.new("open", mesh))
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    grid = Grid.from_bounds((-1, -1, -1), (1, 1, 1), 20)
    inside, n_ambiguous = scene_io._voxelize(verts, tris, grid)
    assert n_ambiguous > 0
    zc = grid.element_centers()[..., 2]
    assert not inside[zc > 1.0].any()


def test_points_in_mesh_matches_sphere(scene):
    obj = uv_sphere(radius=1.0, segments=64, rings=32)
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.3, 1.3, size=(4000, 3))
    radius = np.linalg.norm(points, axis=1)
    clear = np.abs(radius - 1.0) > 0.02  # away from the polygonal approximation
    inside = scene_io.points_in_mesh(verts, tris, points)
    assert np.array_equal(inside[clear], radius[clear] < 1.0)
    # repeated (x, y) columns share rays
    column = np.array([[0.1, 0.2, z] for z in np.linspace(-1.5, 1.5, 31)])
    inside = scene_io.points_in_mesh(verts, tris, column)
    assert np.array_equal(inside, np.linalg.norm(column, axis=1) < 0.995)


def test_voxelize_speed(scene):
    import time

    obj = uv_sphere(radius=1.0, segments=64, rings=32)
    verts, tris = scene_io.evaluated_world_mesh(obj, _depsgraph())
    grid = Grid.from_bounds(verts.min(0), verts.max(0), 80)
    start = time.perf_counter()
    inside = scene_io.voxelize_mesh(verts, tris, grid)
    elapsed = time.perf_counter() - start
    assert inside.sum() > 0
    assert elapsed < 5.0


# ----------------------------------------------------------------------- regions
def test_region_mask_sphere_and_cube_with_transform(scene):
    sphere = empty(
        "S", "SUPPORT", "SPHERE", size=0.5, location=(1.0, 0.0, 0.0), scale=(2.0, 1.0, 1.0)
    )
    bpy.context.view_layer.update()
    points = np.array([[1.95, 0.0, 0.0], [2.05, 0.0, 0.0], [1.0, 0.45, 0.0], [1.0, 0.55, 0.0]])
    assert scene_io.region_mask_points(sphere, points).tolist() == [True, False, True, False]

    cube = empty("C", "SUPPORT", "CUBE", size=1.0, rotation=(0.0, 0.0, math.pi / 4.0))
    bpy.context.view_layer.update()
    s = math.sqrt(0.5)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.45, 0.0, 0.0], [s, s, 0.0], [0.0, 0.0, 1.01]]
    )
    assert scene_io.region_mask_points(cube, points).tolist() == [True, True, False, True, False]


# ------------------------------------------------------------------------ gather
def test_gather_bridge(scene):
    require("fem")
    settings = bridge(resolution=16)
    problem, info = scene_io.gather_problem(bpy.context)
    grid = problem.grid
    assert grid.shape == (16, 4, 6)
    assert info["n_elems"] == grid.n_elems == int(problem.domain.sum())  # a box fills every voxel
    assert info["domain_name"] == settings.domain.name == "SS_Domain"
    forces = problem.forces.reshape(-1, 3)
    assert forces.sum(axis=0) == pytest.approx([0.0, 0.0, -5000.0], abs=1e-2)
    fixed = problem.fixed_dofs
    nodes = np.unique(fixed // 3)
    xyz = grid.origin + np.stack(np.unravel_index(nodes, grid.nodes_shape), axis=1) * grid.h
    assert (xyz[:, 0] < -1.5).any() and (xyz[:, 0] > 1.5).any()  # both ends
    assert np.all(xyz[:, 2] < 0.2)  # bottom only
    right = nodes[xyz[:, 0] > 0.0]
    assert not np.isin(3 * right, fixed).any()  # the right support is a roller in X
    assert np.isin(3 * right + 2, fixed).all() and np.isin(3 * right + 1, fixed).all()
    assert info["seed_points"].shape == (3, 3)
    assert len(info["loads"]) == 1 and len(info["supports"]) == 2
    position, direction, magnitude = info["loads"][0]
    assert magnitude == 5000.0 and direction == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)
    assert problem.E == pytest.approx(settings.youngs_modulus * 1e9)
    assert problem.nu == pytest.approx(settings.poisson_ratio)


def test_gather_all_demo_presets(scene):
    require("fem")
    for preset in ("BRIDGE", "CANTILEVER", "TOWER", "CHAIR"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bridge(resolution=16, preset=preset)
        problem, info = scene_io.gather_problem(bpy.context)
        assert (
            problem.domain.any() and problem.fixed_dofs.size > 0 and np.any(problem.forces)
        ), preset
        assert not info["warnings"], (preset, info["warnings"])
        if preset == "CHAIR":
            assert problem.passive_void is not None and problem.passive_void.any()
            assert problem.passive_solid is not None and problem.passive_solid.any()


def test_gather_errors(scene):
    require("fem")
    with pytest.raises(ValueError, match="domain"):
        scene_io.gather_problem(bpy.context)
    domain = box("Domain", size=(2.0, 1.0, 1.0))
    scene.stress_sculpt.domain = domain
    with pytest.raises(ValueError, match="No supports"):
        scene_io.gather_problem(bpy.context)
    empty("Fix", "SUPPORT", "CUBE", size=0.2, location=(-1.0, 0.0, 0.0), scale=(1.0, 5.0, 5.0))
    with pytest.raises(ValueError, match="No loads"):
        scene_io.gather_problem(bpy.context)
    load = empty(
        "Push",
        "LOAD",
        "SINGLE_ARROW",
        location=(1.0, 0.0, 0.5),
        rotation=(math.pi, 0, 0),
        magnitude=10.0,
    )
    problem, info = scene_io.gather_problem(bpy.context)
    assert np.any(problem.forces)
    load.stress_sculpt.enabled = False
    with pytest.raises(ValueError, match="No loads"):
        scene_io.gather_problem(bpy.context)


def test_gather_nearest_node_fallbacks(scene):
    require("fem")
    domain = box("Domain", size=(2.0, 1.0, 1.0))
    scene.stress_sculpt.domain = domain
    scene.stress_sculpt.resolution = 10
    empty("Far", "SUPPORT", "SPHERE", size=0.05, location=(-1.5, 0.0, 0.0))
    empty("Load", "LOAD", "SINGLE_ARROW", location=(1.5, 0.0, 0.0), magnitude=100.0)
    problem, info = scene_io.gather_problem(bpy.context)
    assert any("does not touch the domain" in w for w in info["warnings"])
    assert any("outside the domain" in w for w in info["warnings"])
    nodes = np.unique(problem.fixed_dofs // 3)
    xyz = (
        problem.grid.origin
        + np.stack(np.unravel_index(nodes, problem.grid.nodes_shape), 1) * problem.grid.h
    )
    assert np.allclose(xyz[:, 0], -1.0)  # nearest face of the domain
    assert problem.forces.reshape(-1, 3).sum(axis=0) == pytest.approx([0.0, 0.0, 100.0], abs=1e-6)


def test_gather_zones(scene):
    require("fem")
    domain = box("Domain", size=(2.0, 1.0, 1.0))
    settings = scene.stress_sculpt
    settings.domain = domain
    settings.resolution = 20
    empty("Fix", "SUPPORT", "CUBE", size=0.1, location=(-1.0, 0.0, 0.0), scale=(1.0, 6.0, 6.0))
    empty(
        "Load",
        "LOAD",
        "SINGLE_ARROW",
        location=(1.0, 0.0, 0.5),
        rotation=(math.pi, 0, 0),
        magnitude=100.0,
    )
    empty("Solid", "KEEP_SOLID", "SPHERE", size=0.2, location=(0.5, 0.0, 0.0))
    empty("Void", "KEEP_VOID", "CUBE", size=0.15, location=(-0.5, 0.0, 0.0))
    empty("Thick", "THICKEN", "SPHERE", size=0.3, location=(0.0, 0.0, 0.0), strength=0.8)
    empty("Thin", "HOLLOW", "SPHERE", size=0.3, location=(0.2, 0.0, 0.0), strength=0.5)
    empty("Branch", "BRANCH", "SPHERE", size=0.25, location=(-0.3, 0.0, 0.2), strength=0.7)
    zone_mesh = uv_sphere("ZoneMesh", radius=0.2, segments=24, rings=12, location=(0.7, 0.0, 0.2))
    zone_mesh.stress_sculpt.role = "HOLLOW"
    zone_mesh.stress_sculpt.strength = 1.0
    problem, info = scene_io.gather_problem(bpy.context)
    grid = problem.grid
    centres = grid.element_centers()

    def at(point):
        i, j, k = np.floor((np.asarray(point) - grid.origin) / grid.h).astype(int)
        return i, j, k

    assert problem.passive_solid[at((0.5, 0.0, 0.0))]
    assert problem.passive_void[at((-0.5, 0.0, 0.0))]
    assert problem.bias[at((-0.15, 0.0, 0.0))] == pytest.approx(0.8)
    assert problem.bias[at((0.05, 0.0, 0.0))] == pytest.approx(0.3)  # 0.8 - 0.5 overlap
    assert problem.bias[at((0.4, 0.0, 0.0))] == pytest.approx(-0.5)
    assert problem.bias[at((0.75, 0.0, 0.25))] == pytest.approx(-1.0)  # the mesh zone
    assert problem.branch[at((-0.3, 0.0, 0.2))] == pytest.approx(0.7)
    assert problem.branch_or_zero().max() == pytest.approx(0.7)
    # membership is decided by element centres
    expected = np.linalg.norm(centres - (0.5, 0.0, 0.0), axis=-1) <= 0.2
    assert np.array_equal(problem.passive_solid, expected & problem.domain)


def test_gather_bounds_method_fills_bbox(scene):
    require("fem")
    sphere = uv_sphere("Ball", radius=1.0, segments=32, rings=16)
    settings = scene.stress_sculpt
    settings.domain = sphere
    settings.resolution = 10
    empty("Fix", "SUPPORT", "SPHERE", size=0.4, location=(0.0, 0.0, -1.0))
    empty("Load", "LOAD", "SINGLE_ARROW", location=(0.0, 0.0, 1.0), magnitude=10.0)
    problem, _info = scene_io.gather_problem(bpy.context)
    assert problem.domain.sum() < problem.grid.n_elems
    settings.voxelize_method = "BOUNDS"
    problem, _info = scene_io.gather_problem(bpy.context)
    assert problem.domain.all()


def test_gather_removes_unsupported_parts(scene):
    import bmesh

    require("fem")

    # Two separate boxes in one mesh; only the left one is supported.
    bm = bmesh.new()
    for x in (-1.0, 1.0):
        geom = bmesh.ops.create_cube(bm, size=0.8)
        bmesh.ops.translate(bm, vec=(x, 0.0, 0.0), verts=geom["verts"])
    mesh = bpy.data.meshes.new("TwoParts")
    bm.to_mesh(mesh)
    bm.free()
    domain = link(bpy.data.objects.new("TwoParts", mesh))
    settings = scene.stress_sculpt
    settings.domain = domain
    settings.resolution = 24
    empty("Fix", "SUPPORT", "CUBE", size=0.1, location=(-1.4, 0.0, 0.0), scale=(1.0, 5.0, 5.0))
    empty("Load", "LOAD", "SINGLE_ARROW", location=(-0.6, 0.0, 0.4), magnitude=10.0)
    empty("Lost", "LOAD", "SINGLE_ARROW", location=(1.0, 0.0, 0.4), magnitude=10.0)
    problem, info = scene_io.gather_problem(bpy.context)
    assert any("touch no support" in w for w in info["warnings"])
    assert any("parts without supports" in w for w in info["warnings"])
    xyz = problem.grid.element_centers()[problem.domain]
    assert xyz[:, 0].max() < 0.0
    assert problem.forces.reshape(-1, 3).sum(axis=0)[2] == pytest.approx(10.0)


# --------------------------------------------------------------------- signature
def test_setup_signature_tracks_changes(scene):
    settings = bridge(resolution=16)
    first = scene_io.setup_signature(bpy.context)
    assert scene_io.setup_signature(bpy.context) == first
    load = next(o for o in scene.objects if o.stress_sculpt.role == "LOAD")
    original = load.location.copy()
    load.location.x += 0.1
    bpy.context.view_layer.update()
    moved = scene_io.setup_signature(bpy.context)
    assert moved != first
    load.location = original
    bpy.context.view_layer.update()
    assert scene_io.setup_signature(bpy.context) == first
    settings.resolution = 18
    assert scene_io.setup_signature(bpy.context) != first
    settings.resolution = 16
    load.stress_sculpt.enabled = False
    assert scene_io.setup_signature(bpy.context) != first
    load.stress_sculpt.enabled = True
    load.stress_sculpt.magnitude = 1.0
    assert scene_io.setup_signature(bpy.context) != first
    load.stress_sculpt.magnitude = 5000.0
    settings.domain.scale.z = 1.1
    bpy.context.view_layer.update()
    assert scene_io.setup_signature(bpy.context) != first


def test_role_objects_skip_domain_disabled_and_excluded(scene):
    bridge(resolution=16)
    settings = scene.stress_sculpt
    names = [o.name for o in scene_io.role_objects(bpy.context)]
    assert len(names) == 3 and settings.domain.name not in names
    obj = scene.objects[names[0]]
    obj.stress_sculpt.enabled = False
    assert obj.name not in [o.name for o in scene_io.role_objects(bpy.context)]
    assert obj.name in [o.name for o in scene_io.role_objects(bpy.context, include_disabled=True)]


def test_gather_mesh_support_uses_its_interior(scene):
    require("fem")
    settings = scene.stress_sculpt
    settings.domain = box("Domain", size=(2.0, 1.0, 1.0))
    settings.resolution = 20
    anchor = box("Anchor", size=(0.5, 2.0, 2.0), location=(-1.0, 0.0, 0.0))  # covers x <= -0.75
    anchor.stress_sculpt.role = "SUPPORT"
    anchor.stress_sculpt.fix_y = False
    empty("Load", "LOAD", "SINGLE_ARROW", location=(1.0, 0.0, 0.5), magnitude=50.0)
    problem, info = scene_io.gather_problem(bpy.context)
    # Y is free everywhere: fem.check_problem reports the mechanism
    assert len(info["warnings"]) == 1 and "rigid" in info["warnings"][0]
    grid = problem.grid
    nodes = np.unique(problem.fixed_dofs // 3)
    xyz = grid.origin + np.stack(np.unravel_index(nodes, grid.nodes_shape), axis=1) * grid.h
    assert xyz[:, 0].max() <= -0.75 + 1e-9 and np.isclose(xyz[:, 0].min(), -1.0)
    # all nodes of the covered layers, only x and z fixed
    expected_layers = int(round(0.25 / grid.h)) + 1
    assert nodes.size == expected_layers * (grid.nodes_shape[1] * grid.nodes_shape[2])
    components = np.unique(problem.fixed_dofs % 3)
    assert components.tolist() == [0, 2]


def test_voxelize_rays_through_vertices_match_analytic(scene):
    # Octahedron |x| + |y| + |z| <= 1: the central column passes exactly through the
    # top and bottom vertices, the columns at (+-1, 0) / (0, +-1) graze side vertices.
    verts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    tris = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    )
    for shape, origin, h in (
        ((5, 5, 8), (-1.25, -1.25, -1.0), 0.5),
        ((9, 9, 9), (-1.125,) * 3, 0.25),
    ):
        grid = Grid(shape=shape, origin=origin, h=h)
        inside, n_ambiguous = scene_io._voxelize(verts, tris, grid)
        centres = grid.element_centers()
        l1 = np.abs(centres).sum(axis=-1)
        clear = np.abs(l1 - 1.0) > 1e-9
        assert np.array_equal(inside[clear], (l1 < 1.0)[clear]), shape
        assert n_ambiguous == 0
