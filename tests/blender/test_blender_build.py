"""build: meshes, attributes, curves, materials, collections and result objects."""

import math

import numpy as np
import pytest

import bpy
from bhelpers import box, link

from stress_sculpt import build


def _quad_grid(n=3):
    x, y = np.meshgrid(np.arange(n + 1, dtype=float), np.arange(n + 1, dtype=float), indexing="ij")
    verts = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(
        -1, 4
    )
    return verts, quads


def test_mesh_from_faces_quads_and_tris(scene):
    verts, quads = _quad_grid(3)
    mesh = build.mesh_from_faces("Quads", verts, quads)
    assert len(mesh.vertices) == 16 and len(mesh.polygons) == 9 and len(mesh.edges) == 24
    assert all(p.loop_total == 4 for p in mesh.polygons)
    assert not mesh.validate()
    assert all(p.use_smooth for p in mesh.polygons)
    co = np.empty(48, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    assert np.allclose(co.reshape(-1, 3), verts)

    tris = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = build.mesh_from_faces("Tris", verts[[0, 4, 5, 1]], tris, smooth=False)
    assert len(mesh.polygons) == 2 and not any(p.use_smooth for p in mesh.polygons)
    empty_mesh = build.mesh_from_faces("Empty", np.zeros((0, 3)), np.zeros((0, 4), dtype=int))
    assert len(empty_mesh.vertices) == 0 and len(empty_mesh.polygons) == 0


def test_mesh_from_polygons_mixed_sizes(scene):
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [2, 0, 0],
            [3, 1, 0],
            [2.5, 2, 0],
            [1.8, 1.5, 0],
        ],
        float,
    )
    loop_verts = [0, 1, 2, 3, 1, 4, 2, 4, 5, 6, 7, 2]
    loop_starts = [0, 4, 7]
    mesh = build.mesh_from_polygons("Mixed", verts, loop_verts, loop_starts)
    assert [p.loop_total for p in mesh.polygons] == [4, 3, 5]
    assert not mesh.validate()
    arrays = build.mesh_arrays(mesh)
    assert arrays["loop_starts"].tolist() == loop_starts
    assert arrays["loop_verts"].tolist() == loop_verts
    assert arrays["tris"].shape == (2 + 1 + 3, 3)
    assert np.allclose(arrays["normals"][:, 2], 1.0)


def test_mesh_builders_reject_bad_input_without_leaking(scene):
    before = len(bpy.data.meshes)
    with pytest.raises(ValueError):
        build.mesh_from_faces("Bad", np.zeros((3, 3)), np.array([[0, 1, 5]]))
    with pytest.raises(ValueError):
        build.mesh_from_polygons("Bad", np.zeros((4, 3)), [0, 1, 2, 3], [0, 2])
    assert len(bpy.data.meshes) == before


def test_point_attributes(scene):
    verts, quads = _quad_grid(2)
    mesh = build.mesh_from_faces("Attr", verts, quads)
    colors = np.linspace(0.0, 1.0, 9 * 4).reshape(9, 4)
    values = np.arange(9, dtype=float) * 1e6
    build.set_mesh_stress(mesh, colors, values)
    color = mesh.color_attributes[build.STRESS_COLOR_ATTRIBUTE]
    assert color.data_type == "FLOAT_COLOR" and color.domain == "POINT"
    out = np.empty(36, dtype=np.float32)
    color.data.foreach_get("color", out)
    assert np.allclose(out.reshape(9, 4), colors, atol=1e-6)
    assert mesh.color_attributes.active_color.name == "Stress"
    assert mesh.color_attributes.render_color_index == mesh.color_attributes.find("Stress")
    raw = mesh.attributes[build.STRESS_VALUE_ATTRIBUTE]
    assert raw.data_type == "FLOAT" and raw.domain == "POINT"
    got = np.empty(9, dtype=np.float32)
    raw.data.foreach_get("value", got)
    assert np.allclose(got, values)
    # replacing keeps a single layer of each; RGB input gets alpha 1
    build.set_point_colors(mesh, np.zeros((9, 3)))
    build.set_point_values(mesh, np.ones(9))
    assert sum(1 for a in mesh.attributes if a.name == "Stress") == 1
    assert sum(1 for a in mesh.attributes if a.name == "stress") == 1
    mesh.color_attributes["Stress"].data.foreach_get("color", out)
    assert np.allclose(out.reshape(9, 4)[:, 3], 1.0)
    with pytest.raises(ValueError):
        build.set_point_values(mesh, np.ones(4))


def test_curve_from_polylines(scene):
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], float),
        np.array([[0, 0, 1]], float),
        np.array([[0, 0, 2], [0, 1, 2]], float),
    ]
    radii = [np.array([0.2, 0.5, 1.0]), np.array([1.0]), np.array([1.0, 0.5])]
    curve = build.curve_from_polylines("Lines", lines, radii, bevel_depth=0.05)
    assert curve.dimensions == "3D" and curve.bevel_depth == pytest.approx(0.05)
    assert curve.bevel_resolution == 2 and curve.use_fill_caps
    assert len(curve.splines) == 2  # the single-point polyline is skipped
    spline = curve.splines[0]
    assert spline.type == "POLY" and len(spline.points) == 3
    assert [tuple(round(c, 6) for c in p.co) for p in spline.points] == [
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (2, 1, 0, 1),
    ]
    assert [round(p.radius, 6) for p in spline.points] == [0.2, 0.5, 1.0]


def test_materials_created_once_and_wired(scene):
    heat = build.heatmap_material()
    assert build.heatmap_material() is heat or build.heatmap_material() == heat
    nodes = heat.node_tree.nodes
    color = next(
        n for n in nodes if n.bl_idname in {"ShaderNodeVertexColor", "ShaderNodeAttribute"}
    )
    assert getattr(color, "layer_name", getattr(color, "attribute_name", "")) == "Stress"
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    assert bsdf.inputs["Base Color"].is_linked
    assert bsdf.inputs["Roughness"].default_value == pytest.approx(0.5)
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    assert output.inputs["Surface"].is_linked
    tension, compression = build.tension_material(), build.compression_material()
    assert tension.diffuse_color[0] > tension.diffuse_color[2]
    assert compression.diffuse_color[2] > compression.diffuse_color[0]
    sculpture = build.sculpture_material()
    snodes = sculpture.node_tree.nodes
    assert any(n.bl_idname in {"ShaderNodeMix", "ShaderNodeMixRGB"} for n in snodes)
    assert next(n for n in snodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"].is_linked
    names = [m.name for m in bpy.data.materials]
    assert names.count("SS_Heatmap") == 1


def test_collections_and_exclusion(scene):
    results = build.ensure_collection(scene, build.RESULTS_COLLECTION)
    assert build.ensure_collection(scene, build.RESULTS_COLLECTION) == results
    assert results.name in scene.collection.children
    parent = build.ensure_collection(scene, build.VARIANTS_COLLECTION)
    child = bpy.data.collections.new("Variant 1")
    parent.children.link(child)
    build.set_collection_excluded(scene, parent, True)
    layer = build.find_layer_collection(bpy.context.view_layer.layer_collection, child)
    assert layer is not None and layer.exclude  # children follow their parent
    assert build.find_layer_collection(bpy.context.view_layer.layer_collection, parent).exclude
    build.remove_collection(parent)
    assert build.find_child_collection(scene.collection, build.VARIANTS_COLLECTION) is None
    assert "Variant 1" not in bpy.data.collections


def test_assign_result_replaces_data_and_keeps_identity(scene):
    verts, quads = _quad_grid(2)
    mesh = build.mesh_from_faces("A", verts, quads)
    obj = build.assign_result(scene, build.OBJ_EVOLVED, mesh, build.heatmap_material())
    assert obj.name == build.OBJ_EVOLVED and obj.data == mesh
    assert obj.name in build.results_collection(scene).objects
    assert obj.material_slots[0].material.name == "SS_Heatmap"
    custom = bpy.data.materials.new("Mine")
    obj.data.materials[0] = custom
    obj.location = (1.0, 2.0, 3.0)
    old_name = mesh.name
    second = build.mesh_from_faces("B", verts * 2.0, quads)
    again = build.assign_result(scene, build.OBJ_EVOLVED, second, build.heatmap_material())
    assert again == obj and obj.data == second
    assert old_name not in bpy.data.meshes  # the old datablock had no users left
    assert obj.material_slots[0].material == custom  # user material kept
    assert tuple(obj.location) == (0.0, 0.0, 0.0)
    assert build.find_result(scene, build.OBJ_EVOLVED) == obj
    # a curve result under the same name replaces the mesh object
    curve = build.curve_from_polylines("C", [np.eye(3)], [np.ones(3)], 0.1)
    replaced = build.assign_result(scene, build.OBJ_EVOLVED, curve)
    assert replaced.type == "CURVE"
    assert build.remove_result(scene, build.OBJ_EVOLVED)
    assert build.find_result(scene, build.OBJ_EVOLVED) is None


def test_new_data_removes_datablocks_on_error(scene):
    verts, quads = _quad_grid(1)
    with pytest.raises(RuntimeError):
        with build.NewData() as new:
            mesh = new.add(build.mesh_from_faces("Temp", verts, quads))
            name = mesh.name
            raise RuntimeError("boom")
    assert name not in bpy.data.meshes


def test_world_mesh_copy_refines_and_transforms(scene):
    obj = box("Box", size=(4.0, 1.0, 1.0), location=(1.0, 2.0, 3.0))
    obj.scale = (-1.0, 1.0, 1.0)  # mirrored: normals must still point outwards
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = build.world_mesh_copy(obj, depsgraph, "Copy", target_edge=0.25, max_faces=100000)
    arrays = build.mesh_arrays(mesh)
    verts = arrays["verts"]
    assert np.allclose(verts.min(axis=0), (-1.0, 1.5, 2.5)) and np.allclose(
        verts.max(axis=0), (3.0, 2.5, 3.5)
    )
    assert len(mesh.polygons) > 6 * 16
    outward = np.einsum("ij,ij->i", verts - verts.mean(axis=0), arrays["normals"])
    assert np.all(outward > 0.0)
    assert len(mesh.materials) == 0
    budget = build.world_mesh_copy(obj, depsgraph, "Budget", target_edge=1e-3, max_faces=500)
    assert len(budget.polygons) <= 500


def test_text_object_and_bounds(scene):
    collection = build.ensure_collection(scene, build.COMPARE_COLLECTION)
    label = build.text_object("Label", "Variant 1\nmaterial 30%", 0.2, (1.0, 0.0, -0.5), collection)
    assert label.type == "FONT" and label.data.body.startswith("Variant 1")
    assert label.rotation_euler.x == pytest.approx(math.pi / 2.0)
    verts, quads = _quad_grid(2)
    mesh_obj = link(bpy.data.objects.new("M", build.mesh_from_faces("M", verts, quads)))
    mesh_obj.location = (10.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    lo, hi = build.data_bounds(mesh_obj)
    assert np.allclose(lo, (10, 0, 0)) and np.allclose(hi, (12, 2, 0))
    curve = build.curve_from_polylines("C", [np.array([[0, 0, 0], [0, 0, 3.0]])], [np.ones(2)], 0.5)
    curve_obj = link(bpy.data.objects.new("C", curve))
    lo, hi = build.union_bounds([mesh_obj, curve_obj])
    assert np.allclose(lo, (-0.5, -0.5, -0.5)) and np.allclose(hi, (12, 2, 3.5))


def test_remove_objects_purges_unused_data(scene):
    verts, quads = _quad_grid(1)
    mesh = build.mesh_from_faces("Shared", verts, quads)
    a = link(bpy.data.objects.new("A", mesh))
    b = link(bpy.data.objects.new("B", mesh))
    assert build.remove_objects([a]) == 1
    assert "Shared" in bpy.data.meshes  # still used by B
    build.remove_objects([b])
    assert "Shared" not in bpy.data.meshes
