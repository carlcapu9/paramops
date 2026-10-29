"""numpy -> Blender: result meshes and curves, materials, collections and labels.

Main thread only. Core arrays are float64 and are converted to float32 / int32
when written with ``foreach_set``.

Result objects live in the scene's "StressSculpt Results" collection and keep
their identity across re-runs: ``assign_result`` swaps the object's data
(removing the old datablock when nothing else uses it), keeps its materials and
resets its transform to identity (result geometry is built in world space).
Meshes that carry stress get a ``FLOAT_COLOR`` / ``POINT`` color attribute
"Stress" (active and render color) and a ``FLOAT`` / ``POINT`` attribute
"stress" with the raw values.
"""

import math
import re

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

SETUP_COLLECTION = "StressSculpt Setup"
RESULTS_COLLECTION = "StressSculpt Results"
VARIANTS_COLLECTION = "StressSculpt Variants"
COMPARE_COLLECTION = "StressSculpt Compare"

STRESS_COLOR_ATTRIBUTE = "Stress"  # FLOAT_COLOR on POINT
STRESS_VALUE_ATTRIBUTE = "stress"  # FLOAT on POINT (raw values)

OBJ_STRESS = "SS_Stress"
OBJ_EVOLVED = "SS_Evolved"
OBJ_LINES_TENSION = "SS_Lines_Tension"
OBJ_LINES_COMPRESSION = "SS_Lines_Compression"
OBJ_SCULPTURE = "SS_Sculpture"
OBJ_RELIEF = "SS_Relief"
RESULT_OBJECTS = (
    OBJ_STRESS,
    OBJ_EVOLVED,
    OBJ_LINES_TENSION,
    OBJ_LINES_COMPRESSION,
    OBJ_SCULPTURE,
    OBJ_RELIEF,
)

MAT_HEATMAP = "SS_Heatmap"
MAT_TENSION = "SS_Tension"
MAT_COMPRESSION = "SS_Compression"
MAT_SCULPTURE = "SS_Sculpture"

TENSION_COLOR = (0.92, 0.25, 0.06, 1.0)
COMPRESSION_COLOR = (0.07, 0.32, 0.92, 1.0)
CLAY_COLOR = (0.74, 0.56, 0.42, 1.0)
SUPPORT_COLOR = (0.1, 0.35, 1.0, 1.0)
LOAD_COLOR = (1.0, 0.12, 0.08, 1.0)

_NAME_SUFFIX = re.compile(r"\.\d{3,}$")


def base_name(name: str) -> str:
    """``name`` without Blender's ``.001`` style de-duplication suffix."""
    return _NAME_SUFFIX.sub("", name)


# ------------------------------------------------------------------- datablocks
def id_alive(id_block) -> bool:
    """False once the datablock has been removed from ``bpy.data``."""
    if id_block is None:
        return False
    try:
        id_block.name  # noqa: B018 - raises ReferenceError when freed
    except ReferenceError:
        return False
    return True


def purge_unused(id_blocks) -> None:
    """Remove the datablocks of ``id_blocks`` that have no users left."""
    dead = {}
    for id_block in id_blocks:
        if not id_alive(id_block):
            continue
        if id_block.users == 0 and not id_block.use_fake_user:
            dead[id_block.as_pointer()] = id_block
    if dead:
        bpy.data.batch_remove(list(dead.values()))


def remove_objects(objects, purge_data=True) -> int:
    """Delete objects (and their data when it has no other users). Returns the count."""
    datas = []
    count = 0
    for obj in list(objects):
        if not id_alive(obj):
            continue
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        count += 1
        if data is not None:
            datas.append(data)
    if purge_data:
        purge_unused(datas)
    return count


class NewData:
    """Context manager collecting datablocks created while building a result.

    If the block raises, every collected datablock that still exists is removed,
    so a failed build never leaves half-built data behind.
    """

    def __init__(self):
        self.ids = []

    def add(self, id_block):
        self.ids.append(id_block)
        return id_block

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is not None:
            alive = {i.as_pointer(): i for i in self.ids if id_alive(i)}
            if alive:
                try:
                    bpy.data.batch_remove(list(alive.values()))
                except Exception:
                    pass
        return False


# ------------------------------------------------------------------ collections
def find_child_collection(parent, name):
    """Child collection of ``parent`` called ``name`` (ignoring ``.001`` suffixes)."""
    for child in parent.children:
        if base_name(child.name) == name:
            return child
    return None


def ensure_collection(scene, name, parent=None):
    """Child collection ``name`` of ``parent`` (default: scene collection), created on demand."""
    parent = scene.collection if parent is None else parent
    collection = find_child_collection(parent, name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        parent.children.link(collection)
    return collection


def find_layer_collection(layer_collection, collection):
    """The ``LayerCollection`` of ``collection`` below ``layer_collection`` (recursive)."""
    if layer_collection.collection == collection:
        return layer_collection
    for child in layer_collection.children:
        found = find_layer_collection(child, collection)
        if found is not None:
            return found
    return None


def set_collection_excluded(scene, collection, exclude=True) -> None:
    """Exclude (or include) ``collection`` in every view layer of ``scene``."""
    for view_layer in scene.view_layers:
        layer = find_layer_collection(view_layer.layer_collection, collection)
        if layer is not None and layer.exclude != exclude:
            layer.exclude = exclude


def link_to_collection(obj, collection) -> None:
    if collection.objects.get(obj.name) is None:
        collection.objects.link(obj)


def remove_collection(collection, purge_data=True) -> None:
    """Delete a collection, its child collections and all their objects."""
    if not id_alive(collection):
        return
    for child in list(collection.children):
        remove_collection(child, purge_data)
    remove_objects(list(collection.objects), purge_data)
    bpy.data.collections.remove(collection)


def results_collection(scene, create=True):
    if create:
        return ensure_collection(scene, RESULTS_COLLECTION)
    return find_child_collection(scene.collection, RESULTS_COLLECTION)


def find_result(scene, name):
    """Result object ``name`` of ``scene`` (in its results collection), or None."""
    collection = results_collection(scene, create=False)
    if collection is None:
        return None
    for obj in collection.objects:
        if base_name(obj.name) == name:
            return obj
    return None


# ------------------------------------------------------------------- visibility
def set_hidden(obj, hidden: bool, view_layer=None) -> None:
    """Viewport visibility (eye icon) of ``obj``; ignores objects not in the view layer."""
    try:
        if view_layer is not None:
            obj.hide_set(hidden, view_layer=view_layer)
        else:
            obj.hide_set(hidden)
    except (RuntimeError, TypeError):
        pass


def is_visible(obj, view_layer=None) -> bool:
    try:
        return (
            obj.visible_get(view_layer=view_layer) if view_layer is not None else obj.visible_get()
        )
    except (RuntimeError, TypeError):
        return False


# ------------------------------------------------------------------------ meshes
def set_smooth(mesh, smooth=True) -> None:
    if hasattr(mesh, "shade_smooth"):  # Blender 4.1+
        if smooth:
            mesh.shade_smooth()
        else:
            mesh.shade_flat()
    elif len(mesh.polygons):
        mesh.polygons.foreach_set("use_smooth", np.full(len(mesh.polygons), smooth, dtype=bool))


def mesh_from_polygons(name, verts, loop_verts, loop_starts, smooth=True):
    """New mesh from flat polygon arrays (any polygon sizes).

    ``verts`` (V, 3); ``loop_verts`` (L,) vertex index of every face corner;
    ``loop_starts`` (F,) first corner of every face (ascending, faces are
    contiguous runs of corners).
    """
    verts = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
    loop_verts = np.ascontiguousarray(loop_verts, dtype=np.int32).ravel()
    loop_starts = np.ascontiguousarray(loop_starts, dtype=np.int32).ravel()
    if loop_verts.size and (loop_verts.min() < 0 or loop_verts.max() >= verts.shape[0]):
        raise ValueError("Face corner refers to a vertex that does not exist")
    if loop_starts.size and (
        loop_starts[0] != 0
        or np.any(np.diff(loop_starts) < 3)
        or loop_verts.size - loop_starts[-1] < 3
    ):
        raise ValueError("Every face needs at least three corners")
    mesh = bpy.data.meshes.new(name)
    try:
        mesh.vertices.add(verts.shape[0])
        if verts.shape[0]:
            mesh.vertices.foreach_set("co", verts.ravel())
        if loop_starts.size:
            mesh.loops.add(loop_verts.size)
            mesh.loops.foreach_set("vertex_index", loop_verts)
            mesh.polygons.add(loop_starts.size)
            mesh.polygons.foreach_set("loop_start", loop_starts)
            totals = np.diff(np.append(loop_starts, loop_verts.size)).astype(np.int32)
            try:  # derived from loop_start (read-only) since Blender 4.0
                mesh.polygons.foreach_set("loop_total", totals)
            except (AttributeError, RuntimeError, TypeError):
                pass
        mesh.update(calc_edges=True)
        if bpy.app.debug:
            mesh.validate(verbose=True)
        if loop_starts.size:  # explicit both ways: since 4.1 new faces are smooth by default
            set_smooth(mesh, bool(smooth))
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    return mesh


def mesh_from_faces(name, verts, faces, smooth=True):
    """New mesh from ``verts`` (V, 3) and equal-sized ``faces`` (F, k) (tris or quads)."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        faces = faces.reshape(0, 3)
    n_faces, corners = faces.shape
    return mesh_from_polygons(
        name, verts, faces.ravel(), np.arange(n_faces, dtype=np.int64) * corners, smooth
    )


def mesh_arrays(mesh) -> dict:
    """Geometry of a mesh as numpy arrays.

    ``verts`` (V, 3) float64, ``normals`` (V, 3) float64, ``loop_verts`` (L,),
    ``loop_starts`` (F,), ``tris`` (T, 3) int64 (loop triangles).
    """
    n_verts, n_loops, n_polys = len(mesh.vertices), len(mesh.loops), len(mesh.polygons)
    co = np.empty(n_verts * 3, dtype=np.float32)
    normals = np.empty(n_verts * 3, dtype=np.float32)
    loop_verts = np.empty(n_loops, dtype=np.int32)
    loop_starts = np.empty(n_polys, dtype=np.int32)
    if n_verts:
        mesh.vertices.foreach_get("co", co)
        if hasattr(mesh, "vertex_normals"):  # Blender 4.1+
            mesh.vertex_normals.foreach_get("vector", normals)
        else:
            mesh.vertices.foreach_get("normal", normals)
    if n_loops:
        mesh.loops.foreach_get("vertex_index", loop_verts)
    if n_polys:
        mesh.polygons.foreach_get("loop_start", loop_starts)
    mesh.calc_loop_triangles()
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    if tris.size:
        mesh.loop_triangles.foreach_get("vertices", tris)
    return {
        "verts": co.reshape(-1, 3).astype(np.float64),
        "normals": normals.reshape(-1, 3).astype(np.float64),
        "loop_verts": loop_verts.astype(np.int64),
        "loop_starts": loop_starts.astype(np.int64),
        "tris": tris.reshape(-1, 3).astype(np.int64),
    }


def mesh_vertices(mesh) -> np.ndarray:
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    if co.size:
        mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def _face_estimate(mesh) -> int:
    """Faces after triangulating n-gons (each subdivision level multiplies by 4)."""
    n_polys = len(mesh.polygons)
    if n_polys == 0:
        return 0
    totals = np.empty(n_polys, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    return int(np.where(totals > 4, totals - 2, 1).sum())


def _refine_levels(mesh, target_edge, extra_levels, max_faces) -> int:
    """Uniform subdivision levels for ``world_mesh_copy``.

    Enough levels for the typical (90th percentile) edge to reach ``target_edge``,
    plus ``extra_levels``; never more faces than ``max_faces`` nor than about four
    times an ideal mesh of the final edge length (anisotropic low-poly meshes such
    as boxes would otherwise be refined far beyond need along their short edges).
    """
    n_faces = _face_estimate(mesh)
    if n_faces == 0:
        return 0
    extra = max(0, int(extra_levels))
    levels = extra
    budget = float(max_faces)
    if target_edge is not None and target_edge > 0.0 and len(mesh.edges):
        co = mesh_vertices(mesh)
        edges = np.empty(len(mesh.edges) * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
        lengths = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1)
        typical = float(np.percentile(lengths, 90.0))
        if typical > target_edge:
            levels += int(math.ceil(math.log2(typical / target_edge)))
        areas = np.empty(len(mesh.polygons), dtype=np.float32)
        mesh.polygons.foreach_get("area", areas)
        final_edge = target_edge / 2.0**extra
        ideal = float(areas.sum(dtype=np.float64)) / (final_edge * final_edge)
        budget = min(budget, max(4.0 * ideal, float(n_faces)))
    levels = min(levels, 8)
    while levels > 0 and n_faces * 4**levels > budget:
        levels -= 1
    return levels


def subdivide_mesh(mesh, levels) -> None:
    """Simple (shape-keeping) subdivision, ``levels`` times; n-gons are triangulated."""
    if levels <= 0:
        return
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        ngons = [face for face in bm.faces if len(face.verts) > 4]
        if ngons:
            bmesh.ops.triangulate(bm, faces=ngons, quad_method="BEAUTY", ngon_method="BEAUTY")
        for _ in range(levels):
            bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    mesh.update()


def world_mesh_copy(obj, depsgraph, name, target_edge=None, extra_levels=0, max_faces=400000):
    """New mesh: ``obj`` evaluated (modifiers applied) and transformed to world space.

    The copy is refined (simple subdivision, the shape is kept) until its edges
    are about ``target_edge`` long, then subdivided ``extra_levels`` more times,
    never exceeding ``max_faces``. Materials are dropped; normals point outwards
    also for mirrored (negative scale) objects. The caller owns the mesh.
    """
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(obj_eval)
    try:
        mesh.name = name
        matrix = obj_eval.matrix_world.copy()
        mesh.transform(matrix)
        if matrix.determinant() < 0.0:
            if hasattr(mesh, "flip_normals"):
                mesh.flip_normals()
            else:  # pragma: no cover - older API
                bm = bmesh.new()
                bm.from_mesh(mesh)
                bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
                bm.to_mesh(mesh)
                bm.free()
        mesh.materials.clear()
        subdivide_mesh(mesh, _refine_levels(mesh, target_edge, extra_levels, max_faces))
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    return mesh


# -------------------------------------------------------------------- attributes
def _activate_color(mesh, name) -> None:
    """Make color attribute ``name`` the active and the render color (4.2 - 5.x)."""
    attrs = mesh.color_attributes
    for prop, value in (("active_color_name", name), ("default_color_name", name)):
        try:
            setattr(attrs, prop, value)
        except (AttributeError, TypeError, ValueError):
            pass
    attribute = attrs.get(name)
    if attribute is not None:
        try:
            attrs.active_color = attribute
        except (AttributeError, TypeError, ValueError):
            pass
    index = attrs.find(name)
    if index >= 0:
        for prop in ("active_color_index", "render_color_index"):
            try:
                setattr(attrs, prop, index)
            except (AttributeError, TypeError, ValueError):
                pass


def set_point_colors(mesh, colors, name=STRESS_COLOR_ATTRIBUTE) -> None:
    """Per-vertex RGBA (or RGB) colors as a FLOAT_COLOR / POINT attribute (active + render)."""
    colors = np.asarray(colors, dtype=np.float32)
    n_verts = len(mesh.vertices)
    if colors.ndim != 2 or colors.shape[0] != n_verts or colors.shape[1] not in (3, 4):
        raise ValueError(f"Expected ({n_verts}, 4) colors, got {colors.shape}")
    if colors.shape[1] == 3:
        colors = np.concatenate([colors, np.ones((n_verts, 1), dtype=np.float32)], axis=1)
    attrs = mesh.color_attributes
    old = attrs.get(name)
    if old is not None:
        attrs.remove(old)
    attribute = attrs.new(name=name, type="FLOAT_COLOR", domain="POINT")
    if n_verts:
        attribute.data.foreach_set("color", np.ascontiguousarray(colors).ravel())
    _activate_color(mesh, name)


def set_point_values(mesh, values, name=STRESS_VALUE_ATTRIBUTE) -> None:
    """Per-vertex float attribute (FLOAT / POINT)."""
    values = np.ascontiguousarray(values, dtype=np.float32).ravel()
    if values.size != len(mesh.vertices):
        raise ValueError(f"Expected {len(mesh.vertices)} values, got {values.size}")
    attrs = mesh.attributes
    old = attrs.get(name)
    if old is not None and old.name == name:
        attrs.remove(old)
    attribute = attrs.new(name=name, type="FLOAT", domain="POINT")
    if values.size:
        attribute.data.foreach_set("value", values)


def set_mesh_stress(mesh, colors, values) -> None:
    """Both stress attributes of a result mesh."""
    set_point_colors(mesh, colors)
    set_point_values(mesh, values)


# ------------------------------------------------------------------------ curves
def curve_from_polylines(name, points_list, radii_list, bevel_depth, bevel_resolution=2):
    """3D curve with one POLY spline per polyline and per-point radius.

    ``points_list``: sequence of (n, 3) world points; ``radii_list``: matching
    (n,) point radii (multipliers of ``bevel_depth``). Polylines with fewer than
    two points are skipped.
    """
    curve = bpy.data.curves.new(name, "CURVE")
    try:
        curve.dimensions = "3D"
        curve.bevel_depth = float(bevel_depth)
        curve.bevel_resolution = int(bevel_resolution)
        curve.use_fill_caps = True
        for points, radii in zip(points_list, radii_list):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            n_points = points.shape[0]
            if n_points < 2:
                continue
            spline = curve.splines.new("POLY")
            spline.points.add(n_points - 1)
            co = np.ones((n_points, 4), dtype=np.float32)
            co[:, :3] = points
            spline.points.foreach_set("co", co.ravel())
            spline.points.foreach_set(
                "radius", np.ascontiguousarray(radii, dtype=np.float32).ravel()[:n_points]
            )
            spline.use_smooth = True
    except Exception:
        bpy.data.curves.remove(curve)
        raise
    return curve


def text_object(name, body, size, location, collection):
    """Upright (facing -Y), centred FONT object."""
    curve = bpy.data.curves.new(name, "FONT")
    try:
        curve.body = body
        curve.size = float(size)
        curve.align_x = "CENTER"
        curve.align_y = "TOP"
        obj = bpy.data.objects.new(name, curve)
    except Exception:
        bpy.data.curves.remove(curve)
        raise
    obj.location = tuple(float(c) for c in location)
    obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)
    collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------- materials
def _node_tree(material):
    if material.node_tree is None:  # Blender < 5.0 creates node trees on demand
        material.use_nodes = True
    return material.node_tree


def _find_or_new(nodes, node_type, idname):
    for node in nodes:
        if node.type == node_type:
            return node
    return nodes.new(idname)


def _input(node, *names):
    """Input socket by name or identifier (first match), or None."""
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    for name in names:
        for socket in node.inputs:
            if socket.identifier == name:
                return socket
    return None


def _output(node, *names):
    for name in names:
        for socket in node.outputs:
            if socket.identifier == name or socket.name == name:
                return socket
    return node.outputs[0] if len(node.outputs) else None


def _set_input(node, value, *names) -> None:
    socket = _input(node, *names)
    if socket is not None and hasattr(socket, "default_value"):
        try:
            socket.default_value = value
        except (TypeError, ValueError):
            pass


def _principled(material, base_color, roughness):
    """Principled BSDF wired to the material output; returns ``(nodes, links, bsdf)``."""
    tree = _node_tree(material)
    nodes, links = tree.nodes, tree.links
    bsdf = _find_or_new(nodes, "BSDF_PRINCIPLED", "ShaderNodeBsdfPrincipled")
    output = _find_or_new(nodes, "OUTPUT_MATERIAL", "ShaderNodeOutputMaterial")
    bsdf.location = (-200.0, 0.0)
    output.location = (200.0, 0.0)
    surface = _input(output, "Surface")
    if surface is not None and not any(link.from_node == bsdf for link in surface.links):
        links.new(_output(bsdf, "BSDF"), surface)
    _set_input(bsdf, base_color, "Base Color")
    _set_input(bsdf, roughness, "Roughness")
    material.diffuse_color = base_color  # solid-mode viewport color
    material.roughness = roughness
    return nodes, links, bsdf


def _color_attribute_node(nodes, layer_name):
    """Node reading the color attribute ``layer_name``; returns ``(node, color output)``."""
    try:
        node = nodes.new("ShaderNodeVertexColor")
        node.layer_name = layer_name
    except (RuntimeError, AttributeError):
        node = nodes.new("ShaderNodeAttribute")
        node.attribute_name = layer_name
    return node, _output(node, "Color")


def _mix_node(nodes, factor):
    """Color mix node (new Mix node, legacy MixRGB fallback): ``(node, fac, a, b, result)``."""
    try:
        node = nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        node.blend_type = "MIX"
        sockets = (
            _input(node, "Factor_Float"),
            _input(node, "A_Color"),
            _input(node, "B_Color"),
            _output(node, "Result_Color"),
        )
    except (RuntimeError, AttributeError, TypeError):
        node = nodes.new("ShaderNodeMixRGB")
        sockets = (
            _input(node, "Fac"),
            _input(node, "Color1"),
            _input(node, "Color2"),
            _output(node, "Color"),
        )
    if sockets[0] is not None:
        sockets[0].default_value = factor
    return (node,) + sockets


def _new_material(name):
    material = bpy.data.materials.get(name)
    if material is not None:
        return material, False
    return bpy.data.materials.new(name), True


def heatmap_material():
    """``SS_Heatmap``: color attribute "Stress" -> Principled Base Color, roughness 0.5."""
    material, created = _new_material(MAT_HEATMAP)
    if created:
        nodes, links, bsdf = _principled(material, (0.8, 0.8, 0.8, 1.0), 0.5)
        node, color = _color_attribute_node(nodes, STRESS_COLOR_ATTRIBUTE)
        node.location = (-500.0, 100.0)
        base = _input(bsdf, "Base Color")
        if base is not None and color is not None:
            links.new(color, base)
    return material


def _plain_material(name, color, roughness):
    material, created = _new_material(name)
    if created:
        _principled(material, color, roughness)
    return material


def tension_material():
    """``SS_Tension``: warm red / orange."""
    return _plain_material(MAT_TENSION, TENSION_COLOR, 0.4)


def compression_material():
    """``SS_Compression``: cool blue."""
    return _plain_material(MAT_COMPRESSION, COMPRESSION_COLOR, 0.4)


def sculpture_material():
    """``SS_Sculpture``: the "Stress" colors blended into a warm clay / stone look."""
    material, created = _new_material(MAT_SCULPTURE)
    if created:
        nodes, links, bsdf = _principled(material, CLAY_COLOR, 0.65)
        node, color = _color_attribute_node(nodes, STRESS_COLOR_ATTRIBUTE)
        node.location = (-700.0, 100.0)
        mix, _fac, clay_in, stress_in, result = _mix_node(nodes, 0.45)
        mix.location = (-450.0, 100.0)
        if clay_in is not None:
            clay_in.default_value = CLAY_COLOR
        if stress_in is not None and color is not None:
            links.new(color, stress_in)
        base = _input(bsdf, "Base Color")
        if base is not None and result is not None:
            links.new(result, base)
        _set_input(bsdf, 0.15, "Coat Weight", "Clearcoat")
    return material


def _set_materials(data, materials) -> None:
    data.materials.clear()
    for material in materials:
        data.materials.append(material)


# ----------------------------------------------------------------------- results
def _object_type_of(data) -> str:
    identifier = data.bl_rna.identifier
    return {"Mesh": "MESH", "Curve": "CURVE", "TextCurve": "FONT"}.get(
        identifier, identifier.upper()
    )


def assign_result(scene, name, data, material=None, view_layer=None):
    """Give ``data`` (a new Mesh / Curve) to the result object ``name`` of ``scene``.

    The object is created in the results collection if needed; an existing one
    keeps its identity and materials (``material`` is used when it had none),
    its old datablock is removed when unused, its transform becomes identity and
    it is made visible.
    """
    collection = results_collection(scene, create=True)
    obj = find_result(scene, name)
    if obj is not None and obj.type != _object_type_of(data):
        remove_objects([obj])
        obj = None
    if obj is None:
        _set_materials(data, [material] if material is not None else [])
        obj = bpy.data.objects.new(name, data)
        collection.objects.link(obj)
    else:
        old = obj.data
        kept = [m for m in old.materials if m is not None] if old is not None else []
        _set_materials(data, kept or ([material] if material is not None else []))
        obj.data = data
        if old is not None and old != data:
            purge_unused([old])
    obj.matrix_world = Matrix.Identity(4)
    obj.hide_viewport = False
    set_hidden(obj, False, view_layer)
    return obj


def remove_result(scene, name) -> bool:
    obj = find_result(scene, name)
    if obj is None:
        return False
    remove_objects([obj])
    return True


def hide_results(scene, names, view_layer=None) -> None:
    for name in names:
        obj = find_result(scene, name)
        if obj is not None:
            set_hidden(obj, True, view_layer)


# ---------------------------------------------------------------------- viewport
def use_attribute_colors(context) -> None:
    """Show color attributes in every 3D view that is in Solid shading."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D" and space.shading.type == "SOLID":
                    space.shading.color_type = "VERTEX"


def data_bounds(obj):
    """World bounding box ``(bmin, bmax)`` of an object's geometry, or None.

    Reads the data directly (works for objects of excluded collections, which
    the depsgraph does not evaluate); curve bevels are included approximately.
    """
    data = obj.data
    points = None
    pad = 0.0
    if obj.type == "MESH" and len(data.vertices):
        points = mesh_vertices(data)
    elif obj.type == "CURVE":
        chunks = []
        for spline in data.splines:
            if spline.type in {"POLY", "NURBS"} and len(spline.points):
                co = np.empty(len(spline.points) * 4, dtype=np.float32)
                spline.points.foreach_get("co", co)
                chunks.append(co.reshape(-1, 4)[:, :3])
            elif spline.type == "BEZIER" and len(spline.bezier_points):
                co = np.empty(len(spline.bezier_points) * 3, dtype=np.float32)
                spline.bezier_points.foreach_get("co", co)
                chunks.append(co.reshape(-1, 3))
        if chunks:
            points = np.concatenate(chunks).astype(np.float64)
            pad = float(data.bevel_depth)
    if points is None:
        return None
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    world = points @ matrix[:3, :3].T + matrix[:3, 3]
    return world.min(axis=0) - pad, world.max(axis=0) + pad


def union_bounds(objects):
    boxes = [box for box in (data_bounds(obj) for obj in objects) if box is not None]
    if not boxes:
        return None
    return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
