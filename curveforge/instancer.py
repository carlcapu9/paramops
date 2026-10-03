# SPDX-License-Identifier: GPL-3.0-or-later
"""Instancing: rigid segments become instances of their source objects.

The generated mesh carries one extra point per instance with three attributes
(``cf_is_instance``, ``cf_instance`` and ``cf_matrix``). A small Geometry Nodes
modifier, rebuilt automatically when the list of objects changes, turns those
points into instances of the original objects (with their modifiers and
materials) and keeps the rest of the mesh as it is.
"""

import bpy

MODIFIER = "CurveForge Instances"


def _new_node(nodes, idname, location, **props):
    node = nodes.new(idname)
    node.location = location
    for k, v in props.items():
        setattr(node, k, v)
    return node


def _socket(sockets, name, fallback=0):
    sock = sockets.get(name)
    return sock if sock is not None else sockets[fallback]


def _named(nodes, name, data_type, location):
    node = _new_node(nodes, "GeometryNodeInputNamedAttribute", location, data_type=data_type)
    node.inputs["Name"].default_value = name
    return node


def build_group(names):
    ng = bpy.data.node_groups.new(".CurveForge Instancer", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = ng.nodes, ng.links
    gin = _new_node(nodes, "NodeGroupInput", (-900, 0))
    gout = _new_node(nodes, "NodeGroupOutput", (700, 0))
    sep = _new_node(nodes, "GeometryNodeSeparateGeometry", (-650, 0), domain="POINT")
    flag = _named(nodes, "cf_is_instance", "BOOLEAN", (-900, -150))
    links.new(gin.outputs[0], sep.inputs["Geometry"])
    links.new(_socket(flag.outputs, "Attribute"), sep.inputs["Selection"])
    to_inst = _new_node(nodes, "GeometryNodeGeometryToInstance", (-400, -300))
    for i, name in enumerate(names):
        info = _new_node(nodes, "GeometryNodeObjectInfo", (-650, -300 - 180 * i), transform_space="ORIGINAL")
        info.inputs["Object"].default_value = bpy.data.objects.get(name)
        links.new(_socket(info.outputs, "Geometry", -1), to_inst.inputs[0])
    on_points = _new_node(nodes, "GeometryNodeInstanceOnPoints", (-150, 0))
    links.new(_socket(sep.outputs, "Selection"), on_points.inputs["Points"])
    links.new(to_inst.outputs[0], on_points.inputs["Instance"])
    on_points.inputs["Pick Instance"].default_value = True
    index = _named(nodes, "cf_instance", "INT", (-400, -150))
    links.new(_socket(index.outputs, "Attribute"), on_points.inputs["Instance Index"])
    xform = _new_node(nodes, "GeometryNodeSetInstanceTransform", (150, 0))
    links.new(on_points.outputs["Instances"], xform.inputs["Instances"])
    matrix = _named(nodes, "cf_matrix", "FLOAT4X4", (-150, -250))
    links.new(_socket(matrix.outputs, "Attribute"), xform.inputs["Transform"])
    join = _new_node(nodes, "GeometryNodeJoinGeometry", (450, 0))
    links.new(xform.outputs[0], join.inputs[0])
    links.new(_socket(sep.outputs, "Inverted", 1), join.inputs[0])
    links.new(join.outputs[0], gout.inputs[0])
    ng["cf_names"] = list(names)
    return ng


def sync(obj, names):
    """Make sure ``obj`` has the instancing modifier for ``names`` (or none)."""
    mod = obj.modifiers.get(MODIFIER)
    if not names:
        if mod is not None:
            group = mod.node_group
            obj.modifiers.remove(mod)
            if group is not None and group.users == 0:
                bpy.data.node_groups.remove(group)
        return
    if mod is None:
        mod = obj.modifiers.new(MODIFIER, "NODES")
        mod.show_in_editmode = False
    group = mod.node_group
    if group is None or list(group.get("cf_names", [])) != list(names) or group.users > 1:
        new = build_group(names)
        mod.node_group = new
        if group is not None and group.users == 0:
            bpy.data.node_groups.remove(group)
