# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers shared by operators and panels."""

import bpy


def is_scatter(obj):
    try:
        return obj is not None and obj.type == "MESH" and obj.paramops.is_scatter
    except (ReferenceError, AttributeError):
        return False


def scatters_using(curve):
    return [ob for ob in bpy.data.objects if is_scatter(ob) and ob.paramops.path == curve]


def context_scatter(context):
    """The scatter shown in the panel: the active scatter, or the one linked to the active curve."""
    ob = context.active_object if hasattr(context, "active_object") else None
    if ob is None:
        return None
    if is_scatter(ob):
        return ob
    if ob.type == "CURVE":
        sc = ob.paramops.curve_scatter
        if is_scatter(sc) and sc.paramops.path == ob:
            return sc
        users = scatters_using(ob)
        if users:
            return users[0]
    return None


def copy_group(src, dst, skip=()):
    """Copy every property of a PropertyGroup (recursively)."""
    for prop in src.bl_rna.properties:
        pid = prop.identifier
        if pid == "rna_type" or pid in skip:
            continue
        if prop.type == "POINTER":
            val = getattr(src, pid)
            if isinstance(val, bpy.types.PropertyGroup):
                copy_group(val, getattr(dst, pid))
                continue
            if prop.is_readonly:
                continue
            setattr(dst, pid, val)
        elif prop.type == "COLLECTION":
            dcol = getattr(dst, pid)
            dcol.clear()
            for item in getattr(src, pid):
                copy_group(item, dcol.add())
        elif not prop.is_readonly:
            val = getattr(src, pid)
            if getattr(prop, "is_array", False) or getattr(prop, "array_length", 0) > 0:
                val = tuple(val)
            try:
                setattr(dst, pid, val)
            except (TypeError, AttributeError, ValueError):
                pass
