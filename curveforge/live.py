# SPDX-License-Identifier: GPL-3.0-or-later
"""Live updates: rebuild CurveForge objects when their style, curves or segments change."""

import bpy
from bpy.app.handlers import persistent

from . import meshio, output

_state = {"building": False, "suspended": 0, "loading": False}
_last_matrix = {}

SEGMENT_OBJECT_NODES = {"CF_NodeSegment"}
MESH_LIKE = {"MESH", "CURVE", "SURFACE", "FONT", "META"}


class suspended:
    """Blocks automatic rebuilds while a block of settings is being changed."""

    def __enter__(self):
        _state["suspended"] += 1
        return self

    def __exit__(self, *exc):
        _state["suspended"] -= 1
        return False


def _busy():
    return _state["building"] or _state["suspended"] or _state["loading"]


def rebuild(obj, depsgraph=None):
    """Rebuild one CurveForge object now (ignores the Live Update toggle)."""
    if not output.is_output(obj) or _state["building"] or obj.mode == "EDIT":
        return False
    _state["building"] = True
    try:
        ok = output.rebuild(obj, depsgraph)
        _last_matrix[obj.session_uid] = tuple(map(tuple, obj.matrix_world))
    finally:
        _state["building"] = False
    return ok


def request(obj):
    if _busy() or not output.is_output(obj):
        return
    if obj.cf_scatter.live:
        rebuild(obj)


_pending = set()


def tree_changed(tree, deferred=False):
    """Rebuild the objects using ``tree``.

    Changes reported by the tree update callback (links and nodes added or
    removed) are deferred to a timer: rebuilding inside that callback is unsafe
    (Blender is still updating the tree) and several changes get merged.
    """
    if _busy() or tree is None:
        return
    if deferred:
        _pending.add(tree.as_pointer())
        if not bpy.app.timers.is_registered(_flush_timer):
            bpy.app.timers.register(_flush_timer, first_interval=0.0)
        return
    for obj in output.users_of(tree):
        if obj.cf_scatter.live:
            rebuild(obj)


def flush():
    """Run the deferred rebuilds now."""
    if not _pending or _busy():
        return
    pointers = set(_pending)
    _pending.clear()
    for tree in bpy.data.node_groups:
        if tree.as_pointer() in pointers:
            tree_changed(tree)


def _flush_timer():
    flush()
    return None


def style_dependencies(tree, _seen=None):
    """Objects a style reads (curves, segment objects, clipping areas)."""
    deps = set()
    if tree is None:
        return deps
    for node in tree.nodes:
        bid = node.bl_idname
        if bid == "CF_NodeSpline" and node.curve is not None:
            deps.add(node.curve)
        elif bid == "CF_NodeSegment":
            if node.source == "OBJECT" and node.object is not None:
                deps.add(node.object)
            elif node.source == "COLLECTION" and node.collection is not None:
                deps.update(o for o in node.collection.all_objects if o.type in MESH_LIKE)
        elif bid == "CF_NodeLinear" and node.clip_curve is not None:
            deps.add(node.clip_curve)
    return deps


@persistent
def on_depsgraph_update(scene, depsgraph):
    if _busy():
        return
    geo, xform = set(), set()
    for upd in depsgraph.updates:
        idb = upd.id
        if not isinstance(idb, bpy.types.Object):
            continue
        uid = idb.original.session_uid
        if upd.is_updated_geometry:
            geo.add(uid)
        if upd.is_updated_transform:
            xform.add(uid)
    if not geo and not xform:
        return
    changed = geo | xform
    meshio.invalidate(changed)
    dirty = []
    for obj in output.outputs(scene):
        st = obj.cf_scatter
        if not st.live:
            continue
        uid = obj.session_uid
        deps = style_dependencies(st.style)
        hit = any(d.session_uid in changed for d in deps if d != obj)
        if not hit and uid in xform:
            hit = _last_matrix.get(uid) != tuple(map(tuple, obj.matrix_world))
        if hit:
            dirty.append(obj)
    for obj in dirty:
        rebuild(obj, depsgraph)


@persistent
def on_frame_change(scene, depsgraph=None):
    if _busy():
        return
    todo = [o for o in output.outputs(scene) if o.cf_scatter.update_on_frame and o.cf_scatter.live]
    if todo:
        meshio.clear_cache()
        for obj in todo:
            rebuild(obj, depsgraph)


@persistent
def on_load_pre(*_args):
    _state["loading"] = True


@persistent
def on_load_post(*_args):
    _state["loading"] = False
    meshio.clear_cache()
    _last_matrix.clear()


@persistent
def on_undo(*_args):
    meshio.clear_cache()


HANDLERS = (("depsgraph_update_post", on_depsgraph_update), ("frame_change_post", on_frame_change),
            ("load_pre", on_load_pre), ("load_post", on_load_post), ("undo_post", on_undo),
            ("redo_post", on_undo))


def register():
    for name, fn in HANDLERS:
        getattr(bpy.app.handlers, name).append(fn)


def unregister():
    for name, fn in HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        while fn in lst:
            lst.remove(fn)
    if bpy.app.timers.is_registered(_flush_timer):
        bpy.app.timers.unregister(_flush_timer)
    _pending.clear()
    meshio.clear_cache()
