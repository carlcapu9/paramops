# SPDX-License-Identifier: GPL-3.0-or-later
"""Live update: rebuild scatters when their curve, samples or settings change."""

import bpy
from bpy.app.handlers import persistent

from . import builder, samples
from .props import scatter_dependencies
from .utils import is_scatter

_state = {"building": False, "suspended": 0}
_last_matrix = {}


class suspended:
    """Context manager that blocks automatic rebuilds (e.g. while creating a scatter)."""

    def __enter__(self):
        _state["suspended"] += 1
        return self

    def __exit__(self, *exc):
        _state["suspended"] -= 1
        return False


def rebuild(obj, depsgraph=None):
    """Rebuild one scatter now (ignores the Live Update toggle)."""
    if not is_scatter(obj) or _state["building"] or obj.mode == "EDIT":
        return False
    _state["building"] = True
    try:
        ok = builder.build(obj, depsgraph)
        _last_matrix[obj.session_uid] = tuple(map(tuple, obj.matrix_world))
    finally:
        _state["building"] = False
    return ok


def request_update(obj):
    """Called by property updates."""
    if _state["suspended"] or _state["building"] or not is_scatter(obj):
        return
    if obj.paramops.auto_update:
        rebuild(obj)


def request_update_users(obj):
    """An object's own setting changed (e.g. its random weight): rebuild users."""
    if _state["suspended"] or _state["building"]:
        return
    for sc in scatters():
        if sc.paramops.auto_update and obj in scatter_dependencies(sc):
            rebuild(sc)


def scatters(scene=None):
    objs = scene.objects if scene is not None else bpy.data.objects
    return [ob for ob in objs if is_scatter(ob)]


@persistent
def on_depsgraph_update(scene, depsgraph):
    if _state["building"] or _state["suspended"]:
        return
    geo = set()
    xform = set()
    for upd in depsgraph.updates:
        idb = upd.id
        if not isinstance(idb, bpy.types.Object):
            continue
        orig = idb.original
        uid = orig.session_uid
        if upd.is_updated_geometry:
            geo.add(uid)
        if upd.is_updated_transform:
            xform.add(uid)
    if not geo and not xform:
        return
    changed = geo | xform
    samples.invalidate(changed)
    builder.invalidate(changed)
    dirty = []
    for ob in scatters(scene):
        st = ob.paramops
        if not st.auto_update:
            continue
        uid = ob.session_uid
        hit = any(dep.session_uid in changed for dep in scatter_dependencies(ob))
        if not hit and uid in xform:
            hit = _last_matrix.get(uid) != tuple(map(tuple, ob.matrix_world))
        if hit:
            dirty.append(ob)
    for ob in dirty:
        rebuild(ob, depsgraph)


@persistent
def on_frame_change(scene, depsgraph=None):
    if _state["building"] or _state["suspended"]:
        return
    todo = [ob for ob in scatters(scene) if ob.paramops.update_on_frame and ob.paramops.auto_update]
    if not todo:
        return
    samples.clear_cache()
    builder.clear_caches()
    for ob in todo:
        rebuild(ob, depsgraph)


@persistent
def on_load(*_args):
    samples.clear_cache()
    builder.clear_caches()
    _last_matrix.clear()


@persistent
def on_undo(*_args):
    samples.clear_cache()
    builder.clear_caches()


def register():
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    bpy.app.handlers.frame_change_post.append(on_frame_change)
    bpy.app.handlers.load_post.append(on_load)
    bpy.app.handlers.undo_post.append(on_undo)
    bpy.app.handlers.redo_post.append(on_undo)


def unregister():
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, on_depsgraph_update),
                    (bpy.app.handlers.frame_change_post, on_frame_change),
                    (bpy.app.handlers.load_post, on_load),
                    (bpy.app.handlers.undo_post, on_undo),
                    (bpy.app.handlers.redo_post, on_undo)):
        while fn in lst:
            lst.remove(fn)
    samples.clear_cache()
    builder.clear_caches()
