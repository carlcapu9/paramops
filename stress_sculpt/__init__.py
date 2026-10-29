"""Stress Sculpt — turn supports and loads into sculptures that follow stress lines.

Place supports and loads on a design domain, estimate the stress distribution with a
voxel finite-element solver, then let the geometry evolve: zones thicken, hollow out
or branch according to the rules you set, tension lines become tubes and the tubes
fuse into one organic sculpture. Results can be saved as variants and compared.

The numerical core (``stress_sculpt.core``) is pure numpy and importable without
Blender; the Blender modules are only imported when ``bpy`` is available.
"""

bl_info = {
    "name": "Stress Sculpt",
    "author": "ParamOps",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Stress Sculpt",
    "description": "Turn supports and loads into organic sculptures that follow the stress lines",
    "category": "Object",
    "doc_url": "",
    "support": "COMMUNITY",
}

try:
    import bpy  # noqa: F401
except ImportError:  # numpy core used outside Blender (unit tests, scripts)
    bpy = None


# Submodules in dependency order (used for reloading with "Reload Scripts").
_SUBMODULES = (
    "core.grid",
    "core.fem",
    "core.topopt",
    "core.mesher",
    "core.sculpt",
    "core.streamlines",
    "props",
    "jobs",
    "scene_io",
    "build",
    "operators",
    "ui",
    "translations",
)


def _reload_submodules():
    import importlib
    import sys

    for name in _SUBMODULES:
        module = sys.modules.get(f"{__package__}.{name}")
        if module is not None:
            importlib.reload(module)


if bpy is not None:
    if "props" in locals():
        _reload_submodules()

    from . import props, jobs, operators, ui, translations  # noqa: E402

    _REGISTER_ORDER = (props, operators, ui, translations)


def register():
    if bpy is None:
        raise RuntimeError("Stress Sculpt can only be registered inside Blender")
    for module in _REGISTER_ORDER:
        module.register()


def unregister():
    if bpy is None:
        return
    _cancel_running_job()
    for module in reversed(_REGISTER_ORDER):
        try:
            module.unregister()
        except Exception as exc:  # keep unregistering the remaining modules
            print(f"Stress Sculpt: error while unregistering {module.__name__}: {exc}")


def _cancel_running_job():
    """Ask a running background job (if any) to stop before the add-on goes away."""
    runtime = getattr(jobs, "RUNTIME", None)
    if runtime is None:
        return
    cancel = runtime.get("cancel") if isinstance(runtime, dict) else getattr(runtime, "cancel", None)
    if cancel is not None and hasattr(cancel, "set"):
        cancel.set()
