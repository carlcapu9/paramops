"""Fixtures for the Blender-layer tests (``bpy`` as a Python module, background mode).

The add-on is registered once per test session -- the full package through
``stress_sculpt.register()`` when that works, otherwise only ``props`` and
``operators`` (so these tests do not depend on the UI modules) -- and every test
starts from an empty factory scene.

Scene-building helpers live in ``bhelpers.py``; tests needing a numpy core module
that is not available call ``bhelpers.require(...)`` and are skipped.
"""

import importlib

import pytest

bpy = pytest.importorskip("bpy")
bmesh = pytest.importorskip("bmesh")

REGISTRATION = {"mode": None}


def _register():
    import stress_sculpt

    if hasattr(stress_sculpt, "register"):
        try:
            stress_sculpt.register()
            return "full", stress_sculpt.unregister
        except Exception as exc:  # pragma: no cover - only when another module is broken
            print(f"full registration failed ({exc!r}); registering props + operators only")
            try:
                stress_sculpt.unregister()
            except Exception:
                pass
    props = importlib.import_module("stress_sculpt.props")
    operators = importlib.import_module("stress_sculpt.operators")
    props.register()
    operators.register()

    def unregister():
        operators.unregister()
        props.unregister()

    return "partial", unregister


@pytest.fixture(scope="session", autouse=True)
def addon():
    mode, unregister = _register()
    REGISTRATION["mode"] = mode
    yield mode
    try:
        unregister()
    except Exception as exc:  # pragma: no cover
        print(f"unregister failed: {exc!r}")


@pytest.fixture
def scene(addon):
    """A fresh empty scene (factory settings) with the add-on registered."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    from stress_sculpt import jobs

    jobs.RUNTIME.end()
    jobs.clear_cache()
    yield bpy.context.scene
    jobs.RUNTIME.end()
