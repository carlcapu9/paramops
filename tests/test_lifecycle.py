"""Disabling, reloading and re-enabling the add-on with CurveForge objects in the file."""

import os
import sys

import pytest

bpy = pytest.importorskip("bpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import curveforge  # noqa: E402
from curveforge import live  # noqa: E402


def test_unregister_keeps_styles_and_new_file_is_safe():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    curveforge.register()
    try:
        for key in ("WALL", "FENCE"):
            bpy.ops.curveforge.add_template(template=key)
        live.flush()
        bpy.context.evaluated_depsgraph_get()
        styles = {o.name: o.cf_scatter.style.name for o in bpy.data.objects if o.cf_scatter.enabled}
        assert len(styles) == 2
    finally:
        curveforge.unregister()
    # Reload: the objects keep their styles.
    curveforge.register()
    try:
        for name, style in styles.items():
            obj = bpy.data.objects[name]
            assert obj.cf_scatter.style is not None and obj.cf_scatter.style.name == style
            if hasattr(obj, "bl_system_properties_get"):  # 5.0: no stray custom property
                assert "cf_scatter" not in obj.keys()
            assert live.rebuild(obj) and len(obj.data.vertices) > 0
        bpy.context.evaluated_depsgraph_get()
    finally:
        curveforge.unregister()
    # Blender 4.2 crashed here when evaluated copies of the styles outlived their node types.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    assert len(bpy.data.objects) == 0


def test_enable_with_styles_in_file(tmp_path):
    """Open a file with styles while the add-on is off, then enable it."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    curveforge.register()
    try:
        bpy.ops.curveforge.add_template(template="RAILING")
        live.flush()
        path = str(tmp_path / "styles.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path)
    finally:
        curveforge.unregister()
    bpy.ops.wm.open_mainfile(filepath=path)
    bpy.context.evaluated_depsgraph_get()
    curveforge.register()
    try:
        obj = next(o for o in bpy.data.objects if o.cf_scatter.enabled)
        tree = obj.cf_scatter.style
        assert tree is not None and tree.bl_idname == "CF_StyleTree"
        assert all(n.bl_idname.startswith("CF_Node") for n in tree.nodes)
        before = len(obj.data.vertices)
        assert live.rebuild(obj) and len(obj.data.vertices) == before > 0
    finally:
        curveforge.unregister()
    bpy.ops.wm.read_factory_settings(use_empty=True)
