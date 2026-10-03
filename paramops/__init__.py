# SPDX-License-Identifier: GPL-3.0-or-later
"""ParamOps: parametric modular scattering along curves for Blender.

Assign meshes to the sample slots (Start, End, Default, Evenly, Corner,
plus Segment IDs and Markers), pick a curve and the scatter builds itself and
stays live while you edit the curve, the samples or the settings.
"""

bl_info = {
    "name": "ParamOps",
    "author": "ParamOps",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > ParamOps",
    "description": "Parametric modular scattering along curves (start/end, adaptive default, "
                   "evenly, corners, segment IDs, markers, slopes, round corners)",
    "category": "Object",
}

if "bpy" in locals():
    import importlib
    for _mod in (core_pathmath, core_layout, core_meshdata, core_placement,  # noqa: F821
                 samples, writer, curves, props, builder, utils, handlers, ops, demo, ui):  # noqa: F821
        importlib.reload(_mod)

import bpy  # noqa: E402

from .core import layout as core_layout  # noqa: E402,F401
from .core import meshdata as core_meshdata  # noqa: E402,F401
from .core import pathmath as core_pathmath  # noqa: E402,F401
from .core import placement as core_placement  # noqa: E402,F401
from . import samples, writer, curves, props, builder, utils, handlers, ops, demo, ui  # noqa: E402,F401


def register():
    props.register()
    ops.register()
    ui.register()
    handlers.register()


def unregister():
    handlers.unregister()
    ui.unregister()
    ops.unregister()
    props.unregister()
