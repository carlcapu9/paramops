# SPDX-License-Identifier: GPL-3.0-or-later
"""CurveForge: node-based modular scattering along curves for Blender.

Build a *style* in the CurveForge node editor: Spline and Segment nodes feed
operators (Compose, Sequence, Randomize, Conditional, Selector, Mirror,
Transform, Material, UV Transform) and a Linear Generator that lays the
segments out along the curves (Start, End, Corner, Evenly, Default, Marker).
"""

bl_info = {
    "name": "CurveForge",
    "author": "CurveForge",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > CurveForge, Node Editor > CurveForge Style",
    "description": "Node-based modular scattering along curves",
    "category": "Object",
}

if "bpy" in locals():
    import importlib
    for _m in (engine_spline, engine_rail, engine_expr, engine_graph, engine_linear, engine_mesh,  # noqa: F821
               engine_place, engine_generate, sockets, nodes, meshio, curvedata, compiler, output,  # noqa: F821
               live, ops, templates, menus, ui):  # noqa: F821
        importlib.reload(_m)

import bpy  # noqa: E402,F401

from .engine import spline as engine_spline  # noqa: E402
from .engine import rail as engine_rail  # noqa: E402
from .engine import expr as engine_expr  # noqa: E402
from .engine import graph as engine_graph  # noqa: E402
from .engine import linear as engine_linear  # noqa: E402
from .engine import mesh as engine_mesh  # noqa: E402
from .engine import place as engine_place  # noqa: E402
from .engine import generate as engine_generate  # noqa: E402
from . import sockets, nodes, meshio, curvedata, compiler, output, live, ops, templates, menus, ui  # noqa: E402

_MODULES = (sockets, nodes, curvedata, output, ops, menus, ui, live)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    for mod in reversed(_MODULES):
        mod.unregister()
