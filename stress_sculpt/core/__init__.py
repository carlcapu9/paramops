"""Pure-numpy computational core of Stress Sculpt.

Nothing in this package may import ``bpy`` or ``mathutils``: every module here
must run (and be unit-tested) with plain CPython + numpy, and must stay safe to
call from a worker thread while Blender keeps drawing the UI.
"""
