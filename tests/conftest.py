import os
import sys

# The core package is pure python/numpy: import it as a top-level package so the
# tests do not need Blender (the add-on's __init__ imports bpy).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paramops"))
