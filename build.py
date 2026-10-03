"""Package the add-on as an installable Blender extension zip.

    python build.py            ->  dist/curveforge-<version>.zip

Equivalent to ``blender --command extension build --source-dir curveforge``:
the manifest and the package files sit at the root of the archive. Install it
from Blender (4.2+) with Edit > Preferences > Get Extensions > Install from Disk,
or drag the zip into the Blender window.
"""

import os
import re
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "curveforge")
DIST = os.path.join(ROOT, "dist")


def version():
    with open(os.path.join(SRC, "blender_manifest.toml"), encoding="utf8") as fh:
        return re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M).group(1)


def main():
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, "curveforge-%s.zip" % version())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, dirs, files in os.walk(SRC):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                if name.endswith((".pyc", ".pyo")):
                    continue
                path = os.path.join(base, name)
                info = zipfile.ZipInfo(os.path.relpath(path, SRC).replace(os.sep, "/"), (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(path, "rb") as fh:
                    zf.writestr(info, fh.read())
    print("created", out)


if __name__ == "__main__":
    main()
