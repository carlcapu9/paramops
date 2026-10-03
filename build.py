"""Package the add-on as an installable Blender extension zip.

    python build.py            ->  dist/paramops-<version>.zip

Equivalent to ``blender --command extension build --source-dir paramops``:
the manifest and the package files sit at the root of the archive. Install it
from Blender (4.2+) with Edit > Preferences > Get Extensions > Install from Disk.
"""

import os
import re
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "paramops")
DIST = os.path.join(ROOT, "dist")


def version():
    with open(os.path.join(SRC, "blender_manifest.toml"), encoding="utf8") as fh:
        return re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M).group(1)


def main():
    os.makedirs(DIST, exist_ok=True)
    out = os.path.join(DIST, "paramops-%s.zip" % version())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for base, dirs, files in os.walk(SRC):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                if name.endswith((".pyc", ".pyo")):
                    continue
                path = os.path.join(base, name)
                zf.write(path, os.path.relpath(path, SRC))
    print("created", out)


if __name__ == "__main__":
    main()
