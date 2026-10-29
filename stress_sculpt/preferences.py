"""Add-on preferences: compute device of the solver and the optional GPU support.

* **Compute device** - Automatic / CPU / GPU and the number of CPU threads used by
  the numpy solver (0 = all cores). Operators read them with
  :func:`compute_settings` (main thread) and build the solver backend with
  ``core.backend.get_backend(device, threads)`` inside their worker thread.
* **GPU support** - the GPU solver needs PyTorch with CUDA, which Blender does not
  ship. "Install GPU Support" runs ``pip install torch`` from the PyTorch package
  index into Blender's user ``scripts/modules`` folder (Blender adds it to
  ``sys.path`` at start-up) in a background process, shows its progress here, and
  "Uninstall" removes exactly the packages that the installer added.

Works as a legacy add-on and as a Blender 4.2+ extension (``bl_idname`` is the
package name, which is ``bl_ext.<repository>.stress_sculpt`` for extensions). The
installer needs Blender's "Allow Online Access" (``bpy.app.online_access``).
"""

import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import AddonPreferences, Operator

from .core import backend

ADDON_ID = __package__  # top-level package of the add-on / extension

PYTORCH_INDEX = "https://download.pytorch.org/whl/{build}"
MANIFEST_NAME = ".stress_sculpt_gpu_install.json"
MIN_FREE_DISK = 6 * (1 << 30)  # the CUDA build of PyTorch needs ~5 GB once unpacked

CUDA_BUILDS = (
    ("AUTO", "Automatic", "Choose from the NVIDIA driver and GPU (queried with nvidia-smi)"),
    ("cu118", "CUDA 11.8", "For old drivers (>= 452); GTX 900 series to RTX 40 series"),
    ("cu124", "CUDA 12.4", "Drivers >= 525; GTX 900 series to RTX 40 series"),
    ("cu126", "CUDA 12.6", "Drivers >= 525; GTX 900 series to RTX 40 series"),
    ("cu128", "CUDA 12.8", "Drivers >= 570; required by RTX 50 series (Blackwell) cards"),
)

DEVICE_ITEMS = (
    (
        "AUTO",
        "Automatic",
        (
            "Use the GPU when PyTorch with CUDA is installed and the model is large enough, "
            "otherwise the CPU"
        ),
    ),
    ("CPU", "CPU", "Always solve on the CPU (numpy, multi-threaded)"),
    ("GPU", "GPU (CUDA)", "Solve on the NVIDIA GPU (falls back to the CPU if it is unavailable)"),
)


# ------------------------------------------------------------------ preferences
def get_preferences(context=None):
    """The add-on preferences, or ``None`` when they are not registered."""
    ctx = context if context is not None else bpy.context
    try:
        addon = ctx.preferences.addons.get(ADDON_ID)
    except AttributeError:
        return None
    return getattr(addon, "preferences", None) if addon is not None else None


def compute_settings(context=None) -> tuple:
    """``(device, threads)`` from the preferences (``("AUTO", 0)`` without them).

    Read on the main thread (e.g. in an operator's ``prepare``) and pass the plain
    values to the worker, which calls ``backend.get_backend(device, threads)``.
    """
    prefs = get_preferences(context)
    if prefs is None:
        return "AUTO", 0
    return str(prefs.compute_device), int(prefs.cpu_threads)


def solver_compute(compute, context=None):
    """Wrap a job's ``compute(payload, progress, should_stop, emit)`` for the solver device.

    Call on the main thread (it reads the preferences); the returned function, run
    in the worker thread, builds the backend there (the first GPU use imports PyTorch
    and starts CUDA, a few seconds) and makes every FE model created inside
    ``compute`` use it (``backend.using_backend``)::

        job.run(preferences.solver_compute(self.compute, context), payload)
    """
    device, threads = compute_settings(context)

    def run(payload, progress, should_stop, emit):
        if device != "CPU" and backend.cached_torch_info() is None and progress is not None:
            if torch_installed():
                progress(0.0, "Starting the GPU (first use in this session)")
        be = backend.get_backend(device, threads)
        with backend.using_backend(be):
            return compute(payload, progress, should_stop, emit)

    run.__name__ = getattr(compute, "__name__", "compute")
    run.__doc__ = getattr(compute, "__doc__", None)
    return run


_FIND_SPEC = {"time": -1e9, "found": False}


def torch_installed() -> bool:
    """``import torch`` would find a package (cheap check, cached for 5 s; no import)."""
    now = time.monotonic()
    if now - _FIND_SPEC["time"] > 5.0:
        try:
            _FIND_SPEC["found"] = importlib.util.find_spec("torch") is not None
        except (ImportError, ValueError):
            _FIND_SPEC["found"] = False
        _FIND_SPEC["time"] = now
    return _FIND_SPEC["found"]


def status_line(context=None) -> str:
    """One line for the UI: which device the solver will use (never imports torch)."""
    device, threads = compute_settings(context)
    nthreads = backend.resolve_threads(threads)
    cpu = f"CPU ({nthreads} thread{'s' if nthreads != 1 else ''})"
    if device == "CPU":
        return f"Solver: {cpu}"
    info = backend.cached_torch_info()
    if info is None:
        if not torch_installed():
            return f"Solver: {cpu} - GPU support not installed"
        return f"Solver: {cpu} or GPU (checked at the first solve)"
    if info.gpu_ok:
        gpu = info.gpu_label() or "GPU"
        if device == "AUTO":
            return f"Solver: GPU {gpu} (small models on the {cpu})"
        return f"Solver: GPU {gpu}"
    return f"Solver: {cpu} - GPU unavailable: {info.error}"


def modules_dir(create: bool = False) -> str:
    """Blender's user ``scripts/modules`` folder (the install target)."""
    return bpy.utils.user_resource("SCRIPTS", path="modules", create=create)


def online_access_allowed() -> bool:
    return bool(getattr(bpy.app, "online_access", True))


def cuda_platform() -> bool:
    """CUDA builds of PyTorch exist for Windows and Linux on x86-64 / arm64 only."""
    return sys.platform.startswith(("win", "linux"))


OFFLINE_MESSAGE = (
    "Online access is disabled in Blender. Enable it in Edit > Preferences > System > "
    "Network > Allow Online Access (or start Blender with --online-mode), then try again"
)


# ------------------------------------------------------------- CUDA build choice
def query_nvidia_smi(timeout: float = 10.0) -> dict:
    """GPU name, compute capability and driver CUDA version from ``nvidia-smi``.

    Returns ``{}`` when nvidia-smi is missing or fails. Keys: ``name``,
    ``capability`` (float, e.g. 8.9), ``driver`` (str), ``cuda`` (float, e.g. 12.4).
    """
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        for cand in (
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.isfile(cand):
                exe = cand
                break
    if exe is None:
        return {}
    flags = _no_window_flags()
    out = {}
    try:
        res = subprocess.run(
            [exe, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout, creationflags=flags,
        )
        if res.returncode != 0:  # old drivers do not know compute_cap
            res = subprocess.run(
                [exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=timeout, creationflags=flags,
            )
        first = (res.stdout or "").strip().splitlines()
        if res.returncode == 0 and first:
            parts = [p.strip() for p in first[0].split(",")]
            out["name"] = parts[0]
            if len(parts) > 1:
                out["driver"] = parts[1]
            if len(parts) > 2:
                try:
                    out["capability"] = float(parts[2])
                except ValueError:
                    pass
        res = subprocess.run(
            [exe], capture_output=True, text=True, timeout=timeout, creationflags=flags
        )
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", res.stdout or "")
        if m:
            out["cuda"] = float(m.group(1))
    except (OSError, subprocess.SubprocessError, ValueError):
        return out
    return out


def choose_cuda_build(smi: dict) -> tuple:
    """``(build, why)`` for the PyTorch index from :func:`query_nvidia_smi` output."""
    if not smi:
        return "cu126", "nvidia-smi not found: using CUDA 12.6 (any driver >= 525)"
    cap = smi.get("capability")
    cuda = smi.get("cuda", 0.0)
    name = smi.get("name", "NVIDIA GPU")
    if cap is not None and cap >= 10.0:
        why = f"{name} (compute {cap:g}) needs CUDA 12.8"
        if cuda and cuda < 12.8:
            why += f" - update the NVIDIA driver (it supports CUDA {cuda:g})"
        return "cu128", why
    if cap is not None and cap < 7.0:  # Maxwell / Pascal: dropped by the newest builds
        if cuda >= 12.6:
            return "cu126", f"{name} (compute {cap:g}): CUDA 12.6"
        return "cu118", f"{name} (compute {cap:g}), driver CUDA {cuda:g}: CUDA 11.8"
    if cuda >= 12.8:
        return "cu128", f"{name}, driver supports CUDA {cuda:g}: CUDA 12.8"
    if cuda >= 12.6:
        return "cu126", f"{name}, driver supports CUDA {cuda:g}: CUDA 12.6"
    if cuda >= 12.4:
        return "cu124", f"{name}, driver supports CUDA {cuda:g}: CUDA 12.4"
    if cuda >= 11.8:
        return "cu118", f"{name}, driver supports CUDA {cuda:g}: CUDA 11.8"
    return "cu118", f"{name}: the driver is old (CUDA {cuda:g}); update it for best results"


def _no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


# ---------------------------------------------------------------- background job
class _Job:
    """State of the running / last install, uninstall or check (shared with the UI)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.kind = ""  # "INSTALL" | "UNINSTALL" | "CHECK"
        self.running = False
        self.status = ""
        self.lines = deque(maxlen=300)
        self.success = None  # None while running / never run
        self.error = ""
        self.restart_needed = False
        self.started = 0.0
        self.finished = 0.0
        self.proc = None
        self.cancel = threading.Event()
        self.thread = None

    def begin(self, kind: str) -> bool:
        with self.lock:
            if self.running:
                return False
            self.kind = kind
            self.running = True
            self.status = "Starting..."
            self.lines.clear()
            self.success = None
            self.error = ""
            self.started = time.time()
            self.finished = 0.0
            self.proc = None
            self.cancel.clear()
            return True

    def log(self, line: str) -> None:
        line = line.rstrip()
        if line:
            with self.lock:
                self.lines.append(line)

    def set_status(self, text: str) -> None:
        with self.lock:
            self.status = text

    def end(self, success: bool, status: str, error: str = "") -> None:
        with self.lock:
            self.running = False
            self.success = bool(success)
            self.status = status
            self.error = error
            self.finished = time.time()
            self.proc = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "kind": self.kind,
                "running": self.running,
                "status": self.status,
                "lines": list(self.lines),
                "success": self.success,
                "error": self.error,
                "restart_needed": self.restart_needed,
                "elapsed": (self.finished or time.time()) - self.started if self.started else 0.0,
            }


JOB = _Job()


def _clean_env() -> dict:
    env = dict(os.environ)
    for key in ("PIP_USER", "PIP_TARGET", "PIP_PREFIX", "PIP_REQUIRE_VIRTUALENV", "PYTHONHOME"):
        env.pop(key, None)
    env["PIP_REQUIRE_VIRTUALENV"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_NO_INPUT"] = "1"
    return env


_STATUS_PREFIXES = ("Collecting ", "Downloading ", "Installing collected", "Successfully")


def _run_logged(job: _Job, cmd: list, env: dict) -> int:
    """Run ``cmd`` streaming its output into ``job``; returns the exit code."""
    job.log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_no_window_flags(),
    )
    with job.lock:
        job.proc = proc
    try:
        for line in proc.stdout:
            job.log(line)
            text = line.strip()
            if text.startswith(_STATUS_PREFIXES):
                job.set_status(text[:160])
            if job.cancel.is_set():
                proc.terminate()
                break
        return proc.wait()
    finally:
        with job.lock:
            job.proc = None


def _pip_version(python: str, env: dict) -> tuple:
    try:
        res = subprocess.run(
            [python, "-m", "pip", "--version"],
            capture_output=True, text=True, env=env, timeout=120, creationflags=_no_window_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    m = re.match(r"pip\s+(\d+)\.(\d+)", res.stdout or "") if res.returncode == 0 else None
    return (int(m.group(1)), int(m.group(2))) if m else ()


def ensure_pip(job: _Job, python: str, env: dict) -> tuple:
    """pip's version, running ``ensurepip`` first when pip is missing (``()`` = no pip)."""
    ver = _pip_version(python, env)
    if ver:
        return ver
    job.set_status("pip is missing: running ensurepip")
    for extra in ([], ["--user"]):
        _run_logged(job, [python, "-m", "ensurepip", "--upgrade"] + extra, env)
        ver = _pip_version(python, env)
        if ver:
            return ver
    return ()


def _dist_dirs(target: str) -> dict:
    """``{normalized name: dist-info dir}`` of the distributions in ``target``."""
    out = {}
    try:
        entries = os.listdir(target)
    except OSError:
        return out
    for entry in entries:
        if entry.endswith(".dist-info"):
            name = entry[: -len(".dist-info")].rsplit("-", 1)[0]
            out[_normalize(name)] = os.path.join(target, entry)
    return out


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _error_hint(lines: list) -> str:
    text = "\n".join(lines[-80:])
    if "No space left" in text or "Errno 28" in text:
        return "not enough disk space"
    if "WinError 206" in text or ("No such file or directory" in text and sys.platform == "win32"):
        return "Windows path too long: enable long paths (LongPathsEnabled) and retry"
    if "No matching distribution" in text or "Could not find a version" in text:
        return "no PyTorch build for this Python / CUDA choice: pick another CUDA build"
    if "SSL" in text or "CERTIFICATE" in text:
        return "secure connection failed (proxy / firewall?)"
    if "Connection" in text or "Max retries" in text or "Temporary failure" in text:
        return "network error: check the internet connection"
    if "Permission denied" in text or "Access is denied" in text:
        return "permission denied: close other Blender instances and retry"
    return "see the log below"


def install_torch(
    job: _Job,
    target: str,
    index_url: str = "",
    python: str = "",
    extra_args=None,
    package: str = "torch",
) -> bool:
    """Install ``package`` (PyTorch) with pip into ``target``; blocking, logs into ``job``.

    ``index_url`` is the PyTorch index of a CUDA build; ``extra_args`` replace it in
    tests (e.g. ``["--no-index", "--find-links", folder]``). Writes a manifest of the
    newly added distributions (used by :func:`uninstall_torch`). Returns success.
    """
    python = python or sys.executable
    env = _clean_env()
    os.makedirs(target, exist_ok=True)
    try:
        free = shutil.disk_usage(target).free
    except OSError:
        free = None
    if free is not None and free < MIN_FREE_DISK and package == "torch" and not extra_args:
        gb = free / (1 << 30)
        job.end(False, "Not enough disk space", f"{gb:.1f} GB free, about 6 GB needed")
        return False
    job.set_status("Checking pip")
    pip_ver = ensure_pip(job, python, env)
    if not pip_ver:
        job.end(False, "pip is not available", "Blender's Python has no pip and ensurepip failed")
        return False
    before = set(_dist_dirs(target))
    report_dir = tempfile.mkdtemp(prefix="stress_sculpt_pip_")
    report = os.path.join(report_dir, "report.json")
    cmd = [
        python, "-m", "pip", "install", "--upgrade", "--target", target,
        "--disable-pip-version-check", "--no-warn-script-location", "--no-cache-dir",
        "--progress-bar", "off", "--only-binary=:all:",
    ]
    if pip_ver >= (22, 2):
        cmd += ["--report", report]
    cmd += list(extra_args) if extra_args else ["--index-url", index_url]
    cmd.append(package)
    job.set_status(f"Installing {package} (this downloads 2-3 GB, please wait)")
    try:
        code = _run_logged(job, cmd, env)
    except OSError as exc:
        job.end(False, "Could not start pip", str(exc))
        return False
    if job.cancel.is_set():
        job.end(False, "Installation cancelled", "cancelled by the user")
        return False
    if code != 0:
        lines = list(job.lines)
        job.end(False, "Installation failed", f"pip exited with code {code}: {_error_hint(lines)}")
        return False
    installed = []
    try:
        with open(report, "r", encoding="utf-8") as fh:
            for item in json.load(fh).get("install", []):
                meta = item.get("metadata", {})
                installed.append({"name": meta.get("name", ""), "version": meta.get("version", "")})
    except (OSError, ValueError):
        pass
    after = _dist_dirs(target)
    new = set(n for n in after if n not in before)
    try:  # a reinstall / upgrade keeps what an earlier install of ours added
        with open(os.path.join(target, MANIFEST_NAME), "r", encoding="utf-8") as fh:
            old = (_normalize(n) for n in json.load(fh).get("added", []))
            new |= {n for n in old if n in after}
    except (OSError, ValueError):
        pass
    new = sorted(new)
    if not installed:
        installed = [{"name": n, "version": ""} for n in new]
    manifest = {
        "installed_by": "Stress Sculpt",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "source": index_url or " ".join(extra_args or ()),
        "package": package,
        "packages": installed,
        "added": new,
    }
    try:
        with open(os.path.join(target, MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
    except OSError as exc:
        job.log(f"warning: could not write the manifest: {exc}")
    shutil.rmtree(report_dir, ignore_errors=True)
    _activate_target(target)
    return True


def _activate_target(target: str) -> None:
    """Make a freshly created modules folder importable in this session."""
    if os.path.isdir(target) and target not in sys.path:
        sys.path.append(target)
    importlib.invalidate_caches()
    _FIND_SPEC["time"] = -1e9


def _remove_distribution(target: str, dist_info: str, job: _Job) -> tuple:
    """Delete the files of one distribution listed in its RECORD (inside ``target`` only)."""
    removed, failed = 0, []
    target_abs = os.path.realpath(target)
    record = os.path.join(dist_info, "RECORD")
    paths = []
    try:
        with open(record, "r", encoding="utf-8", errors="replace") as fh:
            for row in fh:
                rel = row.split(",")[0].strip()
                if rel:
                    paths.append(rel)
    except OSError:
        paths = []
    dirs = set()
    for rel in paths:
        parts = rel.replace("\\", "/").split("/")
        if parts[0] == ".." and len(parts) >= 2 and parts[-2] in ("bin", "Scripts"):
            # pip --target moves console scripts into <target>/bin (Scripts on Windows)
            rel = os.path.join(parts[-2], parts[-1])
        path = os.path.realpath(os.path.join(target, rel))
        if os.path.commonpath([path, target_abs]) != target_abs or path == target_abs:
            continue  # never touch anything outside the target folder
        if os.path.isfile(path) or os.path.islink(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                failed.append(path)
        parent = os.path.dirname(path)
        while parent.startswith(target_abs) and parent != target_abs:
            dirs.add(parent)
            parent = os.path.dirname(parent)
    for d in sorted(dirs, key=len, reverse=True):  # deepest first, only if empty
        pycache = os.path.join(d, "__pycache__")
        if os.path.isdir(pycache):
            shutil.rmtree(pycache, ignore_errors=True)
        try:
            os.rmdir(d)
        except OSError:
            pass
    if os.path.isdir(dist_info):
        shutil.rmtree(dist_info, ignore_errors=True)
    if failed:
        job.log(f"{len(failed)} files could not be removed (in use?), e.g. {failed[0]}")
    return removed, failed


def uninstall_torch(job: _Job, target: str) -> bool:
    """Remove what :func:`install_torch` added (manifest), or torch + CUDA wheels."""
    dists = _dist_dirs(target)
    manifest_path = os.path.join(target, MANIFEST_NAME)
    names = None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            names = [_normalize(n) for n in json.load(fh).get("added", [])]
    except (OSError, ValueError):
        names = None
    if names is None:  # no manifest: torch and the CUDA runtime wheels only
        names = [
            n for n in dists
            if n in ("torch", "triton", "functorch")
            or n.startswith(("nvidia-", "cuda-"))
        ]
    names = [n for n in names if n in dists]
    if not names:
        _try_remove(manifest_path)
        job.end(True, "Nothing to uninstall", "")
        return True
    total_failed = []
    for i, name in enumerate(names):
        if job.cancel.is_set():
            break
        job.set_status(f"Removing {name} ({i + 1}/{len(names)})")
        removed, failed = _remove_distribution(target, dists[name], job)
        job.log(f"removed {name}: {removed} files")
        total_failed.extend(failed)
    if total_failed:
        job.restart_needed = True
        job.end(
            False,
            "Some files are in use",
            "Restart Blender (keep the device on CPU) and click Uninstall again",
        )
        return False
    _try_remove(manifest_path)
    importlib.invalidate_caches()
    _FIND_SPEC["time"] = -1e9
    return True


def _try_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _install_worker(target: str, build_pref: str) -> None:
    job = JOB
    was_loaded = backend.torch_loaded()
    try:
        if build_pref == "AUTO":
            job.set_status("Checking the NVIDIA driver (nvidia-smi)")
            smi = query_nvidia_smi()
            build, why = choose_cuda_build(smi)
            job.log(f"CUDA build: {build} ({why})")
        else:
            build = build_pref
            job.log(f"CUDA build: {build} (chosen in the preferences)")
        url = PYTORCH_INDEX.format(build=build)
        if not install_torch(job, target, url):
            return
        job.restart_needed = True
        if was_loaded:  # the old PyTorch stays loaded until Blender restarts
            job.end(True, "Installed. Restart Blender to load the new PyTorch")
            return
        job.set_status("Checking the GPU")
        info = backend.torch_info(refresh=True)
        if info.gpu_ok:
            gpu = info.gpu_label()
            job.end(True, f"Installed PyTorch {info.version}: GPU ready ({gpu}). Restart Blender")
        elif info.importable:
            job.end(True, f"Installed PyTorch {info.version}, but: {info.error}. Restart Blender")
        else:
            job.end(True, "Installed. Restart Blender to load PyTorch")
    except Exception as exc:  # never let the thread die silently
        job.end(False, "Installation failed", f"{type(exc).__name__}: {exc}")


def _uninstall_worker(target: str) -> None:
    job = JOB
    try:
        loaded = backend.torch_loaded()
        if uninstall_torch(job, target):
            if loaded:
                job.restart_needed = True
                job.end(True, "Uninstalled. Restart Blender (PyTorch is still loaded)")
            else:
                job.end(True, "GPU support uninstalled")
            if not loaded:
                backend.torch_info(refresh=True)
    except Exception as exc:
        job.end(False, "Uninstall failed", f"{type(exc).__name__}: {exc}")


def _check_worker() -> None:
    job = JOB
    try:
        info = backend.torch_info(refresh=True)
        job.end(True, "; ".join(backend.status_lines(info)))
    except Exception as exc:
        job.end(False, "Check failed", f"{type(exc).__name__}: {exc}")


def start_job(kind: str, target, *args) -> bool:
    if not JOB.begin(kind):
        return False
    workers = {"INSTALL": _install_worker, "UNINSTALL": _uninstall_worker, "CHECK": _check_worker}
    call_args = (target,) + args if kind != "CHECK" else ()
    JOB.thread = threading.Thread(
        target=workers[kind], args=call_args, daemon=True, name=f"stress_sculpt_{kind}"
    )
    JOB.thread.start()
    _start_redraw_timer()
    return True


# ----------------------------------------------------------------------- redraw
def _redraw_preferences() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "PREFERENCES":
                area.tag_redraw()


def _redraw_tick():
    try:
        _redraw_preferences()
    except Exception:
        pass
    return 0.5 if JOB.running else None


def _start_redraw_timer() -> None:
    timers = getattr(bpy.app, "timers", None)
    if timers is not None and not timers.is_registered(_redraw_tick):
        try:
            timers.register(_redraw_tick, first_interval=0.5)
        except Exception:
            pass


_AUTO_CHECKED = {"done": False}


def _maybe_auto_check() -> None:
    """Detect the GPU in the background the first time the preferences are shown."""
    if _AUTO_CHECKED["done"] or JOB.running or backend.cached_torch_info() is not None:
        return
    _AUTO_CHECKED["done"] = True
    if torch_installed():
        start_job("CHECK", None)


# -------------------------------------------------------------------- operators
class STRESS_SCULPT_OT_install_gpu_support(Operator):
    bl_idname = "stress_sculpt.install_gpu_support"
    bl_label = "Install GPU Support (PyTorch CUDA)"
    bl_description = (
        "Download PyTorch with CUDA (2-3 GB) from download.pytorch.org into Blender's user "
        "scripts/modules folder, so the solver can use an NVIDIA GPU"
    )
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return not JOB.running

    def execute(self, context):
        if not cuda_platform():
            self.report({"ERROR"}, "GPU support needs an NVIDIA GPU on Windows or Linux")
            return {"CANCELLED"}
        if not online_access_allowed():
            self.report({"ERROR"}, OFFLINE_MESSAGE)
            return {"CANCELLED"}
        prefs = get_preferences(context)
        build = str(prefs.cuda_build) if prefs is not None else "AUTO"
        try:
            target = modules_dir(create=True)
        except Exception as exc:
            self.report({"ERROR"}, f"Cannot create the scripts/modules folder: {exc}")
            return {"CANCELLED"}
        if not start_job("INSTALL", target, build):
            self.report({"WARNING"}, "Another GPU support job is running")
            return {"CANCELLED"}
        self.report({"INFO"}, "Installing GPU support in the background (see the preferences)")
        return {"FINISHED"}


class STRESS_SCULPT_OT_uninstall_gpu_support(Operator):
    bl_idname = "stress_sculpt.uninstall_gpu_support"
    bl_label = "Uninstall GPU Support"
    bl_description = "Remove the PyTorch packages installed by Stress Sculpt"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return not JOB.running

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        target = modules_dir(create=False)
        if not os.path.isdir(target):
            self.report({"INFO"}, "GPU support is not installed")
            return {"CANCELLED"}
        if not start_job("UNINSTALL", target):
            self.report({"WARNING"}, "Another GPU support job is running")
            return {"CANCELLED"}
        return {"FINISHED"}


class STRESS_SCULPT_OT_check_gpu(Operator):
    bl_idname = "stress_sculpt.check_gpu"
    bl_label = "Check GPU"
    bl_description = "Detect PyTorch and the NVIDIA GPU again (runs in the background)"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return not JOB.running

    def execute(self, context):
        start_job("CHECK", None)
        return {"FINISHED"}


class STRESS_SCULPT_OT_cancel_gpu_job(Operator):
    bl_idname = "stress_sculpt.cancel_gpu_job"
    bl_label = "Cancel"
    bl_description = "Stop the running installation"
    bl_options = {"REGISTER", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return JOB.running and JOB.kind == "INSTALL"

    def execute(self, context):
        JOB.cancel.set()
        with JOB.lock:
            proc = JOB.proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
        return {"FINISHED"}


# ------------------------------------------------------------------ preferences
class STRESS_SCULPT_AddonPreferences(AddonPreferences):
    bl_idname = ADDON_ID

    compute_device: EnumProperty(
        name="Compute Device",
        description="Where the finite-element solver runs",
        items=DEVICE_ITEMS,
        default="AUTO",
    )
    cpu_threads: IntProperty(
        name="CPU Threads",
        description="Threads of the CPU solver (0 = all cores)",
        default=0,
        min=0,
        max=backend.MAX_THREADS,
    )
    cuda_build: EnumProperty(
        name="CUDA Build",
        description="PyTorch build to install (Automatic picks it from the NVIDIA driver)",
        items=CUDA_BUILDS,
        default="AUTO",
    )
    show_gpu_advanced: BoolProperty(name="Advanced", default=False)
    show_gpu_log: BoolProperty(name="Log", default=False)

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(self, "compute_device")
        row = col.row()
        row.prop(self, "cpu_threads")
        col.label(text=f"{backend.default_threads()} CPU cores available")
        layout.label(text=status_line(context), icon="INFO")
        self._draw_gpu(layout)

    def _draw_gpu(self, layout):
        _maybe_auto_check()
        box = layout.box()
        box.label(text="GPU Support (optional, NVIDIA)", icon="PREFERENCES")
        job = JOB.snapshot()
        info = backend.cached_torch_info()
        status = box.column(align=True)
        if job["running"] and job["kind"] == "CHECK":
            status.label(text="Checking PyTorch and the GPU...", icon="TIME")
        elif info is None:
            if not torch_installed():
                status.label(text="PyTorch is not installed: the solver uses the CPU", icon="DOT")
            else:
                status.label(text="PyTorch found, GPU not checked yet", icon="DOT")
        else:
            lines = backend.status_lines(info)
            for line in lines[:-1]:
                status.label(text=line, icon="DOT")
            status.label(text=lines[-1], icon="CHECKMARK" if info.gpu_ok else "ERROR")
            if info.importable and info.path:
                status.label(text=f"Location: {info.path}")
        online = online_access_allowed()
        row = box.row(align=True)
        sub = row.row(align=True)
        sub.enabled = online and not job["running"] and cuda_platform()
        sub.operator(STRESS_SCULPT_OT_install_gpu_support.bl_idname, icon="IMPORT")
        row.operator(
            STRESS_SCULPT_OT_uninstall_gpu_support.bl_idname, text="Uninstall", icon="TRASH"
        )
        row.operator(STRESS_SCULPT_OT_check_gpu.bl_idname, text="", icon="FILE_REFRESH")
        if not cuda_platform():
            box.label(text="GPU support needs an NVIDIA GPU on Windows or Linux", icon="INFO")
        elif not online:
            warn = box.column(align=True)
            warn.alert = True
            warn.label(text="Online access is disabled: nothing can be downloaded", icon="ERROR")
            warn.label(text="Enable Preferences > System > Network > Allow Online Access")
        shown = job["running"] or job["success"] is not None
        if job["kind"] in ("INSTALL", "UNINSTALL") and shown:
            prog = box.column(align=True)
            if job["running"]:
                minutes = int(job["elapsed"] // 60)
                secs = int(job["elapsed"] % 60)
                prog.label(text=f"{job['status']}  ({minutes}:{secs:02d})", icon="TIME")
                prog.operator(STRESS_SCULPT_OT_cancel_gpu_job.bl_idname, icon="CANCEL")
            else:
                icon = "CHECKMARK" if job["success"] else "ERROR"
                prog.label(text=job["status"], icon=icon)
                if job["error"]:
                    prog.label(text=job["error"])
            if job["restart_needed"]:
                prog.label(text="Restart Blender to finish", icon="FILE_REFRESH")
            lines = job["lines"]
            if lines:
                prog.prop(self, "show_gpu_log", icon="TEXT")
                if self.show_gpu_log or job["running"]:
                    logbox = prog.box().column(align=True)
                    for line in lines[-8:]:
                        logbox.label(text=line[:120])
        adv = box.column()
        icon = "TRIA_DOWN" if self.show_gpu_advanced else "TRIA_RIGHT"
        adv.prop(self, "show_gpu_advanced", icon=icon)
        if self.show_gpu_advanced:
            sub = adv.column()
            sub.use_property_split = True
            sub.prop(self, "cuda_build")
            sub.label(text=f"Install folder: {modules_dir(create=False)}")
            sub.label(text="About 3 GB download and 5 GB of disk space")
            sub.label(text="After installing, restart Blender: the solver then uses the GPU")


_CLASSES = (
    STRESS_SCULPT_OT_install_gpu_support,
    STRESS_SCULPT_OT_uninstall_gpu_support,
    STRESS_SCULPT_OT_check_gpu,
    STRESS_SCULPT_OT_cancel_gpu_job,
    STRESS_SCULPT_AddonPreferences,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    JOB.cancel.set()
    with JOB.lock:
        proc = JOB.proc
    if proc is not None:
        try:
            proc.terminate()
        except OSError:
            pass
    timers = getattr(bpy.app, "timers", None)
    if timers is not None and timers.is_registered(_redraw_tick):
        timers.unregister(_redraw_tick)
    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
