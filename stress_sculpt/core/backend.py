"""Compute backends of the FEM solver: numpy on the CPU or PyTorch on an NVIDIA GPU.

* ``NUMPY`` (always available): the matrix-free kernels of ``core.fem`` run on a
  small pool of threads (numpy releases the GIL inside its loops),
* ``TORCH`` (optional): the same solver ported to PyTorch (``core.fem_torch``), on a
  CUDA GPU when the user installed PyTorch with CUDA support (add-on preferences),
  or on the CPU (used by the tests).

Precision: ``"float64"`` everywhere, or ``"mixed"`` - the multigrid preconditioner in
float32 inside a float64 conjugate gradient (the solution is still float64 and meets
the same true-residual tolerance; consumer GPUs run float64 up to 64x slower).

Nothing here imports torch at module import time. :func:`torch_info` imports it on
demand, once, and never raises: a missing or broken PyTorch just means "CPU only",
with the reason kept for the user interface. The detection result is cached for the
process (a lock guards it), everything else is immutable.

Typical use (worker thread of an operator)::

    be = backend.get_backend("AUTO", threads=0)   # never raises
    with backend.using_backend(be):               # FEModels created inside use it
        result = fem.analyze(problem)
"""

from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, field, replace
import importlib
import math
import os
import threading
import time
from typing import Optional


DEVICES = ("AUTO", "CPU", "GPU")
KINDS = ("NUMPY", "TORCH")
PRECISIONS = ("float64", "mixed")

# AUTO uses the GPU only for models with at least this many active elements: below it
# the CPU solve takes a fraction of a second and the GPU start-up (CUDA context, first
# kernels) would dominate.
GPU_MIN_ELEMENTS = 15000

# Default precision of the numpy solver (see the measurements in fem.py's docstring).
CPU_PRECISION = "mixed"
GPU_PRECISION = "mixed"

MAX_THREADS = 64


# ------------------------------------------------------------------------ detection
@dataclass(frozen=True)
class TorchInfo:
    """What PyTorch / CUDA can do in this process (result of :func:`torch_info`)."""

    importable: bool = False  # ``import torch`` works
    version: str = ""  # torch.__version__
    path: str = ""  # folder of the torch package
    cuda_build: str = ""  # CUDA version torch was built for ("" = CPU-only build)
    cuda_available: bool = False  # a CUDA device works (smoke test passed)
    device_name: str = ""
    total_memory: int = 0  # bytes of the first device
    capability: tuple = ()  # (major, minor) compute capability
    device_count: int = 0
    error: str = ""  # why the GPU cannot be used ("" when it can)
    detect_time: float = 0.0  # seconds spent detecting (import + smoke test)

    @property
    def gpu_ok(self) -> bool:
        return bool(self.importable and self.cuda_available and not self.error)

    @property
    def memory_gb(self) -> float:
        return self.total_memory / float(1 << 30)

    def gpu_label(self) -> str:
        """``"NVIDIA GeForce RTX 4090 (24.0 GB)"`` or ``""``."""
        if not self.device_name:
            return ""
        if self.total_memory:
            return f"{self.device_name} ({self.memory_gb:.1f} GB)"
        return self.device_name


_DETECT_LOCK = threading.Lock()
_DETECTED: Optional[TorchInfo] = None


def torch_info(refresh: bool = False) -> TorchInfo:
    """Detect PyTorch and a CUDA GPU (cached; ``refresh=True`` detects again).

    Importing torch takes a few seconds the first time; call this from a worker thread
    (or accept the pause). Never raises.
    """
    global _DETECTED
    with _DETECT_LOCK:
        if _DETECTED is None or refresh:
            if refresh:
                importlib.invalidate_caches()  # a fresh install must be found
            _DETECTED = _detect()
        return _DETECTED


def cached_torch_info() -> Optional[TorchInfo]:
    """The last detection result without detecting (``None`` = not detected yet)."""
    return _DETECTED


def torch_loaded() -> bool:
    """True when torch has been imported in this process (its files may be in use)."""
    import sys

    return "torch" in sys.modules and sys.modules["torch"] is not None


def _import_error_message(exc: BaseException) -> str:
    if isinstance(exc, ImportError) and getattr(exc, "name", None) == "torch":
        return "PyTorch is not installed"
    text = str(exc).strip().splitlines()
    detail = text[0] if text else ""
    return f"PyTorch could not be loaded ({type(exc).__name__}: {detail[:200]})"


def _smoke_test(torch, device) -> None:
    """Run the kernels the solver uses once on ``device`` (raises on failure)."""
    for dtype in (torch.float64, torch.float32):
        x = torch.arange(48, dtype=dtype, device=device)
        idx = torch.arange(24, dtype=torch.int32, device=device)
        a = torch.index_select(x, 0, idx).view(2, 12)
        b = torch.mm(a, a.t())
        y = torch.zeros(48, dtype=dtype, device=device)
        y.index_add_(0, idx, a.reshape(-1))
        s = float((b.sum() + y.sum()).item())
        if not math.isfinite(s):
            raise RuntimeError("the GPU returned invalid numbers")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _detect() -> TorchInfo:
    t0 = time.perf_counter()

    def done(**kw):
        return TorchInfo(detect_time=time.perf_counter() - t0, **kw)

    try:
        import torch  # noqa: F401  (optional dependency)
    except Exception as exc:  # ImportError, OSError (DLL / .so load), RuntimeError ...
        return done(importable=False, error=_import_error_message(exc))
    base = {
        "importable": True,
        "version": str(getattr(torch, "__version__", "?")),
        "path": os.path.dirname(str(getattr(torch, "__file__", "") or "")),
    }
    try:
        base["cuda_build"] = str(getattr(getattr(torch, "version", None), "cuda", None) or "")
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return done(error=f"the CUDA check failed ({type(exc).__name__}: {exc})", **base)
    if not available:
        if not base["cuda_build"]:
            reason = "this PyTorch build has no CUDA support (CPU-only build)"
        else:
            reason = "no usable NVIDIA GPU or driver was found"
        return done(error=reason, **base)
    try:
        base["device_count"] = int(torch.cuda.device_count())
        props = torch.cuda.get_device_properties(0)
        base["device_name"] = str(props.name)
        base["total_memory"] = int(props.total_memory)
        base["capability"] = (int(props.major), int(props.minor))
    except Exception as exc:
        return done(error=f"the GPU could not be queried ({type(exc).__name__}: {exc})", **base)
    try:
        _smoke_test(torch, "cuda:0")
    except Exception as exc:
        msg = str(exc).strip().splitlines()[0][:200] if str(exc).strip() else type(exc).__name__
        hint = ""
        cap = base["capability"]
        if cap and cap[0] >= 12:
            hint = " - RTX 50 series cards need a CUDA 12.8 (cu128) or newer build"
        return done(
            cuda_available=False,
            error=f"PyTorch cannot run on this GPU ({msg}){hint}",
            **base,
        )
    return done(cuda_available=True, **base)


# -------------------------------------------------------------------------- backend
def default_threads() -> int:
    """CPU threads usable by this process (affinity aware, capped)."""
    n = 0
    getaff = getattr(os, "sched_getaffinity", None)
    if getaff is not None:
        try:
            n = len(getaff(0))
        except OSError:
            n = 0
    if n <= 0:
        n = os.cpu_count() or 1
    return max(1, min(MAX_THREADS, int(n)))


@dataclass(frozen=True)
class Backend:
    """Where and how ``FEModel.solve`` runs (immutable, safe to share between threads).

    * ``kind`` ``"NUMPY"`` or ``"TORCH"``, ``device`` the torch device (``"cuda:0"``,
      ``"cpu"``; ``"cpu"`` for numpy),
    * ``threads`` CPU threads of the numpy kernels (also used by the numpy fallback),
    * ``precision`` ``"float64"`` or ``"mixed"`` (float32 multigrid preconditioner),
    * ``min_elements`` torch is used only for models with at least this many active
      elements (smaller ones use numpy; AUTO sets :data:`GPU_MIN_ELEMENTS`),
    * ``description`` for the user, ``reason`` why the requested device is not used
      (empty when it is), ``info`` the detection result (``None`` for plain CPU).
    """

    kind: str = "NUMPY"
    device: str = "cpu"
    threads: int = 1
    precision: str = "float64"
    min_elements: int = 0
    description: str = ""
    reason: str = ""
    info: Optional[TorchInfo] = field(default=None, compare=False)

    def __post_init__(self):
        kind = str(self.kind).upper()
        if kind not in KINDS:
            raise ValueError(f"unknown backend kind {self.kind!r} (expected one of {KINDS})")
        if self.precision not in PRECISIONS:
            raise ValueError(f"unknown precision {self.precision!r} (expected {PRECISIONS})")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "threads", max(1, min(MAX_THREADS, int(self.threads))))
        object.__setattr__(self, "min_elements", max(0, int(self.min_elements)))
        if not self.description:
            object.__setattr__(self, "description", self._default_description())

    @property
    def is_torch(self) -> bool:
        return self.kind == "TORCH"

    @property
    def is_gpu(self) -> bool:
        return self.kind == "TORCH" and str(self.device).startswith("cuda")

    def _default_description(self) -> str:
        if self.kind == "NUMPY":
            return f"CPU ({self.threads} thread{'s' if self.threads != 1 else ''})"
        if self.is_gpu:
            name = self.info.gpu_label() if self.info is not None else ""
            return f"GPU ({name})" if name else f"GPU ({self.device})"
        return f"PyTorch ({self.device})"

    def with_threads(self, threads: int) -> "Backend":
        return replace(self, threads=resolve_threads(threads), description="")


def resolve_threads(threads) -> int:
    """``0`` / ``None`` / negative -> all usable cores."""
    try:
        t = int(threads or 0)
    except (TypeError, ValueError):
        t = 0
    return default_threads() if t <= 0 else max(1, min(MAX_THREADS, t))


def cpu_backend(threads: int = 0, precision: Optional[str] = None, reason: str = "") -> Backend:
    """The numpy backend (``threads`` 0 = all cores)."""
    return Backend(
        kind="NUMPY",
        device="cpu",
        threads=resolve_threads(threads),
        precision=precision or CPU_PRECISION,
        reason=reason,
    )


def torch_backend(
    device: str = "cuda:0",
    precision: str = GPU_PRECISION,
    threads: int = 0,
    min_elements: int = 0,
    info: Optional[TorchInfo] = None,
) -> Backend:
    """An explicit PyTorch backend (no detection; a failing solve falls back to numpy)."""
    return Backend(
        kind="TORCH",
        device=str(device),
        threads=resolve_threads(threads),
        precision=precision,
        min_elements=min_elements,
        info=info,
    )


def get_backend(
    preference: str = "AUTO",
    threads: int = 0,
    precision: Optional[str] = None,
    refresh: bool = False,
) -> Backend:
    """Backend for a device preference ``"AUTO"`` | ``"CPU"`` | ``"GPU"``. Never raises.

    * ``CPU``: numpy with ``threads`` (0 = all cores); torch is not even imported,
    * ``GPU``: PyTorch on the first CUDA device, for every model size,
    * ``AUTO``: the GPU for models of at least :data:`GPU_MIN_ELEMENTS` elements when
      PyTorch with CUDA works, numpy otherwise.

    When the GPU is wanted but unusable the numpy backend is returned with ``reason``
    explaining why (PyTorch missing, CPU-only build, no driver, kernels failing).
    ``precision`` defaults to :data:`GPU_PRECISION` / :data:`CPU_PRECISION`.
    """
    pref = str(preference or "AUTO").upper()
    if pref not in DEVICES:
        pref = "AUTO"
    nthreads = resolve_threads(threads)
    if pref == "CPU":
        return cpu_backend(nthreads, precision)
    try:
        info = torch_info(refresh=refresh)
    except Exception as exc:  # paranoia: torch_info never raises
        info = TorchInfo(error=f"GPU detection failed ({exc})")
    if not info.gpu_ok:
        reason = info.error or "no usable GPU"
        if pref == "GPU":
            reason = f"GPU requested but unavailable: {reason}"
        return cpu_backend(nthreads, precision, reason=reason)
    return torch_backend(
        "cuda:0",
        precision or GPU_PRECISION,
        nthreads,
        min_elements=GPU_MIN_ELEMENTS if pref == "AUTO" else 0,
        info=info,
    )


# --------------------------------------------------------------- per-thread default
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("stress_sculpt_backend", default=None)


def current_backend() -> Optional[Backend]:
    """The backend set by :func:`using_backend` in this thread / context (or ``None``)."""
    return _CURRENT.get()


@contextmanager
def using_backend(backend: Optional[Backend]):
    """Make ``backend`` the default of the FE models created inside the ``with`` block.

    The setting is local to the current thread (context variable): set it inside the
    worker thread that runs the computation. ``None`` restores the plain default.
    """
    token = _CURRENT.set(backend)
    try:
        yield backend
    finally:
        _CURRENT.reset(token)


def status_lines(info: Optional[TorchInfo] = None) -> list:
    """Short human-readable lines about PyTorch / CUDA (for UIs and logs)."""
    info = info if info is not None else cached_torch_info()
    if info is None:
        return ["GPU support not checked yet"]
    lines = []
    if not info.importable:
        lines.append(info.error or "PyTorch is not installed")
        return lines
    build = f"CUDA {info.cuda_build}" if info.cuda_build else "CPU-only build"
    lines.append(f"PyTorch {info.version} ({build})")
    if info.gpu_ok:
        lines.append(f"GPU: {info.gpu_label()}")
    else:
        lines.append(f"GPU unavailable: {info.error}")
    return lines
