"""Background jobs, runtime state and the in-memory result cache.

Long computations (analysis, evolution, line tracing, sculpting) run in a worker
thread so that Blender keeps drawing while numpy crunches; only one job runs at
a time.

* ``RUNTIME`` -- state of the running job, read by the UI on every redraw
  (``RUNTIME.running``, ``.name``, ``.progress`` in [0, 1], ``.message`` and the
  ``.cancel`` event). Only the main thread writes it.
* ``CACHE`` / ``get_cache`` / ``clear_cache`` -- per-scene numpy results reused
  by later steps (the lines reuse the analysis, the sculpture reuses the lines).
  The cache lives in memory only and is dropped when a file is loaded.
* ``ThreadedOperatorMixin`` -- operator base class implementing the protocol
  ``prepare`` (main thread) -> ``compute`` (worker) -> ``finish`` (main thread).

The worker never touches ``bpy``: ``compute`` receives the plain Python / numpy
payload built by ``prepare`` and returns plain data that ``finish`` turns into
Blender objects. Partial results (``emit``) travel through a queue drained by
the modal timer, which keeps only the newest one per tick.
"""

import queue
import threading
import time
import traceback

import bpy
from bpy.app.handlers import persistent

from .core.grid import Cancelled

# Seconds after which a finished job whose modal handler vanished (add-on
# reloaded, handler killed without ``cancel``) no longer blocks new jobs.
STALE_AFTER = 5.0
# Modal timer period [s].
TIMER_STEP = 0.1

_FAILED = object()  # prepare() failed and already reported


# --------------------------------------------------------------------------- job
class Job:
    """Book-keeping of one computation, shared by the worker and the main thread.

    The worker side calls ``report_progress``, ``should_stop`` and ``emit``; the
    main thread reads ``progress_state``, drains ``take_partial`` and inspects
    ``result`` / ``error`` / ``cancelled`` once ``done`` is set.
    """

    def __init__(self, name: str):
        self.name = name
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.error = None  # exception raised by compute (not Cancelled)
        self.error_traceback = ""
        self.cancelled = False
        self.finished_at = None  # time.monotonic() when done was set
        self.thread = None
        self._partials = queue.Queue()
        self._lock = threading.Lock()
        self._last_partial = None
        self._progress = 0.0
        self._message = ""

    # ------------------------------------------------------------ worker side
    def report_progress(self, fraction, message=None) -> None:
        """``progress(fraction, message)`` callback handed to the core code."""
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            value = 0.0
        if value != value:  # NaN
            value = 0.0
        value = min(max(value, 0.0), 1.0)
        with self._lock:
            self._progress = value
            if message:
                self._message = str(message)

    def should_stop(self) -> bool:
        """``should_stop()`` callback handed to the core code."""
        return self.cancel.is_set()

    def emit(self, partial) -> None:
        """Queue a partial result (e.g. a density snapshot) for the main thread."""
        with self._lock:
            self._last_partial = partial
        self._partials.put(partial)

    def run(self, compute, payload) -> None:
        """Run ``compute`` and record its outcome (never raises)."""
        try:
            self.result = compute(payload, self.report_progress, self.should_stop, self.emit)
        except Cancelled:
            self.cancelled = True
        except BaseException as exc:  # reported on the main thread
            self.error = exc
            self.error_traceback = traceback.format_exc()
        finally:
            self.finished_at = time.monotonic()
            self.done.set()

    # -------------------------------------------------------- main-thread side
    def take_partial(self):
        """Newest queued partial result (older ones are dropped), or None."""
        latest = None
        while True:
            try:
                latest = self._partials.get_nowait()
            except queue.Empty:
                return latest

    @property
    def last_partial(self):
        with self._lock:
            return self._last_partial

    def progress_state(self) -> tuple:
        with self._lock:
            return self._progress, self._message


# ----------------------------------------------------------------------- runtime
class Runtime:
    """State of the single running job (read by the UI).

    Attribute access (``RUNTIME.running``) is the primary interface; dict-style
    reads (``RUNTIME["progress"]``, ``RUNTIME.get("message")``) work too.
    """

    __slots__ = ("running", "name", "progress", "message", "cancel", "job", "started")

    def __init__(self):
        self.running = False
        self.name = ""
        self.progress = 0.0
        self.message = ""
        self.cancel = threading.Event()
        self.job = None
        self.started = 0.0

    # dict-like reading
    def __getitem__(self, key):
        if key not in self.__slots__:
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key) -> bool:
        return key in self.__slots__

    def get(self, key, default=None):
        return getattr(self, key) if key in self.__slots__ else default

    def keys(self):
        return list(self.__slots__)

    # state changes (main thread only)
    def begin(self, job: Job) -> None:
        self.running = True
        self.name = job.name
        self.progress = 0.0
        self.message = ""
        self.cancel = job.cancel  # the Cancel operator sets the job's own event
        self.job = job
        self.started = time.monotonic()

    def end(self, job=None) -> None:
        """Mark the job finished (ignored if ``job`` is not the current one)."""
        if job is not None and self.job is not job:
            return
        self.running = False
        self.progress = 0.0
        self.message = ""
        self.job = None

    def request_cancel(self) -> None:
        self.cancel.set()
        if self.running:
            self.message = "Cancelling..."

    def busy(self) -> bool:
        """True while a job runs; also clears a stale job whose handler is gone."""
        if not self.running:
            return False
        job = self.job
        if (
            job is not None
            and job.done.is_set()
            and job.finished_at is not None
            and time.monotonic() - job.finished_at > STALE_AFTER
        ):
            self.end(job)
            return False
        return True

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started if self.running else 0.0


RUNTIME = Runtime()


# ------------------------------------------------------------------------- cache
class SceneCache:
    """Numpy results of one scene.

    * ``analysis``  -- dict ``problem``, ``result`` (fem.AnalysisResult),
      ``signature``, ``grid``, ``info``
    * ``evolution`` -- dict ``problem``, ``state`` (topopt.IterationState),
      ``settings`` (topopt.OptSettings), ``signature``, ``info``, ``max_displacement``
    * ``lines``     -- dict ``lines`` (list of streamlines.Polyline), ``grid``,
      ``h``, ``source``, ``signature``, ``magnitude_ref``
    * ``structure`` -- dict of the last "Generate Structure": ``kind``, ``mode``,
      ``signature`` and, for the organic shape (reused by the ribs), ``problem``,
      ``state``, ``organic_signature``

    Each is ``None`` until computed. Dict-style reads (``cache["lines"]``) work.
    """

    __slots__ = ("analysis", "evolution", "lines", "structure")

    def __init__(self):
        self.analysis = None
        self.evolution = None
        self.lines = None
        self.structure = None

    def __getitem__(self, key):
        if key not in self.__slots__:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        value = getattr(self, key, None) if key in self.__slots__ else None
        return default if value is None else value

    def clear(self) -> None:
        self.analysis = None
        self.evolution = None
        self.lines = None
        self.structure = None


CACHE = {}  # scene name -> SceneCache


def _scene_key(scene) -> str:
    return scene if isinstance(scene, str) else scene.name


def get_cache(scene) -> SceneCache:
    """Cache of ``scene`` (a Scene or a scene name), created on demand."""
    key = _scene_key(scene)
    cache = CACHE.get(key)
    if cache is None:
        cache = CACHE[key] = SceneCache()
    return cache


def clear_cache(scene=None) -> None:
    """Forget the results of ``scene`` (all scenes when None)."""
    if scene is None:
        CACHE.clear()
    else:
        CACHE.pop(_scene_key(scene), None)


# ------------------------------------------------------------------ ui helpers
def redraw_views(context) -> None:
    """Tag 3D views (and the editors that show progress) for redraw."""
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type in {"VIEW_3D", "PROPERTIES", "OUTLINER"}:
                area.tag_redraw()


def _set_status_text(context, text) -> None:
    workspace = getattr(context, "workspace", None)
    if workspace is None:
        return
    try:
        workspace.status_text_set(text)
    except (AttributeError, RuntimeError, TypeError):
        pass


# ------------------------------------------------------------- operator mixin
class ThreadedOperatorMixin:
    """Base for operators whose heavy part runs in a worker thread.

    Subclasses implement:

    * ``prepare(self, context) -> payload`` -- main thread; gathers everything the
      computation needs as plain Python / numpy data. Raise ``ValueError`` with
      an actionable message to abort (reported as an error).
    * ``compute(payload, progress, should_stop, emit) -> result`` -- worker
      thread; pure numpy. Declare it as a ``staticmethod`` (or never touch
      ``self``): the operator is a Blender struct that must not be read from
      another thread. ``emit(obj)`` queues partial results.
    * ``on_partial(self, context, partial)`` -- main thread, optional; shows a
      partial result (only the newest partial of each timer tick is passed).
    * ``finish(self, context, result) -> set | None`` -- main thread; builds the
      Blender data. Returning None means ``{'FINISHED'}``.

    Set ``keep_partial_on_cancel = True`` to have a cancelled job still call
    ``finish(context, self.result_from_partial(last_partial))``.

    ``execute`` runs synchronously (scripts, tests, background mode, redo);
    ``invoke`` starts the worker, a 0.1 s window-manager timer and a modal
    handler (ESC / right-click cancel). Timer, status text and ``RUNTIME`` are
    cleaned up in every exit path.
    """

    job_name = "Working"
    keep_partial_on_cancel = False

    @classmethod
    def poll(cls, context):
        return not RUNTIME.busy()

    # ---------------------------------------------------------- to override
    def prepare(self, context):
        raise NotImplementedError

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        raise NotImplementedError

    def on_partial(self, context, partial):
        pass

    def finish(self, context, result):
        raise NotImplementedError

    def result_from_partial(self, partial):
        return partial

    def is_async(self) -> bool:
        """True while ``prepare`` runs for a threaded (invoked) job."""
        return bool(getattr(self, "_ss_async", False))

    def run_synchronously(self, context) -> bool:
        """Whether ``invoke`` falls back to ``execute`` (background mode, no window)."""
        return bool(bpy.app.background) or getattr(context, "window", None) is None

    # ------------------------------------------------------------- execute
    def execute(self, context):
        if RUNTIME.busy():
            self.report({"ERROR"}, "Another Stress Sculpt job is running: wait for it or cancel it")
            return {"CANCELLED"}
        self._ss_async = False
        payload = self._ss_prepare(context)
        if payload is _FAILED:
            return {"CANCELLED"}
        job = Job(self.job_name)
        RUNTIME.begin(job)
        try:
            job.run(self.compute, payload)
        finally:
            RUNTIME.end(job)
        return self._ss_conclude(context, job)

    # -------------------------------------------------------------- invoke
    def invoke(self, context, _event):
        if self.run_synchronously(context):
            return self.execute(context)
        if RUNTIME.busy():
            self.report({"ERROR"}, "Another Stress Sculpt job is running: wait for it or cancel it")
            return {"CANCELLED"}
        self._ss_async = True
        payload = self._ss_prepare(context)
        self._ss_async = False
        if payload is _FAILED:
            return {"CANCELLED"}

        job = Job(self.job_name)
        compute = self.compute  # looked up on the main thread
        job.thread = threading.Thread(
            target=job.run,
            args=(compute, payload),
            name=f"StressSculpt-{self.job_name}",
            daemon=True,
        )
        self._ss_job = job
        self._ss_timer = None
        RUNTIME.begin(job)
        wm = context.window_manager
        try:
            self._ss_timer = wm.event_timer_add(TIMER_STEP, window=context.window)
            wm.modal_handler_add(self)
            job.thread.start()
        except Exception as exc:  # pragma: no cover - only if Blender refuses the handler
            job.cancel.set()
            self._ss_cleanup(context)
            traceback.print_exc()
            self.report({"ERROR"}, f"Could not start the background job: {exc}")
            return {"CANCELLED"}
        self._ss_update_status(context, job)
        redraw_views(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        job = getattr(self, "_ss_job", None)
        if job is None:
            self._ss_cleanup(context)
            return {"CANCELLED"}

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS" and not job.done.is_set():
            job.cancel.set()
            if RUNTIME.job is job:
                RUNTIME.request_cancel()
            self._ss_update_status(context, job)
            redraw_views(context)
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        partial = job.take_partial()
        if partial is not None and not job.cancel.is_set():
            try:
                self.on_partial(context, partial)
            except Exception:  # a broken preview must not kill the job
                traceback.print_exc()
        self._ss_update_status(context, job)
        redraw_views(context)

        if job.done.is_set():
            self._ss_cleanup(context)
            return self._ss_conclude(context, job)
        return {"PASS_THROUGH"}

    def cancel(self, context):
        """Called by Blender when the modal handler is removed (file load, window closed)."""
        job = getattr(self, "_ss_job", None)
        if job is not None:
            job.cancel.set()
        self._ss_cleanup(context)

    # ------------------------------------------------------------- helpers
    def _ss_prepare(self, context):
        try:
            return self.prepare(context)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"{self.job_name}: {exc}")
        return _FAILED

    def _ss_update_status(self, context, job) -> None:
        fraction, message = job.progress_state()
        if RUNTIME.job is job:
            RUNTIME.progress = fraction
            if job.cancel.is_set():
                RUNTIME.message = "Cancelling..."
            else:
                RUNTIME.message = message
        state = "Cancelling..." if job.cancel.is_set() else message
        text = f"Stress Sculpt - {job.name}: {fraction * 100:.0f}%"
        if state:
            text += f"  {state}"
        if not job.cancel.is_set():
            text += "   (Esc to cancel)"
        _set_status_text(context, text)

    def _ss_cleanup(self, context) -> None:
        timer = getattr(self, "_ss_timer", None)
        if timer is not None:
            try:
                context.window_manager.event_timer_remove(timer)
            except Exception:
                pass
            self._ss_timer = None
        _set_status_text(context, None)
        job = getattr(self, "_ss_job", None)
        RUNTIME.end(job)
        redraw_views(context)

    def _ss_conclude(self, context, job):
        if job.error is not None:
            print(f"Stress Sculpt: {job.name} failed\n{job.error_traceback}")
            message = str(job.error) or type(job.error).__name__
            if isinstance(job.error, MemoryError):
                self.report(
                    {"ERROR"}, f"{job.name}: out of memory, lower the resolution or the detail"
                )
            elif isinstance(job.error, ValueError):
                self.report({"ERROR"}, message)
            else:
                self.report({"ERROR"}, f"{job.name} failed: {message}")
            return {"CANCELLED"}
        if job.cancelled:
            partial = job.last_partial
            if self.keep_partial_on_cancel and partial is not None:
                result = self._ss_finish(context, self.result_from_partial(partial))
                if "FINISHED" in result:
                    self.report({"WARNING"}, f"{job.name} cancelled: kept the last state")
                return result
            self.report({"WARNING"}, f"{job.name} cancelled")
            return {"CANCELLED"}
        return self._ss_finish(context, job.result)

    def _ss_finish(self, context, result):
        try:
            outcome = self.finish(context, result)
        except Exception as exc:
            traceback.print_exc()
            self.report({"ERROR"}, f"{self.job_name}: could not build the result: {exc}")
            return {"CANCELLED"}
        return outcome if outcome else {"FINISHED"}


# ---------------------------------------------------------------------- handlers
@persistent
def _on_load_pre(*_args):
    """A new file is being loaded: stop the running job and forget cached results."""
    RUNTIME.request_cancel()
    clear_cache()


def _remove_load_handlers() -> None:
    # Compare by name too: after a module reload the old function object differs.
    for handler in list(bpy.app.handlers.load_pre):
        if handler is _on_load_pre or (
            getattr(handler, "__name__", "") == _on_load_pre.__name__
            and getattr(handler, "__module__", "") == __name__
        ):
            bpy.app.handlers.load_pre.remove(handler)


def register():
    _remove_load_handlers()
    bpy.app.handlers.load_pre.append(_on_load_pre)


def unregister():
    RUNTIME.request_cancel()
    _remove_load_handlers()
    clear_cache()
