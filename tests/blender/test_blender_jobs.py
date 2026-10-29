"""jobs: Job runner, RUNTIME, cache and the ThreadedOperatorMixin protocol."""

import threading
import time

import pytest

import bpy
from bpy.props import StringProperty

from stress_sculpt import jobs
from stress_sculpt.core.grid import Cancelled, check_stop

LOG = []


class _TestJobBase(jobs.ThreadedOperatorMixin):
    """Shared body (registered operator classes must not subclass each other)."""

    job_name = "Test"

    def prepare(self, context):
        if self.mode == "prepare_error":
            raise ValueError("bad setup")
        return {"mode": self.mode, "async": self.is_async()}

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        progress(0.5, "half way")
        if payload["mode"] == "compute_error":
            raise RuntimeError("boom")
        if payload["mode"] == "cancel":
            emit({"partial": 1})
            raise Cancelled()
        return {"value": 42, "async": payload["async"], "running": jobs.RUNTIME.running}

    def finish(self, context, result):
        LOG.append(result)
        if self.mode == "finish_error":
            raise RuntimeError("finish failed")
        return {"FINISHED"}


class SS_TEST_OT_job(_TestJobBase, bpy.types.Operator):
    bl_idname = "ss_test.job"
    bl_label = "Test Job"

    mode: StringProperty(default="ok")


class SS_TEST_OT_job_keep(_TestJobBase, bpy.types.Operator):
    bl_idname = "ss_test.job_keep"
    bl_label = "Test Job Keep"
    keep_partial_on_cancel = True

    mode: StringProperty(default="ok")


TEST_CLASSES = (SS_TEST_OT_job, SS_TEST_OT_job_keep)


@pytest.fixture
def test_ops(scene):
    for cls in TEST_CLASSES:
        bpy.utils.register_class(cls)
    LOG.clear()
    yield
    for cls in reversed(TEST_CLASSES):
        bpy.utils.unregister_class(cls)


# ---------------------------------------------------------------------- runtime
def test_runtime_dict_and_attribute_access(scene):
    runtime = jobs.RUNTIME
    assert runtime.running is False and runtime["running"] is False
    assert runtime.get("progress") == 0.0 and runtime.get("missing", 7) == 7
    assert "cancel" in runtime and hasattr(runtime["cancel"], "set")
    job = jobs.Job("X")
    runtime.begin(job)
    assert runtime.running and runtime.name == "X" and runtime.cancel is job.cancel
    runtime.request_cancel()
    assert job.cancel.is_set() and runtime.message.startswith("Cancelling")
    runtime.end(jobs.Job("other"))  # not the current job: ignored
    assert runtime.running
    runtime.end(job)
    assert not runtime.running and runtime.job is None


def test_runtime_busy_clears_stale_jobs(scene):
    job = jobs.Job("Stale")
    jobs.RUNTIME.begin(job)
    assert jobs.RUNTIME.busy()
    job.finished_at = time.monotonic() - jobs.STALE_AFTER - 1.0
    job.done.set()
    assert not jobs.RUNTIME.busy() and not jobs.RUNTIME.running


def test_cache_per_scene_and_load_handler(scene):
    cache = jobs.get_cache(scene)
    assert cache.analysis is None and cache.get("lines") is None
    cache.analysis = {"signature": "abc"}
    assert jobs.get_cache(scene.name) is cache and cache["analysis"]["signature"] == "abc"
    jobs.clear_cache(scene)
    assert jobs.get_cache(scene).analysis is None
    jobs.get_cache(scene).lines = {"lines": []}
    bpy.ops.wm.read_factory_settings(use_empty=True)  # load_pre handler drops every cache
    assert jobs.CACHE == {}
    assert (
        sum(1 for h in bpy.app.handlers.load_pre if getattr(h, "__module__", "") == jobs.__name__)
        == 1
    )


# -------------------------------------------------------------------------- job
def test_job_outcomes():
    job = jobs.Job("ok")
    job.run(lambda payload, progress, should_stop, emit: payload * 2, 21)
    assert job.done.is_set() and job.result == 42 and job.error is None and not job.cancelled

    def cancelled(payload, progress, should_stop, emit):
        emit("partial")
        raise Cancelled()

    job = jobs.Job("cancel")
    job.run(cancelled, None)
    assert job.cancelled and job.last_partial == "partial" and job.result is None

    def broken(payload, progress, should_stop, emit):
        raise KeyError("missing")

    job = jobs.Job("error")
    job.run(broken, None)
    assert isinstance(job.error, KeyError) and "KeyError" in job.error_traceback


def test_job_progress_is_clamped():
    job = jobs.Job("p")
    job.report_progress(1.7, "over")
    assert job.progress_state() == (1.0, "over")
    job.report_progress(float("nan"))
    assert job.progress_state() == (0.0, "over")  # message kept when none is given
    job.report_progress("bad", "msg")
    assert job.progress_state() == (0.0, "msg")


def test_job_thread_partials_and_cancel():
    job = jobs.Job("thread")
    started = threading.Event()

    def compute(payload, progress, should_stop, emit):
        for i in range(10000):
            emit(i)
            progress(i / 10000.0, f"step {i}")
            started.set()
            check_stop(should_stop)
            time.sleep(0.0005)
        return "done"

    thread = threading.Thread(target=job.run, args=(compute, None), daemon=True)
    thread.start()
    assert started.wait(2.0)
    time.sleep(0.02)
    latest = job.take_partial()
    assert isinstance(latest, int) and latest >= 0
    job.cancel.set()
    thread.join(5.0)
    assert not thread.is_alive() and job.cancelled and job.result is None
    final = job.take_partial()
    assert final is None or final >= latest
    assert job.last_partial >= latest


# ------------------------------------------------------------- operator protocol
def test_mixin_execute_paths(test_ops):
    assert bpy.ops.ss_test.job(mode="ok") == {"FINISHED"}
    assert LOG[-1] == {"value": 42, "async": False, "running": True}
    assert not jobs.RUNTIME.running
    # background mode: invoke falls back to the synchronous path
    assert bpy.ops.ss_test.job("INVOKE_DEFAULT", mode="ok") == {"FINISHED"}
    assert LOG[-1]["async"] is False

    with pytest.raises(RuntimeError, match="bad setup"):
        bpy.ops.ss_test.job(mode="prepare_error")
    with pytest.raises(RuntimeError, match="boom"):
        bpy.ops.ss_test.job(mode="compute_error")
    assert not jobs.RUNTIME.running
    with pytest.raises(RuntimeError, match="finish failed"):
        bpy.ops.ss_test.job(mode="finish_error")
    assert not jobs.RUNTIME.running

    count = len(LOG)
    assert bpy.ops.ss_test.job(mode="cancel") == {"CANCELLED"}
    assert len(LOG) == count  # finish not called without keep_partial_on_cancel
    assert bpy.ops.ss_test.job_keep(mode="cancel") == {"FINISHED"}
    assert LOG[-1] == {"partial": 1}


def test_poll_blocks_a_second_job(test_ops):
    job = jobs.Job("Other")
    jobs.RUNTIME.begin(job)
    try:
        assert not SS_TEST_OT_job.poll(bpy.context)
        with pytest.raises(RuntimeError):
            bpy.ops.ss_test.job(mode="ok")
        assert bpy.ops.stress_sculpt.cancel() == {"FINISHED"}
        assert job.cancel.is_set()
    finally:
        jobs.RUNTIME.end(job)
    assert SS_TEST_OT_job.poll(bpy.context)


class _Event:
    def __init__(self, type_, value="NOTHING"):
        self.type = type_
        self.value = value


class _FakeOperator(jobs.ThreadedOperatorMixin):
    """The mixin's modal loop driven by hand (background mode has no event loop)."""

    job_name = "Fake"

    def __init__(self, keep=False, fail=False, steps=60):
        self.keep_partial_on_cancel = keep
        self.fail = fail
        self.steps = steps
        self.reports = []
        self.partials = []
        self.finished = None

    def report(self, level, message):
        self.reports.append((next(iter(level)), message))

    def prepare(self, context):
        return {"steps": self.steps, "fail": self.fail}

    @staticmethod
    def compute(payload, progress, should_stop, emit):
        for i in range(payload["steps"]):
            check_stop(should_stop)
            emit(i)
            progress((i + 1) / payload["steps"], f"step {i}")
            time.sleep(0.004)
        if payload["fail"]:
            raise RuntimeError("worker exploded")
        return "result"

    def on_partial(self, context, partial):
        self.partials.append(partial)

    def finish(self, context, result):
        self.finished = result
        return {"FINISHED"}

    def start(self, context):
        job = jobs.Job(self.job_name)
        self._ss_job = job
        self._ss_timer = None
        jobs.RUNTIME.begin(job)
        job.thread = threading.Thread(
            target=job.run, args=(self.compute, self.prepare(context)), daemon=True
        )
        job.thread.start()
        return job


def _drive(op, context, cancel_after=None, timeout=10.0):
    deadline = time.monotonic() + timeout
    ticks = 0
    while time.monotonic() < deadline:
        if cancel_after is not None and ticks == cancel_after:
            assert op.modal(context, _Event("ESC", "PRESS")) == {"RUNNING_MODAL"}
        result = op.modal(context, _Event("TIMER"))
        ticks += 1
        if result != {"PASS_THROUGH"}:
            return result
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_modal_loop_finishes_with_latest_partials(scene):
    op = _FakeOperator()
    op.start(bpy.context)
    assert op.modal(bpy.context, _Event("MOUSEMOVE")) == {"PASS_THROUGH"}
    assert _drive(op, bpy.context) == {"FINISHED"}
    assert op.finished == "result"
    assert op.partials and op.partials == sorted(op.partials)
    assert len(op.partials) < op.steps  # only the newest partial per tick
    assert not jobs.RUNTIME.running


def test_modal_cancel_keeps_last_state(scene):
    op = _FakeOperator(keep=True, steps=400)
    op.start(bpy.context)
    assert _drive(op, bpy.context, cancel_after=3) == {"FINISHED"}
    assert isinstance(op.finished, int) and op.finished < 399
    assert any(level == "WARNING" and "cancelled" in message for level, message in op.reports)
    assert not jobs.RUNTIME.running


def test_modal_cancel_without_partial_support(scene):
    op = _FakeOperator(keep=False, steps=400)
    op.start(bpy.context)
    assert _drive(op, bpy.context, cancel_after=2) == {"CANCELLED"}
    assert op.finished is None and not jobs.RUNTIME.running


def test_modal_worker_error_is_reported(scene):
    op = _FakeOperator(fail=True, steps=5)
    op.start(bpy.context)
    assert _drive(op, bpy.context) == {"CANCELLED"}
    assert any(level == "ERROR" and "worker exploded" in message for level, message in op.reports)
    assert not jobs.RUNTIME.running


def test_cancel_callback_cleans_up(scene):
    op = _FakeOperator(steps=400)
    job = op.start(bpy.context)
    op.cancel(bpy.context)  # Blender removed the handler (file load, window closed)
    assert job.cancel.is_set() and not jobs.RUNTIME.running
    job.thread.join(5.0)
    assert job.cancelled


def test_invoke_in_background_runs_synchronously(test_ops):
    # Blender calls execute() itself in background mode (there is no event); the
    # mixin's own fallback covers invoke() calls without a window.
    assert bpy.ops.ss_test.job("INVOKE_DEFAULT", mode="ok") == {"FINISHED"}
    assert LOG[-1]["async"] is False and not jobs.RUNTIME.running
    assert SS_TEST_OT_job.run_synchronously(None, bpy.context) is True


def test_modal_tick_mirrors_progress_into_runtime(scene):
    op = _FakeOperator(steps=400)
    job = op.start(bpy.context)
    deadline = time.monotonic() + 5.0
    while jobs.RUNTIME.progress == 0.0 and time.monotonic() < deadline:
        assert op.modal(bpy.context, _Event("TIMER")) == {"PASS_THROUGH"}
        time.sleep(0.01)
    assert jobs.RUNTIME.running and jobs.RUNTIME.name == "Fake"
    assert 0.0 < jobs.RUNTIME.progress <= 1.0 and jobs.RUNTIME.message.startswith("step ")
    assert jobs.RUNTIME["progress"] == jobs.RUNTIME.progress
    bpy.ops.stress_sculpt.cancel()  # the Cancel button sets the job's own event
    assert job.cancel.is_set() and jobs.RUNTIME.message.startswith("Cancelling")
    assert _drive(op, bpy.context) == {"CANCELLED"}
    assert not jobs.RUNTIME.running and jobs.RUNTIME.progress == 0.0
