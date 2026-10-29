"""Tests of the compute backends: device detection, threaded / mixed-precision numpy
kernels, the PyTorch solver (on the CPU) and the GPU-support preferences.

The PyTorch tests are skipped when torch is not installed; there is no GPU in the
test environment, so the CUDA paths are exercised with fake ``torch`` modules.
"""

import importlib
import os
import subprocess
import sys
import threading
import types
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from stress_sculpt.core import backend, fem
from stress_sculpt.core.grid import Cancelled, Grid, Problem

from test_fem import cantilever, dense_solve, irregular_problem, node_ids, all_dofs, simp_scale

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:  # not installed / broken
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="PyTorch is not installed")


# ---------------------------------------------------------------------- helpers
@pytest.fixture
def fresh_detection(monkeypatch):
    """Isolate the cached torch detection (restored after the test)."""
    monkeypatch.setattr(backend, "_DETECTED", None)
    yield
    backend._DETECTED = None


@pytest.fixture
def small_blocks(monkeypatch):
    """Tiny slabs / boxes / vector chunks: exercise the parallel decomposition."""
    monkeypatch.setattr(fem, "_SLAB_TARGET", 60)
    monkeypatch.setattr(fem, "_SERIAL_ELEMS", 0)
    monkeypatch.setattr(fem, "_SLAB_MIN_ELEMS", 1)
    monkeypatch.setattr(fem, "_VEC_CHUNK", 1000)


def fake_torch(available=True, cuda="12.4", name="Fake RTX 4070", cap=(8, 9), mem=12 << 30):
    mod = types.ModuleType("torch")
    mod.__version__ = "9.9.9+fake"
    mod.__file__ = os.path.join("fake", "torch", "__init__.py")
    mod.version = SimpleNamespace(cuda=cuda)
    props = SimpleNamespace(name=name, total_memory=mem, major=cap[0], minor=cap[1])
    mod.cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: 1,
        get_device_properties=lambda i: props,
        synchronize=lambda: None,
    )
    return mod


def sparse_problem(n=18, width=2, nz=4):
    """Diagonal bar across an (n, n, nz) box (~25 % of it active: indexed kernel),
    clamped at its i = 0 end, loaded at the other end."""
    g = Grid((n, n, nz), (0.0, 0.0, 0.0), 0.5)
    i, j, k = np.meshgrid(*[np.arange(s) for s in g.shape], indexing="ij")
    dom = np.abs(i - j) <= width
    nodes = node_ids(g)
    touched = g.nodes_of_elements(dom).reshape(g.nodes_shape)
    f = np.zeros(g.n_dofs)
    end = nodes[n][touched[n]]
    f[3 * end + 2] = -1.0
    f[3 * end + 0] = 0.5
    fixed = all_dofs(nodes[0][touched[0]])
    return Problem(g, dom, fixed, f, E=2.0, nu=0.3)


def cpu(threads, precision="float64"):
    return backend.cpu_backend(threads, precision)


# -------------------------------------------------------------------- detection
def test_detection_without_torch(monkeypatch, fresh_detection):
    monkeypatch.setitem(sys.modules, "torch", None)  # "import torch" -> ImportError
    info = backend.torch_info(refresh=True)
    assert not info.importable and not info.gpu_ok
    assert "not installed" in info.error
    for pref in ("AUTO", "GPU"):
        be = backend.get_backend(pref, threads=3)
        assert be.kind == "NUMPY" and be.threads == 3 and not be.is_torch
        assert "not installed" in be.reason
    assert backend.get_backend("GPU").reason.startswith("GPU requested but unavailable")
    assert backend.status_lines(info) == [info.error]


def test_cpu_preference_never_imports_torch(monkeypatch, fresh_detection):
    def boom():
        raise AssertionError("torch must not be imported for the CPU device")

    monkeypatch.setattr(backend, "_detect", boom)
    be = backend.get_backend("CPU", threads=2)
    assert be.kind == "NUMPY" and be.threads == 2 and be.reason == ""
    assert backend.cached_torch_info() is None
    assert backend.get_backend("cpu", 0).threads == backend.default_threads()
    assert backend.get_backend("bogus", 1).kind in ("NUMPY", "TORCH")  # treated as AUTO


def test_detection_broken_torch_import(monkeypatch, fresh_detection):
    class Broken:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "torch":
                raise OSError("libcudart.so.12: cannot open shared object file")
            return None

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Broken()] + sys.meta_path)
    info = backend.torch_info(refresh=True)
    assert not info.importable
    assert "could not be loaded" in info.error and "OSError" in info.error
    be = backend.get_backend("AUTO")
    assert be.kind == "NUMPY" and "could not be loaded" in be.reason


def test_detection_cpu_only_build_and_no_driver(monkeypatch, fresh_detection):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=False, cuda=None))
    info = backend.torch_info(refresh=True)
    assert info.importable and info.version == "9.9.9+fake" and not info.gpu_ok
    assert "CPU-only" in info.error
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available=False, cuda="12.4"))
    info = backend.torch_info(refresh=True)
    assert "no usable NVIDIA GPU" in info.error and info.cuda_build == "12.4"
    assert backend.get_backend("AUTO").kind == "NUMPY"


def test_detection_kernels_failing_on_the_gpu(monkeypatch, fresh_detection):
    # A real fake module has no tensor functions: the smoke test fails like a GPU
    # whose architecture is missing from the PyTorch build.
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cap=(12, 0), name="Fake RTX 5090"))
    info = backend.torch_info(refresh=True)
    assert info.importable and not info.gpu_ok and info.device_name == "Fake RTX 5090"
    assert "cannot run on this GPU" in info.error and "cu128" in info.error
    assert backend.get_backend("GPU").kind == "NUMPY"


def test_detection_working_gpu(monkeypatch, fresh_detection):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setattr(backend, "_smoke_test", lambda torch, device: None)
    info = backend.torch_info(refresh=True)
    assert info.gpu_ok and info.capability == (8, 9) and info.memory_gb == pytest.approx(12.0)
    assert "Fake RTX 4070 (12.0 GB)" in info.gpu_label()
    auto = backend.get_backend("AUTO", threads=2)
    assert auto.kind == "TORCH" and auto.is_gpu and auto.device == "cuda:0"
    assert auto.precision == "mixed" and auto.min_elements == backend.GPU_MIN_ELEMENTS
    assert auto.threads == 2 and "Fake RTX 4070" in auto.description
    gpu = backend.get_backend("GPU")
    assert gpu.is_gpu and gpu.min_elements == 0 and gpu.reason == ""
    assert backend.get_backend("GPU", precision="float64").precision == "float64"
    assert backend.cached_torch_info() is info  # cached
    assert any("GPU: Fake RTX 4070" in line for line in backend.status_lines())


def test_backend_validation_and_context():
    with pytest.raises(ValueError):
        backend.Backend(kind="OPENCL")
    with pytest.raises(ValueError):
        backend.Backend(precision="float16")
    be = backend.Backend(kind="numpy", threads=10_000)
    assert be.kind == "NUMPY" and be.threads == backend.MAX_THREADS
    assert backend.cpu_backend(0).threads == backend.default_threads() >= 1
    assert be.with_threads(3).threads == 3
    prob = cantilever((4, 3, 3))
    assert fem.FEModel(prob).backend.kind == "NUMPY"  # default
    mine = backend.cpu_backend(2, "float64")
    seen = {}
    with backend.using_backend(mine):
        assert fem.FEModel(prob).backend is mine
        assert fem.analyze(prob).info["threads"] == 2

        def other_thread():  # the setting is per thread
            seen["other"] = fem.FEModel(prob).backend

        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert seen["other"] is not mine
    assert fem.FEModel(prob).backend is not mine
    with pytest.raises(TypeError):
        fem.FEModel(prob, backend="CPU")


# ------------------------------------------------------------- numpy kernels
@pytest.mark.parametrize("which", ["solid", "irregular", "sparse"])
def test_threaded_kernels_match_reference(which, small_blocks, monkeypatch):
    if which == "sparse":
        monkeypatch.setattr(fem, "_MIN_BOX_FILL", 1.01)  # force the indexed kernel
    prob = {
        "solid": lambda: cantilever((9, 5, 4)),
        "irregular": irregular_problem,
        "sparse": sparse_problem,
    }[which]()
    rng = np.random.default_rng(8)
    model = fem.FEModel(prob, backend=cpu(3))
    loops = model._fine_loops()
    assert len(loops.slabs) > 3
    assert loops.structured == (which != "sparse")
    u = rng.standard_normal(model.n_dofs)
    s = rng.uniform(0.01, 2.0, model.n_act)
    ref = fem._apply_elements(u, model.edof, model._edof_flat, model.KE, s, model.n_dofs)
    close = np.testing.assert_allclose
    close(model.apply_K(u, s), ref * model.free, rtol=0, atol=1e-13 * np.abs(ref).max())
    ue = u[model.edof]
    energy = np.einsum("ij,ij->i", ue @ model.KE, ue)
    close(model.element_energy(u), energy, rtol=1e-12, atol=1e-14 * energy.max())
    stress = (ue @ model._stress_op.T) * s[:, None]
    close(model.element_stress(u, s), stress, rtol=1e-12, atol=1e-13 * np.abs(stress).max())
    # NaN on dofs that no active element touches is ignored (as before)
    u2 = u.copy()
    u2[~model._touched] = np.nan
    assert np.all(np.isfinite(model.apply_K(u2, s)))


@pytest.mark.parametrize("which", ["solid", "irregular", "sparse"])
def test_threads_give_bitwise_identical_results(which, small_blocks, monkeypatch):
    if which == "sparse":
        monkeypatch.setattr(fem, "_MIN_BOX_FILL", 1.01)  # force the indexed kernel
    prob = {
        "solid": lambda: cantilever((10, 6, 5)),
        "irregular": irregular_problem,
        "sparse": sparse_problem,
    }[which]()
    rng = np.random.default_rng(3)
    u = rng.standard_normal(prob.grid.n_dofs)
    results = {}
    for threads in (1, 4):
        out = []
        runs = (("float64", "MG", 8), ("mixed", "MG", 8), ("float64", "JACOBI", 25))
        for precision, solver, maxiter in runs:
            model = fem.FEModel(prob, solver, backend=cpu(threads, precision),
                                mg_options=fem.MGOptions(coarse_dofs=60))
            scale = simp_scale(model, seed=5)
            x, info = model.solve(scale, tol=1e-14, maxiter=maxiter)  # fixed iteration count
            assert info["threads"] == threads and info["solver"] == solver
            out += [x, np.array([info["iterations"], info["residual"]])]
        model = fem.FEModel(prob, backend=cpu(threads))
        out += [model.apply_K(u, None), model.element_energy(u), model.element_stress(u, None)]
        results[threads] = out
    for a, b in zip(results[1], results[4]):
        assert np.array_equal(a, b)


def test_dense_cholesky_blocks():
    rng = np.random.default_rng(0)
    for n in (7, 48, 97, 230):
        A = rng.standard_normal((n, n))
        K = A @ A.T + n * np.eye(n)
        for threads in (1, 3):
            f = fem._DenseCholesky(K, fem._Workers(threads))
            b = rng.standard_normal(n)
            np.testing.assert_allclose(K @ f.solve(b), b, rtol=0, atol=1e-11 * np.abs(b).max())
            G = f.inverse_factor()
            np.testing.assert_allclose(G.T @ G @ K, np.eye(n), atol=1e-10)
        assert np.array_equal(f.blocks, fem._DenseCholesky(K, fem._Workers(1)).blocks)
    with pytest.raises(np.linalg.LinAlgError):
        fem._DenseCholesky(-np.eye(60), fem._Workers(2))


# ------------------------------------------------------------ mixed precision
@pytest.mark.parametrize("simp", [False, True])
def test_mixed_precision_reaches_the_same_tolerance(simp):
    prob = cantilever((16, 8, 8))
    sols = {}
    for precision in ("float64", "mixed"):
        model = fem.FEModel(prob, "MG", backend=cpu(2, precision))
        scale = simp_scale(model, seed=1) if simp else None
        for tol in (1e-6, 1e-10):
            u, info = model.solve(scale, tol=tol, maxiter=2000)
            assert info["converged"] and info["solver"] == "MG"
            assert info["residual"] <= tol and info["precision"] == precision
            sols[precision, tol] = (u, info["iterations"])
    for tol in (1e-6, 1e-10):
        u64, it64 = sols["float64", tol]
        u32, it32 = sols["mixed", tol]
        assert it32 <= it64 + 3
        assert np.abs(u32 - u64).max() <= 50 * tol * np.abs(u64).max()


def test_mixed_precision_matches_dense_solution_on_irregular_domain():
    prob = irregular_problem()
    model = fem.FEModel(prob, "MG", backend=cpu(2, "mixed"))
    scale = simp_scale(model, seed=2)
    u_ref = dense_solve(prob, model, scale)
    u, info = model.solve(scale, tol=1e-11, maxiter=20000)
    assert info["converged"] and info["precision"] == "mixed" and info["residual"] <= 1e-11
    np.testing.assert_allclose(u, u_ref, rtol=0, atol=1e-8 * np.abs(u_ref).max())
    # the float32 V-cycle itself is positive definite and nearly symmetric
    M = model._mg_preconditioner(scale, precision="mixed")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(prob.grid.n_dofs) * model.free
    y = rng.standard_normal(prob.grid.n_dofs) * model.free
    assert x @ M(x) > 0 and abs(y @ M(x) - x @ M(y)) <= 1e-4 * abs(x @ M(x))


def test_torch_backend_failure_falls_back_to_numpy():
    """A device that cannot work (no CUDA here) -> numpy result, reason reported, sticky."""
    prob = cantilever((10, 4, 4))
    ref, _ = fem.FEModel(prob, backend=cpu(1)).solve(None, tol=1e-9)
    model = fem.FEModel(prob, backend=backend.torch_backend("cuda:99"))
    u, info = model.solve(None, tol=1e-9)
    assert info["converged"] and "backend_fallback" in info and "CPU" in info["backend"]
    np.testing.assert_allclose(u, ref, rtol=0, atol=1e-7 * np.abs(ref).max())
    assert model._torch_error is not None and model._torch is None
    u2, info2 = model.solve(None, tol=1e-9)
    assert "backend_fallback" in info2
    # AUTO keeps small models on the CPU without touching torch
    small = fem.FEModel(prob, backend=backend.torch_backend("cuda:99", min_elements=10**6))
    _, info3 = small.solve(None)
    assert "backend_fallback" not in info3 and small._torch_error is None


# --------------------------------------------------------------- torch solver
def torch_cpu(precision="float64"):
    return backend.torch_backend("cpu", precision, threads=2)


@pytest.fixture(autouse=True)
def torch_single_thread(request):
    """torch's OpenMP pool spins badly on busy shared machines (90x slower measured):
    the CPU-device tests run torch single-threaded (process-wide, restored after)."""
    if not HAVE_TORCH or "torch" not in request.node.name:
        yield
        return
    import torch as T

    old = T.get_num_threads()
    T.set_num_threads(1)
    try:
        yield
    finally:
        T.set_num_threads(old)


def problem_case(name):
    if name == "solid":
        return cantilever((12, 6, 6)), False
    if name == "simp":
        return cantilever((12, 6, 6)), True
    if name == "irregular":
        return irregular_problem(), True  # holes, odd sizes, roller line (partial fixed dofs)
    if name == "sparse":
        return sparse_problem(), False
    if name == "slab":
        g = Grid((14, 12, 2), (0.0, 0.0, 0.0), 1.0)
        nodes = node_ids(g)
        corners = np.concatenate([nodes[i, j, :].ravel() for i in (0, 14) for j in (0, 12)])
        f = np.zeros(g.n_dofs)
        f[3 * nodes[7, 6, 2] + 2] = -1.0
        return Problem(g, np.ones(g.shape, bool), all_dofs(corners), f), False
    raise KeyError(name)


@needs_torch
@pytest.mark.parametrize("name", ["solid", "simp", "irregular", "sparse", "slab"])
@pytest.mark.parametrize("solver", ["MG", "JACOBI"])
@pytest.mark.parametrize("precision", ["float64", "mixed"])
def test_torch_solver_matches_numpy(name, solver, precision):
    prob, simp = problem_case(name)
    opts = fem.MGOptions(coarse_dofs=150)
    ref_model = fem.FEModel(prob, solver, backend=cpu(2, precision), mg_options=opts)
    scale = simp_scale(ref_model, seed=3) if simp else None
    u_ref, info_ref = ref_model.solve(scale, tol=1e-10, maxiter=20000)
    model = fem.FEModel(prob, solver, backend=torch_cpu(precision), mg_options=opts)
    u, info = model.solve(scale, tol=1e-10, maxiter=20000)
    assert "backend_fallback" not in info, info
    assert info["device"] == "cpu" and info["converged"] and info["residual"] <= 1e-10
    assert info["solver"] == info_ref["solver"] == solver
    expected_precision = precision if solver == "MG" else "float64"
    assert info["precision"] == expected_precision
    assert abs(info["iterations"] - info_ref["iterations"]) <= max(2, info_ref["iterations"] // 50)
    np.testing.assert_allclose(u, u_ref, rtol=0, atol=1e-7 * np.abs(u_ref).max())
    assert np.all(u[~model.free] == 0.0)
    # the true residual reported by torch is the numpy definition
    s = np.ones(model.n_act) if scale is None else scale
    free = model.free & (model.diagonal(s) > 0)
    b = np.where(free, prob.forces, 0.0)
    res = np.linalg.norm(b - model.apply_K(u, s) * free) / np.linalg.norm(b)
    assert res == pytest.approx(info["residual"], rel=1e-3, abs=1e-13)


@needs_torch
def test_torch_solver_edge_cases_and_state_reuse():
    prob = cantilever((10, 5, 5))
    g = prob.grid
    zero = Problem(g, prob.domain, prob.fixed_dofs, np.zeros(g.n_dofs))
    for solver in fem.SOLVERS:
        u, info = fem.FEModel(zero, solver, backend=torch_cpu()).solve(None)
        assert np.all(u == 0.0) and info["iterations"] == 0 and info["converged"]
    every = Problem(g, prob.domain, np.arange(g.n_dofs), prob.forces)
    u, info = fem.FEModel(every, backend=torch_cpu()).solve(None)
    assert np.all(u == 0.0) and info["converged"]
    empty = Problem(g, np.zeros(g.shape, bool), prob.fixed_dofs, prob.forces)
    res = fem.analyze(empty, backend=torch_cpu())
    assert np.all(res.u == 0.0)

    model = fem.FEModel(prob, "MG", backend=torch_cpu("mixed"))
    u1, info1 = model.solve(None, tol=1e-8)
    state = model._torch
    levels = state.levels
    u2, info2 = model.solve(None, x0=u1, tol=1e-8)  # warm start: nothing to do
    assert info2["iterations"] == 0 and model._torch is state and state.levels is levels
    ops = state.ops
    u3, info3 = model.solve(None, tol=1e-8)  # same scales: operators reused, same result
    assert state.ops is ops and np.array_equal(u3, u1)
    u4, info4 = model.solve(np.full(model.n_act, 2.0), x0=u1, tol=1e-8)
    assert state.ops is not ops and state.levels is levels
    np.testing.assert_allclose(u4, 0.5 * u1, rtol=0, atol=1e-6 * np.abs(u1).max())
    model.release_device()
    assert model._torch is None
    u5, _ = model.solve(None, tol=1e-8)
    assert np.array_equal(u5, u1)  # rebuilt state, deterministic
    # mechanism: finite, not converged
    pinned = Problem(g, prob.domain, np.arange(3), prob.forces)
    u, info = fem.FEModel(pinned, backend=torch_cpu()).solve(None, maxiter=200)
    assert not info["converged"] and np.all(np.isfinite(u))


@needs_torch
def test_torch_solver_polls_should_stop():
    prob = cantilever((12, 6, 6))
    for solver in fem.SOLVERS:
        def stop(calls={"n": 0}):  # noqa: B006  (fresh dict per loop iteration)
            calls["n"] += 1
            return calls["n"] > 4

        model = fem.FEModel(prob, solver, backend=torch_cpu("mixed"))
        with pytest.raises(Cancelled):
            model.solve(None, tol=1e-12, should_stop=stop)
        assert model._torch_error is None  # cancelling is not a device failure


@needs_torch
def test_torch_failure_mid_solve_falls_back(monkeypatch):
    from stress_sculpt.core import fem_torch

    prob = cantilever((10, 5, 4))
    ref, _ = fem.FEModel(prob, backend=cpu(1)).solve(None, tol=1e-9)

    def oom(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(fem_torch.TorchSolver, "solve", oom)
    model = fem.FEModel(prob, backend=torch_cpu())
    u, info = model.solve(None, tol=1e-9)
    assert "out of memory" in info["backend_fallback"] and info["converged"]
    np.testing.assert_allclose(u, ref, rtol=0, atol=1e-7 * np.abs(ref).max())
    assert model._torch is None


@needs_torch
def test_torch_mg_matches_numpy_vcycle_in_float64():
    """Same hierarchy, same damping: one float64 V-cycle agrees to round-off."""
    prob = irregular_problem()
    model = fem.FEModel(prob, "MG", backend=torch_cpu(), mg_options=fem.MGOptions(coarse_dofs=60))
    scale = simp_scale(model, seed=9)
    M = model._mg_preconditioner(scale)  # numpy V-cycle (node-major)
    from stress_sculpt.core import fem_torch

    state = fem_torch.TorchSolver(model, model.backend)
    import torch as T

    with T.no_grad():
        diag = state.diagonal(state.fine, state.to_device(scale))
        state.mg_levels()
        state.mg_operators(scale, diag, None, T.float64)
        r = np.random.default_rng(1).standard_normal(model.n_dofs) * model.free
        z = state.vcycle(0, state.to_device(fem._soa(r))).numpy()
    np.testing.assert_allclose(fem._aos(z), M(r), rtol=0, atol=1e-9 * np.abs(M(r)).max())


# ----------------------------------------------------------------- preferences
def make_fake_wheel(folder, name="torch", version="0.0.1"):
    """Minimal pure-python wheel that provides ``import torch``."""
    path = os.path.join(folder, f"{name}-{version}-py3-none-any.whl")
    dist = f"{name}-{version}.dist-info"
    files = {
        f"{name}/__init__.py": f"__version__ = '{version}+fake'\n",
        f"{name}/sub/data.txt": "payload\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist}/entry_points.txt": f"[console_scripts]\n{name}run = {name}:main\n",
    }
    records = [f"{k},," for k in files] + [f"{dist}/RECORD,,"]
    files[f"{dist}/RECORD"] = "\n".join(records) + "\n"
    with zipfile.ZipFile(path, "w") as zf:
        for k, v in files.items():
            zf.writestr(k, v)
    return path


def test_installer_and_uninstaller_with_local_wheel(tmp_path, monkeypatch):
    prefs = pytest.importorskip("stress_sculpt.preferences")
    monkeypatch.setattr(sys, "path", list(sys.path))  # the installer appends the target
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    make_fake_wheel(str(wheels))
    target = tmp_path / "modules"
    other = target / "otherpkg-1.0.dist-info"  # someone else's package: must survive
    other.mkdir(parents=True)
    (target / "otherpkg").mkdir()
    (target / "otherpkg" / "__init__.py").write_text("x = 1\n")
    (other / "RECORD").write_text("otherpkg/__init__.py,,\n")
    job = prefs._Job()
    job.begin("INSTALL")
    offline = ["--no-index", "--find-links", str(wheels)]
    ok = prefs.install_torch(job, str(target), extra_args=offline)
    assert ok, (job.error, list(job.lines)[-10:])
    assert (target / "torch" / "__init__.py").is_file()
    manifest = (target / prefs.MANIFEST_NAME).read_text()
    assert '"torch"' in manifest and "otherpkg" not in manifest.split('"added"')[1]
    assert any("Successfully installed" in line for line in job.lines)
    assert str(target) in sys.path
    script_dirs = [target / d for d in ("bin", "Scripts") if (target / d).is_dir()]
    scripts = [p for d in script_dirs for p in d.iterdir()]
    assert any(p.name.startswith("torchrun") for p in scripts)  # console script of the wheel
    # reinstall over it: still tracked as ours
    job.begin("INSTALL")
    assert prefs.install_torch(job, str(target), extra_args=offline)
    assert '"torch"' in (target / prefs.MANIFEST_NAME).read_text().split('"added"')[1]
    job.begin("UNINSTALL")
    assert prefs.uninstall_torch(job, str(target))
    assert not (target / "torch").exists()
    assert not any(p.exists() for p in scripts)
    assert not any(p.name.startswith("torch-") for p in target.iterdir())
    assert (target / "otherpkg" / "__init__.py").is_file() and other.is_dir()
    assert not (target / prefs.MANIFEST_NAME).exists()
    # uninstall with nothing left is harmless
    job.begin("UNINSTALL")
    assert prefs.uninstall_torch(job, str(target)) and job.status == "Nothing to uninstall"


def test_cuda_build_choice():
    prefs = pytest.importorskip("stress_sculpt.preferences")
    choose = prefs.choose_cuda_build
    assert choose({})[0] == "cu126"
    assert choose({"name": "RTX 5090", "capability": 12.0, "cuda": 12.8})[0] == "cu128"
    build, why = choose({"name": "RTX 5080", "capability": 12.0, "cuda": 12.4})
    assert build == "cu128" and "update the NVIDIA driver" in why
    assert choose({"name": "RTX 4090", "capability": 8.9, "cuda": 12.9})[0] == "cu128"
    assert choose({"name": "RTX 3060", "capability": 8.6, "cuda": 12.6})[0] == "cu126"
    assert choose({"name": "RTX 3060", "capability": 8.6, "cuda": 12.4})[0] == "cu124"
    assert choose({"name": "RTX 2070", "capability": 7.5, "cuda": 12.0})[0] == "cu118"
    assert choose({"name": "GTX 1080", "capability": 6.1, "cuda": 12.8})[0] == "cu126"
    assert choose({"name": "GTX 970", "capability": 5.2, "cuda": 11.4})[0] == "cu118"
    builds = [b[0] for b in prefs.CUDA_BUILDS]
    assert {"AUTO", "cu118", "cu124"} <= set(builds)


def test_preferences_register_and_offline_refusal():
    bpy = pytest.importorskip("bpy")
    prefs = importlib.import_module("stress_sculpt.preferences")
    # the Blender-layer tests may have registered the whole add-on for the session
    owned = not prefs.STRESS_SCULPT_AddonPreferences.is_registered
    if owned:
        prefs.register()
    try:
        assert prefs.STRESS_SCULPT_AddonPreferences.is_registered
        assert prefs.STRESS_SCULPT_AddonPreferences.bl_idname == "stress_sculpt"
        assert prefs.compute_settings() == ("AUTO", 0)  # add-on not enabled: defaults
        assert prefs.status_line().startswith("Solver: ")
        if not prefs.online_access_allowed():  # default of the bpy module
            with pytest.raises(RuntimeError, match="Online access is disabled"):
                bpy.ops.stress_sculpt.install_gpu_support()
        assert bpy.ops.stress_sculpt.check_gpu.poll()
    finally:
        if owned:
            prefs.unregister()
    if owned:
        assert not prefs.STRESS_SCULPT_AddonPreferences.is_registered
        prefs.register()  # register / unregister cycles cleanly
        prefs.unregister()


def test_addon_preferences_when_enabled():
    """Enable the whole add-on in a fresh Blender process: the preferences attach."""
    pytest.importorskip("bpy")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys, bpy, addon_utils\n"
        f"sys.path.insert(0, {root!r})\n"
        "mod = addon_utils.enable('stress_sculpt', default_set=True, handle_error=None)\n"
        "assert mod is not None\n"
        "from stress_sculpt import preferences as P\n"
        "p = P.get_preferences()\n"
        "assert p is not None and p.compute_device == 'AUTO' and p.cpu_threads == 0\n"
        "p.compute_device = 'CPU'; p.cpu_threads = 3\n"
        "assert P.compute_settings() == ('CPU', 3), P.compute_settings()\n"
        "assert P.status_line() == 'Solver: CPU (3 threads)', P.status_line()\n"
        "addon_utils.disable('stress_sculpt', default_set=True)\n"
        "assert not P.STRESS_SCULPT_AddonPreferences.is_registered\n"
        "print('OK')\n"
    )
    cmd = [sys.executable, "-c", script]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout[-2000:] + res.stderr[-2000:]


class _MockLayout:
    """Records UILayout calls (no real layout exists in background mode)."""

    def __init__(self, log):
        self.log = log
        self.enabled = True
        self.alert = False
        self.use_property_split = False
        self.use_property_decorate = True

    def _child(self, *args, **kwargs):
        return _MockLayout(self.log)

    column = row = box = _child

    def label(self, text="", icon="NONE"):
        assert isinstance(text, str)
        self.log.append(("label", text, icon))

    def prop(self, data, name, **kwargs):
        assert hasattr(data, name), name
        self.log.append(("prop", name))

    def operator(self, idname, **kwargs):
        assert idname.startswith("stress_sculpt."), idname
        self.log.append(("operator", idname))
        return SimpleNamespace()


def test_preferences_draw_in_every_state(monkeypatch, fresh_detection):
    prefs = pytest.importorskip("stress_sculpt.preferences")
    cls = prefs.STRESS_SCULPT_AddonPreferences
    monkeypatch.setattr(prefs, "_maybe_auto_check", lambda: None)  # no background import
    monkeypatch.setattr(prefs, "JOB", prefs._Job())

    def draw(**state):
        log = []
        fake = SimpleNamespace(
            layout=_MockLayout(log), compute_device="AUTO", cpu_threads=0, cuda_build="AUTO",
            show_gpu_advanced=True, show_gpu_log=True,
        )
        fake.__dict__.update(state)
        fake._draw_gpu = types.MethodType(cls._draw_gpu, fake)
        cls.draw(fake, bpy_context())
        return log

    def labels(log):
        return " | ".join(entry[1] for entry in log if entry[0] == "label")

    # never checked
    log = draw()
    assert ("operator", "stress_sculpt.install_gpu_support") in log
    assert "Solver:" in labels(log)
    # torch missing
    monkeypatch.setattr(backend, "_DETECTED", backend.TorchInfo(error="PyTorch is not installed"))
    assert "PyTorch is not installed" in labels(draw())
    # a working GPU
    info = backend.TorchInfo(
        importable=True, version="2.6.0+cu124", path="/x/torch", cuda_build="12.4",
        cuda_available=True, device_name="NVIDIA GeForce RTX 4070", total_memory=12 << 30,
    )
    monkeypatch.setattr(backend, "_DETECTED", info)
    text = labels(draw(compute_device="GPU"))
    assert "RTX 4070" in text and "PyTorch 2.6.0+cu124" in text
    # an install in progress, then failed
    prefs.JOB.begin("INSTALL")
    prefs.JOB.log("Collecting torch")
    prefs.JOB.set_status("Downloading torch-2.6.0 (2.5 GB)")
    log = draw()
    assert ("operator", "stress_sculpt.cancel_gpu_job") in log and "Downloading" in labels(log)
    prefs.JOB.end(False, "Installation failed", "network error: check the internet connection")
    assert "network error" in labels(draw())
    assert "stress_sculpt.cancel_gpu_job" not in str(draw())


def bpy_context():
    bpy = pytest.importorskip("bpy")
    return bpy.context


@needs_torch
def test_topology_optimization_runs_on_the_torch_backend():
    """End to end: the Optimizer's model picks the thread's backend (no topopt change)."""
    topopt = pytest.importorskip("stress_sculpt.core.topopt")
    prob = cantilever((14, 6, 6))
    settings = topopt.OptSettings(volfrac=0.4, max_iter=4, solver_tol=1e-8)
    histories = {}
    for name, be in (("numpy", cpu(2, "float64")), ("torch", torch_cpu("mixed"))):
        with backend.using_backend(be):
            opt = topopt.Optimizer(prob, settings)
        assert opt.model.backend is be
        states = [opt.step() for _ in range(3)]
        histories[name] = (np.array([s.compliance for s in states]), states[-1].density)
        assert getattr(opt.model, "_torch_error", None) is None
    c_np, d_np = histories["numpy"]
    c_t, d_t = histories["torch"]
    np.testing.assert_allclose(c_t, c_np, rtol=1e-5)
    np.testing.assert_allclose(d_t, d_np, rtol=0, atol=1e-4)


def test_solver_compute_wrapper_sets_the_backend(monkeypatch, fresh_detection):
    prefs = pytest.importorskip("stress_sculpt.preferences")
    monkeypatch.setattr(prefs, "compute_settings", lambda context=None: ("CPU", 3))
    seen = {}

    def compute(payload, progress, should_stop, emit):
        seen["model"] = fem.FEModel(cantilever((4, 3, 3)))
        return payload + 1

    wrapped = prefs.solver_compute(compute)
    result = {}
    t = threading.Thread(target=lambda: result.update(r=wrapped(41, None, None, None)))
    t.start()
    t.join()
    assert result["r"] == 42
    assert seen["model"].backend.kind == "NUMPY" and seen["model"].threads == 3
    assert backend.current_backend() is None  # nothing leaks into this thread
