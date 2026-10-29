"""PyTorch implementation of :meth:`.fem.FEModel.solve` (CUDA GPU, or CPU for testing).

The algorithm mirrors the numpy solver of :mod:`.fem` step by step (same tolerance
definition, warm start, restarts, best-iterate fallback, Jacobi fallback, should_stop
polling every iteration, progress every 0.2 s):

* the multigrid hierarchy (levels, coarsening factors, fixed dofs) is built by the
  numpy code and uploaded once per model; per solve the level scales are coarsened
  on the host (cheap) and everything else runs on the device: diagonals, Lanczos
  estimates of the Jacobi damping, the smoother, the restriction / prolongation
  (exact transposes, same slicing formulas as numpy), and the dense coarsest solve
  (Cholesky on the device, ``K^-1 = G^T G`` applied with two matrix-vector products),
* matrix-free products: ``index_select`` of the 24 element dofs (component-major
  numbering, int32 indices), one ``mm`` with ``KE`` per chunk of elements and eight
  ``index_add_`` calls (one per local node, so the indices of a call are unique: no
  atomic conflicts, deterministic sums), in bounded chunks of elements,
* ``precision="mixed"``: the V-cycle (smoother, transfers, level stiffness) runs in
  float32 inside the float64 conjugate gradient, which then uses the flexible
  Polak-Ribiere update; the coarsest solve and the true-residual test stay float64.
  This is what makes consumer GPUs fast (their float64 rate is 1/32 - 1/64 of float32).

Device memory: per model the level index arrays (~100 bytes per element), about 15
vectors per level and two element buffers of at most ``_CHUNK`` elements; they are
reused by every solve of the model (topology optimization solves ~40 times) and freed
by :func:`release` / ``FEModel.release_device``. Errors (CUDA out of memory, driver
problems) propagate: ``FEModel.solve`` catches them and falls back to numpy.

Importing this module requires torch.
"""

import time

import numpy as np
import torch

from . import fem as _fem
from .grid import check_stop

_CHUNK = 1 << 18  # elements per matrix-free chunk: bounds the (24, chunk) temporaries
_COARSE_MAX_DEVICE = 6000  # coarsest systems up to this size are factored on the device


def _soa_index_t(edof: np.ndarray, n_nodes: int) -> np.ndarray:
    return (edof % 3) * int(n_nodes) + edof // 3


class _DeviceLevel:
    """Geometry of one multigrid level on the device (built once per model)."""

    def __init__(self, level, device, index_dtype):
        self.shape = tuple(level.shape)
        self.nodes_shape = tuple(level.nodes_shape)
        self.n_nodes = level.n_nodes
        self.n_dofs = level.n_dofs
        self.factors = tuple(level.factors)
        self.n_act = int(level.elems.size)
        self.device = device
        e = _soa_index_t(level.edof, level.n_nodes)
        self.chunks = []  # (start, stop, (24, m) contiguous dof ids)
        for s in range(0, self.n_act, _CHUNK):
            t = min(self.n_act, s + _CHUNK)
            idx = torch.from_numpy(np.ascontiguousarray(e[s:t].T))
            idx = idx.to(device=device, dtype=index_dtype)
            self.chunks.append((s, t, idx))
        self.ke = {torch.float64: torch.from_numpy(np.ascontiguousarray(level.ke)).to(device)}
        self.ke[torch.float32] = self.ke[torch.float64].to(torch.float32)
        self.ke_diag = torch.diagonal(self.ke[torch.float64]).contiguous()
        self.free_geo = torch.from_numpy(_fem._soa(level.free)).to(device)
        self.lambda_bound = float(level.lambda_bound)

    def element_ke(self, dtype):
        return self.ke[dtype]


class _DeviceOp:
    """Scale-dependent part of a level for one solve (device tensors)."""

    __slots__ = ("dtype", "scale", "freef", "dinv", "omega", "ke", "G", "GT", "cidx")

    def __init__(self):
        self.dtype = torch.float64
        self.scale = None
        self.freef = None
        self.dinv = None
        self.omega = 0.0
        self.ke = None
        self.G = None
        self.GT = None
        self.cidx = None


class TorchSolver:
    """Device state of one :class:`~.fem.FEModel` (cached as ``model._torch``)."""

    def __init__(self, model, backend):
        self.model = model
        self.device = torch.device(backend.device)
        self.precision = backend.precision
        n_dofs = model.n_dofs
        self.index_dtype = torch.int32 if 3 * n_dofs < np.iinfo(np.int32).max else torch.int64
        fine_np = _fem._Level(
            shape=model.grid.shape,
            sizes=np.full(3, model.grid.h),
            elems=model.elems,
            edof=model.edof,
            edof_flat=model._edof_flat,
            ke=model.KE,
            fixed=model.fixed,
            free=model.free,
            lambda_bound=_fem._element_lambda_bound(model.KE),
        )
        self.fine = _DeviceLevel(fine_np, self.device, self.index_dtype)
        self.levels = None  # device levels (MG), lazily
        self.ops = None
        self.ops_key = None
        self._buf = {}

    # ----------------------------------------------------------------- helpers
    def release(self) -> None:
        self.levels = None
        self.ops = None
        self.ops_key = None
        self._buf = {}
        self.fine = None
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _buffers(self, dtype, m: int):
        key = (dtype, "elem")
        ue, fe = self._buf.get(key, (None, None))
        if ue is None or ue.numel() < 24 * m:
            size = 24 * min(_CHUNK, max(m, 1))
            ue = torch.empty(size, dtype=dtype, device=self.device)
            fe = torch.empty(size, dtype=dtype, device=self.device)
            self._buf[key] = (ue, fe)
        return ue, fe

    def to_device(self, a: np.ndarray, dtype=None) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        return t if dtype is None else t.to(dtype)

    # ----------------------------------------------------------------- kernels
    def apply(self, lvl: _DeviceLevel, x, scale, ke, out=None):
        """``sum_e s_e KE x_e`` (component-major, nothing masked)."""
        if out is None:
            out = torch.zeros(lvl.n_dofs, dtype=x.dtype, device=self.device)
        else:
            out.zero_()
        for s, t, idx in lvl.chunks:
            m = t - s
            ue_b, fe_b = self._buffers(x.dtype, m)
            ue = ue_b[: 24 * m]
            fe = fe_b[: 24 * m]
            torch.index_select(x, 0, idx.view(-1), out=ue)
            torch.mm(ke, ue.view(24, m), out=fe.view(24, m))
            fe2 = fe.view(24, m)
            fe2.mul_(scale[s:t])
            for a in range(8):  # one call per local node: unique indices, no conflicts
                rows = slice(3 * a, 3 * a + 3)
                out.index_add_(0, idx[rows].reshape(-1), fe2[rows].reshape(-1))
        return out

    def diagonal(self, lvl: _DeviceLevel, scale64):
        out = torch.zeros(lvl.n_dofs, dtype=torch.float64, device=self.device)
        for s, t, idx in lvl.chunks:
            vals = lvl.ke_diag[:, None] * scale64[s:t][None, :]
            for a in range(8):
                rows = slice(3 * a, 3 * a + 3)
                out.index_add_(0, idx[rows].reshape(-1), vals[rows].reshape(-1))
        return out

    def level_apply(self, lvl, op, x):
        y = self.apply(lvl, x, op.scale, op.ke)
        y.mul_(op.freef)
        return y

    def prolong(self, xc, fine: _DeviceLevel, coarse: _DeviceLevel):
        a = xc.view((3,) + coarse.nodes_shape)
        for axis in range(3):
            if fine.factors[axis] == 2:
                a = _prolong_axis(a, axis + 1, fine.nodes_shape[axis])
        return a.reshape(-1)

    def restrict(self, rf, fine: _DeviceLevel, coarse: _DeviceLevel):
        a = rf.view((3,) + fine.nodes_shape)
        for axis in range(3):
            if fine.factors[axis] == 2:
                a = _restrict_axis(a, axis + 1, coarse.nodes_shape[axis])
        return a.reshape(-1)

    # --------------------------------------------------------------- multigrid
    def mg_levels(self):
        if self.levels is None:
            levels_np = self.model._mg_geometry()  # raises _MGSetupError
            dev = [self.fine]
            for level in levels_np[1:]:
                dev.append(_DeviceLevel(level, self.device, self.index_dtype))
            self.fine.factors = tuple(levels_np[0].factors)
            self.levels = dev
        return self.levels

    def mg_operators(self, scale_np, diag64, should_stop, dtype):
        key = (dtype,)
        if (
            self.ops is not None
            and self.ops_key == key
            and np.array_equal(self._ops_scale, scale_np)
        ):
            return self.ops
        model = self.model
        levels_np = model._mg_geometry()
        levels = self.mg_levels()
        opts = model.mg_options
        scales = model._level_scales(levels_np, scale_np)
        ops = []
        last = len(levels) - 1
        for li, (lv_np, lvl) in enumerate(zip(levels_np, levels)):
            check_stop(should_stop)
            op = _DeviceOp()
            op.dtype = dtype
            if li == last:
                # dense coarsest solve (float64), assembled on the host
                if li == 0:
                    diag_np = model.diagonal(scale_np)
                else:
                    diag_np = _fem._assembled_diagonal(
                        lv_np.edof, np.diag(lv_np.ke), scales[li], lv_np.n_dofs
                    )
                free_np = lv_np.free & (diag_np > 0.0)
                idx, K = _fem._assemble_dense(lv_np, scales[li], free_np)
                op.freef = self.to_device(_fem._soa(free_np), dtype)
                op.cidx = self.to_device(_fem._soa_index(idx, lv_np.n_nodes))
                if K is not None:
                    op.G, op.GT = self._coarse_factor(K)
                ops.append(op)
                continue
            s64 = self.to_device(scales[li])
            d = diag64 if li == 0 else self.diagonal(lvl, s64)
            free = lvl.free_geo & (d > 0.0)
            d_inv = _safe_inverse(d, free)
            op.scale = s64.to(dtype)
            op.ke = lvl.element_ke(dtype)
            op.freef = free.to(dtype)
            if opts.omega is not None:
                omega = float(opts.omega)
            else:
                rng = np.random.default_rng(1234567)  # same start vector as numpy
                v0 = self.to_device(_fem._soa(rng.standard_normal(lvl.n_dofs)), dtype) * op.freef
                d_isqrt = torch.sqrt(d_inv).to(dtype)
                lam = self.lanczos(lvl, op, v0, d_isqrt, opts.lanczos_steps, should_stop)
                lam_used = lvl.lambda_bound
                if np.isfinite(lam) and lam > 0.0:
                    lam_used = min(lam_used, opts.lambda_margin * max(lam, 1.0))
                omega = opts.omega_scale / lam_used
            op.omega = omega
            op.dinv = (omega * d_inv).to(dtype)
            ops.append(op)
        self.ops = ops
        self.ops_key = key
        self._ops_scale = scale_np.copy()
        return ops

    def _coarse_factor(self, K: np.ndarray):
        n = K.shape[0]
        dev = self.device if n <= _COARSE_MAX_DEVICE else torch.device("cpu")
        Kt = torch.from_numpy(K).to(dev)
        mean_diag = float(np.mean(np.diag(K)))
        eye = torch.eye(n, dtype=torch.float64, device=dev)
        for reg in _fem._COARSE_REGULARIZATION:
            L, info = torch.linalg.cholesky_ex(Kt + (reg * mean_diag) * eye)
            if int(info.item()) != 0:
                continue
            G = torch.linalg.solve_triangular(L, eye, upper=False)
            if not bool(torch.isfinite(G).all().item()):
                continue
            G = G.to(self.device)
            return G, G.t().contiguous()
        raise _fem._MGSetupError("coarsest multigrid matrix is not positive definite")

    def lanczos(self, lvl, op, v0, d_isqrt, steps, should_stop) -> float:
        v = v0.clone()
        nv = float(torch.linalg.vector_norm(v).item())
        if nv == 0.0:
            return 0.0
        v /= nv
        v_prev = torch.zeros_like(v)
        beta = 0.0
        alphas, betas = [], []
        for _ in range(max(1, steps)):
            check_stop(should_stop)
            w = d_isqrt * self.level_apply(lvl, op, d_isqrt * v)
            alpha = float(torch.dot(w, v).item())
            w -= alpha * v + beta * v_prev
            alphas.append(alpha)
            beta = float(torch.linalg.vector_norm(w).item())
            if not np.isfinite(beta) or beta <= 1e-12 * max(abs(alpha), 1e-300):
                break
            betas.append(beta)
            v_prev, v = v, w / beta
        k = len(alphas)
        T = np.diag(alphas) + np.diag(betas[: k - 1], 1) + np.diag(betas[: k - 1], -1)
        return float(np.linalg.eigvalsh(T)[-1])

    def vcycle(self, li, r, should_stop=None):
        levels, ops = self.levels, self.ops
        lvl, op = levels[li], ops[li]
        if li == len(levels) - 1:
            x = torch.zeros(lvl.n_dofs, dtype=op.dtype, device=self.device)
            if op.G is not None:
                rc = r[op.cidx].to(torch.float64)
                z = torch.mv(op.GT, torch.mv(op.G, rc))
                x[op.cidx] = z.to(op.dtype)
            return x
        sweeps = max(1, int(self.model.mg_options.sweeps))
        x = op.dinv * r
        ax = torch.empty_like(x)
        for _ in range(sweeps - 1):
            self.apply(lvl, x, op.scale, op.ke, ax)
            x.addcmul_(op.dinv, r - ax)
        if li == 0:
            check_stop(should_stop)
        self.apply(lvl, x, op.scale, op.ke, ax)
        res = (r - ax).mul_(op.freef)
        coarse, cop = levels[li + 1], ops[li + 1]
        rc = self.restrict(res, lvl, coarse).mul_(cop.freef)
        xc = self.vcycle(li + 1, rc)
        corr = self.prolong(xc, lvl, coarse)
        x.addcmul_(corr, op.freef)
        if li == 0:
            check_stop(should_stop)
        for _ in range(sweeps):
            self.apply(lvl, x, op.scale, op.ke, ax)
            x.addcmul_(op.dinv, r - ax)
        return x

    # ------------------------------------------------------------------- solve
    def solve(self, scale_np, x0, tol, maxiter, should_stop, progress):
        model = self.model
        dev = self.device
        tol = float(tol)
        maxiter = max(0, int(maxiter))
        f64 = torch.float64
        vdtype = torch.float32 if self.precision == "mixed" else f64
        be = model.backend
        info = {
            "iterations": 0,
            "residual": 0.0,
            "converged": True,
            "solver": model.solver,
            "backend": be.description,
            "device": str(dev),
            "threads": model.threads,
            "precision": "float64",
        }
        n = model.n_dofs
        fine = self.fine
        scale64 = self.to_device(scale_np)
        diag = self.diagonal(fine, scale64)
        free = fine.free_geo & (diag > 0.0)
        b = self.to_device(_fem._soa(model.problem.forces)) * free
        bnorm = float(torch.linalg.vector_norm(b).item())
        if bnorm == 0.0 or not bool(free.any().item()):
            return np.zeros(n), info
        freef = free.to(f64)
        jacobi_dinv = _safe_inverse(diag, free)
        ke64 = fine.element_ke(f64)

        def apply(v):
            y = self.apply(fine, v, scale64, ke64)
            y.mul_(freef)
            return y

        def jacobi(v):
            return jacobi_dinv * v

        def dot(a, c) -> float:
            return float(torch.dot(a, c).item())

        precond = jacobi
        flexible = False
        info["solver"] = "JACOBI"
        if model.solver == "MG":
            if progress is not None:
                progress(0.0, "Preparing the multigrid solver")
            try:
                self.mg_levels()
                self.mg_operators(scale_np, diag, should_stop, vdtype)
            except _fem._MGSetupError as exc:
                info["fallback"] = str(exc)
            else:
                if vdtype == torch.float32:

                    def precond(v):
                        nrm = torch.linalg.vector_norm(v)
                        inv = torch.where(nrm > 0, 1.0 / nrm, torch.zeros_like(nrm))
                        z = self.vcycle(0, (v * inv).to(vdtype), should_stop)
                        return z.to(f64) * nrm

                    flexible = True
                    info["precision"] = "mixed"
                else:
                    precond = lambda v: self.vcycle(0, v, should_stop)  # noqa: E731
                info["solver"] = "MG"

        x = torch.zeros(n, dtype=f64, device=dev)
        r = b.clone()
        if x0 is not None and np.all(np.isfinite(x0)):
            xw = self.to_device(_fem._soa(x0)) * freef
            rw = b - apply(xw)
            if float(torch.linalg.vector_norm(rw).item()) < bnorm:
                x, r = xw, rw
        rel = float(torch.linalg.vector_norm(r).item()) / bnorm
        best_x, best_rel = x.clone(), rel
        rel_true = rel
        it = 0
        restarts = 0
        breakdown = False
        last_report = time.perf_counter()
        log_tol = np.log(tol) if 0.0 < tol < 1.0 else None

        while rel > tol and it < maxiter:
            z = precond(r)
            rz = dot(r, z)
            if not (np.isfinite(rz) and rz > 0.0):
                if info["solver"] != "MG":
                    breakdown = True
                    break
                precond = jacobi
                flexible = False
                info["solver"] = "JACOBI"
                info["precision"] = "float64"
                info["fallback"] = "multigrid preconditioner was not positive definite"
                continue
            p = z.clone()
            rz_t = torch.dot(r, z)  # device copy: the step length stays on the device
            switch = False
            rel_true = None
            while True:
                check_stop(should_stop)
                q = apply(p)
                pq_t = torch.dot(p, q)
                alpha_t = rz_t / pq_t
                x.add_(p * alpha_t)
                r.sub_(q * alpha_t)
                # one host sync for both scalars; a bad step (pq <= 0) is discarded
                # below by returning the best iterate, as in the numpy solver
                pq, rr = torch.stack((pq_t, torch.dot(r, r))).tolist()
                if not (np.isfinite(pq) and pq > 0.0):
                    breakdown = True
                    break
                alpha = rz / pq
                it += 1
                rel = float(np.sqrt(rr)) / bnorm if rr >= 0.0 else float("nan")
                if not np.isfinite(rel):
                    breakdown = True
                    break
                if rel < best_rel:
                    best_rel = rel
                    best_x.copy_(x)
                if rel <= tol or it >= maxiter:
                    break
                z = precond(r)
                rz_t = torch.dot(r, z)
                if flexible:
                    rz_new, zq = torch.stack((rz_t, torch.dot(z, q))).tolist()
                else:
                    rz_new = float(rz_t.item())
                if not (np.isfinite(rz_new) and rz_new > 0.0):
                    if info["solver"] == "MG":
                        precond = jacobi
                        flexible = False
                        info["solver"] = "JACOBI"
                        info["precision"] = "float64"
                        info["fallback"] = "multigrid preconditioner was not positive definite"
                        switch = True
                    else:
                        breakdown = True
                    break
                if flexible:  # Polak-Ribiere: z_new . (r_new - r_old) / rz
                    beta = -alpha * zq / rz
                else:
                    beta = rz_new / rz
                p.mul_(beta).add_(z)
                rz = rz_new
                if progress is not None:
                    now = time.perf_counter()
                    if now - last_report > 0.2:
                        last_report = now
                        frac = 0.0
                        if log_tol is not None and rel > 0.0:
                            frac = float(np.clip(np.log(rel) / log_tol, 0.0, 1.0))
                        msg = f"Solving on {be.description}: residual {rel:.1e} (iteration {it})"
                        progress(frac, msg)
            if breakdown:
                break
            r = b - apply(x)
            rel = rel_true = float(torch.linalg.vector_norm(r).item()) / bnorm
            if not switch:
                if rel <= tol or it >= maxiter or restarts >= 3:
                    break
                restarts += 1

        if breakdown or rel_true is None or rel_true > tol:
            if breakdown or best_rel < rel or not bool(torch.isfinite(x).all().item()):
                x = best_x
            rel_true = float(torch.linalg.vector_norm(b - apply(x)).item()) / bnorm
        info["iterations"] = it
        info["residual"] = rel_true
        info["converged"] = bool(rel_true <= tol)
        if breakdown:
            info["breakdown"] = True
        x.mul_(freef)
        u = _fem._aos(x.cpu().numpy())
        return u, info


def _safe_inverse(d, mask):
    """``1 / d`` where ``mask``, 0 elsewhere (float64)."""
    return torch.where(mask, 1.0 / torch.where(mask, d, torch.ones_like(d)), torch.zeros_like(d))


# ----------------------------------------------------------------- transfers
def _prolong_axis(a, axis: int, n_fine: int):
    """Same formula as ``fem._prolong_axis`` (torch tensors)."""
    src = a.movedim(axis, 0)
    n_even = (n_fine + 1) // 2
    n_odd = n_fine // 2
    out = torch.empty((n_fine,) + tuple(src.shape[1:]), dtype=a.dtype, device=a.device)
    out[0::2] = src[:n_even]
    out[1::2] = (src[:n_odd] + src[1 : n_odd + 1]).mul_(0.5)
    return out.movedim(0, axis)


def _restrict_axis(a, axis: int, n_coarse: int):
    """Exact transpose of :func:`_prolong_axis` (same formula as numpy)."""
    src = a.movedim(axis, 0)
    n_fine = src.shape[0]
    n_even = (n_fine + 1) // 2
    n_odd = n_fine // 2
    out = torch.zeros((n_coarse,) + tuple(src.shape[1:]), dtype=a.dtype, device=a.device)
    out[:n_even] += src[0::2]
    half = src[1::2] * 0.5
    out[:n_odd] += half
    out[1 : n_odd + 1] += half
    return out.movedim(0, axis)


# ------------------------------------------------------------------------ entry
def solve(model, scale, x0, tol, maxiter, should_stop, progress) -> tuple:
    """Solve with the model's torch backend (state cached on the model).

    ``scale`` and ``x0`` are validated numpy arrays (see ``FEModel.solve``). Raises
    whatever torch raises (the caller falls back to numpy); ``Cancelled`` passes.
    """
    state = model._torch
    if state is None or state.fine is None:
        state = TorchSolver(model, model.backend)
        model._torch = state
    with torch.no_grad():
        return state.solve(scale, x0, tol, maxiter, should_stop, progress)


def release(model) -> None:
    state = getattr(model, "_torch", None)
    if state is not None:
        state.release()
        model._torch = None
