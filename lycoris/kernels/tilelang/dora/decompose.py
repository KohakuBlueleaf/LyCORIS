"""DoRA weight decompose y = M * (mult*(d/||M_row|| - 1) + 1), M = w [+ delta],
in TileLang on the model-ranked tuner, one kernel per direction (eps =
finfo(dtype).eps). Rows are reduced along the contiguous axis; the column
form runs on the transposed copy. See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer
from ...tuner.model import Work, cdiv, rank
from ..common import CompileError, compiled, tl_dtype
from . import apply

F32 = "float32"
# The per-direction kernels of ``.apply``.
APPLY = {"impl": "apply"}
# Largest resident tile (fp32 elements of M per program).
_RESIDENT_TILE = 16384
_KERNEL_DTYPES = (torch.float16, torch.float32)
_ZERO = T.float32(0)


def _fwd_prim(rows, cols, bm, bn, resident, threads, w_dt, d_dt, has_delta):
    """y = M * s, s = mult*(d/n - 1) + 1, n = ||M_row|| + eps, and the norms;
    a program owns bm rows (whole rows in registers when resident)."""
    chunks = cdiv(cols, bn)
    exact = rows % bm == 0 and cols % bn == 0

    @T.prim_func
    def main(
        w: T.Tensor((rows, cols), w_dt),
        dl: T.Tensor((rows, cols), w_dt),
        d: T.Tensor((rows,), d_dt),
        norm: T.Tensor((rows,), F32),
        y: T.Tensor((rows, cols), w_dt),
        mult: T.float32,
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(rows, bm), threads=threads) as bx:
            m_f = T.alloc_fragment((bm, bn), F32)
            sq = T.alloc_fragment((bm, bn), F32)
            part = T.alloc_fragment((bm,), F32)
            acc = T.alloc_fragment((bm,), F32)
            s = T.alloc_fragment((bm,), F32)
            T.clear(acc)
            # Pass 1: sum of M^2 per row.
            for k in T.serial(chunks):
                for i, j in T.Parallel(bm, bn):
                    r = bx * bm + i
                    c = k * bn + j
                    if exact:
                        m_f[i, j] = T.cast(w[r, c], F32)
                        if has_delta:
                            m_f[i, j] += T.cast(dl[r, c], F32)
                    else:
                        ok = (r < rows) and (c < cols)
                        rr = T.min(r, rows - 1)
                        cc = T.min(c, cols - 1)
                        m_f[i, j] = T.if_then_else(ok, T.cast(w[rr, cc], F32), _ZERO)
                        if has_delta:
                            m_f[i, j] += T.if_then_else(
                                ok, T.cast(dl[rr, cc], F32), _ZERO
                            )
                    sq[i, j] = m_f[i, j] * m_f[i, j]
                T.reduce_sum(sq, part, dim=1)
                for i in T.Parallel(bm):
                    acc[i] += part[i]
            for i in T.Parallel(bm):
                r = T.min(bx * bm + i, rows - 1)
                n = T.sqrt(acc[i]) + eps
                s[i] = mult * (T.cast(d[r], F32) / n - 1.0) + 1.0
                if bx * bm + i < rows:
                    norm[bx * bm + i] = n
            # Pass 2: y = M * s, M from registers when resident.
            for k in T.serial(chunks):
                for i, j in T.Parallel(bm, bn):
                    r = bx * bm + i
                    c = k * bn + j
                    if resident:
                        if exact:
                            y[r, c] = T.cast(m_f[i, j] * s[i], w_dt)
                        elif r < rows and c < cols:
                            y[r, c] = T.cast(m_f[i, j] * s[i], w_dt)
                    elif exact:
                        if has_delta:
                            y[r, c] = T.cast(
                                (T.cast(w[r, c], F32) + T.cast(dl[r, c], F32)) * s[i],
                                w_dt,
                            )
                        else:
                            y[r, c] = T.cast(T.cast(w[r, c], F32) * s[i], w_dt)
                    elif r < rows and c < cols:
                        if has_delta:
                            y[r, c] = T.cast(
                                (T.cast(w[r, c], F32) + T.cast(dl[r, c], F32)) * s[i],
                                w_dt,
                            )
                        else:
                            y[r, c] = T.cast(T.cast(w[r, c], F32) * s[i], w_dt)

    return main


def _bwd_prim(rows, cols, bm, bn, resident, threads, w_dt, d_dt, has_delta):
    """gM = g*s + coef*M with coef = -mult*d*rowdot/n^3 and gd = mult*rowdot/n,
    rowdot = sum_c g*M; resident holds g and M between the two passes."""
    chunks = cdiv(cols, bn)
    exact = rows % bm == 0 and cols % bn == 0

    @T.prim_func
    def main(
        g: T.Tensor((rows, cols), w_dt),
        w: T.Tensor((rows, cols), w_dt),
        dl: T.Tensor((rows, cols), w_dt),
        d: T.Tensor((rows,), d_dt),
        norm: T.Tensor((rows,), F32),
        gw: T.Tensor((rows, cols), w_dt),
        gd: T.Tensor((rows,), d_dt),
        mult: T.float32,
    ):
        with T.Kernel(T.ceildiv(rows, bm), threads=threads) as bx:
            m_f = T.alloc_fragment((bm, bn), F32)
            g_f = T.alloc_fragment((bm, bn), F32)
            gm = T.alloc_fragment((bm, bn), F32)
            part = T.alloc_fragment((bm,), F32)
            acc = T.alloc_fragment((bm,), F32)
            s = T.alloc_fragment((bm,), F32)
            coef = T.alloc_fragment((bm,), F32)
            T.clear(acc)
            # Pass 1: rowdot = sum_c g*M.
            for k in T.serial(chunks):
                for i, j in T.Parallel(bm, bn):
                    r = bx * bm + i
                    c = k * bn + j
                    if exact:
                        g_f[i, j] = T.cast(g[r, c], F32)
                        m_f[i, j] = T.cast(w[r, c], F32)
                        if has_delta:
                            m_f[i, j] += T.cast(dl[r, c], F32)
                    else:
                        ok = (r < rows) and (c < cols)
                        rr = T.min(r, rows - 1)
                        cc = T.min(c, cols - 1)
                        g_f[i, j] = T.if_then_else(ok, T.cast(g[rr, cc], F32), _ZERO)
                        m_f[i, j] = T.if_then_else(ok, T.cast(w[rr, cc], F32), _ZERO)
                        if has_delta:
                            m_f[i, j] += T.if_then_else(
                                ok, T.cast(dl[rr, cc], F32), _ZERO
                            )
                    gm[i, j] = g_f[i, j] * m_f[i, j]
                T.reduce_sum(gm, part, dim=1)
                for i in T.Parallel(bm):
                    acc[i] += part[i]
            for i in T.Parallel(bm):
                r = T.min(bx * bm + i, rows - 1)
                n = norm[r]
                dv = T.cast(d[r], F32)
                s[i] = mult * (dv / n - 1.0) + 1.0
                coef[i] = -mult * dv * acc[i] / (n * n * n)
                if bx * bm + i < rows:
                    gd[bx * bm + i] = T.cast(mult * acc[i] / n, d_dt)
            # Pass 2: gM = g*s + coef*M.
            for k in T.serial(chunks):
                for i, j in T.Parallel(bm, bn):
                    r = bx * bm + i
                    c = k * bn + j
                    if resident:
                        if exact:
                            gw[r, c] = T.cast(
                                g_f[i, j] * s[i] + coef[i] * m_f[i, j], w_dt
                            )
                        elif r < rows and c < cols:
                            gw[r, c] = T.cast(
                                g_f[i, j] * s[i] + coef[i] * m_f[i, j], w_dt
                            )
                    elif exact:
                        if has_delta:
                            gw[r, c] = T.cast(
                                T.cast(g[r, c], F32) * s[i]
                                + coef[i]
                                * (T.cast(w[r, c], F32) + T.cast(dl[r, c], F32)),
                                w_dt,
                            )
                        else:
                            gw[r, c] = T.cast(
                                T.cast(g[r, c], F32) * s[i]
                                + coef[i] * T.cast(w[r, c], F32),
                                w_dt,
                            )
                    elif r < rows and c < cols:
                        if has_delta:
                            gw[r, c] = T.cast(
                                T.cast(g[r, c], F32) * s[i]
                                + coef[i]
                                * (T.cast(w[r, c], F32) + T.cast(dl[r, c], F32)),
                                w_dt,
                            )
                        else:
                            gw[r, c] = T.cast(
                                T.cast(g[r, c], F32) * s[i]
                                + coef[i] * T.cast(w[r, c], F32),
                                w_dt,
                            )

    return main


def _plan(dev, rows, cols, reads, writes, backward):
    """Resident and streamed configs, model-ranked, then torch and the
    ``.apply`` kernels; reads / writes are bytes per element of one pass."""
    # whole rows padded to a multiple of 128 lanes (a fragment splits evenly over threads)
    whole = cdiv(cols, 128) * 128
    space = [
        {"RESIDENT": True, "BM": m, "BN": whole, "threads": th}
        for m in (1, 2, 4, 8)
        for th in (128, 256, 512)
        if m * whole * (2 if backward else 1) <= _RESIDENT_TILE
        and (m * whole) % th == 0
    ]
    space += [
        {"RESIDENT": False, "BM": m, "BN": n, "threads": th}
        for m in (1, 2, 4, 8)
        for n in (512, 1024, 2048)
        for th in (128, 256)
        if n <= cols
    ]

    def work(c):
        tile = c["BM"] * c["BN"]
        # streamed configs read every operand twice (the second read L2-hot)
        passes = 1 if c["RESIDENT"] else 2
        return Work(
            grid=cdiv(rows, c["BM"]),
            warps=c["threads"] // 32,
            acc_fp32=tile * (3 if backward else 2),
            operand_bytes=tile * reads,
            smem_stage=0,
            stages=1,
            hbm_bytes=rows * cols * (reads + writes),
            vector_flops=(10.0 if backward else 6.0) * rows * cols,
            l2_bytes=rows * cols * (passes * reads + writes),
        )

    return [*rank(dev, space, work), TORCH, APPLY]


def _eager_fwd(w, dscale, mult, delta):
    mg = w.float() if delta is None else w.float() + delta.float()
    n = mg.norm(dim=1) + torch.finfo(w.dtype).eps
    s = mult * (dscale.reshape(-1).float() / n - 1.0) + 1.0
    return (mg * s[:, None]).to(w.dtype), n


def _fwd_run(cfg, w, dscale, mult, delta):
    """(y, norms) of the row form on the contiguous (rows, cols) w."""
    if cfg == TORCH:
        return _eager_fwd(w, dscale, mult, delta)
    if cfg == APPLY:
        return apply.dora_fwd(w, dscale, mult, 0, delta)
    rows, cols = w.shape
    d = dscale.reshape(-1)
    kernel = compiled(
        _fwd_prim,
        w.device,
        rows,
        cols,
        cfg["BM"],
        cfg["BN"],
        cfg["RESIDENT"],
        cfg["threads"],
        tl_dtype(w.dtype),
        tl_dtype(d.dtype),
        delta is not None,
    )
    y = torch.empty_like(w)
    norms = torch.empty(rows, device=w.device, dtype=torch.float32)
    dl = delta if delta is not None else w
    kernel(w, dl, d, norms, y, float(mult), float(torch.finfo(w.dtype).eps))
    return y, norms


def _bwd_run(cfg, grad, w, dscale, norms, mult, delta):
    if cfg == TORCH:
        mg = w.float() if delta is None else w.float() + delta.float()
        g = grad.float()
        rdot = (g * mg).sum(dim=1)
        d = dscale.reshape(-1).float()
        s = mult * (d / norms - 1.0) + 1.0
        coef = -mult * d * rdot / norms**3
        gw = g * s[:, None] + coef[:, None] * mg
        return gw.to(w.dtype), (mult * rdot / norms).to(dscale.dtype)
    if cfg == APPLY:
        gw, gd = apply.dora_bwd(grad, w, dscale, norms, mult, 0, delta)
        return gw, gd.to(dscale.dtype)
    rows, cols = w.shape
    d = dscale.reshape(-1)
    kernel = compiled(
        _bwd_prim,
        w.device,
        rows,
        cols,
        cfg["BM"],
        cfg["BN"],
        cfg["RESIDENT"],
        cfg["threads"],
        tl_dtype(w.dtype),
        tl_dtype(d.dtype),
        delta is not None,
    )
    gw = torch.empty_like(w)
    gd = torch.empty(rows, device=w.device, dtype=d.dtype)
    dl = delta if delta is not None else w
    kernel(grad, w, dl, d, norms, gw, gd, float(mult))
    return gw, gd


def _rows(t, row_axis):
    """t as the contiguous (rows, cols) row form."""
    return (t if row_axis == 0 else t.t()).contiguous()


def dora_fwd(w, dscale, mult: float = 1.0, row_axis: int = 0, delta=None):
    """(y, fp32 norms) of the weight decompose on M = w [+ delta]; w and delta
    are 2-D, y in w's dtype."""
    wr = _rows(w, row_axis)
    dr = None if delta is None else _rows(delta.to(w.dtype), row_axis)
    if w.dtype not in _KERNEL_DTYPES or dscale.dtype not in _KERNEL_DTYPES:
        y, n = _fwd_run(APPLY, wr, dscale, mult, dr)
        return (y if row_axis == 0 else y.t()), n
    dev = device_info(w.device)
    rows, cols = wr.shape
    e = w.element_size()
    reads = e * (2 if delta is not None else 1)
    key = (
        "tl.dora.fwd",
        (dtype_key(w.dtype), dtype_key(dscale.dtype), delta is not None),
        (rows, cols),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, rows, cols, reads, e, False),
        measurer(
            dev,
            2 * w.numel() * e,
            lambda: (copy_of(wr),),
            lambda c, s: lambda: _fwd_run(c, s[0], dscale, mult, dr),
        ),
    )
    y, n = guarded(
        dev, key, cfg, lambda c: _fwd_run(c, wr, dscale, mult, dr), CompileError
    )
    return (y if row_axis == 0 else y.t()), n


def dora_bwd(grad, w, dscale, norms, mult: float = 1.0, row_axis: int = 0, delta=None):
    """(gM, gd) of the weight decompose: gM is the gradient of M = w [+ delta]
    in w's dtype, gd in dscale's dtype."""
    wr, gr = _rows(w, row_axis), _rows(grad.to(w.dtype), row_axis)
    dr = None if delta is None else _rows(delta.to(w.dtype), row_axis)
    if w.dtype not in _KERNEL_DTYPES or dscale.dtype not in _KERNEL_DTYPES:
        gw, gd = _bwd_run(APPLY, gr, wr, dscale, norms, mult, dr)
        return (gw if row_axis == 0 else gw.t()), gd
    dev = device_info(w.device)
    rows, cols = wr.shape
    e = w.element_size()
    reads = e * (3 if delta is not None else 2)
    key = (
        "tl.dora.bwd",
        (dtype_key(w.dtype), dtype_key(dscale.dtype), delta is not None),
        (rows, cols),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, rows, cols, reads, e, True),
        measurer(
            dev,
            3 * w.numel() * e,
            lambda: (copy_of(gr),),
            lambda c, s: lambda: _bwd_run(c, s[0], wr, dscale, norms, mult, dr),
        ),
    )
    gw, gd = guarded(
        dev,
        key,
        cfg,
        lambda c: _bwd_run(c, gr, wr, dscale, norms, mult, dr),
        CompileError,
    )
    return (gw if row_axis == 0 else gw.t()), gd
