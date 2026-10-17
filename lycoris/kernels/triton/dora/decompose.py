"""DoRA weight decompose y = M * (mult*(d/||M_row|| - 1) + 1), M = w [+ delta],
on the model-ranked tuner, one kernel per direction (eps = finfo(dtype).eps).
See docs/kernels/designs.md.
"""

import torch
import triton
import triton.language as tl

from ...tuner import TORCH, device_info, select
from ...tuner.measure import (
    copy_of,
    dtype_key,
    grid_configs,
    guarded,
    launch,
    measurer,
    pow2,
)
from ...tuner.model import Work, cdiv, rank
from ..common import OutOfResources

# Largest resident tile (fp32 elements of M per program).
_RESIDENT_TILE = 16384


@triton.jit
def _merged(w_ptr, dl_ptr, off, m, has_delta: tl.constexpr):
    """The M = w [+ delta] tile in fp32."""
    mg = tl.load(w_ptr + off, mask=m, other=0.0).to(tl.float32)
    if has_delta:
        mg += tl.load(dl_ptr + off, mask=m, other=0.0).to(tl.float32)
    return mg


@triton.jit
def _fwd_kernel(
    w_ptr,
    dl_ptr,
    d_ptr,
    norm_ptr,
    y_ptr,
    rows,
    cols,
    sr,
    sc,
    mult,
    eps,
    has_delta: tl.constexpr,
    resident: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """y = M * s with s = mult*(d/n - 1) + 1, n = ||M_row|| + eps, and the norms.

    Grid: (row tiles,); resident holds the whole rows of M in registers.
    """
    pid = tl.program_id(0)
    rm = pid * block_m + tl.arange(0, block_m)
    mm = rm < rows
    if resident:
        rn = tl.arange(0, block_n)
        m = mm[:, None] & (rn < cols)[None, :]
        off = rm[:, None] * sr + rn[None, :] * sc
        mg = _merged(w_ptr, dl_ptr, off, m, has_delta)
        n = tl.sqrt(tl.sum(mg * mg, axis=1)) + eps
    else:
        acc = tl.zeros((block_m,), tl.float32)
        for c0 in range(0, cols, block_n):
            rn = c0 + tl.arange(0, block_n)
            m = mm[:, None] & (rn < cols)[None, :]
            off = rm[:, None] * sr + rn[None, :] * sc
            mg = _merged(w_ptr, dl_ptr, off, m, has_delta)
            acc += tl.sum(mg * mg, axis=1)
        n = tl.sqrt(acc) + eps
    tl.store(norm_ptr + rm, n, mask=mm)
    d = tl.load(d_ptr + rm, mask=mm, other=1.0).to(tl.float32)
    s = mult * (d / n - 1.0) + 1.0

    # y = M * s, M from registers when resident.
    for c0 in range(0, cols, block_n):
        rn = c0 + tl.arange(0, block_n)
        m = mm[:, None] & (rn < cols)[None, :]
        off = rm[:, None] * sr + rn[None, :] * sc
        if not resident:
            mg = _merged(w_ptr, dl_ptr, off, m, has_delta)
        tl.store(y_ptr + off, (mg * s[:, None]).to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _bwd_kernel(
    g_ptr,
    w_ptr,
    dl_ptr,
    d_ptr,
    norm_ptr,
    gw_ptr,
    gd_ptr,
    rows,
    cols,
    sr,
    sc,
    mult,
    has_delta: tl.constexpr,
    resident: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """gM = g*s + coef*M with coef = -mult*d*rowdot/n^3, and gd = mult*rowdot/n,
    rowdot = sum_c g*M; resident holds g and M between the two passes."""
    pid = tl.program_id(0)
    rm = pid * block_m + tl.arange(0, block_m)
    mm = rm < rows
    if resident:
        rn = tl.arange(0, block_n)
        m = mm[:, None] & (rn < cols)[None, :]
        off = rm[:, None] * sr + rn[None, :] * sc
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        mg = _merged(w_ptr, dl_ptr, off, m, has_delta)
        rdot = tl.sum(g * mg, axis=1)
    else:
        rdot = tl.zeros((block_m,), tl.float32)
        for c0 in range(0, cols, block_n):
            rn = c0 + tl.arange(0, block_n)
            m = mm[:, None] & (rn < cols)[None, :]
            off = rm[:, None] * sr + rn[None, :] * sc
            g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
            mg = _merged(w_ptr, dl_ptr, off, m, has_delta)
            rdot += tl.sum(g * mg, axis=1)
    d = tl.load(d_ptr + rm, mask=mm, other=1.0).to(tl.float32)
    n = tl.load(norm_ptr + rm, mask=mm, other=1.0)
    s = mult * (d / n - 1.0) + 1.0
    coef = -mult * d * rdot / (n * n * n)
    tl.store(gd_ptr + rm, (mult * rdot / n).to(gd_ptr.dtype.element_ty), mask=mm)
    for c0 in range(0, cols, block_n):
        rn = c0 + tl.arange(0, block_n)
        m = mm[:, None] & (rn < cols)[None, :]
        off = rm[:, None] * sr + rn[None, :] * sc
        if not resident:
            g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
            mg = _merged(w_ptr, dl_ptr, off, m, has_delta)
        gw = g * s[:, None] + coef[:, None] * mg
        tl.store(gw_ptr + off, gw.to(gw_ptr.dtype.element_ty), mask=m)


def _plan(dev, rows, cols, reads, writes, backward):
    """Resident and streamed configs, model-ranked, then torch; reads / writes
    are bytes per element of one pass."""
    resident = [
        {"RESIDENT": True, "BLOCK_M": m, "num_warps": w, "num_stages": 1}
        for m in (1, 2, 4, 8)
        for w in (2, 4, 8, 16)
        if m * pow2(cols) <= _RESIDENT_TILE
    ]
    streamed = grid_configs(
        dev,
        RESIDENT=(False,),
        BLOCK_M=(1, 2, 4, 8),
        BLOCK_N=(512, 1024, 2048),
        num_warps=(4, 8),
        num_stages=(1,),
    )

    def work(c):
        bm = c["BLOCK_M"]
        tile = bm * (pow2(cols) if c["RESIDENT"] else c["BLOCK_N"])
        # streamed configs read every operand twice (the second read L2-hot)
        passes = 1 if c["RESIDENT"] else 2
        return Work(
            grid=cdiv(rows, bm),
            warps=c["num_warps"],
            acc_fp32=tile * (2 if backward else 1),
            operand_bytes=tile * reads,
            smem_stage=0,
            stages=1,
            hbm_bytes=rows * cols * (reads + writes),
            vector_flops=(10.0 if backward else 6.0) * rows * cols,
            l2_bytes=rows * cols * (passes * reads + writes),
        )

    return [*rank(dev, resident + streamed, work), TORCH]


def _axes(w: torch.Tensor, row_axis: int):
    if row_axis == 0:
        return w.shape[0], w.shape[1], w.stride(0), w.stride(1)
    return w.shape[1], w.shape[0], w.stride(1), w.stride(0)


def _eager_fwd(w, dscale, mult, row_axis, delta):
    mg = w.float() if delta is None else w.float() + delta.float()
    n = mg.norm(dim=1 - row_axis) + torch.finfo(w.dtype).eps
    s = mult * (dscale.reshape(-1).float() / n - 1.0) + 1.0
    shape = (-1, 1) if row_axis == 0 else (1, -1)
    return (mg * s.reshape(shape)).to(w.dtype), n


def _fwd_run(cfg, w, dscale, mult, row_axis, delta):
    if cfg == TORCH:
        return _eager_fwd(w, dscale, mult, row_axis, delta)
    rows, cols, sr, sc = _axes(w, row_axis)
    y = torch.empty_like(w)
    norms = torch.empty(rows, device=w.device, dtype=torch.float32)
    block_n = pow2(cols) if cfg["RESIDENT"] else cfg["BLOCK_N"]
    _fwd_kernel[(triton.cdiv(rows, cfg["BLOCK_M"]),)](
        w,
        delta if delta is not None else w,
        dscale.reshape(-1),
        norms,
        y,
        rows,
        cols,
        sr,
        sc,
        mult,
        torch.finfo(w.dtype).eps,
        has_delta=delta is not None,
        resident=cfg["RESIDENT"],
        block_m=cfg["BLOCK_M"],
        block_n=block_n,
        **launch(cfg),
    )
    return y, norms


def dora_fwd(w, dscale, mult: float = 1.0, row_axis: int = 0, delta=None):
    """(y, fp32 norms) of the weight decompose on M = w [+ delta]; w and delta
    are 2-D, y in w's dtype."""
    w = w.contiguous()
    delta = None if delta is None else delta.contiguous()
    dev = device_info(w.device)
    rows, cols, *_ = _axes(w, row_axis)
    e = w.element_size()
    reads = e * (2 if delta is not None else 1)
    key = (
        "dora.fwd",
        (dtype_key(w.dtype), row_axis, delta is not None),
        (rows, cols),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, rows, cols, reads, e, False),
        measurer(
            dev,
            2 * w.numel() * e,
            lambda: (copy_of(w),),
            lambda c, s: lambda: _fwd_run(c, s[0], dscale, mult, row_axis, delta),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _fwd_run(c, w, dscale, mult, row_axis, delta),
        OutOfResources,
    )


def _bwd_run(cfg, grad, w, dscale, norms, mult, row_axis, delta):
    if cfg == TORCH:
        mg = w.float() if delta is None else w.float() + delta.float()
        shape = (-1, 1) if row_axis == 0 else (1, -1)
        g = grad.float()
        rdot = (g * mg).sum(dim=1 - row_axis)
        d = dscale.reshape(-1).float()
        s = mult * (d / norms - 1.0) + 1.0
        coef = -mult * d * rdot / norms**3
        gw = g * s.reshape(shape) + coef.reshape(shape) * mg
        return gw.to(w.dtype), (mult * rdot / norms).to(dscale.dtype)
    rows, cols, sr, sc = _axes(w, row_axis)
    gw = torch.empty_like(w)
    gd = torch.empty(rows, device=w.device, dtype=dscale.dtype)
    block_n = pow2(cols) if cfg["RESIDENT"] else cfg["BLOCK_N"]
    _bwd_kernel[(triton.cdiv(rows, cfg["BLOCK_M"]),)](
        grad,
        w,
        delta if delta is not None else w,
        dscale.reshape(-1),
        norms,
        gw,
        gd,
        rows,
        cols,
        sr,
        sc,
        mult,
        has_delta=delta is not None,
        resident=cfg["RESIDENT"],
        block_m=cfg["BLOCK_M"],
        block_n=block_n,
        **launch(cfg),
    )
    return gw, gd


def dora_bwd(grad, w, dscale, norms, mult: float = 1.0, row_axis: int = 0, delta=None):
    """(gM, gd) of the weight decompose: gM is the gradient of M = w [+ delta]
    in w's dtype, gd in dscale's dtype."""
    grad, w = grad.contiguous(), w.contiguous()
    delta = None if delta is None else delta.contiguous()
    dev = device_info(w.device)
    rows, cols, *_ = _axes(w, row_axis)
    e = w.element_size()
    reads = e * (3 if delta is not None else 2)
    key = (
        "dora.bwd",
        (dtype_key(w.dtype), row_axis, delta is not None),
        (rows, cols),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, rows, cols, reads, e, True),
        measurer(
            dev,
            3 * w.numel() * e,
            lambda: (copy_of(grad),),
            lambda c, s: lambda: _bwd_run(
                c, s[0], w, dscale, norms, mult, row_axis, delta
            ),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _bwd_run(c, grad, w, dscale, norms, mult, row_axis, delta),
        OutOfResources,
    )
