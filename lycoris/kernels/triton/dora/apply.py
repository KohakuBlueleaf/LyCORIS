"""F7: DoRA weight-decompose epilogue, one kernel per direction.

y = w * s(w) rowwise with s = m*(d/||w_row|| - 1) + 1. Grid is 1D over row
tiles and both column passes live in the same CTA: pass 1 reduces the row
(sumsq, and the g.w dot in the backward), pass 2 applies the scale. W is
therefore read twice, but the second read is L2-hot — per-CTA row bytes
BM*I*eb times the resident CTAs stays inside this card's L2 at real shapes —
so the second pass costs cache bandwidth, not DRAM, and one kernel boundary
disappears against the two-launch form.

Row/column axis is handled by transposed strides. The eps guard matches
apply_weight_decompose (torch.finfo(dtype).eps).
"""

import torch
import triton
import triton.language as tl

from ...plans import dora as plan
from ...plans import tune
from ...plans.device import resolve_device


@triton.jit
def _dora_fwd_kernel(
    w_ptr,
    d_ptr,
    norm_ptr,
    y_ptr,
    ROWS,
    COLS,
    swr,
    swc,
    syr,
    syc,
    mult,
    eps,
    RESIDENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = w * (mult*(d/||w_row|| - 1) + 1), one CTA per row block.

    RESIDENT keeps the whole row in registers between the norm and the scale,
    so W is read ONCE: traffic is read + write rather than two reads and a
    write, which is a third of the bytes off a purely bandwidth-bound op. It
    needs BLOCK_N >= COLS, so a row too wide for the register file falls to
    the two-pass form instead.
    """
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < ROWS
    if RESIDENT:
        rn = tl.arange(0, BLOCK_N)
        m = mm[:, None] & (rn < COLS)[None, :]
        w = tl.load(
            w_ptr + rm[:, None] * swr + rn[None, :] * swc, mask=m, other=0.0
        ).to(tl.float32)
        n = tl.sqrt(tl.sum(w * w, axis=1)) + eps
        tl.store(norm_ptr + rm, n, mask=mm)
        d = tl.load(d_ptr + rm, mask=mm, other=1.0).to(tl.float32)
        s = mult * (d / n - 1.0) + 1.0
        tl.store(
            y_ptr + rm[:, None] * syr + rn[None, :] * syc,
            (w * s[:, None]).to(y_ptr.dtype.element_ty),
            mask=m,
        )
    else:
        acc = tl.zeros((BLOCK_M,), tl.float32)
        for c0 in range(0, COLS, BLOCK_N):
            rn = c0 + tl.arange(0, BLOCK_N)
            m = mm[:, None] & (rn < COLS)[None, :]
            w = tl.load(
                w_ptr + rm[:, None] * swr + rn[None, :] * swc, mask=m, other=0.0
            ).to(tl.float32)
            acc += tl.sum(w * w, axis=1)
        n = tl.sqrt(acc) + eps
        tl.store(norm_ptr + rm, n, mask=mm)
        d = tl.load(d_ptr + rm, mask=mm, other=1.0).to(tl.float32)
        s = mult * (d / n - 1.0) + 1.0
        for c0 in range(0, COLS, BLOCK_N):
            rn = c0 + tl.arange(0, BLOCK_N)
            m = mm[:, None] & (rn < COLS)[None, :]
            w = tl.load(
                w_ptr + rm[:, None] * swr + rn[None, :] * swc, mask=m, other=0.0
            ).to(tl.float32)
            tl.store(
                y_ptr + rm[:, None] * syr + rn[None, :] * syc,
                (w * s[:, None]).to(y_ptr.dtype.element_ty),
                mask=m,
            )


@triton.jit
def _dora_bwd_kernel(
    g_ptr,
    w_ptr,
    d_ptr,
    norm_ptr,
    gw_ptr,
    gd_ptr,
    ROWS,
    COLS,
    sgr,
    sgc,
    swr,
    swc,
    sor,
    soc,
    mult,
    RESIDENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Row-dot pass + grad pass for one row tile, in one CTA.

    gw = g*s + mult*d*(-rowdot/n^3)*w and gd = mult*rowdot/n, with
    s = mult*(d/n - 1) + 1 and rowdot = sum_c g*w. RESIDENT holds both rows
    between the two passes, so g and w are each read once.
    """
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < ROWS
    if RESIDENT:
        rn = tl.arange(0, BLOCK_N)
        m = mm[:, None] & (rn < COLS)[None, :]
        g = tl.load(
            g_ptr + rm[:, None] * sgr + rn[None, :] * sgc, mask=m, other=0.0
        ).to(tl.float32)
        w = tl.load(
            w_ptr + rm[:, None] * swr + rn[None, :] * swc, mask=m, other=0.0
        ).to(tl.float32)
        rdot = tl.sum(g * w, axis=1)
        d = tl.load(d_ptr + rm, mask=mm, other=1.0).to(tl.float32)
        n = tl.load(norm_ptr + rm, mask=mm, other=1.0)
        s = mult * (d / n - 1.0) + 1.0
        coef = -mult * d * rdot / (n * n * n)
        tl.store(gd_ptr + rm, (mult * rdot / n).to(gd_ptr.dtype.element_ty), mask=mm)
        tl.store(
            gw_ptr + rm[:, None] * sor + rn[None, :] * soc,
            (g * s[:, None] + coef[:, None] * w).to(gw_ptr.dtype.element_ty),
            mask=m,
        )
        return
    # Pass 1: rowdot = sum_c g*w (the norm is reused from the forward).
    rd = tl.zeros((BLOCK_M,), tl.float32)
    for c0 in range(0, COLS, BLOCK_N):
        rn = c0 + tl.arange(0, BLOCK_N)
        m = mm[:, None] & (rn < COLS)[None, :]
        g = tl.load(
            g_ptr + rm[:, None] * sgr + rn[None, :] * sgc, mask=m, other=0.0
        ).to(tl.float32)
        w = tl.load(
            w_ptr + rm[:, None] * swr + rn[None, :] * swc, mask=m, other=0.0
        ).to(tl.float32)
        rd += tl.sum(g * w, axis=1)
    d = tl.load(d_ptr + rm, mask=mm, other=1.0).to(tl.float32)
    n = tl.load(norm_ptr + rm, mask=mm, other=1.0)
    s = mult * (d / n - 1.0) + 1.0
    coef = -mult * d * rd / (n * n * n)
    tl.store(gd_ptr + rm, (mult * rd / n).to(gd_ptr.dtype.element_ty), mask=mm)
    # Pass 2: gw = g*s + coef*w.
    for c0 in range(0, COLS, BLOCK_N):
        rn = c0 + tl.arange(0, BLOCK_N)
        m = mm[:, None] & (rn < COLS)[None, :]
        g = tl.load(
            g_ptr + rm[:, None] * sgr + rn[None, :] * sgc, mask=m, other=0.0
        ).to(tl.float32)
        w = tl.load(
            w_ptr + rm[:, None] * swr + rn[None, :] * swc, mask=m, other=0.0
        ).to(tl.float32)
        tl.store(
            gw_ptr + rm[:, None] * sor + rn[None, :] * soc,
            (g * s[:, None] + coef[:, None] * w).to(gw_ptr.dtype.element_ty),
            mask=m,
        )


def _strides2(w: torch.Tensor, row_axis: int):
    if row_axis == 0:
        return w.shape[0], w.shape[1], w.stride(0), w.stride(1)
    return w.shape[1], w.shape[0], w.stride(1), w.stride(0)


def _pick(rows, cols, w, name, factory):
    return tune.tuned(
        name,
        (rows, cols, str(w.dtype)),
        lambda: plan.topk_row_reduce(rows, cols, resolve_device(), w.element_size()),
        factory,
    )


# A resident row costs BLOCK_M*BLOCK_N fp32 registers per CTA. 4096 of them
# over 128 threads is 32 per thread; past that the register file spills and
# the two-pass form is cheaper than the traffic it saves.
RESIDENT_BUDGET = 4096
RESIDENT_COLS = 4096


def _resident(cols: int) -> tuple[bool, int, int]:
    """(hold the rows, BLOCK_M, BLOCK_N) — BLOCK_N covers COLS when resident."""
    if cols > RESIDENT_COLS:
        return False, 0, 0
    bn = max(16, 1 << (cols - 1).bit_length())
    return True, max(1, RESIDENT_BUDGET // bn), bn


def dora_fwd(
    w: torch.Tensor,
    dscale: torch.Tensor,
    mult: float = 1.0,
    row_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(y, norms) in one launch; norms are saved for the backward."""
    rows, cols, sr, sc = _strides2(w, row_axis)
    y = torch.empty_like(w)
    _, _, oyr, oyc = _strides2(y, row_axis)
    norms = torch.empty(rows, device=w.device, dtype=torch.float32)
    eps = torch.finfo(w.dtype).eps

    res, res_bm, res_bn = _resident(cols)

    def launch(p, o_norm, o_y):
        _dora_fwd_kernel[(triton.cdiv(rows, res_bm if res else p.bm),)](
            w,
            dscale.reshape(-1),
            o_norm,
            o_y,
            rows,
            cols,
            sr,
            sc,
            oyr,
            oyc,
            mult,
            eps,
            RESIDENT=res,
            BLOCK_M=res_bm if res else p.bm,
            BLOCK_N=res_bn if res else p.bn,
            BLOCK_K=1,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        s1, s2 = torch.empty_like(norms), torch.empty_like(y)
        return lambda: launch(p, s1, s2)

    best = _pick(rows, cols, w, "triton.dora.fwd", factory)
    launch(best, norms, y)
    return y, norms


def dora_bwd(
    grad: torch.Tensor,
    w: torch.Tensor,
    dscale: torch.Tensor,
    norms: torch.Tensor,
    mult: float = 1.0,
    row_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(gw, gd) in one launch."""
    rows, cols, sr, sc = _strides2(w, row_axis)
    _, _, gr, gc = _strides2(grad, row_axis)
    gw = torch.empty_like(w)
    _, _, owr, owc = _strides2(gw, row_axis)
    # Each CTA owns its rows outright, so gd is stored in the parameter dtype
    # directly — no fp32 staging buffer and no cast launch.
    gd = torch.empty(rows, device=w.device, dtype=dscale.dtype)
    res, res_bm, res_bn = _resident(cols)

    def launch(p, o_gw, o_gd):
        _dora_bwd_kernel[(triton.cdiv(rows, res_bm if res else p.bm),)](
            grad,
            w,
            dscale.reshape(-1),
            norms,
            o_gw,
            o_gd,
            rows,
            cols,
            gr,
            gc,
            sr,
            sc,
            owr,
            owc,
            mult,
            RESIDENT=res,
            BLOCK_M=res_bm if res else p.bm,
            BLOCK_N=res_bn if res else p.bn,
            BLOCK_K=1,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        s1, s2 = torch.empty_like(gw), torch.empty_like(gd)
        return lambda: launch(p, s1, s2)

    best = _pick(rows, cols, w, "triton.dora.bwd", factory)
    launch(best, gw, gd)
    return gw, gd
