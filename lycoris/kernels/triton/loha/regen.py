"""LoHa bypass y = [base +] gamma * x @ ((a1 @ b1) * (a2 @ b2))^T on the
model-ranked tuner, forward and backward. See docs/kernels/designs.md.
"""

import torch
import triton
import triton.language as tl

from ...limits import MAX_HADA_RANK
from ...tuner import TORCH, device_info, select
from ...tuner.measure import (
    copy_of,
    dtype_key,
    grid_configs,
    guarded,
    launch,
    measurer,
    pad16,
    token_bucket,
)
from ...tuner.model import Work, cdiv, rank, shared_reads
from ..common import OutOfResources, dot_prec, load2d
from . import bypass
from .hada import MAX_RANK, loha_merge, loha_merge_bwd

# Bypass forms: DeltaW then one addmm with the base, or DeltaW then a matmul and an add.
MATERIALIZE = {"impl": "materialize"}
MATERIALIZE_ADD = {"impl": "materialize_add"}
WHOLE = {"impl": "whole"}


@triton.jit
def _bypass_regen_kernel(
    x_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_out,
    n_in,
    rank,
    sxt,
    sxi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    sbt,
    sbo,
    syt,
    syo,
    scale,
    has_base: tl.constexpr,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """y = [base +] scale * x @ DeltaW^T, the DeltaW tile generated per K step.

    Grid: (token tiles, output column tiles).
    """
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * block_m + tl.arange(0, block_m)
    rn = pid_n * block_n + tl.arange(0, block_n)
    rr = tl.arange(0, block_r)
    mt = rt < n_tok
    mn = rn < n_out
    mr = rr < rank

    # Up-factor rows of this output tile, held for the whole K loop.
    a1 = load2d(a1_ptr, rn, rr, sa1o, sa1r, mn, mr)
    a2 = load2d(a2_ptr, rn, rr, sa2o, sa2r, mn, mr)

    # acc (tokens, outputs) = x @ DeltaW^T over the input axis.
    acc = tl.zeros((block_m, block_n), tl.float32)
    for k0 in range(0, n_in, block_k):
        rk = k0 + tl.arange(0, block_k)
        mk = rk < n_in
        x = load2d(x_ptr, rt, rk, sxt, sxi, mt, mk)
        b1 = load2d(b1_ptr, rr, rk, sb1r, sb1i, mr, mk)
        b2 = load2d(b2_ptr, rr, rk, sb2r, sb2i, mr, mk)

        # DeltaW tile (outputs, k) = (a1 @ b1) * (a2 @ b2), then acc += x @ tile^T.
        w = tl.dot(a1, b1, input_precision=prec) * tl.dot(a2, b2, input_precision=prec)
        acc = tl.dot(x, tl.trans(w.to(x.dtype)), acc, input_precision=prec)

    # Epilogue: scale, add base, store.
    y = acc * scale
    if has_base:
        y += load2d(b_ptr, rt, rn, sbt, sbo, mt, mn).to(tl.float32)
    tl.store(
        y_ptr + rt[:, None] * syt + rn[None, :] * syo,
        y.to(y_ptr.dtype.element_ty),
        mask=mt[:, None] & mn[None, :],
    )


def _plan_bypass(dev, t, o, i, r, e, has_base):
    """Regen configs model-ranked, then the materialize forms, the
    single-launch kernel and torch."""
    alternatives = [MATERIALIZE, MATERIALIZE_ADD, TORCH]
    rp = pad16(r)
    if rp > MAX_RANK:
        return alternatives
    alternatives.append(WHOLE)
    space = grid_configs(
        dev,
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(32, 64),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, bk = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]
        tt, tn = cdiv(t, bm), cdiv(o, bn)
        return Work(
            grid=tt * tn,
            warps=c["num_warps"],
            acc_fp32=bm * bn + 2 * bn * bk,
            operand_bytes=0,
            smem_stage=(bm * bk + 2 * (bn + bk) * rp) * e,
            stages=c["num_stages"],
            hbm_bytes=t * i * e * tn
            + t * o * e * (2 if has_base else 1)
            + shared_reads(dev, 2 * (o + i) * rp * e, tt * tn),
            mma_flops=2.0 * tt * bm * tn * bn * i + 4.0 * tt * tn * bn * i * rp,
            vector_flops=tt * tn * bn * i,
            # each tile reads its x rows, both ups' rows and both downs whole
            l2_bytes=tt * tn * (bm * i + 2 * bn * rp + 2 * rp * i) * e
            + t * o * e * (2 if has_base else 1),
        )

    return [*rank(dev, space, work), *alternatives]


def _bypass_run(cfg, flat, factors, scale, base2, out_dtype):
    if cfg == TORCH:
        a1, b1, a2, b2 = factors
        y = flat @ ((a1 @ b1) * (a2 @ b2)).t() * scale
        return (y if base2 is None else base2 + y).to(out_dtype)
    if cfg == WHOLE:
        y = bypass.loha_bypass_fwd(flat, *factors, scale)
        return (y if base2 is None else base2 + y).to(out_dtype)
    a1, b1, a2, b2 = factors
    t, i = flat.shape
    o, r = a1.shape
    if cfg in (MATERIALIZE, MATERIALIZE_ADD):
        delta = loha_merge(a1, b1, a2, b2, scale)
        same = base2 is not None and base2.dtype == flat.dtype
        if same and cfg == MATERIALIZE:
            return torch.addmm(base2, flat, delta.t())
        y = flat @ delta.t()
        if base2 is None:
            return y
        return y.add_(base2) if same else (base2 + y).to(out_dtype)
    out = torch.empty(t, o, device=flat.device, dtype=out_dtype)
    b = base2 if base2 is not None else out
    grid = (triton.cdiv(t, cfg["BLOCK_M"]), triton.cdiv(o, cfg["BLOCK_N"]))
    _bypass_regen_kernel[grid](
        flat,
        a1,
        b1,
        a2,
        b2,
        b,
        out,
        t,
        o,
        i,
        r,
        *flat.stride(),
        *a1.stride(),
        *b1.stride(),
        *a2.stride(),
        *b2.stride(),
        *b.stride(),
        *out.stride(),
        scale,
        has_base=base2 is not None,
        prec=dot_prec(flat.dtype),
        block_r=pad16(r),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_k=cfg["BLOCK_K"],
        **launch(cfg),
    )
    return out


def loha_bypass_fwd(x, a1, b1, a2, b2, gamma: float = 1.0, base=None):
    """[base +] gamma * x @ ((a1 @ b1) * (a2 @ b2))^T for x (T, I); the result
    in base's dtype when given, else x's."""
    dev = device_info(x.device)
    t, i = x.shape
    o, r = a1.shape
    out_dtype = base.dtype if base is not None else x.dtype
    factors = (a1, b1, a2, b2)
    key = (
        "loha.bypass_fwd",
        (dtype_key(x.dtype), dtype_key(out_dtype), base is not None),
        (token_bucket(t), o, i, r),
    )

    def inputs():
        return copy_of(x), copy_of(base)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass(dev, t, o, i, r, x.element_size(), base is not None),
        measurer(
            dev,
            t * (i + o) * x.element_size(),
            inputs,
            lambda c, s: lambda: _bypass_run(c, s[0], factors, gamma, s[1], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _bypass_run(c, x, factors, gamma, base, out_dtype),
        OutOfResources,
    )


def _bypass_bwd_materialized(grad, x, a1, b1, a2, b2, gamma):
    delta = loha_merge(a1, b1, a2, b2, gamma)
    gx = grad @ delta
    gw = grad.t() @ x
    return (gx, *loha_merge_bwd(gw, a1, b1, a2, b2, gamma))


def loha_bypass_bwd(grad, x, a1, b1, a2, b2, gamma: float = 1.0):
    """(gx, g_a1, g_b1, g_a2, g_b2) of the LoHa bypass in the compute dtype."""
    r = a1.shape[1]
    if r > MAX_HADA_RANK:
        return _bypass_bwd_materialized(grad, x, a1, b1, a2, b2, gamma)
    dev = device_info(x.device)
    key = (
        "loha.bypass_bwd",
        (dtype_key(x.dtype),),
        (token_bucket(x.shape[0]), a1.shape[0], x.shape[1], r),
    )

    def run(cf, g, xx):
        if cf == WHOLE:
            return bypass.loha_bypass_bwd(g, xx, a1, b1, a2, b2, gamma)
        return _bypass_bwd_materialized(g, xx, a1, b1, a2, b2, gamma)

    cfg = select(
        dev,
        key,
        lambda: [MATERIALIZE, WHOLE],
        measurer(
            dev,
            2 * grad.numel() * grad.element_size(),
            lambda: (copy_of(grad), copy_of(x)),
            lambda cf, s: lambda: run(cf, *s),
        ),
    )
    return run(cfg, grad, x)
