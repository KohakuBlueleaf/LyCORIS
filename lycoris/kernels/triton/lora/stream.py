"""LoRA bypass y = [base +] gamma * (x @ down^T) @ up^T on the model-ranked
tuner (linear layout), forward and backward. See docs/kernels/designs.md.
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
    pad16,
    token_bucket,
)
from ...tuner.model import Work, cdiv, rank, shared_reads, staged
from ..common import OutOfResources, dot_prec, load2d
from . import bypass

# Largest padded rank the fused bypass holds in registers.
MAX_RANK = 128
ATOMIC = {"impl": "atomic"}


@triton.jit
def _bypass_fwd_kernel(
    x_ptr,
    d_ptr,
    u_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_out,
    n_in,
    rank,
    sxt,
    sxi,
    sdr,
    sdi,
    suo,
    sur,
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
    split_o: tl.constexpr,
):
    """y = [base +] scale * (x @ down^T) @ up^T for one token tile and one slice of O.

    Grid: (token tiles, split_o output slices).
    """
    pid_t = tl.program_id(0)
    pid_o = tl.program_id(1)
    rt = pid_t * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mt = rt < n_tok
    mr = rr < rank

    # h (tokens, rank) in fp32 registers: x tile @ down^T over the input axis.
    h = tl.zeros((block_m, block_r), tl.float32)
    for i0 in range(0, n_in, block_k):
        ri = i0 + tl.arange(0, block_k)
        mi = ri < n_in
        xv = load2d(x_ptr, rt, ri, sxt, sxi, mt, mi)
        dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)
        h = tl.dot(xv, dv, h, input_precision=prec)
    # h in x's dtype, as the two-Linear chain rounds it.
    hs = h.to(x_ptr.dtype.element_ty)

    # Output columns [lo, hi) of this program's slice, streamed block_n at a time.
    per = tl.cdiv(tl.cdiv(n_out, block_n), split_o) * block_n
    lo = pid_o * per
    hi = tl.minimum(lo + per, n_out)
    for o0 in range(lo, hi, block_n):
        ro = o0 + tl.arange(0, block_n)
        mo = ro < n_out
        uv = load2d(u_ptr, rr, ro, sur, suo, mr, mo)
        y = tl.dot(hs, uv, input_precision=prec) * scale
        if has_base:
            y += load2d(b_ptr, rt, ro, sbt, sbo, mt, mo).to(tl.float32)
        tl.store(
            y_ptr + rt[:, None] * syt + ro[None, :] * syo,
            y.to(y_ptr.dtype.element_ty),
            mask=mt[:, None] & mo[None, :],
        )


@triton.jit
def _bypass_bwd_kernel(
    x_ptr,
    d_ptr,
    u_ptr,
    g_ptr,
    gx_ptr,
    h_ptr,
    q_ptr,
    n_tok,
    n_out,
    n_in,
    rank,
    sxt,
    sxi,
    sdr,
    sdi,
    suo,
    sur,
    sgt,
    sgo,
    sgxt,
    sgxi,
    scale,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """gx = scale * (g @ up) @ down per token tile, storing h and q for the GEMMs.

    Grid: (token tiles,).
    """
    pid = tl.program_id(0)
    rt = pid * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mt = rt < n_tok
    mr = rr < rank
    tr_mask = mt[:, None] & mr[None, :]

    # h = x @ down^T (tokens, rank), stored for the g_up GEMM.
    h = tl.zeros((block_m, block_r), tl.float32)
    for i0 in range(0, n_in, block_k):
        ri = i0 + tl.arange(0, block_k)
        mi = ri < n_in
        xv = load2d(x_ptr, rt, ri, sxt, sxi, mt, mi)
        dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)
        h = tl.dot(xv, dv, h, input_precision=prec)
    tl.store(
        h_ptr + rt[:, None] * rank + rr[None, :],
        h.to(h_ptr.dtype.element_ty),
        mask=tr_mask,
    )

    # q = g @ up (tokens, rank), stored for the g_down GEMM.
    q = tl.zeros((block_m, block_r), tl.float32)
    for o0 in range(0, n_out, block_n):
        ro = o0 + tl.arange(0, block_n)
        mo = ro < n_out
        gv = load2d(g_ptr, rt, ro, sgt, sgo, mt, mo)
        uv = load2d(u_ptr, ro, rr, suo, sur, mo, mr)
        q = tl.dot(gv, uv, q, input_precision=prec)
    qs = q.to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + rt[:, None] * rank + rr[None, :], qs, mask=tr_mask)

    # gx chunk = scale * q @ down over the input axis.
    for i0 in range(0, n_in, block_k):
        ri = i0 + tl.arange(0, block_k)
        mi = ri < n_in
        dv = load2d(d_ptr, rr, ri, sdr, sdi, mr, mi)
        gxv = tl.dot(qs, dv, input_precision=prec) * scale
        tl.store(
            gx_ptr + rt[:, None] * sgxt + ri[None, :] * sgxi,
            gxv.to(gx_ptr.dtype.element_ty),
            mask=mt[:, None] & mi[None, :],
        )


def _plan_bypass_fwd(dev, t, o, i, r, e, has_base):
    """Bypass forward configs, model-ranked, then the alternatives."""
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH]
    space = grid_configs(
        dev,
        BLOCK_M=(16, 32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(32, 64, 128),
        SPLIT_O=(1, 2, 4, 8),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, bk, split = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["SPLIT_O"]
        tiles = cdiv(t, bm)
        out_bytes = t * o * e * (2 if has_base else 1)
        return Work(
            grid=tiles * split,
            warps=c["num_warps"],
            acc_fp32=bm * rp + bm * bn,
            operand_bytes=0,
            smem_stage=max((bm * bk + bk * rp) * e, rp * bn * e),
            stages=c["num_stages"],
            hbm_bytes=t * i * e * split
            + shared_reads(dev, (i + o) * rp * e, tiles * split)
            + out_bytes,
            mma_flops=2.0 * tiles * bm * rp * (i * split + o),
            # every program reads all of down and its 1/split of up
            l2_bytes=t * i * e * split
            + tiles * split * (i + cdiv(o, split)) * rp * e
            + out_bytes,
            # loads in flight: loop 1 an x tile per stage; loop 2 a base tile (an up tile without base)
            phases=(
                (t * i * e * split, staged(dev, bm * bk * e, c["num_stages"])),
                (out_bytes, (bm * bn if has_base else bn * rp) * e),
            ),
        )

    return [*rank(dev, space, work), TORCH, ATOMIC]


def _plan_bypass_bwd(dev, t, o, i, r, e):
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH, ATOMIC]
    space = grid_configs(
        dev,
        BLOCK_M=(16, 32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_K=(32, 64, 128),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, bk = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"]
        tiles = cdiv(t, bm)
        return Work(
            grid=tiles,
            warps=c["num_warps"],
            acc_fp32=2 * bm * rp + bm * bk,
            operand_bytes=0,
            smem_stage=max((bm * bk + bk * rp) * e, (bm * bn + bn * rp) * e),
            stages=c["num_stages"],
            hbm_bytes=(2 * t * i + t * o + 2 * t * r) * e
            + shared_reads(dev, (i + o) * rp * e, tiles),
            mma_flops=2.0 * tiles * bm * rp * (2 * i + o),
            # every token tile reads both factors whole
            l2_bytes=(2 * t * i + t * o + 2 * t * r) * e + tiles * (i + o) * rp * e,
        )

    return [*rank(dev, space, work), TORCH, ATOMIC]


def _torch_bypass(flat, u, d, scale, base2, out_dtype):
    """The torch composition: two GEMMs, the scale and base add in the second's
    epilogue when base is in the compute dtype."""
    h = flat @ d.t()
    if base2 is not None and base2.dtype == flat.dtype:
        return torch.addmm(base2, h, u.t(), alpha=scale).to(out_dtype)
    y = torch.addmm(h.new_empty(()), h, u.t(), beta=0, alpha=scale)
    return (y if base2 is None else base2 + y).to(out_dtype)


def _bypass_fwd_run(cfg, flat, u, d, scale, base2, out_dtype):
    if cfg == TORCH:
        return _torch_bypass(flat, u, d, scale, base2, out_dtype)
    if cfg == ATOMIC:
        y = bypass.lora_bypass_fwd(flat, u, d, scale)
        return (y if base2 is None else base2 + y).to(out_dtype)
    t, i = flat.shape
    o, r = u.shape
    out = torch.empty(t, o, device=flat.device, dtype=out_dtype)
    b = base2 if base2 is not None else out
    grid = (triton.cdiv(t, cfg["BLOCK_M"]), cfg["SPLIT_O"])
    _bypass_fwd_kernel[grid](
        flat,
        d,
        u,
        b,
        out,
        t,
        o,
        i,
        r,
        *flat.stride(),
        *d.stride(),
        *u.stride(),
        *b.stride(),
        *out.stride(),
        scale,
        has_base=base2 is not None,
        prec=dot_prec(flat.dtype),
        block_r=pad16(r),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_k=cfg["BLOCK_K"],
        split_o=cfg["SPLIT_O"],
        **launch(cfg),
    )
    return out


def _bypass_bwd_run(cfg, g2, flat, u, d, scale):
    """(gx, g_up, g_down) in the compute dtype, all operands already in it."""
    if cfg == TORCH:
        h = flat @ d.t()
        q = g2 @ u
        return q @ d * scale, g2.t() @ h * scale, q.t() @ flat * scale
    if cfg == ATOMIC:
        return bypass.lora_bypass_bwd(flat, u, d, g2, scale)
    t, i = flat.shape
    o, r = u.shape
    gx = torch.empty_like(flat)
    h = torch.empty(t, r, device=flat.device, dtype=flat.dtype)
    q = torch.empty_like(h)
    _bypass_bwd_kernel[(triton.cdiv(t, cfg["BLOCK_M"]),)](
        flat,
        d,
        u,
        g2,
        gx,
        h,
        q,
        t,
        o,
        i,
        r,
        *flat.stride(),
        *d.stride(),
        *u.stride(),
        *g2.stride(),
        *gx.stride(),
        scale,
        prec=dot_prec(flat.dtype),
        block_r=pad16(r),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_k=cfg["BLOCK_K"],
        **launch(cfg),
    )
    gu = torch.mm(g2.t(), h).mul_(scale)
    gd = torch.mm(q.t(), flat).mul_(scale)
    return gx, gu, gd


def lora_bypass_fwd(x, up, down, gamma: float = 1.0, base=None, out_dtype=None):
    """[base +] gamma * (x @ down^T) @ up^T; x (T, I) and the factors in one
    compute dtype, base (T, O) or None; the result in ``out_dtype`` (default
    base's dtype, else x's)."""
    dev = device_info(x.device)
    t, i = x.shape
    o, r = up.shape
    out_dtype = out_dtype or (base.dtype if base is not None else x.dtype)
    key = (
        "lora.bypass_fwd",
        (dtype_key(x.dtype), dtype_key(out_dtype), base is not None),
        (token_bucket(t), o, i, r),
    )

    def inputs():
        return copy_of(x), copy_of(base)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass_fwd(dev, t, o, i, r, x.element_size(), base is not None),
        measurer(
            dev,
            (t * (i + o * (2 if base is not None else 1))) * x.element_size(),
            inputs,
            lambda c, s: lambda: _bypass_fwd_run(
                c, s[0], up, down, gamma, s[1], out_dtype
            ),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _bypass_fwd_run(c, x, up, down, gamma, base, out_dtype),
        OutOfResources,
    )


def lora_bypass_bwd(x, up, down, grad, gamma: float = 1.0):
    """(gx, g_up, g_down) of the bypass in the compute dtype."""
    dev = device_info(x.device)
    t, i = x.shape
    o, r = up.shape
    key = (
        "lora.bypass_bwd",
        (dtype_key(x.dtype),),
        (token_bucket(t), o, i, r),
    )

    def inputs():
        return copy_of(grad), copy_of(x)

    cfg = select(
        dev,
        key,
        lambda: _plan_bypass_bwd(dev, t, o, i, r, x.element_size()),
        measurer(
            dev,
            2 * t * (i + o) * x.element_size(),
            inputs,
            lambda c, s: lambda: _bypass_bwd_run(c, s[0], s[1], up, down, gamma),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _bypass_bwd_run(c, grad, x, up, down, gamma),
        OutOfResources,
    )
