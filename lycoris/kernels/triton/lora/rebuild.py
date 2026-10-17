"""LoRA merge out = [w +] gamma * up @ down on the model-ranked tuner, forward
and backward. See docs/kernels/designs.md.
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
)
from ...tuner.model import Work, cdiv, rank, shared_reads, tiled_reads
from ..common import OutOfResources, dot_prec, load2d
from . import bypass
from .stream import MAX_RANK

# The role-split backward holding each whole reduction in registers.
WHOLE = {"impl": "whole"}


@triton.jit
def _merge_fwd_kernel(
    u_ptr,
    d_ptr,
    w_ptr,
    out_ptr,
    n_out,
    n_in,
    rank,
    suo,
    sur,
    sdr,
    sdi,
    swo,
    swi,
    soo,
    soi,
    scale,
    add_base: tl.constexpr,
    prec: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_r: tl.constexpr,
):
    """out = [w +] scale * (up @ down), one program per output tile.

    Grid: (output row tiles, input column tiles).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rn = pid_n * block_n + tl.arange(0, block_n)
    mm = rm < n_out
    mn = rn < n_in

    # acc (block_m, block_n) = up tile @ down tile, reduced over rank.
    acc = tl.zeros((block_m, block_n), tl.float32)
    for r0 in range(0, rank, block_r):
        rk = r0 + tl.arange(0, block_r)
        mk = rk < rank
        uv = load2d(u_ptr, rm, rk, suo, sur, mm, mk)
        dv = load2d(d_ptr, rk, rn, sdr, sdi, mk, mn)
        acc = tl.dot(uv, dv, acc, input_precision=prec)

    # Epilogue: scale, add the weight tile, store.
    out = acc * scale
    if add_base:
        out += load2d(w_ptr, rm, rn, swo, swi, mm, mn).to(tl.float32)
    tl.store(
        out_ptr + rm[:, None] * soo + rn[None, :] * soi,
        out.to(out_ptr.dtype.element_ty),
        mask=mm[:, None] & mn[None, :],
    )


@triton.jit
def _merge_bwd_kernel(
    g_ptr,
    u_ptr,
    d_ptr,
    gu_ptr,
    gd_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    suo,
    sur,
    sdr,
    sdi,
    n_up,
    n_down,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """fp32 partials of g_up (first n_up*S programs) and g_down (the rest), slab s per split s."""
    pid = tl.program_id(0)
    splits = tl.num_programs(0) // (n_up + n_down)
    rr = tl.arange(0, block_r)
    mr = rr < rank
    ct = u_ptr.dtype.element_ty

    # g_up role: rows of g_up (block_m, rank) = g @ down^T over split `part` of the input axis.
    if pid < n_up * splits:
        part = pid // n_up
        rm = (pid % n_up) * block_m + tl.arange(0, block_m)
        mm = rm < n_out
        per = tl.cdiv(tl.cdiv(n_in, block_k), splits) * block_k
        lo = part * per
        hi = tl.minimum(lo + per, n_in)
        acc = tl.zeros((block_m, block_r), tl.float32)
        for i0 in range(lo, hi, block_k):
            ri = i0 + tl.arange(0, block_k)
            mi = ri < hi
            gv = load2d(g_ptr, rm, ri, sgo, sgi, mm, mi).to(ct)
            dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)
            acc = tl.dot(gv, dv, acc, input_precision=prec)
        tl.store(
            gu_ptr + (part * n_out + rm)[:, None] * rank + rr[None, :],
            acc,
            mask=mm[:, None] & mr[None, :],
        )
    # g_down role: columns of g_down (rank, block_m) = up^T @ g over split `part_d` of the output axis.
    else:
        q = pid - n_up * splits
        part_d = q // n_down
        rn = (q % n_down) * block_m + tl.arange(0, block_m)
        mn = rn < n_in
        per_d = tl.cdiv(tl.cdiv(n_out, block_k), splits) * block_k
        lo_d = part_d * per_d
        hi_d = tl.minimum(lo_d + per_d, n_out)
        acc_d = tl.zeros((block_r, block_m), tl.float32)
        for o0 in range(lo_d, hi_d, block_k):
            ro = o0 + tl.arange(0, block_k)
            mo = ro < hi_d
            gt = load2d(g_ptr, ro, rn, sgo, sgi, mo, mn).to(ct)
            ut = load2d(u_ptr, rr, ro, sur, suo, mr, mo)
            acc_d = tl.dot(ut, gt, acc_d, input_precision=prec)
        tl.store(
            gd_ptr + (part_d * rank + rr)[:, None] * n_in + rn[None, :],
            acc_d,
            mask=mr[:, None] & mn[None, :],
        )


@triton.jit
def _merge_bwd_tile_kernel(
    g_ptr,
    u_ptr,
    d_ptr,
    gu_ptr,
    gd_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    suo,
    sur,
    sdr,
    sdi,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """fp32 partials of g_up (slab per split) and g_down (slab per row tile) from one read of g.

    Grid: (output row tiles, splits of the input axis).
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    splits = tl.num_programs(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mm = rm < n_out
    mr = rr < rank
    ct = u_ptr.dtype.element_ty

    # up^T rows of this row tile, held for every g tile; acc = this split's g_up partial.
    ut = load2d(u_ptr, rr, rm, sur, suo, mr, mm)
    acc = tl.zeros((block_m, block_r), tl.float32)

    # Input columns [lo, hi) of this split, one g tile at a time.
    per = tl.cdiv(tl.cdiv(n_in, block_n), splits) * block_n
    lo = pid_s * per
    hi = tl.minimum(lo + per, n_in)
    for i0 in range(lo, hi, block_n):
        ri = i0 + tl.arange(0, block_n)
        mi = ri < hi
        gv = load2d(g_ptr, rm, ri, sgo, sgi, mm, mi).to(ct)
        dv = load2d(d_ptr, ri, rr, sdi, sdr, mi, mr)

        # g_up += g @ down^T; g_down tile = up^T @ g, stored per row tile.
        acc = tl.dot(gv, dv, acc, input_precision=prec)
        gd = tl.dot(ut, gv, input_precision=prec)
        tl.store(
            gd_ptr + (pid_m * rank + rr)[:, None] * n_in + ri[None, :],
            gd,
            mask=mr[:, None] & mi[None, :],
        )
    tl.store(
        gu_ptr + (pid_s * n_out + rm)[:, None] * rank + rr[None, :],
        acc,
        mask=mm[:, None] & mr[None, :],
    )


def _plan_merge(dev, o, i, r, e_w, e_f):
    rp = pad16(r)
    brs = tuple(sorted({min(rp, 16), min(rp, 32), min(rp, 64)}))
    space = grid_configs(
        dev,
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128, 256),
        BLOCK_R=brs,
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, br = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_R"]
        tm, tn = cdiv(o, bm), cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["num_warps"],
            acc_fp32=bm * bn,
            operand_bytes=0,
            smem_stage=(bm + bn) * br * e_f,
            stages=c["num_stages"],
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, (o + i) * rp * e_f, max(tm, tn)),
            mma_flops=2.0 * tm * bm * tn * bn * rp,
            l2_bytes=2 * o * i * e_w + tiled_reads(o, i, rp, bm, bn, e_f),
        )

    return [*rank(dev, space, work), TORCH]


def _plan_merge_bwd(dev, o, i, r, e_g, e_f):
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH]
    roles = grid_configs(
        dev,
        LAYOUT=("roles",),
        BLOCK_M=(32, 64, 128),
        BLOCK_K=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=(2, 4, 8),
    )
    tile = grid_configs(
        dev,
        LAYOUT=("tile",),
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, split = c["BLOCK_M"], c["SPLIT"]
        weights = shared_reads(dev, (o + i) * rp * e_f, cdiv(o, bm) * split)
        if c["LAYOUT"] == "roles":
            bk = c["BLOCK_K"]
            return Work(
                grid=(cdiv(o, bm) + cdiv(i, bm)) * split,
                warps=c["num_warps"],
                acc_fp32=bm * rp,
                operand_bytes=0,
                smem_stage=(bm * bk + bk * rp) * e_f,
                stages=c["num_stages"],
                hbm_bytes=2 * o * i * e_g + 2 * split * (o + i) * rp * 4 + weights,
                mma_flops=4.0 * o * i * rp,
                # g_up programs read down's columns, g_down programs up's rows
                l2_bytes=2 * o * i * e_g
                + (cdiv(o, bm) * i + cdiv(i, bm) * o) * rp * e_f
                + 2 * split * (o + i) * rp * 4,
            )
        bn = c["BLOCK_N"]
        tm = cdiv(o, bm)
        return Work(
            grid=tm * split,
            warps=c["num_warps"],
            acc_fp32=bm * rp + rp * bn,
            operand_bytes=0,
            smem_stage=(bm * bn + bn * rp + rp * bm) * e_f,
            stages=c["num_stages"],
            hbm_bytes=o * i * e_g + 2 * (split * o + tm * i) * rp * 4 + weights,
            mma_flops=4.0 * o * i * rp,
            # each (row tile, split) reads its slice of down and the tile's rows of up
            l2_bytes=o * i * e_g
            + (tm * i + tm * split * bm) * rp * e_f
            + 2 * (split * o + tm * i) * rp * 4,
        )

    return [*rank(dev, roles + tile, work), TORCH, WHOLE]


def _merge_run(cfg, u, d, scale, base, out_dtype):
    o, r = u.shape
    i = d.shape[1]
    if cfg == TORCH:
        if base is not None and base.dtype == u.dtype == out_dtype:
            return torch.addmm(base, u, d, alpha=scale)
        delta = (u @ d) * scale
        return (delta if base is None else base + delta).to(out_dtype)
    out = torch.empty(o, i, device=u.device, dtype=out_dtype)
    w = base if base is not None else out
    grid = (triton.cdiv(o, cfg["BLOCK_M"]), triton.cdiv(i, cfg["BLOCK_N"]))
    _merge_fwd_kernel[grid](
        u,
        d,
        w,
        out,
        o,
        i,
        r,
        *u.stride(),
        *d.stride(),
        *w.stride(),
        *out.stride(),
        scale,
        add_base=base is not None,
        prec=dot_prec(u.dtype),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_r=cfg["BLOCK_R"],
        **launch(cfg),
    )
    return out


def lora_merge(up, down, gamma: float = 1.0, base=None, out_dtype=None):
    """[base +] gamma * (up @ down) stored in ``out_dtype`` (default the
    factors'); base is (O, I) or None."""
    dev = device_info(up.device)
    o, r = up.shape
    i = down.shape[1]
    out_dtype = out_dtype or up.dtype
    e_w = torch.finfo(out_dtype).bits // 8
    key = (
        "lora.merge_fwd",
        (dtype_key(up.dtype), dtype_key(out_dtype), base is not None),
        (o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_merge(dev, o, i, r, e_w, up.element_size()),
        measurer(
            dev,
            2 * o * i * e_w,
            lambda: (copy_of(base),),
            lambda c, s: lambda: _merge_run(c, up, down, gamma, s[0], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _merge_run(c, up, down, gamma, base, out_dtype),
        OutOfResources,
    )


def _merge_bwd_run(cfg, g2, u, d, scale):
    """(g_up, g_down) from the (O, I) grad, in the factors' dtype."""
    if cfg == TORCH:
        g = (g2 * scale).to(u.dtype)
        return g @ d.t(), u.t() @ g
    if cfg == WHOLE:
        return bypass.lora_merge_bwd(g2, u, d, scale)
    o, r = u.shape
    i = d.shape[1]
    prec = dot_prec(u.dtype)
    bm = cfg["BLOCK_M"]
    if cfg["LAYOUT"] == "tile":
        n_m = triton.cdiv(o, bm)
        split = max(1, min(cfg["SPLIT"], triton.cdiv(i, cfg["BLOCK_N"])))
        gu = torch.empty(split, o, r, device=u.device, dtype=torch.float32)
        gd = torch.empty(n_m, r, i, device=u.device, dtype=torch.float32)
        _merge_bwd_tile_kernel[(n_m, split)](
            g2,
            u,
            d,
            gu,
            gd,
            o,
            i,
            r,
            *g2.stride(),
            *u.stride(),
            *d.stride(),
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_n=cfg["BLOCK_N"],
            **launch(cfg),
        )
    else:
        n_up, n_down = triton.cdiv(o, bm), triton.cdiv(i, bm)
        split = max(1, min(cfg["SPLIT"], triton.cdiv(min(o, i), cfg["BLOCK_K"])))
        gu = torch.empty(split, o, r, device=u.device, dtype=torch.float32)
        gd = torch.empty(split, r, i, device=u.device, dtype=torch.float32)
        _merge_bwd_kernel[((n_up + n_down) * split,)](
            g2,
            u,
            d,
            gu,
            gd,
            o,
            i,
            r,
            *g2.stride(),
            *u.stride(),
            *d.stride(),
            n_up,
            n_down,
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_k=cfg["BLOCK_K"],
            **launch(cfg),
        )
    return gu.sum(0).mul_(scale).to(u.dtype), gd.sum(0).mul_(scale).to(u.dtype)


def lora_merge_bwd(grad, up, down, gamma: float = 1.0):
    """(g_up, g_down) of gamma * up @ down from the (O, I) grad, in the
    factors' dtype."""
    dev = device_info(up.device)
    o, r = up.shape
    i = down.shape[1]
    key = (
        "lora.merge_bwd",
        (dtype_key(grad.dtype), dtype_key(up.dtype)),
        (o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_merge_bwd(dev, o, i, r, grad.element_size(), up.element_size()),
        measurer(
            dev,
            o * i * grad.element_size(),
            lambda: (copy_of(grad),),
            lambda c, s: lambda: _merge_bwd_run(c, s[0], up, down, gamma),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _merge_bwd_run(c, grad, up, down, gamma),
        OutOfResources,
    )
