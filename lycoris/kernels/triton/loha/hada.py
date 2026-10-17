"""LoHa merge DeltaW = [base +] gamma * (a1 @ b1) * (a2 @ b2) on the
model-ranked tuner, forward and backward (the bypass: ``.regen``).
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
    pad16,
)
from ...tuner.model import Work, cdiv, rank, shared_reads, tiled_reads
from ..common import OutOfResources, dot_prec, load2d

# Largest padded rank the regen bypass and the merge backward hold in registers.
MAX_RANK = 128


@triton.jit
def _merge_fwd_kernel(
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    w_ptr,
    out_ptr,
    n_out,
    n_in,
    rank,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
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
    """out = [w +] scale * (a1@b1) * (a2@b2), one program per output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rn = pid_n * block_n + tl.arange(0, block_n)
    mm = rm < n_out
    mn = rn < n_in

    # acc1 = a1 @ b1 tile, acc2 = a2 @ b2 tile, reduced over rank.
    acc1 = tl.zeros((block_m, block_n), tl.float32)
    acc2 = tl.zeros((block_m, block_n), tl.float32)
    for r0 in range(0, rank, block_r):
        rk = r0 + tl.arange(0, block_r)
        mk = rk < rank
        a1 = load2d(a1_ptr, rm, rk, sa1o, sa1r, mm, mk)
        b1 = load2d(b1_ptr, rk, rn, sb1r, sb1i, mk, mn)
        a2 = load2d(a2_ptr, rm, rk, sa2o, sa2r, mm, mk)
        b2 = load2d(b2_ptr, rk, rn, sb2r, sb2i, mk, mn)
        acc1 = tl.dot(a1, b1, acc1, input_precision=prec)
        acc2 = tl.dot(a2, b2, acc2, input_precision=prec)

    # Epilogue: hadamard, scale, add the weight tile, store.
    out = acc1 * acc2 * scale
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
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    ga1_ptr,
    gb1_ptr,
    ga2_ptr,
    gb2_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    scale,
    n_rows,
    n_cols,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """fp32 partials of the four gradients: row-side (first n_rows*S programs), then column-side."""
    pid = tl.program_id(0)
    splits = tl.num_programs(0) // (n_rows + n_cols)
    rr = tl.arange(0, block_r)
    mr = rr < rank
    ct = a1_ptr.dtype.element_ty

    # Row role: ga1, ga2 (block_m, rank) for one row tile over split `part` of the columns.
    if pid < n_rows * splits:
        part = pid // n_rows
        rm = (pid % n_rows) * block_m + tl.arange(0, block_m)
        mm = rm < n_out
        amask = mm[:, None] & mr[None, :]
        a1 = load2d(a1_ptr, rm, rr, sa1o, sa1r, mm, mr)
        a2 = load2d(a2_ptr, rm, rr, sa2o, sa2r, mm, mr)
        ga1 = tl.zeros((block_m, block_r), tl.float32)
        ga2 = tl.zeros((block_m, block_r), tl.float32)
        per = tl.cdiv(tl.cdiv(n_in, block_n), splits) * block_n
        lo = part * per
        hi = tl.minimum(lo + per, n_in)
        for i0 in range(lo, hi, block_n):
            rn = i0 + tl.arange(0, block_n)
            mn = rn < hi
            b1 = load2d(b1_ptr, rr, rn, sb1r, sb1i, mr, mn)
            b2 = load2d(b2_ptr, rr, rn, sb2r, sb2i, mr, mn)
            g = load2d(g_ptr, rm, rn, sgo, sgi, mm, mn).to(tl.float32) * scale

            # e1 = g * (a2 @ b2), e2 = g * (a1 @ b1) on this tile.
            e1 = (g * tl.dot(a2, b2, input_precision=prec)).to(ct)
            e2 = (g * tl.dot(a1, b1, input_precision=prec)).to(ct)

            # ga1 += e1 @ b1^T, ga2 += e2 @ b2^T.
            ga1 = tl.dot(e1, tl.trans(b1), ga1, input_precision=prec)
            ga2 = tl.dot(e2, tl.trans(b2), ga2, input_precision=prec)

        off = (part * n_out + rm)[:, None] * rank + rr[None, :]
        tl.store(ga1_ptr + off, ga1, mask=amask)
        tl.store(ga2_ptr + off, ga2, mask=amask)
    # Column role: gb1, gb2 (rank, block_n) for one column tile over split `part_c` of the rows.
    else:
        q = pid - n_rows * splits
        part_c = q // n_cols
        rn = (q % n_cols) * block_n + tl.arange(0, block_n)
        mn = rn < n_in
        bmask = mr[:, None] & mn[None, :]
        b1 = load2d(b1_ptr, rr, rn, sb1r, sb1i, mr, mn)
        b2 = load2d(b2_ptr, rr, rn, sb2r, sb2i, mr, mn)
        gb1 = tl.zeros((block_r, block_n), tl.float32)
        gb2 = tl.zeros((block_r, block_n), tl.float32)
        per_c = tl.cdiv(tl.cdiv(n_out, block_m), splits) * block_m
        lo_c = part_c * per_c
        hi_c = tl.minimum(lo_c + per_c, n_out)
        for o0 in range(lo_c, hi_c, block_m):
            rm = o0 + tl.arange(0, block_m)
            mm = rm < hi_c
            a1 = load2d(a1_ptr, rm, rr, sa1o, sa1r, mm, mr)
            a2 = load2d(a2_ptr, rm, rr, sa2o, sa2r, mm, mr)
            g = load2d(g_ptr, rm, rn, sgo, sgi, mm, mn).to(tl.float32) * scale

            # gb1 += a1^T @ (g * P2), gb2 += a2^T @ (g * P1).
            e1 = (g * tl.dot(a2, b2, input_precision=prec)).to(ct)
            e2 = (g * tl.dot(a1, b1, input_precision=prec)).to(ct)
            gb1 = tl.dot(tl.trans(a1), e1, gb1, input_precision=prec)
            gb2 = tl.dot(tl.trans(a2), e2, gb2, input_precision=prec)

        off_c = (part_c * rank + rr)[:, None] * n_in + rn[None, :]
        tl.store(gb1_ptr + off_c, gb1, mask=bmask)
        tl.store(gb2_ptr + off_c, gb2, mask=bmask)


@triton.jit
def _merge_bwd_tile_kernel(
    g_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    ga1_ptr,
    gb1_ptr,
    ga2_ptr,
    gb2_ptr,
    n_out,
    n_in,
    rank,
    sgo,
    sgi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    scale,
    prec: tl.constexpr,
    block_r: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """fp32 partials of the four gradients from one read of g: ga per split, gb per row tile.

    Grid: (output row tiles, splits of the input axis).
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    splits = tl.num_programs(1)
    rm = pid_m * block_m + tl.arange(0, block_m)
    rr = tl.arange(0, block_r)
    mm = rm < n_out
    mr = rr < rank
    ct = a1_ptr.dtype.element_ty

    # Up-factor rows of this row tile; ga1, ga2 accumulate this split's partials.
    a1 = load2d(a1_ptr, rm, rr, sa1o, sa1r, mm, mr)
    a2 = load2d(a2_ptr, rm, rr, sa2o, sa2r, mm, mr)
    ga1 = tl.zeros((block_m, block_r), tl.float32)
    ga2 = tl.zeros((block_m, block_r), tl.float32)

    # Input columns [lo, hi) of this split, one g tile at a time.
    per = tl.cdiv(tl.cdiv(n_in, block_n), splits) * block_n
    lo = pid_s * per
    hi = tl.minimum(lo + per, n_in)
    for i0 in range(lo, hi, block_n):
        rn = i0 + tl.arange(0, block_n)
        mn = rn < hi
        bmask = mr[:, None] & mn[None, :]
        b1 = load2d(b1_ptr, rr, rn, sb1r, sb1i, mr, mn)
        b2 = load2d(b2_ptr, rr, rn, sb2r, sb2i, mr, mn)
        g = load2d(g_ptr, rm, rn, sgo, sgi, mm, mn).to(tl.float32) * scale

        # e1 = g * P2, e2 = g * P1 for this tile, built once for all four gradients.
        e1 = (g * tl.dot(a2, b2, input_precision=prec)).to(ct)
        e2 = (g * tl.dot(a1, b1, input_precision=prec)).to(ct)

        # ga += e @ b^T; gb tile = a^T @ e, stored per row tile.
        ga1 = tl.dot(e1, tl.trans(b1), ga1, input_precision=prec)
        ga2 = tl.dot(e2, tl.trans(b2), ga2, input_precision=prec)
        off_b = (pid_m * rank + rr)[:, None] * n_in + rn[None, :]
        tl.store(
            gb1_ptr + off_b, tl.dot(tl.trans(a1), e1, input_precision=prec), mask=bmask
        )
        tl.store(
            gb2_ptr + off_b, tl.dot(tl.trans(a2), e2, input_precision=prec), mask=bmask
        )
    off_a = (pid_s * n_out + rm)[:, None] * rank + rr[None, :]
    amask = mm[:, None] & mr[None, :]
    tl.store(ga1_ptr + off_a, ga1, mask=amask)
    tl.store(ga2_ptr + off_a, ga2, mask=amask)


def _plan_merge(dev, o, i, r, e_w, e_f):
    rp = pad16(r)
    brs = tuple(sorted({min(rp, 16), min(rp, 32), min(rp, 64)}))
    space = grid_configs(
        dev,
        BLOCK_M=(32, 64, 128),
        BLOCK_N=(32, 64, 128),
        BLOCK_R=brs,
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, br = c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_R"]
        tm, tn = cdiv(o, bm), cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["num_warps"],
            acc_fp32=2 * bm * bn,
            operand_bytes=0,
            smem_stage=2 * (bm + bn) * br * e_f,
            stages=c["num_stages"],
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, 2 * (o + i) * rp * e_f, max(tm, tn)),
            mma_flops=4.0 * tm * bm * tn * bn * rp,
            vector_flops=3.0 * o * i,
            l2_bytes=2 * o * i * e_w + 2 * tiled_reads(o, i, rp, bm, bn, e_f),
        )

    return [*rank(dev, space, work), TORCH]


def _plan_merge_bwd(dev, o, i, r, e_g, e_f):
    rp = pad16(r)
    if rp > MAX_RANK:
        return [TORCH]
    # XPU Triton miscompiles "roles" below 8 warps when the rank tile is partly masked.
    roles_warps = (8,) if dev.kind == "xpu" and rp != r else (2, 4, 8)
    roles = grid_configs(
        dev,
        LAYOUT=("roles",),
        BLOCK_M=(32, 64),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=roles_warps,
    )
    tile = grid_configs(
        dev,
        LAYOUT=("tile",),
        BLOCK_M=(32, 64),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 2, 4, 8, 16),
        num_warps=(2, 4, 8),
    )

    def work(c):
        bm, bn, split = c["BLOCK_M"], c["BLOCK_N"], c["SPLIT"]
        weights = shared_reads(dev, 2 * (o + i) * rp * e_f, cdiv(o, bm) * split)
        tile_smem = (2 * bm * rp + 2 * rp * bn + 2 * bm * bn) * e_f
        if c["LAYOUT"] == "roles":
            return Work(
                grid=(cdiv(o, bm) + cdiv(i, bn)) * split,
                warps=c["num_warps"],
                acc_fp32=2 * (bm * rp + bm * bn),
                operand_bytes=0,
                smem_stage=tile_smem,
                stages=c["num_stages"],
                hbm_bytes=2 * o * i * e_g + 2 * 2 * split * (o + i) * rp * 4 + weights,
                mma_flops=16.0 * o * i * rp,
                vector_flops=6.0 * o * i,
                # row roles read both ups' rows and both downs' split; column roles the transpose
                l2_bytes=2 * o * i * e_g
                + 2
                * (o * split + cdiv(o, bm) * i + i * split + cdiv(i, bn) * o)
                * rp
                * e_f
                + 2 * 2 * split * (o + i) * rp * 4,
            )
        tm = cdiv(o, bm)
        return Work(
            grid=tm * split,
            warps=c["num_warps"],
            acc_fp32=2 * (bm * rp + bm * bn + rp * bn),
            operand_bytes=0,
            smem_stage=tile_smem,
            stages=c["num_stages"],
            hbm_bytes=o * i * e_g + 2 * 2 * (split * o + tm * i) * rp * 4 + weights,
            mma_flops=12.0 * o * i * rp,
            vector_flops=3.0 * o * i,
            # each (row tile, split) reads both ups' rows and both downs' split
            l2_bytes=o * i * e_g
            + 2 * (tm * split * bm + tm * i) * rp * e_f
            + 2 * 2 * (split * o + tm * i) * rp * 4,
        )

    return [*rank(dev, roles + tile, work), TORCH]


def _merge_run(cfg, factors, scale, base, out_dtype):
    a1, b1, a2, b2 = factors
    o, r = a1.shape
    i = b1.shape[1]
    if cfg == TORCH:
        delta = (a1 @ b1) * (a2 @ b2) * scale
        return (delta if base is None else base + delta).to(out_dtype)
    out = torch.empty(o, i, device=a1.device, dtype=out_dtype)
    w = base if base is not None else out
    grid = (triton.cdiv(o, cfg["BLOCK_M"]), triton.cdiv(i, cfg["BLOCK_N"]))
    _merge_fwd_kernel[grid](
        a1,
        b1,
        a2,
        b2,
        w,
        out,
        o,
        i,
        r,
        *a1.stride(),
        *b1.stride(),
        *a2.stride(),
        *b2.stride(),
        *w.stride(),
        *out.stride(),
        scale,
        add_base=base is not None,
        prec=dot_prec(a1.dtype),
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        block_r=cfg["BLOCK_R"],
        **launch(cfg),
    )
    return out


def loha_merge(a1, b1, a2, b2, gamma: float = 1.0, base=None, out_dtype=None):
    """[base +] gamma * (a1 @ b1) * (a2 @ b2) in ``out_dtype`` (default the
    factors'); the factors share one compute dtype, base is (O, I) or None."""
    dev = device_info(a1.device)
    o, r = a1.shape
    i = b1.shape[1]
    out_dtype = out_dtype or a1.dtype
    e_w = torch.finfo(out_dtype).bits // 8
    factors = (a1, b1, a2, b2)
    key = (
        "loha.merge_fwd",
        (dtype_key(a1.dtype), dtype_key(out_dtype), base is not None),
        (o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_merge(dev, o, i, r, e_w, a1.element_size()),
        measurer(
            dev,
            2 * o * i * e_w,
            lambda: (copy_of(base),),
            lambda c, s: lambda: _merge_run(c, factors, gamma, s[0], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _merge_run(c, factors, gamma, base, out_dtype),
        OutOfResources,
    )


def _merge_bwd_run(cfg, g2, factors, scale):
    a1, b1, a2, b2 = factors
    if cfg == TORCH:
        gw = g2.to(a1.dtype) * scale
        e1 = gw * (a2 @ b2)
        e2 = gw * (a1 @ b1)
        return e1 @ b1.t(), a1.t() @ e1, e2 @ b2.t(), a2.t() @ e2
    o, r = a1.shape
    i = b1.shape[1]
    bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
    prec = dot_prec(a1.dtype)
    n_rows = triton.cdiv(o, bm)
    strides = (*g2.stride(), *a1.stride(), *b1.stride(), *a2.stride(), *b2.stride())
    dev_ = a1.device
    if cfg["LAYOUT"] == "tile":
        split = max(1, min(cfg["SPLIT"], triton.cdiv(i, bn)))
        ga = [
            torch.empty(split, o, r, device=dev_, dtype=torch.float32) for _ in range(2)
        ]
        gb = [
            torch.empty(n_rows, r, i, device=dev_, dtype=torch.float32)
            for _ in range(2)
        ]
        _merge_bwd_tile_kernel[(n_rows, split)](
            g2,
            a1,
            b1,
            a2,
            b2,
            ga[0],
            gb[0],
            ga[1],
            gb[1],
            o,
            i,
            r,
            *strides,
            scale,
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_n=bn,
            **launch(cfg),
        )
    else:
        n_cols = triton.cdiv(i, bn)
        split = max(1, min(cfg["SPLIT"], triton.cdiv(o, bm), triton.cdiv(i, bn)))
        ga = [
            torch.empty(split, o, r, device=dev_, dtype=torch.float32) for _ in range(2)
        ]
        gb = [
            torch.empty(split, r, i, device=dev_, dtype=torch.float32) for _ in range(2)
        ]
        _merge_bwd_kernel[((n_rows + n_cols) * split,)](
            g2,
            a1,
            b1,
            a2,
            b2,
            ga[0],
            gb[0],
            ga[1],
            gb[1],
            o,
            i,
            r,
            *strides,
            scale,
            n_rows,
            n_cols,
            prec=prec,
            block_r=pad16(r),
            block_m=bm,
            block_n=bn,
            **launch(cfg),
        )
    return tuple(p.sum(0).to(a1.dtype) for p in (ga[0], gb[0], ga[1], gb[1]))


def loha_merge_bwd(grad, a1, b1, a2, b2, gamma: float = 1.0):
    """(g_a1, g_b1, g_a2, g_b2) of gamma * (a1 @ b1) * (a2 @ b2) from the (O, I)
    grad, in the factors' dtype."""
    dev = device_info(a1.device)
    o, r = a1.shape
    i = b1.shape[1]
    factors = (a1, b1, a2, b2)
    key = (
        "loha.merge_bwd",
        (dtype_key(grad.dtype), dtype_key(a1.dtype)),
        (o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_merge_bwd(dev, o, i, r, grad.element_size(), a1.element_size()),
        measurer(
            dev,
            o * i * grad.element_size(),
            lambda: (copy_of(grad),),
            lambda c, s: lambda: _merge_bwd_run(c, s[0], factors, gamma),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _merge_bwd_run(c, grad, factors, gamma),
        OutOfResources,
    )
