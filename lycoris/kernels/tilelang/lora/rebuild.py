"""LoRA merge in TileLang on the model-ranked tuner (fp16, sm_70+).

Forward: out = [w +] gamma * up @ down, one program per output tile, one GEMM
over the padded rank, the base add in the epilogue. Backward: a program owns a
row tile and 1/SPLIT of the columns and reads each grad tile once for both
gradients: g_up's rows accumulate over the program's columns, g_down is stored
per row tile; fp32 partials summed in torch. Volta's MMA path has no
transposed A operand, so up^T is staged instead. The role-split kernels
(``.merge`` / ``.bypass``) and the torch composition are timed alongside;
non-fp16 operands take them. See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer
from ...tuner.model import Work, rank, shared_reads, tiled_reads
from ..common import (
    CompileError,
    cdiv,
    compiled,
    fits,
    pad_to,
    padded_cols,
    padded_rows,
    partitions,
    tl_dtype,
)
from . import bypass, merge

F16 = "float16"
F32 = "float32"
MAX_RANK = 128
WHOLE = {"impl": "whole"}


def _merge_prim(o, i, rp, bm, bn, threads, base_dt, out_dt):
    """out = [w +] scale * up @ down; grid (column tiles bx, row tiles by)."""
    plain_only = base_dt is None
    base_dt = base_dt or out_dt

    @T.macro
    def product(up, down, bx, by, acc):
        u_s = T.alloc_shared((bm, rp), F16)
        d_s = T.alloc_shared((rp, bn), F16)
        T.clear(acc)
        T.copy(up[by * bm, 0], u_s)
        T.copy(down[0, bx * bn], d_s)
        T.gemm(u_s, d_s, acc)

    @T.prim_func
    def with_base(
        up: T.Tensor((o, rp), F16),
        down: T.Tensor((rp, i), F16),
        w: T.Tensor((o, i), base_dt),
        out: T.Tensor((o, i), out_dt),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            acc = T.alloc_fragment((bm, bn), F32)
            w_f = T.alloc_fragment((bm, bn), base_dt)
            product(up, down, bx, by, acc)
            T.copy(w[by * bm, bx * bn], w_f)
            for a, b in T.Parallel(bm, bn):
                acc[a, b] = w_f[a, b] + scale * acc[a, b]
            T.copy(acc, out[by * bm, bx * bn])

    @T.prim_func
    def plain(
        up: T.Tensor((o, rp), F16),
        down: T.Tensor((rp, i), F16),
        out: T.Tensor((o, i), out_dt),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            acc = T.alloc_fragment((bm, bn), F32)
            product(up, down, bx, by, acc)
            for a, b in T.Parallel(bm, bn):
                acc[a, b] = scale * acc[a, b]
            T.copy(acc, out[by * bm, bx * bn])

    return plain if plain_only else with_base


def _merge_bwd_prim(o, i, rp, bm, bn, threads, split):
    """fp32 partials: gu[split] = g @ down^T per row tile, gd[row tile] = up^T @ g; grid (splits, row tiles)."""
    n_m = cdiv(o, bm)
    per = cdiv(cdiv(i, bn), split)

    @T.prim_func
    def main(
        g: T.Tensor((o, i), F16),
        up_t: T.Tensor((rp, o), F16),
        down: T.Tensor((rp, i), F16),
        gu: T.Tensor((split, o, rp), F32),
        gd: T.Tensor((n_m, rp, i), F32),
    ):
        with T.Kernel(split, n_m, threads=threads) as (bs, bm_i):
            g_s = T.alloc_shared((bm, bn), F16)
            d_s = T.alloc_shared((rp, bn), F16)
            ut_s = T.alloc_shared((rp, bm), F16)
            gu_f = T.alloc_fragment((bm, rp), F32)
            gd_f = T.alloc_fragment((rp, bn), F32)

            # up^T of this row tile, held for every g tile; gu_f accumulates this split.
            T.clear(gu_f)
            T.copy(up_t[0, bm_i * bm], ut_s)

            # Each g tile of this split: gu += g @ down^T, gd tile = up^T @ g.
            for n in T.serial(per):
                if (bs * per + n) * bn < i:
                    T.copy(g[bm_i * bm, (bs * per + n) * bn], g_s)
                    T.copy(down[0, (bs * per + n) * bn], d_s)
                    T.gemm(g_s, d_s, gu_f, transpose_B=True)
                    T.clear(gd_f)
                    T.gemm(ut_s, g_s, gd_f)
                    T.copy(gd_f, gd[bm_i, 0, (bs * per + n) * bn])
            T.copy(gu_f, gu[bs, bm_i * bm, 0])

    return main


def _plan_merge(dev, o, i, r, e_w):
    rp = pad_to(r)
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th}
        for bm in (32, 64, 128)
        for bn in (64, 128, 256)
        for th in (128, 256)
    ]
    legal = [
        c
        for c in cfgs
        if fits(dev, (c["BM"] + c["BN"]) * rp)
        and partitions(dev, c["threads"], (c["BM"], c["BN"]))
    ]

    def work(c):
        bm, bn = c["BM"], c["BN"]
        tm, tn = cdiv(o, bm), cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["threads"] // 32,
            acc_fp32=bm * bn,
            operand_bytes=0,
            smem_stage=2 * (bm + bn) * rp,
            stages=1,
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, 2 * (o + i) * rp, max(tm, tn)),
            mma_flops=2.0 * tm * bm * tn * bn * rp,
            l2_bytes=2 * o * i * e_w + tiled_reads(o, i, rp, bm, bn, 2),
        )

    return [*rank(dev, legal, work, mma=True), TORCH, WHOLE]


def _merge_run(cfg, up, down, scale, base, out_dtype):
    if cfg == TORCH and base is not None and base.dtype == up.dtype == out_dtype:
        # cuBLAS with the scale as alpha and the base as the addend
        return torch.addmm(base, up, down, alpha=scale)
    if cfg == TORCH:
        delta = (up @ down) * scale
        return (delta if base is None else base + delta).to(out_dtype)
    if cfg == WHOLE:
        return merge.lora_merge_fwd(
            up, down, None, None, base, scale, "plain", out_dtype
        )
    o, r = up.shape
    i = down.shape[1]
    rp = pad_to(r)
    u, d = padded_cols(up, rp), padded_rows(down, rp)
    base_dt = None if base is None else tl_dtype(base.dtype)
    kernel = compiled(
        _merge_prim,
        up.device,
        o,
        i,
        rp,
        cfg["BM"],
        cfg["BN"],
        cfg["threads"],
        base_dt,
        tl_dtype(out_dtype),
    )
    out = torch.empty(o, i, device=up.device, dtype=out_dtype)
    if base is None:
        kernel(u, d, out, scale)
    else:
        kernel(u, d, base.contiguous(), out, scale)
    return out


def lora_merge(up, down, gamma: float = 1.0, base=None, out_dtype=None):
    """[base +] gamma * (up @ down) stored in ``out_dtype`` (default the
    factors'); base is (O, I) or None."""
    out_dtype = out_dtype or up.dtype
    if up.dtype != torch.float16:
        return merge.lora_merge_fwd(
            up, down, None, None, base, gamma, "plain", out_dtype
        )
    dev = device_info(up.device)
    o, r = up.shape
    i = down.shape[1]
    e_w = torch.finfo(out_dtype).bits // 8
    key = (
        "tl.lora.merge_fwd",
        (
            dtype_key(up.dtype),
            dtype_key(out_dtype),
            tl_dtype(base.dtype) if base is not None else None,
        ),
        (o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_merge(dev, o, i, r, e_w),
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
        CompileError,
    )


def _plan_merge_bwd(dev, o, i, r, e_g):
    rp = pad_to(r)
    if rp > MAX_RANK:
        return [TORCH, WHOLE]
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th, "SPLIT": s}
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for th in (128, 256)
        for s in (1, 2, 4, 8, 16)
    ]
    legal = [
        c
        for c in cfgs
        if fits(dev, c["BM"] * c["BN"] + rp * (c["BN"] + c["BM"]))
        and partitions(dev, c["threads"], (c["BM"], rp), (rp, c["BN"]))
    ]

    def work(c):
        bm, bn, split = c["BM"], c["BN"], c["SPLIT"]
        tm = cdiv(o, bm)
        # g read once; fp32 partials: g_up per split, g_down per row tile
        return Work(
            grid=tm * split,
            warps=c["threads"] // 32,
            acc_fp32=bm * rp + rp * bn,
            operand_bytes=0,
            smem_stage=2 * (bm * bn + rp * (bn + bm)),
            stages=1,
            hbm_bytes=o * i * e_g
            + 2 * (split * o + tm * i) * rp * 4
            + shared_reads(dev, 2 * (o + i) * rp, tm * split),
            mma_flops=4.0 * o * i * rp,
            # each (row tile, split) reads its slice of down and the tile's rows of up
            l2_bytes=o * i * e_g
            + 2 * (tm * i + tm * split * bm) * rp
            + 2 * (split * o + tm * i) * rp * 4,
        )

    return [*rank(dev, legal, work, mma=True), TORCH, WHOLE]


def _merge_bwd_run(cfg, g2, up, down, scale):
    """(g_up, g_down) from the (O, I) grad, in the factors' dtype."""
    if cfg == TORCH and up.dtype == g2.dtype:
        # cuBLAS with the scale as alpha
        gu = torch.addmm(up.new_empty(up.shape), g2, down.t(), beta=0, alpha=scale)
        gd = torch.addmm(down.new_empty(down.shape), up.t(), g2, beta=0, alpha=scale)
        return gu, gd
    if cfg == TORCH:
        g = (g2 * scale).to(up.dtype)
        return g @ down.t(), up.t() @ g
    if cfg == WHOLE:
        return bypass.lora_merge_bwd(g2, up, down, scale)
    o, r = up.shape
    i = down.shape[1]
    rp = pad_to(r)
    bm, bn = cfg["BM"], cfg["BN"]
    split = max(1, min(cfg["SPLIT"], cdiv(i, bn)))
    ut = padded_rows(up.t(), rp)
    d = padded_rows(down, rp)
    gu = torch.empty(split, o, rp, device=up.device, dtype=torch.float32)
    gd = torch.empty(cdiv(o, bm), rp, i, device=up.device, dtype=torch.float32)
    kernel = compiled(
        _merge_bwd_prim, up.device, o, i, rp, bm, bn, cfg["threads"], split
    )
    kernel(g2.contiguous(), ut, d, gu, gd)
    return (
        gu.sum(0)[:, :r].mul_(scale).to(up.dtype),
        gd.sum(0)[:r].mul_(scale).to(up.dtype),
    )


def lora_merge_bwd(grad, up, down, gamma: float = 1.0):
    """(g_up, g_down) of gamma * up @ down from the (O, I) grad, in the
    factors' dtype."""
    if up.dtype != torch.float16 or grad.dtype != torch.float16:
        return bypass.lora_merge_bwd(grad, up, down, gamma)
    dev = device_info(up.device)
    o, r = up.shape
    i = down.shape[1]
    key = (
        "tl.lora.merge_bwd",
        (dtype_key(grad.dtype),),
        (o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_merge_bwd(dev, o, i, r, grad.element_size()),
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
        CompileError,
    )
