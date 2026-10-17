"""LoHa in TileLang on the model-ranked tuner (fp16, sm_70+): DeltaW = (a1 @ b1) * (a2 @ b2).

merge: one program per output tile builds both rank products into fp32
fragments and folds the hadamard, the scale and the base add into the
epilogue. merge backward ("tile"): a program owns a row tile and 1/SPLIT of
the columns; each grad tile builds both products once and feeds all four
gradients; fp32 partials summed in torch, the up factors staged transposed
(Volta's MMA has no transposed A operand). bypass: materialize gamma * DeltaW
with the merge kernel and run one cuBLAS GEMM with the base as its addend;
backward gx = g @ DeltaW, g_W = g^T x, then the merge backward. The
single-launch kernels (``.bypass`` / ``..lora.merge``) and the torch
composition are timed alongside; non-fp16 operands take them.
See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...limits import MAX_HADA_RANK
from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer, token_bucket
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
from ..lora import merge
from . import bypass

F16 = "float16"
F32 = "float32"
MAX_RANK = 128
WHOLE = {"impl": "whole"}
MATERIALIZE = {"impl": "materialize"}


def _merge_prim(o, i, rp, bm, bn, threads, base_dt, out_dt):
    """out = [w +] scale * (a1 @ b1) * (a2 @ b2); grid (column tiles bx, row tiles by)."""
    plain_only = base_dt is None
    base_dt = base_dt or out_dt

    @T.macro
    def products(a1, b1, a2, b2, bx, by, acc1, acc2):
        a1_s = T.alloc_shared((bm, rp), F16)
        b1_s = T.alloc_shared((rp, bn), F16)
        a2_s = T.alloc_shared((bm, rp), F16)
        b2_s = T.alloc_shared((rp, bn), F16)
        T.copy(a1[by * bm, 0], a1_s)
        T.copy(b1[0, bx * bn], b1_s)
        T.copy(a2[by * bm, 0], a2_s)
        T.copy(b2[0, bx * bn], b2_s)
        T.clear(acc1)
        T.clear(acc2)
        T.gemm(a1_s, b1_s, acc1)
        T.gemm(a2_s, b2_s, acc2)

    @T.prim_func
    def with_base(
        a1: T.Tensor((o, rp), F16),
        b1: T.Tensor((rp, i), F16),
        a2: T.Tensor((o, rp), F16),
        b2: T.Tensor((rp, i), F16),
        w: T.Tensor((o, i), base_dt),
        out: T.Tensor((o, i), out_dt),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            acc1 = T.alloc_fragment((bm, bn), F32)
            acc2 = T.alloc_fragment((bm, bn), F32)
            w_f = T.alloc_fragment((bm, bn), base_dt)
            products(a1, b1, a2, b2, bx, by, acc1, acc2)
            T.copy(w[by * bm, bx * bn], w_f)
            for p, q in T.Parallel(bm, bn):
                acc1[p, q] = w_f[p, q] + scale * acc1[p, q] * acc2[p, q]
            T.copy(acc1, out[by * bm, bx * bn])

    @T.prim_func
    def plain(
        a1: T.Tensor((o, rp), F16),
        b1: T.Tensor((rp, i), F16),
        a2: T.Tensor((o, rp), F16),
        b2: T.Tensor((rp, i), F16),
        out: T.Tensor((o, i), out_dt),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(i, bn), T.ceildiv(o, bm), threads=threads) as (bx, by):
            acc1 = T.alloc_fragment((bm, bn), F32)
            acc2 = T.alloc_fragment((bm, bn), F32)
            products(a1, b1, a2, b2, bx, by, acc1, acc2)
            for p, q in T.Parallel(bm, bn):
                acc1[p, q] = scale * acc1[p, q] * acc2[p, q]
            T.copy(acc1, out[by * bm, bx * bn])

    return plain if plain_only else with_base


def _merge_bwd_prim(o, i, rp, bm, bn, threads, split):
    """fp32 partials of the four gradients; grid (splits, row tiles), each g tile read once."""
    n_m = cdiv(o, bm)
    per = cdiv(cdiv(i, bn), split)

    @T.prim_func
    def main(
        g: T.Tensor((o, i), F16),
        a1: T.Tensor((o, rp), F16),
        a1t: T.Tensor((rp, o), F16),
        b1: T.Tensor((rp, i), F16),
        a2: T.Tensor((o, rp), F16),
        a2t: T.Tensor((rp, o), F16),
        b2: T.Tensor((rp, i), F16),
        ga1: T.Tensor((split, o, rp), F32),
        gb1: T.Tensor((n_m, rp, i), F32),
        ga2: T.Tensor((split, o, rp), F32),
        gb2: T.Tensor((n_m, rp, i), F32),
        scale: T.float32,
    ):
        with T.Kernel(split, n_m, threads=threads) as (bs, bmi):
            a1_s = T.alloc_shared((bm, rp), F16)
            a2_s = T.alloc_shared((bm, rp), F16)
            a1t_s = T.alloc_shared((rp, bm), F16)
            a2t_s = T.alloc_shared((rp, bm), F16)
            b1_s = T.alloc_shared((rp, bn), F16)
            b2_s = T.alloc_shared((rp, bn), F16)
            g_f = T.alloc_fragment((bm, bn), F16)
            p1 = T.alloc_fragment((bm, bn), F32)
            p2 = T.alloc_fragment((bm, bn), F32)
            e1_s = T.alloc_shared((bm, bn), F16)
            e2_s = T.alloc_shared((bm, bn), F16)
            ga1_f = T.alloc_fragment((bm, rp), F32)
            ga2_f = T.alloc_fragment((bm, rp), F32)
            gb_f = T.alloc_fragment((rp, bn), F32)

            # Up factors of this row tile (and their transposes), held for every column tile.
            T.copy(a1[bmi * bm, 0], a1_s)
            T.copy(a2[bmi * bm, 0], a2_s)
            T.copy(a1t[0, bmi * bm], a1t_s)
            T.copy(a2t[0, bmi * bm], a2t_s)
            T.clear(ga1_f)
            T.clear(ga2_f)

            # Each column tile of this split.
            for n in T.serial(per):
                if (bs * per + n) * bn < i:
                    col = (bs * per + n) * bn
                    T.copy(b1[0, col], b1_s)
                    T.copy(b2[0, col], b2_s)
                    T.copy(g[bmi * bm, col], g_f)

                    # p1, p2 = the two rank products; e1 = scale*g*p2, e2 = scale*g*p1.
                    T.clear(p1)
                    T.clear(p2)
                    T.gemm(a1_s, b1_s, p1)
                    T.gemm(a2_s, b2_s, p2)
                    for p, q in T.Parallel(bm, bn):
                        e1_s[p, q] = scale * g_f[p, q] * p2[p, q]
                        e2_s[p, q] = scale * g_f[p, q] * p1[p, q]

                    # ga += e @ b^T; gb tile = a^T @ e, stored per row tile.
                    T.gemm(e1_s, b1_s, ga1_f, transpose_B=True)
                    T.gemm(e2_s, b2_s, ga2_f, transpose_B=True)
                    T.clear(gb_f)
                    T.gemm(a1t_s, e1_s, gb_f)
                    T.copy(gb_f, gb1[bmi, 0, col])
                    T.clear(gb_f)
                    T.gemm(a2t_s, e2_s, gb_f)
                    T.copy(gb_f, gb2[bmi, 0, col])
            T.copy(ga1_f, ga1[bs, bmi * bm, 0])
            T.copy(ga2_f, ga2[bs, bmi * bm, 0])

    return main


def _padded(factors, rp):
    a1, b1, a2, b2 = factors
    return (
        padded_cols(a1, rp),
        padded_rows(b1, rp),
        padded_cols(a2, rp),
        padded_rows(b2, rp),
    )


def _plan_merge(dev, o, i, r, e_w):
    rp = pad_to(r)
    if rp > MAX_RANK:
        return [TORCH, WHOLE]
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th}
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for th in (128, 256)
    ]
    legal = [
        c
        for c in cfgs
        if fits(dev, 2 * c["BM"] * rp + 2 * rp * c["BN"])
        and partitions(dev, c["threads"], (c["BM"], c["BN"]))
    ]

    def work(c):
        bm, bn = c["BM"], c["BN"]
        tm, tn = cdiv(o, bm), cdiv(i, bn)
        return Work(
            grid=tm * tn,
            warps=c["threads"] // 32,
            acc_fp32=2 * bm * bn,
            operand_bytes=0,
            smem_stage=2 * (2 * bm * rp + 2 * rp * bn),
            stages=1,
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, 4 * (o + i) * rp, max(tm, tn)),
            mma_flops=4.0 * tm * bm * tn * bn * rp,
            vector_flops=3.0 * o * i,
            l2_bytes=2 * o * i * e_w + 2 * tiled_reads(o, i, rp, bm, bn, 2),
        )

    return [*rank(dev, legal, work, mma=True), TORCH, WHOLE]


def _merge_run(cfg, factors, scale, base, out_dtype):
    if cfg == TORCH:
        a1, b1, a2, b2 = factors
        delta = (a1 @ b1) * (a2 @ b2) * scale
        return (delta if base is None else base + delta).to(out_dtype)
    if cfg == WHOLE:
        return merge.lora_merge_fwd(*factors, base, scale, "hada", out_dtype)
    o, r = factors[0].shape
    i = factors[1].shape[1]
    rp = pad_to(r)
    base_dt = None if base is None else tl_dtype(base.dtype)
    kernel = compiled(
        _merge_prim,
        factors[0].device,
        o,
        i,
        rp,
        cfg["BM"],
        cfg["BN"],
        cfg["threads"],
        base_dt,
        tl_dtype(out_dtype),
    )
    out = torch.empty(o, i, device=factors[0].device, dtype=out_dtype)
    if base is None:
        kernel(*_padded(factors, rp), out, scale)
    else:
        kernel(*_padded(factors, rp), base.contiguous(), out, scale)
    return out


def loha_merge(a1, b1, a2, b2, gamma: float = 1.0, base=None, out_dtype=None):
    """[base +] gamma * (a1 @ b1) * (a2 @ b2) in ``out_dtype`` (default the
    factors'); the factors share one compute dtype, base is (O, I) or None."""
    out_dtype = out_dtype or a1.dtype
    factors = (a1, b1, a2, b2)
    if a1.dtype != torch.float16:
        return merge.lora_merge_fwd(*factors, base, gamma, "hada", out_dtype)
    dev = device_info(a1.device)
    o, r = a1.shape
    i = b1.shape[1]
    e_w = torch.finfo(out_dtype).bits // 8
    key = (
        "tl.loha.merge_fwd",
        (dtype_key(out_dtype), tl_dtype(base.dtype) if base is not None else None),
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
            lambda c, s: lambda: _merge_run(c, factors, gamma, s[0], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _merge_run(c, factors, gamma, base, out_dtype),
        CompileError,
    )


def _plan_merge_bwd(dev, o, i, r, e_g):
    rp = pad_to(r)
    if rp > MAX_RANK:
        return [TORCH, WHOLE]
    cfgs = [
        {"BM": bm, "BN": bn, "threads": th, "SPLIT": s}
        for bm in (32, 64)
        for bn in (32, 64, 128)
        for th in (128, 256)
        for s in (1, 2, 4, 8, 16)
    ]
    legal = [
        c
        for c in cfgs
        if fits(dev, 4 * c["BM"] * rp + 2 * rp * c["BN"] + 2 * c["BM"] * c["BN"])
        and partitions(
            dev, c["threads"], (c["BM"], c["BN"]), (c["BM"], rp), (rp, c["BN"])
        )
    ]

    def work(c):
        bm, bn, split = c["BM"], c["BN"], c["SPLIT"]
        tm = cdiv(o, bm)
        # products built once per grad tile; fp32 partials row-side per split, column-side per row tile
        return Work(
            grid=tm * split,
            warps=c["threads"] // 32,
            acc_fp32=2 * (bm * bn + bm * rp + rp * bn),
            operand_bytes=0,
            smem_stage=2 * (4 * bm * rp + 2 * rp * bn + 2 * bm * bn),
            stages=1,
            hbm_bytes=o * i * e_g
            + 4 * (split * o + tm * i) * rp * 4
            + shared_reads(dev, 4 * (o + i) * rp, tm * split),
            mma_flops=12.0 * o * i * rp,
            vector_flops=3.0 * o * i,
            # each (row tile, split) reads both ups (and transposes) and both downs' split
            l2_bytes=o * i * e_g
            + 2 * (2 * tm * split * bm + tm * i) * rp * 2
            + 4 * (split * o + tm * i) * rp * 4,
        )

    return [*rank(dev, legal, work, mma=True), TORCH, WHOLE]


def _merge_bwd_run(cfg, g2, factors, scale):
    a1, b1, a2, b2 = factors
    if cfg == TORCH:
        gw = (g2 * scale).to(a1.dtype)
        e1 = gw * (a2 @ b2)
        e2 = gw * (a1 @ b1)
        return e1 @ b1.t(), a1.t() @ e1, e2 @ b2.t(), a2.t() @ e2
    if cfg == WHOLE:
        return merge.loha_merge_bwd(g2, *factors, scale)
    o, r = a1.shape
    i = b1.shape[1]
    rp = pad_to(r)
    bm, bn = cfg["BM"], cfg["BN"]
    split = max(1, min(cfg["SPLIT"], cdiv(i, bn)))
    pa1, pb1, pa2, pb2 = _padded(factors, rp)
    dev = a1.device
    ga = [torch.empty(split, o, rp, device=dev, dtype=torch.float32) for _ in range(2)]
    gb = [
        torch.empty(cdiv(o, bm), rp, i, device=dev, dtype=torch.float32)
        for _ in range(2)
    ]
    kernel = compiled(_merge_bwd_prim, dev, o, i, rp, bm, bn, cfg["threads"], split)
    kernel(
        g2.contiguous(),
        pa1,
        pa1.t().contiguous(),
        pb1,
        pa2,
        pa2.t().contiguous(),
        pb2,
        ga[0],
        gb[0],
        ga[1],
        gb[1],
        scale,
    )
    grads = (
        ga[0].sum(0)[:, :r],
        gb[0].sum(0)[:r],
        ga[1].sum(0)[:, :r],
        gb[1].sum(0)[:r],
    )
    return tuple(g.to(a1.dtype) for g in grads)


def loha_merge_bwd(grad, a1, b1, a2, b2, gamma: float = 1.0):
    """(g_a1, g_b1, g_a2, g_b2) of gamma * (a1 @ b1) * (a2 @ b2) from the (O, I)
    grad, in the factors' dtype."""
    factors = (a1, b1, a2, b2)
    if a1.dtype != torch.float16 or grad.dtype != torch.float16:
        return merge.loha_merge_bwd(grad, *factors, gamma)
    dev = device_info(a1.device)
    o, r = a1.shape
    i = b1.shape[1]
    key = (
        "tl.loha.merge_bwd",
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
            lambda c, s: lambda: _merge_bwd_run(c, s[0], factors, gamma),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _merge_bwd_run(c, grad, factors, gamma),
        CompileError,
    )


def _bypass_run(cfg, x, factors, gamma, base, out_dtype):
    if cfg == WHOLE:
        y = bypass._bypass_delta(x, *factors, gamma)
        return (y if base is None else base + y).to(out_dtype)
    delta = loha_merge(*factors, gamma)
    if cfg == MATERIALIZE and base is not None and base.dtype == x.dtype:
        return torch.addmm(base, x, delta.t()).to(out_dtype)
    y = x @ delta.t()
    return (y if base is None else base + y).to(out_dtype)


def loha_bypass_fwd(x, a1, b1, a2, b2, gamma: float = 1.0, base=None):
    """[base +] gamma * x @ ((a1 @ b1) * (a2 @ b2))^T for x (T, I); the result
    in base's dtype when given, else x's."""
    out_dtype = base.dtype if base is not None else x.dtype
    factors = (a1, b1, a2, b2)
    if x.dtype != torch.float16:
        return _bypass_run(WHOLE, x, factors, gamma, base, out_dtype)
    dev = device_info(x.device)
    t, i = x.shape
    o, r = a1.shape
    key = (
        "tl.loha.bypass_fwd",
        (dtype_key(out_dtype), base is not None),
        (token_bucket(t), o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: [MATERIALIZE, WHOLE],
        measurer(
            dev,
            t * (i + o) * 2,
            lambda: (copy_of(x), copy_of(base)),
            lambda c, s: lambda: _bypass_run(c, s[0], factors, gamma, s[1], out_dtype),
        ),
    )
    return _bypass_run(cfg, x, factors, gamma, base, out_dtype)


def _bypass_bwd_materialized(grad, x, factors, gamma):
    delta = loha_merge(*factors, gamma)
    gx = grad @ delta
    gw = grad.t() @ x
    return (gx, *loha_merge_bwd(gw, *factors, gamma))


def loha_bypass_bwd(grad, x, a1, b1, a2, b2, gamma: float = 1.0):
    """(gx, g_a1, g_b1, g_a2, g_b2) of the LoHa bypass in the compute dtype."""
    factors = (a1, b1, a2, b2)
    r = a1.shape[1]
    if x.dtype != torch.float16:
        return bypass.loha_bypass_bwd(grad, x, *factors, gamma)
    if r > MAX_HADA_RANK:
        return _bypass_bwd_materialized(grad, x, factors, gamma)
    dev = device_info(x.device)
    key = (
        "tl.loha.bypass_bwd",
        (dtype_key(x.dtype),),
        (token_bucket(x.shape[0]), a1.shape[0], x.shape[1], r),
    )

    def run(cf, g, xx):
        if cf == WHOLE:
            return bypass.loha_bypass_bwd(g, xx, *factors, gamma)
        return _bypass_bwd_materialized(g, xx, factors, gamma)

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
