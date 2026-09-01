"""F2 (TileLang): generated-B hadamard delta apply, mirroring the Triton twin."""

import tilelang
import tilelang.language as T
import torch

from ...gradbuf import GradPack
from ...plans import lora as plan
from ...plans import tune
from ...plans.cost import SENTINEL_MATERIALIZE
from ...plans.device import resolve_device
from ..lora.merge import loha_merge_bwd, lora_merge_fwd


@tilelang.jit(out_idx=[5])
def _hada_delta(TT, O, I, R, dtype, bm=64, bn=64, bk=64, threads=128):
    br = max(16, R)

    @T.prim_func
    def main(
        x: T.Tensor((TT, I), dtype),
        a1: T.Tensor((O, R), dtype),
        b1: T.Tensor((R, I), dtype),
        a2: T.Tensor((O, R), dtype),
        b2: T.Tensor((R, I), dtype),
        y: T.Tensor((TT, O), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(O, bn), T.ceildiv(TT, bm), threads=threads) as (bx, by):
            x_s = T.alloc_shared((bm, bk), dtype)
            an1_s = T.alloc_shared((bn, br), dtype)
            an2_s = T.alloc_shared((bn, br), dtype)
            bk1_s = T.alloc_shared((br, bk), dtype)
            bk2_s = T.alloc_shared((br, bk), dtype)
            p1 = T.alloc_fragment((bk, bn), "float")
            p2 = T.alloc_fragment((bk, bn), "float")
            w_s = T.alloc_shared((bk, bn), dtype)
            acc = T.alloc_fragment((bm, bn), "float")

            T.copy(a1[bx * bn, 0], an1_s)
            T.copy(a2[bx * bn, 0], an2_s)
            T.clear(acc)
            for it in T.Pipelined(T.ceildiv(I, bk), num_stages=1):
                T.copy(x[by * bm, it * bk], x_s)
                T.copy(b1[0, it * bk], bk1_s)
                T.copy(b2[0, it * bk], bk2_s)
                T.clear(p1)
                T.gemm(bk1_s, an1_s, p1, transpose_A=True, transpose_B=True)
                T.clear(p2)
                T.gemm(bk2_s, an2_s, p2, transpose_A=True, transpose_B=True)
                for i, j in T.Parallel(bk, bn):
                    w_s[i, j] = T.cast(p1[i, j] * p2[i, j], dtype)
                T.gemm(x_s, w_s, acc)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < TT and bx * bn + j < O:
                    y[by * bm + i, bx * bn + j] = T.cast(acc[i, j] * gamma, dtype)

    return main


@tilelang.jit
def _loha_bypass_bwd(TT, O, I, R, dtype, bm=64, bn=64, threads=128):
    """One launch for the whole bypass backward, role-split on a linear grid.

    Both roles share one gemm sequence (contraction chunk = bm), so only load
    addressing and store guards branch on the role — never a T.gemm (the
    fragment-layout lesson from the DoRA kernel). Role A (dw tiles): main
    gemm is gW = g^T@x over t; per-iteration p1/p2 regenerate the SAME fixed
    factor tile, so the epilogue can consume them. Role B (dx tiles): main
    gemm is gx = g@W over o with W = p1*p2 regenerated per chunk.
    """
    br = max(16, R)
    nin = -(-I // bn)
    ga_tiles = -(-O // bm) * nin
    grid = ga_tiles + -(-TT // bm) * nin
    steps = -(-max(TT, O) // bm)

    @T.prim_func
    def main(
        g: T.Tensor((TT, O), dtype),
        x: T.Tensor((TT, I), dtype),
        a1: T.Tensor((O, R), dtype),
        b1: T.Tensor((R, I), dtype),
        a2: T.Tensor((O, R), dtype),
        b2: T.Tensor((R, I), dtype),
        gx: T.Tensor((TT, I), dtype),
        ga1: T.Tensor((O, R), "float32"),
        gb1: T.Tensor((R, I), "float32"),
        ga2: T.Tensor((O, R), "float32"),
        gb2: T.Tensor((R, I), "float32"),
        gamma: T.float32,
    ):
        with T.Kernel(grid, threads=threads) as bx:
            a1_s = T.alloc_shared((bm, br), dtype)
            a2_s = T.alloc_shared((bm, br), dtype)
            b1_s = T.alloc_shared((br, bn), dtype)
            b2_s = T.alloc_shared((br, bn), dtype)
            aa_s = T.alloc_shared((bm, bm), dtype)
            bb_s = T.alloc_shared((bm, bn), dtype)
            p1 = T.alloc_fragment((bm, bn), "float")
            p2 = T.alloc_fragment((bm, bn), "float")
            acc = T.alloc_fragment((bm, bn), "float")
            e_s = T.alloc_shared((bm, bn), dtype)
            ta_f = T.alloc_fragment((bm, br), "float")
            tb_f = T.alloc_fragment((br, bn), "float")

            pm = T.if_then_else(bx < ga_tiles, bx // nin, (bx - ga_tiles) // nin)
            pn = T.if_then_else(bx < ga_tiles, bx % nin, (bx - ga_tiles) % nin)
            rows0 = pm * bm
            cols0 = pn * bn

            # b-factor tiles at this CTA's I columns (both roles).
            for i, j in T.Parallel(br, bn):
                ok = (i < R) and (cols0 + j < I)
                b1_s[i, j] = T.if_then_else(
                    ok, b1[T.min(i, R - 1), T.min(cols0 + j, I - 1)], T.cast(0, dtype)
                )
                b2_s[i, j] = T.if_then_else(
                    ok, b2[T.min(i, R - 1), T.min(cols0 + j, I - 1)], T.cast(0, dtype)
                )

            T.clear(acc)
            for it in T.serial(steps):
                # a-factor slices: dw = the fixed tile rows, dx = this o chunk.
                for i, j in T.Parallel(bm, br):
                    row = T.if_then_else(bx < ga_tiles, rows0 + i, it * bm + i)
                    ok = (row < O) and (j < R)
                    a1_s[i, j] = T.if_then_else(
                        ok, a1[T.min(row, O - 1), T.min(j, R - 1)], T.cast(0, dtype)
                    )
                    a2_s[i, j] = T.if_then_else(
                        ok, a2[T.min(row, O - 1), T.min(j, R - 1)], T.cast(0, dtype)
                    )
                # p1 = a1@b1, p2 = a2@b2 (dx: the W chunk; dw: the fixed tile).
                T.clear(p1)
                T.gemm(a1_s, b1_s, p1)
                T.clear(p2)
                T.gemm(a2_s, b2_s, p2)
                # Main A: dw = g[t, o-tile] (contract t), dx = g[t-tile, o] (contract o).
                for k, i in T.Parallel(bm, bm):
                    v = T.alloc_var(dtype)
                    v = T.cast(0, dtype)
                    if bx < ga_tiles:
                        if (it * bm + k < TT) and (rows0 + i < O):
                            v = g[it * bm + k, rows0 + i]
                    else:
                        if (rows0 + i < TT) and (it * bm + k < O):
                            v = g[rows0 + i, it * bm + k]
                    aa_s[k, i] = v
                # Main B: dw = x chunk, dx = W chunk = p1*p2.
                for k, j in T.Parallel(bm, bn):
                    v = T.alloc_var(dtype)
                    v = T.cast(0, dtype)
                    if bx < ga_tiles:
                        if (it * bm + k < TT) and (cols0 + j < I):
                            v = x[it * bm + k, cols0 + j]
                    else:
                        v = T.cast(p1[k, j] * p2[k, j], dtype)
                    bb_s[k, j] = v
                # acc += A^T@B: dw -> gW = g^T@x, dx -> gx = g@W.
                T.gemm(aa_s, bb_s, acc, transpose_A=True)

            # dw epilogue (guarded stores only; gemms run in every CTA):
            # e1 = gamma*gW*p2 -> ga1 += e1@b1^T, gb1 += a1^T@e1; then the twin.
            for i, j in T.Parallel(bm, bn):
                e_s[i, j] = T.cast(acc[i, j] * gamma * p2[i, j], dtype)
            T.clear(ta_f)
            T.gemm(e_s, b1_s, ta_f, transpose_B=True)
            for i, j in T.Parallel(bm, br):
                if (bx < ga_tiles) and (rows0 + i < O) and (j < R):
                    T.atomic_add(ga1[rows0 + i, j], ta_f[i, j])
            T.clear(tb_f)
            T.gemm(a1_s, e_s, tb_f, transpose_A=True)
            for i, j in T.Parallel(br, bn):
                if (bx < ga_tiles) and (i < R) and (cols0 + j < I):
                    T.atomic_add(gb1[i, cols0 + j], tb_f[i, j])
            for i, j in T.Parallel(bm, bn):
                e_s[i, j] = T.cast(acc[i, j] * gamma * p1[i, j], dtype)
            T.clear(ta_f)
            T.gemm(e_s, b2_s, ta_f, transpose_B=True)
            for i, j in T.Parallel(bm, br):
                if (bx < ga_tiles) and (rows0 + i < O) and (j < R):
                    T.atomic_add(ga2[rows0 + i, j], ta_f[i, j])
            T.clear(tb_f)
            T.gemm(a2_s, e_s, tb_f, transpose_A=True)
            for i, j in T.Parallel(br, bn):
                if (bx < ga_tiles) and (i < R) and (cols0 + j < I):
                    T.atomic_add(gb2[i, cols0 + j], tb_f[i, j])
            # dx store: gx = gamma * acc.
            for i, j in T.Parallel(bm, bn):
                if (bx >= ga_tiles) and (rows0 + i < TT) and (cols0 + j < I):
                    gx[rows0 + i, cols0 + j] = T.cast(acc[i, j] * gamma, dtype)

    return main


def _canon(x, a1, b1, a2, b2):
    return (
        x.contiguous(),
        a1.contiguous(),
        b1.contiguous(),
        a2.contiguous(),
        b2.contiguous(),
    )


def _delta_run(xc, a1c, b1c, a2c, b2c, t, o, i, r, gamma, tag):
    def build(p):
        fn = _hada_delta(
            t,
            o,
            i,
            r,
            str(xc.dtype).split(".")[-1],
            bm=p.bm,
            bn=p.bn,
            bk=p.bk,
            threads=32 * p.warps,
        )
        return lambda: fn(xc, a1c, b1c, a2c, b2c, float(gamma))

    def materialize_run():
        w = lora_merge_fwd(a1c, b1c, a2c, b2c, gamma=gamma, mode="hada")
        return xc @ w.transpose(0, 1)

    shortlist = lambda: [
        *plan.topk_delta(t, o, i, r, xc.element_size(), resolve_device()),
        SENTINEL_MATERIALIZE,
    ]

    def factory(p):
        return materialize_run if p.limiter == "materialize" else build(p)

    best = tune.tuned(
        f"tilelang.loha.{tag}",
        (tune.bucket_tokens(t), o, i, r, str(xc.dtype)),
        shortlist,
        factory,
    )
    return materialize_run() if best.limiter == "materialize" else build(best)()


def loha_bypass_fwd(x, a1, b1, a2, b2, gamma=1.0):
    t = x.shape[0]
    o, r = a1.shape
    i = b1.shape[1]
    xc, a1c, b1c, a2c, b2c = _canon(x, a1, b1, a2, b2)
    return _delta_run(xc, a1c, b1c, a2c, b2c, t, o, i, r, gamma, "delta")


def loha_bypass_bwd(grad, x, a1, b1, a2, b2, gamma=1.0):
    """(gx, ga1, gb1, ga2, gb2) of the bypass chain in ONE launch."""
    t, o = grad.shape
    i = x.shape[1]
    r = a1.shape[1]
    if r > 128:
        raise ValueError("hadamard delta backward supports rank <= 128")
    gx = torch.empty_like(x)
    # One fp32 allocation for all four atomic targets: one fill, one cast.
    pack = GradPack(x.device, (o, r), (r, i), (o, r), (r, i))
    ga1, gb1, ga2, gb2 = pack
    xc, a1c, b1c, a2c, b2c = _canon(x, a1, b1, a2, b2)
    gc = grad.contiguous()

    def run(p, o_gx, o1, o2, o3, o4):
        fn = _loha_bypass_bwd(
            t,
            o,
            i,
            r,
            str(x.dtype).split(".")[-1],
            bm=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        fn(gc, xc, a1c, b1c, a2c, b2c, o_gx, o1, o2, o3, o4, float(gamma))

    def materialize_run():
        w = lora_merge_fwd(a1c, b1c, a2c, b2c, gamma=gamma, mode="hada")
        gw = gc.transpose(0, 1) @ xc
        return (
            gc @ w,
            *loha_merge_bwd(gw, a1c, b1c, a2c, b2c, gamma=gamma),
        )

    def factory(p):
        if p.limiter == "materialize":
            return materialize_run
        s0 = torch.empty_like(gx)
        scratch = pack.like()
        return lambda: run(p, s0, *scratch)

    shortlist = lambda: [
        *plan.topk_hada_bypass_bwd(t, o, i, r, x.element_size(), resolve_device()),
        SENTINEL_MATERIALIZE,
    ]
    best = tune.tuned(
        "tilelang.loha.bypass_bwd",
        (tune.bucket_tokens(t), o, i, r, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "materialize":
        return materialize_run()
    run(best, gx, ga1, gb1, ga2, gb2)
    return (gx, *pack.to(a1.dtype))
