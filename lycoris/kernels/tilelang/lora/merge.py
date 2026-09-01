"""F1 (TileLang): low-rank rebuild — same design as the Triton twin.

Rank fits one K tile (padded shared buffers zero-fill out-of-bounds), so the
whole DeltaW is two shared-loads and one or two gemms per output tile.
"""

import tilelang
import tilelang.language as T
import torch

from ...plans import lora as plan
from ...plans import tune
from ...plans.cost import SENTINEL_EAGER
from ...plans.device import resolve_device
from ..common import dstr


@tilelang.jit(out_idx=[5])
def _rebuild(O, I, R, mode, add_base, dtype, bm=64, bn=64, threads=128):
    br = max(16, R)

    @T.prim_func
    def main(
        a1: T.Tensor((O, R), dtype),
        b1: T.Tensor((R, I), dtype),
        a2: T.Tensor((O, R), dtype),
        b2: T.Tensor((R, I), dtype),
        base: T.Tensor((O, I), dtype),
        out: T.Tensor((O, I), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(I, bn), T.ceildiv(O, bm), threads=threads) as (bx, by):
            a_s = T.alloc_shared((bm, br), dtype)
            b_s = T.alloc_shared((br, bn), dtype)
            acc1 = T.alloc_fragment((bm, bn), "float")
            acc2 = T.alloc_fragment((bm, bn), "float")
            T.copy(a1[by * bm, 0], a_s)
            T.copy(b1[0, bx * bn], b_s)
            T.clear(acc1)
            T.gemm(a_s, b_s, acc1)
            if mode > 0:
                T.copy(a2[by * bm, 0], a_s)
                T.copy(b2[0, bx * bn], b_s)
                T.clear(acc2)
                T.gemm(a_s, b_s, acc2)
            for i, j in T.Parallel(bm, bn):
                if mode == 1:
                    acc1[i, j] = acc1[i, j] * acc2[i, j] * gamma
                elif mode == 2:
                    acc1[i, j] = (acc1[i, j] + acc2[i, j]) * gamma
                else:
                    acc1[i, j] = acc1[i, j] * gamma
                if add_base:
                    acc1[i, j] += base[by * bm + i, bx * bn + j]
            T.copy(acc1, out[by * bm, bx * bn])

    return main


@tilelang.jit(out_idx=[7])
def _rebuild_tucker(O, I, K, R, hada, dtype, bm=64, bn=64, threads=128):
    br = max(16, R)

    @T.prim_func
    def main(
        a1: T.Tensor((O, R), dtype),
        t1: T.Tensor((R, R, K), dtype),
        b1: T.Tensor((R, I), dtype),
        a2: T.Tensor((O, R), dtype),
        t2: T.Tensor((R, R, K), dtype),
        b2: T.Tensor((R, I), dtype),
        gamma_t: T.Tensor((1,), "float32"),
        out: T.Tensor((O, I, K), dtype),
    ):
        with T.Kernel(T.ceildiv(I, bn), T.ceildiv(O, bm), K, threads=threads) as (
            bx,
            by,
            bk,
        ):
            a_s = T.alloc_shared((bm, br), dtype)
            t_s = T.alloc_shared((br, br), dtype)
            b_s = T.alloc_shared((br, bn), dtype)
            at_f = T.alloc_fragment((bm, br), "float")
            at_s = T.alloc_shared((bm, br), dtype)
            acc1 = T.alloc_fragment((bm, bn), "float")
            acc2 = T.alloc_fragment((bm, bn), "float")

            for i, j in T.Parallel(br, br):
                if i < R and j < R:
                    t_s[i, j] = t1[i, j, bk]
                else:
                    t_s[i, j] = T.cast(0, dtype)
            T.copy(a1[by * bm, 0], a_s)
            T.clear(at_f)
            T.gemm(a_s, t_s, at_f)
            T.copy(at_f, at_s)
            T.copy(b1[0, bx * bn], b_s)
            T.clear(acc1)
            T.gemm(at_s, b_s, acc1)

            if hada:
                for i, j in T.Parallel(br, br):
                    if i < R and j < R:
                        t_s[i, j] = t2[i, j, bk]
                    else:
                        t_s[i, j] = T.cast(0, dtype)
                T.copy(a2[by * bm, 0], a_s)
                T.clear(at_f)
                T.gemm(a_s, t_s, at_f)
                T.copy(at_f, at_s)
                T.copy(b2[0, bx * bn], b_s)
                T.clear(acc2)
                T.gemm(at_s, b_s, acc2)
                for i, j in T.Parallel(bm, bn):
                    acc1[i, j] = acc1[i, j] * acc2[i, j] * gamma_t[0]
            else:
                for i, j in T.Parallel(bm, bn):
                    acc1[i, j] = acc1[i, j] * gamma_t[0]

            for i, j in T.Parallel(bm, bn):
                if by * bm + i < O and bx * bn + j < I:
                    out[by * bm + i, bx * bn + j, bk] = T.cast(acc1[i, j], dtype)

    return main


@tilelang.jit
def _loha_merge_bwd(O, I, R, dtype, bm=64, bn=64, threads=128):
    """All four hadamard factor grads, one role-split 1D launch, no atomics.

    bx < ga: own an O tile, reduce over I -> ga1, ga2. Otherwise: own an I
    tile, reduce over O -> gb1, gb2. Each CTA holds its whole reduction, so
    the grads are plain dtype stores — no fp32 scratch, zero-fill or cast, and
    the result is deterministic. Both roles run the same gemm sequence; only
    addressing and the store guards branch, never a T.gemm.
    """
    br = max(16, R)
    ga = -(-O // bm)
    # Role A walks I in bn steps, role B walks O in bm steps; one shared count
    # covers both because every out-of-range tile is masked to zero.
    steps = max(-(-I // bn), -(-O // bm))

    @T.prim_func
    def main(
        g: T.Tensor((O, I), dtype),
        a1: T.Tensor((O, R), dtype),
        b1: T.Tensor((R, I), dtype),
        a2: T.Tensor((O, R), dtype),
        b2: T.Tensor((R, I), dtype),
        ga1: T.Tensor((O, R), dtype),
        gb1: T.Tensor((R, I), dtype),
        ga2: T.Tensor((O, R), dtype),
        gb2: T.Tensor((R, I), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(ga + T.ceildiv(I, bn), threads=threads) as bx:
            a1_s = T.alloc_shared((bm, br), dtype)
            a2_s = T.alloc_shared((bm, br), dtype)
            b1_s = T.alloc_shared((br, bn), dtype)
            b2_s = T.alloc_shared((br, bn), dtype)
            g_s = T.alloc_shared((bm, bn), dtype)
            p1 = T.alloc_fragment((bm, bn), "float")
            p2 = T.alloc_fragment((bm, bn), "float")
            e_s = T.alloc_shared((bm, bn), dtype)
            ga1_f = T.alloc_fragment((bm, br), "float")
            ga2_f = T.alloc_fragment((bm, br), "float")
            gb1_f = T.alloc_fragment((br, bn), "float")
            gb2_f = T.alloc_fragment((br, bn), "float")

            T.clear(ga1_f)
            T.clear(ga2_f)
            T.clear(gb1_f)
            T.clear(gb2_f)
            for it in T.serial(steps):
                # Role A pins the O tile and walks I; role B does the mirror.
                o0 = T.if_then_else(bx < ga, bx * bm, it * bm)
                i0 = T.if_then_else(bx < ga, it * bn, (bx - ga) * bn)
                for i, j in T.Parallel(bm, br):
                    ok = (o0 + i < O) and (j < R)
                    a1_s[i, j] = T.if_then_else(
                        ok, a1[T.min(o0 + i, O - 1), T.min(j, R - 1)], T.cast(0, dtype)
                    )
                    a2_s[i, j] = T.if_then_else(
                        ok, a2[T.min(o0 + i, O - 1), T.min(j, R - 1)], T.cast(0, dtype)
                    )
                for i, j in T.Parallel(br, bn):
                    ok = (i < R) and (i0 + j < I)
                    b1_s[i, j] = T.if_then_else(
                        ok, b1[T.min(i, R - 1), T.min(i0 + j, I - 1)], T.cast(0, dtype)
                    )
                    b2_s[i, j] = T.if_then_else(
                        ok, b2[T.min(i, R - 1), T.min(i0 + j, I - 1)], T.cast(0, dtype)
                    )
                for i, j in T.Parallel(bm, bn):
                    ok = (o0 + i < O) and (i0 + j < I)
                    g_s[i, j] = T.if_then_else(
                        ok,
                        g[T.min(o0 + i, O - 1), T.min(i0 + j, I - 1)],
                        T.cast(0, dtype),
                    )
                # p1 = a1@b1, p2 = a2@b2 over this (O, I) tile.
                T.clear(p1)
                T.gemm(a1_s, b1_s, p1)
                T.clear(p2)
                T.gemm(a2_s, b2_s, p2)
                # e1 = gamma*G*p2: ga1 += e1@b1^T and gb1 += a1^T@e1.
                for i, j in T.Parallel(bm, bn):
                    e_s[i, j] = T.cast(
                        T.cast(g_s[i, j], "float32") * gamma * p2[i, j], dtype
                    )
                T.gemm(e_s, b1_s, ga1_f, transpose_B=True)
                T.gemm(a1_s, e_s, gb1_f, transpose_A=True)
                # e2 = gamma*G*p1: the 2-side twin.
                for i, j in T.Parallel(bm, bn):
                    e_s[i, j] = T.cast(
                        T.cast(g_s[i, j], "float32") * gamma * p1[i, j], dtype
                    )
                T.gemm(e_s, b2_s, ga2_f, transpose_B=True)
                T.gemm(a2_s, e_s, gb2_f, transpose_A=True)
            for i, j in T.Parallel(bm, br):
                if (bx < ga) and (bx * bm + i < O) and (j < R):
                    ga1[bx * bm + i, j] = T.cast(ga1_f[i, j], dtype)
                    ga2[bx * bm + i, j] = T.cast(ga2_f[i, j], dtype)
            for i, j in T.Parallel(br, bn):
                if (bx >= ga) and (i < R) and ((bx - ga) * bn + j < I):
                    gb1[i, (bx - ga) * bn + j] = T.cast(gb1_f[i, j], dtype)
                    gb2[i, (bx - ga) * bn + j] = T.cast(gb2_f[i, j], dtype)

    return main


def lora_merge_fwd(a1, b1, a2=None, b2=None, base=None, gamma=1.0, mode="plain"):
    mode_id = {"plain": 0, "hada": 1, "sum": 2}[mode]
    o, r = a1.shape
    i = b1.shape[1]
    a1c = a1.contiguous()
    b1c = b1.contiguous()
    a2c = a2.contiguous() if a2 is not None else a1c
    b2c = b2.contiguous() if b2 is not None else b1c
    basec = base.contiguous() if base is not None else a1c.new_zeros(o, i)

    def build(p):
        fn = _rebuild(
            o,
            i,
            r,
            mode_id,
            base is not None,
            dstr(a1),
            bm=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        return lambda: fn(a1c, b1c, a2c, b2c, basec, float(gamma))

    def eager_run():
        acc = a1c @ b1c
        if mode_id == 1:
            acc = acc * (a2c @ b2c)
        elif mode_id == 2:
            acc = acc + a2c @ b2c
        acc = acc * gamma
        return acc + basec if base is not None else acc

    shortlist = lambda: [
        *plan.topk_rebuild(o, i, r, mode_id == 1, a1.element_size(), resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        return eager_run if p.limiter == "eager" else build(p)

    best = tune.tuned(
        "tilelang.lora.merge_fwd",
        (o, i, r, mode_id, base is not None, str(a1.dtype)),
        shortlist,
        factory,
    )
    return eager_run() if best.limiter == "eager" else build(best)()


def lora_tucker_fwd(a1, t1, b1, a2=None, t2=None, b2=None, gamma=1.0):
    o, r = a1.shape
    i = b1.shape[1]
    k = t1.shape[2]
    if r > 64:
        raise ValueError("tucker rebuild kernel supports rank <= 64")
    hada = a2 is not None
    a1c = a1.contiguous()
    t1c = t1.contiguous()
    b1c = b1.contiguous()
    a2c = a2.contiguous() if hada else a1c
    t2c = t2.contiguous() if hada else t1c
    b2c = b2.contiguous() if hada else b1c
    gam = torch.tensor([float(gamma)], device=a1.device, dtype=torch.float32)

    def build(p):
        fn = _rebuild_tucker(
            o, i, k, r, hada, dstr(a1), bm=p.bm, bn=p.bn, threads=32 * p.warps
        )
        return lambda: fn(a1c, t1c, b1c, a2c, t2c, b2c, gam)

    shortlist = lambda: plan.topk_rebuild(
        o, i, r, hada, a1.element_size(), resolve_device()
    )
    best = tune.tuned(
        "tilelang.lora.tucker_fwd",
        (o, i, k, r, hada, str(a1.dtype)),
        shortlist,
        build,
    )
    return build(best)()


def loha_merge_bwd(grad, a1, b1, a2, b2, gamma=1.0):
    """Role-split, so each CTA owns its whole reduction and writes the
    parameter dtype directly — no fp32 scratch, zero-fill or cast launch."""
    o, r = a1.shape
    i = b1.shape[1]
    if r > 128:
        raise ValueError("hadamard backward kernel supports rank <= 128")
    ga1 = torch.empty(o, r, device=a1.device, dtype=a1.dtype)
    gb1 = torch.empty(r, i, device=a1.device, dtype=b1.dtype)
    ga2 = torch.empty(o, r, device=a1.device, dtype=a2.dtype)
    gb2 = torch.empty(r, i, device=a1.device, dtype=b2.dtype)
    gc = grad.contiguous()
    a1c, b1c = a1.contiguous(), b1.contiguous()
    a2c, b2c = a2.contiguous(), b2.contiguous()

    def run(p, o1, o2, o3, o4):
        fn = _loha_merge_bwd(o, i, r, dstr(a1), bm=p.bm, bn=p.bn, threads=32 * p.warps)
        fn(gc, a1c, b1c, a2c, b2c, o1, o2, o3, o4, float(gamma))

    def factory(p):
        s = [torch.empty_like(t) for t in (ga1, gb1, ga2, gb2)]
        return lambda: run(p, *s)

    shortlist = lambda: plan.topk_hada_bwd(o, i, r, a1.element_size(), resolve_device())
    best = tune.tuned(
        "tilelang.loha.merge_bwd",
        (o, i, r, str(a1.dtype)),
        shortlist,
        factory,
    )
    run(best, ga1, gb1, ga2, gb2)
    return ga1, gb1, ga2, gb2
