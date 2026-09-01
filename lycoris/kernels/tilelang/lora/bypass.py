"""LoCon/LoRA bypass + plain merge backward (TileLang), mirroring the Triton
twins: bypass fwd/bwd are single 1D token-tiled kernels keeping h = x@down^T
and q = g@up on-chip; merge bwd is one role-split launch where both roles
share one gemm shape (tile, bk) @ (bk, br) so only addressing branches on the
role, never a T.gemm (the fragment-layout lesson from the DoRA kernel).
"""

import tilelang
import tilelang.language as T
import torch

from ...gradbuf import GradPack
from ...plans import lora as plan
from ...plans import tune
from ...plans.cost import SENTINEL_EAGER
from ...plans.device import resolve_device


def _dt(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


@tilelang.jit(out_idx=[3])
def _bypass_fwd(TT, O, I, R, dtype, bm=64, bn=64, bk=64, threads=128):
    br = max(16, R)

    @T.prim_func
    def main(
        x: T.Tensor((TT, I), dtype),
        d: T.Tensor((R, I), dtype),
        u: T.Tensor((O, R), dtype),
        y: T.Tensor((TT, O), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(TT, bm), threads=threads) as bx:
            x_s = T.alloc_shared((bm, bk), dtype)
            d_s = T.alloc_shared((br, bk), dtype)
            u_s = T.alloc_shared((bn, br), dtype)
            h_f = T.alloc_fragment((bm, br), "float")
            h_s = T.alloc_shared((bm, br), dtype)
            y_f = T.alloc_fragment((bm, bn), "float")

            T.clear(h_f)
            for it in T.Pipelined(T.ceildiv(I, bk), num_stages=1):
                T.copy(x[bx * bm, it * bk], x_s)
                T.copy(d[0, it * bk], d_s)
                T.gemm(x_s, d_s, h_f, transpose_B=True)
            for i, j in T.Parallel(bm, br):
                h_s[i, j] = T.cast(h_f[i, j] * gamma, dtype)
            for ot in T.serial(T.ceildiv(O, bn)):
                T.copy(u[ot * bn, 0], u_s)
                T.clear(y_f)
                T.gemm(h_s, u_s, y_f, transpose_B=True)
                for i, j in T.Parallel(bm, bn):
                    if bx * bm + i < TT and ot * bn + j < O:
                        y[bx * bm + i, ot * bn + j] = T.cast(y_f[i, j], dtype)

    return main


@tilelang.jit(out_idx=[4])
def _bypass_bwd(TT, O, I, R, dtype, bm=64, bn=64, bk=64, threads=128):
    br = max(16, R)

    @T.prim_func
    def main(
        x: T.Tensor((TT, I), dtype),
        d: T.Tensor((R, I), dtype),
        u: T.Tensor((O, R), dtype),
        g: T.Tensor((TT, O), dtype),
        gx: T.Tensor((TT, I), dtype),
        gu: T.Tensor((O, R), "float32"),
        gd: T.Tensor((R, I), "float32"),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(TT, bm), threads=threads) as bx:
            x_s = T.alloc_shared((bm, bk), dtype)
            d_s = T.alloc_shared((br, bk), dtype)
            u_s = T.alloc_shared((bn, br), dtype)
            g_s = T.alloc_shared((bm, bn), dtype)
            h_f = T.alloc_fragment((bm, br), "float")
            h_s = T.alloc_shared((bm, br), dtype)
            q_f = T.alloc_fragment((bm, br), "float")
            q_s = T.alloc_shared((bm, br), dtype)
            gu_f = T.alloc_fragment((bn, br), "float")
            gx_f = T.alloc_fragment((bm, bk), "float")
            gd_f = T.alloc_fragment((br, bk), "float")

            T.clear(h_f)
            for it in T.Pipelined(T.ceildiv(I, bk), num_stages=1):
                T.copy(x[bx * bm, it * bk], x_s)
                T.copy(d[0, it * bk], d_s)
                T.gemm(x_s, d_s, h_f, transpose_B=True)
            T.copy(h_f, h_s)

            T.clear(q_f)
            for ot in T.serial(T.ceildiv(O, bn)):
                T.copy(g[bx * bm, ot * bn], g_s)
                T.copy(u[ot * bn, 0], u_s)
                T.gemm(g_s, u_s, q_f)
                T.clear(gu_f)
                T.gemm(g_s, h_s, gu_f, transpose_A=True)
                for i, j in T.Parallel(bn, br):
                    if ot * bn + i < O and j < R:
                        T.atomic_add(gu[ot * bn + i, j], gu_f[i, j] * gamma)
            T.copy(q_f, q_s)

            for it in T.serial(T.ceildiv(I, bk)):
                T.copy(d[0, it * bk], d_s)
                T.clear(gx_f)
                T.gemm(q_s, d_s, gx_f)
                for i, j in T.Parallel(bm, bk):
                    if bx * bm + i < TT and it * bk + j < I:
                        gx[bx * bm + i, it * bk + j] = T.cast(gx_f[i, j] * gamma, dtype)
                T.copy(x[bx * bm, it * bk], x_s)
                T.clear(gd_f)
                T.gemm(q_s, x_s, gd_f, transpose_A=True)
                for i, j in T.Parallel(br, bk):
                    if i < R and it * bk + j < I:
                        T.atomic_add(gd[i, it * bk + j], gd_f[i, j] * gamma)

    return main


@tilelang.jit
def _merge_bwd(O, I, R, dtype, bm=64, bk=64, threads=128):
    br = max(16, R)
    ga = -(-O // bm)
    steps = -(-max(O, I) // bk)

    @T.prim_func
    def main(
        g: T.Tensor((O, I), dtype),
        u: T.Tensor((O, R), dtype),
        d: T.Tensor((R, I), dtype),
        gu: T.Tensor((O, R), dtype),
        gd: T.Tensor((R, I), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(ga + T.ceildiv(I, bm), threads=threads) as bx:
            g_s = T.alloc_shared((bm, bk), dtype)
            w_s = T.alloc_shared((bk, br), dtype)
            acc = T.alloc_fragment((bm, br), "float")

            T.clear(acc)
            for it in T.serial(steps):
                # Role A (bx < ga): rows of g times down^T over the i axis.
                # Role B: columns of g (transposed load) times up over o.
                for i, j in T.Parallel(bm, bk):
                    g_s[i, j] = T.if_then_else(
                        bx < ga,
                        T.if_then_else(
                            (bx * bm + i < O) and (it * bk + j < I),
                            g[
                                T.min(bx * bm + i, O - 1),
                                T.min(it * bk + j, I - 1),
                            ],
                            T.cast(0, dtype),
                        ),
                        T.if_then_else(
                            ((bx - ga) * bm + i < I) and (it * bk + j < O),
                            g[
                                T.min(it * bk + j, O - 1),
                                T.min((bx - ga) * bm + i, I - 1),
                            ],
                            T.cast(0, dtype),
                        ),
                    )
                for i, j in T.Parallel(bk, br):
                    w_s[i, j] = T.if_then_else(
                        bx < ga,
                        T.if_then_else(
                            (j < R) and (it * bk + i < I),
                            d[T.min(j, R - 1), T.min(it * bk + i, I - 1)],
                            T.cast(0, dtype),
                        ),
                        T.if_then_else(
                            (j < R) and (it * bk + i < O),
                            u[T.min(it * bk + i, O - 1), T.min(j, R - 1)],
                            T.cast(0, dtype),
                        ),
                    )
                T.gemm(g_s, w_s, acc)
            for i, j in T.Parallel(bm, br):
                if bx < ga:
                    if bx * bm + i < O and j < R:
                        gu[bx * bm + i, j] = T.cast(acc[i, j] * gamma, dtype)
                else:
                    if (bx - ga) * bm + i < I and j < R:
                        gd[j, (bx - ga) * bm + i] = T.cast(acc[i, j] * gamma, dtype)

    return main


def lora_bypass_fwd(x, up, down, gamma: float = 1.0) -> torch.Tensor:
    t, i = x.shape
    o, r = up.shape
    xc, uc, dc = x.contiguous(), up.contiguous(), down.contiguous()
    eb = x.element_size()

    def eager_run():
        return (xc @ dc.transpose(0, 1)) @ uc.transpose(0, 1) * gamma

    def build(p):
        fn = _bypass_fwd(
            t, o, i, r, _dt(x), bm=p.bm, bn=p.bn, bk=p.bk, threads=32 * p.warps
        )
        return lambda: fn(xc, dc, uc, float(gamma))

    shortlist = lambda: [
        *plan.topk_bypass_fwd(t, o, i, r, eb, resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        if p.limiter == "eager":
            return eager_run
        return build(p)

    best = tune.tuned(
        "tilelang.lora.bypass_fwd",
        (tune.bucket_tokens(t), o, i, r, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "eager":
        return eager_run()
    return build(best)()


def lora_bypass_bwd(x, up, down, grad, gamma: float = 1.0):
    t, i = x.shape
    o, r = up.shape
    xc, uc, dc, gc = (
        x.contiguous(),
        up.contiguous(),
        down.contiguous(),
        grad.contiguous(),
    )
    eb = x.element_size()
    # One fp32 allocation for both atomic targets: one zero-fill, one cast.
    pack = GradPack(x.device, (o, r), (r, i))
    gu, gd = pack

    def eager_run():
        h = xc @ dc.transpose(0, 1)
        q = gc @ uc
        return (
            q @ dc * gamma,
            gc.transpose(0, 1) @ h * gamma,
            q.transpose(0, 1) @ xc * gamma,
        )

    def run(p, o_gu, o_gd):
        fn = _bypass_bwd(
            t, o, i, r, _dt(x), bm=p.bm, bn=p.bn, bk=p.bk, threads=32 * p.warps
        )
        return fn(xc, dc, uc, gc, o_gu, o_gd, float(gamma))

    shortlist = lambda: [
        *plan.topk_bypass_bwd(t, o, i, r, eb, resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        if p.limiter == "eager":
            return eager_run
        scratch = pack.like()
        return lambda: run(p, *scratch)

    best = tune.tuned(
        "tilelang.lora.bypass_bwd",
        (tune.bucket_tokens(t), o, i, r, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "eager":
        return eager_run()
    gx = run(best, gu, gd)
    g_up, g_down = pack.to(up.dtype)
    return gx, g_up, g_down


def lora_merge_bwd(grad, up, down, gamma: float = 1.0):
    o, r = up.shape
    i = down.shape[1]
    gc, uc, dc = grad.contiguous(), up.contiguous(), down.contiguous()
    eb = up.element_size()
    gu = torch.empty(o, r, device=up.device, dtype=up.dtype)
    gd = torch.empty(r, i, device=up.device, dtype=down.dtype)

    def eager_run():
        g = gc * gamma
        return g @ dc.transpose(0, 1), uc.transpose(0, 1) @ g

    def run(p, o_gu, o_gd):
        fn = _merge_bwd(o, i, r, _dt(up), bm=p.bm, bk=p.bk, threads=32 * p.warps)
        fn(gc, uc, dc, o_gu, o_gd, float(gamma))

    shortlist = lambda: [
        *plan.topk_merge_bwd(o, i, r, eb, resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        if p.limiter == "eager":
            return eager_run
        s1, s2 = torch.empty_like(gu), torch.empty_like(gd)
        return lambda: run(p, s1, s2)

    best = tune.tuned(
        "tilelang.lora.merge_bwd",
        (o, i, r, str(up.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "eager":
        return eager_run()
    run(best, gu, gd)
    return gu, gd
