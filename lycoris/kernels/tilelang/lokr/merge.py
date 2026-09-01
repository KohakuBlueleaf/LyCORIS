"""F3 (TileLang): Kronecker rebuild, mirroring the Triton twin.

Both directions are gridded ON the Kronecker structure — one CTA owns one
(u, v) outer pair and one (C, D) sub-tile — so w1[u,v] is a scalar and the
w2 tile is a plain 2D access: no div/mod gather, and generating the factors
from their low-rank halves in-kernel costs one rank-length reduction plus one
small gemm. The backward chains the sub-factor grads in the same launch, so
neither direction materializes a factor or issues a host mm.
"""

import tilelang
import tilelang.language as T

from ...gradbuf import GradPack
from ...plans import lokr as plan
from ...plans import tune
from ...plans.device import resolve_device


@tilelang.jit(out_idx=[3])
def _lokr_full_merge_fwd(A, B, C, D, add_base, dtype, bm=32, bn=64, threads=128):
    O = A * C
    I = B * D

    @T.prim_func
    def main(
        w1: T.Tensor((A, B), dtype),
        w2: T.Tensor((C, D), dtype),
        base: T.Tensor((O, I), dtype),
        out: T.Tensor((O, I), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(I, bn), T.ceildiv(O, bm), threads=threads) as (bx, by):
            # DeltaW[u*C+c, v*D+d] = gamma * w1[u,v] * w2[c,d] (+ base).
            for i, j in T.Parallel(bm, bn):
                ro = by * bm + i
                ri = bx * bn + j
                if ro < O and ri < I:
                    v = (
                        T.cast(w1[ro // C, ri // D], "float32")
                        * T.cast(w2[ro % C, ri % D], "float32")
                        * gamma
                    )
                    if add_base:
                        v += T.cast(base[ro, ri], "float32")
                    out[ro, ri] = T.cast(v, dtype)

    return main


@tilelang.jit(out_idx=[5])
def _lokr_merge_fwd(
    A, B, C, D, R1, R2, gen1, gen2, add_base, dtype, bm=32, bn=64, threads=128
):
    O = A * C
    I = B * D
    pr = max(16, R1, R2)
    nc = -(-C // bm)
    nd = -(-D // bn)

    @T.prim_func
    def main(
        w1a: T.Tensor((A, R1 if gen1 else B), dtype),
        w1b: T.Tensor((max(R1, 1), B), dtype),
        w2a: T.Tensor((C, R2 if gen2 else D), dtype),
        w2b: T.Tensor((max(R2, 1), D), dtype),
        base: T.Tensor((O, I), dtype),
        out: T.Tensor((O, I), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(A * B * nc * nd, threads=threads) as bx:
            a2_s = T.alloc_shared((bm, pr), dtype)
            b2_s = T.alloc_shared((pr, bn), dtype)
            w2_f = T.alloc_fragment((bm, bn), "float")
            w1_f = T.alloc_fragment((pr,), "float")
            w1_r = T.alloc_fragment((1,), "float")

            uv = bx // (nc * nd)
            rest = bx % (nc * nd)
            u = uv // B
            v = uv % B
            c0 = (rest // nd) * bm
            d0 = (rest % nd) * bn

            # w1[u, v]: scalar, either loaded or reduced over the rank axis.
            if gen1:
                for k in T.Parallel(pr):
                    w1_f[k] = T.if_then_else(
                        k < R1,
                        T.cast(w1a[u, T.min(k, R1 - 1)], "float32")
                        * T.cast(w1b[T.min(k, R1 - 1), v], "float32"),
                        T.cast(0, "float32"),
                    )
                T.reduce_sum(w1_f, w1_r, dim=0)
            else:
                w1_r[0] = T.cast(w1a[u, v], "float32")

            # w2 tile: loaded, or generated as w2a[c-tile,:] @ w2b[:,d-tile].
            if gen2:
                for i, k in T.Parallel(bm, pr):
                    ok = (c0 + i < C) and (k < R2)
                    a2_s[i, k] = T.if_then_else(
                        ok,
                        w2a[T.min(c0 + i, C - 1), T.min(k, R2 - 1)],
                        T.cast(0, dtype),
                    )
                for k, j in T.Parallel(pr, bn):
                    ok = (k < R2) and (d0 + j < D)
                    b2_s[k, j] = T.if_then_else(
                        ok,
                        w2b[T.min(k, R2 - 1), T.min(d0 + j, D - 1)],
                        T.cast(0, dtype),
                    )
                T.clear(w2_f)
                T.gemm(a2_s, b2_s, w2_f)
            else:
                for i, j in T.Parallel(bm, bn):
                    ok = (c0 + i < C) and (d0 + j < D)
                    w2_f[i, j] = T.if_then_else(
                        ok,
                        T.cast(
                            w2a[T.min(c0 + i, C - 1), T.min(d0 + j, D - 1)], "float32"
                        ),
                        T.cast(0, "float32"),
                    )

            # DeltaW[u*C+c, v*D+d] = gamma * w1[u,v] * w2[c,d] (+ base).
            for i, j in T.Parallel(bm, bn):
                if c0 + i < C and d0 + j < D:
                    ro = u * C + c0 + i
                    ri = v * D + d0 + j
                    val = w1_r[0] * w2_f[i, j] * gamma
                    if add_base:
                        val += T.cast(base[ro, ri], "float32")
                    out[ro, ri] = T.cast(val, dtype)

    return main


@tilelang.jit
def _lokr_merge_bwd(A, B, C, D, R1, R2, gen1, gen2, dtype, bm=32, bn=64, threads=128):
    pr = max(16, R1, R2)
    nc = -(-C // bm)
    nd = -(-D // bn)
    ga = A * B

    @T.prim_func
    def main(
        g: T.Tensor((A * C, B * D), dtype),
        w1a: T.Tensor((A, R1 if gen1 else B), dtype),
        w1b: T.Tensor((max(R1, 1), B), dtype),
        w2a: T.Tensor((C, R2 if gen2 else D), dtype),
        w2b: T.Tensor((max(R2, 1), D), dtype),
        g1a: T.Tensor((A, R1 if gen1 else B), "float32"),
        g1b: T.Tensor((max(R1, 1), B), "float32"),
        g2a: T.Tensor((C, R2 if gen2 else D), "float32"),
        g2b: T.Tensor((max(R2, 1), D), "float32"),
        gamma: T.float32,
    ):
        with T.Kernel(ga + nc * nd, threads=threads) as bx:
            acc = T.alloc_fragment((bm, bn), "float")
            red = T.alloc_fragment((bm,), "float")
            red2 = T.alloc_fragment((1,), "float")
            a2_s = T.alloc_shared((bm, pr), dtype)
            b2_s = T.alloc_shared((pr, bn), dtype)
            w2_f = T.alloc_fragment((bm, bn), "float")
            gw2_s = T.alloc_shared((bm, bn), dtype)
            ga2_f = T.alloc_fragment((bm, pr), "float")
            gb2_f = T.alloc_fragment((pr, bn), "float")
            w1_f = T.alloc_fragment((pr,), "float")
            w1_r = T.alloc_fragment((1,), "float")

            if bx < ga:
                # Role A: gw1[u,v] = gamma * sum_{c,d} G[u*C+c, v*D+d]*w2[c,d].
                u = bx // B
                v = bx % B
                T.clear(acc)
                for ct in T.serial(nc):
                    for dt in T.serial(nd):
                        if gen2:
                            for i, k in T.Parallel(bm, pr):
                                ok = (ct * bm + i < C) and (k < R2)
                                a2_s[i, k] = T.if_then_else(
                                    ok,
                                    w2a[T.min(ct * bm + i, C - 1), T.min(k, R2 - 1)],
                                    T.cast(0, dtype),
                                )
                            for k, j in T.Parallel(pr, bn):
                                ok = (k < R2) and (dt * bn + j < D)
                                b2_s[k, j] = T.if_then_else(
                                    ok,
                                    w2b[T.min(k, R2 - 1), T.min(dt * bn + j, D - 1)],
                                    T.cast(0, dtype),
                                )
                            T.clear(w2_f)
                            T.gemm(a2_s, b2_s, w2_f)
                        else:
                            for i, j in T.Parallel(bm, bn):
                                ok = (ct * bm + i < C) and (dt * bn + j < D)
                                w2_f[i, j] = T.if_then_else(
                                    ok,
                                    T.cast(
                                        w2a[
                                            T.min(ct * bm + i, C - 1),
                                            T.min(dt * bn + j, D - 1),
                                        ],
                                        "float32",
                                    ),
                                    T.cast(0, "float32"),
                                )
                        for i, j in T.Parallel(bm, bn):
                            rc = ct * bm + i
                            rd = dt * bn + j
                            if rc < C and rd < D:
                                acc[i, j] += (
                                    T.cast(g[u * C + rc, v * D + rd], "float32")
                                    * w2_f[i, j]
                                )
                T.reduce_sum(acc, red, dim=1)
                T.reduce_sum(red, red2, dim=0)
                # Chain the scalar: g_w1a[u,:] += gw1*w1b[:,v], and the twin.
                if gen1:
                    for k in T.Parallel(pr):
                        if k < R1:
                            T.atomic_add(
                                g1a[u, k],
                                red2[0] * gamma * T.cast(w1b[k, v], "float32"),
                            )
                            T.atomic_add(
                                g1b[k, v],
                                red2[0] * gamma * T.cast(w1a[u, k], "float32"),
                            )
                else:
                    g1a[u, v] = red2[0] * gamma
            else:
                # Role B: gw2[c,d] = gamma * sum_{u,v} w1[u,v]*G[u*C+c, v*D+d].
                rest = bx - ga
                c0 = (rest // nd) * bm
                d0 = (rest % nd) * bn
                T.clear(acc)
                for u in T.serial(A):
                    for v in T.serial(B):
                        if gen1:
                            for k in T.Parallel(pr):
                                w1_f[k] = T.if_then_else(
                                    k < R1,
                                    T.cast(w1a[u, T.min(k, R1 - 1)], "float32")
                                    * T.cast(w1b[T.min(k, R1 - 1), v], "float32"),
                                    T.cast(0, "float32"),
                                )
                            T.reduce_sum(w1_f, w1_r, dim=0)
                        else:
                            w1_r[0] = T.cast(w1a[u, v], "float32")
                        for i, j in T.Parallel(bm, bn):
                            rc = c0 + i
                            rd = d0 + j
                            if rc < C and rd < D:
                                acc[i, j] += w1_r[0] * T.cast(
                                    g[u * C + rc, v * D + rd], "float32"
                                )
                if gen2:
                    # Chain the tile: g_w2a += gw2@w2b^T, g_w2b += w2a^T@gw2.
                    for i, j in T.Parallel(bm, bn):
                        gw2_s[i, j] = T.cast(acc[i, j] * gamma, dtype)
                    for i, k in T.Parallel(bm, pr):
                        ok = (c0 + i < C) and (k < R2)
                        a2_s[i, k] = T.if_then_else(
                            ok,
                            w2a[T.min(c0 + i, C - 1), T.min(k, R2 - 1)],
                            T.cast(0, dtype),
                        )
                    for k, j in T.Parallel(pr, bn):
                        ok = (k < R2) and (d0 + j < D)
                        b2_s[k, j] = T.if_then_else(
                            ok,
                            w2b[T.min(k, R2 - 1), T.min(d0 + j, D - 1)],
                            T.cast(0, dtype),
                        )
                    T.clear(ga2_f)
                    T.gemm(gw2_s, b2_s, ga2_f, transpose_B=True)
                    for i, k in T.Parallel(bm, pr):
                        if c0 + i < C and k < R2:
                            T.atomic_add(g2a[c0 + i, k], ga2_f[i, k])
                    T.clear(gb2_f)
                    T.gemm(a2_s, gw2_s, gb2_f, transpose_A=True)
                    for k, j in T.Parallel(pr, bn):
                        if k < R2 and d0 + j < D:
                            T.atomic_add(g2b[k, d0 + j], gb2_f[k, j])
                else:
                    for i, j in T.Parallel(bm, bn):
                        if c0 + i < C and d0 + j < D:
                            g2a[c0 + i, d0 + j] = acc[i, j] * gamma

    return main


def lokr_full_merge_fwd(w1, w2, base=None, gamma=1.0):
    a, b = w1.shape
    c, d = w2.shape
    w1c = w1.contiguous()
    w2c = w2.contiguous()
    basec = base.contiguous() if base is not None else w2c.new_zeros(a * c, b * d)
    dt = str(w2.dtype).split(".")[-1]

    def build(p):
        fn = _lokr_full_merge_fwd(
            a,
            b,
            c,
            d,
            base is not None,
            dt,
            bm=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        return lambda: fn(w1c, w2c, basec, float(gamma))

    shortlist = lambda: plan.topk_rebuild(
        a * c, b * d, base is not None, w2.element_size(), resolve_device()
    )
    best = tune.tuned(
        "tilelang.lokr.full_merge_fwd",
        (a, b, c, d, base is not None, str(w2.dtype)),
        shortlist,
        build,
    )
    return build(best)()


def _halves(w1a, w1b, w2a, w2b, shape):
    """Padded stand-ins for absent halves, so the prim_func shapes are fixed."""
    b, d = shape[1], shape[3]
    r1 = w1b.shape[0] if w1b is not None else 0
    r2 = w2b.shape[0] if w2b is not None else 0
    z1 = w1a.new_zeros(max(r1, 1), b)
    z2 = w2a.new_zeros(max(r2, 1), d)
    return (
        w1a.contiguous(),
        w1b.contiguous() if w1b is not None else z1,
        w2a.contiguous(),
        w2b.contiguous() if w2b is not None else z2,
        r1,
        r2,
    )


def lokr_merge_fwd(w1a, w1b, w2a, w2b, shape, base=None, gamma=1.0):
    a, b, c, d = shape
    w1ac, w1bc, w2ac, w2bc, r1, r2 = _halves(w1a, w1b, w2a, w2b, shape)
    basec = base.contiguous() if base is not None else w2ac.new_zeros(a * c, b * d)
    dt = str(w2a.dtype).split(".")[-1]

    def build(p):
        fn = _lokr_merge_fwd(
            a,
            b,
            c,
            d,
            r1,
            r2,
            w1b is not None,
            w2b is not None,
            base is not None,
            dt,
            bm=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        return lambda: fn(w1ac, w1bc, w2ac, w2bc, basec, float(gamma))

    shortlist = lambda: plan.topk_rebuild(
        a * c, b * d, base is not None, w2a.element_size(), resolve_device()
    )
    best = tune.tuned(
        "tilelang.lokr.merge_fwd",
        (a, b, c, d, r1, r2, base is not None, str(w2a.dtype)),
        shortlist,
        build,
    )
    return build(best)()


def lokr_merge_bwd(grad, w1a, w1b, w2a, w2b, shape, gamma=1.0):
    a, b, c, d = shape
    gc = grad.contiguous()
    w1ac, w1bc, w2ac, w2bc, r1, r2 = _halves(w1a, w1b, w2a, w2b, shape)
    dt = str(grad.dtype).split(".")[-1]
    # One fp32 allocation for every grad this launch writes: one fill, one cast.
    pack = GradPack(grad.device, w1ac.shape, w1bc.shape, w2ac.shape, w2bc.shape)
    g1a, g1b, g2a, g2b = pack

    def run(p, o1, o2, o3, o4):
        fn = _lokr_merge_bwd(
            a,
            b,
            c,
            d,
            r1,
            r2,
            w1b is not None,
            w2b is not None,
            dt,
            bm=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        fn(gc, w1ac, w1bc, w2ac, w2bc, o1, o2, o3, o4, float(gamma))

    def factory(p):
        scratch = pack.like()
        return lambda: run(p, *scratch)

    shortlist = lambda: plan.topk_rebuild_bwd(
        a, b, c, d, grad.element_size(), resolve_device()
    )
    best = tune.tuned(
        "tilelang.lokr.merge_bwd",
        (a, b, c, d, r1, r2, str(grad.dtype)),
        shortlist,
        factory,
    )
    run(best, g1a, g1b, g2a, g2b)
    o1a, o1b, o2a, o2b = pack.to(w1a.dtype)
    return (
        o1a,
        o1b if w1b is not None else None,
        o2a,
        o2b if w2b is not None else None,
    )


def lokr_full_merge_bwd(grad, w1, w2, gamma=1.0):
    """Kept for the whole-factor path (no low-rank halves): the same
    role-split launch with both GEN flags off."""
    a, b = w1.shape
    c, d = w2.shape
    g1a, _, g2a, _ = lokr_merge_bwd(grad, w1, None, w2, None, (a, b, c, d), gamma)
    return g1a, g2a
