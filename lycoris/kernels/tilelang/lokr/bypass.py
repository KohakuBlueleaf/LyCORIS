"""F4 (TileLang): grouped Kronecker bypass apply.

Same math as the Triton twin; the per-token transpose between the two small
gemms is an explicit indexed shared-memory copy (TileLang gives full control,
no permute intrinsic needed).
"""

import tilelang
import tilelang.language as T

from ...gradbuf import GradPack
from ...plans import lokr as plan
from ...plans import tune
from ...plans.device import resolve_device


def _pad16(v: int) -> int:
    p = 16
    while p < v:
        p *= 2
    return p


@tilelang.jit(out_idx=[3])
def _lokr_bypass_fwd(TT, A, B, C, D, bt, dtype, threads=128):
    pa, pb, pc, pd = _pad16(A), _pad16(B), _pad16(C), _pad16(D)

    @T.prim_func
    def main(
        x: T.Tensor((TT, B * D), dtype),
        w1: T.Tensor((A, B), dtype),
        w2: T.Tensor((C, D), dtype),
        y: T.Tensor((TT, A * C), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(TT, bt), threads=threads) as (pid,):
            x_s = T.alloc_shared((bt * pb, pd), dtype)
            w2t_s = T.alloc_shared((pd, pc), dtype)
            w1t_s = T.alloc_shared((pb, pa), dtype)
            m_f = T.alloc_fragment((bt * pb, pc), "float")
            m_s = T.alloc_shared((bt * pb, pc), dtype)
            mp_s = T.alloc_shared((bt * pc, pb), dtype)
            y_f = T.alloc_fragment((bt * pc, pa), "float")

            for i, j in T.Parallel(bt * pb, pd):
                tok = pid * bt + i // pb
                bi = i % pb
                if tok < TT and bi < B and j < D:
                    x_s[i, j] = x[tok, bi * D + j]
                else:
                    x_s[i, j] = T.cast(0, dtype)
            for i, j in T.Parallel(pd, pc):
                if i < D and j < C:
                    w2t_s[i, j] = w2[j, i]
                else:
                    w2t_s[i, j] = T.cast(0, dtype)
            T.clear(m_f)
            T.gemm(x_s, w2t_s, m_f)
            T.copy(m_f, m_s)
            for i, j in T.Parallel(bt * pc, pb):
                mp_s[i, j] = m_s[(i // pc) * pb + j, i % pc]
            for i, j in T.Parallel(pb, pa):
                if i < B and j < A:
                    w1t_s[i, j] = w1[j, i]
                else:
                    w1t_s[i, j] = T.cast(0, dtype)
            T.clear(y_f)
            T.gemm(mp_s, w1t_s, y_f)
            for i, j in T.Parallel(bt * pc, pa):
                tok = pid * bt + i // pc
                ci = i % pc
                if tok < TT and ci < C and j < A:
                    y[tok, j * C + ci] = T.cast(y_f[i, j] * gamma, dtype)

    return main


@tilelang.jit(out_idx=[4])
def _lokr_bypass_bwd(TT, A, B, C, D, bt, dtype, threads=64):
    pa, pb, pc, pd = _pad16(A), _pad16(B), _pad16(C), _pad16(D)

    @T.prim_func
    def main(
        g: T.Tensor((TT, A * C), dtype),
        x: T.Tensor((TT, B * D), dtype),
        w1: T.Tensor((A, B), dtype),
        w2: T.Tensor((C, D), dtype),
        gx: T.Tensor((TT, B * D), dtype),
        gw1: T.Tensor((A, B), "float32"),
        gw2: T.Tensor((C, D), "float32"),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(TT, bt), threads=threads) as (pid,):
            g_s = T.alloc_shared((bt * pc, pa), dtype)
            w1_s = T.alloc_shared((pa, pb), dtype)
            n_f = T.alloc_fragment((bt * pc, pb), "float")
            n_s = T.alloc_shared((bt * pc, pb), dtype)
            np_s = T.alloc_shared((bt * pb, pc), dtype)
            w2_s = T.alloc_shared((pc, pd), dtype)
            gx_f = T.alloc_fragment((bt * pb, pd), "float")
            x_s = T.alloc_shared((bt * pb, pd), dtype)
            w2t_s = T.alloc_shared((pd, pc), dtype)
            m_f = T.alloc_fragment((bt * pb, pc), "float")
            mp_s = T.alloc_shared((bt * pc, pb), dtype)
            gw1_f = T.alloc_fragment((pa, pb), "float")
            gw2_f = T.alloc_fragment((pc, pd), "float")

            for i, j in T.Parallel(bt * pc, pa):
                tok = pid * bt + i // pc
                ci = i % pc
                if tok < TT and ci < C and j < A:
                    g_s[i, j] = g[tok, j * C + ci]
                else:
                    g_s[i, j] = T.cast(0, dtype)
            for i, j in T.Parallel(pa, pb):
                if i < A and j < B:
                    w1_s[i, j] = w1[i, j]
                else:
                    w1_s[i, j] = T.cast(0, dtype)
            T.clear(n_f)
            T.gemm(g_s, w1_s, n_f)
            T.copy(n_f, n_s)
            for i, j in T.Parallel(bt * pb, pc):
                np_s[i, j] = n_s[(i // pb) * pc + j, i % pb]
            for i, j in T.Parallel(pc, pd):
                if i < C and j < D:
                    w2_s[i, j] = w2[i, j]
                else:
                    w2_s[i, j] = T.cast(0, dtype)
            T.clear(gx_f)
            T.gemm(np_s, w2_s, gx_f)
            for i, j in T.Parallel(bt * pb, pd):
                tok = pid * bt + i // pb
                bi = i % pb
                if tok < TT and bi < B and j < D:
                    gx[tok, bi * D + j] = T.cast(gx_f[i, j] * gamma, dtype)

            for i, j in T.Parallel(bt * pb, pd):
                tok = pid * bt + i // pb
                bi = i % pb
                if tok < TT and bi < B and j < D:
                    x_s[i, j] = x[tok, bi * D + j]
                else:
                    x_s[i, j] = T.cast(0, dtype)
            for i, j in T.Parallel(pd, pc):
                if i < D and j < C:
                    w2t_s[i, j] = w2[j, i]
                else:
                    w2t_s[i, j] = T.cast(0, dtype)
            T.clear(m_f)
            T.gemm(x_s, w2t_s, m_f)
            for i, j in T.Parallel(bt * pc, pb):
                mp_s[i, j] = T.cast(m_f[(i // pc) * pb + j, i % pc], dtype)
            T.clear(gw1_f)
            T.gemm(g_s, mp_s, gw1_f, transpose_A=True)
            for i, j in T.Parallel(pa, pb):
                if i < A and j < B:
                    T.atomic_add(gw1[i, j], gw1_f[i, j] * gamma)
            T.clear(gw2_f)
            T.gemm(np_s, x_s, gw2_f, transpose_A=True)
            for i, j in T.Parallel(pc, pd):
                if i < C and j < D:
                    T.atomic_add(gw2[i, j], gw2_f[i, j] * gamma)

    return main


def lokr_bypass_fwd(x, w1, w2, gamma=1.0):
    t = x.shape[0]
    a, b = w1.shape
    c, d = w2.shape
    if max(_pad16(a), _pad16(b), _pad16(c), _pad16(d)) > 128:
        raise ValueError("kron apply factors too large for the kernel")
    xc, w1c, w2c = x.contiguous(), w1.contiguous(), w2.contiguous()
    dt = str(x.dtype).split(".")[-1]

    def build(p):
        fn = _lokr_bypass_fwd(t, a, b, c, d, p.bm, dt, threads=32 * p.warps)
        return lambda: fn(xc, w1c, w2c, float(gamma))

    shortlist = lambda: plan.topk_apply(
        t, a, b, c, d, x.element_size(), resolve_device()
    )
    best = tune.tuned(
        "tilelang.lokr.bypass_fwd",
        (tune.bucket_tokens(t), a, b, c, d, str(x.dtype)),
        shortlist,
        build,
    )
    return build(best)()


def lokr_bypass_bwd(grad, x, w1, w2, gamma=1.0):
    t = x.shape[0]
    a, b = w1.shape
    c, d = w2.shape
    # One fp32 allocation for both atomic targets: one zero-fill, one cast.
    pack = GradPack(x.device, (a, b), (c, d))
    gw1, gw2 = pack
    gc, xc = grad.contiguous(), x.contiguous()
    w1c, w2c = w1.contiguous(), w2.contiguous()
    dt = str(x.dtype).split(".")[-1]

    def run(p, o1, o2):
        fn = _lokr_bypass_bwd(t, a, b, c, d, p.bm, dt, threads=32 * p.warps)
        return fn(gc, xc, w1c, w2c, o1, o2, float(gamma))

    def factory(p):
        scratch = pack.like()
        return lambda: run(p, *scratch)

    shortlist = lambda: plan.topk_apply(
        t, a, b, c, d, x.element_size(), resolve_device()
    )
    best = tune.tuned(
        "tilelang.lokr.bypass_bwd",
        (tune.bucket_tokens(t), a, b, c, d, str(x.dtype)),
        shortlist,
        factory,
    )
    gx = run(best, gw1, gw2)
    return (gx, *pack.to(w1.dtype))
