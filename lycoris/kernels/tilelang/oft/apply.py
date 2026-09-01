"""F5 (TileLang): Diag-OFT with the Cayley transform fused in-kernel.

Mirrors the Triton twin: one launch per direction, Gauss-Jordan inverse of
(I - q) in shared memory (no pivoting: (I-q)(I-q)^T = I + q^T q; padded rows
hold identity so elimination is a no-op there), rescale/shift folded, partial
gRf chained in-kernel (gQ = (I+R)^T gR M^T is linear in gR). Grid 2D
(k, col_groups). Apply orientation is R^T per the eager einsum contract.
"""

import tilelang
import tilelang.language as T
import torch

from ...gradbuf import GradPack
from ...plans import oft as plan
from ...plans import tune
from ...plans.device import resolve_device


def _ps(s: int) -> int:
    p = 16
    while p < s:
        p *= 2
    if p > 32:
        raise ValueError(f"in-kernel Cayley supports block size <= 32, got {s}")
    return p


def _layouts(x: torch.Tensor, weight: bool):
    if weight:
        return 1, x.shape[1], 0, x.stride(1), x.stride(0), x.shape[1]
    if x.dim() == 2:
        return x.shape[0], 1, x.stride(0), 0, x.stride(1), x.shape[0]
    nb = x.shape[0]
    ln = 1
    for d in x.shape[2:]:
        ln *= d
    v = x.reshape(nb, x.shape[1], ln)
    return nb, ln, v.stride(0), v.stride(2), v.stride(1), nb * ln


def _dt(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


def oft_fwd(blocks, x, rescale=None, cscale=1.0, shift=True, weight=True):
    k, s, _ = blocks.shape
    xc = x.contiguous()
    nb, ln, sb, sl, sc, cols = _layouts(xc, weight)
    res = (
        rescale.reshape(-1).to(x.dtype).contiguous()
        if rescale is not None
        else xc.new_zeros(k * s)
    )
    bc = blocks.contiguous()
    flat = xc.reshape(-1)

    def build(p):
        fn = _oft_fwd(
            k,
            s,
            cols,
            nb,
            ln,
            rescale is not None,
            shift,
            weight,
            _dt(x),
            cg=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        return lambda: fn(bc, res, flat, float(cscale), sb, sl, sc)

    best = tune.tuned(
        "tilelang.oft.fwd",
        (
            k,
            s,
            tune.bucket_tokens(cols),
            weight,
            shift,
            rescale is not None,
            str(x.dtype),
        ),
        lambda: plan.topk_fused(k, s, cols, x.element_size(), resolve_device()),
        build,
    )
    return build(best)().view_as(x)


def oft_bwd(blocks, x, grad, rescale=None, cscale=1.0, shift=True, weight=True):
    k, s, _ = blocks.shape
    xc = x.contiguous()
    gc = grad.contiguous()
    nb, ln, sb, sl, sc, cols = _layouts(xc, weight)
    res = (
        rescale.reshape(-1).to(x.dtype).contiguous()
        if rescale is not None
        else xc.new_zeros(k * s)
    )
    bc = blocks.contiguous()

    # One fp32 allocation for both atomic targets: one zero-fill, one cast.
    pack = GradPack(x.device, (k, s, s), (k * s,))
    gb, gres = pack

    def run(p, o_gb, o_gres):
        fn = _oft_bwd(
            k,
            s,
            cols,
            nb,
            ln,
            rescale is not None,
            shift,
            weight,
            _dt(x),
            cg=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        return fn(
            bc,
            res,
            xc.reshape(-1),
            gc.reshape(-1),
            o_gb,
            o_gres,
            float(cscale),
            sb,
            sl,
            sc,
        )

    def factory(p):
        scratch = pack.like()
        return lambda: run(p, *scratch)

    best = tune.tuned(
        "tilelang.oft.bwd",
        (
            k,
            s,
            tune.bucket_tokens(cols),
            weight,
            shift,
            rescale is not None,
            str(x.dtype),
        ),
        lambda: plan.topk_fused(k, s, cols, x.element_size(), resolve_device()),
        factory,
    )
    gx = run(best, gb, gres)
    o_gb, o_gres = pack.to(blocks.dtype)
    return gx.view_as(x), o_gb, o_gres if rescale is not None else None


@tilelang.jit(out_idx=[3])
def _oft_fwd(
    K, S, COLS, NB, L, rescale_on, shift, weight, dtype, cg=4, bn=64, threads=64
):
    ps = _ps(S)

    @T.prim_func
    def main(
        blocks: T.Tensor((K, S, S), dtype),
        res: T.Tensor((K * S,), dtype),
        x: T.Tensor((NB * L * K * S,), dtype),
        out: T.Tensor((NB * L * K * S,), dtype),
        cscale: T.float32,
        sxb: T.int32,
        sxl: T.int32,
        sxc: T.int32,
    ):
        with T.Kernel(K, cg, threads=threads) as (bk, bg):
            a_s = T.alloc_shared((ps, ps), "float32")
            m_s = T.alloc_shared((ps, ps), "float32")
            q_s = T.alloc_shared((ps, ps), "float32")
            rf_s = T.alloc_shared((ps, ps), "float32")
            x_s = T.alloc_shared((ps, bn), "float32")
            o_f = T.alloc_fragment((ps, bn), "float")

            for i, j in T.Parallel(ps, ps):
                qv = T.if_then_else(
                    (i < S) and (j < S),
                    (
                        T.cast(blocks[bk, i, j], "float32")
                        - T.cast(blocks[bk, j, i], "float32")
                    )
                    * cscale,
                    T.cast(0, "float32"),
                )
                q_s[i, j] = qv
                eye = T.if_then_else(i == j, T.cast(1, "float32"), T.cast(0, "float32"))
                a_s[i, j] = eye - qv
                m_s[i, j] = eye
            for jj in T.serial(ps):
                for i, j in T.Parallel(ps, ps):
                    if i != jj:
                        fac = a_s[i, jj] / a_s[jj, jj]
                        a_s[i, j] = a_s[i, j] - fac * a_s[jj, j]
                        m_s[i, j] = m_s[i, j] - fac * m_s[jj, j]
                for j in T.Parallel(ps):
                    piv = a_s[jj, jj]
                    a_s[jj, j] = a_s[jj, j] / piv
                    m_s[jj, j] = m_s[jj, j] / piv
            # rf[i, j] = res[i] * R[j, i] - shift*I; R = (I+q) @ M
            for i, j in T.Parallel(ps, ps):
                acc = T.alloc_var("float32")
                acc = T.cast(0, "float32")
                for kk in T.serial(ps):
                    lhs = (
                        T.if_then_else(
                            j == kk, T.cast(1, "float32"), T.cast(0, "float32")
                        )
                        + q_s[j, kk]
                    )
                    acc += lhs * m_s[kk, i]
                scale = T.if_then_else(
                    rescale_on and (i < S),
                    T.cast(res[bk * S + i], "float32"),
                    T.cast(1, "float32"),
                )
                sh = T.if_then_else(
                    shift and (i == j), T.cast(1, "float32"), T.cast(0, "float32")
                )
                rf_s[i, j] = scale * acc - sh

            span = T.ceildiv(COLS, cg)
            for c0 in T.serial(T.ceildiv(span, bn)):
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    ok = (i < S) and (col < (bg + 1) * span) and (col < COLS)
                    x_s[i, j] = T.if_then_else(
                        ok,
                        T.cast(x[tb * sxb + tlp * sxl + (bk * S + i) * sxc], "float32"),
                        T.cast(0, "float32"),
                    )
                T.clear(o_f)
                T.gemm(rf_s, x_s, o_f)
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    if (i < S) and (col < (bg + 1) * span) and (col < COLS):
                        out[tb * sxb + tlp * sxl + (bk * S + i) * sxc] = T.cast(
                            o_f[i, j], dtype
                        )

    return main


@tilelang.jit(out_idx=[4])
def _oft_bwd(
    K, S, COLS, NB, L, rescale_on, shift, weight, dtype, cg=4, bn=64, threads=64
):
    ps = _ps(S)

    @T.prim_func
    def main(
        blocks: T.Tensor((K, S, S), dtype),
        res: T.Tensor((K * S,), dtype),
        x: T.Tensor((NB * L * K * S,), dtype),
        g: T.Tensor((NB * L * K * S,), dtype),
        gx: T.Tensor((NB * L * K * S,), dtype),
        gb: T.Tensor((K, S, S), "float32"),
        gres: T.Tensor((K * S,), "float32"),
        cscale: T.float32,
        sxb: T.int32,
        sxl: T.int32,
        sxc: T.int32,
    ):
        with T.Kernel(K, cg, threads=threads) as (bk, bg):
            a_s = T.alloc_shared((ps, ps), "float32")
            m_s = T.alloc_shared((ps, ps), "float32")
            q_s = T.alloc_shared((ps, ps), "float32")
            r_s = T.alloc_shared((ps, ps), "float32")
            rf_s = T.alloc_shared((ps, ps), "float32")
            x_s = T.alloc_shared((ps, bn), "float32")
            g_s = T.alloc_shared((ps, bn), "float32")
            gx_f = T.alloc_fragment((ps, bn), "float")
            grf_f = T.alloc_fragment((ps, ps), "float")
            grf_s = T.alloc_shared((ps, ps), "float32")
            tmp_s = T.alloc_shared((ps, ps), "float32")

            for i, j in T.Parallel(ps, ps):
                qv = T.if_then_else(
                    (i < S) and (j < S),
                    (
                        T.cast(blocks[bk, i, j], "float32")
                        - T.cast(blocks[bk, j, i], "float32")
                    )
                    * cscale,
                    T.cast(0, "float32"),
                )
                q_s[i, j] = qv
                eye = T.if_then_else(i == j, T.cast(1, "float32"), T.cast(0, "float32"))
                a_s[i, j] = eye - qv
                m_s[i, j] = eye
            for jj in T.serial(ps):
                for i, j in T.Parallel(ps, ps):
                    if i != jj:
                        fac = a_s[i, jj] / a_s[jj, jj]
                        a_s[i, j] = a_s[i, j] - fac * a_s[jj, j]
                        m_s[i, j] = m_s[i, j] - fac * m_s[jj, j]
                for j in T.Parallel(ps):
                    piv = a_s[jj, jj]
                    a_s[jj, j] = a_s[jj, j] / piv
                    m_s[jj, j] = m_s[jj, j] / piv
            for i, j in T.Parallel(ps, ps):
                acc = T.alloc_var("float32")
                acc = T.cast(0, "float32")
                for kk in T.serial(ps):
                    lhs = (
                        T.if_then_else(
                            i == kk, T.cast(1, "float32"), T.cast(0, "float32")
                        )
                        + q_s[i, kk]
                    )
                    acc += lhs * m_s[kk, j]
                r_s[i, j] = acc
            for i, j in T.Parallel(ps, ps):
                scale = T.if_then_else(
                    rescale_on and (i < S),
                    T.cast(res[bk * S + i], "float32"),
                    T.cast(1, "float32"),
                )
                sh = T.if_then_else(
                    shift and (i == j), T.cast(1, "float32"), T.cast(0, "float32")
                )
                rf_s[i, j] = scale * r_s[j, i] - sh

            T.clear(grf_f)
            span = T.ceildiv(COLS, cg)
            for c0 in T.serial(T.ceildiv(span, bn)):
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    ok = (i < S) and (col < (bg + 1) * span) and (col < COLS)
                    x_s[i, j] = T.if_then_else(
                        ok,
                        T.cast(x[tb * sxb + tlp * sxl + (bk * S + i) * sxc], "float32"),
                        T.cast(0, "float32"),
                    )
                    g_s[i, j] = T.if_then_else(
                        ok,
                        T.cast(g[tb * sxb + tlp * sxl + (bk * S + i) * sxc], "float32"),
                        T.cast(0, "float32"),
                    )
                T.clear(gx_f)
                T.gemm(rf_s, g_s, gx_f, transpose_A=True)
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    if (i < S) and (col < (bg + 1) * span) and (col < COLS):
                        gx[tb * sxb + tlp * sxl + (bk * S + i) * sxc] = T.cast(
                            gx_f[i, j], dtype
                        )
                T.gemm(g_s, x_s, grf_f, transpose_B=True)

            T.copy(grf_f, grf_s)
            for i in T.Parallel(ps):
                if rescale_on and (i < S):
                    acc = T.alloc_var("float32")
                    acc = T.cast(0, "float32")
                    for j in T.serial(ps):
                        acc += grf_s[i, j] * r_s[j, i]
                    T.atomic_add(gres[bk * S + i], acc)
            # gR[j,i] = res[i]*gRf[i,j]; gQ = (I+R)^T gR M^T; gB = (gQ-gQ^T)*cscale
            for i, j in T.Parallel(ps, ps):
                scale = T.if_then_else(
                    rescale_on and (j < S),
                    T.cast(res[bk * S + j], "float32"),
                    T.cast(1, "float32"),
                )
                tmp_s[i, j] = scale * grf_s[j, i]
            for i, j in T.Parallel(ps, ps):
                acc = T.alloc_var("float32")
                acc = T.cast(0, "float32")
                for kk in T.serial(ps):
                    lhs = (
                        T.if_then_else(
                            kk == i, T.cast(1, "float32"), T.cast(0, "float32")
                        )
                        + r_s[kk, i]
                    )
                    acc += lhs * tmp_s[kk, j]
                rf_s[i, j] = acc
            # gq = u @ M^T, then gB = (gq - gq^T) * cscale, atomically.
            for i, j in T.Parallel(ps, ps):
                acc = T.alloc_var("float32")
                acc = T.cast(0, "float32")
                for kk in T.serial(ps):
                    acc += rf_s[i, kk] * m_s[j, kk]
                tmp_s[i, j] = acc
            for i, j in T.Parallel(ps, ps):
                if (i < S) and (j < S):
                    T.atomic_add(gb[bk, i, j], (tmp_s[i, j] - tmp_s[j, i]) * cscale)

    return main
