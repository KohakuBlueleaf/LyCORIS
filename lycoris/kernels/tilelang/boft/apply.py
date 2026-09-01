"""F6 (TileLang): BOFT butterfly with per-stage Cayley computed in-kernel.

Mirrors the Triton twin: one launch per stage, Gauss-Jordan inverse of (I - q)
in shared memory, multiplier fold (b*scale + (1-scale)I), last stage folds
rescale and the diff subtraction; the per-stage grad kernel emits both the
chained g_next = Rf^T g and the partial gRf -> g_blocks atomics. Grid 2D
(n_blocks, col_groups). Eager applies R (einsum contracts the second index),
so the forward is untransposed. The bijective butterfly scatter trips
TileLang's data-race check falsely (each (block, row) owns a distinct origin
channel), so that check is disabled per kernel.

Stage-i permutation (functional/boft.py): kd = 2^i * b/2;
s -> o = (s // (2*kd)) * 2*kd + (s % 2) * kd + (s % (2*kd)) // 2.
"""

import tilelang
import tilelang.language as T
import torch

from ...gradbuf import GradPack
from ...plans import boft as plan
from ...plans import tune
from ...plans.device import resolve_device

_NO_RACE = {"tl.disable_data_race_check": True}


def _ps(s: int) -> int:
    p = 16
    while p < s:
        p *= 2
    if p > 32:
        raise ValueError(f"in-kernel Cayley supports block size <= 32, got {s}")
    return p


def _layout(x: torch.Tensor, axis: int):
    if axis in (-1, x.dim() - 1):
        flat = x.reshape(-1, x.shape[-1])
        return flat.shape[0], 1, flat.stride(0), 0, flat.stride(1), flat.shape[1]
    if axis == 0:
        flat = x.reshape(x.shape[0], -1)
        return 1, flat.shape[1], 0, flat.stride(1), flat.stride(0), flat.shape[0]
    nb = x.shape[0]
    ln = 1
    for d in x.shape[2:]:
        ln *= d
    v = x.reshape(nb, x.shape[1], ln)
    return nb, ln, v.stride(0), v.stride(2), v.stride(1), x.shape[1]


def _dt(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


def _pick(blocks, x, axis, name):
    m, nblk, s, _ = blocks.shape
    nb, ln = _layout(x, axis)[:2]

    def factory(p):
        scratch = torch.empty_like(x)
        return lambda: _run_stage(
            blocks, 0, x, scratch, axis, False, 0, 1.0, 1.0, None, None, False, p
        )

    return tune.tuned(
        name,
        (nblk, s, tune.bucket_tokens(nb * ln), m, axis, str(x.dtype)),
        lambda: plan.topk_fused(
            nblk, s, nb * ln, m, x.element_size(), resolve_device()
        ),
        factory,
    )


def _run_stage(
    blocks,
    i,
    src,
    dst,
    axis,
    trans,
    offset,
    cscale,
    scale,
    rescale,
    org,
    diff,
    p,
    last=False,
):
    nblk, s = blocks.shape[1], blocks.shape[2]
    nb, ln, sb, sl, sa, n = _layout(src, axis)
    kd = (2 ** (i + offset)) * (s // 2)
    res = (
        rescale.reshape(-1).to(src.dtype).contiguous()
        if rescale is not None
        else src.new_zeros(n)
    )
    org_ = (org if org is not None else src).contiguous()
    fn = _bf_stage(
        nblk,
        s,
        nb * ln,
        nb,
        ln,
        trans,
        last,
        last and rescale is not None,
        last and diff,
        _dt(src),
        cg=p.bm,
        bn=p.bn,
        threads=32 * p.warps,
    )
    fn(
        blocks[i].contiguous(),
        res,
        src.reshape(-1),
        org_.reshape(-1),
        dst.reshape(-1),
        float(cscale),
        float(scale),
        kd,
        sb,
        sl,
        sa,
    )
    return dst


def boft_fwd(
    blocks,
    x,
    axis=0,
    cscale=1.0,
    scale=1.0,
    rescale=None,
    diff=False,
    trans=False,
    reverse=False,
    offset=0,
):
    """All m stages, Cayley in-kernel; last stage folds rescale and diff."""
    blocks = blocks.contiguous()
    m = blocks.shape[0]
    xc = x.contiguous()
    p = _pick(blocks, xc, axis, "tilelang.boft.fwd")
    out = torch.empty_like(xc)
    tmp = torch.empty_like(xc) if m > 1 else out
    order = list(range(m - 1, -1, -1) if reverse else range(m))
    src = xc
    for step, i in enumerate(order):
        into_out = (m - 1 - step) % 2 == 0
        dst = out if into_out else tmp
        last = step == m - 1 and not trans
        _run_stage(
            blocks,
            i,
            src,
            dst,
            axis,
            trans,
            offset,
            cscale,
            scale,
            rescale,
            xc,
            diff,
            p,
            last=last,
        )
        src = dst
    return out


def boft_bwd(blocks, x, grad, axis=0, cscale=1.0, scale=1.0):
    """(gx, g_blocks): per-stage grad kernels chain g and gRf in-kernel; stage
    inputs come from prefix replays (nothing cached)."""
    blocks = blocks.contiguous()
    m, nblk, s, _ = blocks.shape
    xc = x.contiguous()
    # One fp32 buffer for every stage's atomic target: one fill, one cast.
    pack = GradPack(x.device, (m, nblk, s, s))
    (gb,) = pack
    g = grad.contiguous()
    bufs = [torch.empty_like(g)]
    if m > 1:
        bufs.append(torch.empty_like(g))
    p = _pick(blocks, xc, axis, "tilelang.boft.fwd_bwd")

    def run(p, i, src_g, stage_in, dst_g, o_gb):
        nb, ln, sb, sl, sa, _ = _layout(stage_in, axis)
        kd = (2**i) * (s // 2)
        fn = _bf_grad(
            nblk,
            s,
            nb * ln,
            nb,
            ln,
            _dt(x),
            cg=p.bm,
            bn=p.bn,
            threads=32 * p.warps,
        )
        fn(
            blocks[i].contiguous(),
            src_g.reshape(-1),
            stage_in.reshape(-1),
            dst_g.reshape(-1),
            o_gb,
            float(cscale),
            float(scale),
            kd,
            sb,
            sl,
            sa,
        )

    for step, i in enumerate(range(m - 1, -1, -1)):
        stage_in = (
            boft_fwd(blocks[:i], xc, axis, cscale, scale, trans=False) if i > 0 else xc
        )
        gnext = bufs[step % len(bufs)]
        run(p, i, g, stage_in, gnext, gb[i])
        g = gnext
    return g.view_as(x), pack.to(blocks.dtype)[0]


@tilelang.jit(pass_configs=_NO_RACE)
def _bf_stage(
    K, S, COLS, NB, L, TRANS, LAST, RESCALE, DIFF, dtype, cg=1, bn=64, threads=64
):
    ps = _ps(S)

    @T.prim_func
    def main(
        blocks: T.Tensor((K, S, S), dtype),
        res: T.Tensor((K * S,), dtype),
        x: T.Tensor((NB * L * K * S,), dtype),
        org: T.Tensor((NB * L * K * S,), dtype),
        out: T.Tensor((NB * L * K * S,), dtype),
        cscale: T.float32,
        scale: T.float32,
        kd: T.int32,
        sxb: T.int32,
        sxl: T.int32,
        sxa: T.int32,
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
            # Rf = R*scale + (1-scale)I with R = (I+q) @ M; transposed if TRANS.
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
                eye = T.if_then_else(i == j, T.cast(1, "float32"), T.cast(0, "float32"))
                if TRANS:
                    rf_s[j, i] = acc * scale + (T.cast(1, "float32") - scale) * eye
                else:
                    rf_s[i, j] = acc * scale + (T.cast(1, "float32") - scale) * eye

            span = T.ceildiv(COLS, cg)
            for c0 in T.serial(T.ceildiv(span, bn)):
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    idx = bk * S + i
                    oo = (
                        (idx // (2 * kd)) * 2 * kd
                        + (idx % 2) * kd
                        + (idx % (2 * kd)) // 2
                    )
                    ok = (i < S) and (col < (bg + 1) * span) and (col < COLS)
                    x_s[i, j] = T.if_then_else(
                        ok,
                        T.cast(x[tb * sxb + tlp * sxl + oo * sxa], "float32"),
                        T.cast(0, "float32"),
                    )
                T.clear(o_f)
                T.gemm(rf_s, x_s, o_f)
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    idx = bk * S + i
                    oo = (
                        (idx // (2 * kd)) * 2 * kd
                        + (idx % 2) * kd
                        + (idx % (2 * kd)) // 2
                    )
                    if (i < S) and (col < (bg + 1) * span) and (col < COLS):
                        val = T.alloc_var("float32")
                        val = o_f[i, j]
                        if LAST and RESCALE:
                            val = val * T.cast(res[oo], "float32")
                        if LAST and DIFF:
                            val = val - T.cast(
                                org[tb * sxb + tlp * sxl + oo * sxa], "float32"
                            )
                        out[tb * sxb + tlp * sxl + oo * sxa] = T.cast(val, dtype)

    return main


@tilelang.jit(pass_configs=_NO_RACE)
def _bf_grad(K, S, COLS, NB, L, dtype, cg=1, bn=64, threads=64):
    ps = _ps(S)

    @T.prim_func
    def main(
        blocks: T.Tensor((K, S, S), dtype),
        g: T.Tensor((NB * L * K * S,), dtype),
        x: T.Tensor((NB * L * K * S,), dtype),
        gn: T.Tensor((NB * L * K * S,), dtype),
        gb: T.Tensor((K, S, S), "float32"),
        cscale: T.float32,
        scale: T.float32,
        kd: T.int32,
        sxb: T.int32,
        sxl: T.int32,
        sxa: T.int32,
    ):
        with T.Kernel(K, cg, threads=threads) as (bk, bg):
            a_s = T.alloc_shared((ps, ps), "float32")
            m_s = T.alloc_shared((ps, ps), "float32")
            q_s = T.alloc_shared((ps, ps), "float32")
            r_s = T.alloc_shared((ps, ps), "float32")
            rf_s = T.alloc_shared((ps, ps), "float32")
            x_s = T.alloc_shared((ps, bn), "float32")
            g_s = T.alloc_shared((ps, bn), "float32")
            gn_f = T.alloc_fragment((ps, bn), "float")
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
                eye = T.if_then_else(i == j, T.cast(1, "float32"), T.cast(0, "float32"))
                rf_s[i, j] = r_s[i, j] * scale + (T.cast(1, "float32") - scale) * eye

            T.clear(grf_f)
            span = T.ceildiv(COLS, cg)
            for c0 in T.serial(T.ceildiv(span, bn)):
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    idx = bk * S + i
                    oo = (
                        (idx // (2 * kd)) * 2 * kd
                        + (idx % 2) * kd
                        + (idx % (2 * kd)) // 2
                    )
                    ok = (i < S) and (col < (bg + 1) * span) and (col < COLS)
                    x_s[i, j] = T.if_then_else(
                        ok,
                        T.cast(x[tb * sxb + tlp * sxl + oo * sxa], "float32"),
                        T.cast(0, "float32"),
                    )
                    g_s[i, j] = T.if_then_else(
                        ok,
                        T.cast(g[tb * sxb + tlp * sxl + oo * sxa], "float32"),
                        T.cast(0, "float32"),
                    )
                T.clear(gn_f)
                T.gemm(rf_s, g_s, gn_f, transpose_A=True)
                for i, j in T.Parallel(ps, bn):
                    col = bg * span + c0 * bn + j
                    tb = col // L
                    tlp = col % L
                    idx = bk * S + i
                    oo = (
                        (idx // (2 * kd)) * 2 * kd
                        + (idx % 2) * kd
                        + (idx % (2 * kd)) // 2
                    )
                    if (i < S) and (col < (bg + 1) * span) and (col < COLS):
                        gn[tb * sxb + tlp * sxl + oo * sxa] = T.cast(gn_f[i, j], dtype)
                T.gemm(g_s, x_s, grf_f, transpose_B=True)

            T.copy(grf_f, grf_s)
            # gR = gRf*scale; gQ = (I+R)^T gR M^T; gB = (gQ - gQ^T)*cscale.
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
                    acc += lhs * grf_s[kk, j] * scale
                tmp_s[i, j] = acc
            for i, j in T.Parallel(ps, ps):
                acc = T.alloc_var("float32")
                acc = T.cast(0, "float32")
                for kk in T.serial(ps):
                    acc += tmp_s[i, kk] * m_s[j, kk]
                rf_s[i, j] = acc
            for i, j in T.Parallel(ps, ps):
                if (i < S) and (j < S):
                    T.atomic_add(gb[bk, i, j], (rf_s[i, j] - rf_s[j, i]) * cscale)

    return main
