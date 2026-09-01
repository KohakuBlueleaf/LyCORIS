"""F2: generated-B delta apply for hadamard products (LoHa bypass).

y = gamma * x @ DeltaW^T with DeltaW = (a1@b1)*(a2@b2) generated tile-wise in
registers — DeltaW never exists in memory, forward or backward. Plain
low-rank never needs this (its chain factors through (T, r) intermediates);
see design/families/lowrank.md.
"""

import torch
import triton
import triton.language as tl

from ...gradbuf import GradPack
from ...plans import lora as plan
from ...plans import tune
from ...plans.cost import SENTINEL_MATERIALIZE
from ...plans.device import resolve_device
from ..common import rank_block
from ..lora.merge import loha_merge_bwd, lora_merge_fwd


@triton.jit
def _gen_w_tile(
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    ro,
    ri,
    mo,
    mi,
    R,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    PREC: tl.constexpr,
    BR: tl.constexpr,
):
    """DeltaW[ro, ri] tile: (a1@b1)*(a2@b2) rows=out, cols=in."""
    rr = tl.arange(0, BR)
    mr = rr < R
    a1 = tl.load(
        a1_ptr + ro[:, None] * sa1o + rr[None, :] * sa1r,
        mask=mo[:, None] & mr[None, :],
        other=0.0,
    )
    b1 = tl.load(
        b1_ptr + rr[:, None] * sb1r + ri[None, :] * sb1i,
        mask=mr[:, None] & mi[None, :],
        other=0.0,
    )
    a2 = tl.load(
        a2_ptr + ro[:, None] * sa2o + rr[None, :] * sa2r,
        mask=mo[:, None] & mr[None, :],
        other=0.0,
    )
    b2 = tl.load(
        b2_ptr + rr[:, None] * sb2r + ri[None, :] * sb2i,
        mask=mr[:, None] & mi[None, :],
        other=0.0,
    )
    p1 = tl.dot(a1, b1, input_precision=PREC)
    p2 = tl.dot(a2, b2, input_precision=PREC)
    return p1 * p2


@triton.jit
def _loha_bypass_fwd_kernel(
    x_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    y_ptr,
    T,
    O,
    I,
    R,
    sxt,
    sxi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    syt,
    syo,
    gamma,
    TRANS_W: tl.constexpr,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y[t, o] = gamma * sum_i x[t, i] * W[o, i]  (TRANS_W=False: sum_o g W[o,i])."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    rt = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mt = rt < T
    mn = rn < O
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, I, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mk = rk < I
        x = tl.load(
            x_ptr + rt[:, None] * sxt + rk[None, :] * sxi,
            mask=mt[:, None] & mk[None, :],
            other=0.0,
        )
        if TRANS_W:
            w = _gen_w_tile(
                a1_ptr,
                b1_ptr,
                a2_ptr,
                b2_ptr,
                rn,
                rk,
                mn,
                mk,
                R,
                sa1o,
                sa1r,
                sb1r,
                sb1i,
                sa2o,
                sa2r,
                sb2r,
                sb2i,
                PREC,
                BR,
            )
            w = tl.trans(w)
        else:
            w = _gen_w_tile(
                a1_ptr,
                b1_ptr,
                a2_ptr,
                b2_ptr,
                rk,
                rn,
                mk,
                mn,
                R,
                sa1o,
                sa1r,
                sb1r,
                sb1i,
                sa2o,
                sa2r,
                sb2r,
                sb2i,
                PREC,
                BR,
            )
        acc = tl.dot(x, w.to(x.dtype), acc, input_precision=PREC)
    omask = mt[:, None] & mn[None, :]
    tl.store(
        y_ptr + rt[:, None] * syt + rn[None, :] * syo,
        (acc * gamma).to(y_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _loha_bypass_bwd_dw_kernel(
    g_ptr,
    x_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    ga1_ptr,
    gb1_ptr,
    ga2_ptr,
    gb2_ptr,
    T,
    O,
    I,
    R,
    sgt,
    sgo,
    sxt,
    sxi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    gamma,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Factor grads via on-the-fly gDeltaW = g^T @ x, contracted immediately."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ro = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ri = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, BR)
    mo = ro < O
    mi = ri < I
    mr = rr < R
    gw = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for t0 in range(0, T, BLOCK_K):
        rt = t0 + tl.arange(0, BLOCK_K)
        mt = rt < T
        g = tl.load(
            g_ptr + rt[:, None] * sgt + ro[None, :] * sgo,
            mask=mt[:, None] & mo[None, :],
            other=0.0,
        )
        x = tl.load(
            x_ptr + rt[:, None] * sxt + ri[None, :] * sxi,
            mask=mt[:, None] & mi[None, :],
            other=0.0,
        )
        gw = tl.dot(tl.trans(g), x, gw, input_precision=PREC)
    gw *= gamma

    a1 = tl.load(
        a1_ptr + ro[:, None] * sa1o + rr[None, :] * sa1r,
        mask=mo[:, None] & mr[None, :],
        other=0.0,
    )
    b1 = tl.load(
        b1_ptr + rr[:, None] * sb1r + ri[None, :] * sb1i,
        mask=mr[:, None] & mi[None, :],
        other=0.0,
    )
    a2 = tl.load(
        a2_ptr + ro[:, None] * sa2o + rr[None, :] * sa2r,
        mask=mo[:, None] & mr[None, :],
        other=0.0,
    )
    b2 = tl.load(
        b2_ptr + rr[:, None] * sb2r + ri[None, :] * sb2i,
        mask=mr[:, None] & mi[None, :],
        other=0.0,
    )
    p1 = tl.dot(a1, b1, input_precision=PREC)
    p2 = tl.dot(a2, b2, input_precision=PREC)
    e1 = (gw * p2).to(b1.dtype)
    e2 = (gw * p1).to(b2.dtype)
    amask = mo[:, None] & mr[None, :]
    bmask = mr[:, None] & mi[None, :]
    tl.atomic_add(
        ga1_ptr + ro[:, None] * R + rr[None, :],
        tl.dot(e1, tl.trans(b1), input_precision=PREC),
        mask=amask,
    )
    tl.atomic_add(
        gb1_ptr + rr[:, None] * I + ri[None, :],
        tl.dot(tl.trans(a1), e1, input_precision=PREC),
        mask=bmask,
    )
    tl.atomic_add(
        ga2_ptr + ro[:, None] * R + rr[None, :],
        tl.dot(e2, tl.trans(b2), input_precision=PREC),
        mask=amask,
    )
    tl.atomic_add(
        gb2_ptr + rr[:, None] * I + ri[None, :],
        tl.dot(tl.trans(a2), e2, input_precision=PREC),
        mask=bmask,
    )


@triton.jit
def _loha_bypass_bwd_kernel(
    g_ptr,
    x_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    gx_ptr,
    ga1_ptr,
    gb1_ptr,
    ga2_ptr,
    gb2_ptr,
    T,
    O,
    I,
    R,
    sgt,
    sgo,
    sxt,
    sxi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    sgxt,
    sgxi,
    gamma,
    GA,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One launch for the whole bypass backward, role-split on the linear pid:
    pids [0, GA) are dw tiles, the rest are dx tiles. gDeltaW never exists.
    """
    pid = tl.program_id(0)
    nin = tl.cdiv(I, BLOCK_N)
    rr = tl.arange(0, BR)
    mr = rr < R
    if pid < GA:
        # dw role over an (O, I) tile: gW = gamma * sum_t g[t,o] x[t,i].
        pm = pid // nin
        pn = pid % nin
        ro = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        ri = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mo = ro < O
        mi = ri < I
        gw = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for t0 in range(0, T, BLOCK_K):
            rt = t0 + tl.arange(0, BLOCK_K)
            mt = rt < T
            g = tl.load(
                g_ptr + rt[:, None] * sgt + ro[None, :] * sgo,
                mask=mt[:, None] & mo[None, :],
                other=0.0,
            )
            x = tl.load(
                x_ptr + rt[:, None] * sxt + ri[None, :] * sxi,
                mask=mt[:, None] & mi[None, :],
                other=0.0,
            )
            gw = tl.dot(tl.trans(g), x, gw, input_precision=PREC)
        gw *= gamma
        # e1 = gW * (a2@b2), e2 = gW * (a1@b1)  (hadamard chain rule).
        a1 = tl.load(
            a1_ptr + ro[:, None] * sa1o + rr[None, :] * sa1r,
            mask=mo[:, None] & mr[None, :],
            other=0.0,
        )
        b1 = tl.load(
            b1_ptr + rr[:, None] * sb1r + ri[None, :] * sb1i,
            mask=mr[:, None] & mi[None, :],
            other=0.0,
        )
        a2 = tl.load(
            a2_ptr + ro[:, None] * sa2o + rr[None, :] * sa2r,
            mask=mo[:, None] & mr[None, :],
            other=0.0,
        )
        b2 = tl.load(
            b2_ptr + rr[:, None] * sb2r + ri[None, :] * sb2i,
            mask=mr[:, None] & mi[None, :],
            other=0.0,
        )
        p1 = tl.dot(a1, b1, input_precision=PREC)
        p2 = tl.dot(a2, b2, input_precision=PREC)
        e1 = (gw * p2).to(b1.dtype)
        e2 = (gw * p1).to(b2.dtype)
        amask = mo[:, None] & mr[None, :]
        bmask = mr[:, None] & mi[None, :]
        # ga1 += e1@b1^T, gb1 += a1^T@e1 (and the 2-side twins), atomic fp32.
        tl.atomic_add(
            ga1_ptr + ro[:, None] * R + rr[None, :],
            tl.dot(e1, tl.trans(b1), input_precision=PREC),
            mask=amask,
        )
        tl.atomic_add(
            gb1_ptr + rr[:, None] * I + ri[None, :],
            tl.dot(tl.trans(a1), e1, input_precision=PREC),
            mask=bmask,
        )
        tl.atomic_add(
            ga2_ptr + ro[:, None] * R + rr[None, :],
            tl.dot(e2, tl.trans(b2), input_precision=PREC),
            mask=amask,
        )
        tl.atomic_add(
            gb2_ptr + rr[:, None] * I + ri[None, :],
            tl.dot(tl.trans(a2), e2, input_precision=PREC),
            mask=bmask,
        )
    else:
        # dx role over a (T, I) tile: gx = gamma * g @ W, W generated per chunk.
        pid2 = pid - GA
        pm = pid2 // nin
        pn = pid2 % nin
        rt = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mt = rt < T
        mn = rn < I
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, O, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            mk = rk < O
            g = tl.load(
                g_ptr + rt[:, None] * sgt + rk[None, :] * sgo,
                mask=mt[:, None] & mk[None, :],
                other=0.0,
            )
            # W[o, i] tile = (a1@b1)*(a2@b2) at rows rk, cols rn.
            w = _gen_w_tile(
                a1_ptr,
                b1_ptr,
                a2_ptr,
                b2_ptr,
                rk,
                rn,
                mk,
                mn,
                R,
                sa1o,
                sa1r,
                sb1r,
                sb1i,
                sa2o,
                sa2r,
                sb2r,
                sb2i,
                PREC,
                BR,
            )
            acc = tl.dot(g, w.to(g.dtype), acc, input_precision=PREC)
        tl.store(
            gx_ptr + rt[:, None] * sgxt + rn[None, :] * sgxi,
            (acc * gamma).to(gx_ptr.dtype.element_ty),
            mask=mt[:, None] & mn[None, :],
        )


def loha_bypass_bwd(grad, x, a1, b1, a2, b2, gamma=1.0):
    """(gx, ga1, gb1, ga2, gb2) of the bypass chain in ONE launch."""
    t, out_o = grad.shape
    out_i = x.shape[1]
    r = a1.shape[1]
    if r > 128:
        raise ValueError("hadamard delta backward supports rank <= 128")
    gx = torch.empty_like(x)
    # One fp32 allocation for all four atomic targets: one fill, one cast.
    pack = GradPack(x.device, (out_o, r), (r, out_i), (out_o, r), (r, out_i))
    ga1, gb1, ga2, gb2 = pack

    def launch(p, o_gx, o1, o2, o3, o4):
        nin = triton.cdiv(out_i, p.bn)
        ga = triton.cdiv(out_o, p.bm) * nin
        grid = ga + triton.cdiv(t, p.bm) * nin
        _loha_bypass_bwd_kernel[(grid,)](
            grad,
            x,
            a1,
            b1,
            a2,
            b2,
            o_gx,
            o1,
            o2,
            o3,
            o4,
            t,
            out_o,
            out_i,
            r,
            *grad.stride(),
            *x.stride(),
            *a1.stride(),
            *b1.stride(),
            *a2.stride(),
            *b2.stride(),
            *o_gx.stride(),
            gamma,
            ga,
            PREC="ieee" if x.dtype == torch.float32 else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=p.bk,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def materialize_run():
        w = lora_merge_fwd(a1, b1, a2, b2, gamma=gamma, mode="hada")
        gw = grad.transpose(0, 1) @ x
        return (grad @ w, *loha_merge_bwd(gw, a1, b1, a2, b2, gamma=gamma))

    shortlist = lambda: [
        *plan.topk_hada_bypass_bwd(
            t, out_o, out_i, r, x.element_size(), resolve_device()
        ),
        SENTINEL_MATERIALIZE,
    ]

    def factory(p):
        if p.limiter == "materialize":
            return materialize_run
        s0 = torch.empty_like(gx)
        scratch = pack.like()
        return lambda: launch(p, s0, *scratch)

    best = tune.tuned(
        "triton.loha.bypass_bwd",
        (tune.bucket_tokens(t), out_o, out_i, r, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "materialize":
        return materialize_run()
    launch(best, gx, ga1, gb1, ga2, gb2)
    return (gx, *pack.to(a1.dtype))


def _launch_delta(x, a1, b1, a2, b2, gamma, out_cols, trans_w):
    t = x.shape[0]
    r = a1.shape[1]
    inner = x.shape[1]
    y = torch.empty(t, out_cols, device=x.device, dtype=x.dtype)

    def launch(p, dst):
        _loha_bypass_fwd_kernel[(triton.cdiv(t, p.bm), triton.cdiv(out_cols, p.bn))](
            x,
            a1,
            b1,
            a2,
            b2,
            dst,
            t,
            out_cols,
            inner,
            r,
            *x.stride(),
            *a1.stride(),
            *b1.stride(),
            *a2.stride(),
            *b2.stride(),
            *dst.stride(),
            gamma,
            TRANS_W=trans_w,
            PREC="ieee" if x.dtype == torch.float32 else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=p.bk,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def materialize_run():
        w = lora_merge_fwd(a1, b1, a2, b2, gamma=gamma, mode="hada")
        return x @ (w.transpose(0, 1) if trans_w else w)

    shortlist = lambda: [
        *plan.topk_delta(t, out_cols, inner, r, x.element_size(), resolve_device()),
        SENTINEL_MATERIALIZE,
    ]

    def factory(p):
        if p.limiter == "materialize":
            return materialize_run
        return lambda: launch(p, y)

    best = tune.tuned(
        "triton.loha.bypass_fwd",
        (tune.bucket_tokens(t), out_cols, inner, r, trans_w, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "materialize":
        return materialize_run()
    launch(best, y)
    return y


def loha_bypass_fwd(x, a1, b1, a2, b2, gamma=1.0):
    """y(T, O) = gamma * x(T, I) @ DeltaW(O, I)^T, DeltaW generated in-kernel."""
    return _launch_delta(x, a1, b1, a2, b2, gamma, a1.shape[0], trans_w=True)


def loha_bypass_bwd_dx(grad, a1, b1, a2, b2, gamma=1.0):
    """gx(T, I) = gamma * grad(T, O) @ DeltaW(O, I), DeltaW generated in-kernel."""
    return _launch_delta(grad, a1, b1, a2, b2, gamma, b1.shape[1], trans_w=False)


def loha_bypass_bwd_dw(grad, x, a1, b1, a2, b2, gamma=1.0):
    """All four factor grads in one pass; gDeltaW tiles never hit memory."""
    t, out_o = grad.shape
    out_i = x.shape[1]
    r = a1.shape[1]
    if r > 128:
        raise ValueError("hadamard delta backward supports rank <= 128")
    f32 = torch.float32
    ga1 = torch.zeros(out_o, r, device=x.device, dtype=f32)
    gb1 = torch.zeros(r, out_i, device=x.device, dtype=f32)
    ga2 = torch.zeros(out_o, r, device=x.device, dtype=f32)
    gb2 = torch.zeros(r, out_i, device=x.device, dtype=f32)

    def launch(p, o1, o2, o3, o4):
        _loha_bypass_bwd_dw_kernel[
            (triton.cdiv(out_o, p.bm), triton.cdiv(out_i, p.bn))
        ](
            grad,
            x,
            a1,
            b1,
            a2,
            b2,
            o1,
            o2,
            o3,
            o4,
            t,
            out_o,
            out_i,
            r,
            *grad.stride(),
            *x.stride(),
            *a1.stride(),
            *b1.stride(),
            *a2.stride(),
            *b2.stride(),
            gamma,
            PREC="ieee" if x.dtype == torch.float32 else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=p.bk,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def materialize_run():
        return loha_merge_bwd(grad.transpose(0, 1) @ x, a1, b1, a2, b2, gamma=gamma)

    def factory(p):
        if p.limiter == "materialize":
            return materialize_run
        s1, s2 = torch.empty_like(ga1), torch.empty_like(gb1)
        s3, s4 = torch.empty_like(ga2), torch.empty_like(gb2)
        return lambda: launch(p, s1, s2, s3, s4)

    shortlist = lambda: [
        *plan.topk_delta_dw(t, out_o, out_i, r, x.element_size(), resolve_device()),
        SENTINEL_MATERIALIZE,
    ]
    best = tune.tuned(
        "triton.loha.bypass_dw",
        (tune.bucket_tokens(t), out_o, out_i, r, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "materialize":
        return materialize_run()
    launch(best, ga1, gb1, ga2, gb2)
    return (
        ga1.to(a1.dtype),
        gb1.to(b1.dtype),
        ga2.to(a2.dtype),
        gb2.to(b2.dtype),
    )
