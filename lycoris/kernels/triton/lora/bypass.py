"""LoCon/LoRA bypass as single fused kernels (linear layout).

Forward: one kernel, 1D grid over token tiles — each CTA computes
h = x@down^T once (i-loop), then streams the o-loop emitting
y = gamma*h@up^T. The 2-launch chain's h round-trip and second dispatch
disappear; factor rereads stay in L2 (design/loop4-grid-math.md).

Backward: one kernel, same grid — h is rebuilt in-registers (i-loop),
q = g@up accumulates while g_up partials are atomically added (o-loop),
then a second i-loop stores gx = gamma*q@down and atomics g_down partials.
Intermediates h and q are rounded to the storage dtype before their second
dot, which is exactly the rounding the eager two-Linear chain applies.
"""

import torch
import triton
import triton.language as tl

from ...gradbuf import GradPack
from ...plans import lora as plan
from ...plans import tune
from ...plans.cost import SENTINEL_EAGER
from ...plans.device import resolve_device
from ..common import rank_block


@triton.jit
def _bypass_fwd_kernel(
    x_ptr,
    d_ptr,
    u_ptr,
    out_ptr,
    T,
    O,
    I,
    R,
    sxt,
    sxi,
    sdr,
    sdi,
    suo,
    sur,
    sot,
    soo,
    gamma,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y = gamma * (x @ down^T) @ up^T, one CTA per token tile.

    1D over tokens: h is computed once and reused across the whole O axis, so
    x is read exactly once. Splitting O for more programs was measured worse —
    it re-reads x O/BN times, and at these sizes that traffic outweighs the
    parallelism (33.5 us against 10.9 us at T=512, I=O=1280, r=16). BM is the
    only parallelism lever, so it stays small.
    """
    pid_t = tl.program_id(0)
    rt = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
    rr = tl.arange(0, BR)
    mt = rt < T
    mr = rr < R
    # h = x @ down^T for this token tile (i-loop over BLOCK_K).
    h = tl.zeros((BLOCK_M, BR), tl.float32)
    for i0 in range(0, I, BLOCK_K):
        ri = i0 + tl.arange(0, BLOCK_K)
        mi = ri < I
        xv = tl.load(
            x_ptr + rt[:, None] * sxt + ri[None, :] * sxi,
            mask=mt[:, None] & mi[None, :],
            other=0.0,
        )
        dv = tl.load(
            d_ptr + rr[:, None] * sdr + ri[None, :] * sdi,
            mask=mr[:, None] & mi[None, :],
            other=0.0,
        )
        h = tl.dot(xv, tl.trans(dv), h, input_precision=PREC)
    hs = (h * gamma).to(x_ptr.dtype.element_ty)
    # y-tile = h @ up^T, streamed over the O axis in BLOCK_N chunks.
    for o0 in range(0, O, BLOCK_N):
        ro = o0 + tl.arange(0, BLOCK_N)
        mo = ro < O
        uv = tl.load(
            u_ptr + ro[:, None] * suo + rr[None, :] * sur,
            mask=mo[:, None] & mr[None, :],
            other=0.0,
        )
        y = tl.dot(hs, tl.trans(uv), input_precision=PREC)
        tl.store(
            out_ptr + rt[:, None] * sot + ro[None, :] * soo,
            y.to(out_ptr.dtype.element_ty),
            mask=mt[:, None] & mo[None, :],
        )


@triton.jit
def _bypass_bwd_kernel(
    x_ptr,
    d_ptr,
    u_ptr,
    g_ptr,
    gx_ptr,
    gu_ptr,
    gd_ptr,
    T,
    O,
    I,
    R,
    sxt,
    sxi,
    sdr,
    sdi,
    suo,
    sur,
    sgt,
    sgo,
    gamma,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """gx, g_up, g_down of y = gamma*(x@down^T)@up^T for one token tile."""
    pid = tl.program_id(0)
    rt = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rr = tl.arange(0, BR)
    mt = rt < T
    mr = rr < R
    # Pass 1: rebuild h = x @ down^T (never cached — it is T x r of traffic).
    h = tl.zeros((BLOCK_M, BR), tl.float32)
    for i0 in range(0, I, BLOCK_K):
        ri = i0 + tl.arange(0, BLOCK_K)
        mi = ri < I
        xv = tl.load(
            x_ptr + rt[:, None] * sxt + ri[None, :] * sxi,
            mask=mt[:, None] & mi[None, :],
            other=0.0,
        )
        dv = tl.load(
            d_ptr + rr[:, None] * sdr + ri[None, :] * sdi,
            mask=mr[:, None] & mi[None, :],
            other=0.0,
        )
        h = tl.dot(xv, tl.trans(dv), h, input_precision=PREC)
    hs = h.to(x_ptr.dtype.element_ty)
    # Pass 2: q = g @ up accumulates while g_up += gamma * g^T @ h.
    q = tl.zeros((BLOCK_M, BR), tl.float32)
    for o0 in range(0, O, BLOCK_N):
        ro = o0 + tl.arange(0, BLOCK_N)
        mo = ro < O
        gv = tl.load(
            g_ptr + rt[:, None] * sgt + ro[None, :] * sgo,
            mask=mt[:, None] & mo[None, :],
            other=0.0,
        )
        uv = tl.load(
            u_ptr + ro[:, None] * suo + rr[None, :] * sur,
            mask=mo[:, None] & mr[None, :],
            other=0.0,
        )
        q = tl.dot(gv, uv, q, input_precision=PREC)
        gu = tl.dot(tl.trans(gv), hs, input_precision=PREC) * gamma
        tl.atomic_add(
            gu_ptr + ro[:, None] * R + rr[None, :],
            gu,
            mask=mo[:, None] & mr[None, :],
        )
    # Pass 3: gx = gamma * q @ down, and g_down += gamma * q^T @ x.
    qs = q.to(x_ptr.dtype.element_ty)
    for i0 in range(0, I, BLOCK_K):
        ri = i0 + tl.arange(0, BLOCK_K)
        mi = ri < I
        dv = tl.load(
            d_ptr + rr[:, None] * sdr + ri[None, :] * sdi,
            mask=mr[:, None] & mi[None, :],
            other=0.0,
        )
        gxv = tl.dot(qs, dv, input_precision=PREC) * gamma
        tl.store(
            gx_ptr + rt[:, None] * sxt + ri[None, :] * sxi,
            gxv.to(gx_ptr.dtype.element_ty),
            mask=mt[:, None] & mi[None, :],
        )
        xv = tl.load(
            x_ptr + rt[:, None] * sxt + ri[None, :] * sxi,
            mask=mt[:, None] & mi[None, :],
            other=0.0,
        )
        gd = tl.dot(tl.trans(qs), xv, input_precision=PREC) * gamma
        tl.atomic_add(
            gd_ptr + rr[:, None] * I + ri[None, :],
            gd,
            mask=mr[:, None] & mi[None, :],
        )


@triton.jit
def _merge_bwd_kernel(
    g_ptr,
    u_ptr,
    d_ptr,
    gu_ptr,
    gd_ptr,
    O,
    I,
    R,
    sgo,
    sgi,
    suo,
    sur,
    sdr,
    sdi,
    gamma,
    GA,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Role-split: pids [0, GA) reduce g_up rows, the rest g_down columns.

    Both roles hold the full reduction in registers — no atomics, direct
    dtype stores with a single rounding.
    """
    pid = tl.program_id(0)
    rr = tl.arange(0, BR)
    mr = rr < R
    if pid < GA:
        # Role A: g_up = gamma * G @ down^T, reduced over the I axis.
        rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mm = rm < O
        acc = tl.zeros((BLOCK_M, BR), tl.float32)
        for i0 in range(0, I, BLOCK_K):
            ri = i0 + tl.arange(0, BLOCK_K)
            mi = ri < I
            gv = tl.load(
                g_ptr + rm[:, None] * sgo + ri[None, :] * sgi,
                mask=mm[:, None] & mi[None, :],
                other=0.0,
            )
            dv = tl.load(
                d_ptr + rr[:, None] * sdr + ri[None, :] * sdi,
                mask=mr[:, None] & mi[None, :],
                other=0.0,
            )
            acc = tl.dot(gv, tl.trans(dv), acc, input_precision=PREC)
        tl.store(
            gu_ptr + rm[:, None] * R + rr[None, :],
            (acc * gamma).to(gu_ptr.dtype.element_ty),
            mask=mm[:, None] & mr[None, :],
        )
    else:
        # Role B: g_down = gamma * up^T @ G, reduced over the O axis.
        rn = (pid - GA) * BLOCK_N + tl.arange(0, BLOCK_N)
        mn = rn < I
        acc = tl.zeros((BR, BLOCK_N), tl.float32)
        for o0 in range(0, O, BLOCK_K):
            ro = o0 + tl.arange(0, BLOCK_K)
            mo = ro < O
            gv = tl.load(
                g_ptr + ro[:, None] * sgo + rn[None, :] * sgi,
                mask=mo[:, None] & mn[None, :],
                other=0.0,
            )
            uv = tl.load(
                u_ptr + ro[:, None] * suo + rr[None, :] * sur,
                mask=mo[:, None] & mr[None, :],
                other=0.0,
            )
            acc = tl.dot(tl.trans(uv), gv, acc, input_precision=PREC)
        tl.store(
            gd_ptr + rr[:, None] * I + rn[None, :],
            (acc * gamma).to(gd_ptr.dtype.element_ty),
            mask=mr[:, None] & mn[None, :],
        )


def _ieee(*tensors: torch.Tensor) -> bool:
    return any(t is not None and t.dtype == torch.float32 for t in tensors)


def lora_bypass_fwd(x, up, down, gamma: float = 1.0) -> torch.Tensor:
    """y = gamma * (x @ down^T) @ up^T, one kernel; x is (T, I) contiguous-ish."""
    t, i = x.shape
    o, r = up.shape
    out = torch.empty(t, o, device=x.device, dtype=x.dtype)
    eb = x.element_size()

    def launch(p, dst):
        _bypass_fwd_kernel[(triton.cdiv(t, p.bm),)](
            x,
            down,
            up,
            dst,
            t,
            o,
            i,
            r,
            *x.stride(),
            *down.stride(),
            *up.stride(),
            *dst.stride(),
            gamma,
            PREC="ieee" if _ieee(x) else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=p.bk,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def eager_run():
        return (x @ down.transpose(0, 1)) @ up.transpose(0, 1) * gamma

    shortlist = lambda: [
        *plan.topk_bypass_fwd(t, o, i, r, eb, resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        if p.limiter == "eager":
            return eager_run
        return lambda: launch(p, out)

    best = tune.tuned(
        "triton.lora.bypass_fwd",
        (tune.bucket_tokens(t), o, i, r, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "eager":
        return eager_run()
    launch(best, out)
    return out


def lora_bypass_bwd(x, up, down, grad, gamma: float = 1.0):
    """(gx, g_up, g_down) of the bypass chain in one kernel launch."""
    t, i = x.shape
    o, r = up.shape
    gx = torch.empty_like(x)
    # One fp32 allocation for both atomic targets: one zero-fill, one cast.
    pack = GradPack(x.device, (o, r), (r, i))
    gu, gd = pack
    eb = x.element_size()

    def launch(p, o_gx, o_gu, o_gd):
        _bypass_bwd_kernel[(triton.cdiv(t, p.bm),)](
            x,
            down,
            up,
            grad,
            o_gx,
            o_gu,
            o_gd,
            t,
            o,
            i,
            r,
            *x.stride(),
            *down.stride(),
            *up.stride(),
            *grad.stride(),
            gamma,
            PREC="ieee" if _ieee(x, grad) else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=p.bk,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def eager_run():
        h = x @ down.transpose(0, 1)
        q = grad @ up
        return (
            q @ down * gamma,
            grad.transpose(0, 1) @ h * gamma,
            q.transpose(0, 1) @ x * gamma,
        )

    shortlist = lambda: [
        *plan.topk_bypass_bwd(t, o, i, r, eb, resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        if p.limiter == "eager":
            return eager_run
        scratch = pack.like()
        s1 = torch.empty_like(gx)
        return lambda: launch(p, s1, *scratch)

    best = tune.tuned(
        "triton.lora.bypass_bwd",
        (tune.bucket_tokens(t), o, i, r, str(x.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "eager":
        return eager_run()
    launch(best, gx, gu, gd)
    g_up, g_down = pack.to(up.dtype)
    return gx, g_up, g_down


def lora_merge_bwd(grad, up, down, gamma: float = 1.0):
    """(g_up, g_down) of DeltaW = gamma*up@down in one role-split launch."""
    o, r = up.shape
    i = down.shape[1]
    gu = torch.empty(o, r, device=up.device, dtype=up.dtype)
    gd = torch.empty(r, i, device=up.device, dtype=down.dtype)
    eb = up.element_size()

    def launch(p, o_gu, o_gd):
        ga = triton.cdiv(o, p.bm)
        _merge_bwd_kernel[(ga + triton.cdiv(i, p.bn),)](
            grad,
            up,
            down,
            o_gu,
            o_gd,
            o,
            i,
            r,
            *grad.stride(),
            *up.stride(),
            *down.stride(),
            gamma,
            ga,
            PREC="ieee" if _ieee(up, grad) else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=p.bk,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def eager_run():
        g = grad * gamma
        return g @ down.transpose(0, 1), up.transpose(0, 1) @ g

    shortlist = lambda: [
        *plan.topk_merge_bwd(o, i, r, eb, resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        if p.limiter == "eager":
            return eager_run
        s1, s2 = torch.empty_like(gu), torch.empty_like(gd)
        return lambda: launch(p, s1, s2)

    best = tune.tuned(
        "triton.lora.merge_bwd",
        (o, i, r, str(up.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "eager":
        return eager_run()
    launch(best, gu, gd)
    return gu, gd
