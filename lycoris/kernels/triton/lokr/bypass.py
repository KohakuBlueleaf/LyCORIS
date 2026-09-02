"""F4: grouped Kronecker bypass apply (LoKr, linear layout).

Per token X = reshape(x, (b, d)); y = vec(w1 @ (X @ w2^T)) computed as two
small dots on stacked token tiles entirely in registers — no transpose
copies, no (.., b, c) intermediates in memory. Backward recomputes the mid
products and accumulates factor grads atomically in fp32.
Conv-spatial LoKr bypass stays on the torch path in v1 (see progress log).
"""

import torch
import triton
import triton.language as tl

from ...gradbuf import GradPack
from ...plans import lokr as plan
from ...plans import tune
from ...plans.device import resolve_device


@triton.jit
def _lokr_bypass_fwd_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    y_ptr,
    T,
    A,
    B,
    C,
    D,
    sxt,
    sxi,
    s1a,
    s1b,
    s2c,
    s2d,
    syt,
    syo,
    gamma,
    PREC: tl.constexpr,
    BT: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    PC: tl.constexpr,
    PD: tl.constexpr,
):
    pid = tl.program_id(0)

    rows = tl.arange(0, BT * PB)
    ti = rows // PB
    bi = rows % PB
    tok = pid * BT + ti
    rd = tl.arange(0, PD)
    xmask = (tok < T)[:, None] & (bi < B)[:, None] & (rd < D)[None, :]
    x2 = tl.load(
        x_ptr + tok[:, None] * sxt + (bi[:, None] * D + rd[None, :]) * sxi,
        mask=xmask,
        other=0.0,
    )

    rc = tl.arange(0, PC)
    w2t = tl.load(
        w2_ptr + rc[None, :] * s2c + rd[:, None] * s2d,
        mask=(rc < C)[None, :] & (rd < D)[:, None],
        other=0.0,
    )
    m = tl.dot(x2, w2t, input_precision=PREC)
    m2 = tl.reshape(tl.permute(tl.reshape(m, (BT, PB, PC)), (0, 2, 1)), (BT * PC, PB))

    ra = tl.arange(0, PA)
    rb = tl.arange(0, PB)
    w1t = tl.load(
        w1_ptr + ra[None, :] * s1a + rb[:, None] * s1b,
        mask=(ra < A)[None, :] & (rb < B)[:, None],
        other=0.0,
    )
    y2 = tl.dot(m2.to(w1t.dtype), w1t, input_precision=PREC)

    rows2 = tl.arange(0, BT * PC)
    ti2 = rows2 // PC
    ci = rows2 % PC
    tok2 = pid * BT + ti2
    ymask = (tok2 < T)[:, None] & (ci < C)[:, None] & (ra < A)[None, :]
    tl.store(
        y_ptr + tok2[:, None] * syt + (ra[None, :] * C + ci[:, None]) * syo,
        (y2 * gamma).to(y_ptr.dtype.element_ty),
        mask=ymask,
    )


@triton.jit
def _lokr_bypass_bwd_kernel(
    g_ptr,
    x_ptr,
    w1_ptr,
    w2_ptr,
    gx_ptr,
    gw1_ptr,
    gw2_ptr,
    T,
    A,
    B,
    C,
    D,
    sgt,
    sgo,
    sxt,
    sxi,
    s1a,
    s1b,
    s2c,
    s2d,
    gamma,
    PREC: tl.constexpr,
    BT: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    PC: tl.constexpr,
    PD: tl.constexpr,
):
    pid = tl.program_id(0)

    rows2 = tl.arange(0, BT * PC)
    ti2 = rows2 // PC
    ci = rows2 % PC
    tok2 = pid * BT + ti2
    ra = tl.arange(0, PA)
    gmask = (tok2 < T)[:, None] & (ci < C)[:, None] & (ra < A)[None, :]
    g2 = tl.load(
        g_ptr + tok2[:, None] * sgt + (ra[None, :] * C + ci[:, None]) * sgo,
        mask=gmask,
        other=0.0,
    )

    rb = tl.arange(0, PB)
    w1 = tl.load(
        w1_ptr + ra[:, None] * s1a + rb[None, :] * s1b,
        mask=(ra < A)[:, None] & (rb < B)[None, :],
        other=0.0,
    )
    n2 = tl.dot(g2, w1, input_precision=PREC)
    np2 = tl.reshape(tl.permute(tl.reshape(n2, (BT, PC, PB)), (0, 2, 1)), (BT * PB, PC))

    rc = tl.arange(0, PC)
    rd = tl.arange(0, PD)
    w2 = tl.load(
        w2_ptr + rc[:, None] * s2c + rd[None, :] * s2d,
        mask=(rc < C)[:, None] & (rd < D)[None, :],
        other=0.0,
    )
    gx2 = tl.dot(np2.to(w2.dtype), w2, input_precision=PREC)

    rows = tl.arange(0, BT * PB)
    ti = rows // PB
    bi = rows % PB
    tok = pid * BT + ti
    xmask = (tok < T)[:, None] & (bi < B)[:, None] & (rd < D)[None, :]
    tl.store(
        gx_ptr + tok[:, None] * sxt + (bi[:, None] * D + rd[None, :]) * sxi,
        (gx2 * gamma).to(gx_ptr.dtype.element_ty),
        mask=xmask,
    )

    x2 = tl.load(
        x_ptr + tok[:, None] * sxt + (bi[:, None] * D + rd[None, :]) * sxi,
        mask=xmask,
        other=0.0,
    )
    w2t = tl.trans(w2)
    m = tl.dot(x2, w2t.to(x2.dtype), input_precision=PREC)
    mp = tl.reshape(tl.permute(tl.reshape(m, (BT, PB, PC)), (0, 2, 1)), (BT * PC, PB))
    gw1 = tl.dot(tl.trans(g2), mp.to(g2.dtype), input_precision=PREC)
    tl.atomic_add(
        gw1_ptr + ra[:, None] * B + rb[None, :],
        gw1 * gamma,
        mask=(ra < A)[:, None] & (rb < B)[None, :],
    )
    gw2 = tl.dot(tl.trans(np2.to(x2.dtype)), x2, input_precision=PREC)
    tl.atomic_add(
        gw2_ptr + rc[:, None] * D + rd[None, :],
        gw2 * gamma,
        mask=(rc < C)[:, None] & (rd < D)[None, :],
    )


#: The apply kernel holds every factor tile in registers, so each factor dim
#: may pad to a power of two no larger than this.
MAX_KRON_FACTOR = 128


def kron_factors_supported(a: int, b: int, c: int, d: int) -> bool:
    """Whether factor dims (a, b, c, d) fit the apply kernel's register tiles.

    This is the dispatcher's scope test for the fused apply: oversized factors
    (typical LoKr, whose second factor spans out/factor by in/factor) step
    down a backend tier instead of raising at launch time.
    """
    return max(a, b, c, d) <= MAX_KRON_FACTOR


def _pads(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    pads = tuple(max(16, 1 << (max(1, v) - 1).bit_length()) for v in (a, b, c, d))
    if max(pads) > MAX_KRON_FACTOR:
        raise ValueError(f"kron apply factors too large for the kernel: {pads}")
    return pads


def lokr_bypass_fwd(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """y(T, A*C) = gamma * vec(w1 @ reshape(x_t) @ w2^T) per token."""
    t = x.shape[0]
    a, b = w1.shape
    c, d = w2.shape
    pa, pb, pc, pd = _pads(a, b, c, d)
    y = torch.empty(t, a * c, device=x.device, dtype=x.dtype)

    def launch(p, dst):
        _lokr_bypass_fwd_kernel[(triton.cdiv(t, p.bm),)](
            x,
            w1,
            w2,
            dst,
            t,
            a,
            b,
            c,
            d,
            *x.stride(),
            *w1.stride(),
            *w2.stride(),
            *dst.stride(),
            gamma,
            PREC="ieee" if x.dtype == torch.float32 else "tf32",
            BT=p.bm,
            PA=pa,
            PB=pb,
            PC=pc,
            PD=pd,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    shortlist = lambda: plan.topk_apply(
        t, a, b, c, d, x.element_size(), resolve_device()
    )
    best = tune.tuned(
        "triton.lokr.bypass_fwd",
        (tune.bucket_tokens(t), a, b, c, d, str(x.dtype)),
        shortlist,
        lambda p: (lambda: launch(p, y)),
    )
    launch(best, y)
    return y


def lokr_bypass_bwd(
    grad: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = x.shape[0]
    a, b = w1.shape
    c, d = w2.shape
    pa, pb, pc, pd = _pads(a, b, c, d)
    gx = torch.empty_like(x)
    # One fp32 allocation for both atomic targets: one zero-fill, one cast.
    pack = GradPack(x.device, (a, b), (c, d))
    gw1, gw2 = pack

    def launch(p, o0, o1, o2):
        _lokr_bypass_bwd_kernel[(triton.cdiv(t, p.bm),)](
            grad,
            x,
            w1,
            w2,
            o0,
            o1,
            o2,
            t,
            a,
            b,
            c,
            d,
            *grad.stride(),
            *x.stride(),
            *w1.stride(),
            *w2.stride(),
            gamma,
            PREC="ieee" if x.dtype == torch.float32 else "tf32",
            BT=p.bm,
            PA=pa,
            PB=pb,
            PC=pc,
            PD=pd,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        s0 = torch.empty_like(gx)
        scratch = pack.like()
        return lambda: launch(p, s0, *scratch)

    shortlist = lambda: plan.topk_apply(
        t, a, b, c, d, x.element_size(), resolve_device()
    )
    best = tune.tuned(
        "triton.lokr.bypass_bwd",
        (tune.bucket_tokens(t), a, b, c, d, str(x.dtype)),
        shortlist,
        factory,
    )
    launch(best, gx, gw1, gw2)
    return (gx, *pack.to(w1.dtype))
