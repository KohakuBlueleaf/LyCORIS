"""F7: broadcast channel scale (IA3 and friends).

Data is viewed (outer, C, inner) contiguous; y = x * (alpha + gamma * w[c]).
alpha=1 gives the merged/bypass form, alpha=0 the diff form. The backward
fuses gx with an atomic fp32 channel reduction for gw.
"""

import torch
import triton
import triton.language as tl

from ...plans import dora as plan
from ...plans import tune
from ...plans.device import resolve_device


@triton.jit
def _scale_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    C,
    INNER,
    alpha,
    gamma,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < N
    c = (idx // INNER) % C
    x = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + idx, (x * (alpha + gamma * w)).to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _scale_bwd_kernel(
    g_ptr,
    x_ptr,
    w_ptr,
    gx_ptr,
    gw_ptr,
    N,
    C,
    INNER,
    alpha,
    gamma,
    BLOCK: tl.constexpr,
    SLICES: tl.constexpr,
):
    """gw partials land in (SLICES, C) fp32 (contention / SLICES); the
    wrapper sums them."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < N
    c = (idx // INNER) % C
    g = tl.load(g_ptr + idx, mask=m, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32)
    tl.store(
        gx_ptr + idx, (g * (alpha + gamma * w)).to(gx_ptr.dtype.element_ty), mask=m
    )
    tl.atomic_add(gw_ptr + (pid % SLICES) * C + c, g * x * gamma, mask=m)


def _view3(x: torch.Tensor, channel_axis: int) -> tuple[torch.Tensor, int, int]:
    xc = x.contiguous()
    if channel_axis in (-1, x.dim() - 1):
        return xc, x.shape[-1], 1
    if channel_axis == 0:
        inner = xc.numel() // x.shape[0]
        return xc, x.shape[0], inner
    if channel_axis == 1:
        inner = 1
        for d in x.shape[2:]:
            inner *= d
        return xc, x.shape[1], inner
    raise ValueError("channel_axis must be 0, 1 or -1")


def ia3_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    channel_axis: int,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    xc, c, inner = _view3(x, channel_axis)
    y = torch.empty_like(xc)
    n = xc.numel()
    wc = w.reshape(-1).contiguous()

    def launch(p, dst):
        _scale_fwd_kernel[(triton.cdiv(n, p.bm),)](
            xc,
            wc,
            dst,
            n,
            c,
            inner,
            alpha,
            gamma,
            BLOCK=p.bm,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    best = tune.tuned(
        "triton.ia3.fwd",
        (tune.bucket_tokens(n), c, inner, str(x.dtype)),
        lambda: plan.topk_elementwise(n, 2.0, resolve_device(), x.element_size()),
        lambda p: (lambda: launch(p, y)),
    )
    launch(best, y)
    return y.view_as(x)


def ia3_bwd(
    grad: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    channel_axis: int,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    xc, c, inner = _view3(x, channel_axis)
    gc, _, _ = _view3(grad, channel_axis)
    gx = torch.empty_like(xc)
    slices = 32
    gw = torch.zeros(slices, c, device=x.device, dtype=torch.float32)
    n = xc.numel()
    wc = w.reshape(-1).contiguous()

    def launch(p, o1, o2):
        _scale_bwd_kernel[(triton.cdiv(n, p.bm),)](
            gc,
            xc,
            wc,
            o1,
            o2,
            n,
            c,
            inner,
            alpha,
            gamma,
            BLOCK=p.bm,
            SLICES=slices,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        s1, s2 = torch.empty_like(gx), torch.empty_like(gw)
        return lambda: launch(p, s1, s2)

    best = tune.tuned(
        "triton.ia3.fwd_bwd",
        (tune.bucket_tokens(n), c, inner, str(x.dtype)),
        lambda: plan.topk_elementwise(n, 3.0, resolve_device(), x.element_size()),
        factory,
    )
    launch(best, gx, gw)
    return gx.view_as(x), gw.sum(0).to(w.dtype).view_as(w)
