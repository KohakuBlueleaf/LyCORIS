"""LoRA: the four fused kernels (merge fwd/bwd, bypass fwd/bwd).

Merge forward is one 2D-tiled kernel; merge backward is one role-split kernel
(g_up rows ++ g_down columns, no atomics). Bypass forward and backward are
single 1D token-tiled kernels that keep h = x@down^T and q = g@up in
registers. Tucker backward recomputes via einsum on conv-sized tensors
(disclosed exception: spatial-K tucker chains are conv contractions).

Operands may mix fp16/bf16/fp32 across x and the module weights: ``promote``
puts them in one compute dtype for the kernel, and each gradient goes back in
its own leaf's dtype.
"""

import torch

from ..ops import get_ops
from ..precision import promote, restore


class LowRankRebuildFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up, down, scale, backend):
        (up_c, down_c), _, _ = promote(up, down)
        ctx.save_for_backward(up_c, down_c)
        ctx.dtypes = (up.dtype, down.dtype)
        ctx.scale = scale
        ctx.backend = backend
        return get_ops(backend).lora_merge_fwd(up_c, down_c, gamma=scale)

    @staticmethod
    def backward(ctx, grad):
        up, down = ctx.saved_tensors
        (grad_c,), _, _ = promote(grad.contiguous())
        g_up, g_down = get_ops(ctx.backend).lora_merge_bwd(
            grad_c.to(up.dtype), up, down, gamma=ctx.scale
        )
        return (*restore([g_up, g_down], ctx.dtypes), None, None)


class LowRankBypassFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up, down, scale, backend):
        (x_c, up_c, down_c), _, out_dtype = promote(x, up, down)
        ctx.save_for_backward(x_c, up_c, down_c)
        ctx.dtypes = (x.dtype, up.dtype, down.dtype)
        ctx.scale = scale
        ctx.backend = backend
        y = get_ops(backend).lora_bypass_fwd(x_c, up_c, down_c, gamma=scale)
        return y if y.dtype == out_dtype else y.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        x, up, down = ctx.saved_tensors
        gx, g_up, g_down = get_ops(ctx.backend).lora_bypass_bwd(
            x, up, down, grad.contiguous().to(x.dtype), gamma=ctx.scale
        )
        return (*restore([gx, g_up, g_down], ctx.dtypes), None, None)


class TuckerRebuildFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up, mid, down, scale, backend):
        ctx.save_for_backward(up, mid, down)
        ctx.scale = scale
        k = mid.shape[2:].numel()
        out = get_ops(backend).lora_tucker_fwd(
            up, mid.reshape(*mid.shape[:2], k), down, gamma=scale
        )
        return out.reshape(up.shape[0], down.shape[1], *mid.shape[2:])

    @staticmethod
    def backward(ctx, grad):
        up, mid, down = ctx.saved_tensors
        grad = grad * ctx.scale
        g_up = torch.einsum("o i ..., p q ..., q i -> o p", grad, mid, down)
        g_mid = torch.einsum("o p, o i ..., q i -> p q ...", up, grad, down)
        g_down = torch.einsum("o p, p q ..., o i ... -> q i", up, mid, grad)
        return g_up, g_mid, g_down, None, None


def _scale(gamma) -> float:
    return float(gamma) if not isinstance(gamma, torch.Tensor) else float(gamma.item())


def locon_diff_weight(down, up, mid=None, gamma=1.0, backend=None):
    """DeltaW for LoCon; drop-in for functional.locon.diff_weight."""
    r = down.shape[0]
    out_o = up.shape[0]
    if mid is None:
        shape = (out_o, *down.shape[1:])
        out = LowRankRebuildFn.apply(
            up.reshape(out_o, r),
            down.reshape(r, -1),
            _scale(gamma),
            backend,
        )
        return out.reshape(shape)
    up2 = up.reshape(out_o, r)
    down2 = down.reshape(r, down.shape[1])
    return TuckerRebuildFn.apply(up2, mid, down2, _scale(gamma), backend)


def locon_bypass_diff(x, down, up, gamma=1.0, backend=None):
    """Linear bypass delta: gamma * (x @ down^T) @ up^T, one kernel each way.

    x is (..., I); conv layouts stay on the functional path (a conv bypass is
    a convolution, not a token GEMM).
    """
    r = down.shape[0]
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    out = LowRankBypassFn.apply(
        flat,
        up.reshape(up.shape[0], r),
        down.reshape(r, -1),
        _scale(gamma),
        backend,
    )
    return out.reshape(*lead, up.shape[0])
