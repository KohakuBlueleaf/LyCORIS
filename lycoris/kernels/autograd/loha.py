"""LoHa: fused hadamard rebuild and true weight-free bypass.

Signatures mirror ``lycoris.functional.loha`` (w1d/w1u naming and order) so
tests compare directly. Tucker backward recomputes the two rebuilds in torch
(conv-sized tensors); everything else runs in the fused kernels.
"""

import torch

from ..ops import get_ops
from ..precision import promote, restore


class HadaRebuildFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1d, w1u, w2d, w2u, scale, backend):
        (a, b, c, d), _, _ = promote(w1d, w1u, w2d, w2u)
        ctx.save_for_backward(a, b, c, d)
        ctx.dtypes = (w1d.dtype, w1u.dtype, w2d.dtype, w2u.dtype)
        ctx.scale = scale
        ctx.backend = backend
        ops = get_ops(backend)
        return ops.lora_merge_fwd(b, a, d, c, gamma=scale, mode="hada")

    @staticmethod
    def backward(ctx, grad):
        w1d, w1u, w2d, w2u = ctx.saved_tensors
        ops = get_ops(ctx.backend)
        g_w1u, g_w1d, g_w2u, g_w2d = ops.loha_merge_bwd(
            grad.contiguous().to(w1d.dtype), w1u, w1d, w2u, w2d, gamma=ctx.scale
        )
        grads = restore([g_w1d, g_w1u, g_w2d, g_w2u], ctx.dtypes)
        return (*grads, None, None)


class HadaTuckerRebuildFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1d, w1u, t2, w2d, w2u, scale, backend):
        ctx.save_for_backward(t1, w1d, w1u, t2, w2d, w2u)
        ctx.scale = scale
        ops = get_ops(backend)
        k = t1.shape[2:].numel()
        out = ops.lora_tucker_fwd(
            w1u.transpose(0, 1),
            t1.reshape(*t1.shape[:2], k),
            w1d,
            w2u.transpose(0, 1),
            t2.reshape(*t2.shape[:2], k),
            w2d,
            gamma=scale,
        )
        return out.reshape(w1u.shape[1], w1d.shape[1], *t1.shape[2:])

    @staticmethod
    def backward(ctx, grad):
        t1, w1d, w1u, t2, w2d, w2u = ctx.saved_tensors
        grad = grad * ctx.scale
        reb1 = torch.einsum("i j ..., j r, i p -> p r ...", t1, w1d, w1u)
        reb2 = torch.einsum("i j ..., j r, i p -> p r ...", t2, w2d, w2u)

        gw = grad * reb2
        g_w1u = torch.einsum("p r ..., i j ..., j r -> p i", gw, t1, w1d).transpose(
            0, 1
        )
        g_w1d = torch.einsum("p r ..., i j ..., i p -> j r", gw, t1, w1u)
        g_t1 = torch.einsum("p r ..., j r, i p -> i j ...", gw, w1d, w1u)

        gw = grad * reb1
        g_w2u = torch.einsum("p r ..., i j ..., j r -> p i", gw, t2, w2d).transpose(
            0, 1
        )
        g_w2d = torch.einsum("p r ..., i j ..., i p -> j r", gw, t2, w2u)
        g_t2 = torch.einsum("p r ..., j r, i p -> i j ...", gw, w2d, w2u)
        return g_t1, g_w1d, g_w1u, g_t2, g_w2d, g_w2u, None, None


class HadaDeltaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1d, w1u, w2d, w2u, scale, backend):
        (xc, a, b, c, d), _, out_dtype = promote(x, w1d, w1u, w2d, w2u)
        ctx.save_for_backward(xc, a, b, c, d)
        ctx.dtypes = (x.dtype, w1d.dtype, w1u.dtype, w2d.dtype, w2u.dtype)
        ctx.scale = scale
        ctx.backend = backend
        y = get_ops(backend).loha_bypass_fwd(xc, b, a, d, c, gamma=scale)
        return y if y.dtype == out_dtype else y.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        x, w1d, w1u, w2d, w2u = ctx.saved_tensors
        ops = get_ops(ctx.backend)
        gx, g_w1u, g_w1d, g_w2u, g_w2d = ops.loha_bypass_bwd(
            grad.contiguous().to(x.dtype), x, w1u, w1d, w2u, w2d, gamma=ctx.scale
        )
        grads = restore([gx, g_w1d, g_w1u, g_w2d, g_w2u], ctx.dtypes)
        return (*grads, None, None)


def _scale(gamma) -> float:
    return float(gamma) if not isinstance(gamma, torch.Tensor) else float(gamma.item())


def loha_diff_weight(w1d, w1u, w2d, w2u, t1=None, t2=None, gamma=1.0, backend=None):
    """DeltaW for LoHa; drop-in for functional.loha.diff_weight (same layout)."""
    if t1 is not None:
        return HadaTuckerRebuildFn.apply(
            t1, w1d, w1u, t2, w2d, w2u, _scale(gamma), backend
        )
    out_o = w1u.shape[0]
    out_i = w1d.shape[1]
    return HadaRebuildFn.apply(w1d, w1u, w2d, w2u, _scale(gamma), backend).reshape(
        out_o, out_i
    )


def loha_bypass_diff(x, w1d, w1u, w2d, w2u, gamma=1.0, backend=None):
    """Linear bypass: y_diff = gamma * x @ DeltaW^T with DeltaW never built.

    x may carry leading batch dims; conv layouts route through the fused
    rebuild plus a conv instead (wrapper decision, not this function).
    """
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    y = HadaDeltaFn.apply(flat, w1d, w1u, w2d, w2u, _scale(gamma), backend)
    return y.reshape(*lead, y.shape[-1])
