"""LoHa: fused hadamard rebuild and true weight-free bypass.

Signatures mirror ``lycoris.functional.loha`` (w1d/w1u naming and order) so
tests compare directly. Tucker backward recomputes the two rebuilds in torch
(conv-sized tensors); everything else runs in the fused kernels, reached
through the custom ops in ``..library``.
"""

import torch

from .. import library as lib
from ..precision import folds, promote, restore


class HadaRebuildFn(torch.autograd.Function):
    """[base +] scale * (w1u @ w1d) * (w2u @ w2d); base (the original weight) is
    added in the kernel's epilogue and the result stored in its dtype."""

    @staticmethod
    def forward(ctx, w1d, w1u, w2d, w2u, scale, backend, base):
        backend = backend or ""
        (a, b, c, d), _, _ = promote(w1d, w1u, w2d, w2u)
        ctx.save_for_backward(a, b, c, d)
        ctx.dtypes = (w1d.dtype, w1u.dtype, w2d.dtype, w2u.dtype)
        ctx.scale = scale
        ctx.backend = backend
        ctx.base_dtype = None if base is None else base.dtype
        out_dtype = None if base is None else base.dtype
        return lib.lora_merge_fwd(b, a, d, c, base, scale, "hada", backend, out_dtype)

    @staticmethod
    def backward(ctx, grad):
        w1d, w1u, w2d, w2u = ctx.saved_tensors
        grad = grad.contiguous()
        g_w1u, g_w1d, g_w2u, g_w2d = lib.loha_merge_bwd(
            grad.to(w1d.dtype),
            w1u,
            w1d,
            w2u,
            w2d,
            ctx.scale,
            ctx.backend,
        )
        grads = restore([g_w1d, g_w1u, g_w2d, g_w2u], ctx.dtypes)
        g_base = None if ctx.base_dtype is None else grad.to(ctx.base_dtype)
        return (*grads, None, None, g_base)


class HadaTuckerRebuildFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1d, w1u, t2, w2d, w2u, scale, backend):
        (t1, w1d, w1u, t2, w2d, w2u), _, _ = promote(t1, w1d, w1u, t2, w2d, w2u)
        ctx.save_for_backward(t1, w1d, w1u, t2, w2d, w2u)
        ctx.scale = scale
        k = t1.shape[2:].numel()
        out = lib.lora_tucker_fwd(
            w1u.transpose(0, 1),
            t1.reshape(*t1.shape[:2], k),
            w1d,
            w2u.transpose(0, 1),
            t2.reshape(*t2.shape[:2], k),
            w2d,
            scale,
            backend or "",
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
    """[base +] scale * x @ DeltaW^T; base (the base layer's output) is added in
    the kernel's store and returned in its own dtype."""

    @staticmethod
    def forward(ctx, x, w1d, w1u, w2d, w2u, scale, backend, base):
        backend = backend or ""
        (xc, a, b, c, d), _, out_dtype = promote(x, w1d, w1u, w2d, w2u)
        ctx.save_for_backward(xc, a, b, c, d)
        ctx.dtypes = (x.dtype, w1d.dtype, w1u.dtype, w2d.dtype, w2u.dtype)
        ctx.scale = scale
        ctx.backend = backend
        ctx.base_dtype = None if base is None else base.dtype
        y = lib.loha_bypass_fwd(xc, b, a, d, c, scale, backend, base)
        if base is not None:
            return y
        return y if y.dtype == out_dtype else y.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        x, w1d, w1u, w2d, w2u = ctx.saved_tensors
        gx, g_w1u, g_w1d, g_w2u, g_w2d = lib.loha_bypass_bwd(
            grad.contiguous().to(x.dtype), x, w1u, w1d, w2u, w2d, ctx.scale, ctx.backend
        )
        grads = restore([gx, g_w1d, g_w1u, g_w2d, g_w2u], ctx.dtypes)
        g_base = None if ctx.base_dtype is None else grad.to(ctx.base_dtype)
        return (*grads, None, None, g_base)


def _scale(gamma) -> float:
    return float(gamma) if not isinstance(gamma, torch.Tensor) else float(gamma.item())


def loha_diff_weight(
    w1d, w1u, w2d, w2u, t1=None, t2=None, gamma=1.0, backend=None, base=None
):
    """DeltaW for LoHa; drop-in for functional.loha.diff_weight (same layout).
    With ``base`` (the original weight) the result is base + DeltaW, the add
    folded into the kernel's epilogue."""
    if t1 is not None:
        out = HadaTuckerRebuildFn.apply(
            t1, w1d, w1u, t2, w2d, w2u, _scale(gamma), backend
        )
        return out if base is None else base + out
    out_o = w1u.shape[0]
    out_i = w1d.shape[1]
    fold = base is not None and folds(base, w1d, w1u, w2d, w2u)
    b2 = base.reshape(out_o, out_i) if fold else None
    out = HadaRebuildFn.apply(w1d, w1u, w2d, w2u, _scale(gamma), backend, b2)
    out = out.reshape(out_o, out_i)
    if base is not None and not fold:
        return base + out.reshape(base.shape)
    return out if base is None else out.reshape(base.shape)


def loha_bypass_diff(x, w1d, w1u, w2d, w2u, gamma=1.0, backend=None, base=None):
    """Linear bypass: y_diff = gamma * x @ DeltaW^T; with ``base`` (the base
    layer's output, (..., O)) the result is base + y_diff, the add folded into
    the kernel's store.

    x may carry leading batch dims; conv layouts route through the fused
    rebuild plus a conv instead (wrapper decision, not this function).
    """
    lead = x.shape[:-1]
    o = w1u.shape[0]
    flat = x.reshape(-1, x.shape[-1])
    fold = base is not None and folds(base, x, w1d, w1u, w2d, w2u)
    b2 = base.reshape(-1, o) if fold else None
    y = HadaDeltaFn.apply(flat, w1d, w1u, w2d, w2u, _scale(gamma), backend, b2)
    y = y.reshape(*lead, o)
    return base + y if base is not None and not fold else y
