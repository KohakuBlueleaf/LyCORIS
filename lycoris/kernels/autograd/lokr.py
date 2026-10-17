"""LoKr: fused kron rebuild and grouped bypass apply.

The rebuild pair generates w1 = w1a@w1b and w2 = w2a@w2b inside the kernel
and chains the sub-factor grads there too, so neither direction materializes
a factor or issues a host mm. Tucker w2 (spatial t2) still builds in torch —
that contraction is conv-shaped, not Kronecker. Conv-spatial bypass routes
through rebuild (v1 scope). Kernels are reached through ``..library``.
"""

import torch

from .. import library as lib
from ..precision import folds, promote, restore


class KronGenRebuildFn(torch.autograd.Function):
    """Both-full goes to the output-tiled gather kernel, which coalesces the
    write; a factorized side goes to the generating kernel, which needs the
    Kronecker-aligned grid to build that side's tile."""

    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, shape, scale, backend, base):
        backend = backend or ""
        (a, b, c, d), _, _ = promote(w1a, w1b, w2a, w2b)
        ctx.save_for_backward(a, b, c, d)
        ctx.dtypes = tuple(
            t.dtype if t is not None else None for t in (w1a, w1b, w2a, w2b)
        )
        ctx.shape = shape
        ctx.scale = scale
        ctx.backend = backend
        ctx.gen1 = w1b is not None
        ctx.gen2 = w2b is not None
        ctx.base_dtype = None if base is None else base.dtype
        out_dtype = None if base is None else base.dtype
        if not (ctx.gen1 or ctx.gen2):
            return lib.lokr_full_merge_fwd(a, c, base, scale, backend, out_dtype)
        return lib.lokr_merge_fwd(
            a, b, c, d, list(shape), base, scale, backend, out_dtype
        )

    @staticmethod
    def backward(ctx, grad):
        w1a, w1b, w2a, w2b = ctx.saved_tensors
        grad = grad.contiguous().to(w1a.dtype)
        if not (ctx.gen1 or ctx.gen2):
            g1a, g2a = lib.lokr_full_merge_bwd(grad, w1a, w2a, ctx.scale, ctx.backend)
            got = restore([g1a, None, g2a, None], ctx.dtypes)
            return (*got, None, None, None, _base_grad(grad, ctx.base_dtype))
        g1a, g1b, g2a, g2b = lib.lokr_merge_bwd(
            grad,
            w1a,
            w1b if ctx.gen1 else None,
            w2a,
            w2b if ctx.gen2 else None,
            ctx.shape,
            ctx.scale,
            ctx.backend,
        )
        got = restore([g1a, g1b, g2a, g2b], ctx.dtypes)
        return (*got, None, None, None, _base_grad(grad, ctx.base_dtype))


def _base_grad(grad, dtype):
    return None if dtype is None else grad.to(dtype)


class KronApplyFn(torch.autograd.Function):
    """[base +] scale * x @ kron(w1, w2)^T; base (the base layer's output) is
    added in the kernel's store and returned in its own dtype."""

    @staticmethod
    def forward(ctx, x, w1, w2, scale, backend, base):
        backend = backend or ""
        (xc, w1c, w2c), _, out_dtype = promote(x, w1, w2)
        ctx.save_for_backward(xc, w1c, w2c)
        ctx.dtypes = (x.dtype, w1.dtype, w2.dtype)
        ctx.scale = scale
        ctx.backend = backend
        ctx.base_dtype = None if base is None else base.dtype
        y = lib.lokr_bypass_fwd(xc, w1c, w2c, scale, backend, base)
        if base is not None:
            return y
        return y if y.dtype == out_dtype else y.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        x, w1, w2 = ctx.saved_tensors
        gx, gw1, gw2 = lib.lokr_bypass_bwd(
            grad.contiguous().to(x.dtype), x, w1, w2, ctx.scale, ctx.backend
        )
        g_base = None if ctx.base_dtype is None else grad.to(ctx.base_dtype)
        return (*restore([gx, gw1, gw2], ctx.dtypes), None, None, g_base)


def _build_w2(w2, w2a, w2b, t2):
    if w2 is not None:
        return w2
    if t2 is not None:
        return torch.einsum("i j ..., i p, j r -> p r ...", t2, w2a, w2b)
    return w2a @ w2b


def rank_scale(w1a, w2a, gamma):
    """functional.lokr's scale = gamma / rank convention, defined once.

    Type-preserving: a tensor gamma stays a tensor for the eager path, and the
    kernel entry points take the float of it.
    """
    if w1a is not None:
        rank = w1a.shape[1]
    elif w2a is not None:
        rank = w2a.shape[1]
    else:
        rank = gamma
    return gamma / rank


def lokr_kron_weight(
    w1, w1a, w1b, w2, w2a, w2b, t2=None, scale=1.0, backend=None, base=None
):
    """DeltaW = scale * kron(w1, w2), the scale-explicit form; with ``base``
    (the original weight) base + DeltaW, the add folded into the kernel."""
    # Kronecker halves go to the kernel unbuilt; only tucker w2 needs torch.
    gen1 = w1 is None
    a, b = (w1a.shape[0], w1b.shape[1]) if gen1 else w1.shape
    if w2 is None and t2 is None:
        w2a_, w2b_ = w2a, w2b
        c, d = w2a.shape[0], w2b.shape[1]
        k_shape = ()
    else:
        built = _build_w2(w2, w2a, w2b, t2)
        k_shape = built.shape[2:]
        w2a_, w2b_ = built.reshape(built.shape[0], -1), None
        c, d = w2a_.shape
    factors = (w1a if gen1 else w1, w1b if gen1 else None, w2a_, w2b_)
    fold = (
        base is not None
        and not k_shape
        and folds(base, *(f for f in factors if f is not None))
    )
    out = KronGenRebuildFn.apply(
        *factors,
        (a, b, c, d),
        float(scale),
        backend,
        base.reshape(a * c, b * d) if fold else None,
    )
    out = out.reshape(a * c, b * d, *k_shape)
    if base is not None and not fold:
        return base + out
    return out


def lokr_diff_weight(w1, w1a, w1b, w2, w2a, w2b, t2=None, gamma=1.0, backend=None):
    """lokr_kron_weight under functional.lokr's scale = gamma / rank."""
    return lokr_kron_weight(
        w1, w1a, w1b, w2, w2a, w2b, t2, rank_scale(w1a, w2a, gamma), backend
    )


def lokr_kron_bypass(
    x, w1, w1a, w1b, w2, w2a, w2b, t2=None, scale=1.0, backend=None, base=None
):
    """Linear bypass: per-token vec(w1 @ X @ w2^T) * scale, DeltaW never built;
    with ``base`` (the base layer's output, (..., O)) the result is base + that,
    the add folded into the kernel's store."""
    w1_ = w1 if w1 is not None else w1a @ w1b
    w2_ = _build_w2(w2, w2a, w2b, t2)
    if w2_.dim() != 2:
        raise ValueError("conv-spatial LoKr bypass routes through rebuild (v1)")
    lead = x.shape[:-1]
    o = w1_.shape[0] * w2_.shape[0]
    flat = x.reshape(-1, x.shape[-1])
    fold = base is not None and folds(base, x, w1_, w2_)
    b2 = base.reshape(-1, o) if fold else None
    y = KronApplyFn.apply(flat, w1_, w2_, float(scale), backend, b2)
    y = y.reshape(*lead, o)
    return base + y if base is not None and not fold else y


def lokr_bypass_diff(
    x, w1, w1a, w1b, w2, w2a, w2b, t2=None, gamma=1.0, backend=None, base=None
):
    """lokr_kron_bypass under functional.lokr's scale = gamma / rank."""
    return lokr_kron_bypass(
        x, w1, w1a, w1b, w2, w2a, w2b, t2, rank_scale(w1a, w2a, gamma), backend, base
    )
