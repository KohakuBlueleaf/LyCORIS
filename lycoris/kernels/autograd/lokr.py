"""LoKr: fused kron rebuild and grouped bypass apply.

The rebuild pair generates w1 = w1a@w1b and w2 = w2a@w2b inside the kernel
and chains the sub-factor grads there too, so neither direction materializes
a factor or issues a host mm. Tucker w2 (spatial t2) still builds in torch —
that contraction is conv-shaped, not Kronecker. Conv-spatial bypass routes
through rebuild (v1 scope).
"""

import torch

from ..ops import get_ops
from ..precision import promote, restore


class KronGenRebuildFn(torch.autograd.Function):
    """Both-full goes to the output-tiled gather kernel, which coalesces the
    write; a factorized side goes to the generating kernel, which needs the
    Kronecker-aligned grid to build that side's tile."""

    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, shape, scale, backend):
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
        ops = get_ops(backend)
        if not (ctx.gen1 or ctx.gen2):
            return ops.lokr_full_merge_fwd(a, c, gamma=scale)
        return ops.lokr_merge_fwd(a, b, c, d, shape, gamma=scale)

    @staticmethod
    def backward(ctx, grad):
        w1a, w1b, w2a, w2b = ctx.saved_tensors
        ops = get_ops(ctx.backend)
        grad = grad.contiguous().to(w1a.dtype)
        if not (ctx.gen1 or ctx.gen2):
            g1a, g2a = ops.lokr_full_merge_bwd(grad, w1a, w2a, gamma=ctx.scale)
            got = restore([g1a, None, g2a, None], ctx.dtypes)
            return (*got, None, None, None)
        g1a, g1b, g2a, g2b = ops.lokr_merge_bwd(
            grad,
            w1a,
            w1b if ctx.gen1 else None,
            w2a,
            w2b if ctx.gen2 else None,
            ctx.shape,
            gamma=ctx.scale,
        )
        got = restore([g1a, g1b, g2a, g2b], ctx.dtypes)
        return (*got, None, None, None)


class KronApplyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, scale, backend):
        ctx.save_for_backward(x, w1, w2)
        ctx.scale = scale
        ctx.backend = backend
        return get_ops(backend).lokr_bypass_fwd(x, w1, w2, gamma=scale)

    @staticmethod
    def backward(ctx, grad):
        x, w1, w2 = ctx.saved_tensors
        gx, gw1, gw2 = get_ops(ctx.backend).lokr_bypass_bwd(
            grad.contiguous(), x, w1, w2, gamma=ctx.scale
        )
        return gx, gw1, gw2, None, None


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


def lokr_kron_weight(w1, w1a, w1b, w2, w2a, w2b, t2=None, scale=1.0, backend=None):
    """DeltaW = scale * kron(w1, w2), the scale-explicit form."""
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
    out = KronGenRebuildFn.apply(
        w1a if gen1 else w1,
        w1b if gen1 else None,
        w2a_,
        w2b_,
        (a, b, c, d),
        float(scale),
        backend,
    )
    return out.reshape(a * c, b * d, *k_shape)


def lokr_diff_weight(w1, w1a, w1b, w2, w2a, w2b, t2=None, gamma=1.0, backend=None):
    """lokr_kron_weight under functional.lokr's scale = gamma / rank."""
    return lokr_kron_weight(
        w1, w1a, w1b, w2, w2a, w2b, t2, rank_scale(w1a, w2a, gamma), backend
    )


def lokr_kron_bypass(x, w1, w1a, w1b, w2, w2a, w2b, t2=None, scale=1.0, backend=None):
    """Linear bypass: per-token vec(w1 @ X @ w2^T) * scale, DeltaW never built."""
    w1_ = w1 if w1 is not None else w1a @ w1b
    w2_ = _build_w2(w2, w2a, w2b, t2)
    if w2_.dim() != 2:
        raise ValueError("conv-spatial LoKr bypass routes through rebuild (v1)")
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    y = KronApplyFn.apply(flat, w1_, w2_, float(scale), backend)
    return y.reshape(*lead, y.shape[-1])


def lokr_bypass_diff(x, w1, w1a, w1b, w2, w2a, w2b, t2=None, gamma=1.0, backend=None):
    """lokr_kron_bypass under functional.lokr's scale = gamma / rank."""
    return lokr_kron_bypass(
        x, w1, w1a, w1b, w2, w2a, w2b, t2, rank_scale(w1a, w2a, gamma), backend
    )
