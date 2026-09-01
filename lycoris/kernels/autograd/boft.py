"""BOFT through the fully fused per-stage kernels.

Raw oft_blocks go straight to the kernels: Cayley, the multiplier fold
(b*scale + (1-scale)*I) and — on the last stage — rescale and the diff
subtraction all happen in-kernel. Eager applies R (einsum contracts the
second index), so the forward runs untransposed and the backward chains
R^T applies. The backward replays forward prefixes instead of caching the
m stage inputs (m*O*I bytes); rescale costs one extra replay + reduce only
when it is used. cscale is one host scalar when constraint>0 (documented
exception, same convention as diag_oft).
"""

import torch

from ..ops import get_ops
from ..precision import promote
from .diag_oft import _cscale


class ButterflyFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, x, rescale, axis, cscale, scale, diff, backend):
        ops = get_ops(backend)
        (blocks, x, rescale), _, out_dtype = promote(blocks, x, rescale)
        out = ops.boft_fwd(
            blocks,
            x,
            axis=axis,
            cscale=cscale,
            scale=scale,
            rescale=rescale,
            diff=diff,
        )
        out = out if out.dtype == out_dtype else out.to(out_dtype)
        saved = (blocks, x) if rescale is None else (blocks, x, rescale)
        ctx.save_for_backward(*saved)
        ctx.axis = axis
        ctx.cscale = cscale
        ctx.scale = scale
        ctx.diff = diff
        ctx.backend = backend
        ctx.has_rescale = rescale is not None
        return out

    @staticmethod
    def backward(ctx, grad):
        if ctx.has_rescale:
            blocks, x, rescale = ctx.saved_tensors
        else:
            (blocks, x), rescale = ctx.saved_tensors, None
        ops = get_ops(ctx.backend)
        grad = grad.contiguous()
        axis = ctx.axis % grad.dim()
        g_res = None
        g_chain = grad
        if rescale is not None:
            pre = ops.boft_fwd(
                blocks, x, axis=ctx.axis, cscale=ctx.cscale, scale=ctx.scale
            )
            dims = [d for d in range(grad.dim()) if d != axis]
            g_res = (grad * pre).sum(dims).to(rescale.dtype)
            shape = [1] * grad.dim()
            shape[axis] = -1
            g_chain = (grad * rescale.reshape(shape)).contiguous()
        gx, gb = ops.boft_bwd(
            blocks, x, g_chain, axis=ctx.axis, cscale=ctx.cscale, scale=ctx.scale
        )
        if ctx.diff:
            gx = gx - grad
        return gb, gx, g_res, None, None, None, None, None


def boft_diff_weight(
    org_weight, oft_blocks, rescale=None, constraint=None, scale=1, backend=None
):
    """Mirrors functional.boft.diff_weight: butterfly(W)[*rescale] - W."""
    cscale = _cscale(oft_blocks, constraint)
    w = org_weight.to(oft_blocks.dtype)
    res = None if rescale is None else rescale.reshape(-1)
    return ButterflyFusedFn.apply(
        oft_blocks, w, res, 0, cscale, float(scale), True, backend
    )


def boft_bypass_diff(
    org_out,
    oft_blocks,
    rescale=None,
    constraint=None,
    scale=1,
    need_transpose=False,
    backend=None,
):
    """Mirrors functional.boft.bypass_forward_diff on the channel axis."""
    cscale = _cscale(oft_blocks, constraint)
    axis = 1 if need_transpose else -1
    x = org_out.to(oft_blocks.dtype)
    res = None if rescale is None else rescale.reshape(-1)
    return ButterflyFusedFn.apply(
        oft_blocks, x, res, axis, cscale, float(scale), True, backend
    )
