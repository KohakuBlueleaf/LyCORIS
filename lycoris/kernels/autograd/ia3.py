"""IA3: fused broadcast channel scale, weight-space and activation-space.

x and the scale vector may arrive in different dtypes (an fp32 trained vector
on fp16 activations): the op reads both as they are, computes in fp32 and
returns the promoted dtype, as the eager broadcast multiply does; the backward
saves the operands as they arrived, so a half activation is saved in half.
"""

import torch

from .. import library_apply as lib
from ..precision import restore


class ChannelScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, channel_axis, alpha, gamma, backend):
        backend = backend or ""
        ctx.save_for_backward(x, w)
        ctx.dtypes = (x.dtype, w.dtype)
        ctx.args = (channel_axis, alpha, gamma, backend)
        return lib.ia3_fwd(x, w, channel_axis, alpha, gamma, backend)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        channel_axis, alpha, gamma, backend = ctx.args
        grad = grad.contiguous()
        gx, gw = lib.ia3_bwd(grad, x, w, channel_axis, alpha, gamma, backend)
        gx, gw = restore([gx, gw], ctx.dtypes)
        return gx, gw, None, None, None, None


def ia3_diff_weight(
    org_weight, weight, on_input, multiplier=1.0, diff=True, backend=None
):
    """W * (weight*mult + (0 if diff else 1)) on the in- or out-channel axis."""
    axis = 1 if on_input else 0
    if org_weight.dim() == 2 and on_input:
        axis = -1
    alpha = 0.0 if diff else 1.0
    return ChannelScaleFn.apply(
        org_weight, weight, axis, alpha, float(multiplier), backend
    )


def ia3_bypass(x, weight, channel_axis, multiplier=1.0, diff=False, backend=None):
    """x * (weight*mult + (0 if diff else 1)) matching IA3Module._bypass_forward."""
    alpha = 0.0 if diff else 1.0
    return ChannelScaleFn.apply(
        x, weight, channel_axis, alpha, float(multiplier), backend
    )
