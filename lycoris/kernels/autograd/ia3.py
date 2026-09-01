"""IA3: fused broadcast channel scale, weight-space and activation-space."""

import torch

from ..ops import get_ops


class ChannelScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, channel_axis, alpha, gamma, backend):
        ctx.save_for_backward(x, w)
        ctx.args = (channel_axis, alpha, gamma, backend)
        return get_ops(backend).ia3_fwd(x, w, channel_axis, alpha, gamma)

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        channel_axis, alpha, gamma, backend = ctx.args
        gx, gw = get_ops(backend).ia3_bwd(grad, x, w, channel_axis, alpha, gamma)
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
