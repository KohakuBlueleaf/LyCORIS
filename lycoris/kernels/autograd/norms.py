"""Norm layers: fused scaled adds for the diff weights (trivial family)."""

import torch

from ..ops import get_ops


class AddScaledFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, delta, gamma, backend):
        ctx.gamma = gamma
        return get_ops(backend).add_scaled(base, delta, gamma)

    @staticmethod
    def backward(ctx, grad):
        return grad, grad * ctx.gamma, None, None


def norm_diff_weights(org_w, org_b, w_norm, b_norm, multiplier=1.0, backend=None):
    w = AddScaledFn.apply(org_w, w_norm, float(multiplier), backend)
    b = None
    if b_norm is not None and org_b is not None:
        b = AddScaledFn.apply(org_b, b_norm, float(multiplier), backend)
    return w, b
