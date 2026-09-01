"""DoRA epilogue: y = W' * (m*(d/||row|| - 1) + 1) with fused norm/scale.

The weight-decompose stage is shared by every algorithm that has a decomposed
variant (dora, doha, dokr) — it consumes an already-merged weight, so it is
one kernel regardless of what produced that weight.

Matches apply_weight_decompose semantics including the finfo-eps guard and
the multiplier interpolation; wd_on_out=False uses the column axis.
"""

import torch

from ..ops import get_ops
from ..precision import promote, restore


class DoraScaleFn(torch.autograd.Function):
    """Norms are cached (one fp32 per row, O(rows)); recomputing them would
    cost a second full pass over W, which is the expensive side here."""

    @staticmethod
    def forward(ctx, w, dscale, mult, row_axis, backend):
        ops = get_ops(backend)
        (wc, dc), _, out_dtype = promote(w, dscale)
        w2d = wc.reshape(wc.shape[0], -1)
        y, norms = ops.dora_fwd(w2d, dc, mult, row_axis)
        ctx.save_for_backward(wc, dc, norms)
        ctx.dtypes = (w.dtype, dscale.dtype)
        ctx.args = (mult, row_axis, backend)
        y = y.view_as(w)
        return y if y.dtype == out_dtype else y.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        w, dscale, norms = ctx.saved_tensors
        mult, row_axis, backend = ctx.args
        ops = get_ops(backend)
        w2d = w.reshape(w.shape[0], -1)
        g2d = grad.reshape(grad.shape[0], -1).contiguous().to(w.dtype)
        gw, gd = ops.dora_bwd(g2d, w2d, dscale, norms, mult, row_axis)
        grads = restore([gw.view_as(w), gd.view_as(dscale)], ctx.dtypes)
        return (*grads, None, None, None)


def apply_dora(weight, dora_scale, multiplier=1.0, wd_on_out=True, backend=None):
    """Weight-decompose scaling of the (already merged) weight tensor.

    wd_on_out=False on a conv weight needs a per-in-channel norm over
    (out, spatial), which the flat 2D column view cannot express — that case
    stays on the eager path.
    """
    if not wd_on_out and weight.dim() > 2:
        raise ValueError("wd_on_out=False conv DoRA stays on the eager path")
    row_axis = 0 if wd_on_out else 1
    return DoraScaleFn.apply(weight, dora_scale, float(multiplier), row_axis, backend)
