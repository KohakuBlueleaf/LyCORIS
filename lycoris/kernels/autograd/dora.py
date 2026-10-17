"""DoRA epilogue: y = W' * (m*(d/||row|| - 1) + 1) with fused norm/scale.

The weight-decompose stage is shared by every algorithm that has a decomposed
variant (dora, doha, dokr) — it consumes an already-merged weight, so it is
one kernel regardless of what produced that weight.

Matches apply_weight_decompose semantics including the finfo-eps guard and
the multiplier interpolation; wd_on_out=False uses the column axis.
"""

import torch

from .. import library_apply as lib
from ..precision import promote, restore


class DoraScaleFn(torch.autograd.Function):
    """Norms are cached (one fp32 per row, O(rows)); recomputing them would
    cost a second full pass over W, which is the expensive side here. With
    ``delta`` the decompose runs on M = w + delta read in the kernel (M never
    stored); w and delta share the gradient of M."""

    @staticmethod
    def forward(ctx, w, dscale, mult, row_axis, backend, delta):
        backend = backend or ""
        (wc, dc, dl), _, out_dtype = promote(w, dscale, delta)
        w2d = wc.reshape(wc.shape[0], -1)
        d2d = None if dl is None else dl.reshape(w2d.shape)
        y, norms = lib.dora_fwd(w2d, dc, mult, row_axis, backend, d2d)
        ctx.save_for_backward(wc, dc, norms, dl)
        ctx.dtypes = (w.dtype, dscale.dtype, None if delta is None else delta.dtype)
        ctx.args = (mult, row_axis, backend)
        y = y.view_as(w)
        return y if y.dtype == out_dtype else y.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        w, dscale, norms, dl = ctx.saved_tensors
        mult, row_axis, backend = ctx.args
        w2d = w.reshape(w.shape[0], -1)
        d2d = None if dl is None else dl.reshape(w2d.shape)
        g2d = grad.reshape(grad.shape[0], -1).contiguous().to(w.dtype)
        gw, gd = lib.dora_bwd(g2d, w2d, dscale, norms, mult, row_axis, backend, d2d)
        gw = gw.view_as(w)
        g_delta = None if dl is None else gw
        grads = restore([gw, gd.view_as(dscale), g_delta], ctx.dtypes)
        return (grads[0], grads[1], None, None, None, grads[2])


def apply_dora(
    weight, dora_scale, multiplier=1.0, wd_on_out=True, backend=None, delta=None
):
    """Weight-decompose scaling of the merged weight: ``weight`` itself, or
    weight + ``delta`` with the add read inside the kernel.

    wd_on_out=False on a conv weight needs a per-in-channel norm over
    (out, spatial), which the flat 2D column view cannot express — that case
    stays on the eager path.
    """
    if not wd_on_out and weight.dim() > 2:
        raise ValueError("wd_on_out=False conv DoRA stays on the eager path")
    row_axis = 0 if wd_on_out else 1
    if delta is not None and delta.shape != weight.shape:
        delta = delta.reshape(weight.shape)
    return DoraScaleFn.apply(
        weight, dora_scale, float(multiplier), row_axis, backend, delta
    )
