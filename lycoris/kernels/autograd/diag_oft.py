"""Diag-OFT: single fused kernel per direction, raw blocks in, grads out.

The Cayley transform, rescale fold and identity shift all live inside the
kernel (loop3-fusion.md); this layer only handles the constraint scalar (a
global norm over the KB-sized blocks — the one permitted host op) and the
eager-compatible reshapes.
"""

import torch

from ..ops import get_ops
from ..precision import promote, restore


def _cscale(oft_blocks, constraint):
    if constraint is None or constraint <= 0:
        return 1.0
    q = oft_blocks - oft_blocks.transpose(-1, -2)
    q_norm = torch.norm(q) + 1e-8
    if float(q_norm) > constraint:
        return constraint / float(q_norm)
    return 1.0


class BlockDiagFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, blocks, x, rescale, cscale, shift, weight, backend):
        (bc, xc, rc), _, out_dtype = promote(blocks, x, rescale)
        ctx.save_for_backward(bc, xc, *((rc,) if rc is not None else ()))
        ctx.dtypes = (
            blocks.dtype,
            x.dtype,
            rescale.dtype if rescale is not None else None,
        )
        ctx.args = (rescale is not None, cscale, shift, weight, backend)
        out = get_ops(backend).oft_fwd(bc, xc, rc, cscale, shift, weight)
        return out if out.dtype == out_dtype else out.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        has_rescale, cscale, shift, weight, backend = ctx.args
        blocks, x = ctx.saved_tensors[:2]
        rescale = ctx.saved_tensors[2] if has_rescale else None
        gx, gb, gres = get_ops(backend).oft_bwd(
            blocks, x, grad.contiguous().to(x.dtype), rescale, cscale, shift, weight
        )
        gres_out = gres.view_as(rescale) if has_rescale else None
        grads = restore([gb, gx, gres_out], ctx.dtypes)
        return (*grads, None, None, None, None)


def diag_oft_diff_weight(
    org_weight, oft_blocks, rescale=None, constraint=None, backend=None
):
    """Mirrors functional.diag_oft.diff_weight in one kernel launch."""
    k, s, _ = oft_blocks.shape
    cscale = _cscale(oft_blocks, constraint)
    w = org_weight.reshape(k * s, -1).contiguous()
    out = BlockDiagFusedFn.apply(oft_blocks, w, rescale, cscale, True, True, backend)
    return out.reshape(org_weight.shape).to(org_weight.dtype)


def diag_oft_bypass_diff(
    org_out,
    oft_blocks,
    rescale=None,
    constraint=None,
    need_transpose=False,
    backend=None,
):
    """Mirrors functional.diag_oft.bypass_forward_diff in one kernel launch."""
    cscale = _cscale(oft_blocks, constraint)
    if need_transpose and org_out.dim() > 2:
        return BlockDiagFusedFn.apply(
            oft_blocks, org_out, rescale, cscale, True, False, backend
        )
    flat = org_out.reshape(-1, org_out.shape[-1]).contiguous()
    out = BlockDiagFusedFn.apply(
        oft_blocks, flat, rescale, cscale, True, False, backend
    )
    return out.reshape(org_out.shape)
