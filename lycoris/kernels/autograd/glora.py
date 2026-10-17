"""GLoRA: fused dual low-rank rebuild DeltaW = gamma * (U @ a2 + b1 @ b2).

U = W @ a1 stays a cuBLAS GEMM in the caller's graph (O x r output, K = I);
torch chains its grad to a1. Backward here is four skinny GEMMs.
"""

import torch

from .. import library as lib
from ..precision import promote, restore


class DualLowRankRebuildFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a2, b1, b2, scale, backend):
        ops, _, out_dtype = promote(u, a2, b1, b2)
        ctx.save_for_backward(*ops)
        ctx.dtypes = (u.dtype, a2.dtype, b1.dtype, b2.dtype)
        ctx.scale = scale
        out = lib.lora_merge_fwd(*ops, None, scale, "sum", backend or "", None)
        return out if out.dtype == out_dtype else out.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        u, a2, b1, b2 = ctx.saved_tensors
        grad = (grad * ctx.scale).to(u.dtype)
        g_u = grad @ a2.transpose(0, 1)
        g_a2 = u.transpose(0, 1) @ grad
        g_b1 = grad @ b2.transpose(0, 1)
        g_b2 = b1.transpose(0, 1) @ grad
        return (*restore([g_u, g_a2, g_b1, g_b2], ctx.dtypes), None, None)


def glora_diff_weight(w, a1, a2, b1, b2, gamma=1.0, backend=None):
    """DeltaW = gamma * (W @ a1 @ a2 + b1 @ b2).

    Module layouts: a1 is (in, r), a2 is (r, in), b1 is (out, r), b2 is
    (r, in) — matching GLoRAModule's Linear weights flattened 2D.
    """
    scale = float(gamma) if not isinstance(gamma, torch.Tensor) else float(gamma.item())
    w2d = w.reshape(w.shape[0], -1)
    u = w2d @ a1.reshape(a1.shape[0], -1)
    out = DualLowRankRebuildFn.apply(
        u,
        a2.reshape(a2.shape[0], -1),
        b1.reshape(b1.shape[0], -1),
        b2.reshape(b2.shape[0], -1),
        scale,
        backend,
    )
    return out.reshape(w.shape)
