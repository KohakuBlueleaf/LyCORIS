"""GLoRA: fused dual low-rank rebuild DeltaW = gamma * (U @ a2 + b1 @ b2).

U = W @ a1 stays a cuBLAS GEMM in the caller's graph (O x r output, K = I);
torch chains its grad to a1. Backward here is four skinny GEMMs.
"""

import torch

from ..ops import get_ops


class DualLowRankRebuildFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a2, b1, b2, scale, backend):
        ctx.save_for_backward(u, a2, b1, b2)
        ctx.scale = scale
        return get_ops(backend).lora_merge_fwd(u, a2, b1, b2, gamma=scale, mode="sum")

    @staticmethod
    def backward(ctx, grad):
        u, a2, b1, b2 = ctx.saved_tensors
        grad = grad * ctx.scale
        g_u = grad @ a2.transpose(0, 1)
        g_a2 = u.transpose(0, 1) @ grad
        g_b1 = grad @ b2.transpose(0, 1)
        g_b2 = b1.transpose(0, 1) @ grad
        return g_u, g_a2, g_b1, g_b2, None, None


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
