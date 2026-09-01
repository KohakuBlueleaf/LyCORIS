"""F7: fused scaled add (merge_to / diff-add / Full)."""

import torch
import triton
import triton.language as tl

from ...plans import dora as plan
from ...plans import tune
from ...plans.device import resolve_device


@triton.jit
def _add_scaled_kernel(base_ptr, delta_ptr, out_ptr, N, gamma, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < N
    b = tl.load(base_ptr + idx, mask=m, other=0.0).to(tl.float32)
    d = tl.load(delta_ptr + idx, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + idx, (b + gamma * d).to(out_ptr.dtype.element_ty), mask=m)


def add_scaled(
    base: torch.Tensor, delta: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    bc = base.contiguous()
    dc = delta.contiguous()
    out = torch.empty_like(bc)
    n = bc.numel()

    def launch(p, dst):
        _add_scaled_kernel[(triton.cdiv(n, p.bm),)](
            bc, dc, dst, n, gamma, BLOCK=p.bm, num_warps=p.warps, num_stages=p.stages
        )

    best = tune.tuned(
        "triton.merge.add_scaled",
        (tune.bucket_tokens(n), str(base.dtype)),
        lambda: plan.topk_elementwise(n, 3.0, resolve_device(), base.element_size()),
        lambda p: (lambda: launch(p, out)),
    )
    launch(best, out)
    return out.view_as(base)
