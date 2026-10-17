"""The rebuild op of every low-rank family, by mode: LoRA's product, LoHa's
hadamard of two products, GLoRA's sum of two (the rank-(r1 + r2) product of the
stacked factors)."""

import torch

from ..loha.hada import loha_merge
from ..lora.rebuild import lora_merge


def lora_merge_fwd(
    a1, b1, a2=None, b2=None, base=None, gamma=1.0, mode="plain", out_dtype=None
):
    """gamma * (a1 @ b1) [* or + (a2 @ b2)] [+ base], stored in ``out_dtype``
    (default a1's). mode: "plain", "hada" (product), "sum" (GLoRA)."""
    match mode:
        case "plain":
            return lora_merge(a1, b1, gamma, base, out_dtype)
        case "sum":
            up = torch.cat([a1, a2.to(a1.dtype)], dim=1)
            down = torch.cat([b1, b2.to(b1.dtype)], dim=0)
            return lora_merge(up, down, gamma, base, out_dtype)
        case "hada":
            a2, b2 = a2.to(a1.dtype), b2.to(a1.dtype)
            return loha_merge(a1, b1, a2, b2, gamma, base, out_dtype)
        case _:
            raise ValueError(f"merge mode {mode!r}")
