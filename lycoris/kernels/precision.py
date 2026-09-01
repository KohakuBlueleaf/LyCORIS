"""Operand dtype policy: what the matmul runs in, what comes back out.

Inputs (x or the original weight) and module weights carry their dtypes
independently, in fp16, bf16 or fp32. fp16 and bf16 never appear together, so
the 16-bit type in play is unambiguous and a mix of the two is a caller bug
rather than a case to resolve.

Accumulation is fp32 under every policy; only the MMA operand type and the
output type are decided here.
"""

import torch

HALF = (torch.float16, torch.bfloat16)
POLICIES = ("mma16", "wide", "tf32")


def half_of(dtypes) -> torch.dtype | None:
    """The 16-bit float among ``dtypes``; raises if both 16-bit types appear."""
    seen = {d for d in dtypes if d in HALF}
    if len(seen) > 1:
        raise ValueError(
            f"fp16 and bf16 must not appear together: {sorted(map(str, seen))}"
        )
    return next(iter(seen), None)


def resolve(*tensors, policy: str = "mma16") -> tuple[torch.dtype, torch.dtype]:
    """(compute dtype, output dtype) for this operand set.

    ``mma16`` runs the matmul in the 16-bit type present, rounding an fp32
    operand down to it; ``wide`` promotes everything to fp32 IEEE; ``tf32``
    keeps fp32 operands but lets them reach the tensor cores. The output is
    always the promoted type, so an fp32 operand yields an fp32 result
    whatever the matmul ran in.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown precision policy {policy!r}; have {POLICIES}")
    dtypes = [t.dtype for t in tensors if t is not None]
    if not dtypes:
        raise ValueError("resolve() needs at least one tensor")
    out = dtypes[0]
    for d in dtypes[1:]:
        out = torch.promote_types(out, d)
    half = half_of(dtypes)
    if policy == "mma16" and half is not None:
        return half, out
    return torch.float32, out


def cast_operands(tensors, compute: torch.dtype):
    """Operands in the compute dtype; a matching tensor is passed through."""
    return [
        None if t is None else (t if t.dtype == compute else t.to(compute))
        for t in tensors
    ]


def promote(*tensors, policy: str = "mma16"):
    """(operands in the compute dtype, compute dtype, output dtype).

    The single entry point the autograd layer uses, so a backend kernel keeps
    its one-dtype assumption while callers may mix fp16/bf16/fp32 freely
    across x, the original weight and the module weights.
    """
    compute, out = resolve(*tensors, policy=policy)
    return cast_operands(tensors, compute), compute, out


def restore(grads, dtypes):
    """Gradients back in each leaf's own dtype."""
    return [
        None if (g is None or d is None) else (g if g.dtype == d else g.to(d))
        for g, d in zip(grads, dtypes)
    ]
