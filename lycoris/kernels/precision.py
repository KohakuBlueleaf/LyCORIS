"""Operand dtypes: what the matmul runs in, what comes back out.

Inputs (x or the original weight) and module weights carry their dtypes
independently, in fp16, bf16 or fp32. fp16 and bf16 never appear together, so
the 16-bit type in play is unambiguous and a mix of the two is a caller bug
rather than a case to resolve.

Accumulation is always fp32; only the MMA operand type and the output type are
decided here.
"""

import torch

HALF = (torch.float16, torch.bfloat16)


def half_of(dtypes) -> torch.dtype | None:
    """The 16-bit float among ``dtypes``; raises if both 16-bit types appear."""
    seen = {d for d in dtypes if d in HALF}
    if len(seen) > 1:
        raise ValueError(
            f"fp16 and bf16 must not appear together: {sorted(map(str, seen))}"
        )
    return next(iter(seen), None)


def resolve(
    *tensors, compute: torch.dtype | None = None
) -> tuple[torch.dtype, torch.dtype]:
    """(compute dtype, output dtype) for this operand set.

    The matmul runs in ``compute`` when given, else in the 16-bit type present
    (an fp32 operand rounded down to it), else fp32. The output is always the
    promoted type, so an fp32 operand yields an fp32 result whatever the
    matmul ran in.
    """
    dtypes = [t.dtype for t in tensors if t is not None]
    if not dtypes:
        raise ValueError("resolve() needs at least one tensor")
    out = dtypes[0]
    for d in dtypes[1:]:
        out = torch.promote_types(out, d)
    if compute is not None:
        return compute, out
    half = half_of(dtypes)
    return (torch.float32 if half is None else half), out


def cast_operands(tensors, compute: torch.dtype):
    """Operands in the compute dtype; a matching tensor is passed through."""
    return [
        None if t is None else (t if t.dtype == compute else t.to(compute))
        for t in tensors
    ]


def promote(*tensors, compute: torch.dtype | None = None):
    """(operands in the compute dtype, compute dtype, output dtype).

    The single entry point the autograd layer uses, so a backend kernel keeps
    its one-dtype assumption while callers may mix fp16/bf16/fp32 freely
    across x, the original weight and the module weights.
    """
    compute, out = resolve(*tensors, compute=compute)
    return cast_operands(tensors, compute), compute, out


def folds(base, *operands) -> bool:
    """Whether base + delta(operands) keeps base's dtype, the dtype a kernel
    with the add folded into its store returns."""
    _, out = resolve(*operands)
    return torch.promote_types(base.dtype, out) == base.dtype


def restore(grads, dtypes):
    """Gradients back in each leaf's own dtype."""
    return [
        None if (g is None or d is None) else (g if g.dtype == d else g.to(d))
        for g, d in zip(grads, dtypes)
    ]
