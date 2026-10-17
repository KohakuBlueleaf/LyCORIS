"""The bar a kernel output meets: no explosion against eager torch at the same precision.

Every output and every gradient is compared with an fp64 reference by two
errors, ``abs`` = max|out - ref| and ``rel`` = max |out - ref| / |ref| over the
elements where ref != 0. A tensor is flagged when it is not finite, or when
BOTH its abs and its rel error exceed ``FACTOR`` times the eager composition's
own, run in the same dtype.
"""

from typing import NamedTuple

import torch
from torch import Tensor

FACTOR = 2.0


class Err(NamedTuple):
    """One tensor's error against its fp64 reference."""

    abs: float
    rel: float
    finite: bool


def error(out: Tensor, ref: Tensor) -> Err:
    """``Err`` of ``out`` against the fp64 reference ``ref`` (same shape)."""
    out = out.detach()
    if not bool(torch.isfinite(out).all()):
        return Err(float("inf"), float("inf"), False)
    ref = ref.detach().double()
    d = (out.double() - ref).abs()
    nz = ref != 0
    rel = (d[nz] / ref[nz].abs()).max().item() if bool(nz.any()) else 0.0
    return Err(d.max().item() if d.numel() else 0.0, rel, True)


def errors(outs: list[Tensor], refs: list[Tensor]) -> list[Err]:
    """``error`` of each tensor against its reference."""
    return [error(o, r) for o, r in zip(outs, refs)]


def flagged(cand: Err, eager: Err, factor: float = FACTOR) -> bool:
    """Non-finite, or worse than ``factor`` x eager in both abs and rel error."""
    if not cand.finite:
        return True
    return cand.abs > factor * eager.abs and cand.rel > factor * eager.rel


def passes(cand: list[Err], eager: list[Err], factor: float = FACTOR) -> bool:
    """No tensor of ``cand`` is flagged against the eager errors."""
    return not any(flagged(c, e, factor) for c, e in zip(cand, eager))
