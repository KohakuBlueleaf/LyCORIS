"""Error statistics of a kernel and of eager torch at the same precision, both against
an fp64 reference: cosine, and abs / rel error each as mean / median / min / max.

A tensor is flagged only on explosion: not finite, a cosine deficit (1 - cos) of at
least ``BLOWUP`` times eager's, or BOTH its abs and its rel error of the same summary
(mean, or median) at least ``BLOWUP`` times eager's. Every comparison is appended
to the JSON-lines file named by ``LYCORIS_PRECISION_LOG`` when set, for the written
comparison against PyTorch.
"""

import json
import os
from typing import NamedTuple

import torch
from torch import Tensor

BLOWUP = 10.0
# Deficits and errors below these are rounding noise of the fp64 reference itself.
_FLOOR = 1e-12


class Stats(NamedTuple):
    finite: bool
    cos: float
    abs_mean: float
    abs_median: float
    abs_min: float
    abs_max: float
    rel_mean: float
    rel_median: float
    rel_min: float
    rel_max: float


def stats(out: Tensor, ref: Tensor) -> Stats:
    """``Stats`` of ``out`` against the fp64 ``ref`` (same shape)."""
    out = out.detach().double().flatten()
    ref = ref.detach().double().flatten()
    if not bool(torch.isfinite(out).all()):
        inf = float("inf")
        return Stats(False, -inf, inf, inf, inf, inf, inf, inf, inf, inf)
    d = (out - ref).abs()
    nz = ref != 0
    rel = d[nz] / ref[nz].abs() if bool(nz.any()) else d.new_zeros(1)
    norm = out.norm() * ref.norm()
    cos = float((out @ ref) / norm) if norm > 0 else float(out.norm() == ref.norm())
    return Stats(
        True,
        cos,
        d.mean().item(),
        d.median().item(),
        d.min().item(),
        d.max().item(),
        rel.mean().item(),
        rel.median().item(),
        rel.min().item(),
        rel.max().item(),
    )


def exploded(cand: Stats, eager: Stats, blowup: float = BLOWUP) -> str | None:
    """The reason ``cand`` exploded against ``eager``, or None."""
    if not cand.finite:
        return "not finite"
    deficit = max(1.0 - cand.cos, _FLOOR)
    if deficit >= blowup * max(1.0 - eager.cos, _FLOOR):
        return f"cosine deficit {deficit:.3e} vs eager {1.0 - eager.cos:.3e}"
    for s in ("mean", "median"):
        ca, cr = getattr(cand, f"abs_{s}"), getattr(cand, f"rel_{s}")
        ea, er = getattr(eager, f"abs_{s}"), getattr(eager, f"rel_{s}")
        if ca >= blowup * max(ea, _FLOOR) and cr >= blowup * max(er, _FLOOR):
            return f"abs_{s} {ca:.3e} vs {ea:.3e} and rel_{s} {cr:.3e} vs {er:.3e}"
    return None


def worse(cand: Stats, eager: Stats) -> list[str]:
    """The summaries where ``cand`` is worse than eager (for the written comparison)."""
    out = []
    if cand.cos < eager.cos:
        out.append("cos")
    for f in Stats._fields[2:]:
        if getattr(cand, f) > getattr(eager, f):
            out.append(f)
    return out


def record(what: str, cand: Stats, eager: Stats) -> None:
    """Append one comparison to ``LYCORIS_PRECISION_LOG`` when it is set."""
    path = os.environ.get("LYCORIS_PRECISION_LOG")
    if not path:
        return
    row = {
        "what": what,
        "kernel": cand._asdict(),
        "eager": eager._asdict(),
        "worse": worse(cand, eager),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def judge(what: str, out: Tensor, eager_out: Tensor, ref: Tensor) -> str | None:
    """Record kernel and eager stats of ``what`` against ``ref``; the explosion, or None."""
    cand, eager = stats(out, ref), stats(eager_out, ref)
    record(what, cand, eager)
    why = exploded(cand, eager)
    return None if why is None else f"{what}: {why}"
