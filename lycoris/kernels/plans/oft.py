"""Planner for the F5 block-diagonal kernels (Diag-OFT)."""

import math

from .cost import TilePlan, bandwidth_ms, rank, warps_legal, wave_eff
from .device import Device
from .tune import SHORTLIST

BLOCK = (32, 64, 128, 256)
WARPS = (2, 4, 8)


def _ps(s: int) -> int:
    return max(16, 1 << (max(1, s) - 1).bit_length())


def _topk_stream(
    rows_mixed: int,
    stream: int,
    s: int,
    k: int,
    eb: int,
    dev: Device,
    grad: bool,
    topk: int,
) -> list[TilePlan]:
    """Shared shape logic: (ps x bn) apply tiles or (ps x ps) grad reductions
    streaming ``stream`` columns/tokens per block row group."""
    ps = _ps(s)
    safe = [
        TilePlan(0, bn, 0, w, 1, math.inf, "safe")
        for bn in (64, 32)
        for w in (2, 1)
        if warps_legal([(ps, ps) if grad else (ps, bn)], w)
    ] or [TilePlan(0, 32, 0, 1, 1, math.inf, "safe")]
    if not dev.measured:
        return safe
    cands = []
    for bn in BLOCK:
        for warps in WARPS:
            shape = (ps, ps) if grad else (ps, bn)
            if not warps_legal([shape], warps):
                continue
            tiles = k if grad else k * math.ceil(stream / bn)
            traffic = 2.0 * rows_mixed * stream * eb
            ms = bandwidth_ms(traffic, dev) / max(wave_eff(tiles, dev), 1e-3)
            cands.append(TilePlan(0, bn, 0, warps, 1, ms, "dram"))
    return rank(cands, topk) if cands else safe


def topk_fused(
    k: int, s: int, stream: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """Fused Cayley+apply kernels: bm = col_groups, bn = column tile."""
    ps = _ps(s)
    safe = [
        TilePlan(cg, bn, 0, w, 1, math.inf, "safe")
        for cg in (4, 1)
        for bn in (64, 32)
        for w in (2,)
        if warps_legal([(ps, bn)], w)
    ] or [TilePlan(1, 32, 0, 1, 1, math.inf, "safe")]
    if not dev.measured:
        return safe
    cands = []
    for cg in (1, 2, 4, 8, 16):
        for bn in BLOCK:
            for warps in WARPS:
                for stages in dev.pipeline_depths():
                    if not warps_legal([(ps, bn), (ps, ps)], warps):
                        continue
                    if cg * bn > stream * 2:
                        continue
                    if ps * bn * eb * stages > dev.smem_per_cta:
                        continue
                    tiles = k * cg
                    traffic = 2.0 * k * s * stream * eb
                    ms = bandwidth_ms(traffic, dev) / max(wave_eff(tiles, dev), 1e-3)
                    # Overlap the model cannot see; the depth is measured.
                    ms *= 1.0 if stages == 1 else 0.9
                    cands.append(TilePlan(cg, bn, 0, warps, stages, ms, "dram"))
    return rank(cands, topk) if cands else safe


def topk_apply(
    k: int, s: int, stream: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    return _topk_stream(k * s, stream, s, k, eb, dev, grad=False, topk=topk)


def topk_grad_r(
    k: int, s: int, stream: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    return _topk_stream(k * s, stream, s, k, eb, dev, grad=True, topk=topk)
