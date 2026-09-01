"""Planner for the F6 butterfly kernels (BOFT)."""

import math

from .cost import TilePlan, bandwidth_ms, rank, warps_legal, wave_eff
from .device import Device
from .tune import SHORTLIST

BLOCK = (32, 64, 128, 256)
WARPS = (2, 4, 8)


def _ps(s: int) -> int:
    return max(16, 1 << (max(1, s) - 1).bit_length())


def topk_stage(
    n_blocks: int,
    s: int,
    stream: int,
    m_stages: int,
    eb: int,
    dev: Device,
    topk: int = SHORTLIST,
) -> list[TilePlan]:
    """One plan reused by every stage of one apply chain."""
    ps = _ps(s)
    safe = [
        TilePlan(0, bn, 0, w, 1, math.inf, "safe")
        for bn in (64, 32)
        for w in (2, 1)
        if warps_legal([(ps, bn)], w)
    ] or [TilePlan(0, 32, 0, 1, 1, math.inf, "safe")]
    if not dev.measured:
        return safe
    cands = []
    for bn in BLOCK:
        for warps in WARPS:
            if not warps_legal([(ps, bn)], warps):
                continue
            tiles = n_blocks * math.ceil(stream / bn)
            traffic = 2.0 * m_stages * n_blocks * s * stream * eb
            ms = bandwidth_ms(traffic, dev) / max(wave_eff(tiles, dev), 1e-3)
            cands.append(TilePlan(0, bn, 0, warps, 1, ms, "dram"))
    return rank(cands, topk) if cands else safe


def topk_fused(
    n_blocks: int,
    s: int,
    stream: int,
    m_stages: int,
    eb: int,
    dev: Device,
    topk: int = SHORTLIST,
) -> list[TilePlan]:
    """Cayley-fused stage kernels: bm = col_groups, bn = column tile."""
    ps = _ps(s)
    safe = [
        TilePlan(cg, bn, 0, w, 1, math.inf, "safe")
        for cg in (1, 2)
        for bn in (64, 32)
        for w in (2,)
        if warps_legal([(ps, bn)], w)
    ] or [TilePlan(1, 32, 0, 1, 1, math.inf, "safe")]
    if not dev.measured:
        return safe
    cands = []
    for cg in (1, 2, 4):
        for bn in BLOCK:
            for warps in WARPS:
                for stages in dev.pipeline_depths():
                    if not warps_legal([(ps, bn), (ps, ps)], warps):
                        continue
                    if cg * bn > stream * 2:
                        continue
                    if ps * bn * eb * stages > dev.smem_per_cta:
                        continue
                    tiles = n_blocks * cg
                    traffic = 2.0 * m_stages * n_blocks * s * stream * eb
                    ms = bandwidth_ms(traffic, dev) / max(wave_eff(tiles, dev), 1e-3)
                    # Overlap the model cannot see; the depth is measured.
                    ms *= 1.0 if stages == 1 else 0.9
                    cands.append(TilePlan(cg, bn, 0, warps, stages, ms, "dram"))
    return rank(cands, topk) if cands else safe


def topk_grad_r(
    n_blocks: int, s: int, stream: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    ps = _ps(s)
    safe = [
        TilePlan(0, bn, 0, w, 1, math.inf, "safe")
        for bn in (64, 32)
        for w in (2, 1)
        if warps_legal([(ps, ps)], w)
    ] or [TilePlan(0, 32, 0, 1, 1, math.inf, "safe")]
    if not dev.measured:
        return safe
    cands = []
    for bn in BLOCK:
        for warps in WARPS:
            if not warps_legal([(ps, ps)], warps):
                continue
            traffic = 2.0 * n_blocks * s * stream * eb
            ms = bandwidth_ms(traffic, dev) / max(wave_eff(n_blocks, dev), 1e-3)
            cands.append(TilePlan(0, bn, 0, warps, 1, ms, "dram"))
    return rank(cands, topk) if cands else safe
