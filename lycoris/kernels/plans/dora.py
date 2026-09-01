"""Planner for the F7 pointwise kernels."""

import math

from .cost import TilePlan, bandwidth_ms, rank, wave_eff
from .device import Device
from .tune import SHORTLIST

BLOCKS = (256, 512, 1024, 2048, 4096)
WARPS = (2, 4, 8)
ROW_BM = (8, 16, 32)
ROW_BN = (128, 256, 512)


def topk_elementwise(
    n: int, passes: float, dev: Device, eb: int, topk: int = SHORTLIST
) -> list[TilePlan]:
    """1D kernels; ``passes`` = logical bytes moved per element / eb."""
    safe = [
        TilePlan(1024, 0, 0, 4, 1, math.inf, "safe"),
        TilePlan(256, 0, 0, 4, 1, math.inf, "safe"),
    ]
    if not dev.measured:
        return safe
    cands = []
    for blk in BLOCKS:
        for warps in WARPS:
            if blk < warps * 32:
                continue
            tiles = math.ceil(n / blk)
            ms = bandwidth_ms(n * passes * eb, dev) / max(wave_eff(tiles, dev), 1e-3)
            cands.append(TilePlan(blk, 0, 0, warps, 1, ms, "dram"))
    return rank(cands, topk)


def topk_row_reduce(
    rows: int, cols: int, dev: Device, eb: int, topk: int = SHORTLIST
) -> list[TilePlan]:
    """Row-block reductions and rowwise scales (DoRA)."""
    safe = [
        TilePlan(16, 256, 0, 4, 1, math.inf, "safe"),
        TilePlan(8, 128, 0, 2, 1, math.inf, "safe"),
    ]
    if not dev.measured:
        return safe
    cands = []
    for bm in ROW_BM:
        for bn in ROW_BN:
            for warps in (2, 4):
                if bm * bn < warps * 32:
                    continue
                tiles = math.ceil(rows / bm)
                ms = bandwidth_ms(rows * cols * eb, dev) / max(
                    wave_eff(tiles, dev), 1e-3
                )
                cands.append(TilePlan(bm, bn, 0, warps, 1, ms, "dram"))
    return rank(cands, topk)
