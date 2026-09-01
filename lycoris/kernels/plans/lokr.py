"""Planner for the F3/F4 Kronecker kernels."""

import math

from .cost import (
    TilePlan,
    bandwidth_ms,
    compute_ms,
    pick_limited,
    pipe_tax,
    rank,
    warps_legal,
    wave_eff,
)
from .device import Device
from .tune import SHORTLIST

BLOCK = (16, 32, 64, 128, 256)
WARPS = (2, 4, 8)


def _pad16(v: int) -> int:
    return max(16, 1 << (max(1, v) - 1).bit_length())


def topk_rebuild(
    o: int, i: int, add_base: bool, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """Gather-FMA elementwise: one read (optional) + one write per element.

    Traffic is the same for every tile here, so the model alone ties and the
    shortlist would be six arbitrary equals. The tie-breaks are the terms the
    bandwidth model cannot see: a row shorter than a 128 B cache line wastes
    the transaction, and among equals fewer warps and a smaller tile give
    finer wave granularity. An exhaustive sweep at 1280 square put the true
    optimum at (16, 64, 2) — 2.74 us against 5.6 for the untied pick — so the
    space has to reach bm=16 and the ordering has to surface it.
    """
    if not dev.measured:
        return [
            TilePlan(16, 64, 0, 2, 1, math.inf, "safe"),
            TilePlan(32, 64, 0, 4, 1, math.inf, "safe"),
            TilePlan(64, 64, 0, 4, 1, math.inf, "safe"),
        ]
    cands = []
    line = 128
    for bm in BLOCK:
        for bn in BLOCK:
            for warps in WARPS:
                if bm * bn < warps * 32:
                    continue
                tiles = math.ceil(o / bm) * math.ceil(i / bn)
                traffic = o * i * eb * (2 if add_base else 1)
                ms = bandwidth_ms(traffic, dev) / max(wave_eff(tiles, dev), 1e-3)
                if bn * eb < line:
                    ms *= float(line) / (bn * eb)
                ms *= 1.0 + 0.01 * warps + 0.001 * bm
                cands.append(TilePlan(bm, bn, 0, warps, 1, ms, "dram"))
    return rank(cands, topk)


def topk_rebuild_bwd(
    a: int, b: int, c: int, d: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """Both reduction kernels share a tile plan; grad read dominates."""
    if not dev.measured:
        return [
            TilePlan(32, 64, 0, 4, 1, math.inf, "safe"),
            TilePlan(16, 64, 0, 2, 1, math.inf, "safe"),
        ]
    cands = []
    for bm in (16, 32, 64):
        for bn in (32, 64, 128):
            for warps in (2, 4):
                if bm * bn < warps * 32:
                    continue
                traffic = 2 * (a * c) * (b * d) * eb
                tiles = max(a * b, math.ceil(c / bm) * math.ceil(d / bn))
                ms = bandwidth_ms(traffic, dev) / max(wave_eff(tiles, dev), 1e-3)
                cands.append(TilePlan(bm, bn, 0, warps, 1, ms, "dram"))
    return rank(cands, topk)


def topk_apply(
    t: int,
    a: int,
    b: int,
    c: int,
    d: int,
    eb: int,
    dev: Device,
    topk: int = SHORTLIST,
) -> list[TilePlan]:
    """F4 grouped apply: bt tokens per CTA, two stacked small gemms in SMEM."""
    pa, pb, pc, pd = (_pad16(v) for v in (a, b, c, d))
    inner = max(pb, pc)

    def shapes_of(bt):
        return [(bt * pb, pc), (bt * pc, pa)]

    def smem_of(bt):
        return (bt * pb * pd + pd * pc + pb * pa + 2 * bt * pb * pc + bt * pc * pb) * eb

    bts = sorted({max(1, v) for v in (128 // inner, 64 // inner, 32 // inner, 1)})
    safe = [
        TilePlan(bt, 0, 0, w, 1, math.inf, "safe")
        for bt in bts
        for w in (2, 4)
        if warps_legal(shapes_of(bt), w) and smem_of(bt) <= dev.smem_per_cta
    ] or [TilePlan(1, 0, 0, 1, 1, math.inf, "safe")]
    if not dev.measured:
        return safe
    cands = []
    for bt in bts:
        for warps in WARPS:
            if not warps_legal(shapes_of(bt), warps):
                continue
            if smem_of(bt) > dev.smem_per_cta:
                continue
            tiles = math.ceil(t / bt)
            traffic = t * (b * d + a * c) * eb
            flops = 2.0 * t * (pb * pd * pc + pc * pb * pa)
            ms, limiter = pick_limited(
                bandwidth_ms(traffic, dev),
                compute_ms(flops, dev, pipe_tax(dev, 1)),
            )
            ms /= max(wave_eff(tiles, dev), 1e-3)
            cands.append(TilePlan(bt, 0, 0, warps, 1, ms, limiter))
    return rank(cands, topk) if cands else safe
