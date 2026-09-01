"""Planner for the F1/F2 low-rank kernels (rebuild, delta, backwards)."""

import math

from .cost import (
    TilePlan,
    bandwidth_ms,
    compute_ms,
    occupancy,
    pick_limited,
    pipe_tax,
    rank,
    warps_legal,
    wave_eff,
)
from .device import Device
from .tune import SHORTLIST

BLOCK_M = (32, 64, 128)
BLOCK_N = (32, 64, 128)
BLOCK_K = (32, 64)
WARPS = (2, 4, 8)


def _safe(shapes_of, bk: int = 0, stages: int = 1) -> list[TilePlan]:
    out = []
    for bm, bn in ((64, 64), (32, 32)):
        for warps in (4, 2):
            if warps_legal(shapes_of(bm, bn, bk), warps):
                out.append(TilePlan(bm, bn, bk, warps, stages, math.inf, "safe"))
    return out or [TilePlan(32, 32, bk, 1, stages, math.inf, "safe")]


def _rank_block(r: int) -> int:
    return max(16, 1 << (max(1, r) - 1).bit_length())


def topk_rebuild(
    o: int, i: int, r: int, hada: bool, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    br = min(_rank_block(r), 32)
    shapes_of = lambda bm, bn, bk: [(bm, bn)]
    if not dev.measured:
        return _safe(shapes_of)
    sides = 2 if hada else 1
    cands = []
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for warps in WARPS:
                if not warps_legal([(bm, bn)], warps):
                    continue
                acc = bm * bn / (32 * warps)
                if acc > 168:
                    continue
                smem = (bm * br + br * bn) * eb * sides
                if smem > dev.smem_per_cta:
                    continue
                cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                if cta < 1:
                    continue
                tiles = math.ceil(o / bm) * math.ceil(i / bn)
                traffic = o * i * eb + tiles * (bm + bn) * r * eb * sides
                flops = 2.0 * o * i * r * sides
                ms, limiter = pick_limited(
                    bandwidth_ms(traffic, dev),
                    compute_ms(flops, dev, pipe_tax(dev, cta)),
                )
                ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                cands.append(TilePlan(bm, bn, 0, warps, 1, ms, limiter))
    return rank(cands, topk) if cands else _safe(shapes_of)


def topk_bypass_fwd(
    t: int, o: int, i: int, r: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """LoRA bypass forward: 1D grid over t/bm, h = x@down^T once per CTA,
    then an o-loop emitting y = gamma*h@up^T in bn chunks.

    Splitting O for more programs was measured worse (33.5 us against 10.9 at
    t=512, i=o=1280, r=16): it re-reads x o/bn times. bm is therefore the only
    parallelism lever and small bm is preferred; factor rereads are
    (t/bm)*r*(i+o)*eb and L2-resident.
    """
    br = _rank_block(r)
    shapes_of = lambda bm, bn, bk: [(bm, bk or 32), (bm, bn)]
    if r > 128:
        return []
    if not dev.measured:
        return _safe(shapes_of, bk=32)
    cands = []
    for bm in (16, *BLOCK_M):
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    if not warps_legal(shapes_of(bm, bn, bk), warps):
                        continue
                    acc = (bm * br + bm * bn) / (32 * warps)
                    if acc > 168:
                        continue
                    smem = (bm * bk + br * bk + bn * br + bm * bn) * eb
                    if smem > dev.smem_per_cta:
                        continue
                    cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                    if cta < 1:
                        continue
                    tiles = math.ceil(t / bm)
                    traffic = t * (i + o) * eb + tiles * r * (i + o) * eb
                    flops = 2.0 * t * r * (i + o)
                    ms, limiter = pick_limited(
                        bandwidth_ms(traffic, dev),
                        compute_ms(flops, dev, pipe_tax(dev, cta)),
                    )
                    ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                    for stages in dev.pipeline_depths():
                        if smem * stages > dev.smem_per_cta:
                            continue
                        # Overlap the model cannot see; the depth is measured.
                        scaled = ms * (1.0 if stages == 1 else 0.9)
                        cands.append(
                            TilePlan(bm, bn, bk, warps, stages, scaled, limiter)
                        )
    return rank(cands, topk) if cands else _safe(shapes_of, bk=32)


def topk_bypass_bwd(
    t: int, o: int, i: int, r: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """LoCon bypass backward, one kernel: 1D grid over t/bm. Per CTA:
    i-loop rebuilds h, o-loop accumulates q = g@up and atomics g_up partials,
    second i-loop stores gx = gamma*q@down and atomics g_down partials.

    x is streamed twice (h rebuild + g_down), g once; atomic traffic is
    (t/bm)*r*(o+i)*4 bytes, L2-absorbed at real shapes.
    """
    br = _rank_block(r)
    shapes_of = lambda bm, bn, bk: [(bm, bk or 32), (bm, bn), (bn, br)]
    if r > 128:
        return []
    if not dev.measured:
        return _safe(shapes_of, bk=32)
    cands = []
    for bm in (16, *BLOCK_M):
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    if not warps_legal(shapes_of(bm, bn, bk), warps):
                        continue
                    acc = (2 * bm * br + max(bm * bn, bn * br)) / (32 * warps)
                    if acc > 168:
                        continue
                    smem = (bm * bk + br * bk + bn * br + bm * bn) * eb
                    if smem > dev.smem_per_cta:
                        continue
                    cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                    if cta < 1:
                        continue
                    tiles = math.ceil(t / bm)
                    traffic = (
                        t * (2 * i + o + i) * eb
                        + tiles * 2 * r * (i + o) * eb
                        + tiles * r * (i + o) * 4
                    )
                    flops = 2.0 * t * r * (3 * i + 2 * o)
                    ms, limiter = pick_limited(
                        bandwidth_ms(traffic, dev),
                        compute_ms(flops, dev, pipe_tax(dev, cta)),
                    )
                    ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                    cands.append(TilePlan(bm, bn, bk, warps, 1, ms, limiter))
    return rank(cands, topk) if cands else _safe(shapes_of, bk=32)


def topk_merge_bwd(
    o: int, i: int, r: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """LoCon merge backward, one kernel, role-split 1D grid: pids [0, o/bm)
    reduce g_up rows (i-loop), the rest reduce g_down columns (o-loop).

    G is read twice (once per role) — L2-covered while o*i*eb fits — and both
    roles are full in-register reductions, so there are no atomics.
    """
    br = _rank_block(r)
    shapes_of = lambda bm, bn, bk: [(bm, bk or 32), (br, bn)]
    if r > 128:
        return []
    if not dev.measured:
        return _safe(shapes_of, bk=32)
    cands = []
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    if not warps_legal(shapes_of(bm, bn, bk), warps):
                        continue
                    acc = max(bm * br, br * bn) / (32 * warps)
                    if acc > 168:
                        continue
                    smem = (bm * bk + br * bk + bk * bn + br * bn) * eb
                    if smem > dev.smem_per_cta:
                        continue
                    cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                    if cta < 1:
                        continue
                    tiles = math.ceil(o / bm) + math.ceil(i / bn)
                    traffic = 2.0 * o * i * eb + r * (o + i) * eb
                    flops = 4.0 * o * i * r
                    ms, limiter = pick_limited(
                        bandwidth_ms(traffic, dev),
                        compute_ms(flops, dev, pipe_tax(dev, cta)),
                    )
                    ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                    cands.append(TilePlan(bm, bn, bk, warps, 1, ms, limiter))
    return rank(cands, topk) if cands else _safe(shapes_of, bk=32)


def topk_delta(
    t: int, o: int, i: int, r: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """F2 generated-B apply: grid (t/bm, o/bn), k-loop over i in bk steps."""
    br = _rank_block(r)
    shapes_of = lambda bm, bn, bk: [(bk or 32, bn), (bm, bn)]
    if not dev.measured:
        return _safe(shapes_of, bk=32)
    cands = []
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    for stages in (1, 2, 3):
                        if not warps_legal(shapes_of(bm, bn, bk), warps):
                            continue
                        acc = (bm * bn + 2 * bk * bn) / (32 * warps)
                        if acc > 168:
                            continue
                        smem = (bm * bk + 2 * bn * br + 2 * br * bk + bk * bn) * eb
                        if smem * max(1, stages - 1) > dev.smem_per_cta:
                            continue
                        cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                        if cta < 1:
                            continue
                        tiles = math.ceil(t / bm) * math.ceil(o / bn)
                        flops = 2.0 * t * o * i + 4.0 * r * o * i * math.ceil(t / bm)
                        traffic = (t * i + t * o) * eb + tiles * math.ceil(i / bk) * (
                            (bn + bk) * r * eb * 2
                        )
                        ms, limiter = pick_limited(
                            bandwidth_ms(traffic, dev),
                            compute_ms(flops, dev, pipe_tax(dev, cta)),
                        )
                        ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                        cands.append(TilePlan(bm, bn, bk, warps, stages, ms, limiter))
    return rank(cands, topk) if cands else _safe(shapes_of, bk=32)


def topk_delta_dw(
    t: int, o: int, i: int, r: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """F2 factor grads: grid (o/bm, i/bn), t-loop in bk steps, atomic outputs."""
    br = _rank_block(r)
    shapes_of = lambda bm, bn, bk: [(bm, bn), (bm, br), (br, bn)]
    if not dev.measured:
        return _safe(shapes_of, bk=32)
    cands = []
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    for stages in (1, 2, 3):
                        if not warps_legal(shapes_of(bm, bn, bk), warps):
                            continue
                        acc = bm * bn / (32 * warps)
                        if acc > 168:
                            continue
                        smem = (bk * bm + bk * bn + bm * br + br * bn + bm * bn) * eb
                        if smem * max(1, stages - 1) > dev.smem_per_cta:
                            continue
                        cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                        if cta < 1:
                            continue
                        tiles = math.ceil(o / bm) * math.ceil(i / bn)
                        flops = 2.0 * t * o * i
                        traffic = (t * o + t * i) * eb * max(1, tiles // dev.sms) ** 0.5
                        ms, limiter = pick_limited(
                            bandwidth_ms(traffic, dev),
                            compute_ms(flops, dev, pipe_tax(dev, cta)),
                        )
                        ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                        cands.append(TilePlan(bm, bn, bk, warps, stages, ms, limiter))
    return rank(cands, topk) if cands else _safe(shapes_of, bk=32)


def topk_hada_bwd(
    o: int, i: int, r: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """F1 hadamard merge backward: role-split 1D grid, o/bm tiles reducing
    over I (ga1, ga2) ++ i/bn tiles reducing over O (gb1, gb2).

    Each CTA holds its whole reduction in registers, so there are no atomics
    and the grads are plain dtype stores; the cost is reading G twice
    (2*o*i*eb, L2-covered) and regenerating p1/p2 per tile (6r vs 4r FLOP per
    output element). Waves are o/bm + i/bn.
    """
    br = _rank_block(r)
    shapes_of = lambda bm, bn, bk: [(bm, bn), (bm, br), (br, bn)]
    if not dev.measured:
        return _safe(shapes_of)
    cands = []
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for warps in WARPS:
                if not warps_legal(shapes_of(bm, bn, 0), warps):
                    continue
                acc = (bm * bn + 2 * max(bm * br, br * bn)) / (32 * warps)
                if acc > 168:
                    continue
                smem = (2 * bm * br + 2 * br * bn + bm * bn) * eb
                if smem > dev.smem_per_cta:
                    continue
                cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                if cta < 1:
                    continue
                tiles = math.ceil(o / bm) + math.ceil(i / bn)
                flops = 2.0 * o * i * (6 * r)
                traffic = 2.0 * o * i * eb + 2 * r * (o + i) * eb
                ms, limiter = pick_limited(
                    bandwidth_ms(traffic, dev),
                    compute_ms(flops, dev, pipe_tax(dev, cta)),
                )
                ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                cands.append(TilePlan(bm, bn, 0, warps, 1, ms, limiter))
    return rank(cands, topk) if cands else _safe(shapes_of)


def topk_hada_bypass_bwd(
    t: int, o: int, i: int, r: int, eb: int, dev: Device, topk: int = SHORTLIST
) -> list[TilePlan]:
    """LoHa bypass backward, one kernel, role-split: pids below o/bm*i/bn are
    dw tiles (t-loop building gDeltaW then four atomic factor grads), the
    rest are dx tiles (t/bm x i/bn, o-loop with the generated-W tile).

    Both roles stream ~2*t*o*i MACs; the grid is their concatenation so the
    waves see one launch.
    """
    br = _rank_block(r)
    shapes_of = lambda bm, bn, bk: [(bm, bn), (bm, br), (br, bn)]
    if r > 128:
        return []
    if not dev.measured:
        return _safe(shapes_of, bk=32)
    cands = []
    for bm in BLOCK_M:
        for bn in BLOCK_N:
            for bk in BLOCK_K:
                for warps in WARPS:
                    if not warps_legal(shapes_of(bm, bn, bk), warps):
                        continue
                    acc = (bm * bn + 2 * max(bm * br, br * bn)) / (32 * warps)
                    if acc > 168:
                        continue
                    smem = (bm * bk + bk * bn + 2 * bm * br + 2 * br * bn) * eb
                    if smem > dev.smem_per_cta:
                        continue
                    cta = occupancy(dev, int(acc) + dev.reg_overhead, smem, warps)
                    if cta < 1:
                        continue
                    dw_tiles = math.ceil(o / bm) * math.ceil(i / bn)
                    dx_tiles = math.ceil(t / bm) * math.ceil(i / bn)
                    tiles = dw_tiles + dx_tiles
                    flops = 4.0 * t * o * i + 4.0 * r * o * i * (1 + dx_tiles)
                    traffic = 2 * (t * o + t * i) * eb + dw_tiles * (bm + bn) * r * (
                        2 * eb + 4
                    )
                    ms, limiter = pick_limited(
                        bandwidth_ms(traffic, dev),
                        compute_ms(flops, dev, pipe_tax(dev, cta)),
                    )
                    ms /= max(wave_eff(tiles, dev, cta), 1e-3)
                    cands.append(TilePlan(bm, bn, bk, warps, 1, ms, limiter))
    return rank(cands, topk) if cands else _safe(shapes_of, bk=32)
