"""Shared legality and cost-model pieces for the family planners.

Ranking only has to be monotone-correct: the tuner times the shortlist and
absorbs model error inside it. Legality must be exact — an illegal candidate
wastes a tuning slot at best and a compile failure at worst.
"""

import dataclasses
import math

from .device import Device

MMA_M, MMA_N = 16, 8


@dataclasses.dataclass(frozen=True)
class TilePlan:
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int
    predicted_ms: float
    limiter: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


SENTINEL_EAGER = TilePlan(0, 0, 0, 0, 1, float("inf"), "eager")
SENTINEL_MATERIALIZE = TilePlan(0, 0, 0, 0, 1, float("inf"), "materialize")
# BOFT: whole butterfly in one CTA (fewer round trips) against a launch per
# stage (shorter Cayley chain per CTA). Which wins is a shape question.
SENTINEL_CONE = TilePlan(4, 64, 0, 4, 1, float("inf"), "cone")


def mma_n(dev: Device) -> int:
    """Columns a warp's MMA tile covers: 16 on Volta (TileLang's sm_70 GEMM
    partitions warps in 16x16 units), 8 from Turing on."""
    return 16 if (dev.cc_major, dev.cc_minor) == (7, 0) else MMA_N


def warp_tile(
    bm: int, bn: int, warps: int, n_unit: int = MMA_N, exact: bool = False
) -> tuple[int, int]:
    """Warp split of a CTA tile, squarest legal; (0, 0) when none exists.

    ``exact`` additionally requires every warp to own whole MMA tiles (the
    TileLang GEMM partition rule)."""
    best = (0, 0)
    for wm_count in (1, 2, 4, 8):
        wn_count = warps // wm_count
        if wn_count < 1 or wm_count * wn_count != warps:
            continue
        wm, wn = bm // wm_count, bn // wn_count
        if wm < MMA_M or wn < n_unit:
            continue
        if exact and (bm % (wm_count * MMA_M) or bn % (wn_count * n_unit)):
            continue
        if best == (0, 0) or abs(wm - wn) < abs(best[0] - best[1]):
            best = (wm, wn)
    return best


def warps_legal(
    c_shapes: list[tuple[int, int]],
    warps: int,
    n_unit: int = MMA_N,
    exact: bool = False,
) -> bool:
    return all(warp_tile(m, n, warps, n_unit, exact) != (0, 0) for m, n in c_shapes)


def tilelang_legal(cands: list[TilePlan], dev: Device, shapes_of) -> list[TilePlan]:
    """The candidates whose every ``T.gemm`` accumulator (``shapes_of(plan)``)
    partitions over the plan's warps under TileLang's exact rule on ``dev``;
    sentinel plans (no warps) pass. An illegal tile is a compile failure, so it
    never reaches the tuner. Falls back to the input when nothing is legal."""
    n_unit = mma_n(dev)
    legal = [
        p
        for p in cands
        if p.warps == 0 or warps_legal(shapes_of(p), p.warps, n_unit, exact=True)
    ]
    return legal or cands


def occupancy(dev: Device, est_regs: int, smem: int, warps: int) -> int:
    threads = warps * 32
    by_reg = dev.regs_per_sm // max(est_regs * threads, 1)
    by_smem = dev.smem_per_sm // max(smem, 1)
    by_thread = dev.max_threads_per_sm // threads
    return max(min(by_reg, by_smem, by_thread), 0)


def wave_eff(tiles: int, dev: Device, cta_per_sm: int = 1) -> float:
    slots = dev.sms * max(cta_per_sm, 1)
    waves = tiles / slots
    return waves / max(math.ceil(waves), 1)


def pipe_tax(dev: Device, cta_per_sm: int) -> float:
    return dev.bar_tax[min(max(cta_per_sm, 1), len(dev.bar_tax)) - 1]


def bandwidth_ms(traffic_bytes: float, dev: Device) -> float:
    return traffic_bytes / (dev.dram_bw * 1e9) * 1e3


def compute_ms(flops: float, dev: Device, pipe: float) -> float:
    return flops / (dev.mma_peak * 1e12 * max(pipe, 0.05)) * 1e3


def pick_limited(bw_ms: float, mma_ms: float) -> tuple[float, str]:
    if bw_ms >= mma_ms:
        return bw_ms, "dram"
    return mma_ms, "mma"


def rank(cands: list[TilePlan], topk: int) -> list[TilePlan]:
    cands.sort(key=lambda p: (p.predicted_ms, p.warps))
    return cands[:topk]
