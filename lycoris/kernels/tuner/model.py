"""Analytical cost model: ranks a kernel's legal configs on this device without
running them, from driver limits and rates measured once per process.
See docs/kernels/tuning.md.
"""

import functools
import itertools
import math
from dataclasses import dataclass

import torch

from .device import DeviceInfo

try:
    from . import probes
except ImportError:
    probes = None

_REG_UNIT = 256
_SUB_PARTITIONS = 4
_MAX_REGS_THREAD = 255
_BASE_REGS = 40
# DRAM latency under load before the in-flight curve is measured (capture or compile first).
_DRAM_LATENCY_S = 1e-6
# Program launch and drain per wave.
_LAUNCH_S = 2e-6
_FMA_ITERS = 4096


@dataclass(frozen=True)
class Work:
    """What one launch of one config moves and computes.

    phases lists the program's streaming loops as (DRAM bytes of the launch in
    that loop, bytes one program has in flight in it, pipeline stages included);
    without it the launch is one loop of hbm_bytes with smem_stage x stages (or
    operand_bytes) in flight.
    """

    grid: int
    warps: int
    acc_fp32: int  # fp32 accumulator elements per program
    operand_bytes: int  # register-resident operand bytes per program
    smem_stage: int  # shared-memory bytes per pipeline stage per program
    stages: int
    hbm_bytes: float  # DRAM bytes of the whole launch
    mma_flops: float = 0.0
    vector_flops: float = 0.0
    # bytes all programs load, re-reads of shared operands included
    l2_bytes: float = 0.0
    phases: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Throughput:
    dram_bps: float
    l2_bps: float
    # (bytes in flight per SM, DRAM bytes/s with every SM holding that many), ascending
    inflight_curve: tuple[tuple[float, float], ...]
    mma_flops: float
    fma_flops: float

    def stream_bps(self, per_sm: float, active_share: float) -> float:
        """DRAM bytes/s when active_share of the SMs each keep per_sm bytes in flight."""
        curve = self.inflight_curve
        if per_sm <= curve[0][0]:
            bps = curve[0][1] * per_sm / curve[0][0]
        elif per_sm >= curve[-1][0]:
            bps = curve[-1][1]
        else:
            for (x0, y0), (x1, y1) in itertools.pairwise(curve):
                if per_sm <= x1:
                    f = math.log(per_sm / x0) / math.log(x1 / x0)
                    bps = y0 + f * (y1 - y0)
                    break
        return min(self.dram_bps, bps * active_share)


def _capturing(dev: DeviceInfo) -> bool:
    if dev.kind == "cuda" and torch.cuda.is_current_stream_capturing():
        return True
    return torch.compiler.is_compiling()


def _best_seconds(dev: DeviceInfo, fn, reps: int = 5) -> float:
    api = getattr(torch, dev.kind)
    fn()
    best = math.inf
    for _ in range(reps):
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3)
    return best


@functools.cache
def _measure(dev: DeviceInfo) -> Throughput:
    device = torch.device(dev.kind, dev.index)
    # Random data (constant buffers compress in memory on some parts).
    buf = torch.randn(2, 64 * 1024 * 1024, device=device, dtype=torch.float32)
    dram = (
        2
        * buf[0].nbytes
        / _best_seconds(dev, lambda: torch.mul(buf[0], 1.0, out=buf[1]))
    )
    n = 4096
    a16 = torch.randn(n, n, device=device, dtype=torch.float16)
    mma = 2 * n**3 / _best_seconds(dev, lambda: a16 @ a16)
    if probes is None:
        # No kernel language to probe with: a flat in-flight curve and the
        # elementwise rate of torch for the vector unit.
        x = buf[0][: 16 * 1024 * 1024]
        fma = 2.0 * x.numel() / _best_seconds(dev, lambda: torch.addcmul(x, x, x))
        curve = ((1024.0, dram), (65536.0, dram))
        return Throughput(dram, 2 * dram, curve, mma, fma)

    # L2: every program re-reads a buffer a quarter of L2's size.
    block, nprog = 1024, 8 * dev.units
    src = torch.randn(dev.l2_bytes // 16, device=device, dtype=torch.float32)
    out = torch.empty(nprog * block, device=device, dtype=torch.float32)
    reps = max(1, (1 << 31) // (nprog * block * 4))
    seconds = _best_seconds(
        dev,
        lambda: probes.l2_read_kernel[(nprog,)](
            src, out, src.numel(), reps, block=block
        ),
    )
    l2 = nprog * reps * block * 4 / seconds

    # Bandwidth against bytes in flight: p one-warp programs per SM, one dependent 1 KiB tile each.
    tile = 256
    curve = []
    for per_sm in (1, 2, 4, 8, 16, 32, 64):
        if per_sm * 32 > dev.max_threads_sm or per_sm > dev.max_blocks_sm:
            break
        grid = dev.units * per_sm
        per_prog = (buf[0].numel() // grid) // tile * tile
        out = torch.empty(grid * tile, device=device, dtype=torch.float32)
        seconds = _best_seconds(
            dev,
            lambda grid=grid, per_prog=per_prog, out=out: probes.chain_read_kernel[
                (grid,)
            ](buf[0], out, per_prog, block=tile, num_warps=1, num_stages=1),
        )
        curve.append((per_sm * tile * 4.0, grid * per_prog * 4 / seconds))

    grid = dev.units * 8
    out = torch.empty(grid * block, device=device, dtype=torch.float32)
    seconds = _best_seconds(
        dev,
        lambda: probes.fma_kernel[(grid,)](out, _FMA_ITERS, block=block, num_warps=4),
    )
    fma = 2.0 * grid * block * _FMA_ITERS / seconds
    return Throughput(dram, max(l2, dram), tuple(curve), mma, fma)


def throughput(dev: DeviceInfo) -> Throughput | None:
    """Measured rates; None while capturing before the first measurement."""
    if _measure.cache_info().currsize == 0 and _capturing(dev):
        return None
    return _measure(dev)


def _smem_unit(dev: DeviceInfo) -> int:
    return 256 if dev.capability is not None and dev.capability < (8, 0) else 128


def footprint(dev: DeviceInfo, w: Work) -> tuple[int, int]:
    """(registers per thread, shared-memory bytes per block) of one program."""
    threads = 32 * w.warps
    regs = (
        _BASE_REGS
        + math.ceil(w.acc_fp32 / threads)
        + math.ceil(w.operand_bytes / 4 / threads)
    )
    smem = w.smem_stage * max(w.stages, 1)
    return regs, smem


def blocks_per_sm(dev: DeviceInfo, w: Work) -> int:
    """Resident programs per SM by the occupancy calculator's rules; 0 when one does not fit."""
    threads = 32 * w.warps
    regs, smem = footprint(dev, w)
    if threads > dev.max_threads_sm or regs > _MAX_REGS_THREAD or smem > dev.smem:
        return 0
    per_warp = math.ceil(regs * 32 / _REG_UNIT) * _REG_UNIT
    warps_per_sub = (dev.regs_sm // _SUB_PARTITIONS) // per_warp
    unit = _smem_unit(dev)
    smem_block = math.ceil(smem / unit) * unit + dev.smem_reserved
    return max(
        0,
        min(
            dev.max_blocks_sm,
            dev.max_threads_sm // threads,
            warps_per_sub * _SUB_PARTITIONS // w.warps,
            dev.smem_sm // smem_block if smem_block else dev.max_blocks_sm,
        ),
    )


def predict(dev: DeviceInfo, w: Work, mma: bool | None = None) -> float | None:
    """Model seconds of one launch, or None when a program cannot be resident.
    ``mma`` says whether the kernel's dots reach matrix units (default: the
    device's Triton rule)."""
    blocks = blocks_per_sm(dev, w)
    if blocks < 1:
        return None
    slots = dev.units * blocks
    waves = max(1, math.ceil(w.grid / slots))
    wave_eff = w.grid / (waves * slots)
    rates = throughput(dev)

    # Programs resident in each wave: full waves, then the remainder.
    full, rest = divmod(w.grid, slots)
    resident = [(full * slots / w.grid, slots)] if full else []
    if rest:
        resident.append((rest / w.grid, rest))

    # Each loop streams at the measured bandwidth for the bytes its resident programs keep in flight.
    depth = w.stages if dev.async_copy else 1
    in_flight = w.smem_stage * depth if w.smem_stage else w.operand_bytes
    phases = w.phases or ((w.hbm_bytes, in_flight),)
    mem_s = 0.0
    for nbytes, flight in phases:
        for share, programs in resident:
            active = min(dev.units, programs)
            if rates:
                per_sm = programs * max(flight, 1.0) / active
                bps = rates.stream_bps(per_sm, active / dev.units)
            else:
                bps = min(1e12, programs * max(flight, 1.0) / _DRAM_LATENCY_S)
            mem_s += share * nbytes / bps
    if rates:
        mem_s = max(mem_s, w.l2_bytes / rates.l2_bps)

    compute_s = 0.0
    if rates:
        on_mma = dev.mma if mma is None else mma
        mma_rate = rates.mma_flops if on_mma else rates.fma_flops
        compute_s = w.mma_flops / mma_rate + w.vector_flops / rates.fma_flops
        compute_s /= max(wave_eff, 1e-3)
    return max(mem_s, compute_s) + waves * _LAUNCH_S


def rank(
    dev: DeviceInfo, configs: list[dict], work, mma: bool | None = None
) -> list[dict]:
    """Kernel configs (no "impl" key) that fit, best first; work(config) -> Work."""
    scored = []
    for c in configs:
        if "impl" in c:
            continue
        t = predict(dev, work(c), mma)
        if t is None:
            continue
        scored.append(((t,), len(scored), c))
    scored.sort(key=lambda s: (s[0], s[1]))
    return [c for _, _, c in scored]


def shared_reads(dev: DeviceInfo, nbytes: float, readers: int) -> float:
    """DRAM bytes of an operand every program reads: once while it fits half of L2, else per reader."""
    return nbytes if nbytes <= dev.l2_bytes / 2 else nbytes * readers


def staged(dev: DeviceInfo, nbytes: float, stages: int) -> float:
    """Bytes in flight of a loop that loads nbytes per iteration over stages pipeline stages."""
    return nbytes * (stages if dev.async_copy else 1)


def tiled_reads(rows: int, cols: int, k: int, bm: int, bn: int, e: float) -> float:
    """Bytes all (bm, bn) output tiles of a (rows, cols) product over k load: each its A and B panels."""
    return cdiv(rows, bm) * cdiv(cols, bn) * (bm + bn) * k * e


def cdiv(a: int, b: int) -> int:
    return -(-a // b)
