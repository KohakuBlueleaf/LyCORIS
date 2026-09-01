"""Shared measurement harness for the kernel benches.

Conventions (bench-spec.md, from KohakUwULLM's benchmarking.md): best-of-N
CUDA-event timing after warmup with a 256 MiB L2 flush between iterations,
arms interleaved round-robin keeping each arm's best round, every row also
carrying host-issue time so wall can be separated from device (wall alone was
measured wrong by >2x there on small kernels), peak VRAM per arm, ULP against
an fp64 oracle in the same row, fwd+bwd arms asserting the backward ran.
Measure scripts write JSON; plot scripts draw it, never re-measure.
"""

import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from lycoris.kernels.plans.device import resolve_device

_FLUSH_BYTES = 256 * 1024 * 1024
_flush_buffers: dict[int, torch.Tensor] = {}


def flush_l2() -> None:
    """Evict whatever the previous iteration left in L2 (keyed per device)."""
    index = torch.cuda.current_device()
    buf = _flush_buffers.get(index)
    if buf is None:
        buf = torch.empty(_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
        _flush_buffers[index] = buf
    buf.zero_()


def gpu_busy(index: int = 0) -> tuple[int, int]:
    uuid = f"GPU-{torch.cuda.get_device_properties(index).uuid}"
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={uuid}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        util, mem = out.strip().split(",")
        return int(util), int(mem)
    except (OSError, ValueError, subprocess.SubprocessError):
        return -1, -1


def bench_ms(fn, warmup: int = 25, iters: int = 50, flush: bool = True) -> float:
    """MEDIAN wall time in ms — device PLUS host dispatch.

    Median, not min, so it pairs coherently with the mean host-issue time:
    host_share against a best-case wall reads above 100% and says nothing.
    Best-of lives one level up, across interleaved rounds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    beg = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush:
            flush_l2()
        torch.cuda.synchronize()
        beg[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(beg, end))


def device_ms(fn, warmup: int = 20, iters: int = 50, rounds: int = 3) -> float:
    """Best of ``rounds`` profile windows, each averaging ``iters`` calls.

    A single window over 10 calls put run-to-run spread on a 8 us row at 3x,
    which is larger than the differences being judged.
    """
    return min(_device_window(fn, warmup, iters) for _ in range(rounds))


def _device_window(fn, warmup: int, iters: int) -> float:
    """Summed device-kernel time per call, in ms — device work, no dispatch.

    From the profiler rather than a CUDA-graph replay: capture cannot express
    a path that reads device memory on the host (``torch.linalg.inv`` in the
    OFT reference is one), and a failed capture invalidates the stream, which
    then fails the NEXT unrelated call. Summing kernel durations also works
    for a backward, which graph capture frequently cannot take at all.

    Gaps between kernels are excluded by construction, so this is device work
    and not a wall time; the launch paths here are serial, so summing
    per-kernel self time does not double-count concurrency.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = sum(
        getattr(e, "self_device_time_total", None)
        or getattr(e, "self_cuda_time_total", 0.0)
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    )
    return total_us / iters / 1e3


def host_ms(fn, warmup: int = 10, iters: int = 30) -> float:
    """Host time to ISSUE ``fn``, with no device sync inside the loop."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) * 1e3 / iters
    torch.cuda.synchronize()
    return elapsed


def interleave(
    arms: dict, rounds: int = 3, warmup: int = 10, iters: int = 20, device: bool = True
) -> dict:
    """(wall, host, device) per arm.

    Arms run round-robin and each keeps its best ROUND (the reference measures
    1.8% of thermal drift between the first and fifth timing of one config),
    where a round is itself a median. ``device`` is profiler kernel time, the
    only honest comparator once host dispatch dominates the wall.
    """
    best = {name: float("inf") for name in arms}
    for _ in range(rounds):
        for name, fn in arms.items():
            best[name] = min(best[name], bench_ms(fn, warmup, iters))
    return {
        name: (best[name], host_ms(fn), device_ms(fn) if device else float("nan"))
        for name, fn in arms.items()
    }


def peak_vram(fn) -> int:
    # The flush buffer is allocated lazily, so materialise it before the reset
    # or the first row reports 256 MiB less peak than every row after it.
    flush_l2()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def rel_err(got: torch.Tensor, ref64: torch.Tensor) -> float:
    d = (got.double() - ref64).abs().max().item()
    return d / (ref64.abs().max().item() + 1e-12)


def ulp_err(got: torch.Tensor, ref64: torch.Tensor, dtype, mode: str = "rms") -> float:
    """Max error in units of last place for ``dtype``; 1.0 is exact for it.

    ``rms`` scales by the reference's RMS (GEMMs and reductions, where a
    near-zero output is cancellation, not a small true value); ``elementwise``
    scales by each element's own magnitude (pointwise kernels).
    """
    eps = torch.finfo(dtype).eps
    got = got.detach().double()
    ref = ref64.detach().double()
    if mode == "rms":
        scale = ref.pow(2).mean().sqrt().clamp_min(torch.finfo(dtype).tiny)
    else:
        scale = ref.abs().clamp_min(torch.finfo(dtype).tiny)
    return ((got - ref).abs() / (scale * eps)).max().item()


def clear_grads(*tensors) -> None:
    """Clear leaf grads INSIDE the timed closure: .backward() accumulates, and
    the read-modify-write tax is proportional to the gradient bytes an arm
    owns, so it does not cancel in a ratio between arms."""
    for t in tensors:
        if t is not None:
            t.grad = None


def assert_backward_ran(tensors) -> None:
    for t in tensors:
        if t.grad is None or not bool(t.grad.abs().sum() > 0):
            raise RuntimeError("fwd+bwd arm: a gradient is missing or zero")


def save_rows(path: str, meta: dict, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=1)
    print(f"wrote {len(rows)} rows -> {p}")


def load_rows(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["meta"], data["rows"]


def device_meta() -> dict:
    """Machine facts plus the measured ceilings, so a figure drawn later from
    the JSON alone still says what it was scored against."""
    p = torch.cuda.get_device_properties(0)
    dev = resolve_device()
    return {
        "device": p.name,
        "sms": p.multi_processor_count,
        "torch": torch.__version__,
        "dram_bw": dev.dram_bw,
        "mamf": dev.mma_peak,
        "l2_mib": dev.l2_bytes / 2**20,
        "bw_pattern": "best of copy/read/triad, measured this run",
    }
