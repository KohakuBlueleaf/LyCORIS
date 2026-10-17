"""Planner and measuring helpers shared by the fused kernels: config grids, keys,
the measurer the tuner times candidates with, and the build-failure guard."""

import contextlib
import itertools
import math
from collections.abc import Callable

import torch

from . import TORCH, DeviceInfo, tuner
from .timing import buffer_sets, time_candidates

LAUNCH_ARGS = ("num_warps", "num_stages")
# Relative Frobenius distance a candidate may sit from the torch composition, per dtype.
_AGREE_TOL = {torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 4e-2}


def pad16(n: int) -> int:
    """n rounded up to a power of two, at least 16."""
    return max(16, 1 << (max(1, n) - 1).bit_length())


def pow2(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


def token_bucket(t: int) -> int:
    """Token count bucket: power of two up to 4096, then multiples of 4096."""
    if t <= 4096:
        return pow2(t)
    return -(-t // 4096) * 4096


def dtype_key(dtype: torch.dtype) -> str:
    """Key of a dtype: its name (fp16 and bf16 differ in MMA support)."""
    return str(dtype).removeprefix("torch.")


def launch(cfg: dict) -> dict:
    """The launch options of a config."""
    return {k: cfg[k] for k in LAUNCH_ARGS if k in cfg}


def stage_axis(dev: DeviceInfo) -> tuple[int, ...]:
    """Pipeline depths a config may use on this device."""
    return (2, 3, 4) if dev.async_copy else (1, 2)


def grid_configs(dev: DeviceInfo, **axes) -> list[dict]:
    """Every combination of the named axes, plus the device's stage axis."""
    if "num_stages" not in axes:
        axes["num_stages"] = stage_axis(dev)
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*axes.values())]


def copy_of(t: torch.Tensor | None) -> torch.Tensor | None:
    """A same-layout copy of t (never fresh random data: tuning leaves the RNG alone)."""
    return None if t is None else torch.empty_like(t).copy_(t)


def guarded(dev: DeviceInfo, key, cfg: dict, run: Callable[[dict], object], too_big):
    """run(cfg), or run(TORCH) from now on for key when cfg raises ``too_big``
    (the backend's out-of-resources error)."""
    try:
        return run(cfg)
    except too_big:
        tuner.demote(dev, key)
        return run(TORCH)


def _tensors(result) -> list[torch.Tensor]:
    if isinstance(result, torch.Tensor):
        return [result]
    return [t for t in result if isinstance(t, torch.Tensor) and t.numel() > 0]


def disagreement(got, ref) -> str | None:
    """None when got matches the torch composition's ref, else the mismatch: a
    guard against a miscompiled config, not a precision ranking."""
    for n, (g, r) in enumerate(zip(_tensors(got), _tensors(ref))):
        tol = _AGREE_TOL.get(r.dtype, _AGREE_TOL[torch.bfloat16])
        g, r = g.float(), r.float()
        if not torch.isfinite(g).all():
            return f"output {n} not finite"
        norm = r.norm()
        if norm > 0 and (gap := ((g - r).norm() / norm).item()) > tol:
            return f"output {n} off torch by {gap:.2e} (tolerance {tol:.0e})"
    return None


def _warm(run, cfg: dict, inputs: tuple) -> None:
    """Runs cfg once so its kernels are compiled; a failure surfaces again when validated."""
    with contextlib.suppress(Exception):
        run(cfg, inputs)()


def measurer(
    dev: DeviceInfo,
    bytes_per_call: int,
    make_inputs: Callable[[], tuple],
    run: Callable[[dict, tuple], Callable[[], object]],
) -> Callable[[list[dict]], list[float]]:
    """The tuner's measure(): per-config ms, inf for a config that disagrees with torch."""
    state = {}

    def measure(configs: list[dict]) -> list[float]:
        if not state:
            state["sets"] = [
                make_inputs() for _ in range(buffer_sets(bytes_per_call, dev))
            ]
            state["ref"] = run(TORCH, state["sets"][0])()
        sets = state["sets"]
        # Compile every kernel config of the shortlist, one at a time.
        for c in configs:
            if "impl" not in c:
                _warm(run, c, sets[0])

        valid = []
        for c in configs:
            if c == TORCH:
                valid.append(True)
                continue
            try:
                gap = disagreement(run(c, sets[0])(), state["ref"])
            except Exception as e:  # noqa: BLE001 - a failing config times as inf
                gap = f"{type(e).__name__}: {str(e)[:300]}"
            valid.append(gap is None)
            if gap is not None:
                tuner.note_failure(c, gap)
        makers = [lambda j, c=c: run(c, sets[j]) for c in configs]
        errors: list[str | None] = []
        times = time_candidates(makers, len(sets), dev, errors=errors)
        for c, err in zip(configs, errors):
            if err is not None:
                tuner.note_failure(c, err)
        return [t if ok else math.inf for t, ok in zip(times, valid)]

    return measure
