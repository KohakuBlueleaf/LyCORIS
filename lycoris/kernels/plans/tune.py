"""Time the planner's shortlist once per key and keep the winner.

The model ranks; only its top few are ever run. Results live in an in-proc
cache plus a JSON offline table per device, so a later process does not
repeat the timing. Keys must bucket every varlen axis (``bucket_tokens``)
and include the backend name. Runner factories must be side-effect safe:
tuning writes only to scratch the factory allocated.
"""

import json
import os
from pathlib import Path

import torch

from .cost import TilePlan
from .device import resolve_device

SHORTLIST = 6
ROW_BUCKET = 4096

_CACHE: dict[tuple, TilePlan] = {}
_LOADED = False


def bucket_tokens(t: int) -> int:
    """Pow2 below ROW_BUCKET, ROW_BUCKET-granular ceil above."""
    if t <= 0:
        return 1
    if t < ROW_BUCKET:
        return 1 << (t - 1).bit_length()
    return -(-t // ROW_BUCKET) * ROW_BUCKET


def _table_path() -> Path:
    root = os.environ.get("LYCORIS_KERNEL_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "lycoris_kernels"
    return base / "tuning.json"


def _load_table() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = _table_path()
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for row in json.load(fh):
                _CACHE[tuple(row["key"])] = TilePlan(**row["plan"])
    except (OSError, ValueError, TypeError, KeyError):
        pass


def _save_table() -> None:
    path = _table_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"key": list(k), "plan": v.as_dict()} for k, v in _CACHE.items()]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    except OSError:
        pass


def _time(fn, warmup: int = 10, iters: int = 30, rounds: int = 3) -> float:
    """Best of ``rounds`` passes, each the min of ``iters`` events.

    The pick is made once and then cached for the life of the table, so a
    single noisy pass would freeze a plan that is measurably worse: one pass
    was observed selecting a plan 2x off the shortlist's best.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        beg = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            beg[i].record()
            fn()
            end[i].record()
        torch.cuda.synchronize()
        best = min(best, min(s.elapsed_time(e) for s, e in zip(beg, end)))
    return best


def tuning_enabled() -> bool:
    return os.environ.get("LYCORIS_KERNEL_TUNE", "on").lower() not in ("off", "0")


def tuned(kernel: str, key: tuple, shortlist_fn, runner_factory) -> TilePlan:
    """Best of the shortlist, timed once and cached.

    Cache-first: ``shortlist_fn`` (a thunk, or a plain list) is evaluated only
    on a miss, so steady-state calls never pay the python scoring loop. First
    pick when timing is disabled, fails everywhere, or the list is singular.
    """
    full_key = (resolve_device().name, kernel, *key)
    _load_table()
    if full_key in _CACHE:
        return _CACHE[full_key]
    shortlist = shortlist_fn() if callable(shortlist_fn) else shortlist_fn
    if not shortlist:
        raise ValueError(f"{kernel}: empty shortlist")
    if len(shortlist) == 1 or not tuning_enabled():
        best = shortlist[0]
        _CACHE[full_key] = best
        return best
    best, best_ms = None, float("inf")
    for cand in shortlist:
        try:
            ms = _time(runner_factory(cand))
        except Exception:  # noqa: BLE001, S112 - a failing candidate is skipped
            continue
        if ms < best_ms:
            best, best_ms = cand, ms
    best = best or shortlist[0]
    _CACHE[full_key] = best
    _save_table()
    return best


def clear() -> None:
    _CACHE.clear()
