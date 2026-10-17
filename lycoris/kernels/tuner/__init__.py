"""Config selection for the fused kernels: an analytical model ranks, a short
shortlist is timed once per key, the winner is stored per card.
See docs/kernels/tuning.md.
"""

import contextlib
import importlib.metadata
import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch

from ..plans import tune
from .device import DeviceInfo, device_info

__all__ = [
    "LAUNCH_SHORTLIST",
    "TILE_SHORTLIST",
    "TORCH",
    "DeviceInfo",
    "device_info",
    "select",
    "tuner",
]

TORCH = {"impl": "torch"}
TILE_SHORTLIST = 4
LAUNCH_SHORTLIST = 3
# Config keys that set how a tile is launched, not its shape.
LAUNCH_KEYS = frozenset({"num_warps", "num_stages", "threads"})
TABLE_VERSION = 1
_REPLACE_TRIES = 20

Key = tuple[str, tuple, tuple]


def _freeze(config: dict) -> tuple:
    return tuple(sorted(config.items()))


def _capturing(dev: DeviceInfo) -> bool:
    if dev.kind == "cuda" and torch.cuda.is_current_stream_capturing():
        return True
    return torch.compiler.is_compiling()


def _stamp() -> dict:
    out = {"torch": torch.__version__}
    for dist in ("lycoris-lora", "triton", "triton-windows", "tilelang"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            out[dist] = importlib.metadata.version(dist)
    return out


def _table_path(dev: DeviceInfo) -> Path:
    root = os.environ.get("LYCORIS_KERNEL_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "lycoris_kernels"
    card = re.sub(r"[^a-z0-9]+", "-", dev.name.lower()).strip("-")
    return base / "kernels" / f"{dev.arch}-{card}.json"


def _read(path: Path, stamp: dict) -> dict[Key, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if data.get("version") != TABLE_VERSION or data.get("stamp") != stamp:
        return {}
    return {
        (r["kernel"], tuple(r["cat"]), tuple(r["num"])): r["config"]
        for r in data.get("entries", [])
    }


def _write(path: Path, rows: dict[Key, dict], stamp: dict) -> None:
    """Whole-file write through a temp file and an atomic replace, retried while
    another handle holds the file (Windows); the temp file never outlives it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    entries = [
        {"kernel": k, "cat": list(c), "num": list(n), "config": cfg}
        for (k, c, n), cfg in rows.items()
    ]
    try:
        tmp.write_text(
            json.dumps(
                {"version": TABLE_VERSION, "stamp": stamp, "entries": entries},
                indent=1,
            ),
            encoding="utf-8",
        )
        for attempt in range(_REPLACE_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_TRIES - 1:
                    return
                time.sleep(0.05)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


class Tuner:
    def __init__(self):
        self._chosen: dict[tuple[str, Key], dict] = {}
        self._disk: dict[str, dict[Key, dict]] = {}
        self._stamp: dict | None = None
        # When a list, every tuned key's timings are appended (benches).
        self.record: list[dict] | None = None
        self._failures: dict[tuple, str] = {}
        # Reentrant: a candidate being timed may itself select.
        self._lock = threading.RLock()

    def _table(self, dev: DeviceInfo) -> dict[Key, dict]:
        rows = self._disk.get(dev.name)
        if rows is None:
            if self._stamp is None:
                self._stamp = _stamp()
            rows = self._disk[dev.name] = _read(_table_path(dev), self._stamp)
        return rows

    def select(
        self,
        dev: DeviceInfo,
        key: Key,
        plan: Callable[[], list[dict]],
        measure: Callable[[list[dict]], list[float]],
    ) -> dict:
        """The config for key. plan() returns the legal configs, model-ranked,
        with whole-op alternatives (an "impl" key) anywhere in the list."""
        memo = (dev.name, key)
        config = self._chosen.get(memo)
        if config is not None:
            return config
        with self._lock:
            config = self._chosen.get(memo)
            if config is not None:
                return config
            ranked = plan()
            config = self._table(dev).get(key)
            if config is not None and _freeze(config) not in {
                _freeze(c) for c in ranked
            }:
                config = None
            if config is None:
                if _capturing(dev):
                    # Not remembered: the key is timed once timing is possible.
                    return _model_pick(ranked)
                if tune.tuning_enabled():
                    config = self._tune(dev, key, ranked, measure)
                else:
                    # Remembered for the process, never stored, as in plans.tune.
                    config = _model_pick(ranked)
            self._chosen[memo] = config
            return config

    def _tune(self, dev: DeviceInfo, key: Key, ranked: list[dict], measure) -> dict:
        """Time the tile shortlist and the whole-op alternatives, then launch
        settings of the fastest tile; store the winner."""
        self._failures = {}
        ordered = [c for c in ranked if "impl" not in c]
        tile_axes = {k for c in ordered for k in c} - LAUNCH_KEYS
        best_per_tile, seen = [], set()
        for c in ordered:
            tile = _freeze({a: c.get(a) for a in tile_axes})
            if tile not in seen:
                seen.add(tile)
                best_per_tile.append(c)
        tiles = shortlist(best_per_tile, sorted(tile_axes), TILE_SHORTLIST)
        candidates = tiles + [c for c in ranked if "impl" in c]
        times = measure(candidates)

        # Launch settings of the fastest tile, among the configs sharing its shape.
        timed_tiles = [i for i, c in enumerate(candidates) if "impl" not in c]
        if timed_tiles:
            lead = candidates[min(timed_tiles, key=times.__getitem__)]
            same = [
                c
                for c in ordered
                if c not in candidates
                and all(c.get(a) == lead.get(a) for a in tile_axes)
            ]
            launch_axes = sorted({k for c in same for k in c} & LAUNCH_KEYS)
            variants = shortlist([lead, *same], launch_axes, LAUNCH_SHORTLIST + 1)[1:]
            if variants:
                times += measure(variants)
                candidates += variants

        winner = candidates[min(range(len(candidates)), key=times.__getitem__)]
        if self.record is not None:
            self.record.append(
                {
                    "device": dev.name,
                    "key": key,
                    "timed": [[c, t] for c, t in zip(candidates, times)],
                    "failures": [[dict(f), why] for f, why in self._failures.items()],
                }
            )
        rows = self._table(dev)
        rows[key] = winner
        path = _table_path(dev)
        # Merge with rows another process stored since this one loaded the table.
        merged = {**_read(path, self._stamp), **rows}
        with contextlib.suppress(OSError):
            _write(path, merged, self._stamp)
        return winner

    def note_failure(self, config: dict, reason: str) -> None:
        """A candidate of the key being tuned failed; kept for the record."""
        self._failures[_freeze(config)] = reason

    def demote(self, dev: DeviceInfo, key: Key) -> None:
        """A config this device could not build: the key runs torch from now on."""
        self._chosen[(dev.name, key)] = TORCH

    def reset(self) -> None:
        """Forget every in-process choice and loaded table (a test helper)."""
        with self._lock:
            self._chosen.clear()
            self._disk.clear()
            self._stamp = None


def shortlist(kernels: list[dict], axes: list[str], size: int) -> list[dict]:
    """size configs from the model's order: its best, then in turn per axis the
    best config with a value of that axis not yet picked; the model's order
    fills what is left."""
    if not kernels:
        return []
    picked = [kernels[0]]
    covered = {a: {kernels[0].get(a)} for a in axes}
    while len(picked) < size:
        added = False
        for axis in axes:
            pick = next(
                (
                    c
                    for c in kernels
                    if c not in picked and c.get(axis) not in covered[axis]
                ),
                None,
            )
            if pick is None:
                continue
            picked.append(pick)
            for a in axes:
                covered[a].add(pick.get(a))
            added = True
            if len(picked) == size:
                return picked
        if not added:
            break
    rest = [c for c in kernels if c not in picked]
    return picked + rest[: size - len(picked)]


def _model_pick(ranked: list[dict]) -> dict:
    for c in ranked:
        if "impl" not in c:
            return c
    return ranked[0] if ranked else TORCH


tuner = Tuner()


def select(dev, key, plan, measure) -> dict:
    return tuner.select(dev, key, plan, measure)
