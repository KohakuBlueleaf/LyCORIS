"""Hardware description the family planners score against.

Fields split into three groups: queried from the driver, measured on the card
(mma_peak TFLOP/s for the fp32-accumulate rate, dram_bw and l2_bw GB/s), and
calibrated cost-model constants. Measured fields default to NaN; a device
without them cannot be ranked and the planners fall back to their SAFE
candidate sets. The denominator bench measures them and rewrites the JSON
this module loads (``LYCORIS_DEVICE_JSON`` or the cache dir).
"""

import dataclasses
import json
import math
import os
from pathlib import Path

import torch


@dataclasses.dataclass(frozen=True)
class Device:
    name: str
    sms: int
    regs_per_sm: int
    smem_per_cta: int
    smem_per_sm: int
    max_threads_per_sm: int
    l2_bytes: int
    cc_major: int = 0
    cc_minor: int = 0

    mma_peak: float = float("nan")
    dram_bw: float = float("nan")
    l2_bw: float = float("nan")

    bar_tax: tuple[float, ...] = (0.94, 0.96)
    reg_overhead: int = 64

    @property
    def cp_async(self) -> bool:
        """sm_80+ can stage global->shared asynchronously; sm_70/75 cannot."""
        return self.cc_major >= 8

    def pipeline_depths(self) -> tuple[int, ...]:
        """Loop pipeline depths worth trying on this card.

        Without cp.async a deeper pipeline still double-buffers, but it pays
        shared memory for a synchronous copy, so the space stops at 2.
        """
        return (1, 2, 3, 4) if self.cp_async else (1, 2)

    @classmethod
    def query(cls, index: int = 0, **overrides) -> "Device":
        p = torch.cuda.get_device_properties(index)
        base = {
            "name": p.name,
            "sms": p.multi_processor_count,
            "regs_per_sm": getattr(p, "regs_per_multiprocessor", 65536),
            "smem_per_cta": getattr(p, "shared_memory_per_block_optin", 48 * 1024),
            "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", 64 * 1024),
            "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", 1536),
            "l2_bytes": getattr(p, "l2_cache_size", None)
            or getattr(p, "L2_cache_size", 4 * 2**20),
            "cc_major": p.major,
            "cc_minor": p.minor,
        }
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_json(cls, path: str) -> "Device":
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["bar_tax"] = tuple(raw["bar_tax"])
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    @property
    def measured(self) -> bool:
        return not (math.isnan(self.mma_peak) or math.isnan(self.dram_bw))


# Provisional stated values (mma_peak acc-dtype audit pending); the
# denominator bench measures and rewrites them via to_json.
KNOWN = {
    "NVIDIA GeForce RTX 4090": {"mma_peak": 250.0, "dram_bw": 1000.0},
}

_RESOLVED: dict[int, Device] = {}


def _json_path() -> Path | None:
    explicit = os.environ.get("LYCORIS_DEVICE_JSON")
    if explicit:
        return Path(explicit)
    root = os.environ.get("LYCORIS_KERNEL_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "lycoris_kernels"
    return base / "device.json"


def resolve_device(index: int = 0) -> Device:
    """Queried device + measured numbers from disk or the KNOWN table."""
    if index in _RESOLVED:
        return _RESOLVED[index]
    if not torch.cuda.is_available():
        dev = Device(
            name="cpu",
            sms=1,
            regs_per_sm=65536,
            smem_per_cta=48 * 1024,
            smem_per_sm=64 * 1024,
            max_threads_per_sm=1024,
            l2_bytes=4 * 2**20,
        )
        _RESOLVED[index] = dev
        return dev
    dev = Device.query(index)
    path = _json_path()
    if path is not None and path.exists():
        try:
            stored = Device.from_json(str(path))
            if stored.name == dev.name:
                dev = stored
        except (OSError, ValueError, TypeError, KeyError):
            pass
    if not dev.measured and dev.name in KNOWN:
        dev = dataclasses.replace(dev, **KNOWN[dev.name])
    _RESOLVED[index] = dev
    return dev
