"""Backend resolution: triton, tilelang, torch.compile, or the torch reference.

The preference order depends on the device: Triton issues tensor-core dots
only from sm_80, so on sm_70/sm_75 TileLang (which emits the Volta/Turing MMA
instructions) leads and Triton is used only when asked for by name; from
sm_80 Triton leads. ``LYCORIS_KERNEL_BACKEND`` overrides
(``auto|triton|tilelang|compile|torch``). A backend that fails to import is
absent — never an error at import time, only at explicit request. ``demote``
drops one that imports but cannot run on one device, so a first failure costs
a fallback rather than the process, and never touches another device.
"""

import importlib.util
import os

import torch

_PROBE = {
    "triton": "triton",
    "tilelang": "tilelang",
}
ORDER = ("triton", "tilelang", "compile", "torch")
ORDER_PRE_AMPERE = ("tilelang", "compile", "torch", "triton")
AUTO_PRE_AMPERE = ("tilelang", "compile", "torch")
# The backends that carry a ``kernels.<name>.ops`` module; compile and torch run
# the reference body instead, so they have no op set to import.
FUSED = ("triton", "tilelang")
_available: tuple[str, ...] | None = None
_demoted: dict[object, set[str]] = {}


def _compile_usable() -> bool:
    """torch.compile exists and is not disabled by the environment."""
    if os.environ.get("TORCHDYNAMO_DISABLE", "0") not in ("0", "", "false"):
        return False
    spec = importlib.util.find_spec("torch")
    return spec is not None


def _installed() -> tuple[str, ...]:
    global _available
    if _available is None:
        found = []
        for name, module in _PROBE.items():
            if importlib.util.find_spec(module) is not None:
                found.append(name)
        if _compile_usable():
            found.append("compile")
        found.append("torch")
        _available = tuple(sorted(found, key=ORDER.index))
    return _available


def device_key(device: torch.device | None):
    """Demotion / ordering key of a device: (type, index) or None for 'any'."""
    if device is None:
        return None
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return ("cuda", torch.cuda.current_device())
    return (device.type, device.index)


def pre_ampere(device: torch.device | None) -> bool:
    """A CUDA device below sm_80 (no Triton tensor-core dots, no bf16 MMA)."""
    if device is None or not torch.cuda.is_available():
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index)[0] < 8


def order_for(device: torch.device | None) -> tuple[str, ...]:
    """Full preference order for ``device`` (every backend, for explicit requests)."""
    return ORDER_PRE_AMPERE if pre_ampere(device) else ORDER


def available_backends(device: torch.device | None = None) -> tuple[str, ...]:
    """Installed backends usable on ``device``, in that device's preference order."""
    gone = _demoted.get(device_key(device), set()) | _demoted.get(None, set())
    return tuple(b for b in order_for(device) if b in _installed() and b not in gone)


def fused_backends(device: torch.device | None = None) -> tuple[str, ...]:
    """Available backends that have a fused op set, in preference order."""
    return tuple(b for b in available_backends(device) if b in FUSED)


def demote(name: str, device: torch.device | None = None) -> None:
    """Drop a backend that imported but could not run (on one device, or all)."""
    _demoted.setdefault(device_key(device), set()).add(name)


def resolve_backend(
    requested: str | None = None, device: torch.device | None = None
) -> str:
    """The backend to use on ``device``: the requested one, or the device's first."""
    req = (requested or os.environ.get("LYCORIS_KERNEL_BACKEND", "auto")).lower()
    avail = available_backends(device)
    if req == "auto":
        auto = AUTO_PRE_AMPERE if pre_ampere(device) else ORDER
        for pick in auto:
            if pick in avail:
                return pick
    if req in avail:
        return req
    raise RuntimeError(f"kernel backend {req!r} not available; found {avail}")
