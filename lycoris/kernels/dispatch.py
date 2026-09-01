"""Backend resolution: triton, tilelang, torch.compile, or the torch reference.

Preference is triton > tilelang > compile > torch, resolved once per process;
``LYCORIS_KERNEL_BACKEND`` overrides (``auto|triton|tilelang|compile|torch``).
A backend that fails to import is absent — never an error at import time, only
at explicit request. ``demote`` drops one that imports but cannot run, so a
first failure costs a fallback rather than the process.
"""

import importlib.util
import os

_PROBE = {
    "triton": "triton",
    "tilelang": "tilelang",
}
ORDER = ("triton", "tilelang", "compile", "torch")
# The tiers that carry a ``kernels.<name>.ops`` module; compile and torch run
# the reference body instead, so they have no op set to import.
FUSED = ("triton", "tilelang")
_available: tuple[str, ...] | None = None


def _compile_usable() -> bool:
    """torch.compile exists and is not disabled by the environment."""
    if os.environ.get("TORCHDYNAMO_DISABLE", "0") not in ("0", "", "false"):
        return False
    spec = importlib.util.find_spec("torch")
    return spec is not None


def available_backends() -> tuple[str, ...]:
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


def fused_backends() -> tuple[str, ...]:
    """Available backends that have a fused op set, in preference order."""
    return tuple(b for b in available_backends() if b in FUSED)


def demote(name: str) -> None:
    """Drop a backend that imported but could not run."""
    global _available
    _available = tuple(b for b in available_backends() if b != name)


def resolve_backend(requested: str | None = None) -> str:
    req = (requested or os.environ.get("LYCORIS_KERNEL_BACKEND", "auto")).lower()
    avail = available_backends()
    if req == "auto":
        for pick in ORDER:
            if pick in avail:
                return pick
    if req in avail:
        return req
    raise RuntimeError(f"kernel backend {req!r} not available; found {avail}")
