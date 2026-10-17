"""Shared helpers for the TileLang backend: dtype names, the per-arch compile
cache, padding to the MMA extent, and the shared-memory and warp-partition
legality checks every model-ranked config passes before it is compiled.

TileLang self-discovers its Windows toolchain; if an incompatible clang-cl
shadows PATH (nvcc: "Host compiler targets unsupported OS"), set
``TILELANG_DISABLE_CLANG_CL=1``. See docs/kernels/designs.md.
"""

import threading
from collections.abc import Callable

import tilelang
import torch

DTYPE_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
# T.gemm's smallest legal extent on every target: ranks and small factors pad to it.
MIN_DIM = 16

_CACHE: dict[tuple, object] = {}
_KEY_LOCKS: dict[tuple, threading.Lock] = {}
_LOCK = threading.Lock()


def dstr(t: torch.Tensor) -> str:
    return DTYPE_STR[t.dtype]


def tl_dtype(dtype: torch.dtype | None) -> str | None:
    """TileLang's name of a torch dtype (None passes through)."""
    return None if dtype is None else DTYPE_STR[dtype]


def pad_to(n: int, m: int = MIN_DIM) -> int:
    return -(-n // m) * m


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def partitions(dev, threads: int, *gemms: tuple[int, int]) -> bool:
    """Whether every (M, N) T.gemm tile splits evenly over the program's warps,
    each owning a multiple of 16 rows and of the arch's column unit (16 on
    sm_70, 8 from sm_75); a config that fails this does not compile."""
    warps = threads // 32
    unit = 16 if dev.capability == (7, 0) else 8
    for m, n in gemms:
        if not any(
            warps % wm == 0 and m % (16 * wm) == 0 and n % (unit * (warps // wm)) == 0
            for wm in range(1, warps + 1)
        ):
            return False
    return True


def fp32_gemm_exact(dev) -> bool:
    """Whether a T.gemm on fp32 operands runs IEEE fp32 (FMAs, sm_70 / sm_75);
    from sm_80 it lowers to TF32 MMA."""
    return dev.capability is not None and dev.capability < (8, 0)


def fits(dev, half_elements: int) -> bool:
    """Whether shared tiles of this many 16-bit elements fit one program (the
    device's opt-in limit, less a 1 KiB reserve)."""
    return 2 * half_elements + 1024 <= dev.smem


class CompileError(RuntimeError):
    """A config TileLang could not lower for this device."""


def _pass_configs(arch: str) -> dict:
    """Warp specialization and TMA lowering only on sm_90, the arch with setmaxnreg and TMA."""
    if arch == "sm_90":
        return {}
    return {"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}


def compiled(builder: Callable, device: torch.device, *args):
    """builder(*args) compiled for this device's arch, once per process; a
    failed build is remembered and raised as CompileError."""
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    key = (builder.__module__, builder.__qualname__, arch, *args)
    kernel = _CACHE.get(key)
    if kernel is None:
        with _LOCK:
            lock = _KEY_LOCKS.setdefault(key, threading.Lock())
        with lock:
            kernel = _CACHE.get(key)
            if kernel is None:
                try:
                    kernel = tilelang.compile(
                        builder(*args),
                        target={"kind": "cuda", "arch": arch},
                        pass_configs=_pass_configs(arch),
                    )
                except Exception as e:  # noqa: BLE001 - any lowering failure
                    kernel = CompileError(
                        f"{builder.__qualname__}{args}: {type(e).__name__}: {e}"
                    )
                _CACHE[key] = kernel
    if isinstance(kernel, CompileError):
        raise kernel
    return kernel


def padded_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    """t (rows, c) zero-padded to (rows, cols) contiguous; t itself when it fits."""
    if t.shape[1] == cols and t.is_contiguous():
        return t
    out = torch.zeros(t.shape[0], cols, device=t.device, dtype=t.dtype)
    out[:, : t.shape[1]] = t
    return out


def padded_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    """t (r, cols) zero-padded to (rows, cols) contiguous; t itself when it fits."""
    if t.shape[0] == rows and t.is_contiguous():
        return t
    out = torch.zeros(rows, t.shape[1], device=t.device, dtype=t.dtype)
    out[: t.shape[0]] = t
    return out
