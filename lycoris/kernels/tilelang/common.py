"""Shared helpers for the TileLang backend.

Tile/threads selection lives in ``..plans`` (planner shortlist + measured
tuner); builders receive (blocks, threads) as compile-time arguments.
TileLang self-discovers its Windows toolchain; if an incompatible clang-cl
shadows PATH (nvcc: "Host compiler targets unsupported OS"), set
``TILELANG_DISABLE_CLANG_CL=1``. The nvrtc execution backend removes the
host-toolchain step entirely when ``cuda-python`` is installed.
"""

import torch

DTYPE_STR = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def dstr(t: torch.Tensor) -> str:
    return DTYPE_STR[t.dtype]
