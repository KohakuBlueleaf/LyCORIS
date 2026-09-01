"""Shared helpers for the Triton backend.

Tile/warp selection lives in ``..plans`` (planner shortlist + measured tuner);
kernels here receive plans as explicit launch parameters.
"""


def rank_block(r: int) -> int:
    """tl.dot needs >=16 on every dim; pad the rank axis accordingly."""
    return max(16, 1 << (max(1, r) - 1).bit_length())
