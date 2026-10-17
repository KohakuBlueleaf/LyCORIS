"""Shared helpers for the Triton backend.

Tile/warp selection lives in ``..plans`` (planner shortlist + measured tuner);
kernels here receive plans as explicit launch parameters.
"""

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

__all__ = ["OutOfResources", "dot_prec", "load2d", "rank_block"]


def dot_prec(dtype: torch.dtype) -> str:
    """tl.dot input precision: IEEE for fp32 operands, the operands' own for
    16-bit ones."""
    return "ieee" if dtype == torch.float32 else "tf32"


@triton.jit
def load2d(ptr, rows, cols, s_row, s_col, m_row, m_col):
    """The (rows, cols) tile at ptr, zero where either mask is off."""
    return tl.load(
        ptr + rows[:, None] * s_row + cols[None, :] * s_col,
        mask=m_row[:, None] & m_col[None, :],
        other=0.0,
    )


def rank_block(r: int) -> int:
    """tl.dot needs >=16 on every dim; pad the rank axis accordingly."""
    return max(16, 1 << (max(1, r) - 1).bit_length())
