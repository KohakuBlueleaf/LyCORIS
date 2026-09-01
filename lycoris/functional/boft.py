import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .general import power2factorization, FUNC_LIST
from .diag_oft import get_r
from ..kernels.autograd.boft import boft_bypass_diff, boft_diff_weight
from ..kernels.select import FUSED, call_compiled, choose


def weight_gen(org_weight, max_block_size, boft_m=-1, rescale=False):
    """### boft_weight_gen

    Args:
        org_weight (torch.Tensor): the weight tensor
        max_block_size (int): max block size
        rescale (bool, optional): whether to rescale the weight. Defaults to False.

    Returns:
        torch.Tensor: oft_blocks[, rescale_weight]
    """
    out_dim, *rest = org_weight.shape
    block_size, block_num = power2factorization(out_dim, max_block_size)
    max_boft_m = sum(int(i) for i in f"{block_num-1:b}") + 1
    if boft_m == -1:
        boft_m = max_boft_m
    boft_m = min(boft_m, max_boft_m)
    oft_blocks = torch.zeros(boft_m, block_num, block_size, block_size)
    if rescale is not None:
        return oft_blocks, torch.ones(out_dim, *[1] * len(rest))
    else:
        return oft_blocks, None


def _diff_weight(org_weight, oft_blocks, rescale, constraint, scale):
    """ΔW = butterfly(W)·rescale − W: m stages, stage i mixing within 2^i·b."""
    m, num, b, _ = oft_blocks.shape
    r_b = b // 2
    I = torch.eye(b, device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)
    inp = org = org_weight.to(dtype=r.dtype)

    for i in range(m):
        bi = r[i]  # b_num, b_size, b_size
        g = 2
        k = 2**i * r_b
        # Multiplier interpolates each stage toward identity, not the result.
        if scale != 1:
            bi = bi * scale + (1 - scale) * I
        inp = (
            inp.unflatten(0, (-1, g, k))
            .transpose(1, 2)
            .flatten(0, 2)
            .unflatten(0, (-1, b))
        )
        inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
        inp = inp.flatten(0, 1).unflatten(0, (-1, k, g)).transpose(1, 2).flatten(0, 2)

    if rescale is not None:
        inp = inp * rescale

    return inp - org


def diff_weight(org_weight, *weights, constraint=None, scale=1, backend=None):
    """### boft_diff_weight

    Args:
        org_weight (torch.Tensor): the weight tensor of original model
        weights (tuple[torch.Tensor]): (oft_blocks[, rescale_weight])
        constraint (float, optional): constraint for oft
        scale (float, optional): multiplier, folded into every stage
        backend (str, optional): pin one of triton/tilelang/compile/torch

    Returns:
        torch.Tensor: ΔW
    """
    oft_blocks, rescale = weights
    pick = choose((org_weight, oft_blocks, rescale), backend=backend)
    if pick in FUSED:
        return boft_diff_weight(
            org_weight, oft_blocks, rescale, constraint, scale, backend=pick
        )
    if pick == "compile":
        return call_compiled(
            _diff_weight, org_weight, oft_blocks, rescale, constraint, scale
        )
    return _diff_weight(org_weight, oft_blocks, rescale, constraint, scale)


def _bypass_diff(org_out, oft_blocks, rescale, constraint, need_transpose, scale):
    """Δy = butterfly(y)·rescale − y on the channel axis."""
    m, num, b, _ = oft_blocks.shape
    r_b = b // 2
    I = torch.eye(b, device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)
    inp = org = org_out.to(dtype=r.dtype)
    if need_transpose:
        inp = org = inp.transpose(1, -1)

    for i in range(m):
        bi = r[i]  # b_num, b_size, b_size
        g = 2
        k = 2**i * r_b
        if scale != 1:
            bi = bi * scale + (1 - scale) * I
        # ... (c g k) ->... (c k g)
        # ... (d b) -> ... d b
        inp = (
            inp.unflatten(-1, (-1, g, k))
            .transpose(-2, -1)
            .flatten(-3)
            .unflatten(-1, (-1, b))
        )
        inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
        # ... d b -> ... (d b)
        # ... (c k g) -> ... (c g k)
        inp = inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)

    if rescale is not None:
        inp = inp * rescale.transpose(0, -1)

    inp = inp - org
    if need_transpose:
        inp = inp.transpose(1, -1)
    return inp


def bypass_forward_diff(
    org_out, *weights, constraint=None, need_transpose=False, scale=1, backend=None
):
    """### boft_bypass_forward_diff

    Args:
        org_out (torch.Tensor): the output tensor from original model
        weights (tuple[torch.Tensor]): (oft_blocks[, rescale_weight])
        constraint (float, optional): constraint for oft
        need_transpose (bool, optional): `True` when "dim" is not the last
            axis, as in convolution layers
        scale (float, optional): multiplier, folded into every stage
        backend (str, optional): pin one of triton/tilelang/compile/torch

    Returns:
        torch.Tensor: output tensor
    """
    oft_blocks, rescale = weights
    pick = choose((org_out, oft_blocks, rescale), backend=backend)
    if pick in FUSED:
        return boft_bypass_diff(
            org_out,
            oft_blocks,
            rescale,
            constraint,
            scale,
            need_transpose=need_transpose,
            backend=pick,
        )
    if pick == "compile":
        return call_compiled(
            _bypass_diff,
            org_out,
            oft_blocks,
            rescale,
            constraint,
            need_transpose,
            scale,
        )
    return _bypass_diff(org_out, oft_blocks, rescale, constraint, need_transpose, scale)
