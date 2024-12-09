import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .general import power2factorization, FUNC_LIST
from .diag_oft import get_r


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


def diff_weight(org_weight, *weights, constraint=None):
    """### boft_diff_weight

    Args:
        org_weight (torch.Tensor): the weight tensor of original model
        weights (tuple[torch.Tensor]): (oft_blocks[, rescale_weight])
        constraint (float, optional): constraint for oft

    Returns:
        torch.Tensor: ΔW
    """
    oft_blocks, rescale = weights
    m, num, b, _ = oft_blocks.shape
    r_b = b // 2
    I = torch.eye(b, device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)
    inp = org = org_weight.to(dtype=r.dtype)

    for i in range(m):
        bi = r[i]  # b_num, b_size, b_size
        g = 2
        k = 2**i * r_b
        inp = (
            inp.unflatten(-1, (-1, g, k))
            .transpose(-2, -1)
            .flatten(-3)
            .unflatten(-1, (-1, b))
        )
        inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
        inp = inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)

    if rescale is not None:
        inp = inp * rescale

    return inp - org


def bypass_forward_diff(org_out, *weights, constraint=None, need_transpose=False):
    """### boft_bypass_forward_diff

    Args:
        x (torch.Tensor): the input tensor for original model
        org_out (torch.Tensor): the output tensor from original model
        weights (tuple[torch.Tensor]): (oft_blocks[, rescale_weight])
        constraint (float, optional): constraint for oft
        need_transpose (bool, optional):
            whether to transpose the input and output,
            set to `True` if the original model have "dim" not in the last axis.
            For example: Convolution layers

    Returns:
        torch.Tensor: output tensor
    """
    oft_blocks, rescale = weights
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
