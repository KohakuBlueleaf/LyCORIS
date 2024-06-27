import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .general import power2factorization, FUNC_LIST
from .diag_oft import get_r


def boft_weight_gen(org_weight, max_block_size, boft_m=-1, rescale=False):
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
    max_boft_m = sum(int(i) for i in f"{block_num-1:b}")+1
    if boft_m == -1:
        boft_m = max_boft_m
    boft_m = min(boft_m, max_boft_m)
    oft_blocks = torch.zeros(boft_m, block_num, block_size, block_size)
    if rescale:
        return oft_blocks, torch.ones(out_dim, *[1] * len(rest))
    else:
        return oft_blocks, None


def boft_diff_weight(org_weight, oft_blocks, rescale=None, constraint=None):
    """### boft_diff_weight

    Args:
        TODO

    Returns:
        torch.Tensor: ΔW
    """
    m, num, b, _ = oft_blocks.shape
    r_b = b // 2
    I = torch.eye(b, device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)
    inp = org = org_weight.to(dtype=r.dtype)

    for i in range(m):
        bi = r[i]  # b_num, b_size, b_size
        inp = rearrange(inp, "(c g k) ... -> (c k g) ...", g=2, k=2**i * r_b)
        inp = rearrange(inp, "(d b) ... -> d b ...", b=b)
        inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
        inp = rearrange(inp, "d b ... -> (d b) ...")
        inp = rearrange(inp, "(c k g) ... -> (c g k) ...", g=2, k=2**i * r_b)

    return inp * rescale - org


def boft_bypass_forward_diff(
    org_out, need_transpose, oft_blocks, rescale=None, constraint=None
):
    """### boft_bypass_forward_diff

    Args:
        TODO

    Returns:
        torch.Tensor: output tensor
    """
    m, num, b, _ = oft_blocks.shape
    r_b = b // 2
    I = torch.eye(b, device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)
    inp = org_out.to(dtype=r.dtype)
    if need_transpose:
        inp = org = inp.transpose(1, -1)

    for i in range(m):
        bi = r[i]  # b_num, b_size, b_size
        inp = rearrange(inp, "... (c g k) ->... (c k g)", g=2, k=2**i * r_b)
        inp = rearrange(inp, "... (d b) -> ... d b", b=b)
        inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
        inp = rearrange(inp, "... d b -> ... (d b)")
        inp = rearrange(inp, "... (c k g) -> ... (c g k)", g=2, k=2**i * r_b)

    inp = inp * rescale.transpose(0, -1) - org
    if need_transpose:
        inp = inp.transpose(1, -1)
    return inp


if __name__ == "__main__":
    w = torch.randn(32, 32, 3, 3, 3)
    blocks, rescale = boft_weight_gen(w, 4, -1, True)
    blocks = blocks + torch.randn_like(blocks)*0.01
    extra_args = {"padding": 1}

    x = torch.randn(1, 32, 8, 8, 8)
    y = FUNC_LIST[x.dim()](x, w, **extra_args)
    diff_w = boft_diff_weight(w, blocks, rescale, 0.1)
    diff_y_rebuild = FUNC_LIST[x.dim()](x, diff_w, **extra_args)
    diff_y = boft_bypass_forward_diff(y, w.dim()>2, blocks, rescale, 0.1)
    
    print(y.shape, diff_y.shape, diff_y_rebuild.shape)
    print(w.shape, diff_w.shape)

    print(F.mse_loss(y, y + diff_y))
    print(F.mse_loss(w, w + diff_w))
    print(F.mse_loss(diff_y, diff_y_rebuild))
