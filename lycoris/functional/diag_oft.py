import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general import factorization, FUNC_LIST


def get_r(oft_blocks, I=None, constraint=0):
    if I is None:
        I = torch.eye(oft_blocks.shape[-1], device=oft_blocks.device)
    if I.ndim < oft_blocks.ndim:
        for _ in range(oft_blocks.ndim - I.ndim):
            I = I.unsqueeze(0)
    # for Q = -Q^T
    q = oft_blocks - oft_blocks.transpose(-1, -2)
    normed_q = q
    if constraint > 0:
        q_norm = torch.norm(q) + 1e-8
        if q_norm > constraint:
            normed_q = q * constraint / q_norm
    # use float() to prevent unsupported type
    r = (I + normed_q) @ (I - normed_q).float().inverse()
    return r


def daig_oft_weight_gen(org_weight, max_block_size, rescale=False):
    """### daig_oft_weight_gen

    Args:
        org_weight (torch.Tensor): the weight tensor
        max_block_size (int): max block size
        rescale (bool, optional): whether to rescale the weight. Defaults to False.

    Returns:
        torch.Tensor: oft_blocks[, rescale_weight]
    """
    out_dim, *rest = org_weight.shape
    block_size, block_num = factorization(out_dim, max_block_size)
    oft_blocks = torch.zeros(block_num, block_size, block_size)
    if rescale:
        return oft_blocks, torch.ones(out_dim, *[1] * len(rest))
    else:
        return oft_blocks, None


def daig_oft_diff_weight(org_weight, oft_blocks, rescale=None, constraint=None):
    """### daig_oft_diff_weight

    Args:
        TODO

    Returns:
        torch.Tensor: ΔW
    """
    I = torch.eye(oft_blocks.shape[1], device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)

    block_num, block_size, _ = oft_blocks.shape
    _, *shape = org_weight.shape
    org_weight = org_weight.to(dtype=r.dtype)
    org_weight = org_weight.view(block_num, block_size, *shape)
    # Init R=0, so add I on it to ensure the output of step0 is original model output
    weight = torch.einsum(
        "k n m, k n ... -> k m ...",
        r - I,
        org_weight,
    ).view(-1, *shape)
    if rescale is None:
        weight = rescale * weight
        weight = weight + (rescale - 1) * org_weight
    return weight


def daig_oft_bypass_forward_diff(
    org_out, need_transpose, oft_blocks, rescale=None, constraint=None
):
    """### daig_oft_bypass_forward_diff

    Args:
        TODO

    Returns:
        torch.Tensor: output tensor
    """
    block_num, block_size, _ = oft_blocks.shape
    I = torch.eye(block_size, device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)
    if need_transpose:
        org_out = org_out.transpose(1, -1)
    *shape, _ = org_out.shape
    oft_out = torch.einsum(
        "k n m, ... k n -> ... k m", r - I, org_out.view(*shape, block_num, block_size)
    )
    out = oft_out.view(*shape, -1)
    if rescale is not None:
        out = rescale.transpose(-1, 0) * out
        out = out + (rescale - 1).transpose(-1, 0) * org_out
    if need_transpose:
        out = out.transpose(1, -1)
    return out


if __name__ == "__main__":
    w = torch.randn(32, 32, 3, 3, 3)
    blocks, rescale = daig_oft_weight_gen(w, 4, True)
    blocks = blocks + torch.randn_like(blocks) * 0.01
    extra_args = {"padding": 1}

    x = torch.randn(1, 32, 8, 8, 8)
    y = FUNC_LIST[x.dim()](x, w, **extra_args)
    diff_w = daig_oft_diff_weight(w, blocks, rescale, 0.1)
    diff_y_rebuild = FUNC_LIST[x.dim()](x, diff_w, **extra_args)
    diff_y = daig_oft_bypass_forward_diff(y, w.dim() > 2, blocks, rescale, 0.1)

    print(F.mse_loss(y, y + diff_y))
    print(F.mse_loss(w, w + diff_w))
    print(F.mse_loss(diff_y, diff_y_rebuild))
