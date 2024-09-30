import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general import rebuild_tucker, FUNC_LIST


def weight_gen(org_weight, rank, tucker=True):
    """### weight_gen

    Args:
        org_weight (torch.Tensor): the weight tensor
        rank (int): low rank

    Returns:
        torch.Tensor: down, up[, mid]
    """
    out_dim, in_dim, *k = org_weight.shape
    if k and tucker:
        down = torch.empty(rank, in_dim, *(1 for _ in k))
        up = torch.empty(out_dim, rank, *(1 for _ in k))
        mid = torch.empty(rank, rank, *k)
        nn.init.kaiming_uniform_(down, a=math.sqrt(5))
        nn.init.constant_(up, 0)
        nn.init.kaiming_uniform_(mid, a=math.sqrt(5))
        return down, up, mid
    else:
        down = torch.empty(rank, in_dim)
        up = torch.empty(out_dim, rank)
        nn.init.kaiming_uniform_(down, a=math.sqrt(5))
        nn.init.constant_(up, 0)
        return down, up, None


def diff_weight(*weights: tuple[torch.Tensor], gamma=1.0):
    """### diff_weight

    Get ΔW = BA, where BA is low rank decomposition

    Args:
        weights (tuple[torch.Tensor]): (down, up[, mid])
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: ΔW
    """
    d, u, m = weights
    R, I, *k = d.shape
    O, R, *_ = u.shape
    u = u * gamma

    if m is None:
        result = u.reshape(-1, u.size(1)) @ d.reshape(d.size(0), -1)
    else:
        R, R, *k = m.shape
        u = u.reshape(u.size(0), -1).transpose(0, 1)
        d = d.reshape(d.size(0), -1)
        result = rebuild_tucker(m, u, d)
    return result.reshape(O, I, *k)


def bypass_forward_diff(x, org_out, *weights, gamma=1.0, extra_args={}):
    """### bypass_forward_diff

    Args:
        x (torch.Tensor): input tensor
        weights (tuple[torch.Tensor]): (down, up[, mid])
        gamma (float, optional): scale factor, normally alpha/rank here
        extra_args (dict, optional): extra args for forward func, \
            e.g. padding, stride for Conv1/2/3d

    Returns:
        torch.Tensor: output tensor
    """
    d, u, m = weights
    if m is not None:
        down = FUNC_LIST[d.dim()](x, d)
        mid = FUNC_LIST[d.dim()](down, m, **extra_args)
        up = FUNC_LIST[d.dim()](mid, u)
    else:
        down = FUNC_LIST[d.dim()](x, d, **extra_args)
        up = FUNC_LIST[d.dim()](down, u)
    return up * gamma
