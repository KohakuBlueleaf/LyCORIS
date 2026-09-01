import math

import torch
import torch.nn as nn

from .general import rebuild_tucker, FUNC_LIST
from ..kernels.autograd.locon import locon_bypass_diff, locon_diff_weight
from ..kernels.select import FUSED, call_compiled, choose, static_scale


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


def _diff_weight(d, u, m, gamma):
    """ΔW = gamma·(up @ down); with a tucker mid, ΔW = gamma·(mid ×_p up ×_q down).

    Reference body: what the compile tier compiles and the fused kernels match.
    """
    out_dim, in_dim = u.shape[0], d.shape[1]
    u = u * gamma
    if m is None:
        k = d.shape[2:]
        result = u.reshape(-1, u.size(1)) @ d.reshape(d.size(0), -1)
    else:
        k = m.shape[2:]
        u = u.reshape(u.size(0), -1).transpose(0, 1)
        result = rebuild_tucker(m, u, d.reshape(d.size(0), -1))
    return result.reshape(out_dim, in_dim, *k)


def _bypass_diff(x, d, u, m, gamma, extra_args):
    """y = gamma·up(down(x)), the down/mid/up chain in the layer's own op."""
    if m is not None:
        down = FUNC_LIST[d.dim()](x, d)
        mid = FUNC_LIST[d.dim()](down, m, **extra_args)
        up = FUNC_LIST[d.dim()](mid, u)
    else:
        down = FUNC_LIST[d.dim()](x, d, **extra_args)
        up = FUNC_LIST[d.dim()](down, u)
    return up * gamma


def diff_weight(*weights: tuple[torch.Tensor], gamma=1.0, backend=None):
    """### diff_weight

    Get ΔW = BA, where BA is low rank decomposition

    Args:
        weights (tuple[torch.Tensor]): (down, up[, mid])
        gamma (float, optional): scale factor, normally alpha/rank here
        backend (str, optional): pin one of triton/tilelang/compile/torch;
            the default picks per call, in that order

    Returns:
        torch.Tensor: ΔW
    """
    d, u, m = weights
    pick = choose((d, u, m), supported=static_scale(gamma), backend=backend)
    if pick in FUSED:
        return locon_diff_weight(d, u, m, gamma, backend=pick)
    if pick == "compile":
        return call_compiled(_diff_weight, d, u, m, gamma)
    return _diff_weight(d, u, m, gamma)


def bypass_forward_diff(x, org_out, *weights, gamma=1.0, extra_args={}, backend=None):
    """### bypass_forward_diff

    Args:
        x (torch.Tensor): input tensor
        weights (tuple[torch.Tensor]): (down, up[, mid])
        gamma (float, optional): scale factor, normally alpha/rank here
        extra_args (dict, optional): extra args for forward func, \
            e.g. padding, stride for Conv1/2/3d
        backend (str, optional): pin one of triton/tilelang/compile/torch

    Returns:
        torch.Tensor: output tensor
    """
    d, u, m = weights
    linear = m is None and d.dim() == 2 and not extra_args
    pick = choose((x, d, u), supported=linear and static_scale(gamma), backend=backend)
    if pick in FUSED:
        return locon_bypass_diff(x, d, u, gamma, backend=pick)
    if pick == "compile":
        return call_compiled(_bypass_diff, x, d, u, m, gamma, extra_args)
    return _bypass_diff(x, d, u, m, gamma, extra_args)
