import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general import rebuild_tucker


def lora_diff_weight(d, u, m=None, gamma=1.0):
    """### lora_diff_weight

    Get ΔW = BA, where BA is low rank decomposition

    Args:
        d (torch.Tensor): weight of down proj linear/conv layer
        u (torch.Tensor): weight of up proj linear/conv layer
        m (torch.Tensor, optional): weight of mid proj linear/conv layer, \
            for tucker deomposition
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: ΔW
    """
    assert d.size(0) == u.size(1)
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


FUNC_LIST = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]


def lora_bypass_forward_diff(x, d, u, m=None, gamma=1.0, extra_args={}):
    """### lora_bypass_forward_diff

    Args:
        x (torch.Tensor): input tensor
        d (torch.Tensor): weight of down proj linear/conv layer
        u (torch.Tensor): weight of up proj linear/conv layer
        m (torch.Tensor, optional): weight of mid proj linear/conv layer, \
            for tucker deomposition
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: output tensor
    """
    if m is not None:
        down = FUNC_LIST[d.dim()](x, d)
        mid = FUNC_LIST[d.dim()](down, m, **extra_args)
        up = FUNC_LIST[d.dim()](mid, u)
    else:
        down = FUNC_LIST[d.dim()](x, d, **extra_args)
        up = FUNC_LIST[d.dim()](down, u)
    return up * gamma


if __name__ == "__main__":
    w = torch.randn(32, 32, 3, 3, 3)
    d = torch.randn(4, 32, 1, 1, 1) * 0.01
    u = torch.randn(32, 4, 1, 1, 1) * 0.1
    m = torch.randn(4, 4, 3, 3, 3) * 0.1
    extra_args = {"padding": 1}

    x = torch.randn(1, 32, 8, 8, 8)
    y = FUNC_LIST[d.dim()](x, w, **extra_args)
    diff_w = lora_diff_weight(d, u, m, 1)
    diff_y = lora_bypass_forward_diff(x, d, u, m, 1, extra_args)

    print(F.mse_loss(y, y + diff_y))
    print(F.mse_loss(w, w + diff_w))
