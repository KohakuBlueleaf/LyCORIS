import torch
import torch.nn as nn
import torch.nn.functional as F


def lora_diff_weight(a, b, gamma=1.0):
    """### lora_diff_weight

    Get ΔW = BA, where BA is low rank decomposition

    Args:
        a (torch.Tensor): weight of down proj linear/conv layer
        b (torch.Tensor): weight of up proj linear/conv layer
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: ΔW
    """
    assert a.size(0) == b.size(1)
    R, I, *k = a.shape
    O, R, *_ = b.shape

    result = b.reshape(-1, b.size(1)) @ (a.reshape(a.size(0), -1) * gamma)
    return result.reshape(O, I, *k)


FUNC_LIST = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]


def lora_bypass_forward_diff(x, a, b, gamma=1.0, extra_args={}):
    """### lora_bypass_forward_diff

    Args:
        x (torch.Tensor): input tensor
        a (torch.Tensor): weight of down proj linear/conv layer
        b (torch.Tensor): weight of up proj linear/conv layer
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: output tensor
    """
    assert a.size(0) == b.size(1)
    R, I, *k = a.shape
    O, R, *_ = b.shape

    down = FUNC_LIST[a.dim()](x, a, **extra_args)
    up = FUNC_LIST[a.dim()](down, b)
    return up * gamma


if __name__ == "__main__":
    w = torch.randn(128, 128, 3, 3)
    a = torch.randn(16, 128, 3, 3) * 0.01
    b = torch.randn(128, 16, 1, 1) * 0.1
    extra_args = {"padding": 1}

    x = torch.randn(1, 128, 8, 8)
    y = FUNC_LIST[a.dim()](x, w, **extra_args)
    diff_w = lora_diff_weight(a, b, 4)
    diff_y = lora_bypass_forward_diff(x, a, b, 4, extra_args)

    print(F.mse_loss(y, y + diff_y))
    print(F.mse_loss(w, w + diff_w))
