import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general import FUNC_LIST


class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1d, w1u, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(w1d, w1u, w2d, w2u, scale)
        diff_weight = ((w1u @ w1d) * (w2u @ w2d)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1d, w1u, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2u @ w2d)
        grad_w1u = temp @ w1d.T
        grad_w1d = w1u.T @ temp

        temp = grad_out * (w1u @ w1d)
        grad_w2u = temp @ w2d.T
        grad_w2d = w2u.T @ temp

        del temp
        return grad_w1d, grad_w1u, grad_w2d, grad_w2u, None


class HadaWeightTucker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1d, w1u, t2, w2d, w2u, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1d, w1u, t2, w2d, w2u, scale)

        rebuild1 = torch.einsum("i j ..., j r, i p -> p r ...", t1, w1d, w1u)
        rebuild2 = torch.einsum("i j ..., j r, i p -> p r ...", t2, w2d, w2u)

        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1d, w1u, t2, w2d, w2u, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        temp = torch.einsum("i j ..., j r -> i r ...", t2, w2d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w2u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w1u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w1u.T)
        del grad_w, temp

        grad_w1d = torch.einsum("i r ..., i j ... -> r j", t1, grad_temp)
        grad_t1 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w1d.T)
        del grad_temp

        temp = torch.einsum("i j ..., j r -> i r ...", t1, w1d)
        rebuild = torch.einsum("i j ..., i r -> r j ...", temp, w1u)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w2u = torch.einsum("r j ..., i j ... -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j ..., i r -> r j ...", grad_w, w2u.T)
        del grad_w, temp

        grad_w2d = torch.einsum("i r ..., i j ... -> r j", t2, grad_temp)
        grad_t2 = torch.einsum("i j ..., j r -> i r ...", grad_temp, w2d.T)
        del grad_temp
        return grad_t1, grad_w1d, grad_w1u, grad_t2, grad_w2d, grad_w2u, None


def make_weight(w1d, w1u, w2d, w2u, scale):
    return HadaWeight.apply(w1d, w1u, w2d, w2u, scale)


def make_weight_tucker(t1, w1d, w1u, t2, w2d, w2u, scale):
    return HadaWeightTucker.apply(t1, w1d, w1u, t2, w2d, w2u, scale)


def weight_gen(org_weight, rank, tucker=True):
    """### weight_gen

    Args:
        org_weight (torch.Tensor): the weight tensor
        rank (int): low rank

    Returns:
        torch.Tensor: w1d, w2d, w1u, w2u[, t1, t2]
    """
    out_dim, in_dim, *k = org_weight.shape
    if k and tucker:
        w1d = torch.empty(rank, in_dim)
        w1u = torch.empty(rank, out_dim)
        t1 = torch.empty(rank, rank, *k)
        w2d = torch.empty(rank, in_dim)
        w2u = torch.empty(rank, out_dim)
        t2 = torch.empty(rank, rank, *k)
        nn.init.normal_(t1, std=0.1)
        nn.init.normal_(t2, std=0.1)
    else:
        w1d = torch.empty(rank, in_dim)
        w1u = torch.empty(out_dim, rank)
        w2d = torch.empty(rank, in_dim)
        w2u = torch.empty(out_dim, rank)
        t1 = t2 = None
    nn.init.normal_(w1d, std=1)
    nn.init.constant_(w1u, 0)
    nn.init.normal_(w2d, std=1)
    nn.init.normal_(w2u, std=0.1)
    return w1d, w1u, w2d, w2u, t1, t2


def diff_weight(*weights, gamma=1.0):
    """### diff_weight

    Get ΔW = BA, where BA is low rank decomposition

    Args:
        wegihts (tuple[torch.Tensor]): (w1d, w2d, w1u, w2u[, t1, t2])
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: ΔW
    """
    w1d, w1u, w2d, w2u, t1, t2 = weights
    if t1 is not None and t2 is not None:
        R, I = w1d.shape
        R, O = w1u.shape
        R, R, *k = t1.shape
        result = make_weight_tucker(t1, w1d, w1u, t2, w2d, w2u, gamma)
    else:
        R, I, *k = w1d.shape
        O, R, *_ = w1u.shape
        w1d = w1d.reshape(w1d.size(0), -1)
        w1u = w1u.reshape(-1, w1u.size(1))
        w2d = w2d.reshape(w2d.size(0), -1)
        w2u = w2u.reshape(-1, w2u.size(1))
        result = make_weight(w1d, w1u, w2d, w2u, gamma)

    result = result.reshape(O, I, *k)
    return result


def bypass_forward_diff(x, org_out, *weights, gamma=1.0, extra_args={}):
    """### bypass_forward_diff

    Args:
        x (torch.Tensor): input tensor
        weights (tuple[torch.Tensor]): (w1d, w2d, w1u, w2u[, t1, t2])
        gamma (float, optional): scale factor, normally alpha/rank here
        extra_args (dict, optional): extra args for forward func, \
            e.g. padding, stride for Conv1/2/3d

    Returns:
        torch.Tensor: output tensor
    """
    w1d, w1u, w2d, w2u, t1, t2 = weights
    diff_w = diff_weight(w1d, w1u, w2d, w2u, t1, t2, gamma)
    return FUNC_LIST[w1d.dim() if t1 is None else t1.dim()](x, diff_w, **extra_args)
