import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general import rebuild_tucker, FUNC_LIST
from .general import factorization


def make_kron(w1, w2, scale):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    return rebuild * scale


def lokr_weight_gen(
    org_weight,
    rank,
    tucker=True,
    factor=-1,
    decompose_both=False,
    full_matrix=False,
    unbalanced_factorization=False,
):
    """### lokr_weight_gen

    Args:
        org_weight (torch.Tensor): the weight tensor
        rank (int): low rank

    Returns:
        torch.Tensor | None: w1, w1a, w1b, w2, w2a, w2b, t2
    """
    out_dim, in_dim, *k = org_weight.shape
    lokr_w1 = lokr_w1_a = lokr_w1_b = None
    lokr_w2 = lokr_w2_a = lokr_w2_b = None
    t2 = None

    if k:
        k_size = k
        shape = (out_dim, in_dim, *k_size)

        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)
        if unbalanced_factorization:
            out_l, out_k = out_k, out_l
        shape = ((out_l, out_k), (in_m, in_n), *k_size)  # ((a, b), (c, d), *k_size)
        tucker = tucker and any(i != 1 for i in k_size)
        if (
            decompose_both
            and rank < max(shape[0][0], shape[1][0]) / 2
            and not full_matrix
        ):
            lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], rank))
            lokr_w1_b = nn.Parameter(torch.empty(rank, shape[1][0]))
        else:
            lokr_w1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))  # a*c, 1-mode

        if rank >= max(shape[0][1], shape[1][1]) / 2 or full_matrix:
            use_w2 = True
            lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1], *k_size))
        elif tucker:
            lokr_t2 = nn.Parameter(torch.empty(rank, rank, shape[2], shape[3]))
            lokr_w2_a = nn.Parameter(torch.empty(rank, shape[0][1]))  # b, 1-mode
            lokr_w2_b = nn.Parameter(torch.empty(rank, shape[1][1]))  # d, 2-mode
        else:  # Conv2d not tucker
            # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
            lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], rank))
            lokr_w2_b = nn.Parameter(
                torch.empty(rank, shape[1][1] * torch.tensor(shape[2:]).prod().item())
            )
            # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)
    else:  # Linear
        shape = (out_dim, in_dim)

        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)
        if unbalanced_factorization:
            out_l, out_k = out_k, out_l
        shape = (
            (out_l, out_k),
            (in_m, in_n),
        )  # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
        # smaller part. weight scale
        if decompose_both and rank < max(shape[0][0], shape[1][0]) / 2:
            lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], rank))
            lokr_w1_b = nn.Parameter(torch.empty(rank, shape[1][0]))
        else:
            lokr_w1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))  # a*c, 1-mode
        if rank < max(shape[0][1], shape[1][1]) / 2:
            # bigger part. weight and LoRA. [b, dim] x [dim, d]
            lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], rank))
            lokr_w2_b = nn.Parameter(torch.empty(rank, shape[1][1]))
            # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
        else:
            lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))

    return lokr_w1, lokr_w1_a, lokr_w1_b, lokr_w2, lokr_w2_a, lokr_w2_b, t2


def lokr_diff_weight(w1, w1a, w1b, w2, w2a, w2b, t, gamma=1.0):
    """### lokr_diff_weight

    Args:
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: ΔW
    """
    if w1a is not None:
        rank = w1a.shape[1]
    elif w2a is not None:
        rank = w2a.shape[1]
    else:
        rank = gamma
    scale = gamma / rank
    if w1 is None:
        w1 = w1a @ w1b
    if w2 is None:
        if t is None:
            w2 = w2a @ w2b
        else:
            w2 = rebuild_tucker(t, w2a, w2b)
    return make_kron(w1, w2, scale)


def lokr_bypass_forward_diff(x, w1, w1a, w1b, w2, w2a, w2b, t, gamma=1.0, extra_args={}):
    """### lokr_bypass_forward_diff

    Args:
        x (torch.Tensor): input tensor
        d (torch.Tensor): weight of down proj linear/conv layer
        u (torch.Tensor): weight of up proj linear/conv layer
        m (torch.Tensor, optional): middle weight of tucker decomposition
        gamma (float, optional): scale factor, normally alpha/rank here
        extra_args (dict, optional): extra args for forward func, \
            e.g. padding, stride for Conv1/2/3d

    Returns:
        torch.Tensor: output tensor
    """
    pass


if __name__ == "__main__":
    w = torch.randn(32, 32, 3, 3, 3)
    d, u, m = lokr_weight_gen(w, 4)
    u = u + 0.01
    extra_args = {"padding": 1}

    x = torch.randn(1, 32, 8, 8, 8)
    y = FUNC_LIST[d.dim()](x, w, **extra_args)
    diff_w = lokr_diff_weight(d, u, m, 1)
    diff_y = lokr_bypass_forward_diff(x, d, u, m, 1, extra_args)

    print(F.mse_loss(y, y + diff_y))
    print(F.mse_loss(w, w + diff_w))
