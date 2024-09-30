import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general import rebuild_tucker, FUNC_LIST
from .general import factorization


def make_kron(w1, w2, scale):
    for _ in range(w2.dim() - w1.dim()):
        w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    if scale != 1:
        rebuild = rebuild * scale

    return rebuild


def weight_gen(
    org_weight,
    rank,
    tucker=True,
    factor=-1,
    decompose_both=False,
    full_matrix=False,
    unbalanced_factorization=False,
):
    """### weight_gen

    Args:
        org_weight (torch.Tensor): the weight tensor
        rank (int): low rank

    Returns:
        torch.Tensor | None: w1, w1a, w1b, w2, w2a, w2b, t2
    """
    out_dim, in_dim, *k = org_weight.shape
    w1 = w1a = w1b = None
    w2 = w2a = w2b = None
    t2 = None
    use_w1 = use_w2 = False

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
            w1a = torch.empty(shape[0][0], rank)
            w1b = torch.empty(rank, shape[1][0])
        else:
            use_w1 = True
            w1 = torch.empty(shape[0][0], shape[1][0])  # a*c, 1-mode

        if rank >= max(shape[0][1], shape[1][1]) / 2 or full_matrix:
            use_w2 = True
            w2 = torch.empty(shape[0][1], shape[1][1], *k_size)
        elif tucker:
            t2 = torch.empty(rank, rank, *shape[2:])
            w2a = torch.empty(rank, shape[0][1])  # b, 1-mode
            w2b = torch.empty(rank, shape[1][1])  # d, 2-mode
        else:  # Conv2d not tucker
            # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
            w2a = torch.empty(shape[0][1], rank)
            w2b = torch.empty(rank, shape[1][1], *shape[2:])
            # w1 ⊗ (w2a x w2b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)
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
            w1a = torch.empty(shape[0][0], rank)
            w1b = torch.empty(rank, shape[1][0])
        else:
            use_w1 = True
            w1 = torch.empty(shape[0][0], shape[1][0])  # a*c, 1-mode
        if rank < max(shape[0][1], shape[1][1]) / 2:
            # bigger part. weight and LoRA. [b, dim] x [dim, d]
            w2a = torch.empty(shape[0][1], rank)
            w2b = torch.empty(rank, shape[1][1])
            # w1 ⊗ (w2a x w2b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
        else:
            use_w2 = True
            w2 = torch.empty(shape[0][1], shape[1][1])

    if use_w2:
        torch.nn.init.constant_(w2, 1)
    else:
        if tucker:
            torch.nn.init.kaiming_uniform_(t2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(w2a, a=math.sqrt(5))
        torch.nn.init.constant_(w2b, 1)

    if use_w1:
        torch.nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
    else:
        torch.nn.init.kaiming_uniform_(w1a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(w1b, a=math.sqrt(5))

    return w1, w1a, w1b, w2, w2a, w2b, t2


def diff_weight(*weights, gamma=1.0):
    """### diff_weight

    Args:
        weights (tuple[torch.Tensor]): (w1, w1a, w1b, w2, w2a, w2b, t)
        gamma (float, optional): scale factor, normally alpha/rank here

    Returns:
        torch.Tensor: ΔW
    """
    w1, w1a, w1b, w2, w2a, w2b, t = weights
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
            r, o, *k = w2b.shape
            w2 = w2a @ w2b.view(r, -1)
            w2 = w2.view(-1, o, *k)
        else:
            w2 = rebuild_tucker(t, w2a, w2b)
    return make_kron(w1, w2, scale)


def bypass_forward_diff(h, org_out, *weights, gamma=1.0, extra_args={}):
    """### bypass_forward_diff

    Args:
        weights (tuple[torch.Tensor]): (w1, w1a, w1b, w2, w2a, w2b, t)
        gamma (float, optional): scale factor, normally alpha/rank here
        extra_args (dict, optional): extra args for forward func, \
            e.g. padding, stride for Conv1/2/3d

    Returns:
        torch.Tensor: output tensor
    """
    w1, w1a, w1b, w2, w2a, w2b, t = weights
    use_w1 = w1 is not None
    use_w2 = w2 is not None
    tucker = t is not None
    dim = t.dim() if tucker else w2.dim() if w2 is not None else w2b.dim()
    rank = w1b.size(0) if not use_w1 else w2b.size(0) if not use_w2 else gamma
    scale = gamma / rank
    is_conv = dim > 2
    op = FUNC_LIST[dim]

    if is_conv:
        kw_dict = extra_args
    else:
        kw_dict = {}

    if use_w2:
        ba = w2
    else:
        a = w2b
        b = w2a

        if t is not None:
            a = a.view(*a.shape, *[1] * (dim - 2))
            b = b.view(*b.shape, *[1] * (dim - 2))
        elif is_conv:
            b = b.view(*b.shape, *[1] * (dim - 2))

    if use_w1:
        c = w1
    else:
        c = w1a @ w1b
    uq = c.size(1)

    if is_conv:
        # (b, uq), vq, ...
        B, _, *rest = h.shape
        h_in_group = h.reshape(B * uq, -1, *rest)
    else:
        # b, ..., uq, vq
        h_in_group = h.reshape(*h.shape[:-1], uq, -1)

    if use_w2:
        hb = op(h_in_group, ba, **kw_dict)
    else:
        if is_conv:
            if tucker:
                ha = op(h_in_group, a)
                ht = op(ha, t, **kw_dict)
                hb = op(ht, b)
            else:
                ha = op(h_in_group, a, **kw_dict)
                hb = op(ha, b)
        else:
            ha = op(h_in_group, a, **kw_dict)
            hb = op(ha, b)

    if is_conv:
        # (b, uq), vp, ..., f
        # -> b, uq, vp, ..., f
        # -> b, f, vp, ..., uq
        hb = hb.view(B, -1, *hb.shape[1:])
        h_cross_group = hb.transpose(1, -1)
    else:
        # b, ..., uq, vq
        # -> b, ..., vq, uq
        h_cross_group = hb.transpose(-1, -2)

    hc = F.linear(h_cross_group, c)
    if is_conv:
        # b, f, vp, ..., up
        # -> b, up, vp, ... ,f
        # -> b, c, ..., f
        hc = hc.transpose(1, -1)
        h = hc.reshape(B, -1, *hc.shape[3:])
    else:
        # b, ..., vp, up
        # -> b, ..., up, vp
        # -> b, ..., c
        hc = hc.transpose(-1, -2)
        h = hc.reshape(*hc.shape[:-2], -1)

    return h * scale
