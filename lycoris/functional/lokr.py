import functools
import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .general import rebuild_tucker, FUNC_LIST
from .general import factorization
from ..kernels.autograd.lokr import lokr_kron_bypass, lokr_kron_weight, rank_scale
from ..kernels.dispatch import fused_backends
from ..kernels.select import FUSED, call_compiled, choose, static_scale


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
        torch.nn.init.constant_(w2, 0)
    else:
        if tucker:
            torch.nn.init.kaiming_uniform_(t2, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(w2a, a=math.sqrt(5))
        torch.nn.init.constant_(w2b, 0)

    if use_w1:
        torch.nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
    else:
        torch.nn.init.kaiming_uniform_(w1a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(w1b, a=math.sqrt(5))

    return w1, w1a, w1b, w2, w2a, w2b, t2


def _kron_weight(w1, w1a, w1b, w2, w2a, w2b, t, scale):
    """ΔW = scale · kron(w1, w2), each half rebuilt from its factors first."""
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


def _plain_w2(w2, w2b, t):
    """The kron's second half is a plain matrix, which is the fused layout.

    A conv w2 keeps spatial axes that kron must broadcast over, and a tucker
    w2 rebuilds into exactly that shape.
    """
    return w2.dim() == 2 if w2 is not None else (t is None and w2b.dim() == 2)


@functools.cache
def _apply_factor_cap() -> float:
    """The factor-size bound declared by the available fused apply kernels.

    The kron apply kernel holds every factor tile on chip, so each factor dim
    is bounded (see kernels.*.lokr.bypass.MAX_KRON_FACTOR); the merge kernels
    tile the output and carry no such bound. The strictest available fused
    backend wins; none declaring a cap means the apply is treated as
    unbounded.
    """
    caps = []
    for name in fused_backends():
        try:
            mod = importlib.import_module(f"lycoris.kernels.{name}.lokr.bypass")
        except Exception:  # a backend that cannot import is simply absent here
            continue
        cap = getattr(mod, "MAX_KRON_FACTOR", None)
        if cap is not None:
            caps.append(cap)
    return min(caps) if caps else math.inf


def _apply_supported(w1, w1a, w1b, w2, w2a, w2b, t) -> bool:
    """The layout AND factor-size scope of the fused kron apply kernels.

    Factor dims come off the parameters without building anything: a
    factorized half rebuilds to the outer-product shape of its parts, so
    (out/factor, in/factor) factors — the common LoKr case — are caught here
    rather than at kernel launch.
    """
    # Avoid the lazy backend imports in _apply_factor_cap while an enclosing
    # compiler is tracing. choose() selects the torch body in this case, which
    # the enclosing compiler captures as part of its graph.
    if torch.compiler.is_compiling() or not _plain_w2(w2, w2b, t):
        return False
    a, b = w1.shape if w1 is not None else (w1a.shape[0], w1b.shape[1])
    c, d = w2.shape if w2 is not None else (w2a.shape[0], w2b.shape[1])
    return max(a, b, c, d) <= _apply_factor_cap()


def kron_weight(w1, w1a, w1b, w2, w2a, w2b, t=None, scale=1.0, backend=None):
    """ΔW = scale · kron(w1, w2) with the backend chosen per call.

    Only the larger side is ever factorized, so at most one of w1/w2 arrives
    as a pair and the fused kernel generates that side's tile in-register.
    """
    tensors = (w1, w1a, w1b, w2, w2a, w2b, t)
    pick = choose(
        tensors,
        supported=_plain_w2(w2, w2b, t) and static_scale(scale),
        backend=backend,
    )
    if pick in FUSED:
        return lokr_kron_weight(
            w1, w1a, w1b, w2, w2a, w2b, t, scale=scale, backend=pick
        )
    if pick == "compile":
        return call_compiled(_kron_weight, w1, w1a, w1b, w2, w2a, w2b, t, scale)
    return _kron_weight(w1, w1a, w1b, w2, w2a, w2b, t, scale)


def diff_weight(*weights, gamma=1.0, backend=None):
    """### diff_weight

    Args:
        weights (tuple[torch.Tensor]): (w1, w1a, w1b, w2, w2a, w2b, t)
        gamma (float, optional): scale factor, normally alpha/rank here
        backend (str, optional): pin one of triton/tilelang/compile/torch

    Returns:
        torch.Tensor: ΔW
    """
    w1, w1a, w1b, w2, w2a, w2b, t = weights
    return kron_weight(
        w1,
        w1a,
        w1b,
        w2,
        w2a,
        w2b,
        t,
        scale=rank_scale(w1a, w2a, gamma),
        backend=backend,
    )


def _kron_bypass(h, w1, w1a, w1b, w2, w2a, w2b, t, scale, extra_args):
    """y = scale · vec(w1 · unvec(h) · w2ᵀ), as a grouped op chain on h."""
    use_w1 = w1 is not None
    use_w2 = w2 is not None
    tucker = t is not None
    dim = t.dim() if tucker else w2.dim() if w2 is not None else w2b.dim()
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


def kron_bypass(
    h, w1, w1a, w1b, w2, w2a, w2b, t=None, scale=1.0, extra_args={}, backend=None
):
    """Bypass apply of scale · kron(w1, w2), backend chosen per call.

    The fused path is the linear layout: one kernel per token tile computing
    w1 @ X @ w2ᵀ, so ΔW (a·c by b·d) is never built. That kernel holds the
    factor tiles in registers, so factors beyond its size bound step down a
    tier instead of failing at launch.
    """
    pick = choose(
        (h, w1, w1a, w1b, w2, w2a, w2b),
        supported=_apply_supported(w1, w1a, w1b, w2, w2a, w2b, t)
        and not extra_args
        and static_scale(scale),
        backend=backend,
    )
    if pick in FUSED:
        return lokr_kron_bypass(
            h, w1, w1a, w1b, w2, w2a, w2b, t, scale=scale, backend=pick
        )
    if pick == "compile":
        return call_compiled(
            _kron_bypass, h, w1, w1a, w1b, w2, w2a, w2b, t, scale, extra_args
        )
    return _kron_bypass(h, w1, w1a, w1b, w2, w2a, w2b, t, scale, extra_args)


def bypass_forward_diff(h, org_out, *weights, gamma=1.0, extra_args={}, backend=None):
    """### bypass_forward_diff

    Args:
        weights (tuple[torch.Tensor]): (w1, w1a, w1b, w2, w2a, w2b, t)
        gamma (float, optional): scale factor, normally alpha/rank here
        extra_args (dict, optional): extra args for forward func, \
            e.g. padding, stride for Conv1/2/3d
        backend (str, optional): pin one of triton/tilelang/compile/torch

    Returns:
        torch.Tensor: output tensor
    """
    w1, w1a, w1b, w2, w2a, w2b, t = weights
    # The bypass reads rank off the b factors (w1b/w2b), the rebuild off the a
    # factors; both name the same rank of the one factorized side.
    rank = w1b.size(0) if w1 is None else w2b.size(0) if w2 is None else gamma
    return kron_bypass(
        h,
        w1,
        w1a,
        w1b,
        w2,
        w2a,
        w2b,
        t,
        scale=gamma / rank,
        extra_args=extra_args,
        backend=backend,
    )
