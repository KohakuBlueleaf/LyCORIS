import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.autograd.dora import apply_dora
from ..kernels.autograd.full import full_diff_weight
from ..kernels.select import FUSED, call_compiled, choose, static_scale

FUNC_LIST = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]


def rebuild_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    return rebuild2


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    second value is a value for weight.

    Because of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    """

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def power2factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    m = 2k
    n = 2**p
    m*n = dim
    """
    if factor == -1:
        factor = dimension

    # Find the first solution and check if it is even doable
    m = n = 0
    while m <= factor:
        m += 2
        while dimension % m != 0 and m < dimension:
            m += 2
        if m > factor:
            break
        if sum(int(i) for i in f"{dimension//m:b}") == 1:
            n = dimension // m

    if n == 0:
        return None, n
    return dimension // n, n


def tucker_weight_from_conv(up, down, mid):
    up = up.reshape(up.size(0), up.size(1))
    down = down.reshape(down.size(0), down.size(1))
    return torch.einsum("m n ..., i m, n j -> i j ...", mid, up, down)


def tucker_weight(wa, wb, t):
    temp = torch.einsum("i j ..., j r -> i r ...", t, wb)
    return torch.einsum("i j ..., i r -> r j ...", temp, wa)


def _add_scaled(base, delta, gamma):
    """W = W_org + gamma·ΔW."""
    return base + delta * gamma


def add_scaled(base, delta, gamma=1.0, backend=None):
    """W_org + gamma·ΔW in one pass — the full and norm merge.

    Eager reads delta twice (scale, then add); the fused op reads each operand
    once and writes once.
    """
    pick = choose((base, delta), supported=static_scale(gamma), backend=backend)
    if pick in FUSED:
        return full_diff_weight(base, delta, gamma, backend=pick)
    if pick == "compile":
        return call_compiled(_add_scaled, base, delta, gamma)
    return _add_scaled(base, delta, gamma)


def _weight_decompose(weight, dora_scale, multiplier, wd_on_out):
    """W' = W · (mult·(m/‖W‖ − 1) + 1), the norm per out row or per in column."""
    weight = weight.to(dora_scale.dtype)
    norm_dims = weight.dim() - 1
    if wd_on_out:
        weight_norm = (
            weight.reshape(weight.shape[0], -1)
            .norm(dim=1)
            .reshape(weight.shape[0], *[1] * norm_dims)
        ) + torch.finfo(weight.dtype).eps
    else:
        weight_norm = (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[1], *[1] * norm_dims)
            .transpose(0, 1)
        ) + torch.finfo(weight.dtype).eps

    scale = dora_scale.to(weight.device) / weight_norm
    if multiplier != 1:
        scale = multiplier * (scale - 1) + 1
    return weight * scale


def weight_decompose(weight, dora_scale, multiplier=1, wd_on_out=True, backend=None):
    """DoRA epilogue on an already-merged weight — dora, doha and dokr alike.

    One fused kernel: the norm and the rescale share the pass over W, so W is
    read once instead of the eager chain's three times. wd_on_out=False on a
    conv needs a per-in-channel norm over (out, spatial), which no 2D row view
    expresses, so that case takes the tier below.
    """
    flat = wd_on_out or weight.dim() == 2
    pick = choose(
        (weight, dora_scale),
        supported=flat and static_scale(multiplier),
        backend=backend,
    )
    if pick in FUSED:
        return apply_dora(weight, dora_scale, multiplier, wd_on_out, backend=pick)
    if pick == "compile":
        return call_compiled(
            _weight_decompose, weight, dora_scale, multiplier, wd_on_out
        )
    return _weight_decompose(weight, dora_scale, multiplier, wd_on_out)


def apply_dora_scale(org_weight, rebuild, dora_scale, scale):
    dora_norm_dims = org_weight.dim() - 1
    weight = org_weight + rebuild
    weight = weight.to(dora_scale.dtype)
    weight_norm = (
        weight.transpose(0, 1)
        .reshape(weight.shape[1], -1)
        .norm(dim=1, keepdim=True)
        .reshape(weight.shape[1], *[1] * dora_norm_dims)
        .transpose(0, 1)
    )
    merged_scale1 = weight / weight_norm * dora_scale
    diff_weight = merged_scale1 - org_weight
    return org_weight + diff_weight * scale
