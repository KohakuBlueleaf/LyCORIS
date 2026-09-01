import torch

from .general import factorization
from ..kernels.autograd.diag_oft import diag_oft_bypass_diff, diag_oft_diff_weight
from ..kernels.select import FUSED, call_compiled, choose

# The in-kernel Cayley solves one block per CTA in registers; past 32 the
# Gauss-Jordan tile stops fitting and the eager inverse is the fallback.
MAX_BLOCK = 32


def get_r(oft_blocks, I=None, constraint=0):
    if I is None:
        I = torch.eye(oft_blocks.shape[-1], device=oft_blocks.device)
    if I.ndim < oft_blocks.ndim:
        for _ in range(oft_blocks.ndim - I.ndim):
            I = I.unsqueeze(0)
    # for Q = -Q^T
    q = oft_blocks - oft_blocks.transpose(-1, -2)
    normed_q = q
    if constraint is not None and constraint > 0:
        q_norm = torch.norm(q) + 1e-8
        if q_norm > constraint:
            normed_q = q * constraint / q_norm
    # use float() to prevent unsupported type
    r = (I + normed_q) @ (I - normed_q).float().inverse()
    return r


def weight_gen(org_weight, max_block_size=-1, rescale=False):
    """### weight_gen

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


def _diff_weight(org_weight, oft_blocks, rescale, constraint):
    """ΔW = (R − I)·W blockwise, with the rescale row scaling folded in."""
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
    if rescale is not None:
        weight = rescale * weight
        weight = weight + (rescale - 1) * org_weight.reshape(weight.shape)
    return weight


def diff_weight(org_weight, *weights, constraint=None, backend=None):
    """### diff_weight

    The fused path runs Cayley, the rescale fold and the identity shift in one
    kernel, so R is never materialised.

    Args:
        org_weight (torch.Tensor): the weight tensor of original model
        weights (tuple[torch.Tensor]): (oft_blocks[, rescale_weight])
        constraint (float, optional): constraint for oft
        backend (str, optional): pin one of triton/tilelang/compile/torch

    Returns:
        torch.Tensor: ΔW
    """
    oft_blocks, rescale = weights
    pick = choose(
        (org_weight, oft_blocks, rescale),
        supported=oft_blocks.shape[-1] <= MAX_BLOCK,
        backend=backend,
    )
    if pick in FUSED:
        return diag_oft_diff_weight(
            org_weight, oft_blocks, rescale, constraint, backend=pick
        )
    if pick == "compile":
        return call_compiled(_diff_weight, org_weight, oft_blocks, rescale, constraint)
    return _diff_weight(org_weight, oft_blocks, rescale, constraint)


def _bypass_diff(x, org_out, oft_blocks, rescale, constraint, need_transpose):
    """Δy = (R − I)·y on the channel axis, R shared by every token."""
    block_num, block_size, _ = oft_blocks.shape
    I = torch.eye(block_size, device=oft_blocks.device)
    r = get_r(oft_blocks, I, constraint)
    if need_transpose:
        org_out = org_out.transpose(1, -1)
    org_out = org_out.to(dtype=r.dtype)
    *shape, _ = org_out.shape
    oft_out = torch.einsum(
        "k n m, ... k n -> ... k m", r - I, org_out.view(*shape, block_num, block_size)
    )
    out = oft_out.reshape(*shape, -1)
    if rescale is not None:
        out = rescale.transpose(-1, 0) * out
        out = out + (rescale - 1).transpose(-1, 0) * org_out
    if need_transpose:
        out = out.transpose(1, -1)
    return out


def bypass_forward_diff(
    x, org_out, *weights, constraint=None, need_transpose=False, backend=None
):
    """### bypass_forward_diff

    Args:
        x (torch.Tensor): the input tensor for original model
        org_out (torch.Tensor): the output tensor from original model
        weights (tuple[torch.Tensor]): (oft_blocks[, rescale_weight])
        constraint (float, optional): constraint for oft
        need_transpose (bool, optional): `True` when "dim" is not the last
            axis, as in convolution layers
        backend (str, optional): pin one of triton/tilelang/compile/torch

    Returns:
        torch.Tensor: output tensor
    """
    oft_blocks, rescale = weights
    pick = choose(
        (org_out, oft_blocks, rescale),
        supported=oft_blocks.shape[-1] <= MAX_BLOCK,
        backend=backend,
    )
    if pick in FUSED:
        return diag_oft_bypass_diff(
            org_out, oft_blocks, rescale, constraint, need_transpose, backend=pick
        )
    if pick == "compile":
        return call_compiled(
            _bypass_diff, x, org_out, oft_blocks, rescale, constraint, need_transpose
        )
    return _bypass_diff(x, org_out, oft_blocks, rescale, constraint, need_transpose)
