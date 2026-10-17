"""Custom ops for the apply-style kernels: DoRA, diag-OFT, BOFT, IA3, scaled add.

Same contract as ``library``: fake implementations for tracing, real calls on
the operands' device, outputs that never alias inputs, empty placeholders for
absent optional outputs.
"""

import torch
from torch import Tensor

from .library import NS, _fresh, _run, ops_for


@torch.library.custom_op(f"{NS}::dora_fwd", mutates_args=(), device_types="cuda")
def dora_fwd(
    w: Tensor,
    dscale: Tensor,
    mult: float,
    row_axis: int,
    backend: str,
    delta: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """(y, norms) on M = w [+ delta], the add read in the kernel."""
    kw = {} if delta is None else {"delta": delta}
    ins = (w, dscale, delta)
    return _run(backend, "dora_fwd", ins, w, dscale, mult, row_axis, **kw)


@dora_fwd.register_fake
def _(w, dscale, mult, row_axis, backend, delta=None):
    return w.new_empty(w.shape), w.new_empty(w.shape[row_axis], dtype=torch.float32)


@torch.library.custom_op(f"{NS}::dora_bwd", mutates_args=(), device_types="cuda")
def dora_bwd(
    grad: Tensor,
    w: Tensor,
    dscale: Tensor,
    norms: Tensor,
    mult: float,
    row_axis: int,
    backend: str,
    delta: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    kw = {} if delta is None else {"delta": delta}
    ins = (grad, w, dscale, norms, delta)
    return _run(backend, "dora_bwd", ins, grad, w, dscale, norms, mult, row_axis, **kw)


@dora_bwd.register_fake
def _(grad, w, dscale, norms, mult, row_axis, backend, delta=None):
    return w.new_empty(w.shape), dscale.new_empty(w.shape[row_axis])


@torch.library.custom_op(f"{NS}::oft_fwd", mutates_args=(), device_types="cuda")
def oft_fwd(
    blocks: Tensor,
    x: Tensor,
    rescale: Tensor | None,
    cscale: float,
    shift: bool,
    weight: bool,
    backend: str,
) -> Tensor:
    ins = (blocks, x, rescale)
    return _run(backend, "oft_fwd", ins, blocks, x, rescale, cscale, shift, weight)


@oft_fwd.register_fake
def _(blocks, x, rescale, cscale, shift, weight, backend):
    return x.new_empty(x.shape)


@torch.library.custom_op(f"{NS}::oft_bwd", mutates_args=(), device_types="cuda")
def _oft_bwd(
    blocks: Tensor,
    x: Tensor,
    grad: Tensor,
    rescale: Tensor | None,
    cscale: float,
    shift: bool,
    weight: bool,
    backend: str,
) -> tuple[Tensor, Tensor, Tensor]:
    ins = (blocks, x, grad, rescale)
    with torch.cuda.device(x.device):
        gx, gb, gres = ops_for(backend, x.device).oft_bwd(
            blocks, x, grad, rescale, cscale, shift, weight
        )
    gres = blocks.new_empty(0) if gres is None else gres
    return tuple(_fresh([gx, gb, gres], ins))


@_oft_bwd.register_fake
def _(blocks, x, grad, rescale, cscale, shift, weight, backend):
    k, s, _ = blocks.shape
    gres = blocks.new_empty(k * s) if rescale is not None else blocks.new_empty(0)
    return x.new_empty(x.shape), blocks.new_empty(blocks.shape), gres


def oft_bwd(blocks, x, grad, rescale, cscale, shift, weight, backend):
    """(gx, g_blocks, g_rescale | None)."""
    gx, gb, gres = _oft_bwd(blocks, x, grad, rescale, cscale, shift, weight, backend)
    return gx, gb, (gres if rescale is not None else None)


@torch.library.custom_op(f"{NS}::boft_fwd", mutates_args=(), device_types="cuda")
def boft_fwd(
    blocks: Tensor,
    x: Tensor,
    axis: int,
    cscale: float,
    scale: float,
    rescale: Tensor | None,
    diff: bool,
    backend: str,
) -> Tensor:
    ins = (blocks, x, rescale)
    return _run(
        backend,
        "boft_fwd",
        ins,
        blocks,
        x,
        axis=axis,
        cscale=cscale,
        scale=scale,
        rescale=rescale,
        diff=diff,
    )


@boft_fwd.register_fake
def _(blocks, x, axis, cscale, scale, rescale, diff, backend):
    return x.new_empty(x.shape)


@torch.library.custom_op(f"{NS}::boft_bwd", mutates_args=(), device_types="cuda")
def boft_bwd(
    blocks: Tensor,
    x: Tensor,
    grad: Tensor,
    axis: int,
    cscale: float,
    scale: float,
    backend: str,
) -> tuple[Tensor, Tensor]:
    ins = (blocks, x, grad)
    return _run(
        backend,
        "boft_bwd",
        ins,
        blocks,
        x,
        grad,
        axis=axis,
        cscale=cscale,
        scale=scale,
    )


@boft_bwd.register_fake
def _(blocks, x, grad, axis, cscale, scale, backend):
    return grad.new_empty(grad.shape), blocks.new_empty(blocks.shape)


@torch.library.custom_op(f"{NS}::ia3_fwd", mutates_args=(), device_types="cuda")
def ia3_fwd(
    x: Tensor, w: Tensor, channel_axis: int, alpha: float, gamma: float, backend: str
) -> Tensor:
    return _run(backend, "ia3_fwd", (x, w), x, w, channel_axis, alpha, gamma)


@ia3_fwd.register_fake
def _(x, w, channel_axis, alpha, gamma, backend):
    return x.new_empty(x.shape, dtype=torch.promote_types(x.dtype, w.dtype))


@torch.library.custom_op(f"{NS}::ia3_bwd", mutates_args=(), device_types="cuda")
def ia3_bwd(
    grad: Tensor,
    x: Tensor,
    w: Tensor,
    channel_axis: int,
    alpha: float,
    gamma: float,
    backend: str,
) -> tuple[Tensor, Tensor]:
    ins = (grad, x, w)
    return _run(backend, "ia3_bwd", ins, grad, x, w, channel_axis, alpha, gamma)


@ia3_bwd.register_fake
def _(grad, x, w, channel_axis, alpha, gamma, backend):
    return x.new_empty(x.shape), w.new_empty(w.shape)


@torch.library.custom_op(f"{NS}::add_scaled", mutates_args=(), device_types="cuda")
def add_scaled(base: Tensor, delta: Tensor, gamma: float, backend: str) -> Tensor:
    return _run(backend, "add_scaled", (base, delta), base, delta, gamma)


@add_scaled.register_fake
def _(base, delta, gamma, backend):
    return base.new_empty(base.shape)
