"""The fused op set as torch.library custom ops (``torch.ops.lycoris.*``).

Every backend op the autograd layer calls is registered here once, with a
fake (meta) implementation, so ``torch.compile`` traces through the autograd
Functions with the fused kernels as opaque graph nodes instead of falling back
to the torch body. The real implementation runs on the operands' device, so
the planner and tuner see that device even in a mixed-GPU process. Outputs
never alias an input or each other (a custom-op requirement): a backend result
that shares storage is cloned, and an absent optional output is an empty
placeholder that the wrappers below turn back into None.
"""

import torch
from torch import Tensor

from .dispatch import fused_backends
from .ops import get_ops

NS = "lycoris"


def _fresh(outs, inputs):
    """Outputs contiguous (the fakes' layout), cloned where they share storage
    with an input or an earlier output."""
    taken = {
        t.untyped_storage().data_ptr()
        for t in inputs
        if isinstance(t, Tensor) and t.numel() > 0
    }
    fixed = []
    for o in outs:
        o = o.contiguous()
        if o.numel() > 0:
            ptr = o.untyped_storage().data_ptr()
            if ptr in taken:
                o = o.clone()
                ptr = o.untyped_storage().data_ptr()
            taken.add(ptr)
        fixed.append(o)
    return fixed


def ops_for(backend: str, device: torch.device):
    """The op set of ``backend``; an empty name takes the device's first fused backend."""
    if not backend:
        fused = fused_backends(device)
        if not fused:
            raise RuntimeError(f"no fused backend available on {device}")
        backend = fused[0]
    return get_ops(backend)


def _run(backend, op, inputs, *args, **kwargs):
    device = next(t.device for t in inputs if isinstance(t, Tensor))
    fn = getattr(ops_for(backend, device), op)
    if device.index == torch.cuda.current_device():
        out = fn(*args, **kwargs)
    else:
        with torch.cuda.device(device):
            out = fn(*args, **kwargs)
    single = isinstance(out, Tensor)
    outs = _fresh([out] if single else list(out), inputs)
    return outs[0] if single else tuple(outs)


def _empty(like: Tensor) -> Tensor:
    return like.new_empty(0)


def _opt(t: Tensor | None, like: Tensor) -> Tensor:
    return _empty(like) if t is None else t


def _none(t: Tensor) -> Tensor | None:
    return None if t.numel() == 0 else t


# ---------------------------------------------------------------- lora family
@torch.library.custom_op(f"{NS}::lora_merge_fwd", mutates_args=(), device_types="cuda")
def lora_merge_fwd(
    a1: Tensor,
    b1: Tensor,
    a2: Tensor | None,
    b2: Tensor | None,
    base: Tensor | None,
    gamma: float,
    mode: str,
    backend: str,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """``out_dtype`` (default a1's) is the dtype the fp32 accumulators are
    stored in; the same for every ``out_dtype`` below."""
    ins = (a1, b1, a2, b2, base)
    args = (a1, b1, a2, b2, base, gamma, mode)
    return _run(backend, "lora_merge_fwd", ins, *args, out_dtype=out_dtype)


@lora_merge_fwd.register_fake
def _(a1, b1, a2, b2, base, gamma, mode, backend, out_dtype=None):
    return a1.new_empty(a1.shape[0], b1.shape[1], dtype=out_dtype or a1.dtype)


@torch.library.custom_op(f"{NS}::lora_merge_bwd", mutates_args=(), device_types="cuda")
def lora_merge_bwd(
    grad: Tensor,
    up: Tensor,
    down: Tensor,
    gamma: float,
    backend: str,
) -> tuple[Tensor, Tensor]:
    ins = (grad, up, down)
    return _run(backend, "lora_merge_bwd", ins, grad, up, down, gamma=gamma)


@lora_merge_bwd.register_fake
def _(grad, up, down, gamma, backend):
    return up.new_empty(up.shape), down.new_empty(down.shape)


@torch.library.custom_op(f"{NS}::lora_bypass_fwd", mutates_args=(), device_types="cuda")
def lora_bypass_fwd(
    x: Tensor,
    up: Tensor,
    down: Tensor,
    gamma: float,
    backend: str,
    base: Tensor | None = None,
) -> Tensor:
    """[base +] the bypass delta; the result in base's dtype when given."""
    ins = (x, up, down, base)
    kw = {"gamma": gamma} if base is None else {"gamma": gamma, "base": base}
    return _run(backend, "lora_bypass_fwd", ins, x, up, down, **kw)


@lora_bypass_fwd.register_fake
def _(x, up, down, gamma, backend, base=None):
    dtype = x.dtype if base is None else base.dtype
    return x.new_empty(x.shape[0], up.shape[0], dtype=dtype)


@torch.library.custom_op(f"{NS}::lora_bypass_bwd", mutates_args=(), device_types="cuda")
def lora_bypass_bwd(
    x: Tensor, up: Tensor, down: Tensor, grad: Tensor, gamma: float, backend: str
) -> tuple[Tensor, Tensor, Tensor]:
    ins = (x, up, down, grad)
    return _run(backend, "lora_bypass_bwd", ins, x, up, down, grad, gamma=gamma)


@lora_bypass_bwd.register_fake
def _(x, up, down, grad, gamma, backend):
    return (
        x.new_empty(x.shape),
        up.new_empty(up.shape),
        up.new_empty(down.shape),
    )


@torch.library.custom_op(f"{NS}::lora_tucker_fwd", mutates_args=(), device_types="cuda")
def lora_tucker_fwd(
    a1: Tensor,
    t1: Tensor,
    b1: Tensor,
    a2: Tensor | None,
    t2: Tensor | None,
    b2: Tensor | None,
    gamma: float,
    backend: str,
) -> Tensor:
    ins = (a1, t1, b1, a2, t2, b2)
    return _run(backend, "lora_tucker_fwd", ins, a1, t1, b1, a2, t2, b2, gamma=gamma)


@lora_tucker_fwd.register_fake
def _(a1, t1, b1, a2, t2, b2, gamma, backend):
    return a1.new_empty(a1.shape[0], b1.shape[1], t1.shape[2])


# ---------------------------------------------------------------- loha family
@torch.library.custom_op(f"{NS}::loha_merge_bwd", mutates_args=(), device_types="cuda")
def loha_merge_bwd(
    grad: Tensor,
    a1: Tensor,
    b1: Tensor,
    a2: Tensor,
    b2: Tensor,
    gamma: float,
    backend: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ins = (grad, a1, b1, a2, b2)
    return _run(backend, "loha_merge_bwd", ins, grad, a1, b1, a2, b2, gamma=gamma)


@loha_merge_bwd.register_fake
def _(grad, a1, b1, a2, b2, gamma, backend):
    return tuple(t.new_empty(t.shape) for t in (a1, b1, a2, b2))


@torch.library.custom_op(f"{NS}::loha_bypass_fwd", mutates_args=(), device_types="cuda")
def loha_bypass_fwd(
    x: Tensor,
    a1: Tensor,
    b1: Tensor,
    a2: Tensor,
    b2: Tensor,
    gamma: float,
    backend: str,
    base: Tensor | None = None,
) -> Tensor:
    """[base +] the bypass delta; the result in base's dtype when given."""
    ins = (x, a1, b1, a2, b2, base)
    kw = {"gamma": gamma} if base is None else {"gamma": gamma, "base": base}
    return _run(backend, "loha_bypass_fwd", ins, x, a1, b1, a2, b2, **kw)


@loha_bypass_fwd.register_fake
def _(x, a1, b1, a2, b2, gamma, backend, base=None):
    dtype = x.dtype if base is None else base.dtype
    return x.new_empty(x.shape[0], a1.shape[0], dtype=dtype)


@torch.library.custom_op(f"{NS}::loha_bypass_bwd", mutates_args=(), device_types="cuda")
def loha_bypass_bwd(
    grad: Tensor,
    x: Tensor,
    a1: Tensor,
    b1: Tensor,
    a2: Tensor,
    b2: Tensor,
    gamma: float,
    backend: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    ins = (grad, x, a1, b1, a2, b2)
    return _run(backend, "loha_bypass_bwd", ins, grad, x, a1, b1, a2, b2, gamma=gamma)


@loha_bypass_bwd.register_fake
def _(grad, x, a1, b1, a2, b2, gamma, backend):
    return (x.new_empty(x.shape), *(a1.new_empty(t.shape) for t in (a1, b1, a2, b2)))


# ---------------------------------------------------------------- lokr family
@torch.library.custom_op(
    f"{NS}::lokr_full_merge_fwd", mutates_args=(), device_types="cuda"
)
def lokr_full_merge_fwd(
    w1: Tensor,
    w2: Tensor,
    base: Tensor | None,
    gamma: float,
    backend: str,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    ins = (w1, w2, base)
    kw = dict(gamma=gamma, out_dtype=out_dtype)
    return _run(backend, "lokr_full_merge_fwd", ins, w1, w2, base, **kw)


@lokr_full_merge_fwd.register_fake
def _(w1, w2, base, gamma, backend, out_dtype=None):
    shape = (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])
    return w2.new_empty(shape, dtype=out_dtype or w2.dtype)


@torch.library.custom_op(
    f"{NS}::lokr_full_merge_bwd", mutates_args=(), device_types="cuda"
)
def lokr_full_merge_bwd(
    grad: Tensor,
    w1: Tensor,
    w2: Tensor,
    gamma: float,
    backend: str,
) -> tuple[Tensor, Tensor]:
    ins = (grad, w1, w2)
    return _run(backend, "lokr_full_merge_bwd", ins, grad, w1, w2, gamma=gamma)


@lokr_full_merge_bwd.register_fake
def _(grad, w1, w2, gamma, backend):
    return w1.new_empty(w1.shape), w1.new_empty(w2.shape)


@torch.library.custom_op(f"{NS}::lokr_merge_fwd", mutates_args=(), device_types="cuda")
def lokr_merge_fwd(
    w1a: Tensor,
    w1b: Tensor | None,
    w2a: Tensor,
    w2b: Tensor | None,
    shape: list[int],
    base: Tensor | None,
    gamma: float,
    backend: str,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    ins = (w1a, w1b, w2a, w2b, base)
    args = (w1a, w1b, w2a, w2b, tuple(shape), base, gamma)
    return _run(backend, "lokr_merge_fwd", ins, *args, out_dtype=out_dtype)


@lokr_merge_fwd.register_fake
def _(w1a, w1b, w2a, w2b, shape, base, gamma, backend, out_dtype=None):
    a, b, c, d = shape
    return w2a.new_empty(a * c, b * d, dtype=out_dtype or w2a.dtype)


@torch.library.custom_op(f"{NS}::lokr_merge_bwd", mutates_args=(), device_types="cuda")
def _lokr_merge_bwd(
    grad: Tensor,
    w1a: Tensor,
    w1b: Tensor | None,
    w2a: Tensor,
    w2b: Tensor | None,
    shape: list[int],
    gamma: float,
    backend: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ins = (grad, w1a, w1b, w2a, w2b)
    with torch.cuda.device(grad.device):
        out = ops_for(backend, grad.device).lokr_merge_bwd(
            grad, w1a, w1b, w2a, w2b, tuple(shape), gamma=gamma
        )
    g1a, g1b, g2a, g2b = out
    filled = [g1a, _opt(g1b, g1a), g2a, _opt(g2b, g2a)]
    return tuple(_fresh(filled, ins))


@_lokr_merge_bwd.register_fake
def _(grad, w1a, w1b, w2a, w2b, shape, gamma, backend):
    g1b = w1a.new_empty(w1b.shape) if w1b is not None else w1a.new_empty(0)
    g2b = w1a.new_empty(w2b.shape) if w2b is not None else w1a.new_empty(0)
    return (w1a.new_empty(w1a.shape), g1b, w1a.new_empty(w2a.shape), g2b)


def lokr_merge_bwd(grad, w1a, w1b, w2a, w2b, shape, gamma, backend):
    """(g_w1a, g_w1b | None, g_w2a, g_w2b | None)."""
    g1a, g1b, g2a, g2b = _lokr_merge_bwd(
        grad, w1a, w1b, w2a, w2b, list(shape), gamma, backend
    )
    return (
        g1a,
        (g1b if w1b is not None else None),
        g2a,
        (g2b if w2b is not None else None),
    )


@torch.library.custom_op(f"{NS}::lokr_bypass_fwd", mutates_args=(), device_types="cuda")
def lokr_bypass_fwd(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    gamma: float,
    backend: str,
    base: Tensor | None = None,
) -> Tensor:
    """[base +] the bypass delta; the result in base's dtype when given."""
    kw = {"gamma": gamma} if base is None else {"gamma": gamma, "base": base}
    return _run(backend, "lokr_bypass_fwd", (x, w1, w2, base), x, w1, w2, **kw)


@lokr_bypass_fwd.register_fake
def _(x, w1, w2, gamma, backend, base=None):
    dtype = x.dtype if base is None else base.dtype
    return x.new_empty(x.shape[0], w1.shape[0] * w2.shape[0], dtype=dtype)


@torch.library.custom_op(f"{NS}::lokr_bypass_bwd", mutates_args=(), device_types="cuda")
def lokr_bypass_bwd(
    grad: Tensor, x: Tensor, w1: Tensor, w2: Tensor, gamma: float, backend: str
) -> tuple[Tensor, Tensor, Tensor]:
    ins = (grad, x, w1, w2)
    return _run(backend, "lokr_bypass_bwd", ins, grad, x, w1, w2, gamma=gamma)


@lokr_bypass_bwd.register_fake
def _(grad, x, w1, w2, gamma, backend):
    return x.new_empty(x.shape), w1.new_empty(w1.shape), w1.new_empty(w2.shape)
