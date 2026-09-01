"""Per-call backend choice for the functional and module layers.

``dispatch.resolve_backend`` gives the process-wide preference
(triton > tilelang > compile > torch); this layer narrows it per call. A fused
op covers CUDA tensors of one floating family, and a call outside that scope —
mixed fp16/bf16, or a layout the fused path does not implement — steps one tier
down instead of failing, so the caller never carries a capability test. CPU
operands go straight to eager, since the compile tier's warmup only pays for
itself on device work.

torch.compile is applied to ONE op at a time and cached on the function object,
never to a module: each op then compiles once for the process and a model
mixing algos never recompiles on their union. A tier that imports but cannot
run is demoted on its first failure, which costs a fallback, not the run.
"""

import functools
import os

import torch
import torch._dynamo

from .dispatch import FUSED, available_backends, demote, resolve_backend

# Dynamo/inductor failures only: a shape or dtype error from the op itself
# must propagate, not silently disable the tier.
_COMPILE_FAILED = (
    torch._dynamo.exc.BackendCompilerFailed,
    torch._dynamo.exc.InternalTorchDynamoError,
    torch._dynamo.exc.Unsupported,
)


def on_device(tensors) -> bool:
    """Every operand a floating tensor on one and the same CUDA device."""
    devices = set()
    for t in tensors:
        if t is None:
            continue
        if not (torch.is_tensor(t) and t.is_cuda and t.is_floating_point()):
            return False
        devices.add(t.device)
    return len(devices) == 1


def one_family(tensors) -> bool:
    """fp16 and bf16 never mixed.

    The fused ops promote operands into a single compute dtype; fp16 with bf16
    has no common half, and either would have to widen to fp32 to meet.
    """
    dtypes = {t.dtype for t in tensors if t is not None}
    return not (torch.float16 in dtypes and torch.bfloat16 in dtypes)


def fusable(tensors) -> bool:
    """The operand set a fused kernel can take as it stands."""
    return on_device(tensors) and one_family(tensors)


def choose(tensors, supported: bool = True, backend: str | None = None) -> str:
    """Backend for THIS call: triton > tilelang > compile > torch.

    ``supported`` is the caller's own scope test — the layout this op has a
    fused kernel for. Failing it, or arriving with operands outside the fused
    scope, steps one tier down.

    The compile tier is device-only unless asked for by name: the CPU callers
    here are weight merges, where an inductor warmup costs more than the op it
    replaces.
    """
    name = resolve_backend(backend)
    if name == "torch":
        return name
    device = on_device(tensors)
    if name in FUSED and supported and device and one_family(tensors):
        return name
    asked = (backend or os.environ.get("LYCORIS_KERNEL_BACKEND", "auto")).lower()
    if "compile" in available_backends() and (device or asked == "compile"):
        return "compile"
    return "torch"


def static_scale(gamma) -> bool:
    """A scale the fused ops can take as a launch constant.

    They read gamma as a float, so a tensor carrying grad has to stay in the
    graph as an eager multiply.
    """
    return not (isinstance(gamma, torch.Tensor) and gamma.requires_grad)


@functools.cache
def compiled(fn):
    """One inductor graph per op, cached on the function object.

    ``dynamic=None`` lets the second distinct shape mark itself dynamic, so a
    sweep over per-layer shapes settles into one graph instead of exhausting
    the recompile limit.
    """
    return torch.compile(fn, dynamic=None)


def call_compiled(fn, *args, **kwargs):
    """Compiled call with the eager body as its fallback."""
    try:
        return compiled(fn)(*args, **kwargs)
    except _COMPILE_FAILED:
        demote("compile")
        return fn(*args, **kwargs)
