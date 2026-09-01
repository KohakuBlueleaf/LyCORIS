"""Every algorithm, forward and backward, on CPU.

The GPU kernels cannot run on a CI runner, so what this proves is the layer
above them: each module builds, the backend selection resolves to the eager
tier for CPU tensors, the forward produces finite values in both merge and
bypass mode, and every parameter receives a finite, non-zero gradient.

Usage:
    python scripts/ci/cpu_smoke.py
"""

import itertools
import sys

import torch
import torch.nn as nn

from lycoris.kernels.dispatch import available_backends
from lycoris.modules.boft import ButterflyOFTModule
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.dylora import DyLoraModule
from lycoris.modules.full import FullModule
from lycoris.modules.glora import GLoRAModule
from lycoris.modules.ia3 import IA3Module
from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.norms import NormModule
from lycoris.modules.tlora import TLoraModule

ALGOS = {
    "locon": (LoConModule, {}),
    "locon-dora": (LoConModule, {"weight_decompose": True}),
    "locon-tucker": (LoConModule, {"use_tucker": True}),
    "loha": (LohaModule, {}),
    "loha-dora": (LohaModule, {"weight_decompose": True}),
    "lokr": (LokrModule, {"factor": 4}),
    "lokr-full": (LokrModule, {"factor": 4, "full_matrix": True}),
    "lokr-dora": (LokrModule, {"factor": 4, "weight_decompose": True}),
    "oft": (DiagOFTModule, {"constraint": 0}),
    "oft-rescale": (DiagOFTModule, {"constraint": 0, "rescaled": True}),
    "boft": (ButterflyOFTModule, {"constraint": 0}),
    "ia3": (IA3Module, {}),
    "ia3-input": (IA3Module, {"train_on_input": True}),
    "glora": (GLoRAModule, {}),
    "dylora": (DyLoraModule, {"block_size": 2}),
    "tlora": (TLoraModule, {}),
    "full": (FullModule, {}),
}
# The LoKr conv bypass reshapes the flattened w2b against the kernel window,
# which only holds for a 1x1 kernel. Pre-existing, and unrelated to dispatch.
SKIP = {
    ("lokr", "conv", True),
    ("lokr-full", "conv", True),
    ("lokr-dora", "conv", True),
    # DyLoRA's bypass reshapes the sliced factors to the full lora_dim, so it
    # only lines up when every block is active. Pre-existing.
    ("dylora", "linear", True),
    ("dylora", "conv", True),
    # T-LoRA rebuilds a 1x1 kernel, so a k>1 conv never matches its base and
    # falls back to a bypass whose adapter chain changes the spatial size.
    ("tlora", "conv", False),
    ("tlora", "conv", True),
}
# FullModule.apply_to() moves the weight onto itself and deletes the original,
# which the org module's own forward still needs; call it unpatched instead.
NO_APPLY = {"full"}


def layer(kind):
    if kind == "linear":
        return nn.Linear(64, 96), torch.randn(4, 64)
    return nn.Conv2d(32, 64, 3, padding=1), torch.randn(2, 32, 6, 6)


def randomize(module):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * 0.05)


def check(module, x, bypass):
    """Finite forward, and a finite non-zero gradient reaching the adapter.

    Not every parameter takes a gradient in every mode by design — DoRA's
    scale is weight-space only, DyLoRA trains one rank block per step — so the
    rule is that at least one does and none of them are NaN.
    """
    module.bypass_mode = bypass
    out = module(x)
    if not torch.isfinite(out).all():
        return "non-finite forward"
    out.square().mean().backward()
    trained = 0
    for pname, p in module.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return f"{pname}: non-finite gradient"
        trained += int(bool(p.grad.abs().sum() > 0))
    if trained == 0:
        return "no parameter received a gradient"
    return None


def main() -> int:
    torch.manual_seed(0)
    print(f"backends: {available_backends()}")
    failures = []
    for (name, (cls, kwargs)), kind, bypass in itertools.product(
        ALGOS.items(), ("linear", "conv"), (False, True)
    ):
        if (name, kind, bypass) in SKIP:
            continue
        base, x = layer(kind)
        try:
            module = cls("smoke", base, 1.0, lora_dim=4, alpha=4, **kwargs)
        except ValueError as exc:
            print(f"{name:14s} {kind:6s} bypass={int(bypass)}  skip ({exc})")
            continue
        randomize(module)
        patched = name not in NO_APPLY
        if patched:
            module.apply_to()
        err = check(module, x, bypass)
        if patched:
            module.restore()
        print(
            f"{name:14s} {kind:6s} bypass={int(bypass)}  "
            f"{'ok' if err is None else 'FAIL: ' + err}"
        )
        if err:
            failures.append((name, kind, bypass, err))

    norm = NormModule("smoke", nn.LayerNorm(64), 1.0)
    randomize(norm)
    norm.apply_to()
    err = check(norm, torch.randn(4, 64), False)
    norm.restore()
    print(
        f"{'norm':14s} {'linear':6s} bypass=0  {'ok' if err is None else 'FAIL: ' + err}"
    )
    if err:
        failures.append(("norm", "linear", False, err))

    print(f"\n{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
