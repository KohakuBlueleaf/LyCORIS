from itertools import product

import torch
import torch.nn as nn

from lycoris.modules import (
    LycorisBaseModule,
    LoConModule,
    LohaModule,
    LokrModule,
    FullModule,
    DiagOFTModule,
    ButterflyOFTModule,
    GLoRAModule,
    DyLoraModule,
    IA3Module,
)

modules: list[LycorisBaseModule] = [
    LoConModule,
    LohaModule,
    LokrModule,
    FullModule,
    DiagOFTModule,
    ButterflyOFTModule,
    GLoRAModule,
    DyLoraModule,
    IA3Module,
]
base_module_and_input = [
    lambda dim: (nn.Linear(dim, dim), torch.randn(1, dim)),
    lambda dim: (nn.Conv1d(dim, dim, 3, 1, 1), torch.randn(1, dim, 7)),
    lambda dim: (nn.Conv2d(dim, dim, (3, 3), 1, 1), torch.randn(1, dim, 7, 7)),
    lambda dim: (nn.Conv3d(dim, dim, (3, 2, 1), 1, 1), torch.randn(1, dim, 7, 7, 7)),
]
device_and_dtype = [
    (torch.device("cpu"), torch.float32),
    (torch.device("cuda"), torch.float32),
    (torch.device("cuda"), torch.float16),
    (torch.device("cuda"), torch.bfloat16),
]
weight_decompose = [False, True]
use_tucker = [False, True]
use_scalar = [False, True]


for module, base, (device, dtype), weight_decompose, use_tucker, use_scalar in product(
    modules,
    base_module_and_input,
    device_and_dtype,
    weight_decompose,
    use_tucker,
    use_scalar,
):
    base, test_input = base(16)
    print(
        f"{module.__name__: <18}",
        f"{base.__class__.__name__: <7}",
        f"device={str(device): <5}",
        f"dtype={str(dtype): <15}",
        f"wd={str(weight_decompose): <6}",
        f"tucker={str(use_tucker): <6}",
        f"scalar={str(use_scalar): <6}",
        sep="|| ",
    )
    base = base.to(device, dtype)
    test_input = test_input.to(device, dtype)
    net: LycorisBaseModule = module(
        "test",
        base,
        multiplier=1,
        lora_dim=4,
        alpha=1,
        weight_decompose=weight_decompose,
        use_tucker=use_tucker,
        use_scalar=use_scalar,
    ).to(device, dtype)
    net.apply_to()

    with torch.autocast("cuda", dtype=dtype):
        test_output = net(test_input)
    torch.sum(test_output).backward()
    state_dict = net.state_dict()
    net.load_state_dict(state_dict)
