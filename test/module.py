import unittest
from itertools import product
from parameterized import parameterized

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
    lambda dim: (nn.Conv1d(dim, dim, 3, 1, 1), torch.randn(1, dim, 16)),
    lambda dim: (nn.Conv2d(dim, dim, (3, 3), 1, 1), torch.randn(1, dim, 16, 16)),
    lambda dim: (nn.Conv3d(dim, dim, (3, 3, 3), 1, 1), torch.randn(1, dim, 16, 16, 16)),
]
device_and_dtype = [
    (torch.device("cpu"), torch.float32),
]
weight_decompose = [False, True]
use_tucker = [False, True]
use_scalar = [False, True]

if torch.cuda.is_available():
    device_and_dtype.append((torch.device("cuda"), torch.float32))
    device_and_dtype.append((torch.device("cuda"), torch.float16))
    device_and_dtype.append((torch.device("cuda"), torch.bfloat16))

if torch.backends.mps.is_available():
    device_and_dtype.append((torch.device("mps"), torch.float32))


patch_forward_param_list = list(
    product(
        modules,
        base_module_and_input,
        device_and_dtype,
        weight_decompose,
        use_tucker,
        use_scalar,
    )
)


class LycorisModuleTests(unittest.TestCase):
    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_modules(self, module, base, device_dtype, wd, tucker, scalar):
        base, test_input = base(16)
        device, dtype = device_dtype
        print(
            f"{module.__name__: <18}",
            f"{base.__class__.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            f"wd={str(wd): <6}",
            f"tucker={str(tucker): <6}",
            f"scalar={str(scalar): <6}",
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
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
        ).to(device, dtype)
        net.apply_to()

        with torch.autocast("cuda", dtype=dtype):
            test_output = base(test_input)
        torch.sum(test_output).backward()
        net.apply_max_norm(1.0)
        state_dict = net.state_dict()
        net.load_state_dict(state_dict)
        net.restore()
        net.merge_to()

        # attr access test
        net.org_weight

    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_modules_bypass_mode(
        self, module, base, device_dtype, wd, tucker, scalar
    ):
        base, test_input = base(16)
        if module == FullModule:
            # Full module not support bypass forward
            return
        device, dtype = device_dtype
        print(
            f"{module.__name__: <18}",
            f"{base.__class__.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            f"wd={str(wd): <6}",
            f"tucker={str(tucker): <6}",
            f"scalar={str(scalar): <6}",
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
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
            bypass_mode=True,
        ).to(device, dtype)
        net.apply_to()

        with torch.autocast("cuda", dtype=dtype):
            test_output = base(test_input)
        torch.sum(test_output).backward()
        state_dict = net.state_dict()
        net.load_state_dict(state_dict)

    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_modules_parametrize(
        self, module, base, device_dtype, wd, tucker, scalar
    ):
        base, test_input = base(16)
        if module == FullModule:
            # Full module not support bypass forward
            return
        device, dtype = device_dtype
        print(
            f"{module.__name__: <18}",
            f"{base.__class__.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            f"wd={str(wd): <6}",
            f"tucker={str(tucker): <6}",
            f"scalar={str(scalar): <6}",
            sep="|| ",
        )
        base = base.to(device, dtype)
        test_input = test_input.to(device, dtype)
        net = module.parametrize(
            base,
            "weight",
            1,
            4,
            1,
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
        ).to(device, dtype)

        with torch.autocast("cuda", dtype=dtype):
            test_output = base(test_input)
        torch.sum(test_output).backward()
        state_dict = net.state_dict()
        net.load_state_dict(state_dict)
