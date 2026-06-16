import unittest
from itertools import product
from parameterized import parameterized

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def test_kohya_lokr_discovers_fp8_linear_and_handles_bf16_no_autocast(self):
        from lycoris.kohya import create_network

        class Fp8Linear(nn.Module):
            def __init__(self, in_features, out_features, compute_dtype):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.compute_dtype = compute_dtype
                self.register_buffer("weight", torch.ones(out_features, in_features))
                self.register_buffer("weight_scale", torch.full((out_features,), 0.01))
                self.bias = None

            def forward(self, x):
                scale = self.weight_scale.to(device=x.device, dtype=x.dtype)
                if scale.ndim == 1:
                    scale = scale.unsqueeze(1)
                weight = self.weight.to(device=x.device, dtype=x.dtype) * scale
                return F.linear(x, weight, self.bias)

        class Ideogram4TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaln_modulation = nn.Linear(4, 8, bias=True).to(torch.bfloat16)
                self.qkv = Fp8Linear(4, 4, compute_dtype=torch.bfloat16)

            def forward(self, x):
                return self.adaln_modulation(x), self.qkv(x)

        model = nn.Sequential(Ideogram4TransformerBlock())
        network = create_network(
            1.0,
            2,
            1.0,
            None,
            None,
            model,
            algo="lokr",
            preset="full-lin",
            conv_dim=0,
            dora_wd=True,
        )
        fp8_loras = [
            lora
            for lora in network.unet_loras
            if lora.org_module[0].__class__.__name__ == "Fp8Linear"
        ]
        self.assertGreaterEqual(len(network.unet_loras), 2)
        self.assertEqual(len(fp8_loras), 1)
        self.assertTrue(fp8_loras[0].bypass_mode)
        self.assertTrue(fp8_loras[0].is_quant)
        self.assertTrue(fp8_loras[0].wd)
        with self.assertRaisesRegex(RuntimeError, "weight-only FP8"):
            fp8_loras[0].merge_to()

        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        y_adaln, y_fp8 = model(x)
        self.assertEqual(y_adaln.dtype, torch.bfloat16)
        self.assertEqual(y_fp8.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(y_adaln.float()).all())
        self.assertTrue(torch.isfinite(y_fp8.float()).all())
