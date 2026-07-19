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
    @staticmethod
    def make_fp8_model(*, include_linear=False, include_adaln=False, dim=4):
        class Fp8Linear(nn.Linear):
            def __init__(self, in_features, out_features):
                nn.Module.__init__(self)
                self.in_features = in_features
                self.out_features = out_features
                self.compute_dtype = torch.bfloat16
                self.register_buffer(
                    "weight",
                    torch.ones(out_features, in_features).to(torch.float8_e4m3fn),
                )
                self.register_buffer(
                    "weight_scale", torch.linspace(0.25, 1.0, out_features)
                )
                self.bias = None

            def forward(self, x):
                weight = self.weight.to(x.dtype) * self.weight_scale.to(
                    x.dtype
                ).unsqueeze(1)
                return F.linear(x, weight, self.bias)

        class Ideogram4TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                if include_linear:
                    self.linear = nn.Linear(dim, dim)
                if include_adaln:
                    self.adaln_modulation = nn.Linear(dim, dim * 2, bias=True).to(
                        torch.bfloat16
                    )
                self.fp8_linear = Fp8Linear(dim, dim)

            def forward(self, x):
                fp8_output = self.fp8_linear(x)
                if include_adaln:
                    return self.adaln_modulation(x), fp8_output
                return fp8_output

        return nn.Sequential(Ideogram4TransformerBlock())

    def fp8_and_rebuild_outputs(self, algo, *, weight_decompose=False):
        from lycoris.kohya import create_network
        from lycoris.modules.base import dequantize_weight_only_fp8

        dim = 24
        model = self.make_fp8_model(dim=dim)
        network = create_network(
            0.7,
            2,
            1.0,
            None,
            None,
            model,
            algo=algo,
            preset="full-lin",
            conv_dim=0,
            use_scalar=True,
            dora_wd=weight_decompose,
            warn_on_unmatched=False,
        )
        self.assertEqual(len(network.unet_loras), 1)
        fp8_lora = network.unet_loras[0]

        generator = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            for name, param in fp8_lora.named_parameters():
                if name == "dora_scale":
                    continue
                param.copy_(
                    torch.rand(
                        param.shape,
                        generator=generator,
                        device=param.device,
                        dtype=param.dtype,
                    )
                    * 0.4
                    - 0.2
                )

        fp8_base = fp8_lora.org_module[0]
        reference_base = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            reference_base.weight.copy_(dequantize_weight_only_fp8(fp8_base))
        reference_lora = type(fp8_lora)(
            "reference",
            reference_base,
            multiplier=0.7,
            lora_dim=2,
            alpha=1.0,
            use_scalar=True,
            weight_decompose=weight_decompose,
            bypass_mode=False,
        )
        reference_lora.load_state_dict(fp8_lora.state_dict(), strict=False)

        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        network.eval()
        reference_lora.eval()
        x = torch.linspace(-1, 1, 2 * 3 * dim).reshape(2, 3, dim)
        return model(x), reference_lora(x)

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

        model = self.make_fp8_model(include_adaln=True)
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
        regular_lora = next(
            lora
            for lora in network.unet_loras
            if lora.org_module[0].__class__.__name__ != "Fp8Linear"
        )
        with torch.no_grad():
            for name, param in regular_lora.named_parameters():
                if name != "dora_scale":
                    param.fill_(0.1)
        regular_weight = regular_lora.org_weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "weight-only FP8"):
            fp8_loras[0].merge_to()
        with self.assertRaisesRegex(RuntimeError, "weight-only FP8"):
            fp8_loras[0].onfly_merge()
        with self.assertRaisesRegex(RuntimeError, "weight-only FP8"):
            network.merge_to(
                None,
                model,
                {"lora_unet_dummy": torch.tensor(0)},
                None,
                None,
            )
        torch.testing.assert_close(regular_lora.org_weight, regular_weight)
        with self.assertRaisesRegex(RuntimeError, "weight-only FP8"):
            network.onfly_merge()
        self.assertFalse(
            any(hasattr(lora, "cached_org_weight") for lora in network.loras)
        )

        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
        y_adaln, y_fp8 = model(x)
        self.assertEqual(y_adaln.dtype, torch.bfloat16)
        self.assertEqual(y_fp8.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(y_adaln.float()).all())
        self.assertTrue(torch.isfinite(y_fp8.float()).all())

    def test_kohya_supported_bypass_algorithms_handle_fp8_linear(self):
        from lycoris.kohya import create_network

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for algo, device in product(
            ("lora", "locon", "loha", "glora"), devices
        ):
            with self.subTest(algo=algo, device=device):
                model = self.make_fp8_model().to(device)
                network = create_network(
                    1.0,
                    2,
                    1.0,
                    None,
                    None,
                    model,
                    algo=algo,
                    preset="full-lin",
                    conv_dim=0,
                    warn_on_unmatched=False,
                )
                self.assertEqual(len(network.unet_loras), 1)
                self.assertTrue(
                    network.unet_loras[0].org_module[0].__class__.__name__
                    == "Fp8Linear"
                )

                network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
                network.to(device)
                x = torch.randn(
                    2,
                    3,
                    4,
                    device=device,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                output = model(x)
                output.float().sum().backward()

                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output.float()).all())
                self.assertTrue(
                    any(
                        param.grad is not None
                        and torch.isfinite(param.grad.float()).all()
                        for param in network.parameters()
                        if param.requires_grad
                    )
                )

    def test_kohya_fp8_bypass_matches_rebuilt_weight(self):
        for algo in ("lora", "locon", "loha", "lokr", "glora"):
            with self.subTest(algo=algo):
                actual, expected = self.fp8_and_rebuild_outputs(algo)
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_kohya_lokr_fp8_weight_decompose_matches_rebuilt_weight(self):
        actual, expected = self.fp8_and_rebuild_outputs(
            "lokr", weight_decompose=True
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_kohya_unsupported_bypass_algorithms_skip_fp8_linear(self):
        from lycoris.kohya import create_network

        for algo in ("dylora", "full", "diag-oft", "boft", "tlora"):
            with self.subTest(algo=algo):
                model = self.make_fp8_model(include_linear=True)
                network = create_network(
                    1.0,
                    2,
                    1.0,
                    None,
                    None,
                    model,
                    algo=algo,
                    preset="full-lin",
                    conv_dim=0,
                    block_size=2,
                    warn_on_unmatched=False,
                )
                fp8_loras = [
                    lora
                    for lora in network.unet_loras
                    if lora.org_module[0].__class__.__name__ == "Fp8Linear"
                ]
                self.assertEqual(fp8_loras, [])
                self.assertEqual(len(network.unet_loras), 1)

    def test_kohya_fp8_non_lokr_weight_decompose_is_skipped(self):
        from lycoris.kohya import create_network

        for algo in ("lora", "locon", "loha", "glora"):
            with self.subTest(algo=algo):
                model = self.make_fp8_model(include_linear=True)
                network = create_network(
                    1.0,
                    2,
                    1.0,
                    None,
                    None,
                    model,
                    algo=algo,
                    preset="full-lin",
                    conv_dim=0,
                    dora_wd=True,
                    warn_on_unmatched=False,
                )
                self.assertFalse(
                    any(
                        lora.org_module[0].__class__.__name__ == "Fp8Linear"
                        for lora in network.unet_loras
                    )
                )
