import unittest
import re

from itertools import product
from parameterized import parameterized

import torch
import torch.nn as nn

from lycoris import create_lycoris, create_lycoris_from_weights, LycorisNetwork


class TestNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv3d = nn.Conv3d(dim, dim, (3, 3, 3), 1, 1)
        self.conv2d = nn.Conv2d(dim, dim, (3, 3), 1, 1)
        self.group_norm = nn.GroupNorm(4, dim)
        self.conv1d = nn.Conv1d(dim, dim, 3, 1, 1)
        self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.conv3d(x)
        h = h.flatten(2, 3)
        h = self.group_norm(h)
        h = self.conv2d(h)
        h = h.flatten(2, 3)
        h = self.conv1d(h)
        h = h.transpose(-1, -2)
        h = self.layer_norm(h)
        h = self.linear(h)
        return h


algos: list[str] = [
    "lora",
    "loha",
    "lokr",
    "full",
    "diag-oft",
    "boft",
    "glora",
    # "ia3",
]
device_and_dtype = [
    (torch.device("cpu"), torch.float32),
]
weight_decompose = [
    False,
    True,
]
use_tucker = [
    False,
    True,
]
use_scalar = [
    False,
    True,
]

if torch.cuda.is_available():
    device_and_dtype.append((torch.device("cuda"), torch.float32))
    device_and_dtype.append((torch.device("cuda"), torch.float16))
    device_and_dtype.append((torch.device("cuda"), torch.bfloat16))

if torch.backends.mps.is_available():
    device_and_dtype.append((torch.device("mps"), torch.float32))


patch_forward_param_list = list(
    product(
        algos,
        device_and_dtype,
        weight_decompose,
        use_tucker,
        use_scalar,
    )
)


class FunLinearNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.act(x)
        return x


class FunConvNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class NamedSequential(nn.Module):
    def __init__(self, name, *layers):
        super().__init__()
        self.name = name
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestNetwork2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_layers = NamedSequential(
            "linear_layers",
            FunLinearNetwork(dim),
            FunLinearNetwork(dim),
            FunLinearNetwork(dim),
        )
        self.conv_layers = NamedSequential(
            "conv_layers",
            FunConvNetwork(dim),
            FunConvNetwork(dim),
            FunConvNetwork(dim),
        )

    def forward(self, x):
        x = self.linear_layers(x)
        x = self.conv_layers(x)
        return x

    def named_modules(self, memo=None, prefix=""):
        # Call the parent class named_modules method
        for layer_name, layer in super().named_modules(memo, prefix):
            yield layer_name, layer


class LycorisWrapperTests(unittest.TestCase):
    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_wrapper(self, algo, device_dtype, wd, tucker, scalar):
        device, dtype = device_dtype
        print(
            f"{algo: <18}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            f"wd={str(wd): <6}",
            f"tucker={str(tucker): <6}",
            f"scalar={str(scalar): <6}",
            sep="|| ",
        )
        test_net = TestNetwork(16).to(device, dtype)
        test_lycoris: LycorisNetwork = create_lycoris(
            test_net,
            1,
            algo=algo,
            linear_dim=4,
            linear_alpha=2.0,
            conv_dim=4,
            conv_alpha=2.0,
            dropout=0.0,
            rank_dropout=0.0,
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
            train_norm=True,
        )
        test_lycoris.apply_to()
        test_lycoris.to(device, dtype)

        test_input = torch.randn(1, 16, 8, 8, 8).to(device, dtype)
        test_output = test_net(test_input)
        test_lycoris.restore()

        state_dict = test_lycoris.state_dict()
        test_lycoris_from_weights: LycorisNetwork
        test_lycoris_from_weights, _ = create_lycoris_from_weights(
            1, None, test_net, state_dict
        )
        test_lycoris_from_weights.apply_to()
        test_lycoris_from_weights.to(device, dtype)
        test_output_from_weights = test_net(test_input)

        test_lycoris_from_weights.load_state_dict(test_lycoris.state_dict())

        self.assertTrue(len(test_lycoris.loras) == len(test_lycoris_from_weights.loras))
        self.assertTrue(torch.allclose(test_output, test_output_from_weights))

    def test_lycoris_wrapper_regex_named_modules(
        self,
        device_dtype=(torch.device("cpu"), torch.float32),
    ):
        device, dtype = device_dtype

        # Define the preset configuration with regex
        preset = {
            "name_algo_map": {
                "linear_layers.layers.[0-1]..*": {
                    "algo": "lokr",
                    "factor": 4,
                    "linear_dim": 1000000,
                    "linear_alpha": 1,
                    "full_matrix": True,
                },
                "linear_layers.layers.2..*": {
                    "algo": "lora",
                    "dim": 8,
                },
                "conv_layers.layers.0..*": {
                    "algo": "locon",
                    "dim": 8,
                },
                "conv_layers.layers.[1-2]..*": {
                    "algo": "glora",
                    "dim": 128,
                },
            }
        }

        # Apply the preset to the LycorisNetwork class
        LycorisNetwork.apply_preset(preset)

        # Create the test network and the Lycoris network wrapper
        test_net = TestNetwork2(16).to(device, dtype)
        test_lycoris: LycorisNetwork = create_lycoris(
            test_net,
            multiplier=1,
            linear_dim=16,
            linear_alpha=1,
            train_norm=True,
        )
        test_lycoris.apply_to()
        test_lycoris.to(device, dtype)

        # Assertions to verify the correct modules were created and configured
        assert len(test_lycoris.loras) == 12

        lora_names = sorted([lora.lora_name for lora in test_lycoris.loras])
        expected = sorted(
            [
                "lycoris_linear_layers_layers_0_layer_norm",
                "lycoris_linear_layers_layers_0_linear",
                "lycoris_linear_layers_layers_1_layer_norm",
                "lycoris_linear_layers_layers_1_linear",
                "lycoris_linear_layers_layers_2_layer_norm",
                "lycoris_linear_layers_layers_2_linear",
                "lycoris_conv_layers_layers_0_layer_norm",
                "lycoris_conv_layers_layers_0_conv",
                "lycoris_conv_layers_layers_1_layer_norm",
                "lycoris_conv_layers_layers_1_conv",
                "lycoris_conv_layers_layers_2_layer_norm",
                "lycoris_conv_layers_layers_2_conv",
            ]
        )
        self.assertEqual(lora_names, expected)

        for lora in test_lycoris.loras:
            if (
                "lycoris_linear_layers_layers_0" in lora.lora_name
                or "lycoris_linear_layers_layers_1" in lora.lora_name
            ):
                self.assertEqual(lora.dim, 16)
                if "norm" in lora.lora_name:
                    self.assertEqual(lora.__class__.__name__, "NormModule")
                else:
                    self.assertEqual(lora.__class__.__name__, "LokrModule")
                    self.assertEqual(lora.multiplier, 1)
                    self.assertEqual(lora.full_matrix, True)
            elif "lycoris_linear_layers_layers_2" in lora.lora_name:
                self.assertEqual(lora.dim, 16)
                if "norm" in lora.lora_name:
                    self.assertEqual(lora.__class__.__name__, "NormModule")
                else:
                    self.assertEqual(lora.__class__.__name__, "LoConModule")
            elif "lycoris_conv_layers_layers_0" in lora.lora_name:
                self.assertEqual(lora.dim, 16)
                if "norm" in lora.lora_name:
                    self.assertEqual(lora.__class__.__name__, "NormModule")
                else:
                    self.assertEqual(lora.__class__.__name__, "LoConModule")
            elif (
                "lycoris_linear_layers_layers_1" in lora.lora_name
                or "lycoris_linear_layers_layers_2" in lora.lora_name
            ):
                self.assertEqual(lora.dim, 128)
                if "norm" in lora.lora_name:
                    self.assertEqual(lora.__class__.__name__, "NormModule")
                else:
                    self.assertEqual(lora.__class__.__name__, "GLoRAModule")
