import unittest
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
