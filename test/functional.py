import unittest
from itertools import product
from parameterized import parameterized

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.functional import locon, loha, lokr, diag_oft, boft


EPS_DTYPE = {
    torch.float32: 5e-6,
    torch.float16: 5e-5,
    torch.bfloat16: 5e-4,
}


modules = [locon, loha, lokr, diag_oft, boft]
base_module_and_input_adn_weight = [
    lambda dim: (F.linear, torch.randn(dim, dim), torch.randn(1, dim)),
    lambda dim: (F.conv1d, torch.randn(dim, dim, 3), torch.randn(1, dim, 16)),
    lambda dim: (F.conv2d, torch.randn(dim, dim, 3, 3), torch.randn(1, dim, 16, 16)),
    lambda dim: (
        F.conv3d,
        torch.randn(dim, dim, 3, 3, 3),
        torch.randn(1, dim, 16, 16, 16),
    ),
]
device_and_dtype = [
    (torch.device("cpu"), torch.float32),
]

if torch.cuda.is_available():
    device_and_dtype.append((torch.device("cuda"), torch.float32))
    device_and_dtype.append((torch.device("cuda"), torch.float16))
    device_and_dtype.append((torch.device("cuda"), torch.bfloat16))

if torch.backends.mps.is_available():
    device_and_dtype.append((torch.device("mps"), torch.float32))


patch_forward_param_list = list(
    product(
        modules,
        base_module_and_input_adn_weight,
        device_and_dtype,
    )
)


class LycorisFunctionalTests(unittest.TestCase):
    @parameterized.expand(patch_forward_param_list)
    def test_lycoris_functional(self, module, base, device_dtype):
        func, test_weight, test_input = base(16)
        device, dtype = device_dtype
        print(
            f"{module.__name__: <27}",
            f"{func.__name__: <7}",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            sep="||",
        )

        w = test_weight.to(device, dtype)
        x = test_input.to(device, dtype)
        y = func(x, w)

        params = list(module.weight_gen(w, 4))
        for idx, param in enumerate(params):
            if param is not None:
                param = param.to(device, dtype)
                params[idx] = param + torch.randn_like(param) * 0.01

        if module in {boft, diag_oft}:
            diff_w = module.diff_weight(w, *params)
            diff_y = module.bypass_forward_diff(y, *params, need_transpose=w.ndim > 2)
        else:
            diff_w = module.diff_weight(*params)
            diff_y = module.bypass_forward_diff(x, *params)

        diff_y_from_diff_w = func(x, diff_w.to(x))
        self.assertTrue(
            F.mse_loss(diff_y, diff_y_from_diff_w).item() < EPS_DTYPE[dtype],
            f"Error: {module.__name__} {base.__name__} {device} {dtype} ||"
            f"{F.mse_loss(diff_y, diff_y_from_diff_w).item()}",
        )
