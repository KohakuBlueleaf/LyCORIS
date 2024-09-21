import unittest
from itertools import product
from parameterized import parameterized

import torch

from lycoris.kohya import (
    LycorisNetworkKohya,
    create_network,
    create_network_from_weights,
)
from lycoris.utils import merge, extract_diff

from library.model_util import (
    load_models_from_stable_diffusion_checkpoint as load_sd,
    load_file,
)
from library.sdxl_model_util import (
    load_models_from_sdxl_checkpoint as load_sdxl,
)


algos: list[str] = [
    "lora",
    "loha",
    "lokr",
    "full",
    "diag-oft",
    "boft",
    "glora",
    "ia3",
]
if torch.cuda.is_available():
    device_and_dtype = [
        (torch.device("cuda"), torch.float16),
    ]
else:
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

wrapper_param_list = list(
    product(
        algos,
        device_and_dtype,
        weight_decompose,
        use_tucker,
        use_scalar,
    )
)
extract_param_list = list(
    product(
        device_and_dtype,
    )
)

device, dtype = device_and_dtype[0]
sd_te1, sd_te2, vae, sdxl_unet, *_ = load_sdxl(
    None, "./models/kohaku-xl-beta7.safetensors", "cpu", dtype
)


class LycorisKohyaWrapperTests(unittest.TestCase):
    @parameterized.expand(wrapper_param_list)
    def test_wrapper(self, algo, device_dtype, wd, tucker, scalar):
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
        network = create_network(
            1,
            16,
            16,
            vae,
            [sd_te1, sd_te2],
            sdxl_unet,
            algo=algo,
            conv_dim=16,
            conv_alpha=16.0,
            dropout=0.0,
            rank_dropout=0.0,
            weight_decompose=wd,
            use_tucker=tucker,
            use_scalar=scalar,
            train_norm=True,
        )
        network.apply_to([sd_te1, sd_te2], sdxl_unet, True, True)
        network.restore()
        network = create_network_from_weights(
            1, "", vae, [sd_te1, sd_te2], sdxl_unet, weights_sd=network.state_dict()
        )
        network.merge_to()
        del network
        torch.cuda.empty_cache()

    @parameterized.expand(extract_param_list)
    def test_extract(self, device_dtype):
        device, dtype = device_dtype
        print(
            "Extract",
            f"device={str(device): <5}",
            f"dtype={str(dtype): <15}",
            sep="|| ",
        )
        extract_diff(
            [sd_te1, sd_te2],
            [sd_te1, sd_te2],
            sdxl_unet,
            sdxl_unet,
            mode="fixed",
            linear_mode_param=4,
            conv_mode_param=4,
            extract_device=device,
            use_bias=True,
            sparsity=0.98,
            small_conv=True,
        )
