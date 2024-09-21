import unittest
import re

from itertools import product
from parameterized import parameterized

import torch
import torch.nn as nn

from diffusers import FluxTransformer2DModel

from lycoris import create_lycoris, create_lycoris_from_weights, LycorisNetwork


def reset_globals():
    LycorisNetwork.apply_preset(
        {
            "enable_conv": True,
            "target_module": [
                "Linear",
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "GroupNorm",
                "LayerNorm",
            ],
            "target_name": [],
            "lora_prefix": "lycoris",
            "module_algo_map": {},
            "name_algo_map": {},
            "use_fnmatch": False,
            "exclude_name": [],
        }
    )


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
        try:
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
            test_lycoris.merge_to()

            state_dict = test_lycoris.state_dict()
            test_lycoris_from_weights: LycorisNetwork
            test_lycoris_from_weights, _ = create_lycoris_from_weights(
                1, None, test_net, state_dict
            )
            test_lycoris_from_weights.apply_to()
            test_lycoris_from_weights.to(device, dtype)
            test_output_from_weights = test_net(test_input)

            test_lycoris_from_weights.load_state_dict(test_lycoris.state_dict())

            self.assertTrue(
                len(test_lycoris.loras) == len(test_lycoris_from_weights.loras),
                f"{len(test_lycoris.loras)} != {len(test_lycoris_from_weights.loras)}",
            )
            self.assertTrue(
                torch.allclose(test_output, test_output_from_weights),
                f"diff: {torch.nn.functional.mse_loss(test_output, test_output_from_weights).item()}",
            )
        finally:
            reset_globals()

    def test_lycoris_wrapper_regex_named_modules(
        self,
        device_dtype=(torch.device("cpu"), torch.float32),
    ):
        try:
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

        finally:
            reset_globals()

    def test_diffusers_models_and_state_dicts_target_module_and_module_algo_map(self):
        try:
            transformer = FluxTransformer2DModel.from_config(
                {
                    "attention_head_dim": 128,
                    "guidance_embeds": True,
                    "in_channels": 64,
                    "joint_attention_dim": 4096,
                    "num_attention_heads": 24,
                    "num_layers": 4,
                    "num_single_layers": 4,
                    "patch_size": 1,
                    "pooled_projection_dim": 768,
                }
            )

            # Apply the preset to the LycorisNetwork class
            LycorisNetwork.apply_preset(
                {
                    "target_module": [
                        "FluxTransformerBlock",
                        "FluxSingleTransformerBlock",
                    ],
                    "module_algo_map": {
                        "Attention": {"factor": 6},
                        "FeedForward": {"factor": 4},
                    },
                }
            )

            # Create the test network and the Lycoris network wrapper
            test_lycoris: LycorisNetwork = create_lycoris(
                transformer,
                algo="lokr",
                multiplier=1.0,
                linear_dim=10000,
                linear_alpha=1,
                factor=8,
                full_matrix=True,
            )
            test_lycoris.apply_to()

            def check_dims(lora, factor):
                lokr_w2_shape = lora.state_dict()["lokr_w2"].shape
                return lora.shape[0] // lokr_w2_shape[0] == factor

            first_block = list(
                filter(
                    lambda lor: "lycoris_transformer_blocks_0_" in lor.lora_name,
                    test_lycoris.loras,
                )
            )
            assert sorted([lor.lora_name for lor in first_block]) == sorted(
                [
                    "lycoris_transformer_blocks_0_norm1_linear",
                    "lycoris_transformer_blocks_0_norm1_context_linear",
                    "lycoris_transformer_blocks_0_attn_to_q",
                    "lycoris_transformer_blocks_0_attn_to_k",
                    "lycoris_transformer_blocks_0_attn_to_v",
                    "lycoris_transformer_blocks_0_attn_add_k_proj",
                    "lycoris_transformer_blocks_0_attn_add_v_proj",
                    "lycoris_transformer_blocks_0_attn_add_q_proj",
                    "lycoris_transformer_blocks_0_attn_to_out_0",
                    "lycoris_transformer_blocks_0_attn_to_add_out",
                    "lycoris_transformer_blocks_0_ff_net_0_proj",
                    "lycoris_transformer_blocks_0_ff_net_2",
                    "lycoris_transformer_blocks_0_ff_context_net_0_proj",
                    "lycoris_transformer_blocks_0_ff_context_net_2",
                ]
            )
            first_single_block = list(
                filter(
                    lambda lor: "lycoris_single_transformer_blocks_0_" in lor.lora_name,
                    test_lycoris.loras,
                )
            )
            assert sorted([lor.lora_name for lor in first_single_block]) == sorted(
                [
                    "lycoris_single_transformer_blocks_0_norm_linear",
                    "lycoris_single_transformer_blocks_0_proj_mlp",
                    "lycoris_single_transformer_blocks_0_proj_out",
                    "lycoris_single_transformer_blocks_0_attn_to_q",
                    "lycoris_single_transformer_blocks_0_attn_to_k",
                    "lycoris_single_transformer_blocks_0_attn_to_v",
                ]
            )

            for lora in test_lycoris.loras:
                if "_norm1_" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 8))
                elif "_norm_" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 8))
                elif "_proj_" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 8))
                elif "_attn_" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 6))
                elif "_ff_" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 4))
                else:
                    raise Exception(f"Unknown layer {lora.lora_name}")

            test_lycoris_from_weights: LycorisNetwork
            test_lycoris_from_weights, _ = create_lycoris_from_weights(
                1, None, transformer, test_lycoris.state_dict()
            )
            test_lycoris_from_weights.apply_to()
            test_lycoris_from_weights.load_state_dict(test_lycoris.state_dict())

            self.assertTrue(
                len(test_lycoris.loras) == len(test_lycoris_from_weights.loras)
            )
            for lora, lora_loaded in zip(
                test_lycoris.loras, test_lycoris_from_weights.loras
            ):
                lora_sd = lora.state_dict()
                lora_loaded_sd = lora_loaded.state_dict()
                assert torch.equal(lora_sd["alpha"], lora_loaded_sd["alpha"])
                assert torch.equal(lora_sd["lokr_w1"], lora_loaded_sd["lokr_w1"])
                assert torch.equal(lora_sd["lokr_w2"], lora_loaded_sd["lokr_w2"])

        finally:
            reset_globals()

    def test_diffusers_models_and_state_dicts_whole_model(self):
        try:
            transformer = FluxTransformer2DModel.from_config(
                {
                    "attention_head_dim": 128,
                    "guidance_embeds": True,
                    "in_channels": 64,
                    "joint_attention_dim": 4096,
                    "num_attention_heads": 24,
                    "num_layers": 1,
                    "num_single_layers": 1,
                    "patch_size": 1,
                    "pooled_projection_dim": 768,
                }
            )

            # Apply the preset to the LycorisNetwork class
            LycorisNetwork.apply_preset(
                {"module_algo_map": {"FluxTransformer2DModel": {"factor": 16}}}
            )

            # Create the test network and the Lycoris network wrapper
            test_lycoris: LycorisNetwork = create_lycoris(
                transformer,
                algo="lokr",
                multiplier=1.0,
                linear_dim=10000,
                linear_alpha=1,
                factor=16,
            )
            test_lycoris.apply_to()

            state_dict = test_lycoris.state_dict()
            assert sorted(
                [
                    "lycoris_time_text_embed_timestep_embedder_linear_1.alpha",
                    "lycoris_time_text_embed_timestep_embedder_linear_1.lokr_w1",
                    "lycoris_time_text_embed_timestep_embedder_linear_1.lokr_w2",
                    "lycoris_time_text_embed_timestep_embedder_linear_2.alpha",
                    "lycoris_time_text_embed_timestep_embedder_linear_2.lokr_w1",
                    "lycoris_time_text_embed_timestep_embedder_linear_2.lokr_w2",
                    "lycoris_time_text_embed_guidance_embedder_linear_1.alpha",
                    "lycoris_time_text_embed_guidance_embedder_linear_1.lokr_w1",
                    "lycoris_time_text_embed_guidance_embedder_linear_1.lokr_w2",
                    "lycoris_time_text_embed_guidance_embedder_linear_2.alpha",
                    "lycoris_time_text_embed_guidance_embedder_linear_2.lokr_w1",
                    "lycoris_time_text_embed_guidance_embedder_linear_2.lokr_w2",
                    "lycoris_time_text_embed_text_embedder_linear_1.alpha",
                    "lycoris_time_text_embed_text_embedder_linear_1.lokr_w1",
                    "lycoris_time_text_embed_text_embedder_linear_1.lokr_w2",
                    "lycoris_time_text_embed_text_embedder_linear_2.alpha",
                    "lycoris_time_text_embed_text_embedder_linear_2.lokr_w1",
                    "lycoris_time_text_embed_text_embedder_linear_2.lokr_w2",
                    "lycoris_context_embedder.alpha",
                    "lycoris_context_embedder.lokr_w1",
                    "lycoris_context_embedder.lokr_w2",
                    "lycoris_x_embedder.alpha",
                    "lycoris_x_embedder.lokr_w1",
                    "lycoris_x_embedder.lokr_w2",
                    "lycoris_transformer_blocks_0_norm1_linear.alpha",
                    "lycoris_transformer_blocks_0_norm1_linear.lokr_w1",
                    "lycoris_transformer_blocks_0_norm1_linear.lokr_w2",
                    "lycoris_transformer_blocks_0_norm1_context_linear.alpha",
                    "lycoris_transformer_blocks_0_norm1_context_linear.lokr_w1",
                    "lycoris_transformer_blocks_0_norm1_context_linear.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_to_q.alpha",
                    "lycoris_transformer_blocks_0_attn_to_q.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_to_q.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_to_k.alpha",
                    "lycoris_transformer_blocks_0_attn_to_k.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_to_k.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_to_v.alpha",
                    "lycoris_transformer_blocks_0_attn_to_v.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_to_v.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_add_k_proj.alpha",
                    "lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_add_k_proj.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_add_v_proj.alpha",
                    "lycoris_transformer_blocks_0_attn_add_v_proj.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_add_v_proj.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_add_q_proj.alpha",
                    "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_add_q_proj.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_to_out_0.alpha",
                    "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_to_out_0.lokr_w2",
                    "lycoris_transformer_blocks_0_attn_to_add_out.alpha",
                    "lycoris_transformer_blocks_0_attn_to_add_out.lokr_w1",
                    "lycoris_transformer_blocks_0_attn_to_add_out.lokr_w2",
                    "lycoris_transformer_blocks_0_ff_net_0_proj.alpha",
                    "lycoris_transformer_blocks_0_ff_net_0_proj.lokr_w1",
                    "lycoris_transformer_blocks_0_ff_net_0_proj.lokr_w2",
                    "lycoris_transformer_blocks_0_ff_net_2.alpha",
                    "lycoris_transformer_blocks_0_ff_net_2.lokr_w1",
                    "lycoris_transformer_blocks_0_ff_net_2.lokr_w2",
                    "lycoris_transformer_blocks_0_ff_context_net_0_proj.alpha",
                    "lycoris_transformer_blocks_0_ff_context_net_0_proj.lokr_w1",
                    "lycoris_transformer_blocks_0_ff_context_net_0_proj.lokr_w2",
                    "lycoris_transformer_blocks_0_ff_context_net_2.alpha",
                    "lycoris_transformer_blocks_0_ff_context_net_2.lokr_w1",
                    "lycoris_transformer_blocks_0_ff_context_net_2.lokr_w2",
                    "lycoris_single_transformer_blocks_0_norm_linear.alpha",
                    "lycoris_single_transformer_blocks_0_norm_linear.lokr_w1",
                    "lycoris_single_transformer_blocks_0_norm_linear.lokr_w2",
                    "lycoris_single_transformer_blocks_0_proj_mlp.alpha",
                    "lycoris_single_transformer_blocks_0_proj_mlp.lokr_w1",
                    "lycoris_single_transformer_blocks_0_proj_mlp.lokr_w2",
                    "lycoris_single_transformer_blocks_0_proj_out.alpha",
                    "lycoris_single_transformer_blocks_0_proj_out.lokr_w1",
                    "lycoris_single_transformer_blocks_0_proj_out.lokr_w2",
                    "lycoris_single_transformer_blocks_0_attn_to_q.alpha",
                    "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w1",
                    "lycoris_single_transformer_blocks_0_attn_to_q.lokr_w2",
                    "lycoris_single_transformer_blocks_0_attn_to_k.alpha",
                    "lycoris_single_transformer_blocks_0_attn_to_k.lokr_w1",
                    "lycoris_single_transformer_blocks_0_attn_to_k.lokr_w2",
                    "lycoris_single_transformer_blocks_0_attn_to_v.alpha",
                    "lycoris_single_transformer_blocks_0_attn_to_v.lokr_w1",
                    "lycoris_single_transformer_blocks_0_attn_to_v.lokr_w2",
                    "lycoris_norm_out_linear.alpha",
                    "lycoris_norm_out_linear.lokr_w1",
                    "lycoris_norm_out_linear.lokr_w2",
                    "lycoris_proj_out.alpha",
                    "lycoris_proj_out.lokr_w1",
                    "lycoris_proj_out.lokr_w2",
                ]
            ) == sorted([k for k in state_dict.keys()])

            state_dict = test_lycoris.state_dict()
            test_lycoris_from_weights: LycorisNetwork
            test_lycoris_from_weights, _ = create_lycoris_from_weights(
                1, None, transformer, state_dict
            )
            test_lycoris_from_weights.apply_to()
            test_lycoris_from_weights.load_state_dict(test_lycoris.state_dict())

            self.assertTrue(
                len(test_lycoris.loras) == len(test_lycoris_from_weights.loras)
            )
            for lora, lora_loaded in zip(
                test_lycoris.loras, test_lycoris_from_weights.loras
            ):
                lora_sd = lora.state_dict()
                lora_loaded_sd = lora_loaded.state_dict()
                assert torch.equal(lora_sd["alpha"], lora_loaded_sd["alpha"])
                assert torch.equal(lora_sd["lokr_w1"], lora_loaded_sd["lokr_w1"])
                assert torch.equal(lora_sd["lokr_w2"], lora_loaded_sd["lokr_w2"])
        finally:
            reset_globals()

    def test_diffusers_models_and_state_dicts_fnmatch(self):
        try:
            transformer = FluxTransformer2DModel.from_config(
                {
                    "attention_head_dim": 128,
                    "guidance_embeds": True,
                    "in_channels": 64,
                    "joint_attention_dim": 4096,
                    "num_attention_heads": 24,
                    "num_layers": 4,
                    "num_single_layers": 4,
                    "patch_size": 1,
                    "pooled_projection_dim": 768,
                }
            )

            # Apply the preset to the LycorisNetwork class
            LycorisNetwork.apply_preset(
                {
                    "target_module": [
                        "FluxTransformerBlock",
                        "FluxSingleTransformerBlock",
                    ],
                    "name_algo_map": {
                        "transformer_blocks.[2-3]*": {
                            "algo": "lokr",
                            "factor": 8,
                            "linear_alpha": 1,
                            "full_matrix": True,
                        },
                        "single_transformer_blocks.[1-3]*": {
                            "algo": "lokr",
                            "factor": 12,
                            "linear_alpha": 1,
                            "full_matrix": True,
                        },
                    },
                    "use_fnmatch": True,
                }
            )

            # Create the test network and the Lycoris network wrapper
            test_lycoris: LycorisNetwork = create_lycoris(
                transformer,
                algo="lokr",
                multiplier=1.0,
                linear_alpha=1,
                factor=16,
                full_matrix=True,
            )
            test_lycoris.apply_to()

            def check_dims(lora, factor):
                lokr_w2_shape = lora.state_dict()["lokr_w2"].shape
                return lora.shape[0] // lokr_w2_shape[0] == factor

            for lora in test_lycoris.loras:
                if (
                    "lycoris_transformer_blocks_0" in lora.lora_name
                    or "lycoris_transformer_blocks_1" in lora.lora_name
                ):
                    self.assertTrue(check_dims(lora, 16))
                elif (
                    "lycoris_transformer_blocks_2" in lora.lora_name
                    or "lycoris_transformer_blocks_3" in lora.lora_name
                ):
                    self.assertTrue(check_dims(lora, 8))
                elif "lycoris_single_transformer_blocks_0" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 16))
                elif (
                    "lycoris_single_transformer_blocks_1" in lora.lora_name
                    or "lycoris_single_transformer_blocks_2" in lora.lora_name
                    or "lycoris_single_transformer_blocks_3" in lora.lora_name
                ):
                    self.assertTrue(check_dims(lora, 12))

            test_lycoris_from_weights: LycorisNetwork
            test_lycoris_from_weights, _ = create_lycoris_from_weights(
                1, None, transformer, test_lycoris.state_dict()
            )
            test_lycoris_from_weights.apply_to()
            test_lycoris_from_weights.load_state_dict(test_lycoris.state_dict())

            self.assertTrue(
                len(test_lycoris.loras) == len(test_lycoris_from_weights.loras)
            )
            for lora, lora_loaded in zip(
                test_lycoris.loras, test_lycoris_from_weights.loras
            ):
                lora_sd = lora.state_dict()
                lora_loaded_sd = lora_loaded.state_dict()
                assert torch.equal(lora_sd["alpha"], lora_loaded_sd["alpha"])
                assert torch.equal(lora_sd["lokr_w1"], lora_loaded_sd["lokr_w1"])
                assert torch.equal(lora_sd["lokr_w2"], lora_loaded_sd["lokr_w2"])

        finally:
            reset_globals()

    def test_diffusers_models_and_state_dicts_fnmatch_and_exclude(self):
        try:
            transformer = FluxTransformer2DModel.from_config(
                {
                    "attention_head_dim": 128,
                    "guidance_embeds": True,
                    "in_channels": 64,
                    "joint_attention_dim": 4096,
                    "num_attention_heads": 24,
                    "num_layers": 4,
                    "num_single_layers": 4,
                    "patch_size": 1,
                    "pooled_projection_dim": 768,
                }
            )

            # Apply the preset to the LycorisNetwork class
            LycorisNetwork.apply_preset(
                {
                    "target_module": [
                        "FluxTransformerBlock",
                        "FluxSingleTransformerBlock",
                    ],
                    "name_algo_map": {
                        "transformer_blocks.[2-3]*": {
                            "algo": "lokr",
                            "factor": 8,
                            "linear_alpha": 1,
                            "full_matrix": True,
                        },
                        "single_transformer_blocks.[1-3]*": {
                            "algo": "lokr",
                            "factor": 12,
                            "linear_alpha": 1,
                            "full_matrix": True,
                        },
                    },
                    "use_fnmatch": True,
                    "exclude_name": [
                        "transformer_blocks.1*",
                        "single_transformer_blocks.[2-3]*",
                    ],
                }
            )

            # Create the test network and the Lycoris network wrapper
            test_lycoris: LycorisNetwork = create_lycoris(
                transformer,
                algo="lokr",
                multiplier=1.0,
                linear_alpha=1,
                factor=16,
                full_matrix=True,
            )
            test_lycoris.apply_to()

            def check_dims(lora, factor):
                lokr_w2_shape = lora.state_dict()["lokr_w2"].shape
                return lora.shape[0] // lokr_w2_shape[0] == factor

            for lora in test_lycoris.loras:
                assert "lycoris_transformer_blocks_1" not in lora.lora_name
                assert "lycoris_single_transformer_blocks_2" not in lora.lora_name
                assert "lycoris_single_transformer_blocks_3" not in lora.lora_name
                if "lycoris_transformer_blocks_0" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 16))
                elif (
                    "lycoris_transformer_blocks_2" in lora.lora_name
                    or "lycoris_transformer_blocks_3" in lora.lora_name
                ):
                    self.assertTrue(check_dims(lora, 8))
                elif "lycoris_single_transformer_blocks_0" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 16))
                elif "lycoris_single_transformer_blocks_1" in lora.lora_name:
                    self.assertTrue(check_dims(lora, 12))

            test_lycoris_from_weights: LycorisNetwork
            test_lycoris_from_weights, _ = create_lycoris_from_weights(
                1, None, transformer, test_lycoris.state_dict()
            )
            test_lycoris_from_weights.apply_to()
            test_lycoris_from_weights.load_state_dict(test_lycoris.state_dict())

            self.assertTrue(
                len(test_lycoris.loras) == len(test_lycoris_from_weights.loras)
            )
            for lora, lora_loaded in zip(
                test_lycoris.loras, test_lycoris_from_weights.loras
            ):
                lora_sd = lora.state_dict()
                lora_loaded_sd = lora_loaded.state_dict()
                assert torch.equal(lora_sd["alpha"], lora_loaded_sd["alpha"])
                assert torch.equal(lora_sd["lokr_w1"], lora_loaded_sd["lokr_w1"])
                assert torch.equal(lora_sd["lokr_w2"], lora_loaded_sd["lokr_w2"])

        finally:
            reset_globals()
