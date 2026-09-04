"""Regression for #288: `exclude_name` must apply to lycoris.kohya too.

The wrapper honoured the preset key and the kohya network silently dropped it,
so a preset that targets whole transformer blocks had no way to keep the
modulation/norm layers out. Both walks also have to match the pattern against
the model's own module path, not the name relative to the matched block.
"""

import torch.nn as nn

from lycoris import LycorisNetwork, create_lycoris
from lycoris.kohya import LycorisNetworkKohya


class Block(nn.Module):
    """An Anima-shaped block: attention and MLP beside adaLN and norm."""

    def __init__(self, dim):
        super().__init__()
        self.attn_qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.mlp_fc1 = nn.Linear(dim, dim * 2)
        self.mlp_fc2 = nn.Linear(dim * 2, dim)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.norm = nn.Linear(dim, dim)


class Model(nn.Module):
    def __init__(self, dim=16, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])


EXCLUDE = ["*adaln_modulation*", "*norm*"]
TRAINABLE_PER_BLOCK = 4  # attn_qkv, attn_out, mlp_fc1, mlp_fc2
BLOCKS = 2


def kohya_preset(exclude):
    return {
        "enable_conv": False,
        "unet_target_module": ["Block"],
        "unet_target_name": [],
        "text_encoder_target_module": [],
        "text_encoder_target_name": [],
        "use_fnmatch": True,
        "exclude_name": exclude,
    }


def test_kohya_preset_applies_exclude_name():
    LycorisNetworkKohya.apply_preset(kohya_preset(EXCLUDE))
    assert LycorisNetworkKohya.TARGET_EXCLUDE_NAME == EXCLUDE

    network = LycorisNetworkKohya(
        None,
        Model(),
        1.0,
        lora_dim=4,
        alpha=4,
        network_module="locon",
        warn_on_unmatched=False,
    )
    names = sorted(lora.lora_name for lora in network.unet_loras)
    assert len(names) == BLOCKS * TRAINABLE_PER_BLOCK, names
    assert not [n for n in names if "adaln" in n or "norm" in n], names


def test_kohya_preset_without_exclude_name_keeps_every_layer():
    LycorisNetworkKohya.apply_preset(kohya_preset([]))
    network = LycorisNetworkKohya(
        None,
        Model(),
        1.0,
        lora_dim=4,
        alpha=4,
        network_module="locon",
        warn_on_unmatched=False,
    )
    # The four trainable layers plus adaln_modulation.1 and norm.
    assert len(network.unet_loras) == BLOCKS * (TRAINABLE_PER_BLOCK + 2)


def test_wrapper_excludes_inside_a_matched_block():
    LycorisNetwork.apply_preset(
        {
            "enable_conv": False,
            "target_module": ["Block"],
            "target_name": [],
            "use_fnmatch": True,
            "exclude_name": EXCLUDE,
        }
    )
    network = create_lycoris(Model(), 1.0, linear_dim=4, linear_alpha=4, algo="lora")
    names = sorted(lora.lora_name for lora in network.loras)
    assert len(names) == BLOCKS * TRAINABLE_PER_BLOCK, names
    assert not [n for n in names if "adaln" in n or "norm" in n], names


if __name__ == "__main__":
    test_kohya_preset_applies_exclude_name()
    test_kohya_preset_without_exclude_name_keeps_every_layer()
    test_wrapper_excludes_inside_a_matched_block()
    print("ok")
