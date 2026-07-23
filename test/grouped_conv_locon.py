"""Regression for #260: LoCon on grouped Conv2d must use in/groups weight layout."""

import torch
import torch.nn as nn

from lycoris.modules.locon import LoConModule


def test_grouped_conv2d_locon_forward_and_weight_shape():
    groups = 8
    channels = 32
    assert channels % groups == 0

    conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups, bias=False)
    module = LoConModule(
        "test.grouped",
        conv,
        multiplier=1.0,
        lora_dim=4,
        alpha=4,
        bypass_mode=False,
    )

    assert module.shape == (channels, channels // groups, 3, 3)
    assert module.lora_down.weight.shape == (4, channels // groups, 3, 3)
    assert module.kw_dict["groups"] == groups

    x = torch.randn(2, channels, 16, 16)
    y = module(x)
    assert y.shape == x.shape

    diff, _ = module.get_diff_weight(device=x.device)
    assert diff.shape == conv.weight.shape


def test_depthwise_conv2d_locon_forward():
    channels = 16
    conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
    module = LoConModule(
        "test.depthwise",
        conv,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        bypass_mode=True,  # should be forced off for groups != 1
    )
    assert module.bypass_mode is False
    assert module.shape == (channels, 1, 3, 3)

    x = torch.randn(1, channels, 8, 8)
    y = module(x)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_grouped_conv2d_locon_forward_and_weight_shape()
    test_depthwise_conv2d_locon_forward()
    print("ok")
