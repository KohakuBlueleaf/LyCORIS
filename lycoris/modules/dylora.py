import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModuleCustomSD


class DyLoraModule(ModuleCustomSD):
    """
    Hadamard product Implementaion for Dynamic Low Rank adaptation
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        block_size=1,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        assert lora_dim % block_size == 0, "lora_dim must be a multiple of block_size"
        self.block_count = lora_dim // block_size
        self.block_size = block_size

        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            shape = (out_dim, in_dim * k_size[0] * k_size[1])
            self.op = F.conv2d
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            shape = (out_dim, in_dim)
            self.op = F.linear
            self.extra_args = {}

        self.lora_dim = lora_dim
        self.up_list = nn.ParameterList(
            [torch.empty(shape[0], 1) for i in range(lora_dim)]
        )

        self.down_list = nn.ParameterList(
            [torch.empty(1, shape[1]) for i in range(lora_dim)]
        )

        self.index = 0

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # Need more experiences on init method

        for v in self.down_list:
            torch.nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        for v in self.up_list:
            torch.nn.init.zeros_(v)

        self.multiplier = multiplier
        self.org_module = [org_module]  # remove in applying
        self.org_forward = self.org_module[0].forward
        self.grad_ckpt = False
        self.state_dict()

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        destination["lora_up.weight"] = nn.Parameter(
            torch.concat(list(self.up_list), dim=1)
        )
        destination["lora_down.weight"] = nn.Parameter(
            torch.concat(list(self.down_list))
        )
        return destination

    def apply_to(self):
        self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=1.0):
        up = torch.concat(list(self.up_list), dim=1)
        down = torch.concat(list(self.down_list))
        diff_weight = up @ down * self.scale * multiplier / up.size(1)
        self.org_module[0].weight.data.add_(diff_weight)

    @torch.enable_grad()
    def forward(self, x):
        b = random.randint(0, self.block_count - 1)

        down = torch.concat(
            list(i.data for i in self.down_list[: b * self.block_size])
            + list(self.down_list[b * self.block_size : (b + 1) * self.block_size])
        )
        up = torch.concat(
            list(i.data for i in self.up_list[: b * self.block_size])
            + list(self.up_list[b * self.block_size : (b + 1) * self.block_size]),
            dim=1,
        )

        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        return self.op(
            x,
            self.org_module[0].weight
            + (up @ down).view(self.shape) * self.alpha / (b + 1),
            bias,
            **self.extra_args,
        )
