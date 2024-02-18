import math
from weakref import ref
from collections import OrderedDict, abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModuleCustomSD


class NormModule(ModuleCustomSD):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        rank_dropout_scale=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if isinstance(org_module, nn.LayerNorm):
            self.op = F.layer_norm
            self.dim = org_module.normalized_shape[0]
            self.kw_dict = {
                "normalized_shape": org_module.normalized_shape,
                "eps": org_module.eps,
            }
        elif isinstance(org_module, nn.GroupNorm):
            self.op = F.group_norm
            self.group_num = org_module.num_groups
            self.dim = org_module.num_channels
            self.kw_dict = {"num_groups": org_module.num_groups, "eps": org_module.eps}
        else:
            raise NotImplementedError
        self.w_norm = nn.Parameter(torch.zeros(self.dim))
        self.b_norm = nn.Parameter(torch.zeros(self.dim))

        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.org_forward = self.org_module[0].forward

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=1.0):
        weight, bias = self.make_weight(scale=multiplier)
        self.org_module[0].weight.data.copy_(weight)
        self.org_module[0].bias.data.copy_(bias)

    def make_weight(self, scale=1, device=None):
        org_weight = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        org_bias = self.org_module[0].bias.to(device, dtype=self.b_norm.dtype)
        if self.rank_dropout and self.training:
            drop = (torch.rand(self.dim, device=device) < self.rank_dropout).to(
                self.w_norm.device
            )
            if self.rank_dropout_scale:
                drop /= drop.mean()
        else:
            drop = 1
        drop = (
            torch.rand(self.dim, device=device) < self.rank_dropout
            if self.rank_dropout and self.training
            else 1
        )
        weight = self.w_norm.to(device) * drop * scale
        bias = self.b_norm.to(device) * drop * scale
        return org_weight + weight, org_bias + bias

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier

        w, b = self.make_weight(scale, x.device)
        kw_dict = self.kw_dict | {"weight": w, "bias": b}
        return self.op(x, **kw_dict)
