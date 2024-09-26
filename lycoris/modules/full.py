from functools import cache

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..logging import logger


@cache
def log_bypass_override():
    return logger.warning(
        "Automatic Bypass-Mode detected in algo=full, "
        "override with bypass_mode=False since algo=full not support bypass mode. "
        "If you are using quantized model which require bypass mode, please don't use algo=full. "
    )


class FullModule(LycorisBaseModule):
    name = "full"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = ["diff", "diff_b"]
    weight_list_det = ["diff"]

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
        use_scalar=False,
        rank_dropout_scale=False,
        bypass_mode=None,
        **kwargs,
    ):
        org_bypass = bypass_mode
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
        )
        if bypass_mode and org_bypass is None:
            self.bypass_mode = False
            log_bypass_override()

        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Full algo.")

        if self.is_quant:
            raise ValueError(
                "Quant Linear is not supported and meaningless in Full algo."
            )

        if self.bypass_mode:
            raise ValueError("bypass mode is not supported in Full algo.")

        self.weight = nn.Parameter(torch.zeros_like(org_module.weight))
        if org_module.bias is not None:
            self.bias = nn.Parameter(torch.zeros_like(org_module.bias))
        else:
            self.bias = None
        self.is_diff = True
        self._org_weight = [self.org_module[0].weight.data.cpu().clone()]
        if self.org_module[0].bias is not None:
            self.org_bias = [self.org_module[0].bias.data.cpu().clone()]
        else:
            self.org_bias = None

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, diff, diff_b):
        module = cls(
            lora_name,
            orig_module,
            1,
        )
        module.weight.copy_(diff)
        if diff_b is not None:
            if orig_module.bias is not None:
                module.bias.copy_(diff_b)
            else:
                module.bias = nn.Parameter(diff_b)
        module.is_diff = True
        return module

    @property
    def org_weight(self):
        return self._org_weight[0]

    @org_weight.setter
    def org_weight(self, value):
        self.org_module[0].weight.data.copy_(value)

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        self.weight.data.add_(self.org_module[0].weight.data)
        self._org_weight = [self.org_module[0].weight.data.cpu().clone()]
        delattr(self.org_module[0], "weight")
        if self.org_module[0].bias is not None:
            self.bias.data.add_(self.org_module[0].bias.data)
            self.org_bias = [self.org_module[0].bias.data.cpu().clone()]
            delattr(self.org_module[0], "bias")
        else:
            self.org_bias = None
        self.is_diff = False

    def restore(self):
        self.org_module[0].forward = self.org_forward
        self.org_module[0].weight = nn.Parameter(self._org_weight[0])
        if self.org_bias is not None:
            self.org_module[0].bias = nn.Parameter(self.org_bias[0])

    def custom_state_dict(self):
        sd = {"diff": self.weight.data.cpu() - self._org_weight[0]}
        if self.bias is not None:
            sd["diff_b"] = self.bias.data.cpu() - self.org_bias[0]
        return sd

    def load_weight_prehook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        diff_weight = state_dict.pop(f"{prefix}diff")
        state_dict[f"{prefix}weight"] = diff_weight + self.weight.data.to(diff_weight)
        if f"{prefix}diff_b" in state_dict:
            diff_bias = state_dict.pop(f"{prefix}diff_b")
            state_dict[f"{prefix}bias"] = diff_bias + self.bias.data.to(diff_bias)

    def make_weight(self, scale=1, device=None):
        drop = (
            torch.rand(self.dim, device=device) > self.rank_dropout
            if self.rank_dropout and self.training
            else 1
        )
        if drop != 1 or scale != 1 or self.is_diff:
            diff_w, diff_b = self.get_diff_weight(scale, device=device)
            weight = self.org_weight + diff_w * drop
            if self.org_bias is not None:
                bias = self.org_bias + diff_b * drop
            else:
                bias = None
        else:
            weight = self.weight
            bias = self.bias
        return weight, bias

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        if self.is_diff:
            diff_b = None
            if self.bias is not None:
                diff_b = self.bias * multiplier
            return self.weight * multiplier, diff_b
        org_weight = self.org_module[0].weight.to(device, dtype=self.weight.dtype)
        diff = self.weight.to(device) - org_weight
        diff_b = None
        if shape:
            diff = diff.view(shape)
        if self.bias is not None:
            org_bias = self.org_module[0].bias.to(device, dtype=self.bias.dtype)
            diff_b = self.bias.to(device) - org_bias
        if device is not None:
            diff = diff.to(device)
            if self.bias is not None:
                diff_b = diff_b.to(device)
        if multiplier != 1:
            diff = diff * multiplier
            if diff_b is not None:
                diff_b = diff_b * multiplier
        return diff * multiplier, diff_b

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        weight, bias = self.make_weight(multiplier, device)
        if shape is not None:
            weight = weight.view(shape)
            if bias is not None:
                bias = bias.view(shape[0])
        return weight, bias

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if (
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        ):
            original = True
        else:
            original = False
        if original:
            return self.org_forward(x)
        scale = self.multiplier
        weight, bias = self.make_weight(scale, x.device)
        kw_dict = self.kw_dict | {"weight": weight, "bias": bias}
        return self.op(x, **kw_dict)
