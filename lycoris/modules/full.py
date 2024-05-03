import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModuleCustomSD, LycorisBaseModule


class FullModule(LycorisBaseModule):
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }

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
        bypass_mode=False,
        **kwargs,
    ):
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
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Full algo.")

        self.weight = nn.Parameter(torch.zeros_like(org_module.weight))
        if org_module.bias is not None:
            self.bias = nn.Parameter(torch.zeros_like(org_module.bias))
        else:
            self.bias = None

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
        diff_weight = state_dict["diff"]
        self.weight.data.add_(diff_weight)
        if "diff_b" in state_dict:
            diff_bias = state_dict["diff_b"]
            self.bias.data.add_(diff_bias)

    def make_weight(self, scale=1, device=None, original=False):
        if original:
            weight = self._org_weight[0].to(device, dtype=self.weight.dtype)
            if self.org_bias is not None:
                bias = self.org_bias[0].to(device, dtype=self.bias.dtype)
            else:
                bias = None
            return weight, bias
        drop = (
            torch.rand(self.dim, device=device) > self.rank_dropout
            if self.rank_dropout and self.training
            else 1
        )
        if drop != 1 or scale != 1:
            org_weight = self.org_module[0].weight.to(device, dtype=self.weight.dtype)
            diff = self.weight.to(device) - org_weight
            weight = diff * drop * scale + org_weight
            if self.bias is not None:
                org_bias = self.org_module[0].bias.to(device, dtype=self.bias.dtype)
                diff_b = self.bias.to(device) - org_bias
                bias = diff_b * drop * scale + org_bias
        else:
            weight = self.weight
            bias = self.bias
        return weight, bias

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        weight, bias = self.make_weight(multiplier, device)
        if shape is not None:
            weight = weight.view(shape)
            if bias is not None:
                bias = bias.view(shape[0], 1)
        return weight, bias

    def forward(self, x: torch.Tensor):
        if (
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        ):
            original = True
        else:
            original = False
        scale = self.multiplier
        weight, bias = self.make_weight(scale, x.device, original=original)
        kw_dict = self.kw_dict | {"weight": weight, "bias": bias}
        return self.op(x, **kw_dict)
