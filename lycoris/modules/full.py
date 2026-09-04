from functools import cache

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..functional.general import add_scaled
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

    def _org_bias_tensor(self, device=None, dtype=None):
        """The cached original bias, or None. Cached as a one-element list."""
        if self.org_bias is None:
            return None
        return self.org_bias[0].to(device=device, dtype=dtype)

    def make_weight(self, scale=1, device=None):
        dropping = bool(self.rank_dropout) and self.training
        if self.is_diff and not dropping:
            # self.weight IS the diff here, so the merge is one scaled add.
            weight = add_scaled(self.org_weight.to(device), self.weight, scale)
            bias = None
            if self.bias is not None and self.org_bias is not None:
                bias = add_scaled(self._org_bias_tensor(device), self.bias, scale)
            return weight, bias

        if not dropping and scale == 1:
            # apply_to() folded the original in, so self.weight is the layer.
            return self.weight, self.bias

        # Rebuilt from the cached original rather than from the live module:
        # apply_to() takes the weight off the original, so it is the only copy
        # of the pre-training values left (#228).
        diff_w, diff_b = self.get_diff_weight(scale, device=device)
        if dropping:
            drop = (torch.rand(self.dim, device=device) > self.rank_dropout).to(
                diff_w.dtype
            )
            if self.rank_dropout_scale:
                drop = drop / drop.mean()
            # The mask is per output unit, so it broadcasts down the weight's
            # leading axis and straight along the bias.
            diff_w = diff_w * drop.view(-1, *[1] * (diff_w.dim() - 1))
            if diff_b is not None:
                diff_b = diff_b * drop

        weight = self.org_weight.to(diff_w) + diff_w
        org_bias = self._org_bias_tensor(diff_w.device)
        bias = None if (org_bias is None or diff_b is None) else org_bias + diff_b
        return weight, bias

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        if self.is_diff:
            diff = self.weight.to(device)
            diff_b = None if self.bias is None else self.bias.to(device)
        else:
            # Post-apply_to: self.weight is the absolute weight and the
            # original only survives in the cache.
            org_weight = self.org_weight.to(device=device, dtype=self.weight.dtype)
            diff = self.weight.to(device) - org_weight
            diff_b = None
            if self.bias is not None and self.org_bias is not None:
                org_bias = self._org_bias_tensor(device, self.bias.dtype)
                diff_b = self.bias.to(device) - org_bias
        if shape is not None:
            diff = diff.view(shape)
        if multiplier != 1:
            diff = diff * multiplier
            if diff_b is not None:
                diff_b = diff_b * multiplier
        return diff, diff_b

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        weight, bias = self.make_weight(multiplier, device)
        if shape is not None:
            weight = weight.view(shape)
            if bias is not None:
                bias = bias.view(shape[0])
        return weight, bias

    def forward(self, x: torch.Tensor, *args, **kwargs):
        dropped = bool(
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        )

        if not self.is_diff:
            # apply_to() folded the original weight into this module and took
            # it off the layer, so this call IS the layer's forward: delegating
            # to org_forward would look for a weight that is no longer there.
            if dropped:
                weight = self.org_weight.to(x)
                bias = self._org_bias_tensor(x.device, x.dtype)
            else:
                weight, bias = self.make_weight(self.multiplier, x.device)
            return self.op(x, weight=weight, bias=bias, **self.kw_dict)

        if dropped:
            return self.org_forward(x, *args, **kwargs)

        base = self.org_forward(x, *args, **kwargs)
        weight, bias = self.make_weight(self.multiplier, x.device)

        base_weight = self._current_weight().to(weight.device)
        delta_weight = weight - base_weight

        org_bias = self._current_bias()
        if bias is not None:
            bias = bias.to(x.device)

        if org_bias is not None and bias is not None:
            delta_bias = bias - org_bias.to(bias.device)
        else:
            delta_bias = bias

        delta = self.op(x, weight=delta_weight, bias=delta_bias, **self.kw_dict)
        return base + delta
