import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..logging import warning_once


class NormModule(LycorisBaseModule):
    name = "norm"
    support_module = {
        "layernorm",
        "groupnorm",
    }
    weight_list = ["w_norm", "b_norm"]
    weight_list_det = ["w_norm"]

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
        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            rank_dropout_scale=rank_dropout_scale,
            **kwargs,
        )
        if self.module_type == "unknown":
            if not hasattr(org_module, "weight") or not hasattr(org_module, "_norm"):
                warning_once(f"{type(org_module)} is not supported in Norm algo.")
                self.not_supported = True
                return
            else:
                self.dim = org_module.weight.numel()
                self.not_supported = False
        elif self.module_type not in self.support_module:
            warning_once(f"{self.module_type} is not supported in Norm algo.")
            self.not_supported = True
            return

        self.w_norm = nn.Parameter(torch.zeros(self.dim))
        if hasattr(org_module, "bias"):
            self.b_norm = nn.Parameter(torch.zeros(self.dim))
        if hasattr(org_module, "_norm"):
            self.org_norm = org_module._norm
        else:
            self.org_norm = None

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, w_norm, b_norm):
        module = cls(
            lora_name,
            orig_module,
            1,
        )
        module.w_norm.copy_(w_norm)
        if b_norm is not None:
            module.b_norm.copy_(b_norm)
        return module

    def make_weight(self, scale=1, device=None):
        org_weight = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        if hasattr(self.org_module[0], "bias"):
            org_bias = self.org_module[0].bias.to(device, dtype=self.b_norm.dtype)
        else:
            org_bias = None
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
        if org_bias is not None:
            bias = self.b_norm.to(device) * drop * scale
        return org_weight + weight, org_bias + bias if org_bias is not None else None

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        if self.not_supported:
            return 0, 0
        w = self.w_norm * multiplier
        if device is not None:
            w = w.to(device)
        if shape is not None:
            w = w.view(shape)
        if self.b_norm is not None:
            b = self.b_norm * multiplier
            if device is not None:
                b = b.to(device)
            if shape is not None:
                b = b.view(shape)
        else:
            b = None
        return w, b

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        if self.not_supported:
            return None, None
        diff_w, diff_b = self.get_diff_weight(multiplier, shape, device)
        org_w = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        weight = org_w + diff_w
        if diff_b is not None:
            org_b = self.org_module[0].bias.to(device, dtype=self.b_norm.dtype)
            bias = org_b + diff_b
        else:
            bias = None
        return weight, bias

    def forward(self, x):
        if self.not_supported or (
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        ):
            return self.org_forward(x)
        scale = self.multiplier

        w, b = self.make_weight(scale, x.device)
        if self.org_norm is not None:
            normed = self.org_norm(x)
            scaled = normed * w
            if b is not None:
                scaled += b
            return scaled

        kw_dict = self.kw_dict | {"weight": w, "bias": b}
        return self.op(x, **kw_dict)


if __name__ == "__main__":
    base = nn.LayerNorm(128).cuda()
    norm = NormModule("test", base, 1).cuda()
    print(norm)
    test_input = torch.randn(1, 128).cuda()
    test_output = norm(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)

    base = nn.GroupNorm(4, 128).cuda()
    norm = NormModule("test", base, 1).cuda()
    print(norm)
    test_input = torch.randn(1, 128, 3, 3).cuda()
    test_output = norm(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)
