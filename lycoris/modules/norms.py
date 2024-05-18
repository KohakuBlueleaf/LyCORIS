import torch
import torch.nn as nn

from .base import LycorisBaseModule


class NormModule(LycorisBaseModule):
    support_module = {
        "layernorm",
        "groupnorm",
    }

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
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Norm algo.")

        self.w_norm = nn.Parameter(torch.zeros(self.dim))
        self.b_norm = nn.Parameter(torch.zeros(self.dim))

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

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        w = self.w_norm * multiplier
        b = self.b_norm * multiplier
        if device is not None:
            w = w.to(device)
            b = b.to(device)
        if shape is not None:
            w = w.view(shape)
            b = b.view(shape)
        return w, b

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        org_w = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        org_b = self.org_module[0].bias.to(device, dtype=self.b_norm.dtype)
        diff_w, diff_b = self.get_diff_weight(multiplier, shape, device)
        weight = org_w + diff_w
        bias = org_b + diff_b
        return weight, bias

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier

        w, b = self.make_weight(scale, x.device)
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
