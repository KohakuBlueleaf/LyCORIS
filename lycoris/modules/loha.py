import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional.loha import loha_diff_weight
from ..utils.bnb import LinearNF4


class LohaModule(LycorisBaseModule):
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
        weight_decompose=False,
        bypass_mode=False,
        rs_lora=False,
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
            raise ValueError(f"{self.module_type} is not supported in LoHa algo.")
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora

        w_shape = self.shape
        if self.module_type.startswith("conv"):
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *k_size)
            self.tucker = use_tucker and k_size != (1, 1)
            if self.tucker:
                w_shape = (out_dim, in_dim, *k_size)
            else:
                w_shape = (out_dim, in_dim * torch.tensor(k_size).prod().item())
            self.op = F.conv2d

        if self.tucker:
            self.hada_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w1_a = nn.Parameter(
                torch.empty(lora_dim, w_shape[0])
            )  # out_dim, 1-mode
            self.hada_w1_b = nn.Parameter(
                torch.empty(lora_dim, w_shape[1])
            )  # in_dim , 2-mode

            self.hada_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *w_shape[2:]))
            self.hada_w2_a = nn.Parameter(
                torch.empty(lora_dim, w_shape[0])
            )  # out_dim, 1-mode
            self.hada_w2_b = nn.Parameter(
                torch.empty(lora_dim, w_shape[1])
            )  # in_dim , 2-mode
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

            self.hada_w2_a = nn.Parameter(torch.empty(w_shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, w_shape[1]))

        self.wd = weight_decompose
        if self.wd:
            org_weight: nn.Parameter = org_module.weight
            self.dora_norm_dims = org_weight.dim() - 1
            self.dora_scale = nn.Parameter(
                torch.norm(
                    org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                    dim=1,
                    keepdim=True,
                )
                .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(1, 0)
            ).float()

        if self.dropout:
            print("[WARN]LoHa/LoKr haven't implemented normal dropout yet.")

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.scalar = torch.tensor(1.0)
        # Need more experiments on init method
        if self.tucker:
            torch.nn.init.normal_(self.hada_t1, std=0.1)
            torch.nn.init.normal_(self.hada_t2, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w2_b, std=1)
        if use_scalar:
            torch.nn.init.normal_(self.hada_w2_a, std=0.1)
        else:
            torch.nn.init.constant_(self.hada_w2_a, 0)

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def get_weight(self, shape):
        if self.tucker:
            weight = loha_diff_weight(
                self.hada_w1_b,
                self.hada_w1_a,
                self.hada_w2_b,
                self.hada_w2_a,
                self.hada_t1,
                self.hada_t2,
                gamma=torch.tensor(self.scale),
            )
        else:
            weight = loha_diff_weight(
                self.hada_w1_b,
                self.hada_w1_a,
                self.hada_w2_b,
                self.hada_w2_a,
                None,
                None,
                gamma=torch.tensor(self.scale),
            )
        if shape is not None:
            weight = weight.reshape(shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0)) > self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop
        return weight

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.get_weight(shape) * scale
        if device is not None:
            diff = diff.to(device)
        return diff

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        merged = self.org_module[0].weight.data + self.get_diff_weight(
            multiplier=multiplier, shape=shape, device=device
        )
        if self.wd:
            merged = self.apply_weight_decompose(merged)
        return merged

    def apply_weight_decompose(self, weight):
        weight = weight.to(self.dora_scale.dtype)
        weight_norm = (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        ) + torch.finfo(weight.dtype).eps

        return weight * (self.dora_scale / weight_norm)

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["hada_w1_a"] = self.hada_w1_a * self.scalar
        destination["hada_w1_b"] = self.hada_w1_b
        destination["hada_w2_a"] = self.hada_w2_a
        destination["hada_w2_b"] = self.hada_w2_b
        if self.tucker:
            destination["hada_t1"] = self.hada_t1
            destination["hada_t2"] = self.hada_t2
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = (self.get_weight(self.shape) * self.scalar).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio

    def bypass_forward_diff(self, x, scale=1):
        diff_weight = self.get_weight(self.shape) * self.scalar * scale
        return self.drop(self.op(x, diff_weight, **self.kw_dict))

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.op(
                    x,
                    self.org_module[0].weight.data,
                    (
                        None
                        if self.org_module[0].bias is None
                        else self.org_module[0].bias.data
                    ),
                )
        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)
        else:
            weight = (
                self.org_module[0].weight.data.to(self.dtype)
                + self.get_weight(self.shape) * self.scalar * self.multiplier
            )
            bias = (
                None
                if self.org_module[0].bias is None
                else self.org_module[0].bias.data
            )

            if self.wd:
                weight = self.apply_weight_decompose(weight)
            return self.op(x, weight.view(self.shape), bias, **self.kw_dict)


if __name__ == "__main__":
    device = torch.device("cuda")
    base = nn.Linear(128, 128).to(device)
    loha = LohaModule("test", base, 1, 4, 1, weight_decompose=True).to(device)
    print(loha)
    test_input = torch.randn(1, 128).to(device)
    test_output = loha(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)

    base_4bit = LinearNF4(128, 128)
    base_4bit.load_state_dict(base.state_dict())
    base_4bit.to(device)
    qloha = LohaModule("test", base_4bit, 1, 4, 1, weight_decompose=False).to(device)
    print(qloha)
    test_input = torch.randn(1, 128).to(device)
    test_output = qloha(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)

    base = nn.Conv2d(128, 128, 3, 1, 1)
    loha = LohaModule("test", base, 1, 4, 1, weight_decompose=True, use_tucker=True)
    print(loha)
    test_input = torch.randn(1, 128, 16, 16)
    test_output = loha(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)
