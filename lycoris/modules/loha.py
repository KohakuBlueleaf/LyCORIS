import math

import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..functional.loha import diff_weight as loha_diff_weight


class LohaModule(LycorisBaseModule):
    name = "loha"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "hada_t1",
        "hada_t2",
        "alpha",
        "dora_scale",
    ]
    weight_list_det = ["hada_w1_a"]

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
        wd_on_out=False,
        bypass_mode=None,
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
            self.tucker = use_tucker and any(i != 1 for i in k_size)
            if self.tucker:
                w_shape = (out_dim, in_dim, *k_size)
            else:
                w_shape = (out_dim, in_dim * torch.tensor(k_size).prod().item())

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
        self.wd_on_out = wd_on_out
        if self.wd:
            org_weight = org_module.weight.cpu().clone().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_out:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.reshape(org_weight.shape[0], -1),
                        dim=1,
                        keepdim=True,
                    ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
                ).float()
            else:
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

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
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

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, w1a, w1b, w2a, w2b, t1, t2, alpha, dora_scale
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            w1b.size(0),
            float(alpha),
            use_tucker=t1 is not None,
            weight_decompose=dora_scale is not None,
        )
        module.hada_w1_a.copy_(w1a)
        module.hada_w1_b.copy_(w1b)
        module.hada_w2_a.copy_(w2a)
        module.hada_w2_b.copy_(w2b)
        if t1 is not None:
            module.hada_t1.copy_(t1)
            module.hada_t2.copy_(t2)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def get_weight(self, shape):
        scale = torch.tensor(
            self.scale, dtype=self.hada_w1_b.dtype, device=self.hada_w1_b.device
        )
        if self.tucker:
            weight = loha_diff_weight(
                self.hada_w1_b,
                self.hada_w1_a,
                self.hada_w2_b,
                self.hada_w2_a,
                self.hada_t1,
                self.hada_t2,
                gamma=scale,
            )
        else:
            weight = loha_diff_weight(
                self.hada_w1_b,
                self.hada_w1_a,
                self.hada_w2_b,
                self.hada_w2_a,
                None,
                None,
                gamma=scale,
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
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_diff_weight(multiplier=1, shape=shape, device=device)[0]
        weight = self.org_weight
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff, multiplier)
        else:
            merged = weight + diff * multiplier
        return merged, None

    def apply_weight_decompose(self, weight, multiplier=1):
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_out:
            weight_norm = (
                weight.reshape(weight.shape[0], -1)
                .norm(dim=1)
                .reshape(weight.shape[0], *[1] * self.dora_norm_dims)
            ) + torch.finfo(weight.dtype).eps
        else:
            weight_norm = (
                weight.transpose(0, 1)
                .reshape(weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(0, 1)
            ) + torch.finfo(weight.dtype).eps

        scale = self.dora_scale.to(weight.device) / weight_norm
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1

        return weight * scale

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
            diff_weight = self.get_weight(self.shape).to(self.dtype) * self.scalar
            weight = self.org_module[0].weight.data.to(self.dtype)
            if self.wd:
                weight = self.apply_weight_decompose(
                    weight + diff_weight, self.multiplier
                )
            else:
                weight = weight + diff_weight * self.multiplier
            bias = (
                None
                if self.org_module[0].bias is None
                else self.org_module[0].bias.data
            )
            return self.op(x, weight, bias, **self.kw_dict)
