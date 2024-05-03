import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModuleCustomSD, LycorisBaseModule
from ..logging import logger
from ..utils.bnb import LinearNF4, QuantLinears, log_bypass


@cache
def log_wd():
    return logger.warning(
        "Using weight_decompose=True with LoRA (DoRA) will ignore network_dropout."
        "Only rank dropout and module dropout will be applied"
    )


class LoConModule(LycorisBaseModule):
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
        """if alpha == 0 or None, alpha is rank (no scaling)."""
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
            raise ValueError(f"{self.module_type} is not supported in LoRA/LoCon algo.")
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora

        if isinstance(org_module, nn.Conv2d):
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.tucker = use_tucker and k_size != (1, 1)
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
            self.op = F.linear
            self.extra_args = {}

        if self.module_type in {"conv1d", "conv2d", "conv3d"}:
            op = {
                "conv1d": F.conv1d,
                "conv2d": F.conv2d,
                "conv3d": F.conv3d,
            }[self.module_type]
            module = {
                "conv1d": nn.Conv1d,
                "conv2d": nn.Conv2d,
                "conv3d": nn.Conv3d,
            }[self.module_type]
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.down_op = op
            self.up_op = op
            if use_tucker and k_size != (1, 1):
                self.lora_down = module(in_dim, lora_dim, (1, 1), bias=False)
                self.lora_mid = module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.lora_down = module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.lora_up = module(lora_dim, out_dim, (1, 1), bias=False)
        elif isinstance(org_module, nn.Linear):
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError
        self.shape = org_module.weight.shape

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

        if dropout:
            self.dropout = nn.Dropout(dropout)
            if self.wd:
                log_wd()
        else:
            self.dropout = nn.Identity()
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

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
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        else:
            torch.nn.init.constant_(self.lora_up.weight, 0)
        if self.tucker:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.scalar = torch.ones_like(self.scalar)

    def merge_to(self, multiplier=1.0):
        weight = self.make_weight() * self.scale * multiplier
        self.org_module[0].weight.data.add_(weight)

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        if self.tucker:
            t = self.lora_mid.weight.to(device)
            wa = wa.squeeze(-1, -2)
            wb = wb.squeeze(-1, -2)
            weight = torch.einsum("i j k l, p i, j r -> p r k l", t, wa, wb)
        else:
            weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

        weight = weight.view(self.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0)) > self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop

        return weight * self.scalar.to(device)

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.make_weight(device=device) * scale
        if shape is not None:
            diff = diff.view(shape)
        if device is not None:
            diff = diff.to(device)
        merged = self.org_module[0].weight.data + diff
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

        return weight * (self.dora_scale.to(weight.device) / weight_norm)

    def custom_state_dict(self):
        destination = {}
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["alpha"] = self.alpha
        destination["lora_up.weight"] = self.lora_up.weight * self.scalar
        destination["lora_down.weight"] = self.lora_down.weight
        if self.tucker:
            destination["lora_mid.weight"] = self.lora_mid.weight
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device).norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier

        dtype = next(self.parameters()).dtype
        if not self.bypass_mode:
            weight = (
                self.org_module[0].weight.data.to(device=x.device, dtype=dtype)
                + self.make_weight(x.device).to(device=x.device, dtype=dtype) * scale
            )
            if self.wd:
                weight = self.apply_weight_decompose(weight)
            bias = (
                None
                if self.org_module[0].bias is None
                else self.org_module[0].bias.data
            )
            return self.op(x, weight, bias, **self.extra_args)
        else:
            if self.tucker:
                mid = self.lora_mid(self.lora_down(x))
            else:
                mid = self.lora_down(x)

            if self.rank_dropout and self.training:
                drop = (
                    torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
                ).to(mid.dtype)
                if self.rank_dropout_scale:
                    drop /= drop.mean()
                if (dims := len(x.shape)) == 4:
                    drop = drop.view(1, -1, 1, 1)
                else:
                    drop = drop.view(*[1] * (dims - 1), -1)
                mid = mid * drop

            return self.org_forward(x) + self.dropout(
                self.lora_up(mid) * self.scalar * scale
            )


if __name__ == "__main__":
    base = nn.Linear(128, 128).cuda()
    lokr = LoConModule("test", base, 1, 4, 1, weight_decompose=True).cuda()
    print(lokr)
    test_input = torch.randn(1, 128).cuda()
    test_output = lokr(test_input)
    print(test_output.shape)

    base_4bit = LinearNF4(128, 128)
    base_4bit.load_state_dict(base.state_dict())
    base_4bit.cuda()
    qlocon = LoConModule("test", base_4bit, 1, 4, 1, weight_decompose=False).cuda()
    print(qlocon)
    test_input = torch.randn(1, 128).cuda()
    test_output = qlocon(test_input)
    print(test_output.shape)

    base = nn.Conv2d(128, 128, 3, 1, 1)
    lokr = LoConModule("test", base, 1, 4, 1, weight_decompose=True, use_tucker=True)
    print(lokr)
    test_input = torch.randn(1, 128, 16, 16)
    test_output = lokr(test_input)
    print(test_output.shape)
