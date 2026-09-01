import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional.general import weight_decompose
from ..functional.locon import bypass_forward_diff as bypass_diff, diff_weight
from ..logging import logger


@cache
def log_wd():
    return logger.warning(
        "Using weight_decompose=True with LoRA (DoRA) will ignore network_dropout."
        "Only rank dropout and module dropout will be applied"
    )


class LoConModule(LycorisBaseModule):
    name = "locon"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "lora_mid.weight",
        "alpha",
        "dora_scale",
    ]
    weight_list_det = ["lora_up.weight"]

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
        wd_on_out=True,
        bypass_mode=None,
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

        if self.module_type.startswith("conv"):
            self.isconv = True
            # For general LoCon. in_dim follows torch Conv weight layout (in/groups)
            # so rebuild_weight matches F.conv*d when groups != 1 (#260).
            in_dim = org_module.in_channels // org_module.groups
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            use_tucker = use_tucker and any(i != 1 for i in k_size)
            self.down_op = self.op
            self.up_op = self.op
            if org_module.groups != 1 and self.bypass_mode:
                # Adapter Conv modules are groups=1 and take in/groups channels;
                # bypass forward on the full activation is not valid for grouped
                # originals, so force the weight-rebuild path.
                self.bypass_mode = False
            if use_tucker and any(i != 1 for i in k_size):
                self.lora_down = self.module(in_dim, lora_dim, 1, bias=False)
                self.lora_mid = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.lora_down = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.lora_up = self.module(lora_dim, out_dim, 1, bias=False)
        elif self.module_type == "linear":
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError

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

        if dropout:
            self.dropout = nn.Dropout(dropout)
            if self.wd:
                log_wd()
        else:
            self.dropout = nn.Identity()

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
        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        else:
            torch.nn.init.constant_(self.lora_up.weight, 0)
        if self.tucker:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, up, down, mid, alpha, dora_scale
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            down.size(0),
            float(alpha),
            use_tucker=mid is not None,
            weight_decompose=dora_scale is not None,
        )
        module.lora_up.weight.data.copy_(up)
        module.lora_down.weight.data.copy_(down)
        if mid is not None:
            module.lora_mid.weight.data.copy_(mid)
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

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        t = self.lora_mid.weight.to(device) if self.tucker else None
        # gamma=1: self.scale belongs to the caller and self.scalar can be a
        # parameter, so neither folds into the rebuild's scale.
        weight = diff_weight(wb, wa, t, gamma=1.0).view(self.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=device) > self.rank_dropout).to(
                weight.dtype
            )
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop

        return weight * self.scalar.to(device)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.make_weight(device=device) * scale
        if shape is not None:
            diff = diff.view(shape)
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
        return weight_decompose(weight, self.dora_scale, multiplier, self.wd_on_out)

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

    def bypass_forward_diff(self, x, scale=1):
        if self.rank_dropout and self.training:
            # The mask lives on the r axis between down and up, so this case
            # keeps the chain split; every other case is one dispatched call.
            if self.module_type == "linear":
                mid = F.linear(x, self.lora_down.weight.to(x))
            elif self.tucker:
                mid = self.lora_mid(self.lora_down(x))
            else:
                mid = self.lora_down(x)
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
            if self.module_type == "linear":
                output = F.linear(mid, self.lora_up.weight.to(mid))
            else:
                output = self.lora_up(mid)
            scalar = self.scalar.to(device=output.device, dtype=output.dtype)
            return self.dropout(output * scalar * self.scale * scale)

        # Geometry from the conv that carries it, not the org module: neither
        # lora conv is built with the org dilation or groups.
        op = self.lora_mid if self.tucker else self.lora_down
        extra_args = (
            {
                "stride": op.stride,
                "padding": op.padding,
                "dilation": op.dilation,
                "groups": op.groups,
            }
            if self.isconv
            else {}
        )
        # .to(x) so a quantized or fp8 base still works: the adapter weights
        # meet the activation, not the other way around.
        diff = bypass_diff(
            x,
            None,
            self.lora_down.weight.to(x),
            self.lora_up.weight.to(x),
            self.lora_mid.weight.to(x) if self.tucker else None,
            gamma=self.scale * scale,
            extra_args=extra_args,
        )
        scalar = self.scalar.to(device=diff.device, dtype=diff.dtype)
        return self.dropout(diff * scalar)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        scale = self.scale
        device = x.device

        base_weight = self._current_weight().to(device)
        diff_weight = self.make_weight(device).to(base_weight.dtype) * scale
        if self.wd:
            new_weight = self.apply_weight_decompose(
                base_weight + diff_weight, self.multiplier
            )
        else:
            new_weight = base_weight + diff_weight * self.multiplier

        delta_weight = new_weight - base_weight
        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta
