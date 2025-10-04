import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional import tucker_weight_from_conv


class GLoRAModule(LycorisBaseModule):
    name = "glora"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "a1.weight",
        "a2.weight",
        "b1.weight",
        "b2.weight",
        "bm.weight",
        "alpha",
    ]
    weight_list_det = ["a1.weight"]

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
        bypass_mode=None,
        rs_lora=False,
        **kwargs,
    ):
        """
        f(x) = WX + WAX + BX, where A and B are low-rank matrices
        bypass_forward(x) = W(X+A(X)) + B(X)
        bypass_forward_diff(x) = W(A(X)) + B(X)
        get_merged_weight() = W + WA + B
        get_diff_weight() = WA + B
        """
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
            raise ValueError(f"{self.module_type} is not supported in GLoRA algo.")
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora

        if self.module_type.startswith("conv"):
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            use_tucker = use_tucker and all(i == 1 for i in k_size)
            self.down_op = self.op
            self.up_op = self.op

            # A
            self.a2 = self.module(in_dim, lora_dim, 1, bias=False)
            self.a1 = self.module(lora_dim, in_dim, 1, bias=False)

            # B
            if use_tucker and any(i != 1 for i in k_size):
                self.b2 = self.module(in_dim, lora_dim, 1, bias=False)
                self.bm = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.b2 = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.b1 = self.module(lora_dim, out_dim, 1, bias=False)
        else:
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.a2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.a1 = nn.Linear(lora_dim, in_dim, bias=False)
            self.b2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.b1 = nn.Linear(lora_dim, out_dim, bias=False)

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
        torch.nn.init.kaiming_uniform_(self.a1.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.b1.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.a2.weight, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.b2.weight, a=math.sqrt(5))
        else:
            torch.nn.init.zeros_(self.a2.weight)
            torch.nn.init.zeros_(self.b2.weight)

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, a1, a2, b1, b2, bm, alpha
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            a2.size(0),
            float(alpha),
            use_tucker=bm is not None,
        )
        module.a1.weight.data.copy_(a1)
        module.a2.weight.data.copy_(a2)
        module.b1.weight.data.copy_(b1)
        module.b2.weight.data.copy_(b2)
        if bm is not None:
            module.bm.weight.data.copy_(bm)
        return module

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        destination["a1.weight"] = self.a1.weight
        destination["a2.weight"] = self.a2.weight * self.scalar
        destination["b1.weight"] = self.b1.weight
        destination["b2.weight"] = self.b2.weight * self.scalar
        if self.tucker:
            destination["bm.weight"] = self.bm.weight
        return destination

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
        wa1 = self.a1.weight.view(self.a1.weight.size(0), -1)
        wa2 = self.a2.weight.view(self.a2.weight.size(0), -1)
        orig = self.org_weight

        if self.tucker:
            wb = tucker_weight_from_conv(self.b1.weight, self.b2.weight, self.bm.weight)
        else:
            wb1 = self.b1.weight.view(self.b1.weight.size(0), -1)
            wb2 = self.b2.weight.view(self.b2.weight.size(0), -1)
            wb = wb1 @ wb2
            wb = wb.view(*orig.shape)
        if orig.dim() > 2:
            w_wa1 = torch.einsum("o i ..., i j -> o j ...", orig, wa1)
            w_wa2 = torch.einsum("o i ..., i j -> o j ...", w_wa1, wa2)
        else:
            w_wa2 = (orig @ wa1) @ wa2
        return (wb + w_wa2) * self.scale * self.scalar

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        weight = self.make_weight(device) * multiplier
        if shape is not None:
            weight = weight.view(shape)
        return weight, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff_w, _ = self.get_diff_weight(multiplier, shape, device)
        return self.org_weight + diff_w, None

    def _bypass_forward(self, x, scale=1, diff=False):
        scale = self.scale * scale
        ax_mid = self.a2(x) * scale
        bx_mid = self.b2(x) * scale

        if self.rank_dropout and self.training:
            drop_a = (
                torch.rand(self.lora_dim, device=ax_mid.device) < self.rank_dropout
            ).to(ax_mid.dtype)
            drop_b = (
                torch.rand(self.lora_dim, device=bx_mid.device) < self.rank_dropout
            ).to(bx_mid.dtype)
            if self.rank_dropout_scale:
                drop_a /= drop_a.mean()
                drop_b /= drop_b.mean()
            if (dims := len(x.shape)) == 4:
                drop_a = drop_a.view(1, -1, 1, 1)
                drop_b = drop_b.view(1, -1, 1, 1)
            else:
                drop_a = drop_a.view(*[1] * (dims - 1), -1)
                drop_b = drop_b.view(*[1] * (dims - 1), -1)
            ax_mid = ax_mid * drop_a
            bx_mid = bx_mid * drop_b
        return (
            self.org_forward(
                (0 if diff else x) + self.drop(self.a1(ax_mid)) * self.scale
            )
            + self.drop(self.b1(bx_mid)) * self.scale
        )

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale=scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale=scale, diff=False)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        if self.bypass_mode:
            return self.bypass_forward(x, self.multiplier)
        else:
            weight = (
                self.org_module[0].weight.data.to(self.dtype)
                + self.get_diff_weight(multiplier=self.multiplier)[0]
            )
            bias = (
                None
                if self.org_module[0].bias is None
                else self.org_module[0].bias.data
            )
            return self.op(x, weight, bias, **self.kw_dict)
