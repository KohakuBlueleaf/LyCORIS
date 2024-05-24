import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..utils.bnb import LinearNF4


class GLoRAModule(LycorisBaseModule):
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
            raise ValueError(f"{self.module_type} is not supported in LoRA/LoCon algo.")
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
            self.tucker = use_tucker and k_size != (1, 1)
            self.down_op = self.op
            self.up_op = self.op
            if use_tucker and k_size != (1, 1):
                self.a2 = self.module(in_dim, lora_dim, (1, 1), bias=False)
                self.am = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.a2 = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.a1 = self.module(lora_dim, in_dim, (1, 1), bias=False)
            if use_tucker and k_size != (1, 1):
                self.b2 = self.module(in_dim, lora_dim, (1, 1), bias=False)
                self.bm = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.b2 = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.b1 = self.module(lora_dim, out_dim, (1, 1), bias=False)
        elif isinstance(org_module, nn.Linear):
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.a2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.a1 = nn.Linear(lora_dim, in_dim, bias=False)
            self.b2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.b1 = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError

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
        torch.nn.init.zeros_(self.a2.weight)
        torch.nn.init.kaiming_uniform_(self.b1.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.b2.weight)

    def make_weight(self, device=None):
        wa1 = self.a1.weight.view(self.a1.weight.size(0), -1)
        wa2 = self.a2.weight.view(self.a2.weight.size(0), -1)
        wb1 = self.b1.weight.view(self.b1.weight.size(0), -1)
        wb2 = self.b2.weight.view(self.b2.weight.size(0), -1)
        orig = self.org_weight.view(self.org_module[0].weight.size(0), -1)
        return (wb2 @ wb1) + ((orig @ wa2) @ wa1)

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier

        ax_mid = self.a1(x) * scale
        bx_mid = self.b1(x) * scale

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
            self.org_forward(x + self.dropout(self.a2(ax_mid)) * self.scale)
            + self.dropout(self.b2(bx_mid)) * self.scale
        )


if __name__ == "__main__":
    device = torch.device("cuda")
    module = GLoRAModule
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        base = nn.Linear(128, 128).to(device).half()
        net = module("test", base, 1, 4, 1, weight_decompose=True).to(device)
        print(net)
        test_input = torch.randn(1, 128).to(device).half()
        test_output = net(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)

        base_4bit = LinearNF4(128, 128, device="cuda")
        base_4bit.load_state_dict(base.state_dict())
        base_4bit.to(device)
        qnet = module("test", base_4bit, 1, 4, 1, weight_decompose=False).to(device)
        print(qnet)
        test_input = torch.randn(1, 128).to(device).half()
        test_output = qnet(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)

        base = nn.Conv2d(128, 128, 3, 1, 1).to(device).half()
        net = module("test", base, 1, 4, 1, weight_decompose=True, use_tucker=True).to(
            device
        )
        print(net)
        test_input = torch.randn(1, 128, 16, 16).to(device).half()
        test_output = net(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)

        base = nn.Conv2d(128, 128, 3, 1, 1).to(device).half()
        net = module.parametrize(
            base, "weight", 1, 4, 1, weight_decompose=True, use_tucker=True
        ).to(device)
        print(base)
        test_input = torch.randn(1, 128, 16, 16).to(device).half()
        test_output = base(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)
