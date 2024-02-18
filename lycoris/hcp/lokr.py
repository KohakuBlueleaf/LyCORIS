import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisPluginBlock
from ..modules.lokr import make_kron, make_weight_cp, factorization


class LokrBlock(LycorisPluginBlock):
    wrapable_classes = (nn.Conv2d, nn.Linear)

    def __init__(self, *args, factor=-1, decompose_both=False, **kwargs):
        super(LokrBlock, self).__init__(*args, **kwargs)

        in_m, in_n = factorization(self.shape[1], factor)
        out_l, out_k = factorization(self.shape[0], factor)

        self.shape[0] = (out_l, out_k)
        self.shape[1] = (in_m, in_n)

        if decompose_both and self.dim < max(self.shape[0][0], self.shape[1][0]) / 2:
            self.lokr_w1_a = nn.Parameter(torch.empty(self.shape[0][0], self.dim))
            self.lokr_w1_b = nn.Parameter(torch.empty(self.dim, self.shape[1][0]))
        else:
            self.use_w1 = True
            self.lokr_w1 = nn.Parameter(
                torch.empty(self.shape[0][0], self.shape[1][0])
            )  # a*c, 1-mode

        if self.module_type == "conv":
            if self.dim >= max(self.shape[0][1], self.shape[1][1]) / 2:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(
                    torch.empty(self.shape[0][1], self.shape[1][1], *self.k_size)
                )
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(
                    torch.empty(self.dim, self.dim, self.shape[2], self.shape[3])
                )
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(self.dim, self.shape[0][1])
                )  # b, 1-mode
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(self.dim, self.shape[1][1])
                )  # d, 2-mode
            else:  # Conv2d not cp
                # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
                self.lokr_w2_a = nn.Parameter(torch.empty(self.shape[0][1], self.dim))
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(
                        self.dim, self.shape[1][1] * self.shape[2] * self.shape[3]
                    )
                )
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)
        elif self.module_type == "linear":
            if self.dim < max(self.shape[0][1], self.shape[1][1]) / 2:
                # bigger part. weight and LoRA. [b, dim] x [dim, d]
                self.lokr_w2_a = nn.Parameter(torch.empty(self.shape[0][1], self.dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(self.dim, self.shape[1][1]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(
                    torch.empty(self.shape[0][1], self.shape[1][1])
                )
        else:
            raise NotImplementedError

        if self.use_w2:
            torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        if type(self.alpha) == torch.Tensor:
            self.alpha = (
                self.alpha.detach().float().numpy()
            )  # without casting, bf16 causes error
        alpha = self.dim if self.alpha is None or self.alpha == 0 else self.alpha
        self.scale = alpha / self.dim
        print(self.scale)
        delattr(self, "alpha")
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            (
                self.lokr_w2
                if self.use_w2
                else (
                    make_weight_cp(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
                    if self.cp
                    else self.lokr_w2_a @ self.lokr_w2_b
                )
            ),
            torch.tensor(self.scale),
        )
        assert torch.sum(torch.isnan(weight)) == 0, "lokr init weight is nan"

    def forward(self, orig_weight, org_bias, new_weight, new_bias, *args, **kwargs):
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            (
                self.lokr_w2
                if self.use_w2
                else (
                    make_weight_cp(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
                    if self.tucker
                    else self.lokr_w2_a @ self.lokr_w2_b
                )
            ),
            torch.tensor(self.scale),
        )
        if orig_weight is not None:
            weight = weight.reshape(orig_weight.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0)) < self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            drop /= drop.mean()
            weight *= drop
        return weight, None, None, True, True
