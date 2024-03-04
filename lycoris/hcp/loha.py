import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisPluginBlock
from ..modules.loha import make_weight, make_weight_tucker


class LohaBlock(LycorisPluginBlock):
    wrapable_classes = (nn.Conv2d, nn.Linear)

    def __init__(self, *args, **kwargs):
        super(LohaBlock, self).__init__(*args, **kwargs)

        if self.tucker:
            self.hada_t1 = nn.Parameter(
                torch.empty(self.dim, self.dim, self.shape[2], self.shape[3])
            )
            self.hada_w1_a = nn.Parameter(
                torch.empty(self.dim, self.shape[0])
            )  # out_dim, 1-mode
            self.hada_w1_b = nn.Parameter(
                torch.empty(self.dim, self.shape[1])
            )  # in_dim , 2-mode

            self.hada_t2 = nn.Parameter(
                torch.empty(self.dim, self.dim, self.shape[2], self.shape[3])
            )
            self.hada_w2_a = nn.Parameter(
                torch.empty(self.dim, self.shape[0])
            )  # out_dim, 1-mode
            self.hada_w2_b = nn.Parameter(
                torch.empty(self.dim, self.shape[1])
            )  # in_dim , 2-mode
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(self.shape[0], self.dim))
            self.hada_w1_b = nn.Parameter(torch.empty(self.dim, self.shape[1]))

            self.hada_w2_a = nn.Parameter(torch.empty(self.shape[0], self.dim))
            self.hada_w2_b = nn.Parameter(torch.empty(self.dim, self.shape[1]))

        if type(self.alpha) == torch.Tensor:
            self.alpha = (
                self.alpha.detach().float().numpy()
            )  # without casting, bf16 causes error
        alpha = self.dim if self.alpha is None or self.alpha == 0 else self.alpha
        self.scale = alpha / self.dim
        delattr(self, "alpha")
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        if self.tucker:
            torch.nn.init.normal_(self.hada_t1, std=0.1)
            torch.nn.init.normal_(self.hada_t2, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w2_b, std=1)
        torch.nn.init.constant_(self.hada_w2_a, 0)

    def forward(self, orig_weight, org_bias, new_weight, new_bias, *args, **kwargs):
        if self.tucker:
            weight = make_weight_tucker(
                self.hada_t1,
                self.hada_w1_a,
                self.hada_w1_b,
                self.hada_t2,
                self.hada_w2_a,
                self.hada_w2_b,
                scale=torch.tensor(self.scale),
            )
        else:
            weight = make_weight(
                self.hada_w1_a,
                self.hada_w1_b,
                self.hada_w2_a,
                self.hada_w2_b,
                scale=torch.tensor(self.scale),
            )
        if orig_weight is not None:
            weight = weight.reshape(orig_weight.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0)) < self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            drop /= drop.mean()
            weight *= drop
        return weight, None, None, True, True
