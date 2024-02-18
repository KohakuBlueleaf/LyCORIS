import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import LycorisPluginBlock
from ..modules.lokr import factorization
from ..modules.diag_oft import DiagOFTModule, log_oft_factorize


class DiagOFTBlock(DiagOFTModule, LycorisPluginBlock):
    wrapable_classes = (nn.Conv2d, nn.Linear)

    def __init__(self, *args, constrain=0, rescaled=False, **kwargs):
        LycorisPluginBlock.__init__(self, *args, **kwargs)

        out_dim = self.shape[0]
        self.block_size, self.block_num = factorization(out_dim, self.dim)
        self.scale = 1.0
        # block_num > block_size
        self.rescaled = rescaled
        self.constrain = constrain * out_dim
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(torch.ones(out_dim))
        log_oft_factorize(
            dim=out_dim,
            factor=self.dim,
            num=self.block_num,
            bdim=self.block_size,
        )

    def forward(self, orig_weight, org_bias, new_weight, new_bias, *args, **kwargs):
        device = self.oft_blocks.device
        if self.rank_dropout and self.training:
            drop = (torch.rand(self.oft_blocks, device=device) < self.rank_dropout).to(
                self.oft_blocks.dtype
            )
            if self.rank_dropout_scale:
                drop /= drop.mean()
        else:
            drop = 1

        r = self.get_r()
        org_weight = orig_weight.to(device, dtype=r.dtype)
        org_weight = rearrange(
            org_weight, "(k n) ... -> k n ...", k=self.block_num, n=self.block_size
        )
        # Init R=0, so add I on it to ensure the output of step0 is original model output
        weight = torch.einsum(
            "k n m, k n ... -> k m ...",
            r * self.scale + (1 - self.scale) * self.I,
            org_weight,
        )
        weight = rearrange(weight, "k m ... -> (k m) ...")
        if self.rescaled:
            weight = self.rescale.to(weight) * weight
        # disable update weight for replace the weight directly
        return weight, None, None, False, False
