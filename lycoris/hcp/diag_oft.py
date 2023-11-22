import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import LycorisPluginBlock
from ..modules.lokr import factorization
from ..modules.diag_oft import DiagOFTModule


class DiagOFTBlock(DiagOFTModule, LycorisPluginBlock):
    wrapable_classes = (nn.Conv2d, nn.Linear)

    def __init__(
        self, *args, constrain=0, rescaled=False, **kwargs
    ):
        LycorisPluginBlock.__init__(self, *args, **kwargs)
        
        out_dim = self.shape[0]
        self.block_size, self.block_num = factorization(out_dim, self.dim)
        # block_num > block_size
        self.rescaled = rescaled
        self.constrain = constrain * out_dim
        self.oft_blocks = nn.Parameter(torch.zeros(self.block_num, self.block_size, self.block_size))
        if rescaled:
            self.rescale = nn.Parameter(torch.ones(self.block_num, self.block_size, 1))
    
    def forward(self, orig_weight, org_bias, new_weight, new_bias, *args, **kwargs):
        device = self.oft_blocks.device
        if self.rank_dropout and self.training:
            drop = (torch.rand(self.oft_blocks, device=device) < self.rank_dropout).to(self.oft_blocks.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
        else:
            drop = 1

        r = self.get_r()
        org_weight = orig_weight.to(device, dtype=r.dtype)
        org_weight = rearrange(org_weight, '(k n) ... -> k n ...', k=self.block_num, n=self.block_size)
        # Init R=0, so add I on it to ensure the output of step0 is original model output
        weight = torch.einsum(
            "k n m, k n ... -> k m ...", 
            r - self.I, org_weight
        )
        weight = rearrange(weight, 'k m ... -> (k m) ...')
        return weight, None, None