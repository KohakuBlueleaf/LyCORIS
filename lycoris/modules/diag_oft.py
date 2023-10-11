import math
from weakref import ref
from collections import OrderedDict, abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .lokr import factorization


class DiagOFTModule(nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=4, alpha=1, 
        dropout=0., rank_dropout=0., module_dropout=0.,
        use_tucker=False, use_scalar=False, rank_dropout_scale=False,
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name

        if isinstance(org_module, nn.Linear):
            self.op = F.linear
            self.dim = org_module.out_features
            self.kw_dict = {}
        elif isinstance(org_module, nn.Conv2d):
            self.op = F.conv2d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        else:
            raise NotImplementedError
        
        out_dim = org_module.weight.shape[0]
        self.block_size, self.block_num = factorization(out_dim, lora_dim)
        # block_num > block_size
        self.oft_diag = nn.Parameter(torch.zeros(self.block_num, self.block_size, self.block_size))
        
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout
        
        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def make_weight(self, scale = 1, device=None):
        if self.rank_dropout and self.training:
            drop = (torch.rand(self.dim, device=device) < self.rank_dropout).to(self.oft_diag.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
        else:
            drop = 1
        org_weight = self.org_module[0].weight.to(device, dtype=self.oft_diag.dtype)
        org_weight = rearrange(org_weight, '(k n) ... -> k n ...', k=self.block_num, n=self.block_size)
        
        weight = torch.einsum(
            "k n m, k n ... -> k m ...", 
            self.oft_diag * scale + torch.eye(self.block_size, device=device), 
            org_weight
        )
        weight = rearrange(weight, 'k m ... -> (k m) ...')
        return weight

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.oft_diag.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm/2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired/norm
        
        scaled = ratio.item() != 1.0
        if scaled:
            self.oft_diag *= ratio
        
        return scaled, orig_norm*ratio

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier
        
        w = self.make_weight(scale, x.device)
        kw_dict = self.kw_dict | {"weight": w}
        return self.op(x, **kw_dict)
