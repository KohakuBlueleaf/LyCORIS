import math
from weakref import ref
from collections import OrderedDict, abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullModule(nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=4, alpha=1, 
        dropout=0., rank_dropout=0., module_dropout=0.,
        use_tucker=False, use_scalar=False,
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
        self.diff = nn.Parameter(torch.zeros_like(org_module.weight))
        if org_module.bias is not None:
            self.diff_b = nn.Parameter(torch.zeros_like(org_module.bias))
        else:
            self.diff_b = None
        
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def make_weight(self, scale = 1, device=None):
        drop = (
            torch.rand(self.dim, device=device) < self.rank_dropout 
            if self.rank_dropout and self.training 
            else 1
        )
        org_weight = self.org_module[0].weight.to(device, dtype=self.diff.dtype)
        weight = self.diff.to(device) * drop * scale
        weight = weight + org_weight
        
        if self.diff_b is not None:
            org_bias = self.org_module[0].bias.to(device, dtype=self.diff_b.dtype)
            bias = self.diff_b.to(device) * drop * scale
            bias = bias + org_bias
        else:
            bias = None
        return weight, bias

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.diff.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm/2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired/norm
        
        scaled = ratio.item() != 1.0
        if scaled:
            self.diff *= ratio
            self.diff_b *= ratio
        
        return scaled, orig_norm*ratio

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier
        
        w, b = self.make_weight(scale, x.device)
        kw_dict = self.kw_dict | {"weight": w, "bias": b}
        return self.op(x, **kw_dict)
