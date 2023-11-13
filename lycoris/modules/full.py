import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModuleCustomSD


class FullModule(ModuleCustomSD):
    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=4, alpha=1, 
        dropout=0., rank_dropout=0., module_dropout=0.,
        use_tucker=False, use_scalar=False, rank_dropout_scale=False,
        **kwargs,
    ):
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
        self.weight = nn.Parameter(torch.zeros_like(org_module.weight))
        if org_module.bias is not None:
            self.bias = nn.Parameter(torch.zeros_like(org_module.bias))
        else:
            self.bias= None
        
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.weight.data.copy_(self.org_module[0].weight.data)
        if self.org_module[0].bias is not None:
            self.bias.data.copy_(self.org_module[0].bias.data)

    def custom_state_dict(self):
        sd = {
            'diff': self.weight.data.cpu() - self.org_module[0].weight.data.cpu()
        }
        if self.bias is not None:
            sd['diff_b'] = self.bias.data.cpu() - self.org_module[0].bias.data.cpu()
        return sd

    def make_weight(self, scale = 1, device=None):
        drop = (
            torch.rand(self.dim, device=device) > self.rank_dropout 
            if self.rank_dropout and self.training 
            else 1
        )
        if drop!=1 or scale!=1:
            org_weight = self.org_module[0].weight.to(device, dtype=self.weight.dtype)
            diff = self.weight.to(device) - org_weight
            weight = diff * drop * scale + org_weight
            if self.bias is not None:
                org_bias = self.org_module[0].bias.to(device, dtype=self.bias.dtype)
                diff_b = self.bias.to(device) - org_bias
                bias = diff_b * drop * scale + org_bias
        else:
            weight = self.weight
            bias = self.bias
        return weight, bias

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier
        weight, bias = self.make_weight(scale)
        kw_dict = self.kw_dict | {"weight": weight, "bias": bias}
        return self.op(x, **kw_dict)