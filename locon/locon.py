import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoConModule(nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(self, lora_name, org_module: nn.Module, multiplier=1.0, lora_dim=4, alpha=1, dropout=0.):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == 'Conv2d':
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.lora_down = nn.Conv2d(in_dim, lora_dim, k_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
            self.op = F.conv2d
            self.extra_args = {
                'stride': stride,
                'padding': padding
            }
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
            self.op = F.linear
            self.extra_args = {}
        self.shape = org_module.weight.shape
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self):
        self.org_module[0].forward = self.forward

    def make_weight(self):
        wa = self.lora_up.weight
        wb = self.lora_down.weight
        return (wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)).view(self.shape)

    def forward(self, x):
        return self.op(
            x,
            (self.org_module[0].weight 
             + self.dropout(self.make_weight()) * self.multiplier * self.scale),
            self.org_module[0].bias,
            **self.extra_args,
        )