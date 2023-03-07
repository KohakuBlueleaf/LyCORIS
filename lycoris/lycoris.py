import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LycoConModule(nn.Module):
    """
    Conventional Implementaion for Low Rank Adaptation
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
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        
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
        self.org_module = org_module # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return self.org_forward(x) + self.dropout(self.lora_up(self.lora_down(x))) * self.multiplier * self.scale


class LycoHadaModule(nn.Module):
    """
    Hadamard product Implementaion for Low Rank Adaptation
    """

    def __init__(self, lora_name, org_module: nn.Module, multiplier=1.0, lora_dim=4, alpha=1, dropout=0.):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        
        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            shape = (out_dim, in_dim*k_size[0]*k_size[1])
            self.op = F.conv2d
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            shape = (out_dim, in_dim)
            self.op = F.linear
        
        self.hada_w1_a = nn.Parameter(torch.zeros(shape[0], lora_dim))
        self.hada_w1_b = nn.Parameter(torch.zeros(lora_dim, shape[1]))
        
        self.hada_w2_a = nn.Parameter(torch.zeros(shape[0], lora_dim))
        self.hada_w2_b = nn.Parameter(torch.zeros(lora_dim, shape[1]))
        
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
        torch.nn.init.kaiming_uniform_(self.hada_w1_b, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.hada_w2_b, a=math.sqrt(5))
        torch.nn.init.zeros_(self.hada_w1_a)
        torch.nn.init.zeros_(self.hada_w2_a)

        self.multiplier = multiplier
        self.org_module = org_module # remove in applying

    def apply_to(self):
        self.org_weight = self.org_module.weight.data
        if self.org_module.bias is not None:
            self.org_bias = self.org_module.bias.data
        else:
            self.org_bias = 0
        self.org_module.forward = self.forward
        del self.org_module

    def get_weight(self):
        return (
            (self.hada_w1_a @ self.hada_w1_b) 
            * (self.hada_w2_a @ self.hada_w2_b)
        ).reshape(self.shape)

    def forward(self, x):
        return self.op(
            x, 
            self.org_weight + self.get_weight()*self.scale*self.multiplier,
            self.org_bias,
        )