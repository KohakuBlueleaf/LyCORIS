import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def factorization(dimension: int, factor:int=-1) -> tuple[int, int]:
    '''
    return a tuple of two value of input dimension decomposed by power of 2
    second value is higher or equal than first value.
    
    In LoRA with Kroneckor Product, first value is a value for weight.
    secon value is a value for weight scale.
    
    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two value is slightly different.
    
    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 127, 1   127 -> 127, 1    127 -> 127, 1   127 -> 127, 1   127 -> 127, 1
    128 -> 16, 8    128 -> 64, 2     128 -> 32, 4    128 -> 16, 8    128 -> 16, 8
    250 -> 125, 2   250 -> 125, 2    250 -> 125, 2   250 -> 125, 2   250 -> 125, 2
    360 -> 45, 8    360 -> 180, 2    360 -> 90, 4    360 -> 45, 8    360 -> 45, 8
    512 -> 32, 16   512 -> 256, 2    512 -> 128, 4   512 -> 64, 8    512 -> 32, 16
    1024 -> 32, 32  1024 -> 512, 2   1024 -> 256, 4  1024 -> 128, 8  1024 -> 64, 16
    '''
    
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        return m, n
    if factor == -1:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m<n:
        new_m = m + 1
        while dimension%new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m>factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n

def make_weight(orig_weight, w1a, w1b, w2a, w2b, scale):
    orig_weight, w1a, w1b, w2a, w2b, scale
    diff_weight = torch.kron(w1a@w1b, w2a@w2b)*scale
    orig_weight.reshape(diff_weight.shape) + diff_weight
    return orig_weight.reshape(diff_weight.shape) + diff_weight 
    

def make_weight_cp(orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale):
    a, b, c, d = w1a.shape[1], w1b.shape[1], w2a.shape[1], w2b.shape[2]
    rebuild1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1b, w1a) # [a, b, k1, k2]
    rebuild2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2b, w2a) # [c, d, k1, k2]
    
    temp_ab = torch.ones((a, b, 1, 1))
    temp_cd = torch.ones((c, d, 1, 1))
    
    rebuild1 = torch.kron(rebuild1, temp_cd) # [a, b, k1, k2] -> [ac, bd, k1, k2]
    rebuild2 = torch.kron(temp_ab, rebuild2) # [c, d, k1, k2] -> [ac, bd, k1, k2]
    
    return orig_weight+rebuild1*rebuild2*scale


class LokrModule(nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
        and from KohakuBlueleaf/LyCORIS/lycoris:loha:LoHaModule
    """

    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=4, alpha=1, 
        dropout=0.,
        use_cp=False,
        factor=None
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False

        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            
            in_m, in_n = factorization(in_dim)
            out_l, out_k = factorization(out_dim)
            
            self.cp = use_cp and k_size!=(1, 1)
            if self.cp:
                shape = ((out_l, out_k), (in_m, in_n), *k_size)
            else:
                shape = ((out_l, out_k), (in_m*k_size[0], in_n*k_size[1]))
            self.op = F.conv2d
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            
            in_m, in_n = factorization(in_dim)
            out_l, out_k = factorization(out_dim)
            
            shape = ((out_l, out_k), (in_m, in_n)) # out_dim = a * c, in_dim = b * d
            self.op = F.linear
            self.extra_args = {}
            
        # f = open('./log.txt', 'a', encoding='utf-8')
        # print(f'{self.lora_name} : ({in_dim}, {out_dim}) -> ({in_m}, {out_l})⊗({in_n}, {out_k})', file=f)
        # f.close()
        
        if self.cp:
            self.lokr_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, shape[2], shape[3]))
            self.lokr_w1_a = nn.Parameter(torch.empty(lora_dim, shape[0][0])) # a, 1-mode
            self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0])) # b , 2-mode
            
            self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, shape[2], shape[3]))
            self.lokr_w2_a = nn.Parameter(torch.empty(lora_dim, shape[0][1])) # c, 1-mode
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1])) # d , 2-mode
        else:
            self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
            self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            
            self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        # Same as loha.py
        if self.cp:
            torch.nn.init.normal_(self.lokr_t1, std=0.1)
            torch.nn.init.normal_(self.lokr_t2, std=0.1)
        torch.nn.init.normal_(self.lokr_w1_b, std=1)
        torch.nn.init.normal_(self.lokr_w2_b, std=0.01)
        torch.nn.init.normal_(self.lokr_w1_a, std=1)
        torch.nn.init.constant_(self.lokr_w2_a, 0)

        self.multiplier = multiplier
        self.org_module = [org_module]

    # Same as locon.py
    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def get_weight(self):
        d_weight = self.lokr_w1_a@self.lokr_w1_b
        d_weight = torch.kron(d_weight, self.lokr_w2_a@self.lokr_w2_b)
        return (d_weight).reshape(self.shape)
    
    def forward(self, x):
        if self.cp:
            weight = make_weight_cp(
                self.org_module[0].weight.data, 
                self.lokr_t1, self.lokr_w1_a, self.lokr_w1_b,
                self.lokr_t1, self.lokr_w2_a, self.lokr_w2_b,
                scale = torch.tensor(self.scale*self.multiplier),
            )
        else:
            weight = make_weight(
                self.org_module[0].weight.data, 
                self.lokr_w1_a, self.lokr_w1_b,
                self.lokr_w2_a, self.lokr_w2_b,
                scale = torch.tensor(self.scale*self.multiplier),
            )
        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        return self.op(
            x, 
            weight.view(self.shape),
            bias,
            **self.extra_args
        )