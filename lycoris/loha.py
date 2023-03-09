import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orig_weight, w1a, w1b, w2a, w2b, scale=torch.tensor(1), dropout=nn.Identity()):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a@w1b)*(w2a@w2b)) * scale
        return orig_weight.reshape(diff_weight.shape) + dropout(diff_weight)

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        temp = grad_out*(w2a@w2b)*scale
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a@w1b)*scale
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp
        
        del temp
        return grad_out, grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


def make_weight(orig_weight, w1a, w1b, w2a, w2b, scale):
    return HadaWeight.apply(orig_weight, w1a, w1b, w2a, w2b, scale)


class LohaModule(nn.Module):
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
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            shape = (out_dim, in_dim)
            self.op = F.linear
            self.extra_args = {}
        
        self.hada_w1_a = nn.Parameter(torch.empty(shape[0], lora_dim))
        self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1]))
        
        self.hada_w2_a = nn.Parameter(torch.empty(shape[0], lora_dim))
        self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1]))
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        # Need more experiences on init method
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w2_b, std=0.05)
        torch.nn.init.normal_(self.hada_w1_a, std=1)
        torch.nn.init.constant_(self.hada_w2_a, 0)

        self.multiplier = multiplier
        self.org_module = [org_module] # remove in applying
        self.grad_ckpt = False

    def apply_to(self):
        self.org_module[0].forward = self.forward

    def get_weight(self):
        d_weight = self.hada_w1_a @ self.hada_w1_b
        d_weight *= self.hada_w2_a @ self.hada_w2_b
        return (d_weight).reshape(self.shape)

    @torch.enable_grad()
    def forward(self, x):
        # print(torch.mean(torch.abs(self.orig_w1a.to(x.device) - self.hada_w1_a)), end='\r')
        weight = make_weight(
            self.org_module[0].weight.data, 
            self.hada_w1_a, self.hada_w1_b,
            self.hada_w2_a, self.hada_w2_b,
            scale = torch.tensor(self.scale*self.multiplier),
        )
        
        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        return self.op(
            x, 
            weight.view(self.shape),
            bias,
            **self.extra_args
        )