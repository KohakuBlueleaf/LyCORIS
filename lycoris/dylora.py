import math
import random

from collections import OrderedDict, abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F


class DyLoraModule(nn.Module):
    """
    Hadamard product Implementaion for Dynamic Low Rank adaptation
    """

    def __init__(
        self, 
        lora_name, 
        org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=4, alpha=1, 
        dropout=0.,
        use_cp=False, 
        block_size=1,
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        assert lora_dim % block_size == 0, 'lora_dim must be a multiple of block_size'
        self.block_count = lora_dim//block_size
        self.block_size = block_size
        
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
        
        self.lora_dim = lora_dim
        self.up_list = nn.ParameterList([
            torch.empty(shape[0], 1)
            for i in range(lora_dim)
        ])
        self.up_list.requires_grad_(False)
        self.up_update = [
            torch.zeros_like(self.up_list[i])
            for i in range(lora_dim)
        ]
        
        self.down_list = nn.ParameterList([
            torch.empty(1, shape[1])
            for i in range(lora_dim)
        ])
        self.down_list.requires_grad_(False)
        self.down_update = [
            torch.zeros_like(self.down_list[i])
            for i in range(lora_dim)
        ]
        
        self.index = 0
        
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        # Need more experiences on init method
        
        for v in self.down_list:
            torch.nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        for v in self.up_list:
            torch.nn.init.zeros_(v)
        for i, v in enumerate(self.up_update):
            v.copy_(self.up_list[i])
        for i, v in enumerate(self.down_update):
            v.copy_(self.down_list[i])

        self.multiplier = multiplier
        self.org_module = [org_module] # remove in applying
        self.grad_ckpt = False
        
        self.apply_train(0)
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        destination[f'{prefix}alpha'] = self.alpha
        destination[f'{prefix}lora_up.weight'] = nn.Parameter(
            torch.concat(self.up_update, dim=1)
        )
        destination[f'{prefix}lora_down.weight'] = nn.Parameter(
            torch.concat(self.down_update)
        )
        return destination

    def apply_to(self):
        self.org_module[0].forward = self.forward
    
    def apply_train(self, b:int):
        self.up_list.requires_grad_(False)
        self.down_list.requires_grad_(False)
            
        for i in range(self.index*self.block_size, (self.index+1)*self.block_size):
            self.up_update[i].copy_(self.up_list[i])
            self.down_update[i].copy_(self.down_list[i])
        
        for i in range(b*self.block_size, (b+1)*self.block_size):
            self.up_list[i].copy_(self.up_update[i])
            self.down_list[i].copy_(self.down_update[i])
        
        self.up_list.requires_grad_(True)
        self.down_list.requires_grad_(True)
        self.index = b

    @torch.enable_grad()
    def forward(self, x):
        b = random.randint(0, self.block_count-1)
        if self.up_update[b].device != self.up_list[b].device:
            device = self.up_list[b].device
            for i in range(self.lora_dim):
                self.up_update[i] = self.up_update[i].to(device)
                self.down_update[i] = self.down_update[i].to(device)
        
        if self.training:
            self.apply_train(b)
        down = torch.concat(
            list(self.down_update[:b*self.block_size]) 
            + list(self.down_list[b*self.block_size:(b+1)*self.block_size])
        )
        up = torch.concat(
            list(self.up_update[:b*self.block_size]) 
            + list(self.up_list[b*self.block_size:(b+1)*self.block_size]),
            dim=1
        )
        
        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        return self.op(
            x, 
            self.org_module[0].weight + (up@down).view(self.shape) * self.alpha/(b+1),
            bias,
            **self.extra_args
        )