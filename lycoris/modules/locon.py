import math
from weakref import ref
from collections import OrderedDict, abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoConModule(nn.Module):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0, 
        lora_dim=4, alpha=1, 
        dropout=0., rank_dropout=0., module_dropout=0.,
        use_cp=False,
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.cp = False

        self.scalar = nn.Parameter(torch.tensor(0.0))
        if isinstance(org_module, nn.Conv2d):
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            if use_cp and k_size != (1, 1):
                self.lora_down = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
                self.lora_mid = nn.Conv2d(lora_dim, lora_dim, k_size, stride, padding, bias=False)
                self.cp = True
            else:
                self.lora_down = nn.Conv2d(in_dim, lora_dim, k_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        elif isinstance(org_module, nn.Linear):
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError
        self.shape = org_module.weight.shape
        
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lora_up.weight)
        if self.cp:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        return (wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)).view(self.shape) * self.scalar
    
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
        destination[f'{prefix}lora_up.weight'] = self.lora_up.weight * self.scalar
        destination[f'{prefix}lora_down.weight'] = self.lora_down.weight
        if self.cp:
            destination[f'{prefix}lora_mid.weight'] = self.lora_mid.weight
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device).norm()*self.scale
        norm = torch.clamp(orig_norm, max_norm/2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu()/norm.cpu()
        
        scaled = ratio.item() != 1.0
        if scaled:
            self.scalar *= ratio
        
        return scaled, orig_norm*ratio

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier
        if self.cp:
            mid = self.lora_mid(self.lora_down(x))
        else:
            mid = self.lora_down(x)
        
        if self.rank_dropout and self.training:
            drop = torch.rand(self.lora_dim, device=mid.device) < self.rank_dropout
            if (dims:=len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1]*(dims-1), -1)
            mid = mid * drop
        
        return self.org_forward(x) + self.dropout(self.lora_up(mid) * self.scalar * scale)
