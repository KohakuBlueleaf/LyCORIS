import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import ModuleCustomSD
from .lokr import factorization


class DiagOFTModule(ModuleCustomSD):
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
        constrain=0, rescaled=False,
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
        self.rescaled = rescaled
        self.constrain = constrain
        if self.constrain>0:
            #follow kohya's naming to reduce more works in inference
            self.oft_blocks = nn.Parameter(torch.zeros(self.block_num, self.block_size, self.block_size))
        else:
            # For non-constrained OFT, different formulation so use different naming
            self.oft_diag = nn.Parameter(torch.zeros(self.block_num, self.block_size, self.block_size))
        if rescaled:
            self.rescale = nn.Parameter(torch.zeros(self.block_num, self.block_size))
        self.I: torch.Tensor
        self.register_buffer("I", torch.eye(self.block_size), False)
        
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

        if self.constrain > 0:
            # for Q = -Q^T
            q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constrain:
                normed_q = q * self.constrain / q_norm
            else:
                normed_q = q
            r = (self.I + normed_q) @ (self.I - normed_q).inverse()
        else:
            r = self.oft_diag
        
        if self.rescaled:
            # Noted: not implemented in Kohya
            r = self.rescale * r
        
        org_weight = self.org_module[0].weight.to(device, dtype=r.dtype)
        org_weight = rearrange(org_weight, '(k n) ... -> k n ...', k=self.block_num, n=self.block_size)
        weight = torch.einsum(
            "k n m, k n ... -> k m ...", 
            r * scale + self.I * (1-scale), org_weight
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
