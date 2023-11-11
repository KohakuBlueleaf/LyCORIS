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
        super().__init__()
        
        self.lora_name = lora_name
        self.org_module = [org_module]

    def apply_to(self, **kwargs):
        self.org_weight = self.org_module[0].weight.data.clone().cpu()
        if self.org_module[0].bias is not None:
            self.org_bias = self.org_module[0].bias.data.clone().cpu()
        else:
            self.org_bias = None

    def custom_state_dict(self):
        sd = {
            'diff': self.org_module[0].weight.data.cpu() - self.org_weight
        }
        if self.org_bias is not None:
            sd['diff_b'] = self.org_module[0].bias.data.cpu() - self.org_bias
        return sd