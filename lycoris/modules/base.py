from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logging import logger
from ..utils.bnb import QuantLinears, log_bypass


class ModuleCustomSD(nn.Module):
    def __init__(self):
        super().__init__()
        self._register_load_state_dict_pre_hook(self.load_weight_prehook)
        self.register_load_state_dict_post_hook(self.load_weight_hook)

    def load_weight_prehook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        pass

    def load_weight_hook(self, module, incompatible_keys):
        pass

    def custom_state_dict(self):
        return None

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
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

        if (custom_sd := self.custom_state_dict()) is not None:
            for k, v in custom_sd.items():
                destination[f"{prefix}{k}"] = v
            return destination
        else:
            return super().state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )


class LycorisBaseModule(ModuleCustomSD):
    support_module = {}

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        rank_dropout_scale=False,
        bypass_mode=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.op = F.linear
            self.dim = org_module.out_features
            self.kw_dict = {}
        elif isinstance(org_module, nn.Conv1d):
            self.module_type = "conv1d"
            self.op = F.conv1d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv2d):
            self.module_type = "conv2d"
            self.op = F.conv2d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.Conv3d):
            self.module_type = "conv3d"
            self.op = F.conv3d
            self.dim = org_module.out_channels
            self.kw_dict = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
        elif isinstance(org_module, nn.LayerNorm):
            self.module_type = "layernorm"
            self.op = F.layer_norm
            self.dim = org_module.normalized_shape[0]
            self.kw_dict = {
                "normalized_shape": org_module.normalized_shape,
                "eps": org_module.eps,
            }
        elif isinstance(org_module, nn.GroupNorm):
            self.module_type = "groupnorm"
            self.op = F.group_norm
            self.group_num = org_module.num_groups
            self.dim = org_module.num_channels
            self.kw_dict = {"num_groups": org_module.num_groups, "eps": org_module.eps}
        else:
            self.module_type = "unknown"

        if isinstance(org_module, QuantLinears):
            if not bypass_mode:
                log_bypass()
            bypass_mode = True
        self.bypass_mode = bypass_mode
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.org_forward = self.org_module[0].forward

    @property
    def org_weight(self):
        return self.org_module[0].weight

    @org_weight.setter
    def org_weight(self, value):
        self.org_module[0].weight.data.copy_(value)

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=1.0):
        weight = self.get_weight_difference(
            multiplier, self.org_weight.shape, self.org_weight.device
        )
        self.org_weight = weight

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        raise NotImplementedError

    def bypass_forward(self, x, scale=1):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
