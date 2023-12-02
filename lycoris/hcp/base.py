from typing import Tuple, Callable, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.lora import LoRACompatibleLinear, LoRACompatibleConv

try:
    import hcpdiff
    from hcpdiff.models.plugin import PatchPluginContainer, PatchPluginBlock
    from hcpdiff.utils.utils import isinstance_list
except ImportError:
    print(
        "HCP-Diffusion is not installed, hcp feature will be disabled. "
        "Use `pip install hcpdiff` to install hcpdiff."
    )

from ..utils.logger import logger


class LycorisPluginContainer(PatchPluginContainer):
    def __init__(self, host_name: str, host: nn.Module, parent_block: nn.Module):
        super(LycorisPluginContainer, self).__init__(host_name, host, parent_block)

        self.op = None
        self.module_type = ""
        self.extra_args = {}
        if isinstance(host, nn.Conv2d):
            self.op = F.conv2d
            self.module_type = "conv"
            self.extra_args = {
                "stride": host.stride,
                "padding": host.padding,
                "dilation": host.dilation,
                "groups": host.groups,
            }
        elif isinstance(host, nn.Linear):
            self.op = F.linear
            self.module_type = "linear"
            self.extra_args = {}
        elif isinstance(host, nn.LayerNorm):
            self.op = F.layer_norm
            self.module_type = "norm"
            self.extra_args = {
                "normalized_shape": host.normalized_shape,
                "eps": host.eps,
            }
        elif isinstance(host, nn.GroupNorm):
            self.op = F.group_norm
            self.module_type = "norm"
            self.extra_args = {"num_groups": host.num_groups, "eps": host.eps}
        else:
            logger.warning(
                f"Unsupported host type: '{type(host)}' for LyCORIS plugin. "
                "Will ignore this block."
            )

    def forward(self, *args, **kwargs):
        if self.op is None:
            return self._host(*args, **kwargs)

        if (
            isinstance(self._host, (LoRACompatibleLinear, LoRACompatibleConv))
            and len(args) == 2
        ):
            scale = args[1]
            args = [args[0]]
        else:
            scale = 1

        org_weight = self._host.weight
        org_bias = getattr(self._host, "bias", None)
        new_weight = org_weight
        new_bias = org_bias
        total_diff_output = 0

        for name in self.plugin_names:
            lyco_plugin_block = getattr(self, name)
            (
                diff_weight,
                diff_bias,
                diff_output,
                update_weight,
                update_bias,
            ) = lyco_plugin_block(
                org_weight, org_bias, new_weight, new_bias, *args, **kwargs
            )
            # logger.debug(f'diff weight norm: {torch.norm(diff_weight).detach().cpu().item()}')
            if diff_weight is not None:
                if update_weight:
                    new_weight = new_weight + diff_weight * scale
                else:
                    new_weight = diff_weight
            if diff_bias is not None:
                if update_bias:
                    new_bias = (
                        diff_bias * scale
                        if new_bias is None
                        else new_bias + diff_bias * scale
                    )
                else:
                    new_bias = diff_bias
            if diff_output is not None:
                total_diff_output = total_diff_output + diff_output * scale

        weight_dict = {
            "weight": new_weight,
            "bias": new_bias,
        }
        output = self.op(*args, **weight_dict, **self.extra_args)
        return output + total_diff_output


class LycorisPluginBlock(PatchPluginBlock):
    container_cls = LycorisPluginContainer

    def __init__(
        self,
        *args,
        dim=4,
        alpha=1.0,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        **kwargs,
    ):
        super(LycorisPluginBlock, self).__init__(*args, **kwargs)
        self.module_type = self.container().module_type
        self.dim = dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.tucker = False

        if self.module_type == "conv":
            self.out_dim = self.host().out_channels
            self.in_dim = self.host().in_channels
            self.k_size = self.host().kernel_size
            self.tucker = use_tucker
            self.shape = [self.out_dim, self.in_dim, *self.k_size]
        elif self.module_type == "linear":
            self.out_dim = self.host().out_features
            self.in_dim = self.host().in_features
            self.shape = [self.out_dim, self.in_dim]
        elif self.module_type == "norm":
            self.dim = self.host().weight.shape[0]
        else:
            raise NotImplementedError

    def forward(self, org_weight, org_bias, new_weight, new_bias, *args, **kwargs):
        raise NotImplementedError
