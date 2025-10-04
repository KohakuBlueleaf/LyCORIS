from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from ..utils.quant import QuantLinears, log_bypass, log_suspect


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


@dataclass
class _MergeContext:
    precise: bool
    target_device: torch.device
    target_dtype: torch.dtype
    compute_dtype: torch.dtype
    param_device: torch.device | None
    param_dtype: torch.dtype | None
    module: nn.Module
    weight_param: torch.Tensor
    bias_param: torch.Tensor | None


class LycorisBaseModule(ModuleCustomSD):
    name: str
    dtype_tensor: torch.Tensor
    support_module = {}
    weight_list = []
    weight_list_det = []

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        rank_dropout_scale=False,
        bypass_mode=None,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.not_supported = False

        self.module = type(org_module)
        if isinstance(org_module, nn.Linear):
            self.module_type = "linear"
            self.shape = (org_module.out_features, org_module.in_features)
            self.op = F.linear
            self.dim = org_module.out_features
            self.kw_dict = {}
        elif isinstance(org_module, nn.Conv1d):
            self.module_type = "conv1d"
            self.shape = (
                org_module.out_channels,
                org_module.in_channels,
                *org_module.kernel_size,
            )
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
            self.shape = (
                org_module.out_channels,
                org_module.in_channels,
                *org_module.kernel_size,
            )
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
            self.shape = (
                org_module.out_channels,
                org_module.in_channels,
                *org_module.kernel_size,
            )
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
            self.shape = tuple(org_module.normalized_shape)
            self.op = F.layer_norm
            self.dim = org_module.normalized_shape[0]
            self.kw_dict = {
                "normalized_shape": org_module.normalized_shape,
                "eps": org_module.eps,
            }
        elif isinstance(org_module, nn.GroupNorm):
            self.module_type = "groupnorm"
            self.shape = (org_module.num_channels,)
            self.op = F.group_norm
            self.group_num = org_module.num_groups
            self.dim = org_module.num_channels
            self.kw_dict = {"num_groups": org_module.num_groups, "eps": org_module.eps}
        else:
            self.not_supported = True
            self.module_type = "unknown"

        self.register_buffer("dtype_tensor", torch.tensor(0.0), persistent=False)

        self.is_quant = False
        if isinstance(org_module, QuantLinears):
            if not bypass_mode:
                log_bypass()
            self.is_quant = True
            bypass_mode = True
        if (
            isinstance(org_module, nn.Linear)
            and org_module.__class__.__name__ != "Linear"
        ):
            if bypass_mode is None:
                log_suspect()
                bypass_mode = True
            if bypass_mode == True:
                self.is_quant = True
        self.bypass_mode = bypass_mode
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        ## Dropout things
        # Since LoKr/LoHa/OFT/BOFT are hard to follow the rank_dropout definition from kohya
        # We redefine the dropout procedure here.
        # g(x) = WX + drop(Brank_drop(AX)) for LoCon(lora), bypass
        # g(x) = WX + drop(ΔWX) for any algo except LoCon(lora), bypass
        # g(x) = (W + Brank_drop(A))X for LoCon(lora), rebuid
        # g(x) = (W + rank_drop(ΔW))X for any algo except LoCon(lora), rebuild
        self.drop = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.rank_drop = (
            nn.Identity() if rank_dropout == 0 else nn.Dropout(rank_dropout)
        )

        self.multiplier = multiplier
        self.org_forward = org_module.forward
        self.org_module = [org_module]

    @classmethod
    def parametrize(cls, org_module, attr, *args, **kwargs):
        from .full import FullModule

        if cls is FullModule:
            raise RuntimeError("FullModule cannot be used for parametrize.")
        target_param = getattr(org_module, attr)
        kwargs["bypass_mode"] = False
        if target_param.dim() == 2:
            proxy_module = nn.Linear(
                target_param.shape[0], target_param.shape[1], bias=False
            )
            proxy_module.weight = target_param
        elif target_param.dim() > 2:
            module_type = [
                None,
                None,
                None,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                None,
                None,
            ][target_param.dim()]
            proxy_module = module_type(
                target_param.shape[0],
                target_param.shape[1],
                *target_param.shape[2:],
                bias=False,
            )
            proxy_module.weight = target_param
        module_obj = cls("", proxy_module, *args, **kwargs)
        module_obj.forward = module_obj.parametrize_forward
        module_obj.to(target_param)
        parametrize.register_parametrization(org_module, attr, module_obj)
        return module_obj

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        return any(f"{lora_name}.{k}" in state_dict for k in cls.weight_list_det)

    @classmethod
    def extract_state_dict(cls, state_dict, lora_name):
        return [state_dict.get(f"{lora_name}.{k}", None) for k in cls.weight_list]

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, *weights):
        raise NotImplementedError

    @property
    def dtype(self):
        return self.dtype_tensor.dtype

    @property
    def device(self):
        return self.dtype_tensor.device

    @property
    def org_weight(self):
        return self.org_module[0].weight

    @org_weight.setter
    def org_weight(self, value):
        self.org_module[0].weight.data.copy_(value)

    def _current_weight(self):
        return self.org_module[0].weight.detach()

    def _current_bias(self):
        bias = self.org_module[0].bias
        return None if bias is None else bias.detach()

    def apply_to(self, **kwargs):
        if self.not_supported:
            return

        module = self.org_module[0]
        if not hasattr(module, "_lycoris_original_forward"):
            module._lycoris_original_forward = module.forward

        wrappers = list(getattr(module, "_lycoris_wrappers", []))
        if self in wrappers:
            wrappers.remove(self)

        self.org_forward = module.forward
        wrappers.append(self)

        module._lycoris_wrappers = wrappers
        module.forward = self.forward

    def restore(self):
        if self.not_supported:
            return
        module = self.org_module[0]
        wrappers = list(getattr(module, "_lycoris_wrappers", []))

        if not wrappers:
            module.forward = getattr(
                module, "_lycoris_original_forward", self.org_forward
            )
            return

        try:
            idx = wrappers.index(self)
        except ValueError:
            module.forward = (
                wrappers[-1].forward
                if wrappers
                else getattr(module, "_lycoris_original_forward", self.org_forward)
            )
            return

        wrappers.pop(idx)

        if idx < len(wrappers):
            wrappers[idx].org_forward = self.org_forward

        if wrappers:
            module._lycoris_wrappers = wrappers
            module.forward = wrappers[-1].forward
        else:
            module.forward = getattr(
                module, "_lycoris_original_forward", self.org_forward
            )
            module.__dict__.pop("_lycoris_wrappers", None)
            module.__dict__.pop("_lycoris_original_forward", None)

    def merge_to(self, multiplier=1.0, *, precise: bool = False):
        if self.not_supported:
            return

        ctx = self._prepare_merge_context(precise)

        if precise:
            weight_prec, bias_prec = self._compute_precise_result(ctx, multiplier)
            self._apply_precise_weights(ctx, weight_prec, bias_prec)
        else:
            weight, bias = self.get_merged_weight(
                multiplier,
                ctx.weight_param.shape,
                ctx.target_device,
            )
            self._apply_merged_weights(ctx, weight, bias)

        self._restore_merge_context(ctx)

    def onfly_merge(self, multiplier=1.0):
        if self.not_supported:
            return
        self_device = next(self.parameters()).device
        self_dtype = next(self.parameters()).dtype
        self.to(self.org_weight)
        self.cached_org_weight = self.org_weight.data.cpu()
        self.cached_org_bias = None
        weight, bias = self.get_merged_weight(
            multiplier, self.org_weight.shape, self.org_weight.device
        )
        self.org_weight = weight
        if bias is not None:
            bias = bias.to(self.org_weight)
            if self.org_module[0].bias is not None:
                self.org_module[0].bias.data.copy_(bias)
                self.cached_org_bias = self.org_module[0].bias.data.cpu()
            else:
                self.org_module[0].bias = nn.Parameter(bias)
        if self.org_module[0].bias is not None:
            self.org_module[0].bias = self.org_module[0].bias.to(self.org_weight)
        self.to(self_device, self_dtype)

    def onfly_restore(self):
        if self.not_supported:
            return
        self.org_weight = self.cached_org_weight.to(self.org_weight)
        if self.cached_org_bias is not None:
            self.org_module[0].bias.data.copy_(self.cached_org_bias.to(self.org_weight))
        del self.cached_org_weight
        del self.cached_org_bias

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        return None, None

    def bypass_forward_diff(self, x, scale=1):
        raise NotImplementedError

    def bypass_forward(self, x, scale=1):
        raise NotImplementedError

    def parametrize_forward(self, x: torch.Tensor, *args, **kwargs):
        return self.get_merged_weight(
            multiplier=self.multiplier, shape=x.shape, device=x.device
        )[0].to(x.dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _prepare_merge_context(self, precise: bool) -> _MergeContext:
        module = self.org_module[0]
        weight_param = module.weight
        bias_param = module.bias

        params = tuple(self.parameters())
        first_param = params[0] if params else None
        param_device = first_param.device if first_param is not None else None
        param_dtype = first_param.dtype if first_param is not None else None

        target_device = weight_param.device
        target_dtype = weight_param.dtype
        compute_dtype = torch.float64 if precise else target_dtype

        if first_param is not None:
            self.to(device=target_device, dtype=compute_dtype)
        else:
            self.to(target_device)
            if precise:
                self.to(dtype=compute_dtype)

        if precise:
            self._ensure_precise_snapshot(module, weight_param, bias_param)
            self._load_precise_snapshot(
                module,
                weight_param,
                bias_param,
                target_device,
                compute_dtype,
            )

        return _MergeContext(
            precise=precise,
            target_device=target_device,
            target_dtype=target_dtype,
            compute_dtype=compute_dtype,
            param_device=param_device,
            param_dtype=param_dtype,
            module=module,
            weight_param=weight_param,
            bias_param=bias_param,
        )

    def _apply_merged_weights(
        self,
        ctx: _MergeContext,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        merged_weight = weight.to(ctx.target_dtype)
        ctx.weight_param.data.copy_(merged_weight)

        if bias is not None:
            merged_bias = bias.to(ctx.target_dtype)
            if ctx.bias_param is not None:
                ctx.bias_param.data.copy_(merged_bias)
            else:
                ctx.module.bias = nn.Parameter(merged_bias)
        elif ctx.bias_param is None:
            ctx.module.bias = None

        if ctx.precise:
            ctx.module._lycoris_precise_weight_current = weight.to(torch.float64).cpu()
            if ctx.bias_param is not None:
                if bias is not None:
                    ctx.module._lycoris_precise_bias_current = bias.to(
                        torch.float64
                    ).cpu()
                else:
                    ctx.module._lycoris_precise_bias_current = (
                        ctx.module._lycoris_precise_bias_base
                    )

    def _restore_merge_context(self, ctx: _MergeContext) -> None:
        if ctx.param_device is not None and ctx.param_dtype is not None:
            self.to(device=ctx.param_device, dtype=ctx.param_dtype)
        elif ctx.param_device is not None:
            self.to(ctx.param_device)
        elif ctx.param_dtype is not None:
            self.to(dtype=ctx.param_dtype)

    def _compute_precise_result(
        self, ctx: _MergeContext, multiplier: float
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_weight = ctx.module._lycoris_precise_weight_current
        diff_weight, diff_bias = self.get_diff_weight(
            multiplier=1.0, device=ctx.target_device
        )
        diff_weight_prec = diff_weight.to(torch.float64).cpu()
        new_weight = base_weight + diff_weight_prec * multiplier

        new_bias = None
        if diff_bias is not None:
            diff_bias_prec = diff_bias.to(torch.float64).cpu()
            base_bias = ctx.module._lycoris_precise_bias_current
            if base_bias is None:
                base_bias = torch.zeros_like(diff_bias_prec)
            new_bias = base_bias + diff_bias_prec * multiplier
        else:
            new_bias = ctx.module._lycoris_precise_bias_current

        ctx.module._lycoris_precise_weight_current = new_weight.clone()
        if diff_bias is not None:
            ctx.module._lycoris_precise_bias_current = (
                new_bias.clone() if new_bias is not None else None
            )

        return new_weight, new_bias

    def _apply_precise_weights(
        self,
        ctx: _MergeContext,
        weight_prec: torch.Tensor,
        bias_prec: torch.Tensor | None,
    ) -> None:
        ctx.weight_param.data.copy_(weight_prec.to(ctx.target_device, ctx.target_dtype))

        if bias_prec is not None:
            if ctx.bias_param is not None:
                ctx.bias_param.data.copy_(
                    bias_prec.to(ctx.target_device, ctx.target_dtype)
                )
            else:
                ctx.module.bias = nn.Parameter(
                    bias_prec.to(ctx.target_device, ctx.target_dtype)
                )
        elif ctx.bias_param is None:
            ctx.module.bias = None

    @staticmethod
    def _ensure_precise_snapshot(
        module: nn.Module,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        if not hasattr(module, "_lycoris_precise_weight_base"):
            base = weight.detach().cpu().double()
            module._lycoris_precise_weight_base = base
            module._lycoris_precise_weight_current = base.clone()
        if not hasattr(module, "_lycoris_precise_weight_current"):
            module._lycoris_precise_weight_current = (
                module._lycoris_precise_weight_base.clone()
            )

        if not hasattr(module, "_lycoris_precise_bias_base"):
            if bias is not None:
                base_bias = bias.detach().cpu().double()
            else:
                base_bias = None
            module._lycoris_precise_bias_base = base_bias
            module._lycoris_precise_bias_current = (
                base_bias.clone() if base_bias is not None else None
            )
        if not hasattr(module, "_lycoris_precise_bias_current"):
            module._lycoris_precise_bias_current = (
                module._lycoris_precise_bias_base.clone()
                if module._lycoris_precise_bias_base is not None
                else None
            )

    @staticmethod
    def _load_precise_snapshot(
        module: nn.Module,
        weight_param: torch.Tensor,
        bias_param: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        weight_param.data.copy_(
            module._lycoris_precise_weight_current.to(device=device, dtype=dtype)
        )
        if bias_param is not None:
            bias_snapshot = module._lycoris_precise_bias_current
            if bias_snapshot is None and module._lycoris_precise_bias_base is not None:
                bias_snapshot = module._lycoris_precise_bias_base
                module._lycoris_precise_bias_current = bias_snapshot
            if bias_snapshot is not None:
                bias_param.data.copy_(bias_snapshot.to(device=device, dtype=dtype))
