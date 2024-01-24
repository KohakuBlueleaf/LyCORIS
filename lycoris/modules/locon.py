import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModuleCustomSD


class LoConModule(ModuleCustomSD):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.tucker = False

        if isinstance(org_module, nn.Conv2d):
            self.isconv = True
            # For general LoCon
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            self.down_op = F.conv2d
            self.up_op = F.conv2d
            if use_tucker and k_size != (1, 1):
                self.lora_down = nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
                self.lora_mid = nn.Conv2d(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.lora_down = nn.Conv2d(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.lora_up = nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
        elif isinstance(org_module, nn.Linear):
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
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
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.scalar = torch.tensor(1.0)
        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        else:
            torch.nn.init.constant_(self.lora_up.weight, 0)
        if self.tucker:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.org_forward = self.org_module[0].forward

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        self.scalar = nn.Parameter(torch.ones_like(self.scalar))

    def apply_to(self, is_hypernet=False, **kwargs):
        self.org_forward = self.org_module[0].forward
        if is_hypernet:
            self.org_module[0].forward = self.hypernet_forward
        else:
            self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=1.0):
        weight = self.make_weight() * self.scale * multiplier
        self.org_module[0].weight.data.add_(weight)

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        return (wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)).view(
            self.shape
        ) * self.scalar

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        destination["lora_up.weight"] = self.lora_up.weight * self.scalar
        destination["lora_down.weight"] = self.lora_down.weight
        if self.tucker:
            destination["lora_mid.weight"] = self.lora_mid.weight
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device).norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = ratio.item() != 1.0
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio

    def update_weights(self, down, up, idx):
        self.down, self.up = self.make_lightweight(down.squeeze(1), up.squeeze(1), idx)

    def make_lightweight(self, down, up, seed=None, down_aux=None, up_aux=None):
        if down.dim() == 3:
            down = down.reshape(down.size(0), self.lora_dim, -1)
            up = up.reshape(up.size(0), -1, self.lora_dim)
        else:
            down = down.reshape(self.lora_dim, -1)
            up = up.reshape(-1, self.lora_dim)
        # print(up.shape)
        if seed is None:
            assert down_aux is not None and up_aux is not None
        else:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            if down_aux is None or up_aux is None:
                down_aux = torch.empty(
                    down.size(down.dim() - 1),
                    self.lora_down.weight.size(1),
                    device=down.device,
                )
                up_aux = torch.empty(
                    self.lora_up.weight.size(0), up.size(up.dim() - 2), device=up.device
                )
                nn.init.orthogonal_(down_aux)
                nn.init.orthogonal_(up_aux)
                # print(up_aux.shape)
            torch.set_rng_state(rng_state)
        if down.dim() == 3 and down.size(0) == 1:
            down = down.squeeze(0)
        if up.dim() == 3 and up.size(0) == 1:
            up = up.squeeze(0)
        down = down + 1  # avoid zero grad or slow training, give it a constant
        return (down @ down_aux), (up_aux @ up)

    def apply_lightweight(self, down, up, seed=None, down_aux=None, up_aux=None):
        down_weight, up_weight = self.make_lightweight(down, up, seed, down_aux, up_aux)
        self.lora_down.weight.data = down_weight
        self.lora_up.weight.data = up_weight
        return down_weight, up_weight

    def hypernet_forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier

        down_weight = self.down
        up_weight = self.up

        x_batch = None
        if down_weight.dim() == 3:
            if x.size(0) != down_weight.size(0):
                assert (
                    self.isconv == False
                ), "Convolutional hypernet with batch size mismatch is not supported"
                x_batch = x.size(0)
                x = x.view(down_weight.size(0), -1, *x.shape[1:])

            if self.isconv:
                mid = torch.einsum("ijk, ik... -> ij...", down_weight, x)
            else:
                mid = torch.einsum("ijk, i...k -> i...j", down_weight, x)
        else:
            if self.isconv:
                weight = down_weight.unsqueeze(-1).unsqueeze(-1)
            else:
                weight = down_weight
            mid = self.down_op(x, weight)

        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) < self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if (dims := len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1] * (dims - 1), -1)
            mid = mid * drop

        if up_weight.dim() == 3:
            mid_batch = None
            if mid.size(0) != up_weight.size(0):
                assert (
                    self.isconv == False
                ), "Convolutional hypernet with batch size mismatch is not supported"
                mid_batch = mid.size(0)
                mid = mid.view(up_weight.size(0), -1, *mid.shape[1:])

            if self.isconv:
                up = torch.einsum("ijk, ik... -> ij...", up_weight, mid)
            else:
                up = torch.einsum("ijk, i...k -> i...j", up_weight, mid)

            if mid_batch is not None:
                up = up.view(mid_batch, *up.shape[2:])
        else:
            if self.isconv:
                weight = up_weight.unsqueeze(-1).unsqueeze(-1)
            else:
                weight = up_weight
            up = self.up_op(mid, weight)

        if x_batch is not None:
            up = up.view(x_batch, *up.shape[2:])

        org_out = self.org_forward(x)
        # print(x.shape, org_out.shape, up.shape)
        return org_out + self.dropout(up) * scale

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.scale * self.multiplier
        if self.tucker:
            mid = self.lora_mid(self.lora_down(x))
        else:
            mid = self.lora_down(x)

        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if (dims := len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1] * (dims - 1), -1)
            mid = mid * drop

        return self.org_forward(x) + self.dropout(
            self.lora_up(mid) * self.scalar * scale
        )
