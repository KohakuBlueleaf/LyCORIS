from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional import factorization
from ..logging import logger


@cache
def log_oft_factorize(dim, factor, num, bdim):
    logger.info(
        f"Use OFT(block num: {num}, block dim: {bdim})"
        f" (equivalent to lora_dim={num}) "
        f"for {dim=} and lora_dim={factor=}"
    )


class DiagOFTModule(LycorisBaseModule):
    name = "diag-oft"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "oft_blocks",
        "rescale",
        "alpha",
    ]
    weight_list_det = ["oft_blocks"]

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
        constraint=0,
        rescaled=False,
        bypass_mode=None,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in Diag-OFT algo.")

        out_dim = self.dim
        self.block_size, self.block_num = factorization(out_dim, lora_dim)
        # block_num > block_size
        self.rescaled = rescaled
        self.constraint = constraint * out_dim
        self.register_buffer("alpha", torch.tensor(constraint))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

        log_oft_factorize(
            dim=out_dim,
            factor=lora_dim,
            num=self.block_num,
            bdim=self.block_size,
        )

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        if f"{lora_name}.oft_blocks" in state_dict:
            oft_blocks = state_dict[f"{lora_name}.oft_blocks"]
            if oft_blocks.ndim == 3:
                return True
        return False

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, oft_blocks, rescale, alpha
    ):
        n, s, _ = oft_blocks.shape
        module = cls(
            lora_name,
            orig_module,
            1,
            lora_dim=s,
            constraint=float(alpha),
            rescaled=rescale is not None,
        )
        module.oft_blocks.copy_(oft_blocks)
        if rescale is not None:
            module.rescale.copy_(rescale)
        return module

    @property
    def I(self):
        return torch.eye(self.block_size, device=self.device)

    def get_r(self):
        I = self.I
        # for Q = -Q^T
        q = self.oft_blocks - self.oft_blocks.transpose(1, 2)
        normed_q = q
        if self.constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constraint:
                normed_q = q * self.constraint / q_norm
        # use float() to prevent unsupported type
        r = (I + normed_q) @ (I - normed_q).float().inverse()
        return r

    def make_weight(self, scale=1, device=None, diff=False):
        r = self.get_r()
        _, *shape = self.org_weight.shape
        org_weight = self.org_weight.to(device, dtype=r.dtype)
        org_weight = org_weight.view(self.block_num, self.block_size, *shape)
        # Init R=0, so add I on it to ensure the output of step0 is original model output
        weight = torch.einsum(
            "k n m, k n ... -> k m ...",
            self.rank_drop(r * scale) - scale * self.I + (0 if diff else self.I),
            org_weight,
        ).view(-1, *shape)
        if self.rescaled:
            weight = self.rescale * weight
            if diff:
                weight = weight + (self.rescale - 1) * org_weight
        return weight.to(self.oft_blocks.dtype)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device, diff=True)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.oft_blocks.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired / norm

        scaled = norm != desired
        if scaled:
            self.oft_blocks *= ratio

        return scaled, orig_norm * ratio

    def _bypass_forward(self, x, scale=1, diff=False):
        r = self.get_r()
        org_out = self.org_forward(x)
        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            org_out = org_out.transpose(1, -1)
        *shape, _ = org_out.shape
        org_out = org_out.view(*shape, self.block_num, self.block_size)
        mask = neg_mask = 1
        if self.dropout != 0 and self.training:
            mask = torch.ones_like(org_out)
            mask = self.drop(mask)
            neg_mask = torch.max(mask) - mask
        oft_out = torch.einsum(
            "k n m, ... k n -> ... k m",
            r * scale * mask + (1 - scale) * self.I * neg_mask,
            org_out,
        )
        if diff:
            out = out - org_out
        out = oft_out.view(*shape, -1)
        if self.rescaled:
            out = self.rescale.transpose(-1, 0) * out
            out = out + (self.rescale.transpose(-1, 0) - 1) * org_out
        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            out = out.transpose(1, -1)
        return out

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier

        if self.bypass_mode:
            return self.bypass_forward(x, scale)
        else:
            w = self.make_weight(scale, x.device)
            kw_dict = self.kw_dict | {"weight": w, "bias": self.org_module[0].bias}
            return self.op(x, **kw_dict)
