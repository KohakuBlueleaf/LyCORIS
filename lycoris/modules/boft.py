from functools import cache
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import LycorisBaseModule
from ..functional import power2factorization
from ..logging import logger


@cache
def log_butterfly_factorize(dim, factor, result):
    logger.info(
        f"Use BOFT({int(log2(result[1]))}, {result[0]//2})"
        f" (equivalent to factor={result[0]}) "
        f"for {dim=} and {factor=}"
    )


def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    m, n = power2factorization(dimension, factor)

    if n == 0:
        raise ValueError(
            f"It is impossible to decompose {dimension} with factor {factor} under BOFT constraints."
        )

    log_butterfly_factorize(dimension, factor, (m, n))
    return m, n


class ButterflyOFTModule(LycorisBaseModule):
    name = "boft"
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
            raise ValueError(f"{self.module_type} is not supported in BOFT algo.")

        out_dim = self.dim
        b, m_exp = butterfly_factor(out_dim, lora_dim)
        self.block_size = b
        self.block_num = m_exp
        # BOFT(m, b)
        self.boft_b = b
        self.boft_m = sum(int(i) for i in f"{m_exp-1:b}") + 1
        # block_num > block_size
        self.rescaled = rescaled
        self.constraint = constraint * out_dim
        self.register_buffer("alpha", torch.tensor(constraint))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.boft_m, self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        if f"{lora_name}.oft_blocks" in state_dict:
            oft_blocks = state_dict[f"{lora_name}.oft_blocks"]
            if oft_blocks.ndim == 4:
                return True
        return False

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, oft_blocks, rescale, alpha
    ):
        m, n, s, _ = oft_blocks.shape
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
        q = self.oft_blocks - self.oft_blocks.transpose(-1, -2)
        normed_q = q
        # Diag OFT style constrain
        if self.constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constraint:
                normed_q = q * self.constraint / q_norm
        # use float() to prevent unsupported type
        r = (I + normed_q) @ (I - normed_q).float().inverse()
        return r

    def make_weight(self, scale=1, device=None, diff=False):
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        inp = org = self.org_weight.to(device, dtype=r.dtype)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            g = 2
            k = 2**i * r_b
            if scale != 1:
                bi = bi * scale + (1 - scale) * self.I
            inp = (
                inp.unflatten(-1, (-1, g, k))
                .transpose(-2, -1)
                .flatten(-3)
                .unflatten(-1, (-1, b))
            )
            inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
            inp = (
                inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)
            )

        if self.rescaled:
            inp = inp * self.rescale

        if diff:
            inp = inp - org

        return inp.to(self.oft_blocks.dtype)

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
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        inp = org = self.org_forward(x)
        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            inp = inp.transpose(1, -1)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            g = 2
            k = 2**i * r_b
            if scale != 1:
                bi = bi * scale + (1 - scale) * self.I
            inp = (
                inp.unflatten(-1, (-1, g, k))
                .transpose(-2, -1)
                .flatten(-3)
                .unflatten(-1, (-1, b))
            )
            inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
            inp = (
                inp.flatten(-2).unflatten(-1, (-1, k, g)).transpose(-2, -1).flatten(-3)
            )

        if self.rescaled:
            inp = inp * self.rescale.transpose(0, -1)

        if self.op in {F.conv2d, F.conv1d, F.conv3d}:
            inp = inp.transpose(1, -1)

        if diff:
            inp = inp - org
        return inp

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=False)

    def forward(self, x, *args, **kwargs):
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
