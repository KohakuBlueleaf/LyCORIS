from functools import cache
from math import log2, ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import ModuleCustomSD
from lycoris.logging import logger


@cache
def log_butterfly_factorize(dim, factor, result):
    logger.info(
        f"Use BOFT({int(log2(result[1]))}, {result[0]//2})"
        f" (equivalent to factor={result[0]}) "
        f"for {dim=} and {factor=}"
    )


def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    m = 2k
    n = 2**p
    m*n = dim
    """

    # Find the first solution and check if it is even doable
    m = n = 0
    while m <= factor:
        m += 2
        while dimension % m != 0 and m < dimension:
            m += 2
        if m > factor:
            break
        if sum(int(i) for i in f"{dimension//m:b}") == 1:
            n = dimension // m

    if n == 0:
        raise ValueError(
            f"It is impossible to decompose {dimension} with factor {factor} under BOFT constrains."
        )

    log_butterfly_factorize(dimension, factor, (dimension // n, n))
    return dimension // n, n


class ButterflyOFTModule(ModuleCustomSD):
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
        constrain=0,
        rescaled=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
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
        b, m_exp = butterfly_factor(out_dim, lora_dim)
        self.block_size = b
        self.block_num = m_exp
        # BOFT(m, b)
        self.boft_b = b
        self.boft_m = sum(int(i) for i in f"{m_exp-1:b}") + 1
        # block_num > block_size
        self.rescaled = rescaled
        self.constrain = constrain * out_dim
        self.register_buffer("alpha", torch.tensor(constrain))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.boft_m, self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.org_forward = self.org_module[0].forward

    @property
    def I(self):
        return torch.eye(self.block_size, device=next(self.parameters()).device)

    def apply_to(self, **kwargs):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=1.0):
        weight = self.make_weight(scale=multiplier)
        self.org_module[0].weight.data.copy_(weight)

    def get_r(self):
        I = self.I
        # for Q = -Q^T
        q = self.oft_blocks - self.oft_blocks.transpose(-1, -2)
        normed_q = q
        # Diag OFT style constrain
        if self.constrain > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > self.constrain:
                normed_q = q * self.constrain / q_norm
        # use float() to prevent unsupported type
        r = (I + normed_q) @ (I - normed_q).float().inverse()
        return r

    def make_weight(self, scale=1, device=None):
        if self.rank_dropout and self.training:
            drop = torch.rand(self.dim, device=device) < self.rank_dropout
            drop = drop.to(self.oft_blocks.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
        else:
            drop = 1

        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        inp = self.org_module[0].weight.to(device, dtype=r.dtype)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            if i == 0:
                # Apply multiplier/scale and rescale into first weight
                bi = bi * scale + (1 - scale) * self.I
                if self.rescaled:
                    bi = bi * self.rescale
            inp = rearrange(inp, "(c g k) ... -> (c k g) ...", g=2, k=2**i * r_b)
            inp = rearrange(inp, "(d b) ... -> d b ...", b=b)
            inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
            inp = rearrange(inp, "d b ... -> (d b) ...")
            inp = rearrange(inp, "(c k g) ... -> (c g k) ...", g=2, k=2**i * r_b)

        weight = inp
        if self.rescaled:
            weight = self.rescale * weight
        return weight * drop

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.oft_blocks.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired / norm

        scaled = ratio.item() != 1.0
        if scaled:
            self.oft_blocks *= ratio

        return scaled, orig_norm * ratio

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        scale = self.multiplier

        w = self.make_weight(scale, x.device)
        kw_dict = self.kw_dict | {"weight": w, "bias": self.org_module[0].bias}
        return self.op(x, **kw_dict)


if __name__ == "__main__":
    dim = 320
    test_layer = nn.Linear(dim, dim)
    test_layer.weight = nn.Parameter(torch.eye(dim))
    test_boft = ButterflyOFTModule("test", test_layer, lora_dim=16)
    test_boft.oft_blocks = nn.Parameter(torch.randn_like(test_boft.oft_blocks))

    print(test_boft.make_weight(1, "cpu").shape)
    print(torch.count_nonzero(test_boft.make_weight(1, "cpu")), dim * dim)
