from functools import cache
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import LycorisBaseModule
from ..functional import power2factorization
from ..logging import logger
from ..utils.bnb import LinearNF4


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
            f"It is impossible to decompose {dimension} with factor {factor} under BOFT constrains."
        )

    log_butterfly_factorize(dimension, factor, (m, n))
    return m, n


class ButterflyOFTModule(LycorisBaseModule):
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }

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
        bypass_mode=False,
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
        self.constrain = constrain * out_dim
        self.register_buffer("alpha", torch.tensor(constrain))
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.boft_m, self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

    @property
    def I(self):
        return torch.eye(self.block_size, device=self.device)

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

    def make_weight(self, scale=1, device=None, diff=False):
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        inp = self.org_weight.to(device, dtype=r.dtype)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            if i == 0:
                # Apply multiplier/scale and rescale into first weight
                if self.rescaled:
                    bi = bi * self.rescale
            bi = bi * scale - scale * self.I + (self.I if diff else 0)
            inp = rearrange(inp, "(c g k) ... -> (c k g) ...", g=2, k=2**i * r_b)
            inp = rearrange(inp, "(d b) ... -> d b ...", b=b)
            inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
            inp = rearrange(inp, "d b ... -> (d b) ...")
            inp = rearrange(inp, "(c k g) ... -> (c g k) ...", g=2, k=2**i * r_b)

        return inp

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
        inp = self.org_forward(x)

        for i in range(m):
            bi = r[i]  # b_num, b_size, b_size
            if i == 0:
                # Apply multiplier/scale and rescale into first weight
                if self.rescaled:
                    bi = bi * self.rescale
            bi = bi * scale - scale * self.I + (self.I if diff else 0)
            inp = rearrange(inp, "... (c g k) ->... (c k g)", g=2, k=2**i * r_b)
            inp = rearrange(inp, "... (d b) -> ... d b", b=b)
            inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
            inp = rearrange(inp, "... d b -> ... (d b)")
            inp = rearrange(inp, "... (c k g) -> ... (c g k)", g=2, k=2**i * r_b)

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


if __name__ == "__main__":
    device = torch.device("cuda")
    module = ButterflyOFTModule
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        base = nn.Linear(128, 128).to(device).half()
        net = module("test", base, 1, 4, 1, weight_decompose=True).to(device)
        print(net)
        test_input = torch.randn(1, 128).to(device).half()
        test_output = net(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)

        base_4bit = LinearNF4(128, 128, device="cuda")
        base_4bit.load_state_dict(base.state_dict())
        base_4bit.to(device)
        qnet = module("test", base_4bit, 1, 4, 1, weight_decompose=False).to(device)
        print(qnet)
        test_input = torch.randn(1, 128).to(device).half()
        test_output = qnet(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)

        base = nn.Conv2d(128, 128, 3, 1, 1).to(device).half()
        net = module("test", base, 1, 4, 1, weight_decompose=True, use_tucker=True).to(
            device
        )
        print(net)
        test_input = torch.randn(1, 128, 16, 16).to(device).half()
        test_output = net(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)

        base = nn.Conv2d(128, 128, 3, 1, 1).to(device).half()
        net = module.parametrize(
            base, "weight", 1, 4, 1, weight_decompose=True, use_tucker=True
        ).to(device)
        print(base)
        test_input = torch.randn(1, 128, 16, 16).to(device).half()
        test_output = base(test_input)
        torch.sum(test_output).backward()
        print(test_output.shape)
