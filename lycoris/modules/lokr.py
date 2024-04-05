import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .base import ModuleCustomSD
from ..logging import logger
from ..utils.bnb import LinearNF4, QuantLinears, log_bypass


@cache
def logging_force_full_matrix(lora_dim, dim, factor):
    logger.warning(
        f"lora_dim {lora_dim} is too large for"
        f" dim={dim} and {factor=}"
        ", using full matrix mode."
    )


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor
    second value is higher or equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale.
    second value is a value for weight.

    Because of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples)
    factor
        -1               2                4               8               16               ...
    127 -> 1, 127   127 -> 1, 127    127 -> 1, 127   127 -> 1, 127   127 -> 1, 127
    128 -> 8, 16    128 -> 2, 64     128 -> 4, 32    128 -> 8, 16    128 -> 8, 16
    250 -> 10, 25   250 -> 2, 125    250 -> 2, 125   250 -> 5, 50    250 -> 10, 25
    360 -> 8, 45    360 -> 2, 180    360 -> 4, 90    360 -> 8, 45    360 -> 12, 30
    512 -> 16, 32   512 -> 2, 256    512 -> 4, 128   512 -> 8, 64    512 -> 16, 32
    1024 -> 32, 32  1024 -> 2, 512   1024 -> 4, 256  1024 -> 8, 128  1024 -> 16, 64
    """

    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_weight_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j k l, i p, j r -> p r k l", t, wa, wb)  # [c, d, k1, k2]
    return rebuild2


def make_kron(w1, w2, scale):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    return rebuild * scale


class LokrModule(ModuleCustomSD):
    """
    modifed from kohya-ss/sd-scripts/networks/lora:LoRAModule
        and from KohakuBlueleaf/LyCORIS/lycoris:loha:LoHaModule
        and from KohakuBlueleaf/LyCORIS/lycoris:locon:LoconModule
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
        decompose_both=False,
        factor: int = -1,  # factorization factor
        rank_dropout_scale=False,
        weight_decompose=False,
        full_matrix=False,
        bypass_mode=False,
        rs_lora=False,
        unbalanced_factorization=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        factor = int(factor)
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.tucker = False
        self.use_w1 = False
        self.use_w2 = False
        self.full_matrix = full_matrix
        self.rs_lora = rs_lora

        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.shape = (out_dim, in_dim, *k_size)

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            if unbalanced_factorization:
                out_l, out_k = out_k, out_l
            shape = ((out_l, out_k), (in_m, in_n), *k_size)  # ((a, b), (c, d), *k_size)
            self.tucker = use_tucker and k_size != (1, 1)
            if (
                decompose_both
                and lora_dim < max(shape[0][0], shape[1][0]) / 2
                and not self.full_matrix
            ):
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(
                    torch.empty(shape[0][0], shape[1][0])
                )  # a*c, 1-mode

            if lora_dim >= max(shape[0][1], shape[1][1]) / 2 or self.full_matrix:
                if not self.full_matrix:
                    logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(
                    torch.empty(shape[0][1], shape[1][1], *k_size)
                )
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(
                    torch.empty(lora_dim, lora_dim, shape[2], shape[3])
                )
                self.lokr_w2_a = nn.Parameter(
                    torch.empty(lora_dim, shape[0][1])
                )  # b, 1-mode
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1])
                )  # d, 2-mode
            else:  # Conv2d not tucker
                # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(
                    torch.empty(lora_dim, shape[1][1] * shape[2] * shape[3])
                )
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)

            self.op = F.conv2d
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }

        else:  # Linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.shape = (out_dim, in_dim)

            in_m, in_n = factorization(in_dim, factor)
            out_l, out_k = factorization(out_dim, factor)
            if unbalanced_factorization:
                out_l, out_k = out_k, out_l
            shape = (
                (out_l, out_k),
                (in_m, in_n),
            )  # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
            # smaller part. weight scale
            if decompose_both and lora_dim < max(shape[0][0], shape[1][0]) / 2:
                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
            else:
                self.use_w1 = True
                self.lokr_w1 = nn.Parameter(
                    torch.empty(shape[0][0], shape[1][0])
                )  # a*c, 1-mode
            if lora_dim < max(shape[0][1], shape[1][1]) / 2:
                # bigger part. weight and LoRA. [b, dim] x [dim, d]
                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))

            self.op = F.linear
            self.extra_args = {}

        if isinstance(org_module, QuantLinears):
            if not bypass_mode:
                log_bypass()
            bypass_mode = True
        self.bypass_mode = bypass_mode
        assert not (
            bypass_mode and weight_decompose
        ), "bypass_mode and dora_wd cannot be used together"

        self.wd = weight_decompose
        if self.wd:
            org_weight: nn.Parameter = org_module.weight
            self.dora_norm_dims = org_weight.dim() - 1
            self.dora_scale = nn.Parameter(
                torch.norm(
                    org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                    dim=1,
                    keepdim=True,
                )
                .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(1, 0)
            ).float()

        self.dropout = dropout
        if dropout:
            print("[WARN]LoHa/LoKr haven't implemented normal dropout yet.")
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        if self.use_w2 and self.use_w1:
            # use scale = 1
            alpha = lora_dim

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.scalar = torch.tensor(1.0)

        if self.use_w2:
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

        self.multiplier = multiplier
        self.org_module = [org_module]
        self.org_forward = self.org_module[0].forward
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            (
                self.lokr_w2
                if self.use_w2
                else (
                    make_weight_tucker(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
                    if self.tucker
                    else self.lokr_w2_a @ self.lokr_w2_b
                )
            ),
            torch.tensor(self.multiplier * self.scale),
        )
        assert torch.sum(torch.isnan(weight)) == 0, "weight is nan"

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.scalar = torch.ones_like(self.scalar)

    # Same as locon.py
    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        self.org_module[0].forward = self.org_forward

    def merge_to(self, multiplier=1.0):
        weight = self.get_weight(self.shape)
        self.org_module[0].weight.data.add_(weight * multiplier)

    def get_weight(self, shape):
        weight = make_kron(
            self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b,
            (
                self.lokr_w2
                if self.use_w2
                else (
                    make_weight_tucker(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
                    if self.tucker
                    else self.lokr_w2_a @ self.lokr_w2_b
                )
            ),
            torch.tensor(self.scale),
        )
        if shape is not None:
            weight = weight.reshape(shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0)) > self.rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop
        return weight

    def apply_weight_decompose(self, weight):
        weight = weight.to(self.dora_scale.dtype)
        weight_norm = (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        ) + torch.finfo(weight.dtype).eps

        return weight * (self.dora_scale.to(weight.device) / weight_norm)

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        if self.use_w1:
            destination["lokr_w1"] = self.lokr_w1 * self.scalar
        else:
            destination["lokr_w1_a"] = self.lokr_w1_a * self.scalar
            destination["lokr_w1_b"] = self.lokr_w1_b

        if self.use_w2:
            destination["lokr_w2"] = self.lokr_w2
        else:
            destination["lokr_w2_a"] = self.lokr_w2_a
            destination["lokr_w2_b"] = self.lokr_w2_b
            if self.tucker:
                destination["lokr_t2"] = self.lokr_t2
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.get_weight(self.shape).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            modules = 4 - self.use_w1 - self.use_w2 + (not self.use_w2 and self.tucker)
            if self.use_w1:
                self.lokr_w1 *= ratio ** (1 / modules)
            else:
                self.lokr_w1_a *= ratio ** (1 / modules)
                self.lokr_w1_b *= ratio ** (1 / modules)

            if self.use_w2:
                self.lokr_w2 *= ratio ** (1 / modules)
            else:
                if self.tucker:
                    self.lokr_t2 *= ratio ** (1 / modules)
                self.lokr_w2_a *= ratio ** (1 / modules)
                self.lokr_w2_b *= ratio ** (1 / modules)

        return scaled, orig_norm * ratio

    def bypass_forward(self, h):
        if isinstance(self.org_module[0], nn.Conv2d):
            h = h.transpose(1, -1)
        if self.use_w2:
            ba = self.lokr_w2
        else:
            a = self.lokr_w2_b
            b = self.lokr_w2_a
        if self.use_w1:
            c = self.lokr_w1
        else:
            c = self.lokr_w1_a @ self.lokr_w1_b
        uq = c.size(1)

        if self.use_w2:
            vq = ba.size(1)
            h_in_group = rearrange(h, "b ... (uq vq) -> b ... uq vq", uq=uq, vq=vq)
            hb = F.linear(h_in_group, ba)
        else:
            vq = a.size(1)
            h_in_group = rearrange(h, "b ... (uq vq) -> b ... uq vq", uq=uq, vq=vq)
            ha = F.linear(h_in_group, a)
            hb = F.linear(ha, b)

        h_cross_group = hb.transpose(-1, -2)
        hc = F.linear(h_cross_group, c)

        h = rearrange(hc, "b ... vp up -> b ... (up vp)")
        if isinstance(self.org_module[0], nn.Conv2d):
            h = h.transpose(1, -1)

        return h

    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.op(
                    x,
                    self.org_module[0].weight.data,
                    (
                        None
                        if self.org_module[0].bias is None
                        else self.org_module[0].bias.data
                    ),
                )
        if self.bypass_mode:
            if len(self.shape) > 2 and self.shape[2] > 1:
                return self.org_forward(x) + self.op(
                    x, self.get_weight(self.shape), **self.extra_args
                )
            else:
                return self.org_forward(x) + self.bypass_forward(x)
        else:
            weight = (
                self.org_module[0].weight.data.to(
                    x.device, dtype=next(self.parameters()).dtype
                )
                + self.get_weight(self.shape) * self.scalar * self.multiplier
            )
            bias = (
                None
                if self.org_module[0].bias is None
                else self.org_module[0].bias.data
            )

            if self.wd:
                weight = self.apply_weight_decompose(weight)
            return self.op(x, weight.view(self.shape), bias, **self.extra_args)


if __name__ == "__main__":
    base = nn.Linear(128, 128).cuda()
    lokr = LokrModule("test", base, 1, 4, 1, weight_decompose=True, factor=8).cuda()
    print(lokr)
    test_input = torch.randn(1, 77, 128).cuda()
    test_output = lokr(test_input)
    print(test_output.shape)

    # opt = torch.optim.AdamW(lokr.parameters(), lr=1e-2)
    # for _ in range(100):
    #     x = torch.randn(128, 128).cuda()
    #     t = x / 10
    #     y = lokr(x)
    #     loss = F.mse_loss(y, t)
    #     loss.backward()
    #     opt.step()
    #     opt.zero_grad()
    #     print(loss.item())

    base_4bit = LinearNF4(128, 128)
    base_4bit.load_state_dict(base.state_dict())
    base_4bit.cuda()
    qlocon = LokrModule("test", base_4bit, 1, 4, 1, weight_decompose=False).cuda()
    print(qlocon)
    test_output2 = qlocon(test_input)
    print(test_output2.shape)
    print(F.mse_loss(test_output, test_output2))

    base = nn.Conv2d(128, 128, 3, 1, 1)
    lokr = LokrModule("test", base, 1, 4, 1, weight_decompose=True, use_tucker=True)
    print(lokr)
    test_input = torch.randn(1, 128, 16, 16)
    test_output = lokr(test_input)
    print(test_output.shape)
