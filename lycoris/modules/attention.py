import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from lycoris.utils import default
from lycoris.utils.xformers_utils import XFORMERS_AVAIL, memory_efficient_attention


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(project_in, nn.Linear(inner_dim, dim_out))
        # nn.init.constant_(self.net[-1].weight, 0)
        # nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)


def vanilla_attention(q, k, v, mask, scale=None):
    if scale is None:
        scale = math.sqrt(q.size(-1))
    scores = torch.bmm(q, k.transpose(-1, -2)) / scale
    if mask is not None:
        mask = rearrange(mask, "b ... -> b (...)")
        max_neg_value = -torch.finfo(scores.dtype).max
        mask = repeat(mask, "b j -> (b h) j", h=q.size(-3))
        scores = scores.masked_fill(~mask, max_neg_value)
    p_attn = F.softmax(scores, dim=-1)
    return torch.bmm(p_attn, v)


MEMORY_LAYOUTS = {
    "torch": (
        "b n (h d) -> b h n d",
        "b h n d -> b n (h d)",
        lambda x: (1, x, 1, 1),
    ),
    "xformers": (
        "b n (h d) -> b n h d",
        "b n h d -> b n (h d)",
        lambda x: (1, 1, x, 1),
    ),
    "vanilla": (
        "b n (h d) -> b h n d",
        "b h n d -> b n (h d)",
        lambda x: (1, x, 1, 1),
    ),
}
ATTN_FUNCTION = {
    "vanilla": vanilla_attention,
    "torch": getattr(F, "scaled_dot_product_attention", None),
    "xformers": memory_efficient_attention,
}


class Attention(nn.Module):
    """
    Attention Class without norm and residual
    (You need to wrap them by your self)
    """

    def __init__(
        self,
        in_ch,
        context_ch=None,
        heads=8,
        head_ch=64,
        self_cross=False,
        double_attn=False,
        single_kv_head=False,
        attn_backend="torch",
        cosine_attn=False,
        qk_head_ch=-1,
    ):
        super().__init__()
        if heads == -1:
            assert in_ch % head_ch == 0
            heads = in_ch // head_ch
        if head_ch == -1:
            assert in_ch % heads == 0
            head_ch = in_ch // heads
        if qk_head_ch == -1:
            qk_head_ch = head_ch
        q_ch = heads * qk_head_ch
        k_ch = (1 if single_kv_head else heads) * qk_head_ch
        v_ch = (1 if single_kv_head else heads) * head_ch
        inner_ch = heads * head_ch
        assert inner_ch == in_ch
        use_context = context_ch is not None
        context_ch = default(context_ch, in_ch)

        if attn_backend == "xformers":
            assert XFORMERS_AVAIL
        if attn_backend == "torch":
            assert torch.version.__version__ >= "2.0.0"

        self.heads = heads
        self.self_cross = self_cross
        self.double_attn = double_attn
        self.single_kv_head = single_kv_head
        self.attn = ATTN_FUNCTION[attn_backend]
        self.memory_layout = MEMORY_LAYOUTS[attn_backend]
        self.cosine_attn = cosine_attn

        if cosine_attn:
            self.scale = nn.Parameter(
                torch.ones(MEMORY_LAYOUTS[attn_backend][2](heads))
            )
        else:
            self.scale = None

        self.q = nn.Linear(in_ch, q_ch, bias=False)
        if self_cross and use_context:
            self.k = nn.Linear(in_ch, k_ch, bias=False)
            self.v = nn.Linear(in_ch, v_ch, bias=False)
            self.ck = nn.Linear(context_ch, k_ch, bias=False)
            self.cv = nn.Linear(context_ch, v_ch, bias=False)
        else:
            assert double_attn == False
            self.k = nn.Linear(context_ch, k_ch, bias=False)
            self.v = nn.Linear(context_ch, v_ch, bias=False)

        if double_attn:
            self.out = nn.Linear(inner_ch * 2, in_ch)
        else:
            self.out = nn.Linear(inner_ch, in_ch)

    def forward(self, x: torch.Tensor, context=None, mask=None):
        # Input Projection
        heads = self.heads
        q = self.q(x)
        ck = cv = None
        if self.self_cross:
            k = self.k(x)
            v = self.v(x)
            if context is not None:
                ck = self.ck(context)
                cv = self.cv(context)
                if not self.double_attn:
                    k = torch.concat([k, ck], dim=1)
                    v = torch.concat([v, cv], dim=1)
        else:
            ctx = default(context, x)
            k = self.k(ctx)
            v = self.v(ctx)

        # Rearrange for Attention
        q = rearrange(q, self.memory_layout[0], h=heads)
        if self.single_kv_head:
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            b, _, seq, _ = k.shape
            k = k.expand(b, heads, seq, k.size(3))
            v = v.expand(b, heads, seq, v.size(3))
        else:
            k = rearrange(k, self.memory_layout[0], h=heads)
            v = rearrange(v, self.memory_layout[0], h=heads)

        if self.cosine_attn:
            q = (F.normalize(q, dim=-1) * math.sqrt(q.size(-1))).to(v.dtype)
            k = (F.normalize(k, dim=-1) * self.scale).to(v.dtype)
            if ck is not None and self.double_attn:
                ck = (F.normalize(ck, dim=-1) * self.scale).to(v.dtype)

        # Attention
        out = self.attn(q.contiguous(), k.contiguous(), v.contiguous(), mask)
        out = rearrange(out, self.memory_layout[1], h=heads)
        if self.double_attn:
            out2 = self.attn(q.contiguous(), ck.contiguous(), cv.contiguous(), mask)
            out2 = rearrange(out2, self.memory_layout[1], h=heads)
            out = torch.cat([out, out2], dim=-1)

        # Output Projection
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        context_dim=None,
        gated_ff=True,
        self_cross=False,
        single_kv_head=False,
        attn_backend="torch",
        cosine_attn=False,
        qk_head_ch=-1,
        disable_self_attn=False,
        single_attn=False,
    ):
        super().__init__()
        self.single_attn = single_attn
        self.disable_self_attn = disable_self_attn or single_attn

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            dim,
            context_dim if self.disable_self_attn else None,
            n_heads,
            d_head,
            self_cross,
            single_kv_head,
            attn_backend,
            cosine_attn,
            qk_head_ch,
        )
        if not single_attn:
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = Attention(
                dim,
                context_dim,
                n_heads,
                d_head,
                self_cross,
                single_kv_head,
                attn_backend,
                cosine_attn,
                qk_head_ch,
            )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, glu=gated_ff)

    def forward(self, x, context=None):
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            + x
        )
        if not self.single_attn and (context is not None or self.attn2.self_cross):
            x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
