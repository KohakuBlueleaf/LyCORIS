"""LoKr merge backward of two full factors in TileLang on the model-ranked
tuner: (g_w1, g_w2) of gamma * kron(w1, w2) from one read of the (a*c, b*d)
grad, fp32 partials, no atomics. See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer
from ...tuner.model import Work, rank
from ..common import CompileError, compiled, tl_dtype
from . import merge

F32 = "float32"
ATOMIC = {"impl": "atomic"}
_KERNEL_DTYPES = (torch.float16, torch.float32)


def _bwd_prim(a, b, c, d, bm, bn, split, threads, g_dt, w1_dt, w2_dt):
    """One program per (w2 tile, split of the a*b blocks): the g_w2 tile partial
    sum_uv w1[u, v] * G_uv and, per block, the row sums of G_uv * w2 (g_w1)."""
    per = a * b // split
    td = d // bn

    @T.prim_func
    def main(
        g: T.Tensor((a * c, b * d), g_dt),
        w1: T.Tensor((a, b), w1_dt),
        w2: T.Tensor((c, d), w2_dt),
        p1: T.Tensor((a * b, td, c), F32),
        p2: T.Tensor((split, c, d), F32),
    ):
        # Grid: (w2 column tiles bx, w2 row tiles by, block splits bz).
        with T.Kernel(td, c // bm, split, threads=threads) as (bx, by, bz):
            w2_f = T.alloc_fragment((bm, bn), F32)
            acc = T.alloc_fragment((bm, bn), F32)
            prod = T.alloc_fragment((bm, bn), F32)
            row = T.alloc_fragment((bm,), F32)
            for i, j in T.Parallel(bm, bn):
                w2_f[i, j] = T.cast(w2[by * bm + i, bx * bn + j], F32)
            T.clear(acc)
            for k in T.serial(per):
                uv = bz * per + k
                u = uv // b
                v = uv % b
                # G_uv tile: acc += w1[u, v] * G, prod = G * w2.
                for i, j in T.Parallel(bm, bn):
                    gv = T.cast(g[u * c + by * bm + i, v * d + bx * bn + j], F32)
                    acc[i, j] += T.cast(w1[u, v], F32) * gv
                    prod[i, j] = gv * w2_f[i, j]
                T.reduce_sum(prod, row, dim=1)
                for i in T.Parallel(bm):
                    p1[uv, bx, by * bm + i] = row[i]
            for i, j in T.Parallel(bm, bn):
                p2[bz, by * bm + i, bx * bn + j] = acc[i, j]

    return main


def _plan(dev, a, b, c, d, e_g, e_w, alternatives):
    """Tilings that divide (c, d) exactly, model-ranked, then the alternatives."""
    nuv = a * b
    space = []
    for bm in (4, 8, 16, 32, 64):
        for bn in (8, 16, 32, 64, 128, 256):
            if c % bm or d % bn or bm * bn > 8192:
                continue
            for split in (1, 2, 4, 8, 16, 32, 64):
                if nuv % split:
                    continue
                for threads in (64, 128, 256):
                    if threads <= bm * bn:
                        space.append(
                            {"BM": bm, "BN": bn, "SPLIT": split, "threads": threads}
                        )

    def work(cf):
        bm, bn, split = cf["BM"], cf["BN"], cf["SPLIT"]
        td = d // bn
        # G read once; both partial sets written once and summed in torch
        partial = 4 * (split * c * d + nuv * td * c)
        return Work(
            grid=(c // bm) * td * split,
            warps=cf["threads"] // 32,
            acc_fp32=3 * bm * bn,
            operand_bytes=bm * bn * e_g,
            smem_stage=0,
            stages=1,
            hbm_bytes=nuv * c * d * e_g + 2 * partial + c * d * e_w,
            vector_flops=4.0 * nuv * c * d,
        )

    return [*rank(dev, space, work), *alternatives]


def _bwd_run(cfg, g2, w1, w2, scale):
    a, b = w1.shape
    c, d = w2.shape
    if cfg == TORCH:
        g4 = g2.reshape(a, c, b, d)
        gw1 = torch.einsum("acbd,cd->ab", g4, w2) * scale
        gw2 = torch.einsum("acbd,ab->cd", g4, w1) * scale
        return gw1.to(w1.dtype), gw2.to(w1.dtype)
    if cfg == ATOMIC:
        return merge.lokr_full_merge_bwd(g2, w1, w2, scale)
    bm, bn, split = cfg["BM"], cfg["BN"], cfg["SPLIT"]
    kernel = compiled(
        _bwd_prim,
        g2.device,
        a,
        b,
        c,
        d,
        bm,
        bn,
        split,
        cfg["threads"],
        tl_dtype(g2.dtype),
        tl_dtype(w1.dtype),
        tl_dtype(w2.dtype),
    )
    p1 = torch.empty(a * b, d // bn, c, device=g2.device, dtype=torch.float32)
    p2 = torch.empty(split, c, d, device=g2.device, dtype=torch.float32)
    kernel(g2, w1, w2, p1, p2)
    gw1 = p1.sum((1, 2)).mul_(scale).view(a, b)
    gw2 = (p2[0] if split == 1 else p2.sum(0)).mul_(scale)
    return gw1.to(w1.dtype), gw2.to(w1.dtype)


def lokr_full_merge_bwd(grad, w1, w2, gamma: float = 1.0):
    """(g_w1, g_w2) of gamma * kron(w1, w2) from the (a*c, b*d) grad, in w1's
    dtype."""
    grad, w1, w2 = grad.contiguous(), w1.contiguous(), w2.contiguous()
    alternatives = [TORCH, ATOMIC]
    if any(t.dtype not in _KERNEL_DTYPES for t in (grad, w1, w2)):
        return _bwd_run(ATOMIC, grad, w1, w2, gamma)
    dev = device_info(w2.device)
    a, b = w1.shape
    c, d = w2.shape
    key = (
        "tl.lokr.full_merge_bwd",
        (dtype_key(grad.dtype), dtype_key(w1.dtype)),
        (a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(
            dev, a, b, c, d, grad.element_size(), w2.element_size(), alternatives
        ),
        measurer(
            dev,
            grad.numel() * grad.element_size(),
            lambda: (copy_of(grad),),
            lambda cf, s: lambda: _bwd_run(cf, s[0], w1, w2, gamma),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _bwd_run(cf, grad, w1, w2, gamma),
        CompileError,
    )
