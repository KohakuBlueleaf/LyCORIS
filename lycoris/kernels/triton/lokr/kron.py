"""LoKr merge of two full factors, out = [base +] gamma * kron(w1, w2), on the
model-ranked tuner, forward and backward. See docs/kernels/designs.md.
"""

import torch
import triton
import triton.language as tl

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, grid_configs, guarded, launch, measurer
from ...tuner.model import Work, cdiv, rank, shared_reads
from ..common import OutOfResources, load2d
from . import merge

TILED = {"impl": "tiled"}


@triton.jit
def _merge_fwd_kernel(
    w1_ptr,
    w2_ptr,
    w_ptr,
    out_ptr,
    n_out,
    n_in,
    n_c,
    n_d,
    s1u,
    s1v,
    s2c,
    s2d,
    swo,
    swi,
    soo,
    soi,
    scale,
    add_base: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """out = [w +] scale * kron(w1, w2), one program per output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ro = pid_m * block_m + tl.arange(0, block_m)
    ri = pid_n * block_n + tl.arange(0, block_n)
    mo = ro < n_out
    mi = ri < n_in
    omask = mo[:, None] & mi[None, :]

    # Element (u*n_c + c, v*n_d + d) = w1[u, v] * w2[c, d], gathered per element.
    u, c = ro // n_c, ro % n_c
    v, d = ri // n_d, ri % n_d
    w1 = load2d(w1_ptr, u, v, s1u, s1v, mo, mi)
    w2 = load2d(w2_ptr, c, d, s2c, s2d, mo, mi)

    # Epilogue: scale, add the weight tile, store.
    out = w1.to(tl.float32) * w2.to(tl.float32) * scale
    if add_base:
        out += load2d(w_ptr, ro, ri, swo, swi, mo, mi).to(tl.float32)
    tl.store(
        out_ptr + ro[:, None] * soo + ri[None, :] * soi,
        out.to(out_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _merge_bwd_kernel(
    g_ptr,
    w1_ptr,
    w2_ptr,
    gw1_ptr,
    gw2_ptr,
    n_a,
    n_b,
    n_c,
    n_d,
    sgo,
    sgi,
    s1u,
    s1v,
    s2c,
    s2d,
    scale,
    tiles_c,
    tiles_d,
    split: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """g_w1[u, v] per program (first n_a*n_b), then fp32 g_w2 tile partials per split."""
    pid = tl.program_id(0)

    # g_w1[u, v] = scale * sum over the (c, d) block of g * w2.
    if pid < n_a * n_b:
        u = pid // n_b
        v = pid % n_b
        acc = tl.zeros((block_m, block_n), tl.float32)
        for c0 in range(0, n_c, block_m):
            rc = c0 + tl.arange(0, block_m)
            for d0 in range(0, n_d, block_n):
                rd = d0 + tl.arange(0, block_n)
                m = (rc < n_c)[:, None] & (rd < n_d)[None, :]
                g = tl.load(
                    g_ptr
                    + (u * n_c + rc)[:, None] * sgo
                    + (v * n_d + rd)[None, :] * sgi,
                    mask=m,
                    other=0.0,
                )
                w2 = load2d(w2_ptr, rc, rd, s2c, s2d, rc < n_c, rd < n_d)
                acc += g.to(tl.float32) * w2.to(tl.float32)
        tl.store(
            gw1_ptr + u * n_b + v, (tl.sum(acc) * scale).to(gw1_ptr.dtype.element_ty)
        )
    # g_w2 tile (c, d) partial = sum over this split's (u, v) blocks of w1[u, v] * g block.
    else:
        job = pid - n_a * n_b
        tile = job % (tiles_c * tiles_d)
        part = job // (tiles_c * tiles_d)
        rc = (tile // tiles_d) * block_m + tl.arange(0, block_m)
        rd = (tile % tiles_d) * block_n + tl.arange(0, block_n)
        m = (rc < n_c)[:, None] & (rd < n_d)[None, :]
        acc = tl.zeros((block_m, block_n), tl.float32)
        per = tl.cdiv(n_a * n_b, split)
        for uv in range(part * per, tl.minimum(part * per + per, n_a * n_b)):
            u = uv // n_b
            v = uv % n_b
            w1 = tl.load(w1_ptr + u * s1u + v * s1v).to(tl.float32)
            g = tl.load(
                g_ptr + (u * n_c + rc)[:, None] * sgo + (v * n_d + rd)[None, :] * sgi,
                mask=m,
                other=0.0,
            )
            acc += w1 * g.to(tl.float32)
        tl.store(
            gw2_ptr + (part * n_c + rc)[:, None] * n_d + rd[None, :],
            acc * scale,
            mask=m,
        )


def _plan(dev, a, b, c, d, e_w, e_f):
    o, i = a * c, b * d
    space = grid_configs(
        dev,
        BLOCK_M=(8, 16, 32, 64),
        BLOCK_N=(32, 64, 128, 256),
        num_warps=(1, 2, 4, 8),
        num_stages=(1,),
    )

    def work(cf):
        bm, bn = cf["BLOCK_M"], cf["BLOCK_N"]
        return Work(
            grid=cdiv(o, bm) * cdiv(i, bn),
            warps=cf["num_warps"],
            acc_fp32=0,
            operand_bytes=bm * bn * (2 * e_w + 2 * e_f),
            smem_stage=0,
            stages=1,
            hbm_bytes=2 * o * i * e_w
            + shared_reads(dev, (a * b + c * d) * e_f, cdiv(o, bm) * cdiv(i, bn)),
            vector_flops=2.0 * o * i,
            # every output element gathers one w1 and one w2 value
            l2_bytes=2 * o * i * (e_w + e_f),
        )

    return [*rank(dev, space, work), TORCH, TILED]


def _plan_bwd(dev, a, b, c, d, e_g, e_f):
    o, i = a * c, b * d
    space = grid_configs(
        dev,
        BLOCK_M=(16, 32, 64),
        BLOCK_N=(32, 64, 128),
        SPLIT=(1, 4, 16, 64),
        num_warps=(2, 4, 8),
        num_stages=(1,),
    )

    def work(cf):
        bm, bn, split = cf["BLOCK_M"], cf["BLOCK_N"], min(cf["SPLIT"], a * b)
        tiles = cdiv(c, bm) * cdiv(d, bn)
        return Work(
            grid=a * b + tiles * split,
            warps=cf["num_warps"],
            acc_fp32=bm * bn,
            operand_bytes=bm * bn * (e_g + e_f),
            smem_stage=0,
            stages=1,
            hbm_bytes=2 * o * i * e_g
            + 2 * split * c * d * 4
            + shared_reads(dev, (a * b + c * d) * e_f, a * b),
            vector_flops=4.0 * o * i,
            # the w1 programs read w2 whole each; the w2 programs gather w1 per element
            l2_bytes=2 * o * i * (e_g + e_f) + 2 * split * c * d * 4,
        )

    return [*rank(dev, space, work), TORCH, TILED]


def _run(cfg, w1, w2, scale, base, out_dtype):
    a, b = w1.shape
    c, d = w2.shape
    o, i = a * c, b * d
    if cfg == TORCH:
        delta = torch.kron(w1, w2) * scale
        return (delta if base is None else base + delta).to(out_dtype)
    if cfg == TILED:
        return merge.lokr_full_merge_fwd(w1, w2, base, scale, out_dtype)
    out = torch.empty(o, i, device=w2.device, dtype=out_dtype)
    w = base if base is not None else out
    grid = (triton.cdiv(o, cfg["BLOCK_M"]), triton.cdiv(i, cfg["BLOCK_N"]))
    _merge_fwd_kernel[grid](
        w1,
        w2,
        w,
        out,
        o,
        i,
        c,
        d,
        *w1.stride(),
        *w2.stride(),
        *w.stride(),
        *out.stride(),
        scale,
        add_base=base is not None,
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        **launch(cfg),
    )
    return out


def lokr_full_merge_fwd(w1, w2, base=None, gamma: float = 1.0, out_dtype=None):
    """gamma * kron(w1, w2) [+ base] stored in ``out_dtype`` (default w2's)."""
    dev = device_info(w2.device)
    a, b = w1.shape
    c, d = w2.shape
    out_dtype = out_dtype or w2.dtype
    e_w = torch.finfo(out_dtype).bits // 8
    key = (
        "lokr.full_merge_fwd",
        (
            dtype_key(w1.dtype),
            dtype_key(w2.dtype),
            dtype_key(out_dtype),
            base is not None,
        ),
        (a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, a, b, c, d, e_w, w2.element_size()),
        measurer(
            dev,
            2 * a * b * c * d * e_w,
            lambda: (copy_of(base),),
            lambda cf, s: lambda: _run(cf, w1, w2, gamma, s[0], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _run(cf, w1, w2, gamma, base, out_dtype),
        OutOfResources,
    )


def _bwd_run(cfg, g2, w1, w2, scale):
    a, b = w1.shape
    c, d = w2.shape
    if cfg == TORCH:
        g4 = g2.reshape(a, c, b, d)
        gw1 = torch.einsum("acbd,cd->ab", g4, w2) * scale
        gw2 = torch.einsum("acbd,ab->cd", g4, w1) * scale
        return gw1.to(w1.dtype), gw2.to(w1.dtype)
    if cfg == TILED:
        return merge.lokr_full_merge_bwd(g2, w1, w2, scale)
    gw1 = torch.empty(a, b, device=w1.device, dtype=w1.dtype)
    split = min(cfg["SPLIT"], a * b)
    part = torch.empty(split, c, d, device=w2.device, dtype=torch.float32)
    nc, nd = triton.cdiv(c, cfg["BLOCK_M"]), triton.cdiv(d, cfg["BLOCK_N"])
    _merge_bwd_kernel[(a * b + nc * nd * split,)](
        g2,
        w1,
        w2,
        gw1,
        part,
        a,
        b,
        c,
        d,
        *g2.stride(),
        *w1.stride(),
        *w2.stride(),
        scale,
        nc,
        nd,
        split=split,
        block_m=cfg["BLOCK_M"],
        block_n=cfg["BLOCK_N"],
        **launch(cfg),
    )
    return gw1, part.sum(0).to(w1.dtype)


def lokr_full_merge_bwd(grad, w1, w2, gamma: float = 1.0):
    """(g_w1, g_w2) of gamma * kron(w1, w2) from the (a*c, b*d) grad, in w1's
    dtype."""
    dev = device_info(w2.device)
    a, b = w1.shape
    c, d = w2.shape
    key = (
        "lokr.full_merge_bwd",
        (dtype_key(grad.dtype), dtype_key(w1.dtype)),
        (a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_bwd(dev, a, b, c, d, grad.element_size(), w2.element_size()),
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
        OutOfResources,
    )
