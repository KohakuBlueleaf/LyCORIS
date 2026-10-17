"""LoRA bypass in TileLang on the model-ranked tuner (fp16, sm_70+).

Forward: a program owns BM tokens and 1/SPLIT of the outputs. h = x @ down^T
accumulates over the input axis in an fp32 fragment, is rounded to fp16 in
shared memory (as the two-Linear chain rounds it), then y = [base +] gamma *
h @ up^T streams over the program's output slice with the base added before
the store. The rank pads to 16 with zero rows. Backward: gx is the same kernel
on (down^T, up^T), which also stores its h (= q = g @ up); h = x @ down^T and
the factor gradients are cuBLAS GEMMs with the scale as alpha. The
single-launch kernels (``.bypass``) and the torch composition are timed
alongside; non-fp16 operands take the single-launch kernels.
See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer, token_bucket
from ...tuner.model import Work, rank, shared_reads
from ..common import (
    CompileError,
    cdiv,
    compiled,
    fits,
    pad_to,
    padded_cols,
    padded_rows,
    partitions,
    tl_dtype,
)
from . import bypass

F16 = "float16"
F32 = "float32"
MAX_RANK = 128
WHOLE = {"impl": "whole"}


def _bypass_prim(o, i, rp, bm, bn, bk, threads, split, base_dt, out_dt, store_h=False):
    """y = [base +] scale * (x @ down^T) @ up^T; a program owns bm tokens and
    1/split of the outputs; ``store_h`` also stores the fp16 h (no base)."""
    t = T.symbolic("t")
    per = cdiv(cdiv(o, bn), split)
    plain_only = base_dt is None
    base_dt = base_dt or out_dt

    @T.macro
    def hidden(x, down, bt, h_s):
        """h_s = fp16(x tile @ down^T) for token tile bt."""
        x_s = T.alloc_shared((bm, bk), F16)
        d_s = T.alloc_shared((rp, bk), F16)
        h_f = T.alloc_fragment((bm, rp), F32)
        T.clear(h_f)
        for k in T.Pipelined(T.ceildiv(i, bk), num_stages=1):
            T.copy(x[bt * bm, k * bk], x_s)
            T.copy(down[0, k * bk], d_s)
            T.gemm(x_s, d_s, h_f, transpose_B=True)
        T.copy(h_f, h_s)

    @T.prim_func
    def with_base(
        x: T.Tensor((t, i), F16),
        down: T.Tensor((rp, i), F16),
        up: T.Tensor((o, rp), F16),
        base: T.Tensor((t, o), base_dt),
        y: T.Tensor((t, o), out_dt),
        scale: T.float32,
    ):
        # Grid: (output slices bs, token tiles bt).
        with T.Kernel(split, T.ceildiv(t, bm), threads=threads) as (bs, bt):
            h_s = T.alloc_shared((bm, rp), F16)
            u_s = T.alloc_shared((bn, rp), F16)
            y_f = T.alloc_fragment((bm, bn), F32)
            b_f = T.alloc_fragment((bm, bn), base_dt)
            hidden(x, down, bt, h_s)

            # Each bn-wide output chunk of this slice: y = base + scale * h @ up^T.
            for n in T.serial(per):
                if (bs * per + n) * bn < o:
                    T.copy(up[(bs * per + n) * bn, 0], u_s)
                    T.clear(y_f)
                    T.gemm(h_s, u_s, y_f, transpose_B=True)
                    T.copy(base[bt * bm, (bs * per + n) * bn], b_f)
                    for a, b in T.Parallel(bm, bn):
                        y_f[a, b] = b_f[a, b] + scale * y_f[a, b]
                    T.copy(y_f, y[bt * bm, (bs * per + n) * bn])

    @T.prim_func
    def plain(
        x: T.Tensor((t, i), F16),
        down: T.Tensor((rp, i), F16),
        up: T.Tensor((o, rp), F16),
        y: T.Tensor((t, o), out_dt),
        scale: T.float32,
    ):
        with T.Kernel(split, T.ceildiv(t, bm), threads=threads) as (bs, bt):
            h_s = T.alloc_shared((bm, rp), F16)
            u_s = T.alloc_shared((bn, rp), F16)
            y_f = T.alloc_fragment((bm, bn), F32)
            hidden(x, down, bt, h_s)

            # Each bn-wide output chunk of this slice: y = scale * h @ up^T.
            for n in T.serial(per):
                if (bs * per + n) * bn < o:
                    T.copy(up[(bs * per + n) * bn, 0], u_s)
                    T.clear(y_f)
                    T.gemm(h_s, u_s, y_f, transpose_B=True)
                    for a, b in T.Parallel(bm, bn):
                        y_f[a, b] = scale * y_f[a, b]
                    T.copy(y_f, y[bt * bm, (bs * per + n) * bn])

    @T.prim_func
    def with_h(
        x: T.Tensor((t, i), F16),
        down: T.Tensor((rp, i), F16),
        up: T.Tensor((o, rp), F16),
        y: T.Tensor((t, o), out_dt),
        h_out: T.Tensor((t, rp), F16),
        scale: T.float32,
    ):
        with T.Kernel(split, T.ceildiv(t, bm), threads=threads) as (bs, bt):
            h_s = T.alloc_shared((bm, rp), F16)
            u_s = T.alloc_shared((bn, rp), F16)
            y_f = T.alloc_fragment((bm, bn), F32)
            hidden(x, down, bt, h_s)
            if bs == 0:
                T.copy(h_s, h_out[bt * bm, 0])

            # Each bn-wide output chunk of this slice: y = scale * h @ up^T.
            for n in T.serial(per):
                if (bs * per + n) * bn < o:
                    T.copy(up[(bs * per + n) * bn, 0], u_s)
                    T.clear(y_f)
                    T.gemm(h_s, u_s, y_f, transpose_B=True)
                    for a, b in T.Parallel(bm, bn):
                        y_f[a, b] = scale * y_f[a, b]
                    T.copy(y_f, y[bt * bm, (bs * per + n) * bn])

    if store_h:
        return with_h
    return plain if plain_only else with_base


def _smem(cfg, rp):
    bm, bn, bk = cfg["BM"], cfg["BN"], cfg["BK"]
    return bm * bk + rp * bk + bm * rp + bn * rp


def _legal(dev, cfg, rp):
    bm = cfg["BM"]
    return fits(dev, _smem(cfg, rp)) and partitions(
        dev, cfg["threads"], (bm, rp), (bm, cfg["BN"])
    )


def _plan(dev, t, o, i, r, has_base, alternatives):
    """Legal bypass configs, model-ranked, then the alternatives."""
    rp = pad_to(r)
    if rp > MAX_RANK:
        return alternatives
    space = [
        {"BM": bm, "BN": bn, "BK": bk, "threads": th, "SPLIT": s}
        for bm in (32, 64, 128)
        for bn in (64, 128)
        for bk in (32, 64)
        for th in (128, 256)
        for s in (1, 2, 4, 8)
    ]
    space = [c for c in space if c["SPLIT"] <= cdiv(o, c["BN"]) and _legal(dev, c, rp)]

    def work(c):
        bm, bn, split = c["BM"], c["BN"], c["SPLIT"]
        tiles = cdiv(t, bm)
        # every output slice rebuilds h from x
        return Work(
            grid=tiles * split,
            warps=c["threads"] // 32,
            acc_fp32=bm * rp + bm * bn,
            operand_bytes=0,
            smem_stage=2 * _smem(c, rp),
            stages=1,
            hbm_bytes=2 * (t * i * split + t * o * (2 if has_base else 1))
            + shared_reads(dev, 2 * (i + o) * rp, tiles * split),
            mma_flops=2.0 * tiles * bm * rp * (i * split + o),
            # every program reads all of down and its 1/split of up
            l2_bytes=2
            * (
                t * i * split
                + tiles * split * (i + cdiv(o, split)) * rp
                + t * o * (2 if has_base else 1)
            ),
        )

    return [*rank(dev, space, work, mma=True), *alternatives]


def _run(cfg, flat, up, down, scale, base, out_dtype):
    """[base +] scale * (flat @ down^T) @ up^T in out_dtype; flat, up, down fp16."""
    t, i = flat.shape
    o, r = up.shape
    if cfg == TORCH:
        # cuBLAS with the scale as alpha and a same-dtype base as the addend
        h = flat @ down.t()
        if base is not None and base.dtype == flat.dtype:
            return torch.addmm(base, h, up.t(), alpha=scale).to(out_dtype)
        y = torch.addmm(h.new_empty(t, o), h, up.t(), beta=0, alpha=scale)
        return (y if base is None else base + y).to(out_dtype)
    if cfg == WHOLE:
        y = bypass._bypass_delta(flat, up, down, scale)
        return (y if base is None else base + y).to(out_dtype)
    rp = pad_to(r)
    args = (o, i, rp, cfg["BM"], cfg["BN"], cfg["BK"], cfg["threads"], cfg["SPLIT"])
    base_dt = None if base is None else tl_dtype(base.dtype)
    kernel = compiled(_bypass_prim, flat.device, *args, base_dt, tl_dtype(out_dtype))
    y = torch.empty(t, o, device=flat.device, dtype=out_dtype)
    d, u, x = padded_rows(down, rp), padded_cols(up, rp), flat.contiguous()
    if base is None:
        kernel(x, d, u, y, scale)
    else:
        kernel(x, d, u, base.contiguous(), y, scale)
    return y


def _bypass(name, flat, up, down, scale, base, out_dtype, alternatives):
    dev = device_info(flat.device)
    t, i = flat.shape
    o, r = up.shape
    key = (
        name,
        (
            dtype_key(flat.dtype),
            dtype_key(out_dtype),
            tl_dtype(base.dtype) if base is not None else None,
        ),
        (token_bucket(t), o, i, r),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, t, o, i, r, base is not None, alternatives),
        measurer(
            dev,
            t * (i + o * (2 if base is not None else 1)) * 2,
            lambda: (copy_of(flat), copy_of(base)),
            lambda c, s: lambda: _run(c, s[0], up, down, scale, s[1], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda c: _run(c, flat, up, down, scale, base, out_dtype),
        CompileError,
    )


def lora_bypass_fwd(x, up, down, gamma: float = 1.0, base=None, out_dtype=None):
    """[base +] gamma * (x @ down^T) @ up^T; x (T, I) and the factors in one
    compute dtype, base (T, O) or None; the result in ``out_dtype`` (default
    base's dtype, else x's)."""
    out_dtype = out_dtype or (base.dtype if base is not None else x.dtype)
    if x.dtype != torch.float16:
        y = bypass._bypass_delta(x, up, down, gamma)
        return (y if base is None else base + y).to(out_dtype)
    return _bypass(
        "tl.lora.bypass_fwd", x, up, down, gamma, base, out_dtype, [TORCH, WHOLE]
    )


def lora_bypass_bwd(x, up, down, grad, gamma: float = 1.0):
    """(gx, g_up, g_down) of the bypass in the compute dtype."""
    if x.dtype != torch.float16:
        return bypass.lora_bypass_bwd(x, up, down, grad, gamma)
    g2 = grad.contiguous()
    dev = device_info(x.device)
    key = (
        "tl.lora.bypass_bwd",
        (dtype_key(x.dtype),),
        (token_bucket(x.shape[0]), *up.shape, x.shape[1]),
    )
    gemms = {"impl": "gemms"}

    def run(cf, g, xx):
        if cf == WHOLE:
            return bypass.lora_bypass_bwd(xx, up, down, g, gamma)
        return _bwd_gemms(xx, up, down, g, gamma)

    cfg = select(
        dev,
        key,
        lambda: [gemms, WHOLE],
        measurer(
            dev,
            2 * g2.numel() * 2 + 2 * x.numel() * 2,
            lambda: (copy_of(g2), copy_of(x)),
            lambda cf, s: lambda: run(cf, *s),
        ),
    )
    return run(cfg, g2, x)


def _run_dxq(cfg, g, up, down, scale):
    """(gx = scale * (g @ up) @ down, q = g @ up in fp16): the forward kernel on
    (down^T, up^T), storing its h, which is q."""
    t, o = g.shape
    i, r = down.shape[1], up.shape[1]
    if cfg == TORCH:
        q = g @ up
        return torch.addmm(q.new_empty(t, i), q, down, beta=0, alpha=scale), q
    rp = pad_to(r)
    args = (i, o, rp, cfg["BM"], cfg["BN"], cfg["BK"], cfg["threads"], cfg["SPLIT"])
    kernel = compiled(_bypass_prim, g.device, *args, None, F16, True)
    gx = torch.empty(t, i, device=g.device, dtype=g.dtype)
    q = torch.empty(t, rp, device=g.device, dtype=g.dtype)
    kernel(
        g.contiguous(), padded_rows(up.t(), rp), padded_cols(down.t(), rp), gx, q, scale
    )
    return gx, q[:, :r]


def _dxq(g, up, down, scale):
    dev = device_info(g.device)
    t, o = g.shape
    i, r = down.shape[1], up.shape[1]
    key = ("tl.lora.bypass_dxq", (dtype_key(g.dtype),), (token_bucket(t), i, o, r))
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, t, i, o, r, False, [TORCH]),
        measurer(
            dev,
            t * (i + o) * 2,
            lambda: (copy_of(g),),
            lambda c, s: lambda: _run_dxq(c, s[0], up, down, scale),
        ),
    )
    return guarded(
        dev, key, cfg, lambda c: _run_dxq(c, g, up, down, scale), CompileError
    )


def _bwd_gemms(x, up, down, grad, gamma):
    """gx and q from the forward kernel on (down^T, up^T); h and the factor
    gradients as cuBLAS GEMMs over tokens."""
    gx, q = _dxq(grad, up, down, gamma)
    h = x @ down.t()
    gu = torch.addmm(h.new_empty(up.shape), grad.t(), h, beta=0, alpha=gamma)
    gd = torch.addmm(h.new_empty(down.shape), q.t(), x, beta=0, alpha=gamma)
    return gx, gu, gd
