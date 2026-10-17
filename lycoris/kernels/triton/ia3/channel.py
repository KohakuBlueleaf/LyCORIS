"""IA3 channel scale y = x * (alpha + gamma * w[ch]) on the model-ranked tuner,
mixed x / w dtypes, atomic-free backward. See docs/kernels/designs.md.
"""

import torch
import triton
import triton.language as tl

from ...tuner import TORCH, device_info, select
from ...tuner.measure import (
    copy_of,
    dtype_key,
    grid_configs,
    guarded,
    launch,
    measurer,
    pow2,
    token_bucket,
)
from ...tuner.model import Work, cdiv, rank
from ..common import OutOfResources


@triton.jit
def _fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    n_rows,
    n_cols,
    n_ch,
    alpha,
    gamma,
    ch_on_cols: tl.constexpr,
    block_r: tl.constexpr,
    block_q: tl.constexpr,
):
    """y = x * (alpha + gamma * w[ch]) over x as (rows, cols): ch is the column
    (channels-last, cols = C) or row % C (channels-first, rows = outer * C).

    Grid: (row tiles, column tiles).
    """
    rr = tl.program_id(0) * block_r + tl.arange(0, block_r)
    qq = tl.program_id(1) * block_q + tl.arange(0, block_q)
    mr = rr < n_rows
    mq = qq < n_cols
    off = rr[:, None] * n_cols + qq[None, :]
    m = mr[:, None] & mq[None, :]

    # Scale per column or per row, loaded once per tile.
    if ch_on_cols:
        w = tl.load(w_ptr + qq, mask=mq, other=0.0).to(tl.float32)[None, :]
    else:
        w = tl.load(w_ptr + rr % n_ch, mask=mr, other=0.0).to(tl.float32)[:, None]

    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + off, (x * (alpha + gamma * w)).to(y_ptr.dtype.element_ty), mask=m)


@triton.jit
def _bwd_kernel(
    g_ptr,
    x_ptr,
    w_ptr,
    gx_ptr,
    part_ptr,
    n_ch,
    inner,
    n_cols,
    alpha,
    gamma,
    block_c: tl.constexpr,
    block_j: tl.constexpr,
):
    """gx, and the partial channel sums of g*x*gamma for one (channel tile, column slice).

    Grid: (channel tiles, column splits); column j is (outer j // inner, inner j % inner).
    """
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    splits = tl.num_programs(1)
    rc = pid_c * block_c + tl.arange(0, block_c)
    mc = rc < n_ch
    w = tl.load(w_ptr + rc, mask=mc, other=0.0).to(tl.float32)

    # Columns [lo, hi) of this split; acc = per-channel sum of g * x.
    span = tl.cdiv(tl.cdiv(n_cols, block_j), splits) * block_j
    lo = pid_s * span
    hi = tl.minimum(lo + span, n_cols)
    acc = tl.zeros((block_c,), tl.float32)
    for j0 in range(lo, hi, block_j):
        rj = j0 + tl.arange(0, block_j)
        m = mc[:, None] & (rj < hi)[None, :]
        off = (
            (rj // inner)[None, :] * (n_ch * inner)
            + rc[:, None] * inner
            + (rj % inner)[None, :]
        )
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(
            gx_ptr + off,
            (g * (alpha + gamma * w)[:, None]).to(gx_ptr.dtype.element_ty),
            mask=m,
        )
        acc += tl.sum(g * x, axis=1)
    tl.store(part_ptr + pid_s * n_ch + rc, acc * gamma, mask=mc)


def _plan_fwd(dev, rows, cols, e):
    """(row tile, column tile, warps) configs over the (rows, cols) view, model-ranked, then torch."""
    qmax = pow2(cols)
    space = [
        cf
        for cf in grid_configs(
            dev,
            BR=(1, 2, 4, 8, 16, 32),
            BQ=(64, 128, 256, 512, 1024, 2048, 4096),
            num_warps=(2, 4, 8),
            num_stages=(1,),
        )
        if 512 <= cf["BR"] * cf["BQ"] <= 8192 and cf["BQ"] <= max(64, qmax)
    ]
    n = rows * cols

    def work(cf):
        br, bq = cf["BR"], cf["BQ"]
        # masked lanes past the last column still occupy the tile
        used = min(bq, cols) / bq
        return Work(
            grid=cdiv(rows, br) * cdiv(cols, bq),
            warps=cf["num_warps"],
            acc_fp32=br * bq,
            operand_bytes=br * bq * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=2 * n * e,
            vector_flops=2.0 * n / used,
            # one x tile loaded per program
            phases=((2 * n * e, br * bq * e * used),),
        )

    return [*rank(dev, space, work), TORCH]


def _plan_bwd(dev, outer, c, inner, e):
    if inner == 1:
        space = grid_configs(
            dev,
            BLOCK_C=(32, 64, 128),
            BLOCK_J=(16, 32, 64),
            num_warps=(4, 8),
            num_stages=(1,),
        )
    else:
        space = grid_configs(
            dev,
            BLOCK_C=(1, 2, 4),
            BLOCK_J=(256, 512, 1024),
            num_warps=(4, 8),
            num_stages=(1,),
        )
    cols = outer * inner

    def work(cf):
        bc, bj = cf["BLOCK_C"], cf["BLOCK_J"]
        tiles = cdiv(c, bc)
        splits = max(1, min(cdiv(cols, bj), cdiv(4 * dev.units, tiles)))
        # Read amplification from tiles narrower than a 32-byte sector.
        sector = max(1.0, 32 / (bc * e)) if inner == 1 else max(1.0, 32 / (bj * e))
        return Work(
            grid=tiles * splits,
            warps=cf["num_warps"],
            acc_fp32=bc,
            operand_bytes=3 * bc * bj * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=3 * c * cols * e * sector + 2 * splits * c * 4,
            vector_flops=3.0 * c * cols,
        )

    return [*rank(dev, space, work), TORCH]


def _view3(x: torch.Tensor, channel_axis: int):
    """(contiguous x, outer, C, inner) around the channel axis."""
    axis = channel_axis % x.dim()
    c = x.shape[axis]
    inner = 1
    for size in x.shape[axis + 1 :]:
        inner *= size
    return x.contiguous(), x.numel() // (c * inner), c, inner


def _shape(x, channel_axis):
    shape = [1] * x.dim()
    shape[channel_axis % x.dim()] = -1
    return shape


def ia3_fwd(x, w, channel_axis: int, alpha: float = 1.0, gamma: float = 1.0):
    """x * (alpha + gamma * w[c]) in the promoted dtype of x and w."""
    dev = device_info(x.device)
    xc, outer, c, inner = _view3(x, channel_axis)
    wv = w.reshape(-1).contiguous()
    out_dtype = torch.promote_types(x.dtype, w.dtype)
    # (rows, cols) view: channels-last puts C on the columns, channels-first on the rows.
    rows, cols = (outer, c) if inner == 1 else (outer * c, inner)
    key = (
        "ia3.fwd",
        (dtype_key(x.dtype), dtype_key(w.dtype), inner == 1),
        (token_bucket(rows), cols),
    )

    def run(cfg, xx):
        if cfg == TORCH:
            scale = alpha + gamma * wv.reshape(_shape(x, channel_axis))
            return (xx.view(x.shape) * scale).to(out_dtype)
        y = torch.empty(xx.shape, device=xx.device, dtype=out_dtype)
        grid = (triton.cdiv(rows, cfg["BR"]), triton.cdiv(cols, cfg["BQ"]))
        _fwd_kernel[grid](
            xx,
            wv,
            y,
            rows,
            cols,
            c,
            alpha,
            gamma,
            ch_on_cols=inner == 1,
            block_r=cfg["BR"],
            block_q=cfg["BQ"],
            **launch(cfg),
        )
        return y.view(x.shape)

    cfg = select(
        dev,
        key,
        lambda: _plan_fwd(dev, rows, cols, x.element_size()),
        measurer(
            dev,
            2 * x.numel() * x.element_size(),
            lambda: (copy_of(xc),),
            lambda cf, s: lambda: run(cf, s[0]),
        ),
    )
    return guarded(dev, key, cfg, lambda cf: run(cf, xc), OutOfResources)


def ia3_bwd(grad, x, w, channel_axis: int, alpha: float = 1.0, gamma: float = 1.0):
    """(gx in x's dtype, gw in w's dtype) of :func:`ia3_fwd`."""
    dev = device_info(x.device)
    xc, outer, c, inner = _view3(x, channel_axis)
    g = grad.reshape(x.shape).contiguous()
    wv = w.reshape(-1).contiguous()
    key = (
        "ia3.bwd",
        (dtype_key(g.dtype), dtype_key(x.dtype), dtype_key(w.dtype), inner == 1),
        (token_bucket(outer * inner), c),
    )

    def run(cfg, gg, xx):
        if cfg == TORCH:
            gf = gg.view(x.shape).float()
            gx = gf * (alpha + gamma * wv.float().reshape(_shape(x, channel_axis)))
            dims = [d for d in range(x.dim()) if d != channel_axis % x.dim()]
            gw = (gf * xx.view(x.shape).float()).sum(dims) * gamma
            return gx.to(x.dtype), gw
        gx = torch.empty(xx.shape, device=xx.device, dtype=x.dtype)
        cols = outer * inner
        tiles = triton.cdiv(c, cfg["BLOCK_C"])
        splits = max(
            1, min(triton.cdiv(cols, cfg["BLOCK_J"]), triton.cdiv(4 * dev.units, tiles))
        )
        part = torch.empty(splits, c, device=xx.device, dtype=torch.float32)
        _bwd_kernel[(tiles, splits)](
            gg,
            xx,
            wv,
            gx,
            part,
            c,
            inner,
            cols,
            alpha,
            gamma,
            block_c=cfg["BLOCK_C"],
            block_j=cfg["BLOCK_J"],
            **launch(cfg),
        )
        return gx, part.sum(0)

    cfg = select(
        dev,
        key,
        lambda: _plan_bwd(dev, outer, c, inner, x.element_size()),
        measurer(
            dev,
            3 * x.numel() * x.element_size(),
            lambda: (copy_of(g), copy_of(xc)),
            lambda cf, s: lambda: run(cf, s[0], s[1]),
        ),
    )
    gx, gw = guarded(dev, key, cfg, lambda cf: run(cf, g, xc), OutOfResources)
    return gx.reshape(x.shape), gw.reshape(w.shape).to(w.dtype)
