"""BOFT in TileLang: the rotations solved once, one streaming launch per
butterfly stage on channels-last rows, forward and backward, in x's dtype.
Other layouts take the torch chain or the single-launch kernels.
See docs/kernels/designs.md.
"""

import functools

import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer, pad16, token_bucket
from ...tuner.model import Work, cdiv, rank
from ..common import (
    CompileError,
    compiled,
    fits,
    fp32_gemm_exact,
    partitions,
    tl_dtype,
)
from ..oft.solve import MAX_BLOCK, chain, deltas
from . import apply, stage_kernels

# The single-launch kernels (blocks up to 32).
FUSED = {"impl": "fused"}
FUSED_MAX = 32
# Largest permuted block-diagonal tile (channels) of the rows layout.
_MAX_TILE = 128


def _view(x, axis):
    """x as (outer, C, inner) around the channel axis."""
    axis = axis % x.dim()
    c = x.shape[axis]
    inner = 1
    for size in x.shape[axis + 1 :]:
        inner *= size
    return x.reshape(-1, c, inner), x.numel() // (c * inner), c, inner


def _stage_geometry(i, b):
    """(k, blocks per group) of stage i."""
    return (2**i) * (b // 2), 2**i


def _plan(dev, m, num, b, outer, inner, e, backward):
    """Stage-apply configs of the rows layout, model-ranked, then the
    alternatives; one config serves every stage of the call; e is x's (the
    compute dtype's) element size."""
    rows_n = outer * inner
    c = num * b
    w = e // 2
    space = []
    exact = e == 2 or fp32_gemm_exact(dev)
    for g in (1, 2, 4, 8, 16, 32, 64) if inner == 1 and exact else ():
        tw = g * b
        if not (16 <= tw <= _MAX_TILE and tw % 16 == 0 and num % g == 0):
            continue
        for bt in (32, 64) if backward else (32, 64, 128):
            for th in (128, 256):
                halves = w * tw * tw + bt * tw * ((2 * w + 2) if backward else (w + 2))
                gemms = [(bt, tw), (tw, tw)] if backward else [(bt, tw)]
                if not (fits(dev, halves) and partitions(dev, th, *gemms)):
                    continue
                for occ in (2, 4, 8) if backward else (1,):
                    space.append({"G": g, "BT": bt, "OCC": occ, "threads": th})
    # running activation fp32 in and out, the 16-bit save / operand
    bytes_el = 4 + 4 + (2 if backward else 0)

    def work(cf):
        tw, bt = cf["G"] * b, cf["BT"]
        n_ct = num // cf["G"]
        if backward:
            splits = _splits(dev, rows_n, cf, n_ct)
            grid, partial = splits * n_ct, m * splits * n_ct * tw * tw * 4 * 2
        else:
            grid, partial = cdiv(rows_n, bt) * n_ct, 0
        return Work(
            grid=grid,
            warps=cf["threads"] // 32,
            acc_fp32=bt * tw + (tw * tw if backward else 0),
            operand_bytes=0,
            smem_stage=tw * tw * e + 3 * bt * tw * e,
            stages=1,
            hbm_bytes=m * rows_n * c * bytes_el + partial,
            mma_flops=2.0 * m * rows_n * c * tw * (2 if backward else 1),
        )

    alternatives = [TORCH, FUSED] if pad16(b) <= FUSED_MAX else [TORCH]
    return [*rank(dev, space, work, mma=e == 2), *alternatives]


def _splits(dev, rows, cfg, n_ct):
    return max(1, min(cdiv(rows, cfg["BT"]), cdiv(cfg["OCC"] * dev.units, n_ct)))


def _forward_stages(cfg, dh, x, rescale, diff, save, n_stages, out_dtype):
    """Run stages [0, n_stages) on the (rows, C) x; (last output, 16-bit inputs
    of stages 1..n_stages when ``save``). The last stage folds rescale and the
    diff when n_stages is the full m."""
    m, num, ps, _ = dh.shape
    rows, c = x.shape
    b = c // num
    dhf = dh.reshape(m * num, ps, ps)
    res = (
        rescale.reshape(-1).float().contiguous()
        if rescale is not None
        else torch.empty(c, device=x.device, dtype=torch.float32)
    )
    saved, u = [], x
    for i in range(n_stages):
        k, gpb = _stage_geometry(i, b)
        last = i == m - 1
        v = torch.empty(
            rows, c, device=x.device, dtype=out_dtype if last else torch.float32
        )
        sv = torch.empty(rows, c, device=x.device, dtype=x.dtype) if save else x
        kernel = compiled(
            stage_kernels.rows_fwd_prim,
            x.device,
            c,
            b,
            cfg["G"],
            cfg["BT"],
            cfg["threads"],
            save,
            last,
            last and rescale is not None,
            last and diff,
            (
                tl_dtype(x.dtype),
                tl_dtype(u.dtype),
                tl_dtype(v.dtype),
                tl_dtype(x.dtype),
            ),
        )
        kernel(dhf[i * num : (i + 1) * num], u, v, sv, x, res, k, gpb)
        if save:
            saved.append(sv)
        u = v
    return u, saved


def _to_blocks(u, k, b):
    """(..., C) in natural order -> (..., num, b) stage blocks of stride k."""
    return (
        u.unflatten(-1, (-1, 2, k)).transpose(-2, -1).flatten(-3).unflatten(-1, (-1, b))
    )


def _from_blocks(ub, k):
    """The inverse of :func:`_to_blocks`."""
    return ub.flatten(-2).unflatten(-1, (-1, k, 2)).transpose(-2, -1).flatten(-3)


def _torch_butterfly(dh, x, axis, rescale, diff):
    """The stage chain in torch: v = u + D u per stage, fp32 running activation."""
    m, num, ps, _ = dh.shape
    xv = x.movedim(axis % x.dim(), -1)
    c = xv.shape[-1]
    b = c // num
    u = xv.float()
    for i in range(m):
        k = (2**i) * (b // 2)
        ub = _to_blocks(u, k, b)
        dx = torch.einsum(
            "nij,...nj->...ni", dh[i, :, :b, :b].to(x.dtype), ub.to(x.dtype)
        ).float()
        u = _from_blocks(ub + dx, k)
    if rescale is not None:
        u = u * rescale.reshape(-1).float()
    if diff:
        u = u - xv.float()
    return u.movedim(-1, axis % x.dim())


def _fwd_run(cfg, dh, blocks, x, axis, cscale, scale, rescale, diff):
    if cfg == TORCH:
        return _torch_butterfly(dh, x, axis, rescale, diff)
    if cfg == FUSED:
        return apply.boft_fwd(
            blocks, x, axis=axis, cscale=cscale, scale=scale, rescale=rescale, diff=diff
        )
    c = x.shape[-1]
    out, _ = _forward_stages(
        cfg, dh, x.reshape(-1, c), rescale, diff, False, dh.shape[0], torch.float32
    )
    return out.view(x.shape)


def _key(name, x, axis, blocks, rescale, diff):
    _, outer, _, inner = _view(x, axis)
    m, num, b, _ = blocks.shape
    cat = (dtype_key(x.dtype), rescale is not None, diff, inner == 1)
    return name, cat, (token_bucket(outer * inner), m, num, b, inner)


def _deltas(blocks, cscale, scale, cdt):
    m, num, b, _ = blocks.shape
    ps = pad16(b)
    return deltas(blocks.reshape(m * num, b, b), None, cscale, cdt, scale).view(
        m, num, ps, ps
    )


def boft_fwd(blocks, x, axis=0, cscale=1.0, scale=1.0, rescale=None, diff=False):
    """butterfly(x) [* rescale] [- x] along ``axis`` in fp32; blocks (m, num, b, b)."""
    m, num, b, _ = blocks.shape
    if pad16(b) > MAX_BLOCK:
        return apply.boft_fwd(
            blocks, x, axis=axis, cscale=cscale, scale=scale, rescale=rescale, diff=diff
        )
    dev = device_info(x.device)
    x = x.contiguous()
    dh = _deltas(blocks, cscale, scale, x.dtype)
    _, outer, _, inner = _view(x, axis)
    key = _key("tl.boft.stages_fwd", x, axis, blocks, rescale, diff)
    args = (dh, blocks, x, axis, cscale, scale, rescale, diff)
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, m, num, b, outer, inner, x.element_size(), False),
        measurer(
            dev,
            m * x.numel() * 8,
            lambda: (copy_of(x),),
            lambda c, s: lambda: _fwd_run(
                c, dh, blocks, s[0], axis, cscale, scale, rescale, diff
            ),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _fwd_run(c, *args), CompileError)


@functools.cache
def _nat_index(i, b, g, device):
    """inv[bl, l] = natural tile index of (tile block bl, block index l) at
    stage i, on ``device`` (built once per stage geometry)."""
    rb = b // 2
    k, gpb = _stage_geometry(i, b)
    n = torch.arange(g * b)
    if g >= gpb:
        ch = n
    else:
        h = g * rb
        ch = (n // h) * k + n % h
    rem = ch % (2 * k)
    w = rem % k
    blk = (ch // (2 * k)) * gpb + w // rb
    li = (w % rb) * 2 + rem // k
    inv = torch.empty(g, b, dtype=torch.long)
    inv[blk - blk.min(), li] = n
    return inv.to(device)


def _bwd_stages(cfg, dev, dh, x, g):
    """(gx fp32, G_D (m, num, b, b) fp32) of the stage chain on the (rows, C) x."""
    m, num, ps, _ = dh.shape
    rows, c = x.shape
    b = c // num
    _, saved = _forward_stages(cfg, dh, x, None, False, True, m - 1, torch.float32)
    inputs = [x, *saved]
    dhf = dh.reshape(m * num, ps, ps)
    gv = g.reshape(rows, c).float().contiguous()
    tw = cfg["G"] * b
    n_ct = num // cfg["G"]
    splits = _splits(dev, rows, cfg, n_ct)
    gds = [None] * m
    for i in reversed(range(m)):
        k, gpb = _stage_geometry(i, b)
        u = inputs[i]
        gu = torch.empty(rows, c, device=x.device, dtype=torch.float32)
        part = torch.empty(splits, n_ct, tw, tw, device=x.device, dtype=torch.float32)
        kernel = compiled(
            stage_kernels.rows_bwd_prim,
            x.device,
            c,
            b,
            cfg["G"],
            cfg["BT"],
            splits,
            cfg["threads"],
            (tl_dtype(x.dtype), tl_dtype(u.dtype)),
        )
        kernel(dhf[i * num : (i + 1) * num], gv, u, gu, part, k, gpb)
        inv = _nat_index(i, b, cfg["G"], x.device)
        p = part[0] if splits == 1 else part.sum(0)
        gds[i] = p[:, inv[:, :, None], inv[:, None, :]].reshape(num, b, b)
        gv = gu
    return gv, torch.stack(gds)


def _torch_bwd(dh, x, g, axis):
    """(gx fp32, G_D (m, num, b, b) fp32) of the torch chain, written out
    (it runs inside custom ops, where autograd does not record)."""
    m, num, ps, _ = dh.shape
    ax = axis % x.dim()
    xv = x.movedim(ax, -1)
    b = xv.shape[-1] // num
    d16 = dh[:, :, :b, :b].to(x.dtype)
    u, inputs = xv.float(), []
    for i in range(m):
        k = (2**i) * (b // 2)
        ub = _to_blocks(u, k, b)
        inputs.append(ub.to(x.dtype))
        dx = torch.einsum("nij,...nj->...ni", d16[i], inputs[-1]).float()
        u = _from_blocks(ub + dx, k)
    gv, gds = g.movedim(ax, -1).float(), [None] * m
    for i in reversed(range(m)):
        k = (2**i) * (b // 2)
        gb = _to_blocks(gv, k, b)
        gh = gb.to(x.dtype)
        gds[i] = torch.einsum("...ni,...nj->nij", gh, inputs[i]).float()
        gu = gb + torch.einsum("nij,...ni->...nj", d16[i], gh).float()
        gv = _from_blocks(gu, k)
    return gv.movedim(-1, ax), torch.stack(gds)


def _bwd_run(cfg, dev, dh, blocks, x, g, axis, cscale, scale):
    """(gx fp32, g_blocks)."""
    m, num, b, _ = blocks.shape
    if cfg == FUSED:
        return apply.boft_bwd(blocks, x, g, axis=axis, cscale=cscale, scale=scale)
    if cfg == TORCH:
        gx, gd = _torch_bwd(dh, x, g, axis)
    else:
        c = x.shape[-1]
        gx, gd = _bwd_stages(cfg, dev, dh, x.reshape(-1, c), g)
        gx = gx.view(x.shape)
    gd = (gd * scale).reshape(m * num, b, b)
    gb, _ = chain(blocks.reshape(m * num, b, b), None, gd, dh.dtype, cscale)
    return gx, gb.view(blocks.shape)


def boft_bwd(blocks, x, grad, axis=0, cscale=1.0, scale=1.0):
    """(gx fp32, g_blocks) of :func:`boft_fwd` without rescale and diff (the
    autograd layer applies both)."""
    m, num, b, _ = blocks.shape
    if pad16(b) > MAX_BLOCK:
        return apply.boft_bwd(blocks, x, grad, axis=axis, cscale=cscale, scale=scale)
    dev = device_info(x.device)
    x = x.contiguous()
    dh = _deltas(blocks, cscale, scale, x.dtype)
    _, outer, _, inner = _view(x, axis)
    key = _key("tl.boft.stages_bwd", x, axis, blocks, None, False)
    args = (dh, blocks, x, grad, axis, cscale, scale)
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, m, num, b, outer, inner, x.element_size(), True),
        measurer(
            dev,
            m * x.numel() * 10,
            lambda: (copy_of(grad),),
            lambda c, s: lambda: _bwd_run(
                c, dev, dh, blocks, x, s[0], axis, cscale, scale
            ),
        ),
    )
    return guarded(dev, key, cfg, lambda c: _bwd_run(c, dev, *args), CompileError)
