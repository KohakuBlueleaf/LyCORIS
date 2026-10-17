"""Diag-OFT in TileLang in the I + D form (the rotation solved once per block,
the apply streamed), forward and backward, D and x in fp32 (``oft_fwd_full``).
Layouts: "rows" (channels-last activations, a row tile
against the block-diagonal Dh of a channel tile) and "cols" (one block over a
slice of the columns: weights, channels-first activations).
See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
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
from . import apply
from .solve import MAX_BLOCK, chain, deltas

F32 = "float32"
# The single-launch kernels of ``.apply`` (blocks up to 32), timed for the fp32 form.
SINGLE = {"impl": "single"}
SINGLE_MAX = 32
# DRAM sector size, bytes.
_SECTOR = 32


def _rows_fwd_prim(k, s, bch, bt, rt, threads, rescale_on, shift, dts):
    """out = keep * x + x @ Dh_bd for rt row tiles of one channel tile; dts =
    (compute, x, out) dtypes, x converted to the compute dtype in shared memory."""
    c_dt, x_dt, o_dt = dts
    c, g, ps = k * s, bch // s, pad16(s)
    t = T.symbolic("t")

    @T.prim_func
    def main(
        dh: T.Tensor((k, ps, ps), c_dt),
        res: T.Tensor((c,), F32),
        x: T.Tensor((t, c), x_dt),
        out: T.Tensor((t, c), o_dt),
    ):
        # Grid: (row tile groups bx, channel tiles by).
        with T.Kernel(T.ceildiv(t, bt * rt), k // g, threads=threads) as (bx, by):
            d_s = T.alloc_shared((bch, bch), c_dt)
            x_s = T.alloc_shared((bt, bch), c_dt)
            keep_s = T.alloc_shared((bch,), F32)
            o_f = T.alloc_fragment((bt, bch), F32)
            for j, i in T.Parallel(bch, bch):
                d_s[j, i] = T.if_then_else(
                    j // s == i // s,
                    dh[by * g + j // s, j % s, i % s],
                    T.cast(0, c_dt),
                )
            if rescale_on:
                for i in T.Parallel(bch):
                    keep_s[i] = res[by * bch + i] - shift
            else:
                for i in T.Parallel(bch):
                    keep_s[i] = T.float32(1.0 - shift)
            for n in T.serial(rt):
                T.copy(x[(bx * rt + n) * bt, by * bch], x_s)
                T.clear(o_f)
                T.gemm(x_s, d_s, o_f)
                for r, i in T.Parallel(bt, bch):
                    o_f[r, i] += keep_s[i] * T.cast(x_s[r, i], F32)
                T.copy(o_f, out[(bx * rt + n) * bt, by * bch])

    return main


def _rows_bwd_prim(k, s, bch, bt, splits, threads, rescale_on, shift, dts):
    """gx = keep * g + g @ Dh_bd^T over this program's rows, and its partial of
    P = x^T g (fp32 accumulation); dts = (compute, x and gx, grad) dtypes."""
    c_dt, x_dt, g_dt = dts
    c, g_, ps = k * s, bch // s, pad16(s)
    n_ct = k // g_
    t = T.symbolic("t")

    @T.prim_func
    def main(
        dh: T.Tensor((k, ps, ps), c_dt),
        res: T.Tensor((c,), F32),
        x: T.Tensor((t, c), x_dt),
        g: T.Tensor((t, c), g_dt),
        gx: T.Tensor((t, c), x_dt),
        part: T.Tensor((splits, n_ct, bch, bch), F32),
    ):
        # Grid: (row splits bs, channel tiles by).
        with T.Kernel(splits, n_ct, threads=threads) as (bs, by):
            d_s = T.alloc_shared((bch, bch), c_dt)
            x_s = T.alloc_shared((bt, bch), c_dt)
            gw_s = T.alloc_shared((bt, bch), g_dt)
            g_s = T.alloc_shared((bt, bch), c_dt)
            keep_s = T.alloc_shared((bch,), F32)
            gx_f = T.alloc_fragment((bt, bch), F32)
            p_f = T.alloc_fragment((bch, bch), F32)
            for j, i in T.Parallel(bch, bch):
                d_s[j, i] = T.if_then_else(
                    j // s == i // s,
                    dh[by * g_ + j // s, j % s, i % s],
                    T.cast(0, c_dt),
                )
            if rescale_on:
                for i in T.Parallel(bch):
                    keep_s[i] = res[by * bch + i] - shift
            else:
                for i in T.Parallel(bch):
                    keep_s[i] = T.float32(1.0 - shift)
            T.clear(p_f)
            per = T.ceildiv(T.ceildiv(t, bt), splits)
            for n in T.serial(per):
                r0 = (bs * per + n) * bt
                T.copy(x[r0, by * bch], x_s)
                T.copy(g[r0, by * bch], gw_s)
                for r, i in T.Parallel(bt, bch):
                    g_s[r, i] = T.cast(gw_s[r, i], c_dt)
                T.clear(gx_f)
                T.gemm(g_s, d_s, gx_f, transpose_B=True)
                for r, i in T.Parallel(bt, bch):
                    gx_f[r, i] += keep_s[i] * T.cast(gw_s[r, i], F32)
                T.copy(gx_f, gx[r0, by * bch])
                T.gemm(x_s, g_s, p_f, transpose_A=True)
            T.copy(p_f, part[bs, by, 0, 0])

    return main


def _cols_fwd_prim(k, s, inner, bn, groups, threads, rescale_on, shift, dts):
    """out = keep * x + Dh^T x for one block over one slice of the inner columns
    of one outer index of the (outer, C, inner) view; dts = (compute, x, out)."""
    c_dt, x_dt, o_dt = dts
    c, ps = k * s, pad16(s)
    tiles = cdiv(inner, bn)
    per = cdiv(tiles, groups)
    exact = s == ps
    outer = T.symbolic("outer")

    @T.prim_func
    def main(
        dh: T.Tensor((k, ps, ps), c_dt),
        res: T.Tensor((c,), F32),
        x: T.Tensor((outer, c, inner), x_dt),
        out: T.Tensor((outer, c, inner), o_dt),
    ):
        # Grid: (blocks bk, outer index bo, column groups bg).
        with T.Kernel(k, outer, groups, threads=threads) as (bk, bo, bg):
            dt_s = T.alloc_shared((ps, ps), c_dt)
            x_s = T.alloc_shared((ps, bn), c_dt)
            keep_s = T.alloc_shared((ps,), F32)
            o_f = T.alloc_fragment((ps, bn), F32)
            for i, j in T.Parallel(ps, ps):
                dt_s[i, j] = dh[bk, j, i]
            if rescale_on:
                for i in T.Parallel(ps):
                    keep_s[i] = res[bk * s + T.min(i, s - 1)] - shift
            else:
                for i in T.Parallel(ps):
                    keep_s[i] = T.float32(1.0 - shift)
            for n in T.serial(per):
                c0 = (bg * per + n) * bn
                if c0 < inner:
                    if exact:
                        T.copy(x[bo, bk * s, c0], x_s)
                    else:
                        for i, j in T.Parallel(ps, bn):
                            x_s[i, j] = T.if_then_else(
                                (i < s) and (c0 + j < inner),
                                T.cast(
                                    x[
                                        bo,
                                        bk * s + T.min(i, s - 1),
                                        T.min(c0 + j, inner - 1),
                                    ],
                                    c_dt,
                                ),
                                T.cast(0, c_dt),
                            )
                    T.clear(o_f)
                    T.gemm(dt_s, x_s, o_f)
                    for i, j in T.Parallel(ps, bn):
                        o_f[i, j] += keep_s[i] * T.cast(x_s[i, j], F32)
                    if exact:
                        T.copy(o_f, out[bo, bk * s, c0])
                    else:
                        for i, j in T.Parallel(ps, bn):
                            if (i < s) and (c0 + j < inner):
                                out[bo, bk * s + i, c0 + j] = T.cast(o_f[i, j], o_dt)

    return main


def _cols_bwd_prim(k, s, inner, bn, groups, threads, rescale_on, shift, dts):
    """gx = keep * g + Dh g over one column slice of one block, and its partial
    of P = x g^T at part[block, outer, group]; dts = (compute, x and gx, grad)."""
    c_dt, x_dt, g_dt = dts
    c, ps = k * s, pad16(s)
    tiles = cdiv(inner, bn)
    per = cdiv(tiles, groups)
    exact = s == ps
    outer = T.symbolic("outer")

    @T.prim_func
    def main(
        dh: T.Tensor((k, ps, ps), c_dt),
        res: T.Tensor((c,), F32),
        x: T.Tensor((outer, c, inner), x_dt),
        g: T.Tensor((outer, c, inner), g_dt),
        gx: T.Tensor((outer, c, inner), x_dt),
        part: T.Tensor((k, outer, groups, ps, ps), F32),
    ):
        with T.Kernel(k, outer, groups, threads=threads) as (bk, bo, bg):
            d_s = T.alloc_shared((ps, ps), c_dt)
            x_s = T.alloc_shared((ps, bn), c_dt)
            gw_s = T.alloc_shared((ps, bn), g_dt)
            g_s = T.alloc_shared((ps, bn), c_dt)
            keep_s = T.alloc_shared((ps,), F32)
            gx_f = T.alloc_fragment((ps, bn), F32)
            p_f = T.alloc_fragment((ps, ps), F32)
            T.copy(dh[bk, 0, 0], d_s)
            if rescale_on:
                for i in T.Parallel(ps):
                    keep_s[i] = res[bk * s + T.min(i, s - 1)] - shift
            else:
                for i in T.Parallel(ps):
                    keep_s[i] = T.float32(1.0 - shift)
            T.clear(p_f)
            for n in T.serial(per):
                c0 = (bg * per + n) * bn
                if c0 < inner:
                    if exact:
                        T.copy(x[bo, bk * s, c0], x_s)
                        T.copy(g[bo, bk * s, c0], gw_s)
                    else:
                        for i, j in T.Parallel(ps, bn):
                            ok = (i < s) and (c0 + j < inner)
                            ri = bk * s + T.min(i, s - 1)
                            cj = T.min(c0 + j, inner - 1)
                            x_s[i, j] = T.if_then_else(
                                ok, T.cast(x[bo, ri, cj], c_dt), T.cast(0, c_dt)
                            )
                            gw_s[i, j] = T.if_then_else(
                                ok, g[bo, ri, cj], T.cast(0, g_dt)
                            )
                    for i, j in T.Parallel(ps, bn):
                        g_s[i, j] = T.cast(gw_s[i, j], c_dt)
                    T.clear(gx_f)
                    T.gemm(d_s, g_s, gx_f)
                    for i, j in T.Parallel(ps, bn):
                        gx_f[i, j] += keep_s[i] * T.cast(gw_s[i, j], F32)
                    if exact:
                        T.copy(gx_f, gx[bo, bk * s, c0])
                    else:
                        for i, j in T.Parallel(ps, bn):
                            if (i < s) and (c0 + j < inner):
                                gx[bo, bk * s + i, c0 + j] = T.cast(gx_f[i, j], x_dt)
                    T.gemm(x_s, g_s, p_f, transpose_B=True)
            T.copy(p_f, part[bk, bo, bg, 0, 0])

    return main


def _layout(x: torch.Tensor, weight: bool) -> tuple[torch.Tensor, int, int, int]:
    """x as (outer, C, inner): the channel axis is 0 for a weight, the last for
    a (T, C) activation, 1 for a (B, C, ...) one."""
    if weight:
        return x.reshape(1, x.shape[0], -1), 1, x.shape[0], x.numel() // x.shape[0]
    if x.dim() == 2:
        return x.reshape(x.shape[0], x.shape[1], 1), x.shape[0], x.shape[1], 1
    inner = x.numel() // (x.shape[0] * x.shape[1])
    return x.reshape(x.shape[0], x.shape[1], inner), x.shape[0], x.shape[1], inner


def _rows_legal(dev, k, s, bch, bt, threads, backward, wide):
    if bch % s or bch % 16 or k % (bch // s):
        return False
    w = 2 if wide else 1
    halves = w * bch * bch + bt * bch * ((2 * w + 2) if backward else w)
    gemms = [(bt, bch), (bch, bch)] if backward else [(bt, bch)]
    return fits(dev, halves) and partitions(dev, threads, *gemms)


def _plan(dev, k, s, outer, inner, e, backward, wide=False, alternatives=(TORCH,)):
    """Apply configs of both layouts, model-ranked, then the alternatives;
    ``wide``: fp32 compute operands."""
    ps = pad16(s)
    rows_n = outer * inner
    c = k * s
    space = []
    exact = not wide or fp32_gemm_exact(dev)
    # fp32 (FMA) products keep their row tile at 32 in the backward
    tiles = ((32,) if wide else (32, 64)) if backward else (32, 64, 128)
    if inner == 1 and exact:
        for bch in (16, 32, 48, 64, 96, 128):
            for bt in tiles:
                for th in (128, 256):
                    if not _rows_legal(dev, k, s, bch, bt, th, backward, wide):
                        continue
                    for v in (2, 4, 8) if backward else (1, 4, 16):
                        key = "OCC" if backward else "RT"
                        space.append(
                            {
                                "LAYOUT": "rows",
                                "BCH": bch,
                                "BT": bt,
                                key: v,
                                "threads": th,
                            }
                        )
    for bn in (32, 64, 128) if exact and (inner > 1 or outer == 1) else ():
        for th in (32, 64, 128):
            gemms = [(ps, bn), (ps, ps)] if backward else [(ps, bn)]
            w = 2 if wide else 1
            if partitions(dev, th, *gemms) and fits(dev, w * ps * ps + ps * bn * 4 * w):
                for occ in (2, 4, 8):
                    space.append(
                        {"LAYOUT": "cols", "BN": bn, "OCC": occ, "threads": th}
                    )
    passes = 3 if backward else 2

    def work(cf):
        if cf["LAYOUT"] == "cols":
            bn = cf["BN"]
            groups = _col_groups(dev, outer, inner, cf, k)
            strided = _SECTOR / e if inner == 1 else 1.0
            return Work(
                grid=k * outer * groups,
                warps=cf["threads"] // 32,
                acc_fp32=ps * bn + (ps * ps if backward else 0),
                operand_bytes=0,
                smem_stage=ps * ps * e + passes * ps * bn * e,
                stages=1,
                hbm_bytes=passes * rows_n * c * e * strided
                + k * outer * groups * ps * ps * (4 if backward else e),
                mma_flops=2.0 * rows_n * c * ps * (2 if backward else 1),
            )
        bch, bt = cf["BCH"], cf["BT"]
        n_ct = k // (bch // s)
        if backward:
            splits = _splits(dev, rows_n, cf, n_ct)
            grid, partial = splits * n_ct, splits * n_ct * bch * bch * 4 * 2
        else:
            grid, partial = cdiv(rows_n, bt * cf["RT"]) * n_ct, 0
        return Work(
            grid=grid,
            warps=cf["threads"] // 32,
            acc_fp32=bt * bch + (bch * bch if backward else 0),
            operand_bytes=0,
            smem_stage=bch * bch * e + passes * bt * bch * e,
            stages=1,
            hbm_bytes=passes * rows_n * c * e + grid * bch * bch * e + partial,
            mma_flops=2.0 * rows_n * c * bch * (2 if backward else 1),
        )

    return [*rank(dev, space, work, mma=not wide), *alternatives]


def _col_groups(dev, outer, inner, cfg, k):
    tiles = cdiv(inner, cfg["BN"])
    return max(1, min(tiles, cdiv(cfg["OCC"] * dev.units, k * outer)))


def _splits(dev, rows, cfg, n_ct):
    return max(1, min(cdiv(rows, cfg["BT"]), cdiv(cfg["OCC"] * dev.units, n_ct)))


def _res(rescale, c, like):
    if rescale is None:
        return torch.empty(c, device=like.device, dtype=torch.float32)
    return rescale.reshape(-1).float().contiguous()


def _torch_apply(dh, x, rescale, shift, weight, out_dtype):
    """keep * x + the Dh product in torch (fp32 accumulation)."""
    k, ps, _ = dh.shape
    xv, outer, c, inner = _layout(x, weight)
    s = c // k
    xb = xv.reshape(outer, k, s, inner)
    d = dh[:, :s, :s]
    y = torch.einsum("oksn,ksi->okin", xb.to(dh.dtype), d).float()
    if rescale is not None:
        keep = rescale.reshape(k, s).float() - float(shift)
    else:
        keep = torch.full((k, s), 1.0 - float(shift), device=x.device)
    out = y + keep[None, :, :, None] * xb.float()
    return out.reshape(x.shape).to(out_dtype)


def _fwd_run(cfg, dev, dh, x, rescale, shift, weight, out_dtype, blocks, cscale):
    if cfg == TORCH:
        return _torch_apply(dh, x, rescale, shift, weight, out_dtype)
    if cfg == SINGLE:
        out = apply.oft_fwd(blocks, x, rescale, cscale, shift, weight)
        return out.to(out_dtype)
    k, ps, _ = dh.shape
    xv, outer, c, inner = _layout(x.contiguous(), weight)
    s = c // k
    res = _res(rescale, c, x)
    dts = (tl_dtype(dh.dtype), tl_dtype(x.dtype), tl_dtype(out_dtype))
    flags = (rescale is not None, float(shift), dts)
    if cfg["LAYOUT"] == "rows":
        kernel = compiled(
            _rows_fwd_prim,
            x.device,
            k,
            s,
            cfg["BCH"],
            cfg["BT"],
            cfg["RT"],
            cfg["threads"],
            *flags,
        )
        out = torch.empty(outer, c, device=x.device, dtype=out_dtype)
        kernel(dh, res, xv.view(outer, c), out)
        return out.view(x.shape)
    groups = _col_groups(dev, outer, inner, cfg, k)
    kernel = compiled(
        _cols_fwd_prim, x.device, k, s, inner, cfg["BN"], groups, cfg["threads"], *flags
    )
    out = torch.empty(outer, c, inner, device=x.device, dtype=out_dtype)
    kernel(dh, res, xv, out)
    return out.view(x.shape)


def _key(name, x, k, s, rescale, shift, weight, out_dtype):
    _, outer, _, inner = _layout(x, weight)
    cat = (
        dtype_key(x.dtype),
        dtype_key(out_dtype),
        rescale is not None,
        shift,
        weight,
        inner == 1,
    )
    return name, cat, (token_bucket(outer * inner), k, s, inner)


def _alternatives(s, wide):
    """torch, and the single-launch kernels for the fp32 form."""
    return (TORCH, SINGLE) if wide and pad16(s) <= SINGLE_MAX else (TORCH,)


def _fwd(name, blocks, x, rescale, cscale, shift, weight, out_dtype, cdt):
    k, s, _ = blocks.shape
    if pad16(s) > MAX_BLOCK:
        raise ValueError(f"diag-OFT kernels take blocks up to {MAX_BLOCK}, got {s}")
    dev = device_info(x.device)
    x = x.contiguous()
    wide = cdt == torch.float32
    dh = deltas(blocks, rescale, cscale, cdt)
    _, outer, _, inner = _layout(x, weight)
    key = _key(name, x, k, s, rescale, shift, weight, out_dtype)
    args = (rescale, shift, weight, out_dtype, blocks, cscale)
    cfg = select(
        dev,
        key,
        lambda: _plan(
            dev,
            k,
            s,
            outer,
            inner,
            x.element_size(),
            False,
            wide,
            _alternatives(s, wide),
        ),
        measurer(
            dev,
            2 * x.numel() * x.element_size(),
            lambda: (copy_of(x),),
            lambda c, st: lambda: _fwd_run(c, dev, dh, st[0], *args),
        ),
    )
    return guarded(
        dev, key, cfg, lambda c: _fwd_run(c, dev, dh, x, *args), CompileError
    )


def oft_fwd_full(blocks, x, rescale=None, cscale=1.0, shift=True, weight=True):
    """out = keep * x + the D * rescale product, keep = rescale - shift:
    (rescale_rows * R^T - shift*I) applied to x, D and x in fp32, the result in
    x's dtype."""
    args = (rescale, cscale, shift, weight, x.dtype, torch.float32)
    return _fwd("tl.oft.full_fwd", blocks, x, *args)


def _diag_blocks(part, k, s, g):
    """The (k, s, s) diagonal blocks of per-channel-tile (n_ct, bch, bch) sums."""
    blk = torch.arange(k, device=part.device)
    tile = blk // g
    rows = (blk % g)[:, None] * s + torch.arange(s, device=part.device)
    return part[tile[:, None, None], rows[:, :, None], rows[:, None, :]]


def _bwd_apply(cfg, dev, dh, x, g, rescale, shift, weight):
    """(gx in x's dtype, P (k, s, s) fp32)."""
    k, ps, _ = dh.shape
    xv, outer, c, inner = _layout(x, weight)
    s = c // k
    gv = g.reshape(xv.shape)
    if cfg == TORCH:
        gx = _torch_apply(
            dh.transpose(1, 2).contiguous(), g, rescale, shift, weight, x.dtype
        )
        xb = xv.reshape(outer, k, s, inner).to(dh.dtype)
        gb = gv.reshape(outer, k, s, inner).to(dh.dtype)
        return gx, torch.einsum("okjn,okin->kji", xb, gb).float()
    res = _res(rescale, c, x)
    dts = (tl_dtype(dh.dtype), tl_dtype(x.dtype), tl_dtype(g.dtype))
    flags = (rescale is not None, float(shift), dts)
    gx = torch.empty(xv.shape, device=x.device, dtype=x.dtype)
    if cfg["LAYOUT"] == "rows":
        bch = cfg["BCH"]
        n_ct = k // (bch // s)
        splits = _splits(dev, outer, cfg, n_ct)
        kernel = compiled(
            _rows_bwd_prim,
            x.device,
            k,
            s,
            bch,
            cfg["BT"],
            splits,
            cfg["threads"],
            *flags,
        )
        part = torch.empty(splits, n_ct, bch, bch, device=x.device, dtype=torch.float32)
        kernel(dh, res, xv.view(outer, c), gv.view(outer, c), gx.view(outer, c), part)
        p = _diag_blocks(part[0] if splits == 1 else part.sum(0), k, s, bch // s)
        return gx.view(x.shape), p
    groups = _col_groups(dev, outer, inner, cfg, k)
    kernel = compiled(
        _cols_bwd_prim, x.device, k, s, inner, cfg["BN"], groups, cfg["threads"], *flags
    )
    part = torch.empty(k, outer, groups, ps, ps, device=x.device, dtype=torch.float32)
    kernel(dh, res, xv, gv, gx, part)
    return gx.view(x.shape), part.sum((1, 2))[:, :s, :s]


def _bwd_run(cfg, dev, dh, x, g, rescale, shift, weight, blocks, cscale):
    """(gx, g_blocks, g_rescale fp32 (k*s,), empty without rescale) under cfg."""
    if cfg == SINGLE:
        gx, gb, gres = apply.oft_bwd(blocks, x, g, rescale, cscale, shift, weight)
    else:
        gx, p = _bwd_apply(cfg, dev, dh, x, g, rescale, shift, weight)
        gb, gres = chain(blocks, rescale, p, dh.dtype, cscale)
    if rescale is None:
        return gx, gb, gx.new_empty(0)
    return gx, gb, gres.reshape(-1).float()


def _bwd(name, blocks, x, grad, rescale, cscale, shift, weight, cdt):
    k, s, _ = blocks.shape
    if pad16(s) > MAX_BLOCK:
        raise ValueError(f"diag-OFT kernels take blocks up to {MAX_BLOCK}, got {s}")
    dev = device_info(x.device)
    x = x.contiguous()
    wide = cdt == torch.float32
    dh = deltas(blocks, rescale, cscale, cdt)
    _, outer, _, inner = _layout(x, weight)
    g = grad.reshape(x.shape).contiguous()
    key = _key(name, x, k, s, rescale, shift, weight, g.dtype)
    args = (rescale, shift, weight, blocks, cscale)
    cfg = select(
        dev,
        key,
        lambda: _plan(
            dev,
            k,
            s,
            outer,
            inner,
            x.element_size(),
            True,
            wide,
            _alternatives(s, wide),
        ),
        measurer(
            dev,
            3 * x.numel() * x.element_size(),
            lambda: (copy_of(x), copy_of(g)),
            lambda c, st: lambda: _bwd_run(c, dev, dh, st[0], st[1], *args),
        ),
    )
    gx, gb, gres = guarded(
        dev, key, cfg, lambda c: _bwd_run(c, dev, dh, x, g, *args), CompileError
    )
    return gx, gb, (gres if rescale is not None else None)


def oft_bwd_full(blocks, x, grad, rescale=None, cscale=1.0, shift=True, weight=True):
    """(gx in x's dtype, g_blocks, g_rescale | None) of :func:`oft_fwd_full`."""
    args = (rescale, cscale, shift, weight, torch.float32)
    return _bwd("tl.oft.full_bwd", blocks, x, grad, *args)
