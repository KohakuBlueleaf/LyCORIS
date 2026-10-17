"""IA3 channel scale y = x * (alpha + gamma * w[ch]) in TileLang on the
model-ranked tuner, mixed x / w dtypes, atomic-free backward (fp32 partials
per column split). See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer, token_bucket
from ...tuner.model import Work, cdiv, rank
from ..common import CompileError, compiled, tl_dtype
from . import scale

F32 = "float32"
# The single-launch kernels of ``.scale``.
SCALE = {"impl": "scale"}
_KERNEL_DTYPES = (torch.float16, torch.float32)


def _fwd_prim(cols, n_ch, br, bq, threads, ch_on_cols, dts):
    """y = x * (alpha + gamma * w[ch]) over x as (rows, cols): ch is the column
    (channels-last) or row % C (channels-first)."""
    x_dt, w_dt, y_dt = dts
    rows = T.symbolic("rows")

    @T.prim_func
    def main(
        x: T.Tensor((rows, cols), x_dt),
        w: T.Tensor((n_ch,), w_dt),
        y: T.Tensor((rows, cols), y_dt),
        alpha: T.float32,
        gamma: T.float32,
    ):
        # Grid: (column tiles bx, row tiles by).
        with T.Kernel(T.ceildiv(cols, bq), T.ceildiv(rows, br), threads=threads) as (
            bx,
            by,
        ):
            x_f = T.alloc_fragment((br, bq), F32)
            T.copy(x[by * br, bx * bq], x_f)
            if ch_on_cols:
                for i, j in T.Parallel(br, bq):
                    wv = T.cast(w[T.min(bx * bq + j, n_ch - 1)], F32)
                    x_f[i, j] = x_f[i, j] * (alpha + gamma * wv)
            else:
                for i, j in T.Parallel(br, bq):
                    wv = T.cast(w[(by * br + i) % n_ch], F32)
                    x_f[i, j] = x_f[i, j] * (alpha + gamma * wv)
            T.copy(x_f, y[by * br, bx * bq])

    return main


def _bwd_last_prim(n_ch, bc, bj, splits, threads, dts):
    """Channels-last (outer, C): gx, and the per-split channel sums of
    g * x * gamma for one channel tile."""
    g_dt, x_dt, w_dt = dts
    outer = T.symbolic("outer")

    @T.prim_func
    def main(
        g: T.Tensor((outer, n_ch), g_dt),
        x: T.Tensor((outer, n_ch), x_dt),
        w: T.Tensor((n_ch,), w_dt),
        gx: T.Tensor((outer, n_ch), x_dt),
        part: T.Tensor((splits, n_ch), F32),
        alpha: T.float32,
        gamma: T.float32,
    ):
        # Grid: (channel tiles bx, row splits bs).
        with T.Kernel(T.ceildiv(n_ch, bc), splits, threads=threads) as (bx, bs):
            g_f = T.alloc_fragment((bj, bc), F32)
            x_f = T.alloc_fragment((bj, bc), F32)
            prod = T.alloc_fragment((bj, bc), F32)
            row = T.alloc_fragment((bc,), F32)
            acc = T.alloc_fragment((bc,), F32)
            sc = T.alloc_fragment((bc,), F32)
            for j in T.Parallel(bc):
                sc[j] = alpha + gamma * T.cast(w[T.min(bx * bc + j, n_ch - 1)], F32)
            T.clear(acc)
            per = T.ceildiv(T.ceildiv(outer, bj), splits)
            for it in T.serial(per):
                r0 = (bs * per + it) * bj
                T.copy(g[r0, bx * bc], g_f)
                T.copy(x[r0, bx * bc], x_f)
                for i, j in T.Parallel(bj, bc):
                    prod[i, j] = g_f[i, j] * x_f[i, j]
                    g_f[i, j] = g_f[i, j] * sc[j]
                T.copy(g_f, gx[r0, bx * bc])
                T.reduce_sum(prod, row, dim=0)
                for j in T.Parallel(bc):
                    acc[j] += row[j]
            for j in T.Parallel(bc):
                if bx * bc + j < n_ch:
                    part[bs, bx * bc + j] = acc[j] * gamma

    return main


def _bwd_first_prim(n_ch, inner, bc, bj, splits, threads, dts):
    """Channels-first (outer, C, inner): gx, and the per-split channel sums of
    g * x * gamma for one channel tile."""
    g_dt, x_dt, w_dt = dts
    outer = T.symbolic("outer")
    tiles = cdiv(inner, bj)

    @T.prim_func
    def main(
        g: T.Tensor((outer, n_ch, inner), g_dt),
        x: T.Tensor((outer, n_ch, inner), x_dt),
        w: T.Tensor((n_ch,), w_dt),
        gx: T.Tensor((outer, n_ch, inner), x_dt),
        part: T.Tensor((splits, n_ch), F32),
        alpha: T.float32,
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(n_ch, bc), splits, threads=threads) as (bx, bs):
            prod = T.alloc_fragment((bc, bj), F32)
            row = T.alloc_fragment((bc,), F32)
            acc = T.alloc_fragment((bc,), F32)
            sc = T.alloc_fragment((bc,), F32)
            for i in T.Parallel(bc):
                sc[i] = alpha + gamma * T.cast(w[T.min(bx * bc + i, n_ch - 1)], F32)
            T.clear(acc)
            # Work items are (outer index, inner tile) pairs, split evenly.
            per = T.ceildiv(outer * tiles, splits)
            for it in T.serial(per):
                job = bs * per + it
                o = job // tiles
                i0 = (job % tiles) * bj
                for i, j in T.Parallel(bc, bj):
                    ch = bx * bc + i
                    ok = (job < outer * tiles) and (ch < n_ch) and (i0 + j < inner)
                    oo = T.min(o, outer - 1)
                    cc = T.min(ch, n_ch - 1)
                    jj = T.min(i0 + j, inner - 1)
                    gv = T.if_then_else(ok, T.cast(g[oo, cc, jj], F32), T.float32(0))
                    xv = T.if_then_else(ok, T.cast(x[oo, cc, jj], F32), T.float32(0))
                    prod[i, j] = gv * xv
                    if ok:
                        gx[oo, cc, jj] = T.cast(gv * sc[i], x_dt)
                T.reduce_sum(prod, row, dim=1)
                for i in T.Parallel(bc):
                    acc[i] += row[i]
            for i in T.Parallel(bc):
                if bx * bc + i < n_ch:
                    part[bs, bx * bc + i] = acc[i] * gamma

    return main


def _plan_fwd(dev, rows, cols, e):
    """(row tile, column tile, threads) configs over the (rows, cols) view,
    model-ranked, then torch and the ``.scale`` kernel."""
    space = [
        {"BR": br, "BQ": bq, "threads": th}
        for br in (1, 2, 4, 8, 16, 32)
        for bq in (64, 128, 256, 512, 1024)
        for th in (128, 256)
        if 1024 <= br * bq <= 8192 and bq <= max(64, cols) and br * bq % th == 0
    ]
    n = rows * cols

    def work(cf):
        br, bq = cf["BR"], cf["BQ"]
        used = min(bq, cols) / bq
        return Work(
            grid=cdiv(rows, br) * cdiv(cols, bq),
            warps=cf["threads"] // 32,
            acc_fp32=br * bq,
            operand_bytes=br * bq * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=2 * n * e,
            vector_flops=2.0 * n / used,
            phases=((2 * n * e, br * bq * e * used),),
        )

    return [*rank(dev, space, work), TORCH, SCALE]


def _plan_bwd(dev, outer, c, inner, e):
    if inner == 1:
        tiles = ((bc, bj) for bc in (32, 64, 128) for bj in (16, 32, 64))
    else:
        tiles = ((bc, bj) for bc in (1, 2, 4) for bj in (256, 512, 1024))
    space = [
        {"BC": bc, "BJ": bj, "OCC": occ, "threads": th}
        for bc, bj in tiles
        for occ in (2, 4)
        for th in (128, 256)
        if bc * bj % th == 0
    ]
    cols = outer * inner

    def work(cf):
        bc, bj = cf["BC"], cf["BJ"]
        splits = _splits(dev, outer, inner, c, cf)
        sector = max(1.0, 32 / (bc * e)) if inner == 1 else max(1.0, 32 / (bj * e))
        return Work(
            grid=cdiv(c, bc) * splits,
            warps=cf["threads"] // 32,
            acc_fp32=3 * bc * bj,
            operand_bytes=3 * bc * bj * e,
            smem_stage=0,
            stages=1,
            hbm_bytes=3 * c * cols * e * sector + 2 * splits * c * 4,
            vector_flops=3.0 * c * cols,
        )

    return [*rank(dev, space, work), TORCH, SCALE]


def _splits(dev, outer, inner, c, cfg):
    work = cdiv(outer, cfg["BJ"]) if inner == 1 else outer * cdiv(inner, cfg["BJ"])
    return max(1, min(work, cdiv(cfg["OCC"] * dev.units, cdiv(c, cfg["BC"]))))


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


def _kernel_dtypes(*tensors):
    return all(t.dtype in _KERNEL_DTYPES for t in tensors)


def _rows_cols(x, channel_axis):
    """(contiguous x, outer, C, inner, rows, cols): channels-last puts C on the
    columns of the (rows, cols) view, channels-first on the rows."""
    xc, outer, c, inner = _view3(x, channel_axis)
    rows, cols = (outer, c) if inner == 1 else (outer * c, inner)
    return xc, outer, c, inner, rows, cols


def _fwd_run(cfg, x, w, channel_axis, alpha, gamma):
    """x * (alpha + gamma * w[c]) under cfg, in the promoted dtype."""
    xc, _, c, inner, rows, cols = _rows_cols(x, channel_axis)
    wv = w.reshape(-1).contiguous()
    out_dtype = torch.promote_types(x.dtype, w.dtype)
    if cfg == TORCH:
        return (xc * (alpha + gamma * wv.reshape(_shape(x, channel_axis)))).to(
            out_dtype
        )
    if cfg == SCALE:
        return scale.ia3_fwd(xc, w, channel_axis, alpha, gamma)
    dts = (tl_dtype(x.dtype), tl_dtype(w.dtype), tl_dtype(out_dtype))
    args = (cols, c, cfg["BR"], cfg["BQ"], cfg["threads"], inner == 1, dts)
    kernel = compiled(_fwd_prim, x.device, *args)
    y = torch.empty(rows, cols, device=x.device, dtype=out_dtype)
    kernel(xc.view(rows, cols), wv, y, float(alpha), float(gamma))
    return y.view(x.shape)


def ia3_fwd(x, w, channel_axis: int, alpha: float = 1.0, gamma: float = 1.0):
    """x * (alpha + gamma * w[c]) in the promoted dtype of x and w."""
    if not _kernel_dtypes(x, w):
        return scale.ia3_fwd(x, w, channel_axis, alpha, gamma)
    dev = device_info(x.device)
    xc, _, _, inner, rows, cols = _rows_cols(x, channel_axis)
    key = (
        "tl.ia3.fwd",
        (dtype_key(x.dtype), dtype_key(w.dtype), inner == 1),
        (token_bucket(rows), cols),
    )
    args = (w, channel_axis, alpha, gamma)
    cfg = select(
        dev,
        key,
        lambda: _plan_fwd(dev, rows, cols, x.element_size()),
        measurer(
            dev,
            2 * x.numel() * x.element_size(),
            lambda: (copy_of(xc),),
            lambda cf, s: lambda: _fwd_run(cf, s[0], *args),
        ),
    )
    return guarded(dev, key, cfg, lambda cf: _fwd_run(cf, xc, *args), CompileError)


def _bwd_run(cfg, dev, grad, x, w, channel_axis, alpha, gamma):
    """(gx in x's dtype, gw fp32 (C,)) under cfg."""
    xc, outer, c, inner = _view3(x, channel_axis)
    g = grad.reshape(x.shape).contiguous()
    wv = w.reshape(-1).contiguous()
    if cfg == TORCH:
        gf = g.float()
        gx = gf * (alpha + gamma * wv.float().reshape(_shape(x, channel_axis)))
        dims = [d for d in range(x.dim()) if d != channel_axis % x.dim()]
        return gx.to(x.dtype), (gf * xc.float()).sum(dims) * gamma
    if cfg == SCALE:
        gx, gw = scale.ia3_bwd(g, xc, w, channel_axis, alpha, gamma)
        return gx.to(x.dtype), gw.reshape(-1).float()
    dts = (tl_dtype(g.dtype), tl_dtype(x.dtype), tl_dtype(w.dtype))
    splits = _splits(dev, outer, inner, c, cfg)
    if inner == 1:
        args = (c, cfg["BC"], cfg["BJ"], splits, cfg["threads"], dts)
        kernel = compiled(_bwd_last_prim, x.device, *args)
        shape = (outer, c)
    else:
        args = (c, inner, cfg["BC"], cfg["BJ"], splits, cfg["threads"], dts)
        kernel = compiled(_bwd_first_prim, x.device, *args)
        shape = (outer, c, inner)
    gx = torch.empty(shape, device=x.device, dtype=x.dtype)
    part = torch.empty(splits, c, device=x.device, dtype=torch.float32)
    kernel(g.view(shape), xc.view(shape), wv, gx, part, float(alpha), float(gamma))
    return gx.view(x.shape), (part[0] if splits == 1 else part.sum(0))


def ia3_bwd(grad, x, w, channel_axis: int, alpha: float = 1.0, gamma: float = 1.0):
    """(gx in x's dtype, gw in w's dtype) of :func:`ia3_fwd`."""
    if not _kernel_dtypes(grad, x, w):
        return scale.ia3_bwd(grad, x, w, channel_axis, alpha, gamma)
    dev = device_info(x.device)
    xc, outer, c, inner = _view3(x, channel_axis)
    g = grad.reshape(x.shape).contiguous()
    key = (
        "tl.ia3.bwd",
        (dtype_key(g.dtype), dtype_key(x.dtype), dtype_key(w.dtype), inner == 1),
        (token_bucket(outer * inner), c, inner),
    )
    args = (w, channel_axis, alpha, gamma)
    cfg = select(
        dev,
        key,
        lambda: _plan_bwd(dev, outer, c, inner, x.element_size()),
        measurer(
            dev,
            3 * x.numel() * x.element_size(),
            lambda: (copy_of(g), copy_of(xc)),
            lambda cf, s: lambda: _bwd_run(cf, dev, s[0], s[1], *args),
        ),
    )
    gx, gw = guarded(
        dev, key, cfg, lambda cf: _bwd_run(cf, dev, g, xc, *args), CompileError
    )
    return gx, gw.reshape(w.shape).to(w.dtype)
