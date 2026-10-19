"""LoKr bypass on the token-major GEMM kernels (``gemm_kernels``), for one adapter or a
group of G adapters that share their input: forward, gx with g_w1, and g_w2, each on
the model-ranked tuner against its torch composition. See docs/kernels/designs.md (F4b).
"""

import functools

import torch
import triton

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
from ...tuner.model import Work, cdiv, rank, shared_reads
from ..common import OutOfResources, dot_prec
from .gemm_kernels import _gemm_dx_kernel, _gemm_fwd_kernel, _gemm_gw2_kernel

GEMM = {"impl": "gemm"}
# Largest pow2-padded w1 side the mix tiles take.
MAX_W1 = 128
# Largest fp32 register tile per program (128 x 128).
_TILE = 16384
_ROWS = (16, 32, 64, 128, 256)
# Programs the g_w2 split-K grid aims for, per SM.
_GW2_PER_SM = 2
# The token tiles ride the grid's y axis (CUDA's 65535 bound).
_GRID_Y = 65535


def in_scope(a: int, b: int, c: int, d: int) -> bool:
    """Whether w1 (a, b), w2 (c, d) fit the GEMM layout's mix tiles."""
    return max(pow2(a), pow2(b)) <= MAX_W1 and min(a, b, c, d) >= 1


def _mixes(pa: int, pb: int) -> tuple[str, ...]:
    return ("batch", "diag") if min(pa, pb) >= 16 else ("diag",)


def _mix_legal(mix: str, bt: int, pa: int, pb: int) -> bool:
    ra, rb = bt * pa, bt * pb
    if ra not in _ROWS or rb not in _ROWS:
        return False
    if mix == "diag":
        return ra <= 128 and rb <= 128
    return min(pa, pb) >= 16


def _mix_flops(mix: str, tiles: float, bt: int, pa: int, pb: int, n: int) -> float:
    """MMA flops of the w1 mix over `tiles` (token tile, n columns) tiles."""
    if mix == "batch":
        return 2.0 * tiles * bt * pa * pb * n
    return 2.0 * tiles * (bt * pa) * (bt * pb) * n


@functools.cache
def _scale_tensor(scales: tuple[float, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(scales, dtype=torch.float32, device=device)


def _scales(scales, g: int, device) -> tuple[torch.Tensor | None, float]:
    """(per-group fp32 scale tensor or None, the shared scale); a group's tensor is
    built once per scale tuple and device (the scales are launch constants)."""
    if g == 1:
        return None, float(scales[0])
    return _scale_tensor(tuple(scales), device), 1.0


# ------------------------------------------------------------------- forward
def _plan_fwd(dev, t, g, a, b, c, d, e, has_base):
    pa, pb = pow2(a), pow2(b)
    space = [
        cf
        for cf in grid_configs(
            dev,
            MIX=_mixes(pa, pb),
            BT=(1, 2, 4, 8, 16, 32),
            BN=(32, 64, 128),
            BK=(16, 32, 64),
            num_warps=(4, 8),
        )
        if _mix_legal(cf["MIX"], cf["BT"], pa, pb)
        and cdiv(t, cf["BT"]) <= _GRID_Y
        and cf["BT"] * pb * cf["BN"] <= _TILE
        and cf["BT"] * pa * cf["BN"] <= _TILE
        and cf["BK"] <= max(16, pow2(d))
        and cf["BN"] <= max(32, pow2(c))
    ]

    def work(cf):
        bt, bn, bk = cf["BT"], cf["BN"], cf["BK"]
        ra, rb = bt * pa, bt * pb
        tt, tc = cdiv(t, bt), cdiv(c, bn)
        tiles = g * tt * tc
        x_bytes = t * b * d * e
        y_bytes = g * t * a * c * e * (2 if has_base else 1)
        w_bytes = g * (a * b + c * d) * e
        return Work(
            grid=tiles,
            warps=cf["num_warps"],
            acc_fp32=rb * bn + ra * bn,
            operand_bytes=(pa * pb if cf["MIX"] == "batch" else ra * rb) * e,
            smem_stage=(rb * bk + bk * bn) * e,
            stages=cf["num_stages"],
            hbm_bytes=x_bytes + y_bytes + shared_reads(dev, w_bytes, tiles),
            mma_flops=2.0 * tiles * rb * cdiv(d, bk) * bk * bn
            + _mix_flops(cf["MIX"], tiles, bt, pa, pb, bn),
            l2_bytes=x_bytes * g * tc + tiles * cdiv(d, bk) * bk * bn * e + y_bytes,
        )

    return [*rank(dev, space, work), TORCH]


def _torch_fwd(x, w1s, w2s, scales, bases, out_dtype):
    t = x.shape[0]
    g, a, b = w1s.shape
    c, d = w2s.shape[1:]
    x3 = x.reshape(t, b, d)
    ys = []
    for i in range(g):
        y = torch.matmul(w1s[i], x3 @ w2s[i].t()).reshape(t, a * c) * scales[i]
        ys.append(y if bases is None else bases[i] + y)
    return torch.stack(ys).to(out_dtype)


def _fwd_run(cfg, x, w1s, w2s, scales, bases, out_dtype):
    """(G, T, a*c): [base +] scale_g * vec(w1_g @ X_t @ w2_g^T)."""
    if cfg == TORCH:
        return _torch_fwd(x, w1s, w2s, scales, bases, out_dtype)
    t = x.shape[0]
    g, a, b = w1s.shape
    c, d = w2s.shape[1:]
    out = torch.empty(g, t, a * c, device=x.device, dtype=out_dtype)
    bb = bases if bases is not None else out
    sg, s = _scales(scales, g, x.device)
    n_ct = triton.cdiv(c, cfg["BN"])
    grid = (g * n_ct, triton.cdiv(t, cfg["BT"]))
    _gemm_fwd_kernel[grid](
        x,
        w1s,
        w2s,
        bb,
        out,
        sg if sg is not None else out,
        t,
        a,
        b,
        c,
        d,
        n_ct,
        *x.stride(),
        *w1s.stride(),
        *w2s.stride(),
        *bb.stride(),
        *out.stride(),
        s,
        has_base=bases is not None,
        group_scale=sg is not None,
        prec=dot_prec(x.dtype),
        MIX=cfg["MIX"],
        BT=cfg["BT"],
        PA=pow2(a),
        PB=pow2(b),
        BN=cfg["BN"],
        BK=cfg["BK"],
        **launch(cfg),
    )
    return out


def lokr_gemm_fwd(x, w1s, w2s, scales, bases=None, out_dtype=None):
    """(G, T, a*c) = [bases +] scales[g] * x @ kron(w1s[g], w2s[g])^T for x (T, b*d),
    w1s (G, a, b), w2s (G, c, d), bases (G, T, a*c) or None; one compute dtype."""
    out_dtype = out_dtype or (bases.dtype if bases is not None else x.dtype)
    dev = device_info(x.device)
    t = x.shape[0]
    g, a, b = w1s.shape
    c, d = w2s.shape[1:]
    scales = tuple(float(s) for s in scales)
    key = (
        "lokr.gemm_fwd",
        (dtype_key(x.dtype), dtype_key(out_dtype), bases is not None),
        (token_bucket(t), g, a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_fwd(dev, t, g, a, b, c, d, x.element_size(), bases is not None),
        measurer(
            dev,
            t * (b * d + g * a * c) * x.element_size(),
            lambda: (copy_of(x), copy_of(bases)),
            lambda cf, s: lambda: _fwd_run(cf, s[0], w1s, w2s, scales, s[1], out_dtype),
            x.dtype,
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _fwd_run(cf, x, w1s, w2s, scales, bases, out_dtype),
        OutOfResources,
    )


# --------------------------------------------------------- gx and g_w1
def _plan_dx(dev, t, g, a, b, c, d, e):
    pa, pb = pow2(a), pow2(b)
    space = [
        cf
        for cf in grid_configs(
            dev,
            MIX=_mixes(pa, pb),
            BT=(1, 2, 4, 8, 16, 32),
            BD=(32, 64, 128),
            BK=(16, 32, 64),
            num_warps=(4, 8),
        )
        if _mix_legal(cf["MIX"], cf["BT"], pa, pb)
        and cdiv(t, cf["BT"]) <= _GRID_Y
        and cf["BT"] * pa * cf["BD"] <= _TILE
        and cf["BT"] * pb * cf["BD"] <= _TILE
        and cf["BK"] <= max(16, pow2(c))
        and cf["BD"] <= max(32, pow2(d))
    ]

    def work(cf):
        bt, bd, bk = cf["BT"], cf["BD"], cf["BK"]
        ra, rb = bt * pa, bt * pb
        tt, td = cdiv(t, bt), cdiv(d, bd)
        tiles = tt * td
        g_bytes = g * t * a * c * e
        x_bytes = t * b * d * e
        return Work(
            grid=tiles,
            warps=cf["num_warps"],
            acc_fp32=2 * rb * bd + ra * bd,
            operand_bytes=rb * bd * e,
            smem_stage=(ra * bk + bk * bd) * e,
            stages=cf["num_stages"],
            hbm_bytes=g_bytes + 2 * x_bytes + tiles * g * pa * pb * 4,
            mma_flops=g * 2.0 * tiles * ra * cdiv(c, bk) * bk * bd
            + 2 * g * _mix_flops(cf["MIX"], tiles, bt, pa, pb, bd),
            l2_bytes=g_bytes * td + 2 * x_bytes + g * tiles * cdiv(c, bk) * bk * bd * e,
        )

    return [*rank(dev, space, work), TORCH]


def _torch_dx(grad, x, w1s, w2s, scales):
    t = x.shape[0]
    g, a, b = w1s.shape
    c, d = w2s.shape[1:]
    x3 = x.reshape(t, b, d)
    gx = torch.zeros(t, b * d, device=x.device, dtype=torch.float32)
    gw1 = []
    for i in range(g):
        g3 = grad[i].reshape(t, a, c)
        n = torch.matmul(w1s[i].t(), g3)
        gx += (n @ w2s[i]).reshape(t, b * d).float() * scales[i]
        mid = x3 @ w2s[i].t()
        gw1.append((g3 @ mid.transpose(1, 2)).sum(0).float() * scales[i])
    return gx.to(x.dtype), torch.stack(gw1)


def _dx_run(cfg, grad, x, w1s, w2s, scales):
    """(gx (T, b*d) in x's dtype, g_w1 (G, a, b) fp32) for grad (G, T, a*c)."""
    if cfg == TORCH:
        return _torch_dx(grad, x, w1s, w2s, scales)
    t = x.shape[0]
    g, a, b = w1s.shape
    c, d = w2s.shape[1:]
    pa, pb = pow2(a), pow2(b)
    grid = (triton.cdiv(d, cfg["BD"]), triton.cdiv(t, cfg["BT"]))
    n_prog = grid[0] * grid[1]
    gx = torch.empty_like(x)
    part = torch.empty(g, n_prog, pa, pb, device=x.device, dtype=torch.float32)
    sg, s = _scales(scales, g, x.device)
    _gemm_dx_kernel[grid](
        grad,
        x,
        w1s,
        w2s,
        sg if sg is not None else part,
        gx,
        part,
        t,
        a,
        b,
        c,
        d,
        g,
        n_prog,
        *grad.stride(),
        *x.stride(),
        *w1s.stride(),
        *w2s.stride(),
        *gx.stride(),
        s,
        want_gw1=True,
        group_scale=sg is not None,
        prec=dot_prec(x.dtype),
        MIX=cfg["MIX"],
        BT=cfg["BT"],
        PA=pa,
        PB=pb,
        BD=cfg["BD"],
        BK=cfg["BK"],
        **launch(cfg),
    )
    return gx, part.sum(1)[:, :a, :b]


def lokr_gemm_dx(grad, x, w1s, w2s, scales):
    """(gx, g_w1 fp32 (G, a, b)) of the group bypass for grad (G, T, a*c)."""
    dev = device_info(x.device)
    t = x.shape[0]
    g, a, b = w1s.shape
    c, d = w2s.shape[1:]
    scales = tuple(float(s) for s in scales)
    key = (
        "lokr.gemm_dx",
        (dtype_key(x.dtype),),
        (token_bucket(t), g, a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_dx(dev, t, g, a, b, c, d, x.element_size()),
        measurer(
            dev,
            t * (g * a * c + 2 * b * d) * x.element_size(),
            lambda: (copy_of(grad),),
            lambda cf, s: lambda: _dx_run(cf, s[0], x, w1s, w2s, scales),
            x.dtype,
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _dx_run(cf, grad, x, w1s, w2s, scales),
        OutOfResources,
    )


# ------------------------------------------------------------------- g_w2
def _splits(dev, out_tiles: int, token_tiles: int) -> int:
    """Token splits so the grid holds about _GW2_PER_SM programs per SM."""
    want = max(1, (_GW2_PER_SM * dev.units) // max(out_tiles, 1))
    return max(1, min(want, token_tiles))


def _plan_gw2(dev, t, g, a, b, c, d, e):
    pa, pb = pow2(a), pow2(b)
    space = [
        cf
        for cf in grid_configs(
            dev,
            MIX=_mixes(pa, pb),
            BT=(1, 2, 4, 8, 16, 32),
            BC=(32, 64, 128),
            BD=(32, 64, 128),
            num_warps=(4, 8),
        )
        if _mix_legal(cf["MIX"], cf["BT"], pa, pb)
        and cf["BC"] * cf["BD"] <= _TILE
        and cf["BT"] * pb * cf["BC"] <= _TILE
        and cf["BC"] <= max(32, pow2(c))
        and cf["BD"] <= max(32, pow2(d))
    ]

    def work(cf):
        bt, bc, bd = cf["BT"], cf["BC"], cf["BD"]
        ra, rb = bt * pa, bt * pb
        tt, tc, td = cdiv(t, bt), cdiv(c, bc), cdiv(d, bd)
        splits = _splits(dev, g * tc * td, tt)
        g_bytes = g * t * a * c * e
        x_bytes = t * b * d * e
        return Work(
            grid=g * tc * td * splits,
            warps=cf["num_warps"],
            acc_fp32=bc * bd + rb * bc,
            operand_bytes=(pa * pb if cf["MIX"] == "batch" else ra * rb) * e,
            smem_stage=(ra * bc + rb * bd) * e,
            stages=cf["num_stages"],
            hbm_bytes=g_bytes * td + x_bytes * g * tc + splits * g * c * d * 4,
            mma_flops=2.0 * g * tt * tc * td * rb * bc * bd
            + _mix_flops(cf["MIX"], g * tt * tc * td, bt, pa, pb, bc),
        )

    return [*rank(dev, space, work), TORCH]


def _torch_gw2(grad, x, w1s, scales):
    t = x.shape[0]
    g, a, b = w1s.shape
    c = grad.shape[2] // a
    d = x.shape[1] // b
    x2 = x.reshape(t * b, d)
    out = []
    for i in range(g):
        n = torch.matmul(w1s[i].t(), grad[i].reshape(t, a, c)).reshape(t * b, c)
        out.append((n.t() @ x2).float() * scales[i])
    return torch.stack(out)


def _gw2_run(cfg, grad, x, w1s, c, d, scales):
    """g_w2 (G, c, d) fp32 for grad (G, T, a*c)."""
    if cfg == TORCH:
        return _torch_gw2(grad, x, w1s, scales)
    dev = device_info(x.device)
    t = x.shape[0]
    g, a, b = w1s.shape
    n_ct, n_dt = triton.cdiv(c, cfg["BC"]), triton.cdiv(d, cfg["BD"])
    splits = _splits(dev, g * n_ct * n_dt, triton.cdiv(t, cfg["BT"]))
    part = torch.empty(splits, g, c, d, device=x.device, dtype=torch.float32)
    sg, s = _scales(scales, g, x.device)
    _gemm_gw2_kernel[(g * n_ct * n_dt, splits)](
        grad,
        x,
        w1s,
        sg if sg is not None else part,
        part,
        t,
        a,
        b,
        c,
        d,
        g,
        n_ct,
        n_dt,
        *grad.stride(),
        *x.stride(),
        *w1s.stride(),
        s,
        group_scale=sg is not None,
        prec=dot_prec(x.dtype),
        MIX=cfg["MIX"],
        BT=cfg["BT"],
        PA=pow2(a),
        PB=pow2(b),
        BC=cfg["BC"],
        BD=cfg["BD"],
        **launch(cfg),
    )
    return part.sum(0)


def lokr_gemm_gw2(grad, x, w1s, c, d, scales):
    """g_w2 fp32 (G, c, d) of the group bypass for grad (G, T, a*c)."""
    dev = device_info(x.device)
    t = x.shape[0]
    g, a, b = w1s.shape
    scales = tuple(float(s) for s in scales)
    key = (
        "lokr.gemm_gw2",
        (dtype_key(x.dtype),),
        (token_bucket(t), g, a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_gw2(dev, t, g, a, b, c, d, x.element_size()),
        measurer(
            dev,
            t * (g * a * c + b * d) * x.element_size(),
            lambda: (copy_of(grad),),
            lambda cf, s: lambda: _gw2_run(cf, s[0], x, w1s, c, d, scales),
            x.dtype,
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _gw2_run(cf, grad, x, w1s, c, d, scales),
        OutOfResources,
    )


# ------------------------------------------------------------- entry points
def lokr_gemm_bwd(grad, x, w1s, w2s, scales):
    """(gx, g_w1 (G, a, b), g_w2 (G, c, d)) of the group bypass, the factor grads in
    the factors' dtype, for grad (G, T, a*c)."""
    c, d = w2s.shape[1:]
    gx, gw1 = lokr_gemm_dx(grad, x, w1s, w2s, scales)
    gw2 = lokr_gemm_gw2(grad, x, w1s, c, d, scales)
    return gx, gw1.to(w1s.dtype), gw2.to(w2s.dtype)


def fwd_single(x, w1, w2, scale, base, out_dtype):
    """The single-adapter forward: (T, a*c)."""
    bases = None if base is None else base.unsqueeze(0)
    return lokr_gemm_fwd(x, w1[None], w2[None], (scale,), bases, out_dtype)[0]


def bwd_single(grad, x, w1, w2, scale):
    """The single-adapter backward: (gx, g_w1, g_w2) in the compute dtype."""
    gx, gw1, gw2 = lokr_gemm_bwd(grad[None], x, w1[None], w2[None], (scale,))
    return gx, gw1[0], gw2[0]
