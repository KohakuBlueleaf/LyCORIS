"""LoKr bypass forward Y_t = [base +] gamma * w1 @ X_t @ w2^T on the
model-ranked tuner, the Kronecker product never built (backward: ``.grad``).
See docs/kernels/designs.md.
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
    stage_axis,
    token_bucket,
)
from ...tuner.model import Work, cdiv, rank, shared_reads
from ..common import OutOfResources, dot_prec, load2d
from . import bypass

# Largest fp32 register tile per program (128 x 128).
_TILE_BUDGET = 16384
ATOMIC = {"impl": "atomic"}


@triton.jit
def _bypass_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    sxt,
    sxi,
    s1a,
    s1b,
    s2c,
    s2d,
    sbt,
    sbo,
    syt,
    syo,
    scale,
    has_base: tl.constexpr,
    prec: tl.constexpr,
    block_t: tl.constexpr,
    block_v: tl.constexpr,
    block_d: tl.constexpr,
    block_c: tl.constexpr,
    block_a: tl.constexpr,
):
    """Y[t, u*n_c+c] = [base +] scale * sum_v w1[u,v] sum_d X[t, v*n_d+d] w2[c,d], w1 as a block-diagonal tile.

    Grid: (token tiles, w1 row tiles, w2 row tiles).
    """
    pid_t = tl.program_id(0)
    pid_a = tl.program_id(1)
    pid_c = tl.program_id(2)

    # mid rows are (token, v), output rows (token, u).
    kv = tl.arange(0, block_t * block_v)
    tok = pid_t * block_t + kv // block_v
    kt = kv // block_v
    vl = kv % block_v
    ru2 = tl.arange(0, block_t * block_a)
    ut = ru2 // block_a
    ru = pid_a * block_a + ru2 % block_a
    rc = pid_c * block_c + tl.arange(0, block_c)
    mtok = tok < n_tok
    mc = rc < n_c
    mu = ru < n_a
    ct = x_ptr.dtype.element_ty

    # acc rows (token, u), columns c: the output tile, reduced over w1's columns v.
    acc = tl.zeros((block_t * block_a, block_c), tl.float32)
    for v0 in range(0, n_b, block_v):
        rv = v0 + vl
        mv = rv < n_b

        # mid rows (token, v), columns c: M = X @ w2^T, reduced over d.
        mid = tl.zeros((block_t * block_v, block_c), tl.float32)
        for d0 in range(0, n_d, block_d):
            rd = d0 + tl.arange(0, block_d)
            md = rd < n_d
            xv = tl.load(
                x_ptr + tok[:, None] * sxt + (rv[:, None] * n_d + rd[None, :]) * sxi,
                mask=(mtok & mv)[:, None] & md[None, :],
                other=0.0,
            )
            w2t = load2d(w2_ptr, rd, rc, s2d, s2c, md, mc)
            mid = tl.dot(xv, w2t, mid, input_precision=prec)
        # w1 on the token-diagonal blocks, zero elsewhere.
        w1blk = tl.load(
            w1_ptr + ru[:, None] * s1a + rv[None, :] * s1b,
            mask=(ut[:, None] == kt[None, :]) & mu[:, None] & mv[None, :],
            other=0.0,
        )
        # mid in x's dtype, as eager's chain rounds it.
        acc = tl.dot(w1blk.to(ct), mid.to(ct), acc, input_precision=prec)

    # Epilogue: scale, add base, store rows (token, u) at columns u*n_c + c.
    tok2 = pid_t * block_t + ut
    omask = ((tok2 < n_tok) & mu)[:, None] & mc[None, :]
    col = ru[:, None] * n_c + rc[None, :]
    y = acc * scale
    if has_base:
        base = tl.load(b_ptr + tok2[:, None] * sbt + col * sbo, mask=omask, other=0.0)
        y += base.to(tl.float32)
    tl.store(
        y_ptr + tok2[:, None] * syt + col * syo,
        y.to(y_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _bypass_batch_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    b_ptr,
    y_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    sxt,
    sxi,
    s1a,
    s1b,
    s2c,
    s2d,
    sbt,
    sbo,
    syt,
    syo,
    scale,
    has_base: tl.constexpr,
    prec: tl.constexpr,
    block_t: tl.constexpr,
    block_v: tl.constexpr,
    block_d: tl.constexpr,
    block_c: tl.constexpr,
    block_a: tl.constexpr,
):
    """The same product as _bypass_kernel with tokens as the batch of 3-D dots.

    Grid: (token tiles, w1 row tiles, w2 row tiles).
    """
    pid_t = tl.program_id(0)
    pid_a = tl.program_id(1)
    pid_c = tl.program_id(2)
    tok = pid_t * block_t + tl.arange(0, block_t)
    ru = pid_a * block_a + tl.arange(0, block_a)
    rc = pid_c * block_c + tl.arange(0, block_c)
    mtok = tok < n_tok
    mu = ru < n_a
    mc = rc < n_c
    ct = x_ptr.dtype.element_ty

    # acc (token, u, c): the output tile, reduced over w1's columns v.
    acc = tl.zeros((block_t, block_a, block_c), tl.float32)
    for v0 in range(0, n_b, block_v):
        rv = v0 + tl.arange(0, block_v)
        mv = rv < n_b

        # mid (token, v, c) = X_t @ w2^T per token, reduced over d.
        mid = tl.zeros((block_t, block_v, block_c), tl.float32)
        for d0 in range(0, n_d, block_d):
            rd = d0 + tl.arange(0, block_d)
            md = rd < n_d
            xv = tl.load(
                x_ptr
                + tok[:, None, None] * sxt
                + (rv[None, :, None] * n_d + rd[None, None, :]) * sxi,
                mask=mtok[:, None, None] & mv[None, :, None] & md[None, None, :],
                other=0.0,
            )
            w2t = load2d(w2_ptr, rd, rc, s2d, s2c, md, mc)
            mid = tl.dot(
                xv,
                tl.broadcast_to(w2t[None, :, :], (block_t, block_d, block_c)),
                mid,
                input_precision=prec,
            )

        # acc += w1 tile @ mid per token, mid rounded to x's dtype.
        w1 = load2d(w1_ptr, ru, rv, s1a, s1b, mu, mv)
        w1b = tl.broadcast_to(w1.to(ct)[None, :, :], (block_t, block_a, block_v))
        acc = tl.dot(w1b, mid.to(ct), acc, input_precision=prec)

    # Epilogue: scale, add base, store (token, u*n_c + c).
    omask = mtok[:, None, None] & mu[None, :, None] & mc[None, None, :]
    col = ru[None, :, None] * n_c + rc[None, None, :]
    y = acc * scale
    if has_base:
        base = tl.load(
            b_ptr + tok[:, None, None] * sbt + col * sbo, mask=omask, other=0.0
        )
        y += base.to(tl.float32)
    tl.store(
        y_ptr + tok[:, None, None] * syt + col * syo,
        y.to(y_ptr.dtype.element_ty),
        mask=omask,
    )


def _legal(cfg):
    bt, bv, bc, ba, bd = cfg["BT"], cfg["BV"], cfg["BC"], cfg["BA"], cfg["BD"]
    if cfg["LAYOUT"] == "batch":
        if min(bv, ba, bc, bd) < 16:
            return False
        tiles = (bt * bv * bc, bt * bv * bd, bt * ba * bc, bt * ba * bv)
    else:
        if bt * bv < 16 or bt * ba < 16:
            return False
        tiles = (bt * bv * bc, bt * bv * bd, bt * ba * bc, bt * ba * bt * bv)
    return max(tiles) <= _TILE_BUDGET


def _factor_blocks(n):
    return tuple(sorted({min(pow2(n), v) for v in (4, 8, 16, 32, 64, 128)}))


def _plan(dev, t, a, b, c, d, e, has_base, alternatives):
    """Both layouts' legal configs, model-ranked, then the alternatives."""
    stages = tuple(sorted({1, *stage_axis(dev)}))
    space = [
        cfg
        for cfg in grid_configs(
            dev,
            LAYOUT=("diag", "batch"),
            BT=(1, 2, 4, 8, 16),
            BV=_factor_blocks(b),
            BD=(16, 32, 64),
            BC=(16, 32, 64, 128),
            BA=_factor_blocks(a),
            num_warps=(2, 4, 8),
            num_stages=stages,
        )
        if _legal(cfg)
    ]

    def work(cf):
        bt, bv, bd, bc, ba = cf["BT"], cf["BV"], cf["BD"], cf["BC"], cf["BA"]
        tt, ta, tc, tv = cdiv(t, bt), cdiv(a, ba), cdiv(c, bc), cdiv(b, bv)
        mid = 2.0 * tt * bt * tv * bv * cdiv(d, bd) * bd * tc * bc * ta
        if cf["LAYOUT"] == "diag":
            w1dot = 2.0 * tt * ta * tc * tv * (bt * ba) * (bt * bv) * bc
            w1reg = bt * ba * bt * bv
        else:
            w1dot = 2.0 * tt * ta * tc * tv * bt * ba * bv * bc
            w1reg = ba * bv
        return Work(
            grid=tt * ta * tc,
            warps=cf["num_warps"],
            acc_fp32=bt * bv * bc + bt * ba * bc,
            operand_bytes=w1reg * e,
            smem_stage=(bt * bv * bd + bd * bc) * e,
            stages=cf["num_stages"],
            hbm_bytes=t * b * d * e * ta * tc
            + t * a * c * e * (2 if has_base else 1)
            + shared_reads(dev, (a * b + c * d) * e, tt * ta * tc),
            mma_flops=mid + w1dot,
            # each program reads its x rows, w2's rows once per v block, and its w1 rows
            l2_bytes=t * b * d * e * ta * tc
            + tt * ta * tc * (tv * bc * d + ba * b) * e
            + t * a * c * e * (2 if has_base else 1),
        )

    return [*rank(dev, space, work), *alternatives]


def _apply_run(cfg, flat, w1, w2, scale, base2, out_dtype):
    """[base +] scale * vec(w1 @ X_t @ w2^T); w1 / w2 may be strided views."""
    a, b = w1.shape
    c, d = w2.shape
    t = flat.shape[0]
    if cfg == TORCH:
        x3 = flat.reshape(t, b, d)
        y = torch.matmul(w1, x3 @ w2.t()).reshape(t, a * c) * scale
        return (y if base2 is None else base2 + y).to(out_dtype)
    if cfg == ATOMIC:
        y = bypass.lokr_bypass_fwd(flat, w1.contiguous(), w2.contiguous(), scale)
        return (y if base2 is None else base2 + y).to(out_dtype)
    out = torch.empty(t, a * c, device=flat.device, dtype=out_dtype)
    bb = base2 if base2 is not None else out
    grid = (
        triton.cdiv(t, cfg["BT"]),
        triton.cdiv(a, cfg["BA"]),
        triton.cdiv(c, cfg["BC"]),
    )
    kernel = _bypass_batch_kernel if cfg["LAYOUT"] == "batch" else _bypass_kernel
    kernel[grid](
        flat,
        w1,
        w2,
        bb,
        out,
        t,
        a,
        b,
        c,
        d,
        *flat.stride(),
        *w1.stride(),
        *w2.stride(),
        *bb.stride(),
        *out.stride(),
        scale,
        has_base=base2 is not None,
        prec=dot_prec(flat.dtype),
        block_t=cfg["BT"],
        block_v=cfg["BV"],
        block_d=cfg["BD"],
        block_c=cfg["BC"],
        block_a=cfg["BA"],
        **launch(cfg),
    )
    return out


def _apply(name, flat, w1, w2, scale, base2, out_dtype, alternatives):
    dev = device_info(flat.device)
    a, b = w1.shape
    c, d = w2.shape
    t = flat.shape[0]
    key = (
        name,
        (dtype_key(flat.dtype), dtype_key(out_dtype), base2 is not None),
        (token_bucket(t), a, b, c, d),
    )

    def inputs():
        return copy_of(flat), copy_of(base2)

    cfg = select(
        dev,
        key,
        lambda: _plan(
            dev, t, a, b, c, d, flat.element_size(), base2 is not None, alternatives
        ),
        measurer(
            dev,
            t * (b * d + a * c) * flat.element_size(),
            inputs,
            lambda cf, s: lambda: _apply_run(cf, s[0], w1, w2, scale, s[1], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _apply_run(cf, flat, w1, w2, scale, base2, out_dtype),
        OutOfResources,
    )


def lokr_bypass_fwd(x, w1, w2, gamma: float = 1.0, base=None):
    """[base +] gamma * x @ kron(w1, w2)^T for x (T, b*d), w1 (a, b), w2 (c, d)
    in one compute dtype; the result in base's dtype when given, else x's."""
    out_dtype = base.dtype if base is not None else x.dtype
    small = bypass.kron_factors_supported(*w1.shape, *w2.shape)
    alternatives = [TORCH, ATOMIC] if small else [TORCH]
    return _apply("lokr.bypass_fwd", x, w1, w2, gamma, base, out_dtype, alternatives)
