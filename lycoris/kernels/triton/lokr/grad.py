"""LoKr bypass backward (gx, g_w1, g_w2) on the model-ranked tuner.
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
    token_bucket,
)
from ...tuner.model import Work, cdiv, rank, shared_reads
from ..common import OutOfResources, dot_prec, load2d
from . import apply, bypass

DXN = {"impl": "dxn"}
APPLY = {"impl": "apply"}
ATOMIC = apply.ATOMIC
# Rows a program's block-diagonal tile may span: BT*a and BT*b.
MAX_ROWS = 128


@triton.jit
def _dxn_kernel(
    g_ptr,
    w1_ptr,
    w2_ptr,
    gx_ptr,
    n_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    s1a,
    s1b,
    s2c,
    s2d,
    scale,
    prec: tl.constexpr,
    block_t: tl.constexpr,
    pa: tl.constexpr,
    pb: tl.constexpr,
    block_c: tl.constexpr,
    block_d: tl.constexpr,
):
    """gx rows (token, v) = scale * (blockdiag(w1^T) @ G) @ w2 for one token
    tile and one gx column tile; N rows (token, v) stored by column tile 0.

    Grid: (token tiles, gx column tiles); g is (t, a*c), gx (t, b*d), n (t, b*c).
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    # rows (token, v) of N and gx; columns (token, u) of the block-diagonal tile
    rv = tl.arange(0, block_t * pb)
    tv = rv // pb
    vv = rv % pb
    ru = tl.arange(0, block_t * pa)
    tu = ru // pa
    uu = ru % pa
    tok_v = pid_t * block_t + tv
    tok_u = pid_t * block_t + tu
    mv = (tok_v < n_tok) & (vv < n_b)
    mu = (tok_u < n_tok) & (uu < n_a)
    rd = pid_d * block_d + tl.arange(0, block_d)
    md = rd < n_d
    ct = g_ptr.dtype.element_ty

    # w1^T on the token-diagonal blocks: [(t, v), (t', u)] = w1[u, v] if t == t'.
    w1bdt = tl.load(
        w1_ptr + uu[None, :] * s1a + vv[:, None] * s1b,
        mask=(tv[:, None] == tu[None, :]) & (vv < n_b)[:, None] & (uu < n_a)[None, :],
        other=0.0,
    ).to(ct)
    acc = tl.zeros((block_t * pb, block_d), tl.float32)
    for c0 in range(0, n_c, block_c):
        rc = c0 + tl.arange(0, block_c)
        mc = rc < n_c
        # G rows (token, u), columns c of this chunk.
        gv = tl.load(
            g_ptr + tok_u[:, None] * (n_a * n_c) + (uu[:, None] * n_c + rc[None, :]),
            mask=mu[:, None] & mc[None, :],
            other=0.0,
        )
        nt = tl.dot(w1bdt, gv, input_precision=prec).to(ct)
        if pid_d == 0:
            tl.store(
                n_ptr
                + tok_v[:, None] * (n_b * n_c)
                + (vv[:, None] * n_c + rc[None, :]),
                nt,
                mask=mv[:, None] & mc[None, :],
            )
        w2v = load2d(w2_ptr, rc, rd, s2c, s2d, mc, md)
        acc = tl.dot(nt, w2v, acc, input_precision=prec)
    tl.store(
        gx_ptr + tok_v[:, None] * (n_b * n_d) + (vv[:, None] * n_d + rd[None, :]),
        (acc * scale).to(gx_ptr.dtype.element_ty),
        mask=mv[:, None] & md[None, :],
    )


@triton.jit
def _gw1_kernel(
    g_ptr,
    m_ptr,
    part_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    prec: tl.constexpr,
    block_t: tl.constexpr,
    pa: tl.constexpr,
    pb: tl.constexpr,
    block_c: tl.constexpr,
):
    """part[tile] = sum over the tile's tokens of G_t mid_t^T (fp32 (pa, pb)):
    the token-diagonal blocks of the (BT*a, BT*b) tile product over c.

    Grid: (token tiles,); g is (t, a*c), mid (t, b*c).
    """
    pid = tl.program_id(0)
    ru = tl.arange(0, block_t * pa)
    tu = ru // pa
    uu = ru % pa
    rv = tl.arange(0, block_t * pb)
    tv = rv // pb
    vv = rv % pb
    tok_u = pid * block_t + tu
    tok_v = pid * block_t + tv
    mu = (tok_u < n_tok) & (uu < n_a)
    mv = (tok_v < n_tok) & (vv < n_b)
    z = tl.zeros((block_t * pa, block_t * pb), tl.float32)
    for c0 in range(0, n_c, block_c):
        rc = c0 + tl.arange(0, block_c)
        mc = rc < n_c
        gv = tl.load(
            g_ptr + tok_u[:, None] * (n_a * n_c) + (uu[:, None] * n_c + rc[None, :]),
            mask=mu[:, None] & mc[None, :],
            other=0.0,
        )
        mt = tl.load(
            m_ptr + tok_v[None, :] * (n_b * n_c) + (vv[None, :] * n_c + rc[:, None]),
            mask=mv[None, :] & mc[:, None],
            other=0.0,
        )
        z = tl.dot(gv, mt, z, input_precision=prec)
    # Keep the token-diagonal blocks and sum them over the tile's tokens.
    z = tl.where(tu[:, None] == tv[None, :], z, 0.0)
    z4 = tl.reshape(z, (block_t, pa, block_t, pb))
    blocks = tl.sum(tl.sum(z4, axis=2), axis=0)
    ra = tl.arange(0, pa)
    rb = tl.arange(0, pb)
    tl.store(
        part_ptr + pid * (pa * pb) + ra[:, None] * pb + rb[None, :],
        blocks,
    )


def _tiles(a, b):
    """(pa, pb, block_t values) legal for the block-diagonal tiles."""
    pa, pb = pow2(a), pow2(b)
    bts = [
        bt
        for bt in (1, 2, 4, 8, 16, 32)
        if 16 <= bt * min(pa, pb) and bt * max(pa, pb) <= MAX_ROWS
    ]
    return pa, pb, bts


def _plan_dxn(dev, t, a, b, c, d, e):
    pa, pb, bts = _tiles(a, b)
    if not bts:
        return [TORCH]
    space = grid_configs(
        dev,
        BT=tuple(bts),
        BC=(16, 32, 64),
        BD=(32, 64, 128),
        num_warps=(2, 4, 8),
        num_stages=(1, 2),
    )

    def work(cf):
        bt, bc, bd = cf["BT"], cf["BC"], cf["BD"]
        ra, rb = bt * pa, bt * pb
        tt, td = cdiv(t, bt), cdiv(d, bd)
        return Work(
            grid=tt * td,
            warps=cf["num_warps"],
            acc_fp32=rb * bc + rb * bd,
            operand_bytes=rb * ra * e,
            smem_stage=(ra * bc + bc * bd) * e,
            stages=cf["num_stages"],
            hbm_bytes=e * (t * a * c * td + t * b * d + t * b * c)
            + shared_reads(dev, (c * d + a * b) * e, tt * td),
            mma_flops=2.0 * tt * td * cdiv(c, bc) * bc * (rb * ra + rb * bd),
            # every gx column tile re-reads the token tile's G
            l2_bytes=e * (t * a * c * td + tt * td * c * bd + t * b * d),
        )

    return [*rank(dev, space, work), TORCH]


def _dxn_run(cfg, g, w1, w2, scale, out_dtype):
    """(gx (t, b*d) in out_dtype, N (t*b, c)) for grad g (t, a*c)."""
    a, b = w1.shape
    c, d = w2.shape
    t = g.shape[0]
    if cfg == TORCH:
        n = torch.matmul(w1.t(), g.reshape(t, a, c)).reshape(t * b, c)
        gx = torch.mm(n, w2).mul_(scale)
        return gx.reshape(t, b * d).to(out_dtype), n
    pa, pb = pow2(a), pow2(b)
    gx = torch.empty(t, b * d, device=g.device, dtype=out_dtype)
    n = torch.empty(t * b, c, device=g.device, dtype=g.dtype)
    grid = (triton.cdiv(t, cfg["BT"]), triton.cdiv(d, cfg["BD"]))
    _dxn_kernel[grid](
        g,
        w1,
        w2,
        gx,
        n,
        t,
        a,
        b,
        c,
        d,
        *w1.stride(),
        *w2.stride(),
        scale,
        prec=dot_prec(g.dtype),
        block_t=cfg["BT"],
        pa=pa,
        pb=pb,
        block_c=cfg["BC"],
        block_d=cfg["BD"],
        **launch(cfg),
    )
    return gx, n


def _dxn(g, w1, w2, scale, out_dtype):
    dev = device_info(g.device)
    a, b = w1.shape
    c, d = w2.shape
    t = g.shape[0]
    e = g.element_size()
    key = (
        "lokr.bypass_dxn",
        (dtype_key(g.dtype), dtype_key(out_dtype)),
        (token_bucket(t), a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan_dxn(dev, t, a, b, c, d, e),
        measurer(
            dev,
            t * (a * c + 2 * b * d) * e,
            lambda: (copy_of(g),),
            lambda cf, s: lambda: _dxn_run(cf, s[0], w1, w2, scale, out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _dxn_run(cf, g, w1, w2, scale, out_dtype),
        OutOfResources,
    )


def _plan_gw1(dev, t, a, b, c, e):
    pa, pb, bts = _tiles(a, b)
    if not bts:
        return [TORCH]
    space = grid_configs(
        dev,
        BT=tuple(bts),
        BC=(16, 32, 64, 128),
        num_warps=(2, 4, 8),
        num_stages=(1, 2),
    )

    def work(cf):
        bt, bc = cf["BT"], cf["BC"]
        ra, rb = bt * pa, bt * pb
        tt = cdiv(t, bt)
        # G and mid read once; the cross-token blocks of each tile product are waste
        return Work(
            grid=tt,
            warps=cf["num_warps"],
            acc_fp32=ra * rb,
            operand_bytes=0,
            smem_stage=(ra + rb) * bc * e,
            stages=cf["num_stages"],
            hbm_bytes=e * t * (a + b) * c + tt * pa * pb * 4,
            mma_flops=2.0 * tt * ra * rb * cdiv(c, bc) * bc,
            vector_flops=2.0 * tt * ra * rb,
        )

    return [*rank(dev, space, work), TORCH]


def _gw1_run(cfg, g, mid, a, b, c):
    """sum_t G_t mid_t^T; g (t, a*c), mid (t*b, c)."""
    t = g.shape[0]
    if cfg == TORCH:
        return (g.reshape(t, a, c) @ mid.reshape(t, b, c).transpose(1, 2)).sum(0)
    pa, pb = pow2(a), pow2(b)
    n_t = triton.cdiv(t, cfg["BT"])
    part = torch.empty(n_t, pa, pb, device=g.device, dtype=torch.float32)
    _gw1_kernel[(n_t,)](
        g,
        mid,
        part,
        t,
        a,
        b,
        c,
        prec=dot_prec(g.dtype),
        block_t=cfg["BT"],
        pa=pa,
        pb=pb,
        block_c=cfg["BC"],
        **launch(cfg),
    )
    return part.sum(0)[:a, :b]


def _grad_w1(g, x, w1, w2, scale):
    """gamma * sum_t G_t mid_t^T with mid = X @ w2^T."""
    a, b = w1.shape
    c, d = w2.shape
    t = x.shape[0]
    mid = x.reshape(t * b, d) @ w2.t()
    dev = device_info(g.device)
    key = ("lokr.grad_w1", (dtype_key(g.dtype),), (token_bucket(t), a, b, c))
    cfg = select(
        dev,
        key,
        lambda: _plan_gw1(dev, t, a, b, c, g.element_size()),
        measurer(
            dev,
            t * (a + b) * c * g.element_size(),
            lambda: (copy_of(g),),
            lambda cf, s: lambda: _gw1_run(cf, s[0], mid, a, b, c),
        ),
    )
    out = guarded(
        dev, key, cfg, lambda cf: _gw1_run(cf, g, mid, a, b, c), OutOfResources
    )
    return out.mul_(scale)


def _bwd_dxn(grad, x, w1, w2, gamma):
    """gx and N from one kernel, g_w2 from N, g_w1 from mid."""
    d = w2.shape[1]
    gx, n = _dxn(grad, w1, w2, gamma, x.dtype)
    gw2 = (n.t() @ x.reshape(-1, d)).mul_(gamma)
    gw1 = _grad_w1(grad, x, w1, w2, gamma)
    return gx, gw1.to(w1.dtype), gw2.to(w2.dtype)


def _bwd_apply(grad, x, w1, w2, gamma):
    """gx through the forward kernel on (w1^T, w2^T); N and the factor
    gradients as GEMMs over tokens."""
    a, b = w1.shape
    c, d = w2.shape
    t = x.shape[0]
    gx = apply._apply(
        "lokr.bypass_dx", grad, w1.t(), w2.t(), gamma, None, x.dtype, [TORCH]
    )
    n = torch.matmul(w1.t(), grad.reshape(t, a, c)).reshape(t * b, c)
    gw2 = (n.t() @ x.reshape(t * b, d)).mul_(gamma)
    gw1 = _grad_w1(grad, x, w1, w2, gamma)
    return gx, gw1.to(w1.dtype), gw2.to(w2.dtype)


def lokr_bypass_bwd(grad, x, w1, w2, gamma: float = 1.0):
    """(gx, g_w1, g_w2) of the LoKr bypass, in the compute dtype."""
    grad, x = grad.contiguous(), x.contiguous()
    small = bypass.kron_factors_supported(*w1.shape, *w2.shape)
    dev = device_info(x.device)
    key = (
        "lokr.bypass_bwd",
        (dtype_key(x.dtype),),
        (token_bucket(x.shape[0]), *w1.shape, *w2.shape),
    )

    def run(cf, g, xx):
        if cf == ATOMIC:
            return bypass.lokr_bypass_bwd(g, xx, w1, w2, gamma)
        if cf == APPLY:
            return _bwd_apply(g, xx, w1, w2, gamma)
        return _bwd_dxn(g, xx, w1, w2, gamma)

    cfg = select(
        dev,
        key,
        lambda: [DXN, APPLY, *([ATOMIC] if small else [])],
        measurer(
            dev,
            2 * grad.numel() * grad.element_size(),
            lambda: (copy_of(grad), copy_of(x)),
            lambda cf, s: lambda: run(cf, *s),
        ),
    )
    return run(cfg, grad, x)
