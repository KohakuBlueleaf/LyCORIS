"""LoKr bypass backward in TileLang on the model-ranked tuner (fp16, sm_70+).

gx_t = gamma * w1^T @ G_t @ w2, contracting w1 first (eager's order):
N = blockdiag(w1^T x BT) @ G is a (BT*b, BT*a) x (BT*a, c) GEMM per token tile
and column tile of G, rounded to fp16 and stored once, then gx = N @ w2 over
c. N is also the operand of g_w2 = gamma * N^T X, one GEMM over (token, b)
rows; g_w1 = gamma * sum_t G_t mid_t^T with mid = X @ w2^T (a cuBLAS GEMM)
reduced per token tile as the token-diagonal blocks of one (BT*a, BT*b) tile
product, partials summed in torch. The forward-kernel
formulation (gx through ``.apply`` on (w1^T, w2^T)), the register-resident
atomic kernel (``.bypass``) and the torch composition are timed alongside.
See docs/kernels/designs.md.
"""

import math

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer, token_bucket
from ...tuner.model import Work, rank, shared_reads
from ..common import CompileError, compiled, fits, partitions, tl_dtype
from . import apply, bypass

F16 = "float16"
F32 = "float32"
DXN = {"impl": "dxn"}
APPLY = {"impl": "apply"}
ATOMIC = apply.ATOMIC


def _dxn_prim(a, b, c, d, bt, bc, bd, threads, out_dt):
    """gx = scale * (blockdiag(w1^T) @ G) @ w2 and N = blockdiag(w1^T) @ G;
    grid (gx column tiles bx, token tiles by), N stored by the bx == 0 programs."""
    ra, rb = bt * a, bt * b
    ta = T.symbolic("ta")
    tb = T.symbolic("tb")

    @T.prim_func
    def main(
        gv: T.Tensor((ta, c), F16),
        w1bdt: T.Tensor((rb, ra), F16),
        w2: T.Tensor((c, d), F16),
        gxv: T.Tensor((tb, d), out_dt),
        nv: T.Tensor((tb, c), F16),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(d, bd), T.ceildiv(tb, rb), threads=threads) as (bx, by):
            w1_s = T.alloc_shared((rb, ra), F16)
            g_s = T.alloc_shared((ra, bc), F16)
            w2_s = T.alloc_shared((bc, bd), F16)
            n_f = T.alloc_fragment((rb, bc), F32)
            n_s = T.alloc_shared((rb, bc), F16)
            acc = T.alloc_fragment((rb, bd), F32)
            T.copy(w1bdt, w1_s)
            T.clear(acc)

            # Each column tile of G: N tile = blockdiag(w1^T) @ G tile, then acc += N @ w2.
            for k in T.serial(T.ceildiv(c, bc)):
                T.copy(gv[by * ra, k * bc], g_s)
                T.copy(w2[k * bc, bx * bd], w2_s)
                T.clear(n_f)
                T.gemm(w1_s, g_s, n_f)
                T.copy(n_f, n_s)
                if bx == 0:
                    T.copy(n_s, nv[by * rb, k * bc])
                T.gemm(n_s, w2_s, acc)
            for p, q in T.Parallel(rb, bd):
                acc[p, q] = scale * acc[p, q]
            T.copy(acc, gxv[by * rb, bx * bd])

    return main


def _smem(cfg, a, b):
    ra, rb = cfg["BT"] * a, cfg["BT"] * b
    return rb * ra + ra * cfg["BC"] + cfg["BC"] * cfg["BD"] + rb * cfg["BC"]


def _plan(dev, t, a, b, c, d):
    """Legal tilings of the gx + N kernel, model-ranked, then torch."""
    unit = apply._unit(a, b)
    if unit * max(a, b) > apply.MAX_ROWS:
        return [TORCH]
    space = []
    for mult in (1, 2, 4, 8, 16):
        bt = unit * mult
        if bt * max(a, b) > apply.MAX_ROWS:
            break
        for bc in (32, 64, 128):
            for bd in (32, 64, 128):
                for threads in (64, 128, 256):
                    cfg = {"BT": bt, "BC": bc, "BD": bd, "threads": threads}
                    rb = bt * b
                    if fits(dev, _smem(cfg, a, b)) and partitions(
                        dev, threads, (rb, bc), (rb, bd)
                    ):
                        space.append(cfg)

    def work(cf):
        bt, bc, bd = cf["BT"], cf["BC"], cf["BD"]
        ra, rb = bt * a, bt * b
        tt, td = math.ceil(t / bt), math.ceil(d / bd)
        # G is read once per gx column tile; N is rebuilt per column tile
        return Work(
            grid=tt * td,
            warps=cf["threads"] // 32,
            acc_fp32=rb * bc + rb * bd,
            operand_bytes=0,
            smem_stage=2 * _smem(cf, a, b),
            stages=1,
            hbm_bytes=2 * (t * a * c * td + t * b * d + t * b * c)
            + shared_reads(dev, 2 * (c * d + ra * rb), tt * td),
            mma_flops=2.0 * tt * td * (rb * ra * c + rb * c * bd),
            l2_bytes=2 * (t * a * c * td + tt * td * (c * bd + ra * rb) + t * b * d),
        )

    return [*rank(dev, space, work, mma=True), TORCH]


def _dxn_run(cfg, g, w1, w2, scale, out_dtype):
    """(gx (t, b*d) in out_dtype, N (t*b, c) fp16) for grad g (t, a*c)."""
    a, b = w1.shape
    c, d = w2.shape
    t = g.shape[0]
    if cfg == TORCH:
        n = torch.matmul(w1.t(), g.reshape(t, a, c)).reshape(t * b, c)
        gx = torch.addmm(n.new_empty(t * b, d), n, w2, beta=0, alpha=scale)
        return gx.reshape(t, b * d).to(out_dtype), n
    bt = cfg["BT"]
    w1bdt = torch.block_diag(*([w1.t()] * bt)).contiguous()
    kernel = compiled(
        _dxn_prim,
        g.device,
        a,
        b,
        c,
        d,
        bt,
        cfg["BC"],
        cfg["BD"],
        cfg["threads"],
        tl_dtype(out_dtype),
    )
    gx = torch.empty(t * b, d, device=g.device, dtype=out_dtype)
    n = torch.empty(t * b, c, device=g.device, dtype=torch.float16)
    kernel(g.contiguous().view(t * a, c), w1bdt, w2.contiguous(), gx, n, scale)
    return gx.view(t, b * d), n


def _dxn(g, w1, w2, scale, out_dtype):
    dev = device_info(g.device)
    a, b = w1.shape
    c, d = w2.shape
    t = g.shape[0]
    key = (
        "tl.lokr.bypass_dxn",
        (dtype_key(out_dtype),),
        (token_bucket(t), a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, t, a, b, c, d),
        measurer(
            dev,
            t * (a * c + 2 * b * d) * 2,
            lambda: (copy_of(g),),
            lambda cf, s: lambda: _dxn_run(cf, s[0], w1, w2, scale, out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _dxn_run(cf, g, w1, w2, scale, out_dtype),
        CompileError,
    )


def _gw1_prim(a, b, c, bt, bc, threads):
    """Per token tile, the sum over its tokens of G_t mid_t^T (fp32): the
    diagonal (a, b) blocks of the (BT*a, BT*b) tile product over c."""
    ra, rb = bt * a, bt * b
    ta = T.symbolic("ta")
    tb = T.symbolic("tb")
    n_t = T.symbolic("n_t")

    @T.prim_func
    def main(
        gv: T.Tensor((ta, c), F16),
        mv: T.Tensor((tb, c), F16),
        part: T.Tensor((n_t, a, b), F32),
    ):
        with T.Kernel(T.ceildiv(tb, rb), threads=threads) as by:
            g_s = T.alloc_shared((ra, bc), F16)
            m_s = T.alloc_shared((rb, bc), F16)
            z_f = T.alloc_fragment((ra, rb), F32)
            z_s = T.alloc_shared((ra, rb), F32)
            p_f = T.alloc_fragment((a, b), F32)
            T.clear(z_f)
            for k in T.serial(T.ceildiv(c, bc)):
                # rows past the last token load as zeros
                for p, q in T.Parallel(ra, bc):
                    g_s[p, q] = T.if_then_else(
                        (by * ra + p < ta) and (k * bc + q < c),
                        gv[T.min(by * ra + p, ta - 1), T.min(k * bc + q, c - 1)],
                        T.cast(0, F16),
                    )
                for p, q in T.Parallel(rb, bc):
                    m_s[p, q] = T.if_then_else(
                        (by * rb + p < tb) and (k * bc + q < c),
                        mv[T.min(by * rb + p, tb - 1), T.min(k * bc + q, c - 1)],
                        T.cast(0, F16),
                    )
                T.gemm(g_s, m_s, z_f, transpose_B=True)
            T.copy(z_f, z_s)

            # The token-diagonal blocks of z, summed over the tile's tokens.
            T.clear(p_f)
            for j in T.serial(bt):
                for u, v in T.Parallel(a, b):
                    p_f[u, v] += z_s[j * a + u, j * b + v]
            T.copy(p_f, part[by, 0, 0])

    return main


def _plan_gw1(dev, t, a, b, c):
    """Legal tilings of the g_w1 reduction, model-ranked, then torch."""
    unit = apply._unit(a, b)
    if unit * max(a, b) > apply.MAX_ROWS:
        return [TORCH]
    space = []
    for mult in (1, 2, 4, 8, 16):
        bt = unit * mult
        if bt * max(a, b) > apply.MAX_ROWS:
            break
        for bc in (32, 64, 128):
            for threads in (64, 128, 256):
                ra, rb = bt * a, bt * b
                if fits(dev, (ra + rb) * bc + 2 * ra * rb) and partitions(
                    dev, threads, (ra, rb)
                ):
                    space.append({"BT": bt, "BC": bc, "threads": threads})

    def work(cf):
        bt, bc = cf["BT"], cf["BC"]
        ra, rb = bt * a, bt * b
        tt = math.ceil(t / bt)
        # G and mid read once; the cross-token blocks of each tile product are waste
        return Work(
            grid=tt,
            warps=cf["threads"] // 32,
            acc_fp32=ra * rb,
            operand_bytes=0,
            smem_stage=2 * (ra + rb) * bc + 4 * ra * rb,
            stages=1,
            hbm_bytes=2 * t * (a + b) * c + tt * a * b * 4,
            mma_flops=2.0 * tt * ra * rb * c,
            vector_flops=2.0 * (ra + rb) * c * tt,
        )

    return [*rank(dev, space, work, mma=True), TORCH]


def _gw1_run(cfg, g, mid, a, b, c):
    """sum_t G_t mid_t^T; g (t, a*c), mid (t*b, c)."""
    t = g.shape[0]
    if cfg == TORCH:
        return (g.reshape(t, a, c) @ mid.reshape(t, b, c).transpose(1, 2)).sum(0)
    bt = cfg["BT"]
    n_t = -(-t // bt)
    part = torch.empty(n_t, a, b, device=g.device, dtype=torch.float32)
    kernel = compiled(_gw1_prim, g.device, a, b, c, bt, cfg["BC"], cfg["threads"])
    kernel(g.contiguous().view(t * a, c), mid, part)
    return part.sum(0)


def _grad_w1(g, x, w1, w2, scale):
    """gamma * sum_t G_t mid_t^T with mid = X @ w2^T."""
    a, b = w1.shape
    c, d = w2.shape
    t = x.shape[0]
    mid = x.reshape(t * b, d) @ w2.t()
    if g.dtype != torch.float16:
        return _gw1_run(TORCH, g, mid, a, b, c).mul_(scale)
    dev = device_info(g.device)
    key = ("tl.lokr.grad_w1", (dtype_key(g.dtype),), (token_bucket(t), a, b, c))
    cfg = select(
        dev,
        key,
        lambda: _plan_gw1(dev, t, a, b, c),
        measurer(
            dev,
            t * (a + b) * c * 2,
            lambda: (copy_of(g),),
            lambda cf, s: lambda: _gw1_run(cf, s[0], mid, a, b, c),
        ),
    )
    out = guarded(dev, key, cfg, lambda cf: _gw1_run(cf, g, mid, a, b, c), CompileError)
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
    w1t, w2t = w1.t().contiguous(), w2.t().contiguous()
    if x.dtype == torch.float16:
        gx = apply._apply(
            "tl.lokr.bypass_dx", grad, w1t, w2t, gamma, None, x.dtype, [TORCH]
        )
    else:
        gx = apply._run(TORCH, grad, w1t, w2t, gamma, None, x.dtype)
    n = torch.matmul(w1.t(), grad.reshape(t, a, c)).reshape(t * b, c)
    gw2 = (n.t() @ x.reshape(t * b, d)).mul_(gamma)
    gw1 = _grad_w1(grad, x, w1, w2, gamma)
    return gx, gw1.to(w1.dtype), gw2.to(w2.dtype)


def lokr_bypass_bwd(grad, x, w1, w2, gamma: float = 1.0):
    """(gx, g_w1, g_w2) of the LoKr bypass, in the compute dtype."""
    grad, x = grad.contiguous(), x.contiguous()
    w1, w2 = w1.contiguous(), w2.contiguous()
    small = bypass.kron_factors_supported(*w1.shape, *w2.shape)
    if x.dtype != torch.float16:
        if small:
            return bypass.lokr_bypass_bwd(grad, x, w1, w2, gamma)
        return _bwd_apply(grad, x, w1, w2, gamma)
    dev = device_info(x.device)
    key = (
        "tl.lokr.bypass_bwd",
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
