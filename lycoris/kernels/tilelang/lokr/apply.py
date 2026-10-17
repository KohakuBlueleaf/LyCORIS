"""LoKr bypass in TileLang on the model-ranked tuner (fp16, sm_70+):
Y_t = [B_t +] gamma * w1 @ X_t @ w2^T per token.

Both products are plain 2-D GEMMs over views. A contiguous x (t, b*d) is the
(t*b, d) matrix whose rows are (token, v), so for BT whole tokens M = X @ w2^T
is a (BT*b, d) x (d, c) GEMM; y (t, a*c) is the (t*a, c) matrix with rows
(token, u), and Y = blockdiag(w1 x BT) @ M is a (BT*a, BT*b) x (BT*b, c) GEMM
against the block-diagonal tile, built once per call in torch. BT is the
smallest token count making BT*a and BT*b multiples of 16, times a tuned
factor; w2's axes are tiled, so only w1 is bounded (BT*max(a, b) <= 128). M is
rounded to fp16 in shared memory between the two, as eager's chain rounds it.
The register-resident atomic kernel (``.bypass``, factors up to 128) and the
torch composition are timed alongside; non-fp16 operands take them. The
backward is ``.grad``. See docs/kernels/designs.md.
"""

import math

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import TORCH, device_info, select
from ...tuner.measure import copy_of, dtype_key, guarded, measurer, token_bucket
from ...tuner.model import Work, rank, shared_reads
from ..common import MIN_DIM, CompileError, compiled, fits, partitions, tl_dtype
from . import bypass

F16 = "float16"
F32 = "float32"
# Rows of either GEMM a program may hold: BT*a and BT*b.
MAX_ROWS = 128
ATOMIC = {"impl": "atomic"}


def _apply_prim(a, b, c, d, bt, bc, bd, threads, base_dt, out_dt):
    """Y = [base +] scale * blockdiag(w1) @ (X @ w2^T); grid (w2 row tiles bx, token tiles by)."""
    ra, rb = bt * a, bt * b
    plain_only = base_dt is None
    base_dt = base_dt or out_dt
    tb = T.symbolic("tb")
    ta = T.symbolic("ta")

    @T.macro
    def products(xv, w2, w1bd, bx, by, y_f):
        """y_f = blockdiag(w1) @ fp16(X_tile @ w2_tile^T) for program (bx, by)."""
        x_s = T.alloc_shared((rb, bd), F16)
        w2_s = T.alloc_shared((bc, bd), F16)
        m_f = T.alloc_fragment((rb, bc), F32)
        m_s = T.alloc_shared((rb, bc), F16)
        w1_s = T.alloc_shared((ra, rb), F16)

        # M rows (token, v) = X @ w2^T over d, then fp16 in shared memory.
        T.clear(m_f)
        for k in T.Pipelined(T.ceildiv(d, bd), num_stages=1):
            T.copy(xv[by * rb, k * bd], x_s)
            T.copy(w2[bx * bc, k * bd], w2_s)
            T.gemm(x_s, w2_s, m_f, transpose_B=True)
        T.copy(m_f, m_s)

        # Y rows (token, u) = blockdiag(w1) @ M.
        T.copy(w1bd, w1_s)
        T.clear(y_f)
        T.gemm(w1_s, m_s, y_f)

    @T.prim_func
    def with_base(
        xv: T.Tensor((tb, d), F16),
        w2: T.Tensor((c, d), F16),
        w1bd: T.Tensor((ra, rb), F16),
        basev: T.Tensor((ta, c), base_dt),
        yv: T.Tensor((ta, c), out_dt),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(c, bc), T.ceildiv(tb, rb), threads=threads) as (bx, by):
            y_f = T.alloc_fragment((ra, bc), F32)
            b_f = T.alloc_fragment((ra, bc), base_dt)
            products(xv, w2, w1bd, bx, by, y_f)
            T.copy(basev[by * ra, bx * bc], b_f)
            for p, q in T.Parallel(ra, bc):
                y_f[p, q] = b_f[p, q] + scale * y_f[p, q]
            T.copy(y_f, yv[by * ra, bx * bc])

    @T.prim_func
    def plain(
        xv: T.Tensor((tb, d), F16),
        w2: T.Tensor((c, d), F16),
        w1bd: T.Tensor((ra, rb), F16),
        yv: T.Tensor((ta, c), out_dt),
        scale: T.float32,
    ):
        with T.Kernel(T.ceildiv(c, bc), T.ceildiv(tb, rb), threads=threads) as (bx, by):
            y_f = T.alloc_fragment((ra, bc), F32)
            products(xv, w2, w1bd, bx, by, y_f)
            for p, q in T.Parallel(ra, bc):
                y_f[p, q] = scale * y_f[p, q]
            T.copy(y_f, yv[by * ra, bx * bc])

    return plain if plain_only else with_base


def _unit(a, b):
    """Fewest tokens per program making BT*a and BT*b multiples of 16."""
    ua = MIN_DIM // math.gcd(a, MIN_DIM)
    ub = MIN_DIM // math.gcd(b, MIN_DIM)
    return ua * ub // math.gcd(ua, ub)


def _smem(cfg, a, b):
    ra, rb = cfg["BT"] * a, cfg["BT"] * b
    return rb * cfg["BD"] + cfg["BC"] * cfg["BD"] + rb * cfg["BC"] + ra * rb


def _legal(dev, cfg, a, b):
    ra, rb = cfg["BT"] * a, cfg["BT"] * b
    return fits(dev, _smem(cfg, a, b)) and partitions(
        dev, cfg["threads"], (rb, cfg["BC"]), (ra, cfg["BC"])
    )


def _plan(dev, t, a, b, c, d, has_base, alternatives):
    """Legal tilings, model-ranked, then the alternatives."""
    unit = _unit(a, b)
    if unit * max(a, b) > MAX_ROWS:
        return alternatives
    space = []
    for mult in (1, 2, 4, 8, 16):
        bt = unit * mult
        if bt * max(a, b) > MAX_ROWS:
            break
        for bc in (32, 64, 128):
            for bd in (32, 64):
                for threads in (64, 128, 256):
                    cfg = {"BT": bt, "BC": bc, "BD": bd, "threads": threads}
                    if _legal(dev, cfg, a, b):
                        space.append(cfg)

    def work(cf):
        bt, bc, bd = cf["BT"], cf["BC"], cf["BD"]
        ra, rb = bt * a, bt * b
        tt, tc = math.ceil(t / bt), math.ceil(c / bc)
        # x is read once per column tile of w2; the block-diagonal w1 GEMM costs BT x the dense one
        return Work(
            grid=tt * tc,
            warps=cf["threads"] // 32,
            acc_fp32=rb * bc + ra * bc,
            operand_bytes=0,
            smem_stage=2 * _smem(cf, a, b),
            stages=1,
            hbm_bytes=2 * (t * b * d * tc + t * a * c * (2 if has_base else 1))
            + shared_reads(dev, 2 * (c * d + ra * rb), tt * tc),
            mma_flops=2.0 * tt * tc * (rb * math.ceil(d / bd) * bd * bc + ra * rb * bc),
            # each program reads its x rows, its w2 rows and the block-diagonal w1
            l2_bytes=2
            * (
                t * b * d * tc
                + tt * tc * (bc * d + ra * rb)
                + t * a * c * (2 if has_base else 1)
            ),
        )

    return [*rank(dev, space, work, mma=True), *alternatives]


def _run(cfg, flat, w1, w2, scale, base, out_dtype):
    """[base +] scale * vec(w1 @ X_t @ w2^T); flat (t, b*d), w1 (a, b), w2 (c, d), fp16."""
    a, b = w1.shape
    c, d = w2.shape
    t = flat.shape[0]
    if cfg == TORCH:
        y = torch.matmul(w1, flat.reshape(t, b, d) @ w2.t()).reshape(t, a * c) * scale
        return (y if base is None else base + y).to(out_dtype)
    if cfg == ATOMIC:
        y = bypass._bypass_delta(flat, w1, w2, scale)
        return (y if base is None else base + y).to(out_dtype)
    bt = cfg["BT"]
    w1bd = torch.block_diag(*([w1] * bt)).contiguous()
    base_dt = None if base is None else tl_dtype(base.dtype)
    kernel = compiled(
        _apply_prim,
        flat.device,
        a,
        b,
        c,
        d,
        bt,
        cfg["BC"],
        cfg["BD"],
        cfg["threads"],
        base_dt,
        tl_dtype(out_dtype),
    )
    xv = flat.contiguous().view(t * b, d)
    y = torch.empty(t * a, c, device=flat.device, dtype=out_dtype)
    if base is None:
        kernel(xv, w2.contiguous(), w1bd, y, scale)
    else:
        kernel(xv, w2.contiguous(), w1bd, base.contiguous().view(t * a, c), y, scale)
    return y.view(t, a * c)


def _apply(name, flat, w1, w2, scale, base, out_dtype, alternatives):
    dev = device_info(flat.device)
    a, b = w1.shape
    c, d = w2.shape
    t = flat.shape[0]
    key = (
        name,
        (dtype_key(out_dtype), tl_dtype(base.dtype) if base is not None else None),
        (token_bucket(t), a, b, c, d),
    )
    cfg = select(
        dev,
        key,
        lambda: _plan(dev, t, a, b, c, d, base is not None, alternatives),
        measurer(
            dev,
            t * (b * d + a * c) * 2,
            lambda: (copy_of(flat), copy_of(base)),
            lambda cf, s: lambda: _run(cf, s[0], w1, w2, scale, s[1], out_dtype),
        ),
    )
    return guarded(
        dev,
        key,
        cfg,
        lambda cf: _run(cf, flat, w1, w2, scale, base, out_dtype),
        CompileError,
    )


def _alternatives(w1, w2):
    small = bypass.kron_factors_supported(*w1.shape, *w2.shape)
    return [TORCH, ATOMIC] if small else [TORCH]


def lokr_bypass_fwd(x, w1, w2, gamma: float = 1.0, base=None):
    """[base +] gamma * x @ kron(w1, w2)^T for x (T, b*d), w1 (a, b), w2 (c, d)
    in one compute dtype; the result in base's dtype when given, else x's."""
    out_dtype = base.dtype if base is not None else x.dtype
    w1, w2 = w1.contiguous(), w2.contiguous()
    if x.dtype != torch.float16:
        return _run(_alternatives(w1, w2)[-1], x, w1, w2, gamma, base, out_dtype)
    return _apply(
        "tl.lokr.bypass_fwd", x, w1, w2, gamma, base, out_dtype, _alternatives(w1, w2)
    )
