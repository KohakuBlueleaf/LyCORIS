"""F1: low-rank rebuild kernels (LoCon/LoHa/DyLoRA/GLoRA/LoKr-w2).

Forward materializes DeltaW in one pass with no intermediate products; the
hadamard backward regenerates product tiles in registers instead of caching
or re-materializing them (design/families/lowrank.md). Plain (non-hadamard)
backward stays on cuBLAS in the autograd layer: its factor grads never need
an O x I intermediate, so a kernel buys nothing there.
"""

import torch
import triton
import triton.language as tl

from ...plans import lora as plan
from ...plans import tune
from ...plans.cost import SENTINEL_EAGER
from ...plans.device import resolve_device
from ..common import rank_block


@triton.jit
def _rebuild_kernel(
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    base_ptr,
    out_ptr,
    O,
    I,
    R,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    sbo,
    sbi,
    soo,
    soi,
    gamma,
    MODE: tl.constexpr,
    ADD_BASE: tl.constexpr,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """MODE: 0 = a1@b1, 1 = (a1@b1)*(a2@b2), 2 = a1@b1 + a2@b2."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = rm < O
    mn = rn < I
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for rk0 in range(0, R, BR):
        rk = rk0 + tl.arange(0, BR)
        mk = rk < R
        a1 = tl.load(
            a1_ptr + rm[:, None] * sa1o + rk[None, :] * sa1r,
            mask=mm[:, None] & mk[None, :],
            other=0.0,
        )
        b1 = tl.load(
            b1_ptr + rk[:, None] * sb1r + rn[None, :] * sb1i,
            mask=mk[:, None] & mn[None, :],
            other=0.0,
        )
        acc1 = tl.dot(a1, b1, acc1, input_precision=PREC)
        if MODE > 0:
            a2 = tl.load(
                a2_ptr + rm[:, None] * sa2o + rk[None, :] * sa2r,
                mask=mm[:, None] & mk[None, :],
                other=0.0,
            )
            b2 = tl.load(
                b2_ptr + rk[:, None] * sb2r + rn[None, :] * sb2i,
                mask=mk[:, None] & mn[None, :],
                other=0.0,
            )
            acc2 = tl.dot(a2, b2, acc2, input_precision=PREC)
    if MODE == 1:
        out = acc1 * acc2 * gamma
    elif MODE == 2:
        out = (acc1 + acc2) * gamma
    else:
        out = acc1 * gamma
    omask = mm[:, None] & mn[None, :]
    if ADD_BASE:
        base = tl.load(
            base_ptr + rm[:, None] * sbo + rn[None, :] * sbi, mask=omask, other=0.0
        )
        out += base.to(tl.float32)
    tl.store(
        out_ptr + rm[:, None] * soo + rn[None, :] * soi,
        out.to(out_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _rebuild_tucker_kernel(
    a1_ptr,
    b1_ptr,
    t1_ptr,
    a2_ptr,
    b2_ptr,
    t2_ptr,
    out_ptr,
    O,
    I,
    K,
    R,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    st1p,
    st1q,
    st1k,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    st2p,
    st2q,
    st2k,
    soo,
    soi,
    sok,
    gamma,
    HADA: tl.constexpr,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rr = tl.arange(0, BR)
    mm = rm < O
    mn = rn < I
    mr = rr < R

    a1 = tl.load(
        a1_ptr + rm[:, None] * sa1o + rr[None, :] * sa1r,
        mask=mm[:, None] & mr[None, :],
        other=0.0,
    )
    t1 = tl.load(
        t1_ptr + rr[:, None] * st1p + rr[None, :] * st1q + pid_k * st1k,
        mask=mr[:, None] & mr[None, :],
        other=0.0,
    )
    b1 = tl.load(
        b1_ptr + rr[:, None] * sb1r + rn[None, :] * sb1i,
        mask=mr[:, None] & mn[None, :],
        other=0.0,
    )
    at1 = tl.dot(a1, t1, input_precision=PREC)
    acc = tl.dot(at1.to(b1.dtype), b1, input_precision=PREC)
    if HADA:
        a2 = tl.load(
            a2_ptr + rm[:, None] * sa2o + rr[None, :] * sa2r,
            mask=mm[:, None] & mr[None, :],
            other=0.0,
        )
        t2 = tl.load(
            t2_ptr + rr[:, None] * st2p + rr[None, :] * st2q + pid_k * st2k,
            mask=mr[:, None] & mr[None, :],
            other=0.0,
        )
        b2 = tl.load(
            b2_ptr + rr[:, None] * sb2r + rn[None, :] * sb2i,
            mask=mr[:, None] & mn[None, :],
            other=0.0,
        )
        at2 = tl.dot(a2, t2, input_precision=PREC)
        acc = acc * tl.dot(at2.to(b2.dtype), b2, input_precision=PREC)
    out = acc * gamma
    omask = mm[:, None] & mn[None, :]
    tl.store(
        out_ptr + rm[:, None] * soo + rn[None, :] * soi + pid_k * sok,
        out.to(out_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _loha_merge_bwd_kernel(
    g_ptr,
    a1_ptr,
    b1_ptr,
    a2_ptr,
    b2_ptr,
    ga1_ptr,
    gb1_ptr,
    ga2_ptr,
    gb2_ptr,
    O,
    I,
    R,
    sgo,
    sgi,
    sa1o,
    sa1r,
    sb1r,
    sb1i,
    sa2o,
    sa2r,
    sb2r,
    sb2i,
    gamma,
    GA,
    PREC: tl.constexpr,
    BR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """All four hadamard factor grads, one role-split 1D launch, no atomics.

    pids [0, GA) own an O tile and reduce over I -> ga1, ga2; the rest own an
    I tile and reduce over O -> gb1, gb2. Every CTA holds its whole reduction
    in registers, so the grads are plain dtype stores: no fp32 scratch, no
    zero-fill, no cast launch, and the result is deterministic. G is read
    twice, which is L2-covered at these sizes. Waves are O/BM + I/BN.
    """
    pid = tl.program_id(0)
    rr = tl.arange(0, BR)
    mr = rr < R
    if pid < GA:
        rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mm = rm < O
        amask = mm[:, None] & mr[None, :]
        a1 = tl.load(
            a1_ptr + rm[:, None] * sa1o + rr[None, :] * sa1r, mask=amask, other=0.0
        )
        a2 = tl.load(
            a2_ptr + rm[:, None] * sa2o + rr[None, :] * sa2r, mask=amask, other=0.0
        )
        ga1 = tl.zeros((BLOCK_M, BR), tl.float32)
        ga2 = tl.zeros((BLOCK_M, BR), tl.float32)
        for i0 in range(0, I, BLOCK_N):
            rn = i0 + tl.arange(0, BLOCK_N)
            mn = rn < I
            bmask = mr[:, None] & mn[None, :]
            b1 = tl.load(
                b1_ptr + rr[:, None] * sb1r + rn[None, :] * sb1i, mask=bmask, other=0.0
            )
            b2 = tl.load(
                b2_ptr + rr[:, None] * sb2r + rn[None, :] * sb2i, mask=bmask, other=0.0
            )
            g = (
                tl.load(
                    g_ptr + rm[:, None] * sgo + rn[None, :] * sgi,
                    mask=mm[:, None] & mn[None, :],
                    other=0.0,
                ).to(tl.float32)
                * gamma
            )
            # e1 = gamma*G*(a2@b2), e2 = gamma*G*(a1@b1); ga_k += e_k @ b_k^T.
            e1 = (g * tl.dot(a2, b2, input_precision=PREC)).to(b1.dtype)
            e2 = (g * tl.dot(a1, b1, input_precision=PREC)).to(b2.dtype)
            ga1 = tl.dot(e1, tl.trans(b1), ga1, input_precision=PREC)
            ga2 = tl.dot(e2, tl.trans(b2), ga2, input_precision=PREC)
        tl.store(
            ga1_ptr + rm[:, None] * R + rr[None, :],
            ga1.to(ga1_ptr.dtype.element_ty),
            mask=amask,
        )
        tl.store(
            ga2_ptr + rm[:, None] * R + rr[None, :],
            ga2.to(ga2_ptr.dtype.element_ty),
            mask=amask,
        )
    else:
        rn = (pid - GA) * BLOCK_N + tl.arange(0, BLOCK_N)
        mn = rn < I
        bmask = mr[:, None] & mn[None, :]
        b1 = tl.load(
            b1_ptr + rr[:, None] * sb1r + rn[None, :] * sb1i, mask=bmask, other=0.0
        )
        b2 = tl.load(
            b2_ptr + rr[:, None] * sb2r + rn[None, :] * sb2i, mask=bmask, other=0.0
        )
        gb1 = tl.zeros((BR, BLOCK_N), tl.float32)
        gb2 = tl.zeros((BR, BLOCK_N), tl.float32)
        for o0 in range(0, O, BLOCK_M):
            rm = o0 + tl.arange(0, BLOCK_M)
            mm = rm < O
            amask = mm[:, None] & mr[None, :]
            a1 = tl.load(
                a1_ptr + rm[:, None] * sa1o + rr[None, :] * sa1r, mask=amask, other=0.0
            )
            a2 = tl.load(
                a2_ptr + rm[:, None] * sa2o + rr[None, :] * sa2r, mask=amask, other=0.0
            )
            g = (
                tl.load(
                    g_ptr + rm[:, None] * sgo + rn[None, :] * sgi,
                    mask=mm[:, None] & mn[None, :],
                    other=0.0,
                ).to(tl.float32)
                * gamma
            )
            # Same e_k, contracted the other way: gb_k += a_k^T @ e_k.
            e1 = (g * tl.dot(a2, b2, input_precision=PREC)).to(b1.dtype)
            e2 = (g * tl.dot(a1, b1, input_precision=PREC)).to(b2.dtype)
            gb1 = tl.dot(tl.trans(a1), e1, gb1, input_precision=PREC)
            gb2 = tl.dot(tl.trans(a2), e2, gb2, input_precision=PREC)
        tl.store(
            gb1_ptr + rr[:, None] * I + rn[None, :],
            gb1.to(gb1_ptr.dtype.element_ty),
            mask=bmask,
        )
        tl.store(
            gb2_ptr + rr[:, None] * I + rn[None, :],
            gb2.to(gb2_ptr.dtype.element_ty),
            mask=bmask,
        )


def _ieee(*tensors: torch.Tensor) -> bool:
    return any(t is not None and t.dtype == torch.float32 for t in tensors)


def lora_merge_fwd(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    base: torch.Tensor | None = None,
    gamma: float = 1.0,
    mode: str = "plain",
) -> torch.Tensor:
    """DeltaW = gamma * (a1@b1) [* or + (a2@b2)] [+ base], one pass.

    mode: "plain" (a2/b2 ignored), "hada" (product), "sum" (GLoRA-style).
    """
    out_o, r = a1.shape
    out_i = b1.shape[1]
    mode_id = {"plain": 0, "hada": 1, "sum": 2}[mode]
    out = torch.empty(out_o, out_i, device=a1.device, dtype=a1.dtype)
    z = a1
    a2_, b2_ = (a2, b2) if mode_id > 0 else (z, z)
    base_ = base if base is not None else z
    eb = a1.element_size()

    def launch(p, dst):
        _rebuild_kernel[(triton.cdiv(out_o, p.bm), triton.cdiv(out_i, p.bn))](
            a1,
            b1,
            a2_,
            b2_,
            base_,
            dst,
            out_o,
            out_i,
            r,
            *a1.stride(),
            *b1.stride(),
            *a2_.stride(),
            *b2_.stride(),
            *(base_.stride() if base is not None else (0, 0)),
            *dst.stride(),
            gamma,
            MODE=mode_id,
            ADD_BASE=base is not None,
            PREC="ieee" if _ieee(a1) else "tf32",
            BR=min(rank_block(r), 32),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=1,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def eager_run():
        acc = a1 @ b1
        if mode_id == 1:
            acc = acc * (a2_ @ b2_)
        elif mode_id == 2:
            acc = acc + a2_ @ b2_
        acc = acc * gamma
        return acc + base if base is not None else acc

    shortlist = lambda: [
        *plan.topk_rebuild(out_o, out_i, r, mode_id == 1, eb, resolve_device()),
        SENTINEL_EAGER,
    ]

    def factory(p):
        if p.limiter == "eager":
            return eager_run
        return lambda: launch(p, out)

    best = tune.tuned(
        "triton.lora.merge_fwd",
        (out_o, out_i, r, mode_id, base is not None, str(a1.dtype)),
        shortlist,
        factory,
    )
    if best.limiter == "eager":
        return eager_run()
    launch(best, out)
    return out


def lora_tucker_fwd(
    a1: torch.Tensor,
    t1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor | None = None,
    t2: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """DeltaW[:, :, k] = gamma * (a1 @ t1[..k] @ b1) [* (a2 @ t2[..k] @ b2)].

    a is (O, R) oriented, b is (R, I) oriented (callers pass strides via
    transposed views), t is (R, R, K) with K the flattened spatial size.
    """
    out_o, r = a1.shape
    out_i = b1.shape[1]
    k = t1.shape[2]
    if r > 64:
        raise ValueError("tucker rebuild kernel supports rank <= 64")
    hada = a2 is not None
    out = torch.empty(out_o, out_i, k, device=a1.device, dtype=a1.dtype)
    z = a1
    a2_, b2_, t2_ = (a2, b2, t2) if hada else (z, z, t1)
    eb = a1.element_size()

    def launch(p, dst):
        _rebuild_tucker_kernel[(triton.cdiv(out_o, p.bm), triton.cdiv(out_i, p.bn), k)](
            a1,
            b1,
            t1,
            a2_,
            b2_,
            t2_,
            dst,
            out_o,
            out_i,
            k,
            r,
            *a1.stride(),
            *b1.stride(),
            *t1.stride(),
            *a2_.stride(),
            *b2_.stride(),
            *t2_.stride(),
            *dst.stride(),
            gamma,
            HADA=hada,
            PREC="ieee" if _ieee(a1) else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=1,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    shortlist = lambda: plan.topk_rebuild(out_o, out_i, r, hada, eb, resolve_device())
    best = tune.tuned(
        "triton.lora.tucker_fwd",
        (out_o, out_i, k, r, hada, str(a1.dtype)),
        shortlist,
        lambda p: (lambda: launch(p, out)),
    )
    launch(best, out)
    return out


def loha_merge_bwd(
    grad: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grads of DeltaW = gamma*(a1@b1)*(a2@b2) with tiles regenerated in-kernel.

    Role-split, so each CTA owns its whole reduction and writes the parameter
    dtype directly — no fp32 scratch, no zero-fill, no cast launch.
    """
    out_o, r = a1.shape
    out_i = b1.shape[1]
    if r > 128:
        raise ValueError("hadamard backward kernel supports rank <= 128")
    ga1 = torch.empty(out_o, r, device=a1.device, dtype=a1.dtype)
    ga2 = torch.empty(out_o, r, device=a1.device, dtype=a2.dtype)
    gb1 = torch.empty(r, out_i, device=a1.device, dtype=b1.dtype)
    gb2 = torch.empty(r, out_i, device=a1.device, dtype=b2.dtype)
    eb = a1.element_size()

    def launch(p, o1, o2, o3, o4):
        ga = triton.cdiv(out_o, p.bm)
        _loha_merge_bwd_kernel[(ga + triton.cdiv(out_i, p.bn),)](
            grad,
            a1,
            b1,
            a2,
            b2,
            o1,
            o2,
            o3,
            o4,
            out_o,
            out_i,
            r,
            *grad.stride(),
            *a1.stride(),
            *b1.stride(),
            *a2.stride(),
            *b2.stride(),
            gamma,
            ga,
            PREC="ieee" if _ieee(a1, grad) else "tf32",
            BR=rank_block(r),
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=1,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        s1, s2 = torch.empty_like(ga1), torch.empty_like(gb1)
        s3, s4 = torch.empty_like(ga2), torch.empty_like(gb2)
        return lambda: launch(p, s1, s2, s3, s4)

    shortlist = lambda: plan.topk_hada_bwd(out_o, out_i, r, eb, resolve_device())
    best = tune.tuned(
        "triton.loha.merge_bwd",
        (out_o, out_i, r, str(a1.dtype)),
        shortlist,
        factory,
    )
    launch(best, ga1, gb1, ga2, gb2)
    return ga1, gb1, ga2, gb2
