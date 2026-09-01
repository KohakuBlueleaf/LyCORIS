"""F3: Kronecker rebuild (LoKr).

DeltaW[u*C+c, v*D+d] = gamma * w1[u, v] * w2[c, d] in one gather-elementwise
pass; the factors are KB-scale and L1-resident, so the output write is the
whole cost. w2 generation (low-rank / tucker) stays in torch — it is tiny.
Backward is two block-reduction kernels.
"""

import torch
import triton
import triton.language as tl

from ...gradbuf import GradPack
from ...plans import lokr as plan
from ...plans import tune
from ...plans.device import resolve_device


@triton.jit
def _lokr_full_merge_fwd_kernel(
    w1a_ptr,
    w1b_ptr,
    w2a_ptr,
    w2b_ptr,
    base_ptr,
    out_ptr,
    O,
    I,
    C: tl.constexpr,
    D: tl.constexpr,
    R1,
    R2,
    s1aa,
    s1ar,
    s1br,
    s1bb,
    s2ac,
    s2ar,
    s2br,
    s2bd,
    sbo,
    sbi,
    soo,
    soi,
    gamma,
    ADD_BASE: tl.constexpr,
    GEN1: tl.constexpr,
    GEN2: tl.constexpr,
    PREC: tl.constexpr,
    PR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """DeltaW[u*C+c, v*D+d] = gamma * w1[u,v] * w2[c,d], tiled on the OUTPUT.

    Tiling the output is what coalesces the write, which is the whole cost
    here (the factors are KB and L1-resident). A factorized side does not
    change the grid: only the pointer arithmetic is scattered, so
    w1[u,v] = sum_r w1a[u,r] w1b[r,v] is still a plain dot over gathered
    operands. One body therefore serves both-full, A-factorized and
    B-factorized — and only one side is ever factorized.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ro = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ri = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mo = ro < O
    mi = ri < I
    omask = mo[:, None] & mi[None, :]
    # Row/col split into the two factor index pairs: (u, c) and (v, d).
    u = ro // C
    c = ro % C
    v = ri // D
    d = ri % D
    rr = tl.arange(0, PR)

    if GEN1:
        a1 = tl.load(
            w1a_ptr + u[:, None] * s1aa + rr[None, :] * s1ar,
            mask=mo[:, None] & (rr < R1)[None, :],
            other=0.0,
        )
        b1 = tl.load(
            w1b_ptr + rr[:, None] * s1br + v[None, :] * s1bb,
            mask=(rr < R1)[:, None] & mi[None, :],
            other=0.0,
        )
        w1 = tl.dot(a1, b1, input_precision=PREC)
    else:
        w1 = tl.load(
            w1a_ptr + u[:, None] * s1aa + v[None, :] * s1ar, mask=omask, other=0.0
        ).to(tl.float32)
    if GEN2:
        a2 = tl.load(
            w2a_ptr + c[:, None] * s2ac + rr[None, :] * s2ar,
            mask=mo[:, None] & (rr < R2)[None, :],
            other=0.0,
        )
        b2 = tl.load(
            w2b_ptr + rr[:, None] * s2br + d[None, :] * s2bd,
            mask=(rr < R2)[:, None] & mi[None, :],
            other=0.0,
        )
        w2 = tl.dot(a2, b2, input_precision=PREC)
    else:
        w2 = tl.load(
            w2a_ptr + c[:, None] * s2ac + d[None, :] * s2ar, mask=omask, other=0.0
        ).to(tl.float32)

    out = w1 * w2 * gamma
    if ADD_BASE:
        base = tl.load(
            base_ptr + ro[:, None] * sbo + ri[None, :] * sbi, mask=omask, other=0.0
        )
        out += base.to(tl.float32)
    tl.store(
        out_ptr + ro[:, None] * soo + ri[None, :] * soi,
        out.to(out_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _lokr_merge_fwd_kernel(
    w1a_ptr,
    w1b_ptr,
    w2a_ptr,
    w2b_ptr,
    base_ptr,
    out_ptr,
    A,
    B,
    C,
    D,
    R1,
    R2,
    s1aa,
    s1ar,
    s1br,
    s1bb,
    s2ac,
    s2ar,
    s2br,
    s2bd,
    sbo,
    sbi,
    soo,
    soi,
    gamma,
    NC,
    ND,
    ADD_BASE: tl.constexpr,
    GEN1: tl.constexpr,
    GEN2: tl.constexpr,
    PREC: tl.constexpr,
    PR: tl.constexpr,
    UV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Kron rebuild with a factorized side generated in-kernel.

    A CTA owns one (C, D) sub-tile and a CHUNK of outer (u, v) pairs, so the
    factorized side's tile is generated ONCE and then reused across every pair
    in the chunk: generation cost falls from a*b tiles to
    ceil(C/BM)*ceil(D/BN)*ceil(A*B/UV) of them. w1[u,v] stays a scalar (a
    rank-length dot when that side is the factorized one), so no gather exists
    on either side. Grid = ceil(C/BM)*ceil(D/BN)*ceil(A*B/UV).
    """
    pid = tl.program_id(0)
    pid_uv = tl.program_id(1)
    rc = (pid // ND) * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = (pid % ND) * BLOCK_N + tl.arange(0, BLOCK_N)
    mc = rc < C
    md = rd < D
    rr = tl.arange(0, PR)
    mr1 = rr < R1
    mr2 = rr < R2

    # w2 tile: loaded, or generated as w2a[c-tile, :] @ w2b[:, d-tile]. Hoisted
    # out of the (u, v) loop — it does not depend on the outer pair.
    if GEN2:
        a2 = tl.load(
            w2a_ptr + rc[:, None] * s2ac + rr[None, :] * s2ar,
            mask=mc[:, None] & mr2[None, :],
            other=0.0,
        )
        b2 = tl.load(
            w2b_ptr + rr[:, None] * s2br + rd[None, :] * s2bd,
            mask=mr2[:, None] & md[None, :],
            other=0.0,
        )
        w2t = tl.dot(a2, b2, input_precision=PREC)
    else:
        w2t = tl.load(
            w2a_ptr + rc[:, None] * s2ac + rd[None, :] * s2ar,
            mask=mc[:, None] & md[None, :],
            other=0.0,
        ).to(tl.float32)

    omask = mc[:, None] & md[None, :]
    lo = pid_uv * UV
    hi = tl.minimum(lo + UV, A * B)
    for uv in range(lo, hi):
        u = uv // B
        v = uv % B
        # w1[u, v]: one scalar, loaded or reduced from its factor pair.
        if GEN1:
            a1 = tl.load(w1a_ptr + u * s1aa + rr * s1ar, mask=mr1, other=0.0).to(
                tl.float32
            )
            b1 = tl.load(w1b_ptr + rr * s1br + v * s1bb, mask=mr1, other=0.0).to(
                tl.float32
            )
            w1v = tl.sum(a1 * b1)
        else:
            w1v = tl.load(w1a_ptr + u * s1aa + v * s1ar).to(tl.float32)
        # DeltaW[u*C+c, v*D+d] = gamma * w1[u,v] * w2[c,d] (+ base).
        ro = u * C + rc
        ri = v * D + rd
        out = w1v * w2t * gamma
        if ADD_BASE:
            base = tl.load(
                base_ptr + ro[:, None] * sbo + ri[None, :] * sbi, mask=omask, other=0.0
            )
            out += base.to(tl.float32)
        tl.store(
            out_ptr + ro[:, None] * soo + ri[None, :] * soi,
            out.to(out_ptr.dtype.element_ty),
            mask=omask,
        )


@triton.jit
def _lokr_full_merge_bwd_kernel(
    g_ptr,
    w1_ptr,
    w2_ptr,
    gw1_ptr,
    gw2_ptr,
    A,
    B,
    C,
    D,
    sgo,
    sgi,
    s1u,
    s1v,
    s2c,
    s2d,
    gamma,
    NC,
    ND,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Both factor grads, one role-split launch: pids [0, A*B) each reduce one
    g_w1[u,v]; the rest tile (C, D) x u and reduce g_w2 over v, atomically
    across the u split — a serial (u, v) reduction is a dependent chain of A*B
    scalar loads.
    """
    pid = tl.program_id(0)
    ga = A * B
    if pid < ga:
        # g_w1[u, v] = gamma * sum_{c,d} G[u*C+c, v*D+d] * w2[c, d].
        u = pid // B
        v = pid % B
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for c0 in range(0, C, BLOCK_M):
            rc = c0 + tl.arange(0, BLOCK_M)
            mc = rc < C
            for d0 in range(0, D, BLOCK_N):
                rd = d0 + tl.arange(0, BLOCK_N)
                md = rd < D
                m = mc[:, None] & md[None, :]
                g = tl.load(
                    g_ptr + (u * C + rc)[:, None] * sgo + (v * D + rd)[None, :] * sgi,
                    mask=m,
                    other=0.0,
                ).to(tl.float32)
                w2 = tl.load(
                    w2_ptr + rc[:, None] * s2c + rd[None, :] * s2d, mask=m, other=0.0
                ).to(tl.float32)
                acc += g * w2
        tl.store(gw1_ptr + u * B + v, tl.sum(acc) * gamma)
    else:
        # g_w2[c, d] = gamma * sum_{u,v} w1[u, v] * G[u*C+c, v*D+d].
        pid2 = pid - ga
        tile = pid2 % (NC * ND)
        u = pid2 // (NC * ND)
        rc = (tile // ND) * BLOCK_M + tl.arange(0, BLOCK_M)
        rd = (tile % ND) * BLOCK_N + tl.arange(0, BLOCK_N)
        mc = rc < C
        md = rd < D
        m = mc[:, None] & md[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for v in range(B):
            w1 = tl.load(w1_ptr + u * s1u + v * s1v).to(tl.float32)
            g = tl.load(
                g_ptr + (u * C + rc)[:, None] * sgo + (v * D + rd)[None, :] * sgi,
                mask=m,
                other=0.0,
            ).to(tl.float32)
            acc += w1 * g
        tl.atomic_add(gw2_ptr + rc[:, None] * D + rd[None, :], acc * gamma, mask=m)


def lokr_full_merge_fwd(
    w1: torch.Tensor,
    w2: torch.Tensor,
    base: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """DeltaW = gamma * kron(w1, w2) [+ base]; w2 may be a flattened conv factor.

    Output-tiled, not flat: the flat form recovers (u, c, v, d) with four
    integer div/mods PER ELEMENT, where the tile recovers them once per row
    and once per column vector. Measured 10.0 us flat against 7.6 us tiled at
    1280 square.
    """
    a, b = w1.shape
    c, d = w2.shape
    return _launch_merge_fwd(w1, None, w2, None, (a, b, c, d), base, gamma)


def _launch_merge_fwd(w1a, w1b, w2a, w2b, shape, base, gamma):
    """The one output-tiled body, with either side optionally factorized."""
    a, b, c, d = shape
    r1 = w1b.shape[0] if w1b is not None else 0
    r2 = w2b.shape[0] if w2b is not None else 0
    pr = max(16, 1 << (max(1, r1, r2) - 1).bit_length())
    out_o, out_i = a * c, b * d
    out = torch.empty(out_o, out_i, device=w2a.device, dtype=w2a.dtype)
    base_ = base if base is not None else w1a
    w1b_ = w1b if w1b is not None else w1a
    w2b_ = w2b if w2b is not None else w2a

    def launch(p, dst):
        _lokr_full_merge_fwd_kernel[
            (triton.cdiv(out_o, p.bm), triton.cdiv(out_i, p.bn))
        ](
            w1a,
            w1b_,
            w2a,
            w2b_,
            base_,
            dst,
            out_o,
            out_i,
            c,
            d,
            r1,
            r2,
            *w1a.stride(),
            *w1b_.stride(),
            *w2a.stride(),
            *w2b_.stride(),
            *(base_.stride() if base is not None else (0, 0)),
            *dst.stride(),
            gamma,
            ADD_BASE=base is not None,
            GEN1=w1b is not None,
            GEN2=w2b is not None,
            PREC="ieee" if w2a.dtype == torch.float32 else "tf32",
            PR=pr,
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=1,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    shortlist = lambda: plan.topk_rebuild(
        out_o, out_i, base is not None, w2a.element_size(), resolve_device()
    )
    best = tune.tuned(
        "triton.lokr.merge_fwd",
        (out_o, out_i, c, d, r1, r2, base is not None, str(w2a.dtype)),
        shortlist,
        lambda p: (lambda: launch(p, out)),
    )
    launch(best, out)
    return out


@triton.jit
def _lokr_merge_bwd_kernel(
    g_ptr,
    w1a_ptr,
    w1b_ptr,
    w2a_ptr,
    w2b_ptr,
    g1a_ptr,
    g1b_ptr,
    g2a_ptr,
    g2b_ptr,
    A,
    B,
    C,
    D,
    R1,
    R2,
    sgo,
    sgi,
    s1aa,
    s1ar,
    s1br,
    s1bb,
    s2ac,
    s2ar,
    s2br,
    s2bd,
    gamma,
    NC,
    ND,
    GA,
    GEN1: tl.constexpr,
    GEN2: tl.constexpr,
    PREC: tl.constexpr,
    PR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Both LoKr factor-pair grads in ONE launch, sub-factor chain included.

    Role A (pid < GA), one CTA per (u, v):
      gw1 = gamma * sum_{c,d} G[u*C+c, v*D+d] * w2[c,d], then
      g_w1a[u,:] += gw1*w1b[:,v] and g_w1b[:,v] += gw1*w1a[u,:].
    Role B, one CTA per (C, D) tile:
      gw2 = gamma * sum_{u,v} w1[u,v] * G[u*C+c, v*D+d], then
      g_w2a += gw2@w2b^T and g_w2b += w2a^T@gw2.
    Every w1/w2 value is generated from its halves, so no materialized
    factor — and no host mm — exists on either path.
    """
    pid = tl.program_id(0)
    rr = tl.arange(0, PR)
    mr1 = rr < R1
    mr2 = rr < R2
    if pid < GA:
        u = pid // B
        v = pid % B
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for c0 in range(0, C, BLOCK_M):
            rc = c0 + tl.arange(0, BLOCK_M)
            mc = rc < C
            for d0 in range(0, D, BLOCK_N):
                rd = d0 + tl.arange(0, BLOCK_N)
                md = rd < D
                m = mc[:, None] & md[None, :]
                g = tl.load(
                    g_ptr + (u * C + rc)[:, None] * sgo + (v * D + rd)[None, :] * sgi,
                    mask=m,
                    other=0.0,
                ).to(tl.float32)
                # w2 chunk, generated (or loaded) exactly as in the forward.
                if GEN2:
                    a2 = tl.load(
                        w2a_ptr + rc[:, None] * s2ac + rr[None, :] * s2ar,
                        mask=mc[:, None] & mr2[None, :],
                        other=0.0,
                    )
                    b2 = tl.load(
                        w2b_ptr + rr[:, None] * s2br + rd[None, :] * s2bd,
                        mask=mr2[:, None] & md[None, :],
                        other=0.0,
                    )
                    w2t = tl.dot(a2, b2, input_precision=PREC)
                else:
                    w2t = tl.load(
                        w2a_ptr + rc[:, None] * s2ac + rd[None, :] * s2ar,
                        mask=m,
                        other=0.0,
                    ).to(tl.float32)
                acc += g * w2t
        gw1 = tl.sum(acc) * gamma
        if GEN1:
            # Chain the scalar through w1 = w1a@w1b, in-kernel.
            a1 = tl.load(w1a_ptr + u * s1aa + rr * s1ar, mask=mr1, other=0.0).to(
                tl.float32
            )
            b1 = tl.load(w1b_ptr + rr * s1br + v * s1bb, mask=mr1, other=0.0).to(
                tl.float32
            )
            tl.atomic_add(g1a_ptr + u * R1 + rr, gw1 * b1, mask=mr1)
            tl.atomic_add(g1b_ptr + rr * B + v, gw1 * a1, mask=mr1)
        else:
            tl.store(g1a_ptr + u * B + v, gw1)
    else:
        # Split over u as well as the (C, D) tile: a serial (u, v) reduction is
        # a dependent chain of A*B scalar loads (6656 at llm_mlp shapes).
        pid2 = pid - GA
        tile = pid2 % (NC * ND)
        u = pid2 // (NC * ND)
        rc = (tile // ND) * BLOCK_M + tl.arange(0, BLOCK_M)
        rd = (tile % ND) * BLOCK_N + tl.arange(0, BLOCK_N)
        mc = rc < C
        md = rd < D
        m = mc[:, None] & md[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for v in range(B):
            # w1[u,v] scalar, generated (or loaded) as in the forward.
            if GEN1:
                a1 = tl.load(w1a_ptr + u * s1aa + rr * s1ar, mask=mr1, other=0.0).to(
                    tl.float32
                )
                b1 = tl.load(w1b_ptr + rr * s1br + v * s1bb, mask=mr1, other=0.0).to(
                    tl.float32
                )
                w1v = tl.sum(a1 * b1)
            else:
                w1v = tl.load(w1a_ptr + u * s1aa + v * s1ar).to(tl.float32)
            g = tl.load(
                g_ptr + (u * C + rc)[:, None] * sgo + (v * D + rd)[None, :] * sgi,
                mask=m,
                other=0.0,
            ).to(tl.float32)
            acc += w1v * g
        gw2 = acc * gamma
        if GEN2:
            # Chain the tile through w2 = w2a@w2b, in-kernel.
            a2 = tl.load(
                w2a_ptr + rc[:, None] * s2ac + rr[None, :] * s2ar,
                mask=mc[:, None] & mr2[None, :],
                other=0.0,
            )
            b2 = tl.load(
                w2b_ptr + rr[:, None] * s2br + rd[None, :] * s2bd,
                mask=mr2[:, None] & md[None, :],
                other=0.0,
            )
            tl.atomic_add(
                g2a_ptr + rc[:, None] * R2 + rr[None, :],
                tl.dot(gw2.to(b2.dtype), tl.trans(b2), input_precision=PREC),
                mask=mc[:, None] & mr2[None, :],
            )
            tl.atomic_add(
                g2b_ptr + rr[:, None] * D + rd[None, :],
                tl.dot(tl.trans(a2), gw2.to(a2.dtype), input_precision=PREC),
                mask=mr2[:, None] & md[None, :],
            )
        else:
            # The u axis is split across CTAs, so this tile has several writers.
            tl.atomic_add(g2a_ptr + rc[:, None] * D + rd[None, :], gw2, mask=m)


def lokr_merge_fwd(
    w1a: torch.Tensor,
    w1b: torch.Tensor | None,
    w2a: torch.Tensor,
    w2b: torch.Tensor | None,
    shape: tuple[int, int, int, int],
    base: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """DeltaW = gamma * kron(w1a@w1b, w2a@w2b) [+ base], factors generated
    in-kernel; a None b-half means that side arrives already whole."""
    return _launch_merge_fwd(w1a, w1b, w2a, w2b, shape, base, gamma)


def lokr_merge_bwd(
    grad: torch.Tensor,
    w1a: torch.Tensor,
    w1b: torch.Tensor | None,
    w2a: torch.Tensor,
    w2b: torch.Tensor | None,
    shape: tuple[int, int, int, int],
    gamma: float = 1.0,
):
    """(g_w1a, g_w1b, g_w2a, g_w2b) in one launch; a None b-half means that
    side is whole, and its 'a' grad is the full factor grad."""
    a, b, c, d = shape
    r1 = w1b.shape[0] if w1b is not None else 0
    r2 = w2b.shape[0] if w2b is not None else 0
    pr = max(16, 1 << (max(1, r1, r2) - 1).bit_length())
    # One fp32 allocation for every grad this launch writes: one fill, one cast.
    pack = GradPack(
        grad.device,
        (a, r1 if w1b is not None else b),
        (max(r1, 1), b),
        (c, r2 if w2b is not None else d),
        (max(r2, 1), d),
    )
    g1a, g1b, g2a, g2b = pack
    w1b_ = w1b if w1b is not None else w1a
    w2b_ = w2b if w2b is not None else w2a

    def launch(p, o1, o2, o3, o4):
        nc, nd = triton.cdiv(c, p.bm), triton.cdiv(d, p.bn)
        ga = a * b
        _lokr_merge_bwd_kernel[(ga + nc * nd * a,)](
            grad,
            w1a,
            w1b_,
            w2a,
            w2b_,
            o1,
            o2,
            o3,
            o4,
            a,
            b,
            c,
            d,
            r1,
            r2,
            *grad.stride(),
            *w1a.stride(),
            *w1b_.stride(),
            *w2a.stride(),
            *w2b_.stride(),
            gamma,
            nc,
            nd,
            ga,
            GEN1=w1b is not None,
            GEN2=w2b is not None,
            PREC="ieee" if grad.dtype == torch.float32 else "tf32",
            PR=pr,
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        scratch = pack.like()
        return lambda: launch(p, *scratch)

    shortlist = lambda: plan.topk_rebuild_bwd(
        a, b, c, d, grad.element_size(), resolve_device()
    )
    best = tune.tuned(
        "triton.lokr.merge_bwd",
        (a, b, c, d, r1, r2, str(grad.dtype)),
        shortlist,
        factory,
    )
    launch(best, g1a, g1b, g2a, g2b)
    o1a, o1b, o2a, o2b = pack.to(w1a.dtype)
    return (
        o1a,
        o1b if w1b is not None else None,
        o2a,
        o2b if w2b is not None else None,
    )


def lokr_full_merge_bwd(
    grad: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    a, b = w1.shape
    c, d = w2.shape
    # gw2 is accumulated across the u split, so it is a zeroed atomic target.
    pack = GradPack(grad.device, (a, b), (c, d))
    gw1, gw2 = pack

    def launch(p, o1, o2):
        nc, nd = triton.cdiv(c, p.bm), triton.cdiv(d, p.bn)
        grid = a * b + nc * nd * a
        _lokr_full_merge_bwd_kernel[(grid,)](
            grad,
            w1,
            w2,
            o1,
            o2,
            a,
            b,
            c,
            d,
            *grad.stride(),
            *w1.stride(),
            *w2.stride(),
            gamma,
            nc,
            nd,
            BLOCK_M=p.bm,
            BLOCK_N=p.bn,
            BLOCK_K=1,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        scratch = pack.like()
        return lambda: launch(p, *scratch)

    shortlist = lambda: plan.topk_rebuild_bwd(
        a, b, c, d, grad.element_size(), resolve_device()
    )
    best = tune.tuned(
        "triton.lokr.full_merge_bwd",
        (a, b, c, d, str(grad.dtype)),
        shortlist,
        factory,
    )
    launch(best, gw1, gw2)
    return pack.to(w1.dtype)
