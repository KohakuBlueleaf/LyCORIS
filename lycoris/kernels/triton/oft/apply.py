"""F5: Diag-OFT, Cayley transform fused into single fwd/bwd kernels.

One launch per direction, zero host tensor ops: each CTA rebuilds q = b-b^T,
inverts (I-q) by in-register Gauss-Jordan (well-conditioned since
(I-q)(I-q)^T = I + q^T q; padded rows carry identity, so elimination there is
a no-op), forms R, folds rescale rows and the identity shift, and streams its
column slice. The backward chains its PARTIAL gRf through the fixed (R, M) —
legal because gQ = (I+R)^T gR M^T is linear in gR — with atomic g_blocks /
g_rescale. Grid: 2D (k, col_groups) so small-k shapes still fill the SMs;
Cayley redundancy per col-group is ~2 s^3 FLOP. Eager contract: the einsum
contracts R's FIRST index (apply uses R^T); constraint > 0 passes one
host-computed global-norm scalar.
"""

import torch
import triton
import triton.language as tl

from ...gradbuf import GradPack
from ...plans import oft as plan
from ...plans import tune
from ...plans.device import resolve_device


@triton.jit
def _cayley(b_ptr, sbk, sbi, sbj, pid_k, cscale, S, PS: tl.constexpr):
    """Returns (R, M) fp32 (PS, PS) for block pid_k; M = (I - q)^-1."""
    ri = tl.arange(0, PS)
    rj = tl.arange(0, PS)
    mi = ri < S
    blk = tl.load(
        b_ptr + pid_k * sbk + ri[:, None] * sbi + rj[None, :] * sbj,
        mask=mi[:, None] & mi[None, :],
        other=0.0,
    ).to(tl.float32)
    # Skew part q = (b - b^T)*cscale; Cayley solves R = (I+q)(I-q)^-1.
    q = (blk - tl.trans(blk)) * cscale
    eye = (ri[:, None] == rj[None, :]).to(tl.float32)
    a = eye - q
    m = eye
    # Gauss-Jordan on [a | m]: no pivoting needed since (I-q)(I-q)^T = I+q^Tq.
    for j in tl.static_range(PS):
        is_row = ri[:, None] == j
        is_col = rj[None, :] == j
        # Broadcast pivot row j of both halves, and column j as the factors.
        pr_a = tl.sum(tl.where(is_row, a, 0.0), axis=0)
        pr_m = tl.sum(tl.where(is_row, m, 0.0), axis=0)
        pj = tl.sum(tl.where(rj == j, pr_a, 0.0))
        fac = tl.sum(tl.where(is_col, a, 0.0), axis=1)
        adj = tl.where(is_row, 0.0, fac[:, None])
        # row_i -= (a[i,j]/a[j,j]) * row_j for i != j, then normalize row j.
        a = a - adj * (pr_a / pj)[None, :]
        m = m - adj * (pr_m / pj)[None, :]
        a = tl.where(is_row, pr_a[None, :] / pj, a)
        m = tl.where(is_row, pr_m[None, :] / pj, m)
    r = tl.dot(eye + q, m, input_precision="ieee")
    return r, m


@triton.jit
def _fold_rt(
    r,
    res_ptr,
    srr,
    pid_k,
    S,
    RESCALE: tl.constexpr,
    SHIFT: tl.constexpr,
    PS: tl.constexpr,
):
    """Rf[i, j] = rescale[i] * R[j, i] - shift * delta_ij (the R^T apply)."""
    ri = tl.arange(0, PS)
    rt = tl.trans(r)
    if RESCALE:
        res = tl.load(res_ptr + (pid_k * S + ri) * srr, mask=ri < S, other=1.0)
        rt = res.to(tl.float32)[:, None] * rt
    if SHIFT:
        rt = rt - (ri[:, None] == tl.arange(0, PS)[None, :]).to(tl.float32)
    return rt


@triton.jit
def _bd_fwd_kernel(
    b_ptr,
    res_ptr,
    x_ptr,
    out_ptr,
    NB,
    L,
    K,
    S,
    COLS,
    sbk,
    sbi,
    sbj,
    srr,
    sxb,
    sxl,
    sxc,
    sob,
    sol,
    soc,
    cscale,
    CG,
    RESCALE: tl.constexpr,
    SHIFT: tl.constexpr,
    WEIGHT: tl.constexpr,
    PS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """WEIGHT: out rows = k*S+i, cols stream. else: tokens stream, channels."""
    pid_k = tl.program_id(0)
    pid_g = tl.program_id(1)
    r, _ = _cayley(b_ptr, sbk, sbi, sbj, pid_k, cscale, S, PS)
    rf = _fold_rt(r, res_ptr, srr, pid_k, S, RESCALE, SHIFT, PS)
    ri = tl.arange(0, PS)
    mi = ri < S
    ch = pid_k * S + ri
    span = tl.cdiv(COLS, CG)
    lo = pid_g * span
    hi = tl.minimum(lo + span, COLS)
    for c0 in range(lo, hi, BLOCK_N):
        rn = c0 + tl.arange(0, BLOCK_N)
        mn = rn < hi
        tb = rn // L
        tlp = rn % L
        if WEIGHT:
            x = tl.load(
                x_ptr + ch[:, None] * sxc + rn[None, :] * sxl,
                mask=mi[:, None] & mn[None, :],
                other=0.0,
            ).to(tl.float32)
            out = tl.dot(rf, x, input_precision="ieee")
            tl.store(
                out_ptr + ch[:, None] * soc + rn[None, :] * sol,
                out.to(out_ptr.dtype.element_ty),
                mask=mi[:, None] & mn[None, :],
            )
        else:
            x = tl.load(
                x_ptr + tb[:, None] * sxb + tlp[:, None] * sxl + ch[None, :] * sxc,
                mask=mn[:, None] & mi[None, :],
                other=0.0,
            ).to(tl.float32)
            out = tl.dot(x, tl.trans(rf), input_precision="ieee")
            tl.store(
                out_ptr + tb[:, None] * sob + tlp[:, None] * sol + ch[None, :] * soc,
                out.to(out_ptr.dtype.element_ty),
                mask=mn[:, None] & mi[None, :],
            )


@triton.jit
def _bd_bwd_kernel(
    b_ptr,
    res_ptr,
    x_ptr,
    g_ptr,
    gx_ptr,
    gb_ptr,
    gres_ptr,
    NB,
    L,
    K,
    S,
    COLS,
    sbk,
    sbi,
    sbj,
    srr,
    sxb,
    sxl,
    sxc,
    sgb,
    sgl,
    sgc,
    cscale,
    CG,
    RESCALE: tl.constexpr,
    SHIFT: tl.constexpr,
    WEIGHT: tl.constexpr,
    PS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """gx = Rf^T-apply; partial gRf chained to g_blocks/g_rescale in-kernel."""
    pid_k = tl.program_id(0)
    pid_g = tl.program_id(1)
    r, m = _cayley(b_ptr, sbk, sbi, sbj, pid_k, cscale, S, PS)
    rf = _fold_rt(r, res_ptr, srr, pid_k, S, RESCALE, SHIFT, PS)
    ri = tl.arange(0, PS)
    rj = tl.arange(0, PS)
    mi = ri < S
    ch = pid_k * S + ri
    grf = tl.zeros((PS, PS), tl.float32)
    span = tl.cdiv(COLS, CG)
    lo = pid_g * span
    hi = tl.minimum(lo + span, COLS)
    for c0 in range(lo, hi, BLOCK_N):
        rn = c0 + tl.arange(0, BLOCK_N)
        mn = rn < hi
        tb = rn // L
        tlp = rn % L
        if WEIGHT:
            g = tl.load(
                g_ptr + ch[:, None] * sgc + rn[None, :] * sgl,
                mask=mi[:, None] & mn[None, :],
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + ch[:, None] * sxc + rn[None, :] * sxl,
                mask=mi[:, None] & mn[None, :],
                other=0.0,
            ).to(tl.float32)
            # gx = Rf^T @ g; gRf += g @ x^T (accumulated over this CTA's span).
            gx = tl.dot(tl.trans(rf), g, input_precision="ieee")
            tl.store(
                gx_ptr + ch[:, None] * sxc + rn[None, :] * sxl,
                gx.to(gx_ptr.dtype.element_ty),
                mask=mi[:, None] & mn[None, :],
            )
            grf = tl.dot(g, tl.trans(x), grf, input_precision="ieee")
        else:
            g = tl.load(
                g_ptr + tb[:, None] * sgb + tlp[:, None] * sgl + ch[None, :] * sgc,
                mask=mn[:, None] & mi[None, :],
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + tb[:, None] * sxb + tlp[:, None] * sxl + ch[None, :] * sxc,
                mask=mn[:, None] & mi[None, :],
                other=0.0,
            ).to(tl.float32)
            # gx = g @ Rf (activation layout); gRf += g^T @ x over the span.
            gx = tl.dot(g, rf, input_precision="ieee")
            tl.store(
                gx_ptr + tb[:, None] * sxb + tlp[:, None] * sxl + ch[None, :] * sxc,
                gx.to(gx_ptr.dtype.element_ty),
                mask=mn[:, None] & mi[None, :],
            )
            grf = tl.dot(tl.trans(g), x, grf, input_precision="ieee")

    # chain: gRf -> (gR, g_rescale) -> gQ -> g_blocks; all linear in gRf.
    rt = tl.trans(r)
    if RESCALE:
        res = tl.load(res_ptr + (pid_k * S + ri) * srr, mask=mi, other=1.0).to(
            tl.float32
        )
        # g_rescale[i] = sum_j gRf[i,j] * R^T[i,j]; gR = (rescale * gRf)^T.
        gres = tl.sum(grf * rt, axis=1)
        tl.atomic_add(gres_ptr + (pid_k * S + ri), gres, mask=mi)
        gr = tl.trans(res[:, None] * grf)
    else:
        gr = tl.trans(grf)
    eye = (ri[:, None] == rj[None, :]).to(tl.float32)
    # gQ = (I+R)^T @ gR @ M^T, then gB = (gQ - gQ^T)*cscale (q is skew).
    gq = tl.dot(
        tl.dot(tl.trans(eye + r), gr, input_precision="ieee"),
        tl.trans(m),
        input_precision="ieee",
    )
    gblk = (gq - tl.trans(gq)) * cscale
    tl.atomic_add(
        gb_ptr + pid_k * sbk + ri[:, None] * sbi + rj[None, :] * sbj,
        gblk,
        mask=mi[:, None] & (rj < S)[None, :],
    )


def _ps(s: int) -> int:
    ps = max(16, 1 << (max(1, s) - 1).bit_length())
    if ps > 32:
        raise ValueError(f"in-kernel Cayley supports block size <= 32, got {s}")
    return ps


def _layouts(x: torch.Tensor, weight: bool):
    if weight:
        return 1, x.shape[1], 0, x.stride(1), x.stride(0), x.shape[1]
    if x.dim() == 2:
        return x.shape[0], 1, x.stride(0), 0, x.stride(1), x.shape[0]
    nb = x.shape[0]
    ln = 1
    for d in x.shape[2:]:
        ln *= d
    v = x.reshape(nb, x.shape[1], ln)
    return nb, ln, v.stride(0), v.stride(2), v.stride(1), nb * ln


def oft_fwd(blocks, x, rescale=None, cscale=1.0, shift=True, weight=True):
    """out = (rescale_rows * R^T - shift*I) applied to x; one launch."""
    k, s, _ = blocks.shape
    out = torch.empty_like(x)
    nb, ln, sb, sl, sc, cols = _layouts(x, weight)
    res = rescale.reshape(-1) if rescale is not None else blocks

    def launch(p, dst):
        _bd_fwd_kernel[(k, p.bm)](
            blocks,
            res,
            x,
            dst,
            nb,
            ln,
            k,
            s,
            cols,
            *blocks.stride(),
            res.stride(0) if rescale is not None else 0,
            sb,
            sl,
            sc,
            sb,
            sl,
            sc,
            cscale,
            p.bm,
            RESCALE=rescale is not None,
            SHIFT=shift,
            WEIGHT=weight,
            PS=_ps(s),
            BLOCK_N=p.bn,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    best = tune.tuned(
        "triton.oft.fwd",
        (
            k,
            s,
            tune.bucket_tokens(cols),
            weight,
            shift,
            rescale is not None,
            str(x.dtype),
        ),
        lambda: plan.topk_fused(k, s, cols, x.element_size(), resolve_device()),
        lambda p: (lambda: launch(p, out)),
    )
    launch(best, out)
    return out


def oft_bwd(blocks, x, grad, rescale=None, cscale=1.0, shift=True, weight=True):
    """(gx, g_blocks, g_rescale) in one launch."""
    k, s, _ = blocks.shape
    gx = torch.empty_like(x)
    # One fp32 allocation for both atomic targets: one zero-fill, one cast.
    pack = GradPack(x.device, (k, s, s), (k * s,))
    gb, gres = pack
    nb, ln, sb, sl, sc, cols = _layouts(x, weight)
    _, _, gsb, gsl, gsc, _ = _layouts(grad, weight)
    res = rescale.reshape(-1) if rescale is not None else blocks

    def launch(p, o_gx, o_gb, o_gres):
        _bd_bwd_kernel[(k, p.bm)](
            blocks,
            res,
            x,
            grad,
            o_gx,
            o_gb,
            o_gres,
            nb,
            ln,
            k,
            s,
            cols,
            *blocks.stride(),
            res.stride(0) if rescale is not None else 0,
            sb,
            sl,
            sc,
            gsb,
            gsl,
            gsc,
            cscale,
            p.bm,
            RESCALE=rescale is not None,
            SHIFT=shift,
            WEIGHT=weight,
            PS=_ps(s),
            BLOCK_N=p.bn,
            num_warps=p.warps,
            num_stages=p.stages,
        )

    def factory(p):
        s1 = torch.empty_like(gx)
        scratch = pack.like()
        return lambda: launch(p, s1, *scratch)

    best = tune.tuned(
        "triton.oft.bwd",
        (
            k,
            s,
            tune.bucket_tokens(cols),
            weight,
            shift,
            rescale is not None,
            str(x.dtype),
        ),
        lambda: plan.topk_fused(k, s, cols, x.element_size(), resolve_device()),
        factory,
    )
    launch(best, gx, gb, gres)
    o_gb, o_gres = pack.to(blocks.dtype)
    return gx, o_gb, o_gres if rescale is not None else None
