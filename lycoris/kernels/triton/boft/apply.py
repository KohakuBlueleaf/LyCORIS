"""F6: BOFT butterfly with per-stage Cayley computed in-kernel.

Zero host tensor ops: each stage kernel rebuilds its block's rotation
(Gauss-Jordan via the shared _cayley helper), folds the multiplier
(bi*scale + (1-scale)I) and, on the last stage, the rescale rows and the
diff subtraction. The per-stage grad kernel chains gRf -> g_blocks in-kernel
(linear, as in blockdiag). One launch per stage; stage inputs in the
backward come from prefix replays (never cached — that is m*O*I).

Stage-i permutation contract (functional/boft.py): kd = 2^i * b/2;
s -> o = (s // (2*kd)) * 2*kd + (s % 2) * kd + (s % (2*kd)) // 2.
"""

import torch
import triton
import triton.language as tl

from ...gradbuf import GradPack
from ...plans import boft as plan
from ...plans import tune
from ...plans.cost import SENTINEL_CONE
from ...plans.device import resolve_device
from ..oft.apply import _cayley


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


@triton.jit
def _bf_origin(s, kd):
    """Stage permutation: s -> (s//2kd)*2kd + (s%2)*kd + (s%2kd)//2."""
    q = s // (2 * kd)
    rem = s % (2 * kd)
    ki = rem // 2
    gi = rem % 2
    return q * 2 * kd + gi * kd + ki


@triton.jit
def _bf_cone_kernel(
    b_ptr,
    x_ptr,
    org_ptr,
    res_ptr,
    out_ptr,
    NB,
    L,
    N,
    S,
    M,
    sbm,
    sbk,
    sbi,
    sbj,
    sxb,
    sxl,
    sxa,
    cscale,
    scale,
    CG,
    RESCALE: tl.constexpr,
    DIFF: tl.constexpr,
    NBLK: tl.constexpr,
    PS: tl.constexpr,
    PC: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """All M stages of one channel cone, in one CTA.

    Stage i mixes channels only within a span of 2^i * S, so a cone of
    S * 2^(M-1) consecutive channels is closed under every stage: the whole
    butterfly runs inside the CTA and the tensor is read once and written
    once, against the 2*M round trips a stage-per-launch chain pays.

    Each stage is one (PC, PC) operator applied to the cone tile. It is built
    as P^T @ blockdiag(R) @ P, where P is the stage permutation as a 0/1
    matrix: that keeps every index computation in the operator and away from
    the tile, which registers cannot gather.
    """
    pid_c = tl.program_id(0)
    pid_g = tl.program_id(1)
    rc = tl.arange(0, PC)
    mc = rc < (NBLK * S)
    base = pid_c * (NBLK * S)
    eye_c = (rc[:, None] == rc[None, :]).to(tl.float32)

    # Restricted to a cone the m stages compose to ONE PC x PC map, so the
    # Gauss-Jordan chain is paid per cone, not per column block: doing it per
    # block measured 1183 us against 142 us for the stage-per-launch chain.
    total = eye_c
    for i in range(M):
        kd = (1 << i) * (S // 2)
        group = 2 * kd
        # Block-diagonal position of each cone channel at this stage.
        g = base + rc
        u = g % group
        t = tl.where(u < kd, 2 * u, 2 * (u - kd) + 1)
        s_local = (g // group) * group + t - base
        perm = (tl.arange(0, PC)[:, None] == s_local[None, :]).to(tl.float32)
        # Cayley of the whole cone at once: Gauss-Jordan never crosses a
        # block, so one PC-wide pass gives every block's inverse.
        same = (rc[:, None] // S) == (rc[None, :] // S)
        blk_id = pid_c * NBLK + rc // S
        raw = tl.load(
            b_ptr
            + i * sbm
            + blk_id[:, None] * sbk
            + (rc % S)[:, None] * sbi
            + (rc % S)[None, :] * sbj,
            mask=same & mc[:, None] & mc[None, :],
            other=0.0,
        ).to(tl.float32)
        qf = (raw - tl.trans(raw)) * cscale
        af = eye_c - qf
        mf = eye_c
        for jj in tl.static_range(PC):
            is_row = rc[:, None] == jj
            is_col = rc[None, :] == jj
            pr_a = tl.sum(tl.where(is_row, af, 0.0), axis=0)
            pr_m = tl.sum(tl.where(is_row, mf, 0.0), axis=0)
            pj = tl.sum(tl.where(rc == jj, pr_a, 0.0))
            fac = tl.sum(tl.where(is_col, af, 0.0), axis=1)
            adj = tl.where(is_row, 0.0, fac[:, None])
            af = af - adj * (pr_a / pj)[None, :]
            mf = mf - adj * (pr_m / pj)[None, :]
            af = tl.where(is_row, pr_a[None, :] / pj, af)
            mf = tl.where(is_row, pr_m[None, :] / pj, mf)
        rfull = tl.dot(eye_c + qf, mf, input_precision="ieee")
        bdiag = rfull * scale + (1.0 - scale) * eye_c
        op = tl.dot(
            tl.trans(perm),
            tl.dot(bdiag, perm, input_precision="ieee"),
            input_precision="ieee",
        )
        total = tl.dot(op, total, input_precision="ieee")

    if RESCALE:
        res = tl.load(res_ptr + base + rc, mask=mc, other=1.0).to(tl.float32)
        total = total * res[:, None]
    if DIFF:
        total = total - eye_c

    cols = NB * L
    span = tl.cdiv(cols, CG)
    lo = pid_g * span
    hi = tl.minimum(lo + span, cols)
    for c0 in range(lo, hi, BLOCK_N):
        rn = c0 + tl.arange(0, BLOCK_N)
        tb = rn // L
        tlp = rn % L
        msk = mc[:, None] & (rn < hi)[None, :]
        tile = tl.load(
            x_ptr + tb[None, :] * sxb + tlp[None, :] * sxl + (base + rc)[:, None] * sxa,
            mask=msk,
            other=0.0,
        ).to(tl.float32)
        out = tl.dot(total, tile, input_precision="ieee")
        tl.store(
            out_ptr
            + tb[None, :] * sxb
            + tlp[None, :] * sxl
            + (base + rc)[:, None] * sxa,
            out.to(out_ptr.dtype.element_ty),
            mask=msk,
        )


@triton.jit
def _stage_r(
    b_ptr, sbk, sbi, sbj, pid_b, cscale, scale, S, TRANS: tl.constexpr, PS: tl.constexpr
):
    """Folded stage rotation: scale*R + (1-scale)*I, optionally transposed."""
    r, _ = _cayley(b_ptr, sbk, sbi, sbj, pid_b, cscale, S, PS)
    ri = tl.arange(0, PS)
    eye = (ri[:, None] == tl.arange(0, PS)[None, :]).to(tl.float32)
    rf = r * scale + (1.0 - scale) * eye
    if TRANS:
        rf = tl.trans(rf)
    return rf


@triton.jit
def _bf_stage_kernel(
    b_ptr,
    x_ptr,
    org_ptr,
    res_ptr,
    out_ptr,
    NB,
    L,
    N,
    S,
    KD,
    sbk,
    sbi,
    sbj,
    sxb,
    sxl,
    sxa,
    sob,
    sol,
    soa,
    cscale,
    scale,
    CG,
    TRANS_R: tl.constexpr,
    LAST: tl.constexpr,
    RESCALE: tl.constexpr,
    DIFF: tl.constexpr,
    PS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    rf = _stage_r(b_ptr, sbk, sbi, sbj, pid_b, cscale, scale, S, TRANS_R, PS)
    ri = tl.arange(0, PS)
    mi = ri < S
    o = _bf_origin(pid_b * S + ri, KD)
    cols = NB * L
    span = tl.cdiv(cols, CG)
    lo = pid_g * span
    hi = tl.minimum(lo + span, cols)
    for c0 in range(lo, hi, BLOCK_N):
        rn = c0 + tl.arange(0, BLOCK_N)
        tb = rn // L
        tlp = rn % L
        mn = rn < hi
        x = tl.load(
            x_ptr + tb[None, :] * sxb + tlp[None, :] * sxl + o[:, None] * sxa,
            mask=mi[:, None] & mn[None, :],
            other=0.0,
        ).to(tl.float32)
        # y = Rf @ x for this block's permuted channel rows.
        y = tl.dot(rf, x, input_precision="ieee")
        if LAST:
            if RESCALE:
                res = tl.load(res_ptr + o, mask=mi, other=1.0).to(tl.float32)
                y = y * res[:, None]
            if DIFF:
                org = tl.load(
                    org_ptr + tb[None, :] * sxb + tlp[None, :] * sxl + o[:, None] * sxa,
                    mask=mi[:, None] & mn[None, :],
                    other=0.0,
                ).to(tl.float32)
                y = y - org
        tl.store(
            out_ptr + tb[None, :] * sob + tlp[None, :] * sol + o[:, None] * soa,
            y.to(out_ptr.dtype.element_ty),
            mask=mi[:, None] & mn[None, :],
        )


@triton.jit
def _bf_grad_kernel(
    b_ptr,
    g_ptr,
    x_ptr,
    gn_ptr,
    gb_ptr,
    NB,
    L,
    N,
    S,
    KD,
    sbk,
    sbi,
    sbj,
    sgb,
    sgl,
    sga,
    sxb,
    sxl,
    sxa,
    snb,
    snl,
    sna,
    cscale,
    scale,
    CG,
    PS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One stage of the backward, whole: g_next = Rf^T g stored over this CTA's
    span, and the span's partial gRf chained to g_blocks in-kernel.

    The gRf -> g_blocks chain is linear, so each CTA chains its partial sum and
    atomically adds into the pre-zeroed fp32 slice; grid (nblk, CG) keeps the
    waves full where nblk alone (SDXL: 20) would idle most of the card, and
    fusing the g_next store here reads g once instead of twice per stage.
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    r, m = _cayley(b_ptr, sbk, sbi, sbj, pid_b, cscale, S, PS)
    ri = tl.arange(0, PS)
    rj = tl.arange(0, PS)
    mi = ri < S
    eye = (ri[:, None] == rj[None, :]).to(tl.float32)
    rf = r * scale + (1.0 - scale) * eye
    o = _bf_origin(pid_b * S + ri, KD)
    grf = tl.zeros((PS, PS), tl.float32)
    cols = NB * L
    span = tl.cdiv(cols, CG)
    lo = pid_g * span
    hi = tl.minimum(lo + span, cols)
    for c0 in range(lo, hi, BLOCK_N):
        rn = c0 + tl.arange(0, BLOCK_N)
        tb = rn // L
        tlp = rn % L
        mn = rn < hi
        msk = mi[:, None] & mn[None, :]
        g = tl.load(
            g_ptr + tb[None, :] * sgb + tlp[None, :] * sgl + o[:, None] * sga,
            mask=msk,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + tb[None, :] * sxb + tlp[None, :] * sxl + o[:, None] * sxa,
            mask=msk,
            other=0.0,
        ).to(tl.float32)
        # gRf += g @ x^T (span partial); g_next = Rf^T @ g for the next stage.
        grf = tl.dot(g, tl.trans(x), grf, input_precision="ieee")
        gn = tl.dot(tl.trans(rf), g, input_precision="ieee")
        tl.store(
            gn_ptr + tb[None, :] * snb + tlp[None, :] * snl + o[:, None] * sna,
            gn.to(gn_ptr.dtype.element_ty),
            mask=msk,
        )
    # gR = gRf*scale; gQ = (I+R)^T gR M^T; gB = (gQ - gQ^T)*cscale.
    gr = grf * scale
    gq = tl.dot(
        tl.dot(tl.trans(eye + r), gr, input_precision="ieee"),
        tl.trans(m),
        input_precision="ieee",
    )
    gblk = (gq - tl.trans(gq)) * cscale
    tl.atomic_add(
        gb_ptr + pid_b * sbk + ri[:, None] * sbi + rj[None, :] * sbj,
        gblk,
        mask=mi[:, None] & (rj < S)[None, :],
    )


def _layout(x: torch.Tensor, axis: int):
    if axis in (-1, x.dim() - 1):
        flat = x.reshape(-1, x.shape[-1])
        return flat.shape[0], 1, flat.stride(0), 0, flat.stride(1), flat.shape[1]
    if axis == 0:
        flat = x.reshape(x.shape[0], -1)
        return 1, flat.shape[1], 0, flat.stride(1), flat.stride(0), flat.shape[0]
    nb = x.shape[0]
    ln = 1
    for d in x.shape[2:]:
        ln *= d
    v = x.reshape(nb, x.shape[1], ln)
    return nb, ln, v.stride(0), v.stride(2), v.stride(1), x.shape[1]


def _pick(blocks, x, axis, name, cone=None):
    m, nblk, s, _ = blocks.shape
    nb, ln = _layout(x, axis)[:2]

    def factory(p):
        scratch = torch.empty_like(x)
        if p.limiter == "cone":
            return lambda: _run_cone(blocks, x, axis, 1.0, 1.0, None, False, p, cone)
        return lambda: _run_stage(
            blocks, 0, x, scratch, axis, False, 0, 1.0, 1.0, None, None, False, p
        )

    def shortlist():
        cands = plan.topk_fused(nblk, s, nb * ln, m, x.element_size(), resolve_device())
        return [*cands, SENTINEL_CONE] if cone is not None else cands

    return tune.tuned(
        name,
        (nblk, s, tune.bucket_tokens(nb * ln), m, axis, str(x.dtype)),
        shortlist,
        factory,
    )


def _run_stage(
    blocks,
    i,
    src,
    dst,
    axis,
    trans,
    offset,
    cscale,
    scale,
    rescale,
    org,
    diff,
    p,
    last=False,
):
    nblk, s = blocks.shape[1], blocks.shape[2]
    nb, ln, sb, sl, sa, n = _layout(src, axis)
    _, _, ob, ol, oa, _ = _layout(dst, axis)
    kd = (2 ** (i + offset)) * (s // 2)
    ps = max(16, _next_pow2(s))
    res = rescale if rescale is not None else blocks
    org_ = org if org is not None else src
    _bf_stage_kernel[(nblk, p.bm)](
        blocks[i],
        src,
        org_,
        res,
        dst,
        nb,
        ln,
        n,
        s,
        kd,
        blocks.stride(1),
        blocks.stride(2),
        blocks.stride(3),
        sb,
        sl,
        sa,
        ob,
        ol,
        oa,
        cscale,
        scale,
        p.bm,
        TRANS_R=trans,
        LAST=last,
        RESCALE=last and rescale is not None,
        DIFF=last and diff,
        PS=ps,
        BLOCK_N=p.bn,
        num_warps=p.warps,
        num_stages=p.stages,
    )
    return dst


# PC bounds registers and compile time alike (the Gauss-Jordan is PC unrolled
# steps over PC x PC tiles). Whether the cone beats the stage chain at a legal
# size is a shape question the tuner answers, via SENTINEL_CONE.
CONE_MAX = 32


def _cone_fit(blocks, trans, reverse, offset):
    """(n_cones, blocks_per_cone, PC) when the whole butterfly fits a CTA."""
    m, nblk, s, _ = blocks.shape
    if trans or reverse or offset or m < 2:
        return None
    cone = s * (1 << (m - 1))
    if cone > CONE_MAX or cone > nblk * s or (nblk * s) % cone:
        return None
    return (nblk * s) // cone, cone // s, _next_pow2(cone)


def _run_cone(blocks, x, axis, cscale, scale, rescale, diff, p, cone):
    n_cones, nblk_cone, pc = cone
    m, _, s, _ = blocks.shape
    nb, ln, sb, sl, sa, n = _layout(x, axis)
    out = torch.empty_like(x)
    res = rescale if rescale is not None else blocks
    org = x
    _bf_cone_kernel[(n_cones, p.bm)](
        blocks,
        x,
        org,
        res,
        out,
        nb,
        ln,
        n,
        s,
        m,
        blocks.stride(0),
        blocks.stride(1),
        blocks.stride(2),
        blocks.stride(3),
        sb,
        sl,
        sa,
        cscale,
        scale,
        p.bm,
        RESCALE=rescale is not None,
        DIFF=diff,
        NBLK=nblk_cone,
        PS=max(16, _next_pow2(s)),
        PC=pc,
        BLOCK_N=p.bn,
        num_warps=p.warps,
        num_stages=p.stages,
    )
    return out


def boft_fwd(
    blocks,
    x,
    axis=0,
    cscale=1.0,
    scale=1.0,
    rescale=None,
    diff=False,
    trans=False,
    reverse=False,
    offset=0,
):
    """All m stages, Cayley in-kernel; last stage folds rescale and diff.

    One launch when the channel cone fits a CTA (see _bf_cone_kernel), else a
    launch per stage.
    """
    blocks = blocks.contiguous()
    m = blocks.shape[0]
    cone = _cone_fit(blocks, trans, reverse, offset)
    p = _pick(blocks, x, axis, "triton.boft.fwd", cone)
    if cone is not None and p.limiter == "cone":
        return _run_cone(blocks, x, axis, cscale, scale, rescale, diff, p, cone)
    out = torch.empty_like(x)
    tmp = torch.empty_like(x) if m > 1 else out
    order = list(range(m - 1, -1, -1) if reverse else range(m))
    src = x
    for step, i in enumerate(order):
        into_out = (m - 1 - step) % 2 == 0
        dst = out if into_out else tmp
        last = step == m - 1 and not trans
        _run_stage(
            blocks,
            i,
            src,
            dst,
            axis,
            trans,
            offset,
            cscale,
            scale,
            rescale,
            x,
            diff,
            p,
            last=last,
        )
        src = dst
    return out


def boft_bwd(blocks, x, grad, axis=0, cscale=1.0, scale=1.0):
    """(gx, g_blocks): per-stage grad kernels chain gRf in-kernel; stage
    inputs come from prefix replays (nothing cached)."""
    blocks = blocks.contiguous()
    m, nblk, s, _ = blocks.shape
    ps = max(16, _next_pow2(s))
    # One fp32 buffer for every stage's atomic target: one fill, one cast.
    pack = GradPack(blocks.device, blocks.shape)
    (gb,) = pack
    g = grad
    bufs = [torch.empty_like(grad)]
    if m > 1:
        bufs.append(torch.empty_like(grad))
    p = _pick(blocks, x, axis, "triton.boft.fwd_bwd")
    for step, i in enumerate(range(m - 1, -1, -1)):
        gnext = bufs[step % len(bufs)]
        stage_in = (
            boft_fwd(blocks[:i], x, axis, cscale, scale, trans=False) if i > 0 else x
        )
        nb, ln, sb, sl, sa, n = _layout(stage_in, axis)
        _, _, gsb, gsl, gsa, _ = _layout(g, axis)
        _, _, nnb, nnl, nna, _ = _layout(gnext, axis)
        kd = (2**i) * (s // 2)
        _bf_grad_kernel[(nblk, p.bm)](
            blocks[i],
            g,
            stage_in,
            gnext,
            gb[i],
            nb,
            ln,
            n,
            s,
            kd,
            blocks.stride(1),
            blocks.stride(2),
            blocks.stride(3),
            gsb,
            gsl,
            gsa,
            sb,
            sl,
            sa,
            nnb,
            nnl,
            nna,
            cscale,
            scale,
            p.bm,
            PS=ps,
            BLOCK_N=p.bn,
            num_warps=p.warps,
            num_stages=p.stages,
        )
        g = gnext
    return g, pack.to(blocks.dtype)[0]
