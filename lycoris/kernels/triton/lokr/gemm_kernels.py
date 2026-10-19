"""LoKr bypass as token-major GEMM tiles: x (T, b*d) is the matrix X2 (T*b, d) and
y (T, a*c) is Y2 (T*a, c), so Y2 = (I_T (x) w1) . X2 . w2^T is one GEMM against w2
plus a mix of each token's b rows by w1, done once per tile on the GEMM's result.

MIX picks the mix: "batch" a 3-D dot per token (pow2(a), pow2(b) >= 16), "diag"
one 2-D dot against the token-block-diagonal w1 tile. Every kernel takes G
adapters with stacked factors (G, a, b) / (G, c, d) that share x. See
docs/kernels/designs.md (F4b).
"""

import triton
import triton.language as tl


@triton.jit
def _mix_rows(
    m,
    w1_ptr,
    s1a,
    s1b,
    n_a,
    n_b,
    ct: tl.constexpr,
    prec: tl.constexpr,
    transpose: tl.constexpr,
    MIX: tl.constexpr,
    BT: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
):
    """Rows (token, v) of m (BT*PB, N) mixed to rows (token, u) by w1 (a, b), or rows
    (token, u) of m (BT*PA, N) to rows (token, v) by w1^T when transpose."""
    rows_in: tl.constexpr = PA if transpose else PB
    rows_out: tl.constexpr = PB if transpose else PA
    cols: tl.constexpr = m.shape[1]
    if MIX == "batch":
        ro = tl.arange(0, rows_out)
        ri = tl.arange(0, rows_in)
        if transpose:
            w = tl.load(
                w1_ptr + ri[None, :] * s1a + ro[:, None] * s1b,
                mask=(ri < n_a)[None, :] & (ro < n_b)[:, None],
                other=0.0,
            )
        else:
            w = tl.load(
                w1_ptr + ro[:, None] * s1a + ri[None, :] * s1b,
                mask=(ro < n_a)[:, None] & (ri < n_b)[None, :],
                other=0.0,
            )
        m3 = tl.reshape(m, (BT, rows_in, cols))
        wb = tl.broadcast_to(w.to(ct)[None, :, :], (BT, rows_out, rows_in))
        out = tl.reshape(tl.dot(wb, m3, input_precision=prec), (BT * rows_out, cols))
    else:
        ro = tl.arange(0, BT * rows_out)
        ri = tl.arange(0, BT * rows_in)
        to = ro // rows_out
        ti = ri // rows_in
        if transpose:
            w = tl.load(
                w1_ptr + (ri % rows_in)[None, :] * s1a + (ro % rows_out)[:, None] * s1b,
                mask=(to[:, None] == ti[None, :])
                & ((ri % rows_in) < n_a)[None, :]
                & ((ro % rows_out) < n_b)[:, None],
                other=0.0,
            )
        else:
            w = tl.load(
                w1_ptr + (ro % rows_out)[:, None] * s1a + (ri % rows_in)[None, :] * s1b,
                mask=(to[:, None] == ti[None, :])
                & ((ro % rows_out) < n_a)[:, None]
                & ((ri % rows_in) < n_b)[None, :],
                other=0.0,
            )
        out = tl.dot(w.to(ct), m, input_precision=prec)
    return out


@triton.jit
def _gemm_fwd_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    b_ptr,
    y_ptr,
    sg_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    n_ct,
    sxt,
    sxi,
    s1g,
    s1a,
    s1b,
    s2g,
    s2c,
    s2d,
    sbg,
    sbt,
    sbo,
    syg,
    syt,
    syo,
    scale,
    has_base: tl.constexpr,
    group_scale: tl.constexpr,
    prec: tl.constexpr,
    MIX: tl.constexpr,
    BT: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Y[g, t, u*c + n] = [base +] s_g sum_v w1[g, u, v] sum_k X[t, v*d + k] w2[g, n, k].

    Grid: (G * column tiles, token tiles); a token tile's programs run adjacent, so
    its x rows are read from DRAM once for every group and column tile.
    """
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    grp = pid_n // n_ct
    pid_c = pid_n % n_ct
    ct = x_ptr.dtype.element_ty

    rb = tl.arange(0, BT * PB)
    tok_b = pid_t * BT + rb // PB
    mrow_b = (tok_b < n_tok) & ((rb % PB) < n_b)
    rn = pid_c * BN + tl.arange(0, BN)
    mn = rn < n_c
    w2g = w2_ptr + grp * s2g

    # M rows (token, v): X2 @ w2^T over k = d.
    acc = tl.zeros((BT * PB, BN), tl.float32)
    for k0 in range(0, n_d, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < n_d
        xv = tl.load(
            x_ptr
            + tok_b[:, None] * sxt
            + ((rb % PB)[:, None] * n_d + rk[None, :]) * sxi,
            mask=mrow_b[:, None] & mk[None, :],
            other=0.0,
        )
        wv = tl.load(
            w2g + rn[None, :] * s2c + rk[:, None] * s2d,
            mask=mk[:, None] & mn[None, :],
            other=0.0,
        )
        acc = tl.dot(xv, wv, acc, input_precision=prec)

    # M in x's dtype, as eager's chain rounds it; then the w1 mix.
    y = _mix_rows(
        acc.to(ct),
        w1_ptr + grp * s1g,
        s1a,
        s1b,
        n_a,
        n_b,
        ct,
        prec,
        False,
        MIX,
        BT,
        PA,
        PB,
    )
    if group_scale:
        y = y * tl.load(sg_ptr + grp)
    else:
        y = y * scale

    ra = tl.arange(0, BT * PA)
    tok_a = pid_t * BT + ra // PA
    ua = ra % PA
    omask = ((tok_a < n_tok) & (ua < n_a))[:, None] & mn[None, :]
    col = ua[:, None] * n_c + rn[None, :]
    if has_base:
        base = tl.load(
            b_ptr + grp * sbg + tok_a[:, None] * sbt + col * sbo, mask=omask, other=0.0
        )
        y += base.to(tl.float32)
    tl.store(
        y_ptr + grp * syg + tok_a[:, None] * syt + col * syo,
        y.to(y_ptr.dtype.element_ty),
        mask=omask,
    )


@triton.jit
def _gemm_dx_kernel(
    g_ptr,
    x_ptr,
    w1_ptr,
    w2_ptr,
    sg_ptr,
    gx_ptr,
    part_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    n_g,
    n_prog,
    sgg,
    sgt,
    sgo,
    sxt,
    sxi,
    s1g,
    s1a,
    s1b,
    s2g,
    s2c,
    s2d,
    sxgt,
    sxgi,
    scale,
    want_gw1: tl.constexpr,
    group_scale: tl.constexpr,
    prec: tl.constexpr,
    MIX: tl.constexpr,
    BT: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    BD: tl.constexpr,
    BK: tl.constexpr,
):
    """gx[t, v*d + k] = sum_g s_g sum_u w1[g, u, v] P_g[(t, u), k], P_g = G_g @ w2_g;
    with want_gw1, part[g, prog] = s_g sum_t P_g[t] X[t]^T (pa, pb) over this
    program's k columns, summed over programs by the caller.

    Grid: (d tiles, token tiles); g is (G, T, a*c), gx (T, b*d).
    """
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    prog = pid_t * tl.num_programs(0) + pid_d
    ct = g_ptr.dtype.element_ty

    ra = tl.arange(0, BT * PA)
    tok_a = pid_t * BT + ra // PA
    ua = ra % PA
    mrow_a = (tok_a < n_tok) & (ua < n_a)
    rb = tl.arange(0, BT * PB)
    tok_b = pid_t * BT + rb // PB
    mrow_b = (tok_b < n_tok) & ((rb % PB) < n_b)
    rd = pid_d * BD + tl.arange(0, BD)
    md = rd < n_d
    xoff = tok_b[:, None] * sxt + ((rb % PB)[:, None] * n_d + rd[None, :]) * sxi
    xmask = mrow_b[:, None] & md[None, :]
    if want_gw1:
        xt = tl.load(x_ptr + xoff, mask=xmask, other=0.0)

    gx = tl.zeros((BT * PB, BD), tl.float32)
    for grp in range(0, n_g):
        # P rows (token, u): G2 @ w2 over k = c.
        acc = tl.zeros((BT * PA, BD), tl.float32)
        for k0 in range(0, n_c, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < n_c
            gv = tl.load(
                g_ptr
                + grp * sgg
                + tok_a[:, None] * sgt
                + (ua[:, None] * n_c + rk[None, :]) * sgo,
                mask=mrow_a[:, None] & mk[None, :],
                other=0.0,
            )
            wv = tl.load(
                w2_ptr + grp * s2g + rk[:, None] * s2c + rd[None, :] * s2d,
                mask=mk[:, None] & md[None, :],
                other=0.0,
            )
            acc = tl.dot(gv, wv, acc, input_precision=prec)
        p = acc.to(ct)
        if group_scale:
            sc = tl.load(sg_ptr + grp)
        else:
            sc = scale
        mixed = _mix_rows(
            p, w1_ptr + grp * s1g, s1a, s1b, n_a, n_b, ct, prec, True, MIX, BT, PA, PB
        )
        gx += mixed * sc
        if want_gw1:
            if MIX == "batch":
                p3 = tl.reshape(p, (BT, PA, BD))
                x3 = tl.permute(tl.reshape(xt, (BT, PB, BD)), (0, 2, 1))
                z = tl.dot(p3, x3, input_precision=prec)
                blk = tl.sum(z, axis=0)
            else:
                z = tl.dot(p, tl.trans(xt), input_precision=prec)
                ta = ra // PA
                tb = rb // PB
                z = tl.where(ta[:, None] == tb[None, :], z, 0.0)
                z4 = tl.reshape(z, (BT, PA, BT, PB))
                blk = tl.sum(tl.sum(z4, axis=2), axis=0)
            pa_ = tl.arange(0, PA)
            pb_ = tl.arange(0, PB)
            tl.store(
                part_ptr
                + (grp * n_prog + prog) * (PA * PB)
                + pa_[:, None] * PB
                + pb_[None, :],
                blk * sc,
            )
    tl.store(
        gx_ptr
        + tok_b[:, None] * sxgt
        + ((rb % PB)[:, None] * n_d + rd[None, :]) * sxgi,
        gx.to(gx_ptr.dtype.element_ty),
        mask=xmask,
    )


@triton.jit
def _gemm_gw2_kernel(
    g_ptr,
    x_ptr,
    w1_ptr,
    sg_ptr,
    part_ptr,
    n_tok,
    n_a,
    n_b,
    n_c,
    n_d,
    n_g,
    n_ct,
    n_dt,
    sgg,
    sgt,
    sgo,
    sxt,
    sxi,
    s1g,
    s1a,
    s1b,
    scale,
    group_scale: tl.constexpr,
    prec: tl.constexpr,
    MIX: tl.constexpr,
    BT: tl.constexpr,
    PA: tl.constexpr,
    PB: tl.constexpr,
    BC: tl.constexpr,
    BD: tl.constexpr,
):
    """part[split, g, n, k] = s_g sum over this split's token tiles of
    N_g[(t, v), n] X[(t, v), k], N_g = (I (x) w1_g^T) G_g: the split-K share of
    gw2_g = N_g^T X2, summed over splits by the caller.

    Grid: (G * c tiles * d tiles, splits); a split takes token tiles split,
    split + splits, ...
    """
    pid_o = tl.program_id(0)
    split = tl.program_id(1)
    n_split = tl.num_programs(1)
    grp = pid_o // (n_ct * n_dt)
    rest = pid_o % (n_ct * n_dt)
    rc = (rest // n_dt) * BC + tl.arange(0, BC)
    rd = (rest % n_dt) * BD + tl.arange(0, BD)
    mc = rc < n_c
    md = rd < n_d
    ct = g_ptr.dtype.element_ty
    ra = tl.arange(0, BT * PA)
    ua = ra % PA
    rb = tl.arange(0, BT * PB)
    vb = rb % PB
    w1g = w1_ptr + grp * s1g

    acc = tl.zeros((BC, BD), tl.float32)
    n_tt = tl.cdiv(n_tok, BT)
    for tile in range(split, n_tt, n_split):
        tok_a = tile * BT + ra // PA
        tok_b = tile * BT + rb // PB
        gv = tl.load(
            g_ptr
            + grp * sgg
            + tok_a[:, None] * sgt
            + (ua[:, None] * n_c + rc[None, :]) * sgo,
            mask=((tok_a < n_tok) & (ua < n_a))[:, None] & mc[None, :],
            other=0.0,
        )
        # N rows (token, v) in g's dtype, as eager's chain rounds it.
        n_tile = _mix_rows(gv, w1g, s1a, s1b, n_a, n_b, ct, prec, True, MIX, BT, PA, PB)
        xv = tl.load(
            x_ptr + tok_b[:, None] * sxt + (vb[:, None] * n_d + rd[None, :]) * sxi,
            mask=((tok_b < n_tok) & (vb < n_b))[:, None] & md[None, :],
            other=0.0,
        )
        acc = tl.dot(tl.trans(n_tile.to(ct)), xv, acc, input_precision=prec)
    if group_scale:
        acc = acc * tl.load(sg_ptr + grp)
    else:
        acc = acc * scale
    tl.store(
        part_ptr + ((split * n_g + grp) * n_c + rc[:, None]) * n_d + rd[None, :],
        acc,
        mask=mc[:, None] & md[None, :],
    )
