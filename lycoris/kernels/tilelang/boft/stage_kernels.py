"""TileLang kernels of one BOFT butterfly stage on channels-last rows (forward
and backward), launched by ``.stages``. A program owns a row tile and a
channel tile of G consecutive stage blocks, whose channels are two contiguous
runs of G*b/2 (one run split in two once the tile spans whole groups); stage
i's geometry (k, blocks per group) is a runtime argument.
See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling

from ...tuner.measure import pad16

F32 = "float32"


@T.macro
def _block_tile(m_s, dh, by, k, gpb, tw, g, rb, transposed, c_dt):
    """m_s[n1, n2] = D[blk][l2, l1] (or D[blk][l1, l2] when transposed) where
    tile positions n1, n2 sit in one stage block blk, else 0."""
    for n1, n2 in T.Parallel(tw, tw):
        h = g * rb
        group0 = (by * g) // gpb
        jl0 = (by * g) % gpb
        c1 = T.if_then_else(
            g >= gpb,
            group0 * 2 * k + n1,
            group0 * 2 * k + (n1 // h) * k + jl0 * rb + n1 % h,
        )
        c2 = T.if_then_else(
            g >= gpb,
            group0 * 2 * k + n2,
            group0 * 2 * k + (n2 // h) * k + jl0 * rb + n2 % h,
        )
        r1 = c1 % (2 * k)
        r2 = c2 % (2 * k)
        b1 = (c1 // (2 * k)) * gpb + (r1 % k) // rb
        b2 = (c2 // (2 * k)) * gpb + (r2 % k) // rb
        l1 = ((r1 % k) % rb) * 2 + r1 // k
        l2 = ((r2 % k) % rb) * 2 + r2 // k
        if transposed:
            m_s[n1, n2] = T.if_then_else(b1 == b2, dh[b2, l1, l2], T.cast(0, c_dt))
        else:
            m_s[n1, n2] = T.if_then_else(b1 == b2, dh[b2, l2, l1], T.cast(0, c_dt))


def rows_fwd_prim(c, b, g, bt, threads, save, last, rescale_on, diff, dts):
    """v = u + u @ M for a row tile of one channel tile, M the permuted
    block-diagonal D^T of its G stage blocks; ``save`` also stores v in the
    compute dtype, the last stage folds rescale and the diff."""
    c_dt, u_dt, v_dt, x_dt = dts
    rb, tw, ps = b // 2, g * b, pad16(b)
    h = tw // 2
    t = T.symbolic("t")

    @T.prim_func
    def main(
        dh: T.Tensor((c // b, ps, ps), c_dt),
        u: T.Tensor((t, c), u_dt),
        v: T.Tensor((t, c), v_dt),
        sv: T.Tensor((t, c), c_dt),
        x0: T.Tensor((t, c), x_dt),
        res: T.Tensor((c,), F32),
        k: T.int32,
        gpb: T.int32,
    ):
        # Grid: (row tiles bx, channel tiles by).
        with T.Kernel(T.ceildiv(t, bt), c // tw, threads=threads) as (bx, by):
            m_s = T.alloc_shared((tw, tw), c_dt)
            u_s = T.alloc_shared((bt, tw), c_dt)
            w_s = T.alloc_shared((bt, tw), F32)
            o_f = T.alloc_fragment((bt, tw), F32)
            # Start channels of the tile's two runs of h channels.
            base = ((by * g) // gpb) * 2 * k
            s0 = T.if_then_else(g >= gpb, base, base + ((by * g) % gpb) * rb)
            s1 = T.if_then_else(g >= gpb, s0 + h, s0 + k)
            r0 = bx * bt
            _block_tile(m_s, dh, by, k, gpb, tw, g, rb, False, c_dt)
            T.copy(u[r0 : r0 + bt, s0 : s0 + h], w_s[:, 0:h])
            T.copy(u[r0 : r0 + bt, s1 : s1 + h], w_s[:, h:tw])
            for r, n in T.Parallel(bt, tw):
                u_s[r, n] = T.cast(w_s[r, n], c_dt)
            T.clear(o_f)
            T.gemm(u_s, m_s, o_f)
            for r, n in T.Parallel(bt, tw):
                w_s[r, n] = w_s[r, n] + o_f[r, n]
            if save:
                T.copy(w_s[:, 0:h], sv[r0 : r0 + bt, s0 : s0 + h])
                T.copy(w_s[:, h:tw], sv[r0 : r0 + bt, s1 : s1 + h])
            if last and rescale_on:
                for r, n in T.Parallel(bt, tw):
                    ch = T.if_then_else(n < h, s0 + n, s1 + n - h)
                    w_s[r, n] = w_s[r, n] * res[ch]
            if last and diff:
                T.copy(x0[r0 : r0 + bt, s0 : s0 + h], u_s[:, 0:h])
                T.copy(x0[r0 : r0 + bt, s1 : s1 + h], u_s[:, h:tw])
                for r, n in T.Parallel(bt, tw):
                    w_s[r, n] = w_s[r, n] - T.cast(u_s[r, n], F32)
            T.copy(w_s[:, 0:h], v[r0 : r0 + bt, s0 : s0 + h])
            T.copy(w_s[:, h:tw], v[r0 : r0 + bt, s1 : s1 + h])

    return main


def rows_bwd_prim(c, b, g, bt, splits, threads, dts):
    """g_u = g + g @ M^T over this program's rows, and its partial of
    P = g^T u (natural tile order, fp32 accumulation)."""
    c_dt, u_dt = dts
    rb, tw, ps = b // 2, g * b, pad16(b)
    h = tw // 2
    n_ct = c // tw
    t = T.symbolic("t")

    @T.prim_func
    def main(
        dh: T.Tensor((c // b, ps, ps), c_dt),
        gr: T.Tensor((t, c), F32),
        u: T.Tensor((t, c), u_dt),
        gu: T.Tensor((t, c), F32),
        part: T.Tensor((splits, n_ct, tw, tw), F32),
        k: T.int32,
        gpb: T.int32,
    ):
        # Grid: (row splits bs, channel tiles by).
        with T.Kernel(splits, n_ct, threads=threads) as (bs, by):
            m_s = T.alloc_shared((tw, tw), c_dt)
            gh_s = T.alloc_shared((bt, tw), c_dt)
            gf_s = T.alloc_shared((bt, tw), F32)
            u_s = T.alloc_shared((bt, tw), c_dt)
            o_f = T.alloc_fragment((bt, tw), F32)
            p_f = T.alloc_fragment((tw, tw), F32)
            base = ((by * g) // gpb) * 2 * k
            s0 = T.if_then_else(g >= gpb, base, base + ((by * g) % gpb) * rb)
            s1 = T.if_then_else(g >= gpb, s0 + h, s0 + k)
            _block_tile(m_s, dh, by, k, gpb, tw, g, rb, True, c_dt)
            T.clear(p_f)
            per = T.ceildiv(T.ceildiv(t, bt), splits)
            for it in T.serial(per):
                r0 = (bs * per + it) * bt
                T.copy(gr[r0 : r0 + bt, s0 : s0 + h], gf_s[:, 0:h])
                T.copy(gr[r0 : r0 + bt, s1 : s1 + h], gf_s[:, h:tw])
                T.copy(u[r0 : r0 + bt, s0 : s0 + h], u_s[:, 0:h])
                T.copy(u[r0 : r0 + bt, s1 : s1 + h], u_s[:, h:tw])
                for r, n in T.Parallel(bt, tw):
                    gh_s[r, n] = T.cast(gf_s[r, n], c_dt)
                T.clear(o_f)
                T.gemm(gh_s, m_s, o_f)
                for r, n in T.Parallel(bt, tw):
                    gf_s[r, n] = gf_s[r, n] + o_f[r, n]
                T.copy(gf_s[:, 0:h], gu[r0 : r0 + bt, s0 : s0 + h])
                T.copy(gf_s[:, h:tw], gu[r0 : r0 + bt, s1 : s1 + h])
                T.gemm(gh_s, u_s, p_f, transpose_A=True)
            T.copy(p_f, part[bs, by, 0, 0])

    return main
