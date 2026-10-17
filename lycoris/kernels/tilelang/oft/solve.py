"""The Cayley solves of diag-OFT and BOFT in the I + D form (TileLang), one
program per block: D = R - I from [I - q | 2q], and the block gradients from
the summed P through Z = (I + q)^-1 G_D. See docs/kernels/designs.md.
"""

import tilelang.language as T  # noqa: N812 - TileLang's own spelling
import torch

from ...tuner import device_info
from ...tuner.measure import pad16
from ..common import compiled, fits, tl_dtype

F32 = "float32"
# Largest padded block of the shared-memory solve.
MAX_BLOCK = 64


@T.macro
def _solve(a_s, m_s, col_s, ps):
    """m_s <- a_s^-1 m_s by Gauss-Jordan (a_s = I -/+ q: no pivoting)."""
    for jj in T.serial(ps):
        for i in T.Parallel(ps):
            col_s[i] = a_s[i, jj]
        for i, j in T.Parallel(ps, ps):
            if i != jj:
                fac = col_s[i] / col_s[jj]
                a_s[i, j] = a_s[i, j] - fac * a_s[jj, j]
                m_s[i, j] = m_s[i, j] - fac * m_s[jj, j]
        for j in T.Parallel(ps):
            a_s[jj, j] = a_s[jj, j] / col_s[jj]
            m_s[jj, j] = m_s[jj, j] / col_s[jj]


@T.macro
def _skew_delta(blocks, bk, size, ps, cscale, q_s, a_s, d_s, col_s):
    """q_s = (b - b^T) * cscale and d_s = D = (I - q)^-1 2q in fp32 (zero past size)."""
    for i, j in T.Parallel(ps, ps):
        ok = (i < size) and (j < size)
        bij = T.cast(blocks[bk, T.min(i, size - 1), T.min(j, size - 1)], F32)
        bji = T.cast(blocks[bk, T.min(j, size - 1), T.min(i, size - 1)], F32)
        qv = T.if_then_else(ok, (bij - bji) * cscale, T.float32(0))
        q_s[i, j] = qv
        a_s[i, j] = T.if_then_else(i == j, T.float32(1), T.float32(0)) - qv
        d_s[i, j] = 2 * qv
    _solve(a_s, d_s, col_s, ps)


def _delta_prim(k, size, ps, rescale_on, threads, b_dt, c_dt):
    """Dh[k] = mult * D[k] * rescale (columns), padded (k, ps, ps) in the compute dtype."""

    @T.prim_func
    def main(
        blocks: T.Tensor((k, size, size), b_dt),
        res: T.Tensor((k * size,), F32),
        dh: T.Tensor((k, ps, ps), c_dt),
        cscale: T.float32,
        mult: T.float32,
    ):
        with T.Kernel(k, threads=threads) as bk:
            q_s = T.alloc_shared((ps, ps), F32)
            a_s = T.alloc_shared((ps, ps), F32)
            d_s = T.alloc_shared((ps, ps), F32)
            col_s = T.alloc_shared((ps,), F32)
            _skew_delta(blocks, bk, size, ps, cscale, q_s, a_s, d_s, col_s)
            for j, i in T.Parallel(ps, ps):
                r = T.if_then_else(
                    rescale_on and (i < size),
                    res[bk * size + T.min(i, size - 1)],
                    T.float32(1),
                )
                dh[bk, j, i] = T.cast(d_s[j, i] * mult * r, c_dt)

    return main


def _chain_prim(k, size, ps, rescale_on, threads, b_dt, c_dt):
    """g_blocks and g_rescale of one block from its summed P (fp32, G_D =
    P * rescale); the Z D^T product is one 16-bit MMA."""

    @T.prim_func
    def main(
        blocks: T.Tensor((k, size, size), b_dt),
        res: T.Tensor((k * size,), F32),
        p: T.Tensor((k, size, size), F32),
        gb: T.Tensor((k, size, size), b_dt),
        gres: T.Tensor((k * size,), F32),
        cscale: T.float32,
    ):
        with T.Kernel(k, threads=threads) as bk:
            q_s = T.alloc_shared((ps, ps), F32)
            a_s = T.alloc_shared((ps, ps), F32)
            d_s = T.alloc_shared((ps, ps), F32)
            col_s = T.alloc_shared((ps,), F32)
            p_s = T.alloc_shared((ps, ps), F32)
            z_s = T.alloc_shared((ps, ps), F32)
            zh_s = T.alloc_shared((ps, ps), c_dt)
            dt_s = T.alloc_shared((ps, ps), c_dt)
            gq_f = T.alloc_fragment((ps, ps), F32)
            gq_s = T.alloc_shared((ps, ps), F32)
            _skew_delta(blocks, bk, size, ps, cscale, q_s, a_s, d_s, col_s)
            for j, i in T.Parallel(ps, ps):
                ok = (j < size) and (i < size)
                p_s[j, i] = T.if_then_else(
                    ok, p[bk, T.min(j, size - 1), T.min(i, size - 1)], T.float32(0)
                )
            # g_rescale_i = P[i, i] + sum_j D[j, i] P[j, i]
            if rescale_on:
                for i in T.Parallel(ps):
                    if i < size:
                        acc = T.alloc_var(F32)
                        acc = p_s[i, i]
                        for j in T.serial(ps):
                            acc += d_s[j, i] * p_s[j, i]
                        gres[bk * size + i] = acc
            # Z = (I + q)^-1 G_D with G_D[j, i] = P[j, i] * rescale_i
            for j, i in T.Parallel(ps, ps):
                r = T.if_then_else(
                    rescale_on and (i < size),
                    res[bk * size + T.min(i, size - 1)],
                    T.float32(1),
                )
                a_s[j, i] = (
                    T.if_then_else(i == j, T.float32(1), T.float32(0)) + q_s[j, i]
                )
                z_s[j, i] = p_s[j, i] * r
            _solve(a_s, z_s, col_s, ps)
            for i, j in T.Parallel(ps, ps):
                zh_s[i, j] = T.cast(z_s[i, j], c_dt)
                dt_s[i, j] = T.cast(d_s[i, j], c_dt)
            T.clear(gq_f)
            T.gemm(zh_s, dt_s, gq_f, transpose_B=True)
            for i, j in T.Parallel(ps, ps):
                gq_s[i, j] = gq_f[i, j] + 2 * z_s[i, j]
            for i, j in T.Parallel(ps, ps):
                if (i < size) and (j < size):
                    gb[bk, i, j] = T.cast((gq_s[i, j] - gq_s[j, i]) * cscale, b_dt)

    return main


def _threads(ps):
    """Threads whose warps partition the (ps, ps) MMA tile on every target."""
    return {16: 32, 32: 128}.get(ps, 256)


def _res(rescale, k, s, like):
    if rescale is None:
        return torch.empty(k * s, device=like.device, dtype=torch.float32)
    return rescale.reshape(-1).float().contiguous()


def _solve_fits(device, ps, fp32_tiles, cdt_tiles, cdt) -> bool:
    """Whether a solve program's shared tiles, (ps, ps) each, fit the device."""
    nbytes = (4 * fp32_tiles + cdt.itemsize * cdt_tiles) * ps * ps + 4 * ps
    return fits(device_info(device), -(-nbytes // 2))


def _skew_delta_torch(blocks, cscale):
    """(q, D) of the (k, s, s) blocks in fp32, D = (I - q)^-1 2q."""
    b = blocks.float()
    q = (b - b.transpose(1, 2)) * cscale
    eye = torch.eye(b.shape[-1], device=b.device)
    return q, torch.linalg.solve(eye - q, 2 * q)


def _torch_deltas(blocks, rescale, cscale, cdt, mult, ps):
    k, s, _ = blocks.shape
    _, d = _skew_delta_torch(blocks, cscale)
    if rescale is not None:
        d = d * rescale.reshape(k, 1, s).float()
    dh = torch.zeros(k, ps, ps, device=blocks.device, dtype=cdt)
    dh[:, :s, :s] = d * mult
    return dh


def _torch_chain(blocks, rescale, p, cdt, cscale):
    k, s, _ = blocks.shape
    q, d = _skew_delta_torch(blocks, cscale)
    p = p.float()
    gres = torch.zeros(k * s, device=blocks.device, dtype=torch.float32)
    if rescale is not None:
        gres = (p.diagonal(dim1=1, dim2=2) + (d * p).sum(1)).reshape(-1)
        p = p * rescale.reshape(k, 1, s).float()
    eye = torch.eye(s, device=blocks.device)
    z = torch.linalg.solve(eye + q, p)
    gq = (z.to(cdt) @ d.to(cdt).transpose(1, 2)).float() + 2 * z
    return ((gq - gq.transpose(1, 2)) * cscale).to(blocks.dtype), gres


def deltas(blocks, rescale, cscale, cdt, mult=1.0):
    """Dh = mult * D * rescale (k, ps, ps) in the compute dtype, one program
    per block of the (k, s, s) blocks (torch where the tiles exceed shared
    memory)."""
    blocks = blocks.contiguous()
    k, s, _ = blocks.shape
    ps = pad16(s)
    if not _solve_fits(blocks.device, ps, 3, 0, cdt):
        return _torch_deltas(blocks, rescale, cscale, cdt, mult, ps)
    kernel = compiled(
        _delta_prim,
        blocks.device,
        k,
        s,
        ps,
        rescale is not None,
        _threads(ps),
        tl_dtype(blocks.dtype),
        tl_dtype(cdt),
    )
    dh = torch.empty(k, ps, ps, device=blocks.device, dtype=cdt)
    kernel(blocks, _res(rescale, k, s, blocks), dh, float(cscale), float(mult))
    return dh


def chain(blocks, rescale, p, cdt, cscale):
    """(g_blocks in blocks' dtype, g_rescale fp32 (k*s,)) of the (k, s, s)
    blocks from the summed P (k, s, s) fp32 (G_D = P * rescale); torch where
    the tiles exceed shared memory."""
    blocks = blocks.contiguous()
    k, s, _ = blocks.shape
    ps = pad16(s)
    if not _solve_fits(blocks.device, ps, 6, 2, cdt):
        return _torch_chain(blocks, rescale, p, cdt, cscale)
    kernel = compiled(
        _chain_prim,
        blocks.device,
        k,
        s,
        ps,
        rescale is not None,
        _threads(ps),
        tl_dtype(blocks.dtype),
        tl_dtype(cdt),
    )
    gb = torch.empty_like(blocks)
    gres = torch.empty(k * s, device=blocks.device, dtype=torch.float32)
    kernel(blocks, _res(rescale, k, s, blocks), p.contiguous(), gb, gres, float(cscale))
    return gb, gres
