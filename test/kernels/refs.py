"""fp64 references for the fused kernels.

Every reference is written independently of the kernels (plain torch ops in
float64) so it shares no assumptions with what it checks.
"""

import torch


def lowrank_rebuild(a1, b1, a2=None, b2=None, base=None, gamma=1.0, mode="plain"):
    p1 = a1.double() @ b1.double()
    if mode == "hada":
        out = p1 * (a2.double() @ b2.double())
    elif mode == "sum":
        out = p1 + a2.double() @ b2.double()
    else:
        out = p1
    out = out * gamma
    if base is not None:
        out = out + base.double()
    return out


def tucker_rebuild(a, t, b, gamma=1.0):
    return torch.einsum("op,pqk,qi->oik", a.double(), t.double(), b.double()) * gamma


def hada_delta(x, a1, b1, a2, b2, gamma=1.0):
    dw = (a1.double() @ b1.double()) * (a2.double() @ b2.double())
    return gamma * x.double() @ dw.T


def kron_rebuild(w1, w2, base=None, gamma=1.0):
    out = torch.kron(w1.double(), w2.double()) * gamma
    if base is not None:
        out = out + base.double()
    return out


def kron_apply(x, w1, w2, gamma=1.0):
    return gamma * x.double() @ torch.kron(w1.double(), w2.double()).T


def cayley(blocks, cscale=1.0):
    """R = (I + q)(I - q)^-1 with q = (b - b^T)*cscale, in fp64."""
    b = blocks.double()
    q = (b - b.transpose(-1, -2)) * cscale
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype).expand_as(q)
    return (eye + q) @ torch.linalg.inv(eye - q)


def bd_fused(blocks, x, rescale=None, cscale=1.0, shift=True, weight=True):
    """Diag-OFT apply exactly as the eager path does it (contracting R's
    FIRST index, i.e. an R^T apply), with rescale and the diff shift."""
    r = cayley(blocks, cscale)
    k, s, _ = r.shape
    if weight:
        xv = x.double().reshape(k, s, -1)
        out = torch.einsum("kij,kic->kjc", r, xv).reshape(x.shape)
    else:
        axis_last = x.dim() - 1
        if x.shape[axis_last] == k * s:
            xv = x.double().reshape(*x.shape[:-1], k, s)
            out = torch.einsum("kij,...ki->...kj", r, xv).reshape(x.shape)
        else:
            xv = x.double().reshape(x.shape[0], k, s, -1)
            out = torch.einsum("kij,bkil->bkjl", r, xv).reshape(x.shape)
    if rescale is not None:
        shape = [1] * out.dim()
        shape[0 if weight else (1 if out.dim() > 2 else out.dim() - 1)] = -1
        out = out * rescale.double().reshape(shape)
    return out - x.double() if shift else out


def bd_weight(rf, w):
    k, s, _ = rf.shape
    wv = w.double().reshape(k, s, -1)
    return torch.einsum("kij,kjc->kic", rf.double(), wv).reshape(w.shape)


def bd_act(rf, x, channel_axis=-1):
    k, s, _ = rf.shape
    if channel_axis in (-1, x.dim() - 1):
        xv = x.double().reshape(*x.shape[:-1], k, s)
        out = torch.einsum("kij,...kj->...ki", rf.double(), xv)
        return out.reshape(x.shape)
    xv = x.double().reshape(x.shape[0], k, s, -1)
    out = torch.einsum("kij,bkjl->bkil", rf.double(), xv)
    return out.reshape(x.shape)


def butterfly_blocks(blocks, w, cscale=1.0, scale=1.0, axis=0):
    """BOFT from raw oft_blocks: Cayley + multiplier fold, then the stages."""
    r = cayley(blocks, cscale)
    if scale != 1:
        eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
        r = r * scale + (1 - scale) * eye
    return butterfly_rows(r, w) if axis == 0 else butterfly_cols(r, w)


def butterfly_rows(rs, w):
    """The eager BOFT stage loop from functional/boft.py, in fp64."""
    inp = w.double()
    m, _, b, _ = rs.shape
    r_b = b // 2
    for i in range(m):
        bi = rs[i].double()
        g, kd = 2, (2**i) * r_b
        inp = (
            inp.unflatten(0, (-1, g, kd))
            .transpose(1, 2)
            .flatten(0, 2)
            .unflatten(0, (-1, b))
        )
        inp = torch.einsum("b i j, b j ... -> b i ...", bi, inp)
        inp = inp.flatten(0, 1).unflatten(0, (-1, kd, g)).transpose(1, 2).flatten(0, 2)
    return inp


def butterfly_cols(rs, x):
    inp = x.double()
    m, _, b, _ = rs.shape
    r_b = b // 2
    for i in range(m):
        bi = rs[i].double()
        g, kd = 2, (2**i) * r_b
        inp = (
            inp.unflatten(-1, (-1, g, kd))
            .transpose(-2, -1)
            .flatten(-3)
            .unflatten(-1, (-1, b))
        )
        inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
        inp = inp.flatten(-2).unflatten(-1, (-1, kd, g)).transpose(-2, -1).flatten(-3)
    return inp


def channel_scale(x, w, channel_axis, alpha=1.0, gamma=1.0):
    shape = [1] * x.dim()
    shape[channel_axis] = -1
    return x.double() * (alpha + gamma * w.double().reshape(shape))


def dora_scale(w, dscale, mult=1.0, row_axis=0, eps=0.0):
    dim = 1 if row_axis == 0 else 0
    n = w.double().norm(dim=dim, keepdim=True) + eps
    s = mult * (dscale.double().reshape(n.shape) / n - 1.0) + 1.0
    return w.double() * s
