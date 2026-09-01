"""F7 (TileLang): broadcast channel scale (IA3)."""

import tilelang
import tilelang.language as T
import torch

from ...plans import dora as plan
from ...plans import tune
from ...plans.device import resolve_device


@tilelang.jit(out_idx=[2])
def _scale_fwd(N, C, INNER, dtype, blk=1024, threads=128):
    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        w: T.Tensor((C,), dtype),
        y: T.Tensor((N,), dtype),
        alpha: T.float32,
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, blk), threads=threads) as (pid,):
            for i in T.Parallel(blk):
                idx = pid * blk + i
                if idx < N:
                    c = (idx // INNER) % C
                    y[idx] = T.cast(
                        T.cast(x[idx], "float32")
                        * (alpha + gamma * T.cast(w[c], "float32")),
                        dtype,
                    )

    return main


@tilelang.jit(out_idx=[3])
def _scale_bwd(N, C, INNER, dtype, slices=32, blk=1024, threads=128):
    """gw partials land in (slices, C) fp32 (contention / slices); the
    wrapper sums them."""

    @T.prim_func
    def main(
        g: T.Tensor((N,), dtype),
        x: T.Tensor((N,), dtype),
        w: T.Tensor((C,), dtype),
        gx: T.Tensor((N,), dtype),
        gw: T.Tensor((slices, C), "float32"),
        alpha: T.float32,
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, blk), threads=threads) as (pid,):
            for i in T.Parallel(blk):
                idx = pid * blk + i
                if idx < N:
                    c = (idx // INNER) % C
                    gf = T.cast(g[idx], "float32")
                    gx[idx] = T.cast(
                        gf * (alpha + gamma * T.cast(w[c], "float32")), dtype
                    )
                    T.atomic_add(
                        gw[pid % slices, c], gf * T.cast(x[idx], "float32") * gamma
                    )

    return main


def _dt(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


def _view3(x: torch.Tensor, channel_axis: int):
    xc = x.contiguous()
    if channel_axis in (-1, x.dim() - 1):
        return xc, x.shape[-1], 1
    if channel_axis == 0:
        return xc, x.shape[0], xc.numel() // x.shape[0]
    if channel_axis == 1:
        inner = 1
        for d in x.shape[2:]:
            inner *= d
        return xc, x.shape[1], inner
    raise ValueError("channel_axis must be 0, 1 or -1")


def ia3_fwd(x, w, channel_axis, alpha=1.0, gamma=1.0):
    xc, c, inner = _view3(x, channel_axis)
    flat = xc.reshape(-1)
    wc = w.reshape(-1).contiguous()
    n = flat.numel()

    def build(p):
        fn = _scale_fwd(n, c, inner, _dt(x), blk=p.bm, threads=32 * p.warps)
        return lambda: fn(flat, wc, float(alpha), float(gamma))

    best = tune.tuned(
        "tilelang.ia3.fwd",
        (tune.bucket_tokens(n), c, inner, str(x.dtype)),
        lambda: plan.topk_elementwise(n, 2.0, resolve_device(), x.element_size()),
        build,
    )
    return build(best)().view_as(x)


def ia3_bwd(grad, x, w, channel_axis, alpha=1.0, gamma=1.0):
    xc, c, inner = _view3(x, channel_axis)
    gc, _, _ = _view3(grad, channel_axis)
    slices = 32
    gw = torch.zeros(slices, c, device=x.device, dtype=torch.float32)
    gf = gc.reshape(-1)
    xf = xc.reshape(-1)
    wc = w.reshape(-1).contiguous()
    n = xf.numel()

    def run(p, o_gw):
        fn = _scale_bwd(
            n, c, inner, _dt(x), slices=slices, blk=p.bm, threads=32 * p.warps
        )
        return fn(gf, xf, wc, o_gw, float(alpha), float(gamma))

    def factory(p):
        scratch = torch.empty_like(gw)
        return lambda: run(p, scratch)

    best = tune.tuned(
        "tilelang.ia3.fwd_bwd",
        (tune.bucket_tokens(n), c, inner, str(x.dtype)),
        lambda: plan.topk_elementwise(n, 3.0, resolve_device(), x.element_size()),
        factory,
    )
    gx = run(best, gw)
    return gx.view_as(x), gw.sum(0).to(w.dtype).view_as(w)
