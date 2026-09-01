"""F7 (TileLang): DoRA weight-decompose epilogue, one kernel per direction.

Mirrors the Triton twin: 1D grid over row tiles, both column passes inside
one CTA (reduce, then apply), so the second read of W is L2-hot rather than a
second launch. Both reduction sums are always computed — a branch-dead
fragment breaks TileLang's layout inference — and the forward's dot term is
simply discarded.
"""

import tilelang
import tilelang.language as T
import torch

from ...plans import dora as plan
from ...plans import tune
from ...plans.device import resolve_device


@tilelang.jit
def _dora_fwd(ROWS, COLS, dtype, bm=64, bn=128, threads=128):
    @T.prim_func
    def main(
        w: T.Tensor((ROWS, COLS), dtype),
        d: T.Tensor((ROWS,), "float32"),
        norm: T.Tensor((ROWS,), "float32"),
        y: T.Tensor((ROWS, COLS), dtype),
        mult: T.float32,
        eps: T.float32,
    ):
        with T.Kernel(T.ceildiv(ROWS, bm), threads=threads) as bx:
            blk = T.alloc_fragment((bm, bn), "float")
            acc = T.alloc_fragment((bm,), "float")
            part = T.alloc_fragment((bm,), "float")
            nrm = T.alloc_fragment((bm,), "float")
            scl = T.alloc_fragment((bm,), "float")

            # Pass 1: ||w_row||_2 = sqrt(sum_c w^2) + eps.
            T.clear(acc)
            for c0 in T.serial(T.ceildiv(COLS, bn)):
                for i, j in T.Parallel(bm, bn):
                    ok = (bx * bm + i < ROWS) and (c0 * bn + j < COLS)
                    v = T.if_then_else(
                        ok,
                        T.cast(
                            w[
                                T.min(bx * bm + i, ROWS - 1),
                                T.min(c0 * bn + j, COLS - 1),
                            ],
                            "float32",
                        ),
                        T.cast(0, "float32"),
                    )
                    blk[i, j] = v * v
                T.reduce_sum(blk, part, dim=1)
                for i in T.Parallel(bm):
                    acc[i] += part[i]
            # Pass 2: y = w * s with s = mult*(d/n - 1) + 1.
            for i in T.Parallel(bm):
                nrm[i] = T.sqrt(acc[i]) + eps
                if bx * bm + i < ROWS:
                    norm[bx * bm + i] = nrm[i]
                scl[i] = mult * (
                    T.if_then_else(
                        bx * bm + i < ROWS,
                        d[T.min(bx * bm + i, ROWS - 1)],
                        T.cast(1, "float32"),
                    )
                    / nrm[i]
                    - T.cast(1, "float32")
                ) + T.cast(1, "float32")
            for c0 in T.serial(T.ceildiv(COLS, bn)):
                for i, j in T.Parallel(bm, bn):
                    if (bx * bm + i < ROWS) and (c0 * bn + j < COLS):
                        y[bx * bm + i, c0 * bn + j] = T.cast(
                            T.cast(w[bx * bm + i, c0 * bn + j], "float32") * scl[i],
                            dtype,
                        )

    return main


@tilelang.jit
def _dora_bwd(ROWS, COLS, dtype, bm=64, bn=128, threads=128):
    @T.prim_func
    def main(
        g: T.Tensor((ROWS, COLS), dtype),
        w: T.Tensor((ROWS, COLS), dtype),
        d: T.Tensor((ROWS,), "float32"),
        norm: T.Tensor((ROWS,), "float32"),
        gw: T.Tensor((ROWS, COLS), dtype),
        gd: T.Tensor((ROWS,), dtype),
        mult: T.float32,
    ):
        with T.Kernel(T.ceildiv(ROWS, bm), threads=threads) as bx:
            blk = T.alloc_fragment((bm, bn), "float")
            acc = T.alloc_fragment((bm,), "float")
            part = T.alloc_fragment((bm,), "float")
            scl = T.alloc_fragment((bm,), "float")
            cof = T.alloc_fragment((bm,), "float")

            # Pass 1: rowdot = sum_c g*w (norm comes from the forward).
            T.clear(acc)
            for c0 in T.serial(T.ceildiv(COLS, bn)):
                for i, j in T.Parallel(bm, bn):
                    ok = (bx * bm + i < ROWS) and (c0 * bn + j < COLS)
                    r = T.min(bx * bm + i, ROWS - 1)
                    c = T.min(c0 * bn + j, COLS - 1)
                    blk[i, j] = T.if_then_else(
                        ok,
                        T.cast(g[r, c], "float32") * T.cast(w[r, c], "float32"),
                        T.cast(0, "float32"),
                    )
                T.reduce_sum(blk, part, dim=1)
                for i in T.Parallel(bm):
                    acc[i] += part[i]
            # gd = mult*rowdot/n; s = mult*(d/n - 1) + 1; coef = -mult*d*rowdot/n^3.
            for i in T.Parallel(bm):
                r = T.min(bx * bm + i, ROWS - 1)
                nv = norm[r]
                dv = d[r]
                scl[i] = mult * (dv / nv - T.cast(1, "float32")) + T.cast(1, "float32")
                cof[i] = -mult * dv * acc[i] / (nv * nv * nv)
                if bx * bm + i < ROWS:
                    gd[bx * bm + i] = T.cast(mult * acc[i] / nv, dtype)
            # Pass 2: gw = g*s + coef*w.
            for c0 in T.serial(T.ceildiv(COLS, bn)):
                for i, j in T.Parallel(bm, bn):
                    if (bx * bm + i < ROWS) and (c0 * bn + j < COLS):
                        r = bx * bm + i
                        c = c0 * bn + j
                        gw[r, c] = T.cast(
                            T.cast(g[r, c], "float32") * scl[i]
                            + cof[i] * T.cast(w[r, c], "float32"),
                            dtype,
                        )

    return main


def _dt(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


def _view(w: torch.Tensor, row_axis: int) -> torch.Tensor:
    return w if row_axis == 0 else w.transpose(0, 1)


def _pick(rows, cols, w, name, factory):
    return tune.tuned(
        name,
        (rows, cols, str(w.dtype)),
        lambda: plan.topk_row_reduce(rows, cols, resolve_device(), w.element_size()),
        factory,
    )


def dora_fwd(w, dscale, mult=1.0, row_axis=0):
    """(y, norms) in one launch; norms are saved for the backward."""
    wv = _view(w, row_axis).contiguous()
    rows, cols = wv.shape
    dsc = dscale.reshape(-1).to(torch.float32).contiguous()
    norms = torch.empty(rows, device=w.device, dtype=torch.float32)
    y = torch.empty_like(wv)
    eps = torch.finfo(w.dtype).eps

    def run(p, o_norm, o_y):
        fn = _dora_fwd(rows, cols, _dt(w), bm=p.bm, bn=p.bn, threads=32 * p.warps)
        fn(wv, dsc, o_norm, o_y, float(mult), float(eps))

    def factory(p):
        s1, s2 = torch.empty_like(norms), torch.empty_like(y)
        return lambda: run(p, s1, s2)

    best = _pick(rows, cols, w, "tilelang.dora.fwd", factory)
    run(best, norms, y)
    return (y if row_axis == 0 else y.transpose(0, 1)), norms


def dora_bwd(grad, w, dscale, norms, mult=1.0, row_axis=0):
    """(gw, gd) in one launch."""
    wv = _view(w, row_axis).contiguous()
    gv = _view(grad, row_axis).contiguous()
    rows, cols = wv.shape
    dsc = dscale.reshape(-1).to(torch.float32).contiguous()
    gw = torch.empty_like(wv)
    # Each CTA owns its rows outright, so gd is stored in the parameter dtype.
    gd = torch.empty(rows, device=w.device, dtype=w.dtype)

    def run(p, o_gw, o_gd):
        fn = _dora_bwd(rows, cols, _dt(w), bm=p.bm, bn=p.bn, threads=32 * p.warps)
        fn(gv, wv, dsc, norms, o_gw, o_gd, float(mult))

    def factory(p):
        s1, s2 = torch.empty_like(gw), torch.empty_like(gd)
        return lambda: run(p, s1, s2)

    best = _pick(rows, cols, w, "tilelang.dora.bwd", factory)
    run(best, gw, gd)
    return (gw if row_axis == 0 else gw.transpose(0, 1)), gd
