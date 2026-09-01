"""Every algorithm, every path, small shapes: precision and speed in one pass.

The working loop. Small shapes only — a kernel that is wrong or slow shows it
at 1280x1280 as clearly as at 11008x4096, and this finishes in seconds rather
than tens of minutes.

Target ordering, both stated per row:
    speed      tilelang >= triton > compile > eager
    precision  tilelang ~ triton >= compile/eager

Usage:
    .venv/Scripts/python scripts/bench/kernels/quick.py [--dtype fp16|bf16|fp32]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft
from lycoris.kernels.autograd import (
    apply_dora,
    boft_bypass_diff,
    boft_diff_weight,
    diag_oft_bypass_diff,
    diag_oft_diff_weight,
    ia3_bypass,
    locon_bypass_diff,
    locon_diff_weight,
    loha_bypass_diff,
    loha_diff_weight,
    lokr_bypass_diff,
    lokr_diff_weight,
)
from lycoris.kernels.dispatch import fused_backends
from scripts.bench.kernels.harness import device_ms, ulp_err
from test.kernels import refs

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
BACKENDS = list(fused_backends())
ARMS = ["eager", "compile", *BACKENDS]
DEV = "cuda"


def mk(dt, *shape):
    return torch.randn(*shape, device=DEV, dtype=dt) * 0.1


def case(name, build, ref64, dt, rows):
    """One row per arm: device time and ULP against the fp64 oracle."""
    times, ulps = {}, {}
    for arm in ARMS:
        fn = build(arm)
        if fn is None:
            continue
        out = fn()
        ulps[arm] = ulp_err(out, ref64, dt, "rms")
        times[arm] = device_ms(fn)
    base = times.get("eager", float("nan"))
    row = {"case": name, "times": times, "ulps": ulps, "base": base}
    rows.append(row)
    parts = []
    for arm in ARMS:
        if arm in times:
            speed = base / times[arm] if times[arm] > 0 else float("nan")
            parts.append(
                f"{arm[:4]} {times[arm] * 1e3:7.1f}us {speed:5.2f}x {ulps[arm]:6.2f}"
            )
    print(f"  {name:26s} " + " | ".join(parts))
    return row


def compiled(fn):
    torch._dynamo.reset()
    return torch.compile(fn, dynamic=None)


def run(dt):
    torch.manual_seed(0)
    rows = []
    o = i = 1280
    r, t = 16, 512
    print(f"\n=== {str(dt).split('.')[-1]} | o=i={o} r={r} t={t} ===")
    print(
        "  case                       "
        + " | ".join(f"{a[:4]:<4} time    speed   ulp" for a in ARMS)
    )

    # lora
    up, down, x, w = mk(dt, o, r), mk(dt, r, i), mk(dt, t, i), mk(dt, o, i)
    ref = (up.double() @ down.double()) * 0.5
    case(
        "lora merge",
        lambda a: (
            (lambda: (up @ down) * 0.5)
            if a == "eager"
            else (
                compiled(lambda: (up @ down) * 0.5)
                if a == "compile"
                else (lambda: locon_diff_weight(down, up, None, 0.5, a))
            )
        ),
        ref,
        dt,
        rows,
    )
    ref = ((x.double() @ down.double().T) @ up.double().T) * 0.5
    case(
        "lora bypass",
        lambda a: (
            (lambda: ((x @ down.T) @ up.T) * 0.5)
            if a == "eager"
            else (
                compiled(lambda: ((x @ down.T) @ up.T) * 0.5)
                if a == "compile"
                else (lambda: locon_bypass_diff(x, down, up, 0.5, a))
            )
        ),
        ref,
        dt,
        rows,
    )

    # loha
    w1d, w1u, w2d, w2u = mk(dt, r, i), mk(dt, o, r), mk(dt, r, i), mk(dt, o, r)
    ref = (w1u.double() @ w1d.double()) * (w2u.double() @ w2d.double()) * 0.5
    eager_loha = lambda: (w1u @ w1d) * (w2u @ w2d) * 0.5
    case(
        "loha merge",
        lambda a: (
            eager_loha
            if a == "eager"
            else (
                compiled(eager_loha)
                if a == "compile"
                else (
                    lambda: loha_diff_weight(w1d, w1u, w2d, w2u, gamma=0.5, backend=a)
                )
            )
        ),
        ref,
        dt,
        rows,
    )
    ref = (
        0.5
        * x.double()
        @ ((w1u.double() @ w1d.double()) * (w2u.double() @ w2d.double())).T
    )
    eager_lohab = lambda: 0.5 * x @ ((w1u @ w1d) * (w2u @ w2d)).T
    case(
        "loha bypass",
        lambda a: (
            eager_lohab
            if a == "eager"
            else (
                compiled(eager_lohab)
                if a == "compile"
                else (lambda: loha_bypass_diff(x, w1d, w1u, w2d, w2u, 0.5, a))
            )
        ),
        ref,
        dt,
        rows,
    )

    # lokr: both full, and the larger side factorized
    # scale = gamma / rank, and rank falls back to gamma when neither side is
    # factorized, so gamma=1 is the unit-scale case for both-full.
    a_, b_, c_, d_ = 32, 32, o // 32, i // 32
    w1, w2 = mk(dt, a_, b_), mk(dt, c_, d_)
    ref = torch.kron(w1.double(), w2.double())
    eager_lokr = lambda: torch.kron(w1, w2)
    case(
        "lokr merge (both full)",
        lambda a: (
            eager_lokr
            if a == "eager"
            else (
                compiled(eager_lokr)
                if a == "compile"
                else (
                    lambda: lokr_diff_weight(
                        w1, None, None, w2, None, None, None, 1.0, a
                    )
                )
            )
        ),
        ref,
        dt,
        rows,
    )
    w2a, w2b = mk(dt, c_, r), mk(dt, r, d_)
    ref = torch.kron(w1.double(), w2a.double() @ w2b.double()) * (0.5 / r)
    eager_lokr2 = lambda: torch.kron(w1, w2a @ w2b) * (0.5 / r)
    case(
        "lokr merge (B factored)",
        lambda a: (
            eager_lokr2
            if a == "eager"
            else (
                compiled(eager_lokr2)
                if a == "compile"
                else (
                    lambda: lokr_diff_weight(
                        w1, None, None, None, w2a, w2b, None, 0.5, a
                    )
                )
            )
        ),
        ref,
        dt,
        rows,
    )
    xk = mk(dt, t, b_ * d_)
    ref = xk.double() @ torch.kron(w1.double(), w2.double()).T
    eager_lokrb = lambda: xk @ torch.kron(w1, w2).T
    case(
        "lokr bypass",
        lambda a: (
            eager_lokrb
            if a == "eager"
            else (
                compiled(eager_lokrb)
                if a == "compile"
                else (
                    lambda: lokr_bypass_diff(
                        xk, w1, None, None, w2, None, None, None, 1.0, a
                    )
                )
            )
        ),
        ref,
        dt,
        rows,
    )

    # oft / boft — oracles come from refs (fp64, independent of the kernels)
    s, k = 8, o // 8
    blocks = mk(dt, k, s, s) * 0.5
    ref = refs.bd_fused(blocks, w, None, 1.0, True, True)
    # backend="torch" on the eager arms: the functional API dispatches now.
    eager_oft = lambda: f_oft.diff_weight(w, blocks, None, backend="torch")
    case(
        "oft merge",
        lambda a: (
            eager_oft
            if a == "eager"
            else (
                compiled(eager_oft)
                if a == "compile"
                else (lambda: diag_oft_diff_weight(w, blocks, backend=a))
            )
        ),
        ref,
        dt,
        rows,
    )
    y = mk(dt, t, o)
    ref = refs.bd_fused(blocks, y, None, 1.0, True, False)
    eager_oftb = lambda: f_oft.bypass_forward_diff(
        None, y, blocks, None, backend="torch"
    )
    case(
        "oft bypass",
        lambda a: (
            eager_oftb
            if a == "eager"
            else (
                compiled(eager_oftb)
                if a == "compile"
                else (lambda: diag_oft_bypass_diff(y, blocks, backend=a))
            )
        ),
        ref,
        dt,
        rows,
    )
    m = 3
    bb = mk(dt, m, k, s, s) * 0.5
    ref = refs.butterfly_blocks(bb, w, axis=0) - w.double()
    eager_boft = lambda: f_boft.diff_weight(w, bb, None, backend="torch")
    case(
        "boft merge",
        lambda a: (
            eager_boft
            if a == "eager"
            else (
                compiled(eager_boft)
                if a == "compile"
                else (lambda: boft_diff_weight(w, bb, None, None, 1, a))
            )
        ),
        ref,
        dt,
        rows,
    )
    ref = refs.butterfly_blocks(bb, y, axis=-1) - y.double()
    eager_boftb = lambda: f_boft.bypass_forward_diff(y, bb, None, backend="torch")
    case(
        "boft bypass",
        lambda a: (
            eager_boftb
            if a == "eager"
            else (
                compiled(eager_boftb)
                if a == "compile"
                else (lambda: boft_bypass_diff(y, bb, None, None, 1, False, a))
            )
        ),
        ref,
        dt,
        rows,
    )

    # dora / ia3
    dsc = torch.rand(o, 1, device=DEV, dtype=dt) + 0.5
    eps = torch.finfo(dt).eps
    nrm = w.double().norm(dim=1, keepdim=True) + eps
    ref = w.double() * (0.8 * (dsc.double() / nrm - 1) + 1)
    eager_dora = lambda: w * (0.8 * (dsc / (w.norm(dim=1, keepdim=True) + eps) - 1) + 1)
    case(
        "dora",
        lambda a: (
            eager_dora
            if a == "eager"
            else (
                compiled(eager_dora)
                if a == "compile"
                else (lambda: apply_dora(w, dsc, 0.8, True, a))
            )
        ),
        ref,
        dt,
        rows,
    )
    wch = mk(dt, i)
    ref = x.double() * (1 + 0.9 * wch.double())
    eager_ia3 = lambda: x * (1 + 0.9 * wch)
    case(
        "ia3 bypass",
        lambda a: (
            eager_ia3
            if a == "eager"
            else (
                compiled(eager_ia3)
                if a == "compile"
                else (lambda: ia3_bypass(x, wch, -1, 0.9, False, a))
            )
        ),
        ref,
        dt,
        rows,
    )
    return rows


def report(rows):
    """Does each row meet tilelang >= triton > compile > eager, and the ULP rule."""
    print("\n--- ordering check (speed: tl >= tt > co > ea; ulp: tt/tl <= ea) ---")
    bad = 0
    for row in rows:
        t_, u = row["times"], row["ulps"]
        ok_speed = all(
            t_.get(a, float("inf")) <= t_.get(b, float("inf")) * 1.05
            for a, b in (
                ("tilelang", "triton"),
                ("triton", "compile"),
                ("compile", "eager"),
            )
            if a in t_ and b in t_
        )
        ok_ulp = all(
            u.get(a, 0) <= u.get("eager", 0) * 1.05 for a in BACKENDS if a in u
        )
        if not (ok_speed and ok_ulp):
            bad += 1
            why = []
            if not ok_speed:
                why.append(
                    "speed "
                    + " ".join(f"{a}={t_[a] * 1e3:.1f}us" for a in ARMS if a in t_)
                )
            if not ok_ulp:
                why.append("ulp " + " ".join(f"{a}={u[a]:.1f}" for a in ARMS if a in u))
            print(f"  MISS {row['case']:26s} " + "; ".join(why))
    print(f"  {len(rows) - bad}/{len(rows)} rows meet the target ordering")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=[*DTYPES, "all"])
    args = ap.parse_args()
    picks = list(DTYPES.values()) if args.dtype == "all" else [DTYPES[args.dtype]]
    for dt in picks:
        report(run(dt))


if __name__ == "__main__":
    main()
