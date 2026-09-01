"""Roll every family's JSON into one verdict table.

Answers the goal's question directly: is each fused path faster than eager AND
than torch.compile, on device (kernel) time, and what does it cost in memory
and accuracy. Wall time is reported beside it so a dispatch-bound row is
visible as such.

Usage:
    .venv/Scripts/python scripts/bench/kernels/verdict.py --dir out/bench/kernels
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.bench.kernels.harness import load_rows

FAMILIES = ("lora", "lokr", "oft", "boft", "dora")
BACKENDS = ("triton", "tilelang")


def _ratio(ref, got):
    if not ref or not got or math.isnan(ref) or math.isnan(got) or got <= 0:
        return None
    return ref / got


def _geo(values):
    vals = [v for v in values if v]
    return math.exp(statistics.fmean(math.log(v) for v in vals)) if vals else None


def summarize(rows):
    """Per (sub-family, backend): geomean device speedup vs each baseline."""
    out = []
    for sub in sorted({r["family"] for r in rows}):
        sel = [r for r in rows if r["family"] == sub]
        cases = sorted({r["case"] for r in sel})
        by = {(r["case"], r["arm"]): r for r in sel}
        for be in BACKENDS:
            if not any(a == be for _, a in by):
                continue
            stats = {k: [] for k in ("dev_e", "dev_c", "wall_e", "fb_e", "fb_c")}
            vram, ulp_ours, ulp_eager, worst = [], [], [], None
            for c in cases:
                ours, eag, com = (
                    by.get((c, be)),
                    by.get((c, "eager")),
                    by.get((c, "compile")),
                )
                if not (ours and eag):
                    continue
                pairs = (
                    ("dev_e", eag.get("fwd_dev_ms"), ours.get("fwd_dev_ms")),
                    (
                        "dev_c",
                        (com or {}).get("fwd_dev_ms"),
                        ours.get("fwd_dev_ms"),
                    ),
                    ("wall_e", eag.get("fwd_ms"), ours.get("fwd_ms")),
                    ("fb_e", eag.get("fwdbwd_ms"), ours.get("fwdbwd_ms")),
                    ("fb_c", (com or {}).get("fwdbwd_ms"), ours.get("fwdbwd_ms")),
                )
                for key, ref, got in pairs:
                    val = _ratio(ref, got)
                    if val:
                        stats[key].append(val)
                        if key == "dev_e" and (worst is None or val < worst[1]):
                            worst = (c, val)
                if eag.get("vram_fwdbwd") and ours.get("vram_fwdbwd"):
                    vram.append(eag["vram_fwdbwd"] / ours["vram_fwdbwd"])
                ulp_ours.append(ours.get("ulp") or 0.0)
                ulp_eager.append(eag.get("ulp") or 0.0)
            out.append(
                {
                    "family": sub,
                    "backend": be,
                    "n": len(cases),
                    "dev_vs_eager": _geo(stats["dev_e"]),
                    "dev_vs_compile": _geo(stats["dev_c"]),
                    "wall_vs_eager": _geo(stats["wall_e"]),
                    "fb_vs_eager": _geo(stats["fb_e"]),
                    "fb_vs_compile": _geo(stats["fb_c"]),
                    "vram_vs_eager": _geo(vram),
                    "ulp_ours": max(ulp_ours or [0]),
                    "ulp_eager": max(ulp_eager or [0]),
                    "worst_case": worst,
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/bench/kernels")
    args = ap.parse_args()
    rows = []
    for fam in FAMILIES:
        path = Path(args.dir) / f"{fam}.json"
        if path.exists():
            rows += load_rows(str(path))[1]
    table = summarize(rows)
    head = (
        f"{'family':22s}{'be':9s}{'n':>3s}{'dev/eager':>10s}{'dev/comp':>9s}"
        f"{'fb/eager':>9s}{'fb/comp':>8s}{'vram':>6s}{'ulp':>6s}{'eagULP':>7s}  worst"
    )
    print(head)
    print("-" * len(head))
    fmt = lambda v: f"{v:.2f}" if v else "  -  "
    for r in table:
        worst = r["worst_case"]
        print(
            f"{r['family']:22s}{r['backend']:9s}{r['n']:>3d}"
            f"{fmt(r['dev_vs_eager']):>10s}{fmt(r['dev_vs_compile']):>9s}"
            f"{fmt(r['fb_vs_eager']):>9s}{fmt(r['fb_vs_compile']):>8s}"
            f"{fmt(r['vram_vs_eager']):>6s}{r['ulp_ours']:>6.1f}{r['ulp_eager']:>7.1f}"
            f"  {worst[0] + ' ' + fmt(worst[1]) if worst else '-'}"
        )
    print("\nAll ratios are geometric means over the family's cases; >1 is ours")
    print("faster / smaller / more accurate. dev = summed device-kernel time,")
    print("fb = fwd+bwd wall.")


if __name__ == "__main__":
    main()
