"""Draw every family's figures from the measure scripts' JSON.

One plotter for all families (the figure set is uniform); pass --dir where
the measure scripts wrote their JSON.

Usage:
    .venv/Scripts/python scripts/bench/kernels/plot_all.py --dir out/bench/kernels
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.bench.kernels.harness import load_rows
from scripts.bench.kernels.plotting import draw_family

FAMILIES = ("lora", "lokr", "oft", "boft", "dora")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/bench/kernels")
    args = ap.parse_args()
    for fam in FAMILIES:
        path = Path(args.dir) / f"{fam}.json"
        if not path.exists():
            print(f"skip {fam}: {path} missing")
            continue
        meta, rows = load_rows(str(path))
        for sub in sorted({r["family"] for r in rows}):
            sub_rows = [r for r in rows if r["family"] == sub]
            draw_family(sub_rows, meta, str(Path(args.dir) / f"{sub}.png"))


if __name__ == "__main__":
    main()
