"""Drive the full kernel bench suite: denominators first, then every family.

Fresh process per stage so a failure cannot poison the next measurement.

Usage:
    .venv/Scripts/python scripts/bench/kernels/run_all.py --out out/bench/kernels
"""

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = (
    "denominators.py",
    "lora.py",
    "lokr.py",
    "oft.py",
    "boft.py",
    "dora.py",
    "plot_all.py",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/kernels")
    args = ap.parse_args()
    here = Path(__file__).parent
    for stage in STAGES:
        flag = "--dir" if stage == "plot_all.py" else "--out"
        cmd = [sys.executable, str(here / stage), flag, args.out]
        print("=" * 70)
        print(" ".join(cmd))
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise SystemExit(f"{stage} failed with {res.returncode}")


if __name__ == "__main__":
    main()
