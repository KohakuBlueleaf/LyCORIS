"""Drive the full kernel bench suite: denominators first, then every family.

One process for every stage. A fresh interpreter per stage costs ~10 s of
torch/triton/tilelang import each and throws away the tuning table, the JIT
caches and the flush buffer, which is most of a suite's wall time; ``--isolate``
keeps the old behaviour for the case where a stage must not poison the next.

Usage:
    .venv/Scripts/python scripts/bench/kernels/run_all.py --out out/bench/kernels
"""

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("denominators", "lora", "lokr", "oft", "boft", "dora")
PLOTS = "plot_all"


def _isolated(out: str) -> None:
    here = Path(__file__).parent
    for stage in (*STAGES, PLOTS):
        flag = "--dir" if stage == PLOTS else "--out"
        cmd = [sys.executable, str(here / f"{stage}.py"), flag, out]
        print("=" * 70, "\n", " ".join(cmd), sep="")
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            raise SystemExit(f"{stage} failed with {res.returncode}")


def _in_process(out: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    for stage in (*STAGES, PLOTS):
        module = importlib.import_module(f"scripts.bench.kernels.{stage}")
        flag = "--dir" if stage == PLOTS else "--out"
        sys.argv = [stage, flag, out]
        started = time.perf_counter()
        print("=" * 70, f"\n{stage}", sep="")
        module.main()
        print(f"{stage} took {time.perf_counter() - started:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/kernels")
    ap.add_argument("--isolate", action="store_true", help="one process per stage")
    args = ap.parse_args()
    started = time.perf_counter()
    (_isolated if args.isolate else _in_process)(args.out)
    print(f"suite took {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
