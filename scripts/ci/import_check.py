"""Import every module in the package, each in its own interpreter.

One process per module is what makes this a cycle check: inside a single
process the first successful import hides a cycle in everything imported
after it. Optional backends (triton, tilelang) are absent on CI, so this also
proves the package imports without them.

Usage:
    python scripts/ci/import_check.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "lycoris"


def module_names():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        yield ".".join(parts)


def main() -> int:
    failed = []
    names = list(module_names())
    for name in names:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failed.append((name, proc.stderr.strip().splitlines()[-1:]))
            print(f"FAIL {name}")
            print(proc.stderr)
        else:
            print(f"ok   {name}")
    print(f"\n{len(names) - len(failed)}/{len(names)} modules import cleanly")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
