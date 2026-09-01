"""Import every module in the package, each in its own interpreter.

One process per module is what makes this a cycle check: inside a single
process the first successful import hides a cycle in everything imported
after it. Proving the package imports without the optional backends is the
other half, so a backend's own subpackage is skipped when that backend is not
installed — nothing imports it in that case either.

Usage:
    python scripts/ci/import_check.py
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "lycoris"

sys.path.insert(0, str(ROOT))
from lycoris.kernels.dispatch import _PROBE  # noqa: E402

# {"lycoris.kernels.triton": "triton", ...} — derived from the dispatch table
# so a new backend does not need this script edited too.
BACKEND_ROOTS = {f"lycoris.kernels.{name}": mod for name, mod in _PROBE.items()}


def module_names():
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        yield ".".join(parts)


def skip_reason(name: str) -> str | None:
    for root, requirement in BACKEND_ROOTS.items():
        if name == root or name.startswith(f"{root}."):
            if importlib.util.find_spec(requirement) is None:
                return f"{requirement} not installed"
    return None


def main() -> int:
    failed, skipped, checked = [], 0, 0
    for name in module_names():
        reason = skip_reason(name)
        if reason:
            skipped += 1
            print(f"skip {name} ({reason})")
            continue
        checked += 1
        proc = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failed.append(name)
            print(f"FAIL {name}")
            print(proc.stderr)
        else:
            print(f"ok   {name}")
    print(
        f"\n{checked - len(failed)}/{checked} modules import cleanly, {skipped} skipped"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
