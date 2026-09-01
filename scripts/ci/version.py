"""Read, stamp and bump the single version string in pyproject.toml.

One source of truth: `[project] version`. The workflows call this rather than
each hand-rolling a sed, which is how a nightly ends up with two different
version strings in the same run.

Usage:
    python scripts/ci/version.py read
    python scripts/ci/version.py nightly            # X.Y.Z.devYYYYmmddHHMMSS+sha
    python scripts/ci/version.py bump patch|minor|major
    python scripts/ci/version.py set 4.1.0
    python scripts/ci/version.py write <version>    # stamp pyproject.toml
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
# Anchored to the [project] table's own key, so a dependency pin that happens
# to contain "version = " cannot match.
PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')


def read() -> str:
    match = PATTERN.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no version found in pyproject.toml")
    return match.group(2)


def write(version: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    new, count = PATTERN.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise SystemExit("no version found in pyproject.toml")
    PYPROJECT.write_text(new, encoding="utf-8")


def base(version: str) -> tuple[int, int, int]:
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    return tuple(int(g) for g in core.groups())


def bump(version: str, part: str) -> str:
    major, minor, patch = base(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def nightly(version: str) -> str:
    """PEP 440 dev release plus a +local segment naming the commit.

    Not uploadable to PyPI (it rejects +local), which is the point: nightlies
    are installed from the GitHub release, never from the index.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    sha = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    major, minor, patch = base(version)
    tail = f"+{sha}" if sha else ""
    return f"{major}.{minor}.{patch + 1}.dev{stamp}{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("read", "nightly", "bump", "set", "write"))
    ap.add_argument("value", nargs="?")
    args = ap.parse_args()
    current = read()

    if args.action == "read":
        print(current)
    elif args.action == "nightly":
        print(nightly(current))
    elif args.action == "bump":
        print(bump(current, args.value or "patch"))
    elif args.action in ("set", "write"):
        if not args.value:
            raise SystemExit(f"{args.action} needs a version")
        write(args.value)
        print(args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
