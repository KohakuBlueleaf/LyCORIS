"""Release notes for a version: its Change.md section, or the commit list.

Change.md is the curated source. When it has no section for the version being
released (a patch cut by automation, say), the notes fall back to the commits
since the previous tag, so a release is never published with empty notes.

Usage:
    python scripts/ci/release_notes.py --version 4.0.0
"""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = ROOT / "Change.md"


def changelog_section(version: str) -> str | None:
    if not CHANGELOG.exists():
        return None
    text = CHANGELOG.read_text(encoding="utf-8")
    # "## <date> update to <version>" through the next "## " heading.
    pattern = re.compile(
        rf"(?ms)^##\s+[^\n]*update to {re.escape(version)}\s*$(.*?)(?=^##\s|\Z)"
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def commits_since_last_tag() -> str:
    tags = subprocess.run(
        ["git", "tag", "--sort=-creatordate"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    ).stdout.split()
    span = f"{tags[1]}..HEAD" if len(tags) > 1 else "HEAD"
    log = subprocess.run(
        ["git", "log", "--no-merges", "--pretty=* %s (%h)", span],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    ).stdout.strip()
    return log or "* No changes recorded."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    body = changelog_section(args.version)
    if body is None:
        body = f"### Changes\n\n{commits_since_last_tag()}"
    print(f"## LyCORIS {args.version}\n\n{body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
