# Development

## Checks

```bash
black lycoris/ scripts/ test/          # formatting (CI runs --check)
ruff check lycoris/ scripts/ test/     # breakage-only rule set, pinned in pyproject
python scripts/ci/import_check.py      # every module imports in its own process
python scripts/ci/cpu_smoke.py         # every algorithm, forward + backward, CPU
```

The last two are what CI runs in place of a test suite: the runners have no
GPU, so the gate is that the package imports without the optional backends,
that no import cycle exists, and that every algorithm's eager path produces
finite values and gradients. The full suites live in `test/` and want a GPU:

```bash
python -m unittest discover test -v
python -m unittest test.kernels.test_autograd -v   # kernel parity, needs CUDA
```

Kernel benchmarks are described in [../kernels/benchmarks.md](../kernels/benchmarks.md).

## Versioning

`[project] version` in `pyproject.toml` is the only version string; everything
else reads it through `scripts/ci/version.py`:

```bash
python scripts/ci/version.py read            # 4.0.0
python scripts/ci/version.py bump patch      # 4.0.1  (prints, does not write)
python scripts/ci/version.py nightly         # 4.0.1.dev20260901133241+2fbff9a
python scripts/ci/version.py write 4.1.0     # stamp pyproject.toml
```

Semantics: **major** for a change in what an existing configuration computes
or how it is invoked, **minor** for new algorithms, options and backends,
**patch** for fixes.

## Release channels

| Channel | Trigger | Version | Where it lands |
| --- | --- | --- | --- |
| Nightly | daily cron on `dev`, or manual | `X.Y.(Z+1).devYYYYmmddHHMMSS+sha` | a dated GitHub pre-release, pruned to the last 7 |
| Patch | weekly cron on `main`, or manual | `X.Y.Z+1` | tag → PyPI + GitHub release |
| Release | pushing a `vX.Y.Z` tag, or manual | whatever `pyproject.toml` says | PyPI + GitHub release |

Nightlies carry a PEP 440 `+local` segment naming the commit, which PyPI
rejects by design — they are installed from the GitHub release:

```bash
pip install --upgrade --pre --find-links \
  https://github.com/KohakuBlueleaf/LyCORIS/releases/expanded_assets/nightly-YYYYMMDD lycoris_lora
```

Cutting a formal release by hand:

```bash
python scripts/ci/version.py write 4.1.0
# edit Change.md: add the "## <date> update to 4.1.0" section
git commit -am "bump to version 4.1.0"
git tag v4.1.0 && git push origin main --follow-tags
```

The `Release` workflow refuses a tag that disagrees with `pyproject.toml`,
since a PyPI filename cannot be reused once uploaded. Release notes come from
the matching `Change.md` section, falling back to the commit list.

PyPI upload uses [trusted publishing](https://docs.pypi.org/trusted-publishers/):
register this repository and `release.yml` as a publisher on PyPI, and no API
token needs to exist in the repository secrets.

## Workflows

| File | Does |
| --- | --- |
| `.github/workflows/ci.yml` | format, lint, import/cycle check, CPU smoke, wheel install check |
| `.github/workflows/nightly.yml` | dated pre-release from `dev`, skipped when `dev` has not moved |
| `.github/workflows/auto-patch-release.yml` | cuts `X.Y.Z+1` when enough has landed on `main` (dry-run by default) |
| `.github/workflows/release.yml` | builds, publishes to PyPI, creates the GitHub release |
