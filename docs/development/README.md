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
python scripts/ci/version.py nightly         # 4.0.1.dev20260901133241
python scripts/ci/version.py write 4.1.0     # stamp pyproject.toml
```

Semantics: **major** for a change in what an existing configuration computes
or how it is invoked, **minor** for new algorithms, options and backends,
**patch** for fixes.

## Release channels

| Channel | Trigger | Version | Where it lands |
| --- | --- | --- | --- |
| Nightly | daily cron on `main`, or manual | `X.Y.(Z+1).devYYYYmmddHHMMSS` | PyPI pre-release + GitHub pre-release |
| Patch | weekly cron on `main`, or manual | `X.Y.Z+1` | tag → PyPI + GitHub release |
| Release | pushing a `vX.Y.Z` tag, or manual | whatever `pyproject.toml` says | PyPI + GitHub release |

`release.yml` is the only workflow that builds or uploads anything — nightly
and auto-patch compute a version and dispatch it — so PyPI needs exactly one
trusted publisher, and there is one build path to keep working.

The nightly gate reads the `v*` tags, which every release leaves on the commit
it was cut from. It builds only when `main` carries no release tag at its tip
**and** no nightly has been cut for the current UTC date — so a quiet day, a
re-run and an extra manual dispatch all cost nothing, and a nightly is never a
duplicate of the release before it. `workflow_dispatch` with `force: true`
overrides both checks.

A nightly is a plain PEP 440 dev release, so it installs from the index:

```bash
pip install --upgrade --pre lycoris-lora
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
| `.github/workflows/nightly.yml` | once a day: dispatches Release with a dev version, unless `main`'s tip is already tagged or today's nightly exists |
| `.github/workflows/auto-patch-release.yml` | cuts `X.Y.Z+1` when enough has landed on `main` (dry-run by default) |
| `.github/workflows/release.yml` | the only builder: stamps the version if given one, publishes to PyPI by trusted publishing, cuts the GitHub release |
