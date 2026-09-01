"""Shared figure drawing for the family plot scripts — line plots only.

Conventions follow KohakUwULLM's bench/core/plotting.py + kernels_plot.py:
bold = net of host dispatch with the raw wall faint behind it (the gap is the
dispatch cost), ringed markers where host share exceeds 30% and no corrected
rate is admissible, measured ceilings as crimson axhlines, accuracy beside
throughput, stable Okabe-Ito colours per arm, PNG + SVG. Reads the measure
scripts' JSON only, so a finished run redraws without a GPU.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, colourblind-safe, most-distinct first; arms keep one colour.
ARM_COLORS = {
    "eager": "#7F7F7F",
    "compile": "#0072B2",
    "triton": "#D55E00",
    "tilelang": "#009E73",
}
# Arms that agree exactly (eager/compile, and the two backends on ULP) would
# hide one another, so each arm carries its own dash and marker too.
ARM_STYLE = {
    "eager": ("-", "o"),
    "compile": ((0, (6, 2)), "s"),
    "triton": ("-", "^"),
    "tilelang": ((0, (2, 2)), "D"),
}
CEIL = "#C0392B"
RING = "#8E44AD"
# Past this share of wall, `wall - host` is not admissible as a rate.
HOST_SHARE_SUSPECT = 0.30

_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def new_figure(nrows=1, ncols=1, figsize=None):
    plt.rcParams.update(_RC)
    figsize = figsize or (7.0 * ncols, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, [ax for row in axes for ax in row]


def save_figure(fig, path):
    """PNG next to an SVG of the same name."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".svg")
    plt.close(fig)
    print(f"figure -> {path}")


def _log_scale(ax, values) -> None:
    """Log the axis only when something positive was drawn on it.

    An op with no backward leaves that panel's series entirely NaN, and
    matplotlib refuses a log axis with no positive data.
    """
    if any(v is not None and v > 0 and not math.isnan(v) for v in values):
        ax.set_yscale("log")


def _cases(rows):
    seen = []
    for r in rows:
        if r["case"] not in seen:
            seen.append(r["case"])
    return seen


def _host_share(row, phase):
    ms = row.get(f"{phase}_ms") or 0.0
    return (row.get(f"{phase}_host_ms", 0.0) / ms) if ms else float("nan")


def _net_scale(row, phase):
    """wall/net ratio, or nan where the correction is inadmissible."""
    ms = row.get(f"{phase}_ms") or 0.0
    net = ms - row.get(f"{phase}_host_ms", 0.0)
    if ms <= 0 or net <= 0 or _host_share(row, phase) > HOST_SHARE_SUSPECT:
        return float("nan")
    return ms / net


def _lines(ax, rows, cases, arms, value, phase=None, net=False, log=False):
    """One line per arm over the case axis; bold net, faint wall behind.

    ``value(row)`` returns the raw (wall-based) quantity. With ``net``, the
    bold series divides a time by ``net_ms`` or multiplies a rate by
    ``ms/net_ms`` — the same measurement, corrected, never recomputed.
    """
    xs = list(range(len(cases)))
    drawn = []
    for arm in arms:
        by_case = {r["case"]: r for r in rows if r["arm"] == arm}
        raw = [value(by_case[c]) if c in by_case else float("nan") for c in cases]
        color = ARM_COLORS.get(arm, "#000000")
        if net:
            ax.plot(xs, raw, color=color, alpha=0.25, linewidth=1.2)
            scale = [
                _net_scale(by_case[c], phase) if c in by_case else float("nan")
                for c in cases
            ]
            shown = [
                float("nan") if math.isnan(v) or math.isnan(s) else v * s
                for v, s in zip(raw, scale)
            ]
        else:
            shown = raw
        dash, mark = ARM_STYLE.get(arm, ("-", "o"))
        drawn += list(shown)
        ax.plot(
            xs,
            shown,
            linestyle=dash,
            marker=mark,
            markersize=4.5,
            color=color,
            label=arm,
        )
        if net:
            weak = [
                (x, v)
                for x, c, v in zip(xs, cases, raw)
                if c in by_case and _host_share(by_case[c], phase) > HOST_SHARE_SUSPECT
            ]
            if weak:
                ax.scatter(
                    [x for x, _ in weak],
                    [v for _, v in weak],
                    s=130,
                    facecolors="none",
                    edgecolors=RING,
                    linewidths=1.5,
                    zorder=5,
                )
    ax.set_xticks(xs)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
    if log:
        _log_scale(ax, drawn)
    ax.legend(fontsize=8)


def _pair(ax, rows, cases, arms, dev_key, wall_key, log=False):
    """Device time bold, wall faint behind it; the gap is the dispatch cost."""
    xs = list(range(len(cases)))
    drawn = []
    for arm in arms:
        by_case = {r["case"]: r for r in rows if r["arm"] == arm}
        color = ARM_COLORS.get(arm, "#000000")
        dash, mark = ARM_STYLE.get(arm, ("-", "o"))
        wall = [by_case.get(c, {}).get(wall_key, float("nan")) for c in cases]
        dev = [by_case.get(c, {}).get(dev_key, float("nan")) for c in cases]
        ax.plot(xs, wall, color=color, alpha=0.25, linewidth=1.2)
        ax.plot(
            xs,
            dev,
            linestyle=dash,
            marker=mark,
            markersize=4.5,
            color=color,
            label=arm,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)


def _ceiling(ax, value, label):
    ax.axhline(value, color=CEIL, linestyle="--", linewidth=1.2)
    ax.annotate(
        label,
        (0, value),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=8,
        color=CEIL,
    )


def draw_family(rows, meta, out_png):
    """Six line panels: latency (both phases), efficiency, speedup, VRAM, error."""
    cases = _cases(rows)
    arms = sorted(
        {r["arm"] for r in rows},
        key=lambda a: list(ARM_COLORS).index(a) if a in ARM_COLORS else 99,
    )
    fig, axes = new_figure(3, 2, figsize=(15, 13.5))
    family = rows[0]["family"]

    # Device time is the primary everywhere a rate is drawn: at these sizes the
    # wall is mostly Python dispatch, so a wall-based rate ranks launchers.
    _pair(axes[0], rows, cases, arms, "fwd_dev_ms", "fwd_ms", log=True)
    axes[0].set_ylabel("forward ms (bold = device, faint = wall)")
    axes[0].set_title("Forward latency")

    _pair(axes[1], rows, cases, arms, "fwdbwd_dev_ms", "fwdbwd_ms", log=True)
    axes[1].set_ylabel("fwd+bwd ms (bold = device, faint = wall)")
    axes[1].set_title("Forward + backward latency")

    def dev_bw(row):
        dev = row.get("fwd_dev_ms")
        ms = row.get("fwd_ms")
        pct = row.get("pct_bw")
        if not (dev and ms and pct) or math.isnan(dev) or dev <= 0:
            return float("nan")
        return pct * ms / dev

    _lines(axes[2], rows, cases, arms, dev_bw)
    _ceiling(axes[2], 100.0, f"measured DRAM ceiling {meta.get('dram_bw', 0):.0f} GB/s")
    axes[2].set_ylabel("% of measured DRAM bandwidth (device time)")
    axes[2].set_title("Bandwidth efficiency (logical bytes / device time)")
    # Above the DRAM ceiling is a cache hit, not a fast kernel: say so rather
    # than let the reader take it as bandwidth.
    if any((dev_bw(r) or 0) > 100.0 for r in rows):
        axes[2].annotate(
            f"above 100% = working set is L2-resident "
            f"({meta.get('l2_mib', 0):.0f} MiB), so these rows measure cache",
            (0.02, 0.02),
            xycoords="axes fraction",
            fontsize=8,
            color=CEIL,
        )

    # Speedup against eager on DEVICE time: at these sizes the wall is mostly
    # Python dispatch, and a wall ratio would compare launchers, not kernels.
    base = {r["case"]: r for r in rows if r["arm"] == "eager"}
    for phase, style in (("fwd", "-"), ("fwdbwd", "--")):
        for arm in arms:
            if arm == "eager":
                continue
            by_case = {r["case"]: r for r in rows if r["arm"] == arm}
            ys = []
            for c in cases:
                ref = base.get(c, {}).get(f"{phase}_dev_ms")
                got = by_case.get(c, {}).get(f"{phase}_dev_ms")
                ok = ref and got and not (math.isnan(ref) or math.isnan(got))
                ys.append(ref / got if ok else float("nan"))
            axes[3].plot(
                range(len(cases)),
                ys,
                style,
                marker="o" if phase == "fwd" else "s",
                color=ARM_COLORS.get(arm, "#000"),
                label=f"{arm} {phase}",
            )
    axes[3].axhline(1.0, color=CEIL, linestyle="--", linewidth=1.2)
    axes[3].set_xticks(range(len(cases)))
    axes[3].set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
    axes[3].set_ylabel("device-time speedup vs eager (>1 is faster)")
    axes[3].set_title("Speedup against eager — device (kernel) time")
    axes[3].legend(fontsize=8)

    _lines(
        axes[4],
        rows,
        cases,
        arms,
        lambda r: (r.get("vram_fwdbwd") or 0) / 2**20,
    )
    axes[4].set_ylabel("peak VRAM fwd+bwd (MiB)")
    axes[4].set_title("Peak memory")

    _lines(axes[5], rows, cases, arms, lambda r: r.get("ulp") or r.get("rel_err"))
    axes[5].axhline(1.0, color=CEIL, linestyle="--", linewidth=1.2)
    axes[5].set_yscale("log")
    axes[5].set_ylabel("max error (ULP, worst over grads)")
    axes[5].set_title("Precision — 1.0 ULP is as exact as the dtype allows")

    fig.suptitle(
        f"{family} — {meta.get('device', 'device not recorded')}\n"
        f"ceilings: {meta.get('dram_bw', 0):.0f} GB/s DRAM "
        f"({meta.get('bw_pattern', 'best of copy/read/triad')}), "
        f"{meta.get('mamf', 0):.0f} TFLOP/s matmul, both measured on this card\n"
        "bold = device (kernel) time, faint = wall including host dispatch; "
        "rates are scored on device time, since the wall at these sizes is "
        "mostly Python",
        fontweight="bold",
    )
    save_figure(fig, out_png)

    print(f"{'case':24s} {'arm':9s} {'rel_err':>10s} {'host%':>7s}")
    for case in cases:
        for arm in arms:
            row = next((r for r in rows if r["case"] == case and r["arm"] == arm), None)
            if row:
                share = _host_share(row, "fwd")
                print(
                    f"{case:24s} {arm:9s} {row.get('rel_err', float('nan')):10.2e} "
                    f"{100 * share:6.1f}%"
                )
