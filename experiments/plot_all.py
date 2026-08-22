#!/usr/bin/env python3
"""Render the evaluation figures from the CSVs in results/.

Colors: Okabe-Ito colorblind-safe palette with a fixed system->color mapping
used consistently across every figure.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS  # noqa: E402

FIGS = os.path.join(RESULTS, "figures")

COLORS = {   # fixed entity -> color (Okabe-Ito)
    "DGX-A100": "#999999", "CXL-PNM": "#E69F00", "CENT": "#56B4E9",
    "NeuPIMs": "#009E73", "PAPI": "#F0E442", "PIMphony": "#D55E00",
    "Proteus": "#0072B2",
    "Proteus-Base": "#CCCCCC", "+AS": "#56B4E9", "+RD": "#009E73",
    "+OF": "#E69F00", "+EC": "#0072B2",
}
plt.rcParams.update({"font.size": 8, "axes.grid": True,
                     "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "figure.dpi": 150})


def read(name):
    with open(os.path.join(RESULTS, name)) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def model_batch_labels(rows):
    """Tick labels of a (model, batch, ...) table: the batch of every bar
    group, with the model named once over the middle group it spans."""
    labels = []
    for i, r in enumerate(rows):
        same = [j for j, q in enumerate(rows) if q[0] == r[0]]
        labels.append(f"{r[0]}\nb={r[1]}" if i == same[len(same) // 2]
                      else f"b={r[1]}")
    return labels


def grouped_bars(ax, header, rows, group_labels, log=False):
    systems = header[2:]
    n = len(systems)
    width = 0.8 / n
    for j, s in enumerate(systems):
        xs, ys = [], []
        for i, r in enumerate(rows):
            v = float(r[j + 2])
            xs.append(i + (j - n / 2 + 0.5) * width)
            ys.append(v)
        ax.bar(xs, ys, width * 0.92, label=s, color=COLORS.get(s, "#333"),
               edgecolor="white", linewidth=0.3)
        for x, y in zip(xs, ys):
            if y == 0:
                ax.text(x, ax.get_ylim()[0] if not log else 0.55, "OoM",
                        rotation=90, ha="center", va="bottom", fontsize=5.5,
                        color="#888888")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(group_labels, fontsize=7)
    ax.grid(axis="x", visible=False)


def fig_overall():
    for metric, title, fname in [
            ("throughput_normalized.csv",
             "Decode throughput normalized to CXL-PNM", "overall_throughput"),
            ("energyeff_normalized.csv",
             "Energy efficiency (tokens/J) normalized to CXL-PNM", "overall_energyeff")]:
        header, rows = read(metric)
        fig, ax = plt.subplots(figsize=(9, 2.4))
        grouped_bars(ax, header, rows, model_batch_labels(rows), log=True)
        ax.set_ylabel("norm. to CXL-PNM (log)")
        ax.set_title(title, fontsize=9)
        ax.legend(ncol=7, fontsize=6.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"{fname}.png"), bbox_inches="tight")
        plt.close(fig)


def fig_breakdown():
    header, rows = read("effectiveness_breakdown.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    grouped_bars(ax, header, rows, model_batch_labels(rows))
    ax.set_ylabel("throughput norm.\nto Proteus-Base")
    ax.set_title("Incremental Proteus variants (Sec. V-D)", fontsize=9)
    ax.legend(ncol=5, fontsize=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "effectiveness_breakdown.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_lines(name, xlabel, fname, title):
    header, rows = read(name)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    xs = [r[0] for r in rows]
    for j, s in enumerate(header[1:], start=1):
        ys = [float(r[j]) for r in rows]
        ax.plot(range(len(xs)), [y if y > 0 else None for y in ys],
                marker="o", ms=3.5, lw=1.4, label=s, color=COLORS.get(s, "#333"))
        for i, y in enumerate(ys):
            if y == 0:
                ax.annotate("OoM", (i, 0.1), fontsize=5.5, color=COLORS.get(s),
                            ha="center")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("norm. to CXL-PNM", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname), bbox_inches="tight")
    plt.close(fig)


def fig_scalability():
    header, rows = read("scalability_device.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.0))
    ax = axes[0]
    xs = [r[0] for r in rows]
    ys = [float(r[2]) for r in rows]
    ax.bar(range(len(xs)), ys, 0.55, color=COLORS["Proteus"])
    for i, y in enumerate(ys):
        ax.text(i, y, f"{y:.2f}x", ha="center", va="bottom", fontsize=6.5)
    ax.plot(range(len(xs)), [float(x) for x in xs], ls="--", lw=1,
            color="#999999", label="linear")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_xlabel("devices")
    ax.set_ylabel("norm. to 1 device")
    ax.set_title("Device scaling (Llama-3.1-70B)", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(axis="x", visible=False)

    header, rows = read("scalability_parallel.csv")
    ax = axes[1]
    xs = [r[0] for r in rows]
    ys = [float(r[2]) for r in rows]
    ax.bar(range(len(xs)), ys, 0.55, color=COLORS["Proteus"])
    for i, y in enumerate(ys):
        ax.text(i, y, f"{y:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=7)
    ax.set_ylabel("norm. to PP8")
    ax.set_title("[PP,DP] at total batch 32", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "scalability.png"), bbox_inches="tight")
    plt.close(fig)


def fig_serving():
    path = os.path.join(RESULTS, "serving_dynamics_b32.csv")
    if not os.path.exists(path):
        return
    header, rows = read("serving_dynamics_b32.csv")
    it = [int(r[0]) for r in rows]
    thr = [float(r[4]) for r in rows]
    ctx = [float(r[2]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 2.8), sharex=True)
    axes[0].plot(it, thr, lw=1.0, color=COLORS["Proteus"])
    axes[0].set_ylabel("tokens/s", fontsize=7.5)
    axes[0].set_title("Continuous-batching serving dynamics "
                      "(Mixtral-8x7B, closed loop b=32)", fontsize=8.5)
    axes[1].plot(it, ctx, lw=1.0, color=COLORS["CENT"])
    axes[1].set_ylabel("mean context", fontsize=7.5)
    axes[1].set_xlabel("decode iteration", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "serving_dynamics.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_crossover():
    """Fig. 15: throughput under a perturbed crossover threshold."""
    path = os.path.join(RESULTS, "crossover_sensitivity.csv")
    if not os.path.exists(path):
        return
    header, rows = read("crossover_sensitivity.csv")
    labels = header[2:]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    n = len(rows)
    width = 0.16
    for j, lab in enumerate(labels):
        vals = [float(r[2 + j]) for r in rows]
        ax.bar([i + (j - 2) * width for i in range(n)], vals, width,
               label=lab, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{r[0].split('-')[0]}\nb={r[1]}" for r in rows],
                       fontsize=7)
    ax.axhline(1.0, color="grey", lw=0.6, ls="--")
    ax.set_ylabel("normalized throughput", fontsize=8)
    ax.set_title("Throughput under a perturbed crossover threshold "
                 "(normalized to theta = 32)", fontsize=8.5)
    ax.legend(fontsize=6.5, ncol=5, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "crossover_sensitivity.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_slo():
    """Fig. 13: per-token latency and SLO attainment vs offered load."""
    path = os.path.join(RESULTS, "serving_slo_sweep.csv")
    if not os.path.exists(path):
        return
    header, rows = read("serving_slo_sweep.csv")
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(
            (float(r[3]) / 1e3, float(r[4]), float(r[6]) * 100))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for name, pts in by.items():
        pts = sorted(pts)
        x = [p[0] for p in pts]
        c = COLORS.get(name, None)
        axes[0].plot(x, [p[1] for p in pts], marker="o", ms=2.5, lw=1.0,
                     label=name, color=c)
        axes[1].plot(x, [p[2] for p in pts], marker="o", ms=2.5, lw=1.0,
                     label=name, color=c)
    axes[0].axhline(30.0, color="black", lw=0.8, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("avg per-token latency (ms)", fontsize=8)
    axes[1].axhline(90.0, color="black", lw=0.8, ls="--")
    axes[1].set_ylabel("30 ms SLO attainment (%)", fontsize=8)
    for a in axes:
        a.set_xlabel("achieved load (K tokens/s)", fontsize=8)
        a.tick_params(labelsize=7)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Serving under dynamic load (Mixtral-8x7B, 30-min trace)",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "serving_slo.png"), bbox_inches="tight")
    plt.close(fig)


def fig_area():
    """Fig. 12: per-bank PE area breakdown."""
    path = os.path.join(RESULTS, "area_pe_breakdown.csv")
    if not os.path.exists(path):
        return
    header, rows = read("area_pe_breakdown.csv")
    rows = sorted(rows, key=lambda r: -float(r[3]))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.pie([float(r[3]) for r in rows],
           labels=[f"{r[0]}\n{float(r[3]):.2f}%" for r in rows],
           textprops={"fontsize": 6.5}, startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 0.5})
    total = sum(float(r[2]) for r in rows)
    ax.set_title(f"Per-bank PE area: {total:.3f} mm$^2$", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "area_breakdown.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig_overall()
    fig_breakdown()
    fig_lines("sensitivity_length.csv", "sustained context length",
              "sensitivity_length.png", "Context-length sweep (Mixtral, b=32)")
    fig_lines("sensitivity_batch.csv", "batch size",
              "sensitivity_batch.png", "Batch-size sweep (Mixtral)")
    fig_scalability()
    fig_crossover()
    fig_serving()
    fig_slo()
    fig_area()
    print(f"figures written to {FIGS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
