#!/usr/bin/env python3
"""Effectiveness analysis (Fig. 14, Sec. V-D): throughput
of the incremental Proteus variants (Base, +AS, +RD, +OF, +EC), normalized to
Proteus-Base, on Mixtral-8x7B and Llama-3.1-70B."""
import os
from common import RESULTS, BATCHES, run_cell, write_csv, geomean
from proteus_sim import load_model

MODELS = ["mixtral-8x7b", "llama3-70b"]
CHAIN = ["base", "as", "rd", "of", "ec"]
LABELS = ["Proteus-Base", "+AS", "+RD", "+OF", "+EC"]


def main():
    rows, chains = [], []
    for m in MODELS:
        label = load_model(m)["name"]
        for b in BATCHES:
            thr = [run_cell("proteus", m, b, variant=v).throughput for v in CHAIN]
            rows.append([label, b] + [round(t / thr[0], 3) for t in thr])
            chains.append((m, b, thr))
    write_csv(os.path.join(RESULTS, "effectiveness_breakdown.csv"),
              ["model", "batch"] + LABELS, rows)

    print("\nincremental gain of each mechanism "
          "(geomean over every model and batch):")
    for i, name in enumerate(LABELS[1:], start=1):
        gains = [c[i] / c[i - 1] for _, _, c in chains]
        moe = [c[i] / c[i - 1] for m, _, c in chains if m == "mixtral-8x7b"]
        extra = f"   (MoE only: {geomean(moe):.2f}x)" \
            if name == "+EC" else ""
        print(f"  {name:<4}: {geomean(gains):.2f}x{extra}")
    tot = [c[-1] / c[0] for _, _, c in chains]
    print(f"  Proteus-Base -> full: {geomean(tot):.2f}x "
          f"(max single configuration {max(tot):.2f}x)")


if __name__ == "__main__":
    main()
