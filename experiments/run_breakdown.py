#!/usr/bin/env python3
"""Effectiveness analysis (Fig. "PerformanceBreakdown", Sec. V-C): throughput
of the incremental Proteus variants (Base, +AS, +RD, +OF, +EC), normalized to
Proteus-Base, on Mixtral-8x7B and Llama-3.1-70B."""
import os
from common import RESULTS, BATCHES, run_cell, write_csv, geomean

MODELS = ["mixtral-8x7b", "llama3-70b"]
CHAIN = ["base", "as", "rd", "of", "ec"]
LABELS = ["Proteus-Base", "+AS", "+RD", "+OF", "+EC"]


def main():
    rows, chains = [], []
    for m in MODELS:
        for b in BATCHES:
            thr = [run_cell("proteus", m, b, variant=v).throughput for v in CHAIN]
            rows.append([b] + [round(t / thr[0], 3) for t in thr])
            chains.append((m, b, thr))
    write_csv(os.path.join(RESULTS, "effectiveness_breakdown.csv"),
              ["batch"] + LABELS, rows)

    print("\nincremental gains (geomean over models/batches):")
    for i, name in enumerate(LABELS[1:], start=1):
        # +EC is a no-op on dense models; its geomean covers MoE configs only.
        gains = [c[i] / c[i - 1] for m, _, c in chains
                 if name != "+EC" or m == "mixtral-8x7b"]
        print(f"  {name:<4}: {geomean(gains):.2f}x")
    tot = [c[-1] / c[0] for _, _, c in chains]
    print(f"  total: {geomean(tot):.2f}x (max single config "
          f"{max(tot):.2f}x)")


if __name__ == "__main__":
    main()
