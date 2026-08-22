#!/usr/bin/env python3
"""Overall results (Fig. 11, Sec. V-B): throughput and energy efficiency
of Proteus and six baselines across four models and batch sizes 16-64.
Outputs absolute and CXL-PNM-normalized tables for both metrics."""
import os
from common import (RESULTS, MODELS, BATCHES, SYSTEMS, SYSTEM_LABELS,
                    run_cell, write_csv, geomean)
from proteus_sim import load_model


def main():
    thr_abs, thr_norm, eff_abs, eff_norm = [], [], [], []
    speedup_vs = {s: [] for s in SYSTEMS if s != "proteus"}
    effgain_vs = {s: [] for s in SYSTEMS if s != "proteus"}
    for m in MODELS:
        label = load_model(m)["name"]
        for b in BATCHES:
            row = {s: run_cell(s, m, b) for s in SYSTEMS}
            thr = {s: (r.throughput if r.alive else 0.0) for s, r in row.items()}
            eff = {s: (r.tokens_per_joule if r.alive else 0.0) for s, r in row.items()}
            base_t, base_e = thr["cxl-pnm"], eff["cxl-pnm"]
            thr_abs.append([label, b] + [round(thr[s]) for s in SYSTEMS])
            eff_abs.append([label, b] + [round(eff[s], 2) for s in SYSTEMS])
            thr_norm.append([label, b]
                            + [round(thr[s] / base_t, 4) for s in SYSTEMS])
            eff_norm.append([label, b]
                            + [round(eff[s] / base_e, 4) for s in SYSTEMS])
            for s in speedup_vs:
                if thr[s] > 0:
                    speedup_vs[s].append(thr["proteus"] / thr[s])
                if eff[s] > 0:
                    effgain_vs[s].append(eff["proteus"] / eff[s])

    hdr = ["model", "batch"] + [SYSTEM_LABELS[s] for s in SYSTEMS]
    write_csv(os.path.join(RESULTS, "throughput_absolute.csv"), hdr, thr_abs)
    write_csv(os.path.join(RESULTS, "throughput_normalized.csv"), hdr, thr_norm)
    write_csv(os.path.join(RESULTS, "energyeff_absolute.csv"), hdr, eff_abs)
    write_csv(os.path.join(RESULTS, "energyeff_normalized.csv"), hdr, eff_norm)

    print("\ngeomean speedup of Proteus (throughput | energy eff., "
          "computed over mutually runnable configs):")
    for s in speedup_vs:
        print(f"  vs {SYSTEM_LABELS[s]:<9}: {geomean(speedup_vs[s]):.2f}x | "
              f"{geomean(effgain_vs[s]):.2f}x")


if __name__ == "__main__":
    main()
