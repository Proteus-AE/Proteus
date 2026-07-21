#!/usr/bin/env python3
"""Sensitivity analysis (Fig. "sen", Sec. V-D): Mixtral-8x7B throughput
(a) across sustained context lengths at batch 32 and (b) across batch sizes,
normalized to CXL-PNM."""
import os
from common import RESULTS, run_cell, write_csv

SYSTEMS = ["dgx-a100", "cxl-pnm", "cent", "pimphony", "proteus"]
LABELS = ["DGX-A100", "CXL-PNM", "CENT", "PIMphony", "Proteus"]
CTXS = [1024, 4096, 8192, 32768, 65536, 131072]
BATCHES = [1, 8, 16, 32, 64, 128]
MODEL = "mixtral-8x7b"


def norm_rows(rows_abs):
    out = []
    for key, vals in rows_abs:
        base = vals["cxl-pnm"]
        out.append([key] + [round(v / base, 4) if base else 0 for v in vals.values()])
    return out


def main():
    # (a) context length sweep at batch 32 (sustained context = sweep point)
    rows = []
    for ctx in CTXS:
        vals = {s: (lambda r: r.throughput if r.alive else 0.0)(
            run_cell(s, MODEL, 32, ctx=ctx)) for s in SYSTEMS}
        rows.append((f"{ctx//1024}K", vals))
    write_csv(os.path.join(RESULTS, "sensitivity_length_absolute.csv"),
              ["ctx"] + LABELS, [[k] + [round(v) for v in d.values()] for k, d in rows])
    write_csv(os.path.join(RESULTS, "sensitivity_length.csv"),
              ["ctx"] + LABELS, norm_rows(rows))

    # (b) batch sweep at 2K/6K context
    rows = []
    for b in BATCHES:
        vals = {s: (lambda r: r.throughput if r.alive else 0.0)(
            run_cell(s, MODEL, b)) for s in SYSTEMS}
        rows.append((b, vals))
    write_csv(os.path.join(RESULTS, "sensitivity_batch_absolute.csv"),
              ["batch"] + LABELS, [[k] + [round(v) for v in d.values()] for k, d in rows])
    write_csv(os.path.join(RESULTS, "sensitivity_batch.csv"),
              ["batch"] + LABELS, norm_rows(rows))


if __name__ == "__main__":
    main()
