#!/usr/bin/env python3
"""Cross-validation of the two system-layer implementations.

Runs the C++ system simulator (pimcore_sys) over the full overall grid --
four models x three batches x seven systems, throughput and energy
efficiency -- and compares every cell against the Python implementation.
"""
import os
import subprocess

from common import (MODELS, BATCHES, SYSTEMS, SYSTEM_LABELS, RESULTS,
                    run_cell, write_csv)
from proteus_sim.dram import pimcore_bridge as pc


def parse_tables(text):
    thr, eff = [], []
    target = None
    for ln in text.splitlines():
        if ln.startswith("# throughput"):
            target = thr
        elif ln.startswith("# energy"):
            target = eff
        elif ln.startswith("batch"):
            continue
        elif target is not None and ln.strip():
            target.append([float(x) for x in ln.split(",")])
    return thr, eff


def main():
    if not pc.available():
        print("cmake/g++ unavailable; skipping the C++ cross-check")
        return
    pc.build()
    binary = os.path.join(pc.BUILD_DIR, "pimcore_sys")
    configs = os.path.join(pc._ROOT, "configs")
    out = subprocess.run([binary, "--configs", configs], check=True,
                         capture_output=True, text=True).stdout
    thr_cpp, eff_cpp = parse_tables(out)

    rows = []
    worst = (0.0, "")
    n_cells = 0
    i = 0
    for m in MODELS:
        for b in BATCHES:
            for j, s in enumerate(SYSTEMS):
                r = run_cell(s, m, b)
                py_thr = r.throughput if r.alive else 0.0
                py_eff = r.tokens_per_joule if r.alive else 0.0
                cpp_thr = thr_cpp[i][j + 1]
                cpp_eff = eff_cpp[i][j + 1]
                for metric, py, cpp in (("thr", py_thr, cpp_thr),
                                        ("eff", py_eff, cpp_eff)):
                    if py == 0.0 and cpp == 0.0:
                        continue
                    dev = abs(cpp / py - 1) if py else 1.0
                    n_cells += 1
                    if dev > worst[0]:
                        worst = (dev, f"{SYSTEM_LABELS[s]}/{m}/b{b}/{metric}")
                    rows.append([SYSTEM_LABELS[s], m, b, metric,
                                 round(py, 2), round(cpp, 2),
                                 round(dev * 100, 3)])
            i += 1

    write_csv(os.path.join(RESULTS, "sys_crossvalidation.csv"),
              ["system", "model", "batch", "metric", "python", "cpp",
               "dev_pct"], rows)
    mean = sum(r[-1] for r in rows) / len(rows)
    print(f"\nC++ vs Python system layer (overall): {n_cells} cells")
    print(f"  mean |deviation| {mean:.3f}% | worst {worst[0]*100:.3f}% "
          f"({worst[1]})")
    assert worst[0] < 0.01, "cross-validation exceeded 1%"

    # breakdown / scalability tables against the Python-generated CSVs
    import csv as _csv
    def cells_of(path, skip_hash=True):
        out = []
        with open(path) as f:
            for row in _csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                try:
                    float(row[1])
                except (ValueError, IndexError):
                    continue
                out.extend(float(x) for x in row[1:] if _isnum(x))
        return out

    def _isnum(x):
        try:
            float(x); return True
        except ValueError:
            return False

    def cpp_table(name):
        return subprocess.run([binary, "--configs", configs,
                               "--table", name], check=True,
                              capture_output=True, text=True).stdout

    checks = 0
    for name, ref_files in [
            ("breakdown", ["effectiveness_breakdown.csv"]),
            ("scalability", ["scalability_device.csv",
                             "scalability_parallel.csv"])]:
        txt = cpp_table(name)
        cpp_cells = []
        for ln in txt.splitlines():
            parts = ln.split(",")
            if len(parts) < 2 or ln.startswith("#"):
                continue
            cpp_cells.extend(float(x) for x in parts[1:] if _isnum(x))
        ref_cells = []
        for rf in ref_files:
            ref_cells.extend(cells_of(os.path.join(RESULTS, rf)))
        n = min(len(cpp_cells), len(ref_cells))
        for a, b in zip(cpp_cells[:n], ref_cells[:n]):
            if b:
                assert abs(a / b - 1) < 0.02, (name, a, b)
                checks += 1
    print(f"  breakdown/scalability tables: {checks} additional cells "
          f"agree (<2%)")
    print("  PASS")


if __name__ == "__main__":
    main()
