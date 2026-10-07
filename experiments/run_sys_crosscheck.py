#!/usr/bin/env python3
"""Cross-validation of the two system-layer implementations.

Runs the C++ system simulator (pimcore_sys) over the full overall grid --
four models x three batches x seven systems, throughput and energy
efficiency -- and compares every cell against the Python implementation.
The secondary tables (variant breakdown, device and [PP, DP] scalability,
crossover perturbation) are recomputed here from the Python engine rather
than read from `results/`, so the check never depends on a previous run.
"""
import os
import subprocess

from common import (MODELS, BATCHES, SYSTEMS, SYSTEM_LABELS, RESULTS,
                    CONFIG_DIR, run_cell, write_csv)
from proteus_sim import build_system
from proteus_sim.dram import pimcore_bridge as pc

TOL_PRIMARY = 0.01        # per-cell tolerance on the overall grid
TOL_SECONDARY = 0.02      # per-cell tolerance on the derived tables


def _isnum(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def cpp_table(name):
    """Run one table of the C++ system simulator."""
    return subprocess.run([pc.binary("pimcore_sys"), "--configs", CONFIG_DIR,
                           "--table", name], check=True,
                          capture_output=True, text=True).stdout


def table_cols(text):
    """-> {"<table index>.<column>": [values]} for every numeric column.

    A stream may hold several tables back to back; a row with no numeric
    cell after the first starts a new one."""
    import csv as _csv
    rows = [r for r in _csv.reader(text.splitlines())
            if r and not r[0].startswith("#")]
    out, hdr, body, idx = {}, None, [], 0

    def flush():
        nonlocal hdr, body, idx
        if hdr and body:
            for i, nm in enumerate(hdr):
                col = [float(r[i]) for r in body
                       if i < len(r) and _isnum(r[i])]
                if len(col) == len(body):
                    out[f"{idx}.{nm}"] = col
            idx += 1
        hdr, body = None, []

    for r in rows:
        if not any(_isnum(x) for x in r[1:]):
            flush()
            hdr = r
        elif hdr:
            body.append(r)
    flush()
    return out


def breakdown_cols():
    """Python side of the C++ `--table breakdown` output."""
    from run_breakdown import MODELS as BD_MODELS, CHAIN, LABELS
    cols = {"0." + n: [] for n in ["batch"] + LABELS}
    for m in BD_MODELS:
        for b in BATCHES:
            thr = [run_cell("proteus", m, b, variant=v).throughput
                   for v in CHAIN]
            cols["0.batch"].append(float(b))
            for name, t in zip(LABELS, thr):
                cols["0." + name].append(round(t / thr[0], 3))
    return cols


def scalability_cols():
    """Python side of the C++ `--table scalability` output."""
    from run_scalability import (MODEL, DEVICES, BATCH_PER_GROUP, GROUP,
                                 PP_DP, CTX, max_batch)
    cols = {"0.devices": [], "0.groups": [], "0.layers_per_stage": [],
            "0.tokens_s": [], "0.normalized": [], "1.batch": [],
            "1.tokens_s": [], "1.normalized": []}
    base = None
    for d in DEVICES:
        r = run_cell("proteus", MODEL, BATCH_PER_GROUP, ctx=CTX, devices=d)
        thr = r.throughput if r.alive else 0.0
        base = base or thr
        c = r.counters if r.alive else {}
        cols["0.devices"].append(float(d))
        cols["0.groups"].append(float(d // GROUP))
        cols["0.layers_per_stage"].append(float(c.get("layers_per_stage", 0)))
        cols["0.tokens_s"].append(round(thr, 1))
        cols["0.normalized"].append(round(thr / base, 4) if base else 0.0)
    base = None
    for pp, dp in PP_DP:
        b = max_batch(MODEL, CTX, DEVICES[-1], dp)
        r = run_cell("proteus", MODEL, b, ctx=CTX, devices=DEVICES[-1], dp=dp)
        thr = r.throughput if r.alive else 0.0
        base = base or thr
        cols["1.batch"].append(float(b))
        cols["1.tokens_s"].append(round(thr, 1))
        cols["1.normalized"].append(round(thr / base, 4) if base else 0.0)
    return cols


def crossover_cols():
    """Python side of the C++ `--table crossover` output."""
    from run_crossover import MODELS as CO_MODELS, theta_run
    sched = build_system("proteus").sched
    thetas = [("AI_PIM", sched.ai_pim), ("0.9*theta", 0.9 * sched.theta),
              ("theta", sched.theta), ("1.1*theta", 1.1 * sched.theta),
              ("AI_xPU", sched.ai_xpu)]
    cols = {"0.batch": []}
    cols.update({"0." + n: [] for n, _ in thetas})
    for model, batches in CO_MODELS:
        for b in batches:
            base, _ = theta_run(model, b, sched.theta)
            cols["0.batch"].append(float(b))
            for n, th in thetas:
                t, _ = theta_run(model, b, th)
                cols["0." + n].append(round(t / base, 4) if base else 0.0)
    return cols


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
    out = subprocess.run([pc.binary("pimcore_sys"), "--configs", CONFIG_DIR],
                         check=True, capture_output=True, text=True).stdout
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
    if worst[0] >= TOL_PRIMARY:
        raise SystemExit(f"cross-validation exceeded {TOL_PRIMARY:.0%} "
                         f"at {worst[1]}")

    # Secondary tables: the same quantities recomputed on both sides. Each
    # entry is (name, C++ table, Python producer) and every column of the
    # C++ output must find a Python counterpart.
    checks, missing = 0, []
    for name, cpp_cols, py_cols in [
            ("breakdown", table_cols(cpp_table("breakdown")),
             breakdown_cols()),
            ("scalability", table_cols(cpp_table("scalability")),
             scalability_cols()),
            ("crossover", table_cols(cpp_table("crossover")),
             crossover_cols())]:
        for col, vals in cpp_cols.items():
            ref = py_cols.get(col)
            if ref is None or len(ref) != len(vals):
                missing.append(f"{name}.{col}")
                continue
            for a, b in zip(vals, ref):
                if not b:
                    continue
                if abs(a / b - 1) >= TOL_SECONDARY:
                    raise SystemExit(
                        f"{name}.{col}: C++ {a:g} vs Python {b:g} "
                        f"({abs(a / b - 1) * 100:.2f}% > "
                        f"{TOL_SECONDARY:.0%})")
                checks += 1
    print(f"  secondary tables (breakdown / scalability / crossover): "
          f"{checks} cells agree (<{TOL_SECONDARY:.0%})")
    if missing:
        raise SystemExit("no Python counterpart for: " + ", ".join(missing))
    print("  PASS")


if __name__ == "__main__":
    main()
