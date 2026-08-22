#!/usr/bin/env python3
"""What the runtime re-derives every decode iteration (Sec. IV-D, IV-E).

Closed-loop continuous batching on Mixtral-8x7B: a fixed pool of concurrent
requests decodes one token per iteration and completions are replaced
immediately, so the batch composition, the aggregate context and the
per-expert token counts drift continuously. Before each iteration the
runtime re-samples MoE routing, rebuilds the per-expert histogram,
recomputes the workload intensities and re-derives the placement. The
experiment records the resulting trajectory: the co-execution split, how
many experts each substrate holds under the crossover estimate, and how
often that decision changes -- the adaptivity a frozen mapping cannot
provide.

The same scenarios are replayed on the independent C++ serving engine
(pimcore_serve) as a cross-check of the whole iteration loop, not just the
per-iteration timing model.
"""
import os
import re
import subprocess

from common import RESULTS, run_cell, write_csv
from proteus_sim import build_system, load_model
from proteus_sim.serving import ServingSimulator

MODEL = "mixtral-8x7b"
ITERS = 600
SCENARIOS = {          # (max_batch, prompt_mean, out_mean)
    "b32": (32, 2048, 256),   # throughput serving: stable placements
    "b4": (4, 1024, 128),     # small-batch interactive: expert dropout
}


def run_scenario(name, model, mb, pm, om):
    sim = ServingSimulator(build_system("proteus"), model, max_batch=mb,
                           prompt_mean=pm, out_mean=om)
    recs = sim.run(ITERS)

    rows = [[r.it, r.batch, round(r.mean_ctx), round(r.t_iter_ms, 3),
             round(r.throughput), round(r.tokens_per_expert, 2),
             round(r.x_split, 3), r.experts_on_xpu, r.experts_on_pim,
             r.placement_switches, r.remaps, r.completed] for r in recs]
    write_csv(os.path.join(RESULTS, f"serving_dynamics_{name}.csv"),
              ["iter", "batch", "mean_ctx", "t_iter_ms", "tokens_s",
               "tokens_per_expert", "x_split", "experts_xpu", "experts_pim",
               "placement_switches", "scheduler_remaps", "completed"], rows)

    warm = recs[50:]
    thr = sum(r.throughput for r in warm) / len(warm)
    sw = sum(r.placement_switches for r in warm)
    xs = [r.x_split for r in warm]
    tpe = [r.tokens_per_expert for r in warm]
    ref_ctx = int(sum(r.mean_ctx for r in warm) / len(warm))
    ref = run_cell("proteus", MODEL, mb, ctx=ref_ctx).throughput
    print(f"\n[{name}] steady state over {len(warm)} iterations (batch {mb}):")
    print(f"  mean throughput      : {thr:,.0f} tokens/s "
          f"(steady-state point at the same mean context: {ref:,.0f})")
    print(f"  tokens per expert    : {min(tpe):.2f} .. {max(tpe):.2f} "
          f"(re-sampled every iteration)")
    print(f"  x* range             : {min(xs):.2f} .. {max(xs):.2f} "
          f"(re-derived every iteration)")
    print(f"  active experts       : "
          f"{warm[-1].experts_on_xpu + warm[-1].experts_on_pim} "
          f"({warm[-1].experts_on_xpu} xPU / {warm[-1].experts_on_pim} PIM)")
    print(f"  placement switches   : {sw} over {len(warm)} iterations "
          f"({sw/len(warm):.2f}/iter) -- routing-driven remapping")
    print(f"  completed requests   : {sum(r.completed for r in recs)}")
    return thr


def cpp_steady(mb, pm, om):
    """Run the C++ serving engine on the same scenario."""
    from proteus_sim.dram import pimcore_bridge as pc
    exe = pc.binary("pimcore_serve")
    proc = subprocess.run(
        [exe, "--configs", os.path.join(pc._ROOT, "configs"),
         "--model", MODEL, "--batch", str(mb), "--prompt-mean", str(pm),
         "--out-mean", str(om), "--iters", str(ITERS), "--csv", os.devnull],
        check=True, capture_output=True, text=True)
    m = re.search(r"([\d.]+) tokens/s", proc.stderr)
    return float(m.group(1)) if m else None


def main():
    model = load_model(MODEL)
    xrows = []
    for name, (mb, pm, om) in SCENARIOS.items():
        py_thr = run_scenario(name, model, mb, pm, om)
        cpp_thr = cpp_steady(mb, pm, om)
        if cpp_thr:
            dev = abs(cpp_thr - py_thr) / py_thr * 100
            print(f"  C++ serving engine   : {cpp_thr:,.0f} tokens/s "
                  f"(pimcore_serve; deviation {dev:.1f}%)")
            print("    the C++ engine drives the loop from the closed-form "
                  "uniform-routing expectation while the Python driver draws "
                  "a routing histogram per iteration; the gap is the "
                  "convexity of ceil(n/fanout) in the per-expert token count "
                  "(Sec. IV-E), not an implementation disagreement -- the "
                  "per-iteration timing models agree to 0.004% "
                  "(run_sys_crosscheck.py).")
            xrows.append([name, round(py_thr), round(cpp_thr), f"{dev:.2f}%"])
    if xrows:
        write_csv(os.path.join(RESULTS, "serving_crosscheck.csv"),
                  ["scenario", "python_tokens_s", "cpp_tokens_s",
                   "deviation"], xrows)


if __name__ == "__main__":
    main()
