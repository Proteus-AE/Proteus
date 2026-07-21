#!/usr/bin/env python3
"""Continuous-batching serving dynamics (Sec. IV-D "Runtime Adaptation").

Closed-loop request-level simulation on Mixtral-8x7B: requests with
lognormal prompt/output lengths decode one token per iteration; completions
are immediately replaced, so batch composition, aggregate context, and
per-expert routing drift continuously. The experiment records how the
runtime re-derives placement each iteration: the co-execution split x*, the
number of experts mapped to each substrate by the crossover model, and the
resulting placement switches -- i.e., the adaptivity that a static mapping
cannot provide.
"""
import os

from common import RESULTS, write_csv
from proteus_sim import load_model, build_system
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
             r.mode_switches, r.completed] for r in recs]
    write_csv(os.path.join(RESULTS, f"serving_dynamics_{name}.csv"),
              ["iter", "batch", "mean_ctx", "t_iter_ms", "tokens_s",
               "tokens_per_expert", "x_split", "experts_xpu", "experts_pim",
               "placement_switches", "completed"], rows)

    warm = recs[50:]
    thr = sum(r.throughput for r in warm) / len(warm)
    sw = sum(r.mode_switches for r in warm)
    xs = [r.x_split for r in warm]
    from common import run_cell
    ref_ctx = int(sum(r.mean_ctx for r in warm) / len(warm))
    ref = run_cell("proteus", MODEL, mb, ctx=ref_ctx).throughput
    print(f"\n[{name}] steady state over {len(warm)} iterations "
          f"(batch {mb}):")
    print(f"  mean throughput      : {thr:,.0f} tokens/s "
          f"(steady-state point at same mean ctx: {ref:,.0f})")
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
    """Run the C++ serving engine (pimcore_serve) on the same scenario and
    return its steady-state mean throughput."""
    import re
    import subprocess
    from proteus_sim.dram import pimcore_bridge as pc
    pc.build()
    exe = os.path.join(pc.BUILD_DIR, "pimcore_serve")
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
                  f"(pimcore_serve, independent RNG; deviation {dev:.1f}%)")
            xrows.append([name, round(py_thr), round(cpp_thr),
                          f"{dev:.2f}%"])
    if xrows:
        write_csv(os.path.join(RESULTS, "serving_crosscheck.csv"),
                  ["scenario", "python_tokens_s", "cpp_tokens_s",
                   "deviation"], xrows)


if __name__ == "__main__":
    main()
