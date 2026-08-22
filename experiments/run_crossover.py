#!/usr/bin/env python3
"""Crossover robustness (Fig. 15, Sec. V-E).

Perturbs the analytical crossover threshold theta = F_PIM/BW_out around its
nominal value and measures the resulting throughput, normalized to the
nominal setting. Two extremes bracket the sweep: theta = AI_PIM (the PIM
ridge point, 2) collapses the estimate onto an xPU-centric mapping, while
theta = AI_xPU (the xPU ridge point, 312) collapses it onto a PIM-centric
mapping. The +/-10% points quantify how much a mis-estimated crossover
actually costs, which is what justifies using a coarse first-order estimate
plus runtime adaptation rather than a precise cost model.
"""
import os

from common import RESULTS, write_csv
from proteus_sim import build_system

MODELS = [("mixtral-8x7b", [32, 64]), ("llama3-70b", [64, 128])]


def theta_run(model, batch, theta):
    from proteus_sim import load_model
    from proteus_sim.workload import build_workload
    from common import CTX_IN, CTX_OUT
    sys_ = build_system("proteus")
    nominal = sys_.sched.theta
    sys_.sched.theta = theta if theta > 0 else nominal
    r = sys_.simulate(build_workload(load_model(model), batch, CTX_IN, CTX_OUT))
    return (r.throughput if r.alive else 0.0), nominal


def main():
    nominal = build_system("proteus").sched.theta
    ridge_pim = build_system("proteus").sched.ai_pim
    ridge_xpu = build_system("proteus").sched.ai_xpu
    thetas = [("AI_PIM", ridge_pim), ("0.9*theta", 0.9 * nominal),
              ("theta", nominal), ("1.1*theta", 1.1 * nominal),
              ("AI_xPU", ridge_xpu)]
    rows = []
    worst_pm10, worst_lo, worst_hi = 1.0, 1.0, 1.0
    for model, batches in MODELS:
        for b in batches:
            base, _ = theta_run(model, b, nominal)
            vals = []
            for _, th in thetas:
                t, _ = theta_run(model, b, th)
                vals.append(t / base if base else 0.0)
            rows.append([model, b] + [round(v, 4) for v in vals])
            worst_pm10 = min(worst_pm10, vals[1], vals[3])
            worst_lo = min(worst_lo, vals[0])
            worst_hi = min(worst_hi, vals[4])
    write_csv(os.path.join(RESULTS, "crossover_sensitivity.csv"),
              ["model", "batch"] + [n for n, _ in thetas], rows)

    print(f"\ncrossover threshold theta = {nominal:g} "
          f"(AI_PIM = {ridge_pim:g}, AI_xPU = {ridge_xpu:g})")
    print(f"  within +/-10% of theta : throughput stays within "
          f"{(1 - worst_pm10) * 100:.2f}% of nominal")
    print(f"  theta = AI_PIM ({ridge_pim:g})  : up to "
          f"{1 / worst_lo:.2f}x throughput reduction "
          f"(attention and shared operands forced onto the xPU)")
    print(f"  theta = AI_xPU ({ridge_xpu:g}): up to "
          f"{1 / worst_hi:.2f}x throughput reduction "
          f"(dense operators retained in a compute-bound PIM)")


if __name__ == "__main__":
    main()
