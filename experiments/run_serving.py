#!/usr/bin/env python3
"""Runtime adaptation under dynamic serving loads (Fig. 13, Sec. V-C).

Replays a 30-minute production-style LLM serving trace on Mixtral-8x7B with
continuous batching and up to 64 concurrent requests. Every request reserves
peak KV capacity for its full input plus output length before admission and
waits in FIFO order until that capacity frees; average per-token latency is
completion time minus *arrival* time divided by output length, so queueing
delay is charged to every token. Sweeping the offered load by scaling
inter-arrival times traces each system's latency curve and locates the
largest load at which it still attains the 30 ms per-token SLO for 90% of
requests.

Besides the six baselines the sweep includes **Proteus-Static**, which keeps
the full Proteus hardware -- reconfigurable datapath, memory-side fusion, the
lot -- and differs only in freezing the operator mapping at deployment time
instead of re-deriving it every decode iteration.
"""
import os

from common import RESULTS, write_csv
from proteus_sim import build_system, load_model
from proteus_sim.serving import SloServingSimulator, knee_load
from proteus_sim.workload import build_workload
from trace_gen.gen_requests import read_trace, summarize

MODEL = "mixtral-8x7b"
TRACE = os.path.join(os.path.dirname(__file__), "..", "request_traces",
                     "azure30min.txt")
MAX_CONCURRENT = 64
SLO_MS = 30.0
ATTAINMENT = 0.90
SCALES = [0.14, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50, 1.75,
          2.00, 2.25, 2.50, 3.00]
SYSTEMS = ["dgx-a100", "cxl-pnm", "cent", "neupims", "papi", "pimphony",
           "proteus-static", "proteus"]


def make_system(name, model):
    """Build a system; 'proteus-static' freezes the deployment-time mapping."""
    if name != "proteus-static":
        return build_system(name), None
    sys_ = build_system("proteus")
    ref = sys_.cfg["static_reference"]
    w = build_workload(model, int(ref["concurrency"]), 0, 0,
                       ctx_override=int(ref["context"]))
    return sys_, sys_.deployment_plan(w)


def main():
    rows = read_trace(TRACE)
    print("request trace: " + summarize(rows))
    model = load_model(MODEL)

    table, summary = [], []
    for name in SYSTEMS:
        sys_, plan = make_system(name, model)
        label = "Proteus-Static" if name == "proteus-static" \
            else sys_.cfg["name"]
        floor, curve = None, []
        for sc in SCALES:
            rep = SloServingSimulator(sys_, model, rows,
                                      max_concurrent=MAX_CONCURRENT,
                                      slo_ms=SLO_MS, load_scale=sc,
                                      frozen_plan=plan).run()
            r = rep.row()
            r[0] = label
            r.insert(1, sc)
            table.append(r)
            curve.append(rep)
            if rep.completed:
                floor = rep.mean_per_token_ms if floor is None \
                    else min(floor, rep.mean_per_token_ms)
        sustainable = knee_load(curve, ATTAINMENT)
        summary.append((label, floor or 0.0, sustainable))
        print(f"  {label:<15} latency floor {floor or 0:7.2f} ms   "
              f"sustainable @ {ATTAINMENT:.0%} of {SLO_MS:g} ms SLO: "
              f"{sustainable:8,.0f} tokens/s")

    write_csv(os.path.join(RESULTS, "serving_slo_sweep.csv"),
              ["system", "load_scale", "offered_tokens_s",
               "achieved_tokens_s", "mean_per_token_ms", "p99_per_token_ms",
               "slo_attainment", "mean_batch", "mean_queue_s", "completed"],
              table)
    write_csv(os.path.join(RESULTS, "serving_summary.csv"),
              ["system", "latency_floor_ms", "sustainable_tokens_s"],
              [[a, round(b, 3), round(c)] for a, b, c in summary])

    d = {a: (b, c) for a, b, c in summary}
    p_lat, p_load = d["Proteus"]
    print()
    for other in ("Proteus-Static", "PIMphony", "DGX-A100", "CENT", "CXL-PNM"):
        if other not in d:
            continue
        lat, load = d[other]
        print(f"  Proteus vs {other:<15}: sustainable load "
              f"{p_load / load if load else float('inf'):5.2f}x, "
              f"unsaturated latency {lat / p_lat if p_lat else 0:5.2f}x lower")


if __name__ == "__main__":
    main()
