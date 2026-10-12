#!/usr/bin/env python3
"""Ablation analysis (Fig. 15, Sec. V-D).

Leave-one-out: each mechanism -- adaptive scheduling (AS), the reconfigurable
datapath (RD), memory-side operator fusion (OF) and expert-centric processing
(EC) -- is removed individually from full Proteus on Mixtral-8x7B and
Llama-3.1-70B at batch 16-64.

Duplex-style applies Duplex's expert co-processing policy on the same
hardware (Mixtral-8x7B): attention runs in PIM, dense projections on the xPU,
and routed experts, ranked by token count, are split between the xPU and PIM
so that their concurrent execution is balanced; PIM keeps its default direct
connectivity and no queue-aware revision is applied.

All throughputs are normalized to full Proteus."""
import os
import types

from common import RESULTS, BATCHES, CTX_IN, CTX_OUT, write_csv, geomean
from proteus_sim import build_system, load_model
from proteus_sim.system import FULL
from proteus_sim.workload import build_workload

MODELS = ["mixtral-8x7b", "llama3-70b"]
LOO = [("-AS", FULL - {"as"}), ("-RD", FULL - {"rd"}),
       ("-OF", FULL - {"of"}), ("-EC", FULL - {"ec"})]
LABELS = [k for k, _ in LOO] + ["Duplex-style"]


def _duplex_schedule(self, costs, f, counters):
    """Duplex placement: attention -> PIM, dense GEMMs -> xPU, experts split
    by token count to minimize the estimated block time."""
    q = {"xpu": 0.0, "pim": 0.0}
    by = {"xpu": 0.0, "pim": 0.0}
    modes = {"direct": 0, "broadcast": 0}
    chosen = []

    def put(c, sub):
        q[sub] += c.t_on(sub)
        by[sub] += c.bytes_on(sub)
        if sub == "pim":
            modes[c.pim_mode] += 1
        chosen.append([c, sub, 1.0])

    experts = [c for c in costs if c.name.startswith("expert")]
    for c in costs:
        if c not in experts:
            put(c, "pim" if c.kind == "attention" else "xpu")
    ranked = sorted(experts, key=lambda c: -c.intensity)
    overlap = "rd" in f
    best = None
    for k in range(len(ranked) + 1):
        tx = sum(c.t_est("xpu") for c in ranked[:k])
        tp = sum(c.t_est("pim") for c in ranked[k:])
        obj = max(tx, tp) if overlap else tx + tp
        if best is None or obj < best[0]:
            best = (obj, k)
    for i, c in enumerate(ranked):
        put(c, "xpu" if i < best[1] else "pim")
    return q, by, modes, chosen


def duplex_style_system():
    s = build_system("proteus", features=FULL)
    op_cost = s._op_cost
    s._op_cost = (lambda op, shard, pbw_d, pbw_b, pf, xbw, smallf, f,
                  mode_pref="auto":
                  op_cost(op, shard, pbw_d, pbw_b, pf, xbw, smallf, f, "direct"))
    s._schedule_block = types.MethodType(_duplex_schedule, s)
    return s


def throughput(sys_, model, batch):
    w = build_workload(load_model(model), batch, CTX_IN, CTX_OUT,
                       routing="expected", seed=7)
    r = sys_.simulate(w)
    return r.throughput if r.alive else 0.0


def main():
    rows, agg, ec_moe, dup = [], {k: [] for k, _ in LOO}, [], []
    for m in MODELS:
        label = load_model(m)["name"]
        for b in BATCHES:
            full = throughput(build_system("proteus", features=FULL), m, b)
            r = [throughput(build_system("proteus", features=v), m, b) / full
                 for _, v in LOO]
            for (k, _), x in zip(LOO, r):
                agg[k].append(x)
            d = float("nan")
            if m == "mixtral-8x7b":
                ec_moe.append(r[3])
                d = throughput(duplex_style_system(), m, b) / full
                dup.append(d)
            rows.append([label, b] + [round(x, 3) for x in r]
                        + [round(d, 3) if d == d else "nan"])
    write_csv(os.path.join(RESULTS, "effectiveness_breakdown.csv"),
              ["model", "batch"] + LABELS, rows)
    print("\nthroughput loss of each ablation (geomean over both models):")
    for k, _ in LOO:
        print(f"  {k:<4}: {1.0 / geomean(agg[k]):.2f}x")
    print(f"  -EC on Mixtral-8x7B: {1.0 / geomean(ec_moe):.2f}x")
    print(f"Proteus over Duplex-style on Mixtral-8x7B: {1.0 / geomean(dup):.2f}x")


if __name__ == "__main__":
    main()
