#!/usr/bin/env python3
"""Multi-device scalability (Fig. 17 and Fig. 18, Sec. V-F).

(a) Llama-3.1-405B scaled from 8 to 64 devices. Eight devices inside one CXL
    switch domain form a tensor-parallel group; groups are pipeline stages
    holding whole layers, so 126 layers over G groups leave the trailing
    groups one layer short and the fullest group sets the stage time. Each
    group serves its own in-flight micro-batch of 16 requests, and only one
    b x d_model activation crosses each group boundary.

(b) [PP, DP] combinations on 64 devices at a 32K context. Data parallelism
    replicates the model, so both strategies fetch the same weight volume per
    token at equal batch, but every replica holds its own copy and the KV
    budget -- and with it the sustainable concurrency -- shrinks.
"""
import os

from common import RESULTS, run_cell, write_csv

MODEL = "llama3-405b"
DEVICES = [8, 16, 32, 64]
BATCH_PER_GROUP = 16
GROUP = 8
PP_DP = [(8, 1), (4, 2), (2, 4), (1, 8)]
CTX = 32768


def main():
    rows = []
    base = None
    print(f"\n(a) {MODEL}: pipeline groups of {GROUP} devices, "
          f"{BATCH_PER_GROUP} requests per group")
    for d in DEVICES:
        groups = d // GROUP
        # One in-flight micro-batch of BATCH_PER_GROUP requests per group.
        r = run_cell("proteus", MODEL, BATCH_PER_GROUP, ctx=CTX, devices=d)
        thr = r.throughput if r.alive else 0.0
        base = base or thr
        c = r.counters if r.alive else {}
        rows.append([d, groups, c.get("layers_per_stage", 0),
                     round(c.get("layer_imbalance", 0.0), 4), round(thr),
                     round(thr / base, 3) if base else 0,
                     round(c.get("stage_transfer_ms", 0.0), 4)])
        print(f"  {d:>3} devices ({groups} groups x {GROUP}): "
              f"{c.get('layers_per_stage',0):>3} layers/stage, "
              f"{thr:>9,.0f} tokens/s  ({thr/base:.2f}x)  "
              f"boundary transfer {c.get('stage_transfer_ms',0)*1e3:.1f} us")
    write_csv(os.path.join(RESULTS, "scalability_device.csv"),
              ["devices", "groups", "layers_per_stage", "layer_imbalance",
               "tokens_s", "normalized", "stage_transfer_ms"], rows)

    print(f"\n(b) [PP, DP] on {DEVICES[-1]} devices, {CTX//1024}K context, "
          f"each at the largest batch its capacity permits")
    print("      (a replica must hold one micro-batch per pipeline group)")
    rows = []
    base = None
    for pp, dp in PP_DP:
        b = max_batch(MODEL, CTX, DEVICES[-1], dp)
        r = run_cell("proteus", MODEL, b, ctx=CTX, devices=DEVICES[-1], dp=dp) \
            if b else None
        thr = r.throughput if (r and r.alive) else 0.0
        base = base or thr
        rows.append([f"PP{pp}xDP{dp}", b, round(thr),
                     round(thr / base, 3) if base else 0])
        print(f"  [PP={pp}, DP={dp}]: batch {b:>5}  {thr:>9,.0f} tokens/s  "
              f"({thr/base:.2f}x)")
    write_csv(os.path.join(RESULTS, "scalability_parallel.csv"),
              ["config", "batch", "tokens_s", "normalized"], rows)


def max_batch(model, ctx, devices, dp, hi=2048):
    """Best batch the deployment can actually run.

    Capacity bounds the batch twice over: a micro-batch must fit, and a
    pipelined deployment must hold one micro-batch per group to keep its
    stages busy. The search therefore maximizes throughput over the admissible
    batches rather than taking the largest one that merely fits, which would
    starve the pipeline of in-flight work."""
    best_b, best_t = 0, 0.0
    b = 1
    while b <= hi:
        r = run_cell("proteus", model, b, ctx=ctx, devices=devices, dp=dp)
        if r.alive and r.throughput > best_t:
            best_b, best_t = b, r.throughput
        b = b * 2 if b < 16 else b + 16
    return best_b


if __name__ == "__main__":
    main()
