#!/usr/bin/env python3
"""Generate request-arrival traces for the serving-layer simulation.

Trace format (one request per line, replayable and shareable):

    # arrival_ms  prompt_tokens  output_tokens
    0.0   1732  412
    3.1   2048  256
    ...

Distributions follow production serving studies: lognormal prompt/output
lengths and Poisson arrivals. `--closed-loop` emits arrival_ms = 0 for all
requests (the closed-loop pool model used by the paper's throughput
experiments); a rate in requests/s produces an open-loop trace.

Examples:
  python trace_gen/gen_requests.py -n 256 --closed-loop -o request_traces/pool256.txt
  python trace_gen/gen_requests.py -n 512 --rate 40 --prompt-mean 2048 \
      --output-mean 256 -o request_traces/poisson40.txt
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate(n, rate, prompt_mean, output_mean, seed, closed_loop):
    rng = random.Random(seed)
    t_ms = 0.0
    rows = []
    for _ in range(n):
        prompt = max(64, int(rng.lognormvariate(0, 0.35) * prompt_mean))
        out = max(32, int(rng.lognormvariate(0, 0.45) * output_mean))
        rows.append((0.0 if closed_loop else t_ms, prompt, out))
        if not closed_loop and rate > 0:
            t_ms += rng.expovariate(rate) * 1e3
    return rows


def write_trace(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("# arrival_ms  prompt_tokens  output_tokens\n")
        for a, p, o in rows:
            f.write(f"{a:.1f}\t{p}\t{o}\n")


def read_trace(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            a, p, o = ln.split()
            rows.append((float(a), int(p), int(o)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=256, help="number of requests")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="Poisson arrival rate (requests/s); 0 = closed loop")
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--prompt-mean", type=int, default=2048)
    ap.add_argument("--output-mean", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    rows = generate(args.n, args.rate, args.prompt_mean, args.output_mean,
                    args.seed, args.closed_loop or args.rate <= 0)
    write_trace(args.out, rows)
    print(f"{len(rows)} requests -> {args.out}")


if __name__ == "__main__":
    main()
