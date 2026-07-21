#!/usr/bin/env python3
"""Scalability analysis (Sec. V-E): (a) Proteus on Llama-3.1-70B scaling from
1 to 16 devices with pipeline parallelism; (b) [PP, DP] combinations on eight
devices at a fixed total batch of 32."""
import os
from common import RESULTS, run_cell, write_csv

MODEL = "llama3-70b"
DEVICES = [1, 4, 8, 16]
PP_DP = [(8, 1), (4, 2), (2, 4), (1, 8)]


def main():
    rows = []
    base = None
    for d in DEVICES:
        r = run_cell("proteus", MODEL, 32, devices=d)
        thr = r.throughput if r.alive else 0.0
        base = base or thr
        rows.append([d, round(thr), round(thr / base, 2)])
    write_csv(os.path.join(RESULTS, "scalability_device.csv"),
              ["devices", "Proteus", "normalized"], rows)

    rows = []
    base = None
    for pp, dp in PP_DP:
        r = run_cell("proteus", MODEL, 32, devices=8, dp=dp)
        thr = r.throughput if r.alive else 0.0
        base = base or thr
        rows.append([f"PP{pp}" + (f"-DP{dp}" if dp > 1 else ""),
                     round(thr), round(thr / base, 3)])
    write_csv(os.path.join(RESULTS, "scalability_parallel.csv"),
              ["config", "Proteus", "normalized"], rows)


if __name__ == "__main__":
    main()
