#!/usr/bin/env python3
"""Co-execution policy study (Sec. IV-B unified memory path), executed on
the PimCore C++ backend's host-controller engine.

Sweeps the three host/PIM arbitration policies and offered host loads
against direct- and broadcasting-mode skinny-GEMM kernels, reporting the
PIM kernel slowdown, achieved host bandwidth, and host latency
distribution. The study grounds two claims at command level: (1) with PIM
priority, broadcasting's reduced memory-service demand is what creates
usable host bandwidth; (2) host-priority arbitration bounds host tail
latency at a quantified cost in PIM kernel time -- the trade-off the
work-conserving scheduler of Sec. IV-D navigates.
"""
import os
import subprocess

from common import RESULTS, write_csv
from proteus_sim.dram import pimcore_bridge as pc


def main():
    if not pc.available():
        print("cmake/g++ unavailable; skipping the co-execution study")
        return
    pc.build()
    binary = os.path.join(pc.BUILD_DIR, "pimcore_coexec")
    out = subprocess.run(
        [binary, "--config", pc.config_path("lpddr5x-8533"),
         "--rows", "48", "--vectors", "8"],
        check=True, capture_output=True, text=True).stdout.strip()
    lines = [ln.split(",") for ln in out.splitlines()]
    write_csv(os.path.join(RESULTS, "coexec_policies.csv"),
              lines[0], lines[1:])

    print("host/PIM arbitration under an 8-vector skinny-GEMM "
          "(per channel):")
    print(f"  {'mode':<10}{'policy':<15}{'load':>6}{'PIM slowdown':>14}"
          f"{'host GB/s':>11}{'lat p95':>10}")
    for r in lines[1:]:
        print(f"  {r[0]:<10}{r[1]:<15}{r[2]:>6}{float(r[3]):>14.2f}"
              f"{float(r[4]):>11.2f}{float(r[6]):>9.0f}ns")


if __name__ == "__main__":
    main()
