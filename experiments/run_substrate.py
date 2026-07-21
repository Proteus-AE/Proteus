#!/usr/bin/env python3
"""Memory-substrate comparison at command level (Table "DRAM technologies",
Sec. IV-B), executed on the PimCore C++ backend.

Runs the identical reuse-free GEMV streaming kernel on the three candidate
near-bank substrates -- LPDDR5X-PIM, HBM-PIM, GDDR6-AiM -- and reports the
sustained per-channel and device-level internal bandwidth, row-buffer
behavior, and array-access energy, grounding the C1/C2/C3 substrate
comparison of the paper in command-level measurements. Also cross-validates
the C++ and Python implementations of the LPDDR5X backend against each
other.
"""
import os

from common import RESULTS, write_csv
from proteus_sim.config import load_memory
from proteus_sim.dram import PimChannel
from proteus_sim.dram import pimcore_bridge as pc
from trace_gen import gemv_trace

SUBSTRATES = ["lpddr5x-8533", "hbm-pim", "gddr6-aim"]
ROWS = 96


def main():
    if not pc.available():
        print("cmake/g++ unavailable; skipping the C++ substrate study")
        return
    print("building PimCore (C++) ...")
    pc.build()
    print(pc.run_tests().strip())

    rows = [["substrate", "channel_gbps", "efficiency", "device_tbps",
             "row_hit", "near_bank_pj_bit", "external_pj_bit"]]
    for sub in SUBSTRATES:
        rep = pc.run_kernel(config=sub, kernel="gemv", rows=ROWS)
        ext = pc.run_kernel(config=sub, kernel="gemv", rows=ROWS,
                            external_energy=True)
        s, e = rep["stats"], rep["energy"]
        rows.append([sub,
                     round(s["sustained_pim_bw"] / 1e9, 1),
                     round(s["efficiency"], 3),
                     round(s["sustained_pim_bw"] * rep["device_channels"]
                           / 1e12, 2),
                     round(s["row_hit_rate"], 3),
                     round(e["pj_per_bit"], 2),
                     round(ext["energy"]["pj_per_bit"], 2)])
        print(f"  {sub:<14} {rows[-1][1]:>7} GB/s/ch (eff {rows[-1][2]}) | "
              f"device {rows[-1][3]} TB/s | {rows[-1][5]} / {rows[-1][6]} "
              f"pJ/bit")
    write_csv(os.path.join(RESULTS, "substrate_comparison.csv"),
              rows[0], rows[1:])

    # C++ vs Python cross-validation on the LPDDR5X backend.
    mem = load_memory("lpddr5x-8533")
    py = PimChannel(mem, "direct").execute(gemv_trace(ROWS, mem))
    cpp = pc.run_kernel(config="lpddr5x-8533", kernel="gemv", rows=ROWS)
    dev = abs(cpp["stats"]["time_ns"] / py.time_ns - 1)
    print(f"\ncross-validation (LPDDR5X GEMV, {ROWS} rows/bank):")
    print(f"  python backend : {py.time_ns/1e3:9.1f} us")
    print(f"  C++ backend    : {cpp['stats']['time_ns']/1e3:9.1f} us "
          f"(deviation {dev*100:.1f}%)")
    write_csv(os.path.join(RESULTS, "backend_crossvalidation.csv"),
              ["backend", "time_us", "sustained_gbps"],
              [["python", round(py.time_ns / 1e3, 1),
                round(py.sustained_bw() / 1e9, 1)],
               ["cpp", round(cpp["stats"]["time_ns"] / 1e3, 1),
                round(cpp["stats"]["sustained_pim_bw"] / 1e9, 1)]])


if __name__ == "__main__":
    main()
