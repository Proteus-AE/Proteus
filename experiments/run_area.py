#!/usr/bin/env python3
"""Logic area of the near-bank datapath (Fig. 12, Sec. V-B).

Takes the post-synthesis cell areas of the Verilog under rtl/ (Synopsys
Design Compiler at 28 nm, rtl/syn/dc_compile.tcl), scales them to the 1z-nm
DRAM process, and reports the per-bank PE breakdown, the per-die overhead,
and the per-channel SFU cost. The point of the table is that the incremental
logic is confined to the bank groups: it is a fraction of the DRAM die and,
unlike HBM-PIM, costs no DRAM capacity.

`--dc-report <file>` replaces the tabulated 28 nm areas with the ones parsed
from a `report_area -hierarchy` output, so the figure can be regenerated
from a fresh synthesis run:

    make -C rtl syn                 # writes rtl/reports/pe_area_hier.rpt
    python3 experiments/run_area.py --dc-report rtl/reports/pe_area_hier.rpt

rtl/syn/report_area_pe.rpt is a checked-in report of that shape, so the path
can be exercised without a Design Compiler licence (`make -C rtl syn-check`).
"""
import argparse
import os
import re

import yaml

from common import RESULTS, write_csv

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "area",
                   "near-bank-pe.yaml")

# A row of a `report_area -hierarchy` table:
#
#   Hierarchical cell            Absolute  Percent  Combi-  Noncombi- Black-
#                                Total              national national boxes
#   ------------------------------------------------------------------------
#   pe_mac16                    32754.0202  100.0    0.0000   0.0000   0.0000
#   u_mul_array (mul_array)     19291.0374   58.9    0.0000   0.0000   0.0000
#
# an instance path, the design it references in parentheses (absent on the
# row of the design being reported), and five numeric columns of which the
# first is the cell area of the whole subtree.
HIER_ROW = re.compile(r"^(?P<cell>[\w./$\[\]]+)"
                      r"(?:\s+\((?P<design>\w+)\))?"
                      r"\s+(?P<area>\d+\.\d+)"
                      r"(?:\s+\d+\.\d+){4}\s*$")
HIER_HEAD = "Hierarchical area distribution"


def scale_factor(proc):
    """28 nm logic cell area -> 1z-nm DRAM-process area."""
    shrink = (proc["target_node_nm"] / proc["synth_node_nm"]) ** 2
    return shrink * proc["dram_logic_penalty"]


def parse_dc_report(path, expected):
    """Group areas (um^2) from a `report_area -hierarchy` output.

    A row is keyed by the design it refers to -- the parenthesized reference
    on an instance row, the cell name itself on the row of the design being
    reported -- and the topmost occurrence wins, so a group is the cell area
    of one instance of that design including its subtree. That holds for a
    single top-level report (rtl/reports/pe_cluster_area.rpt lists
    `u_arb (arbiter_4to1)`, `g_pe[0].u_fifo (operand_fifo)`, ...) as well as
    for the per-group report dc_compile.tcl concatenates for this figure.

    `expected` is the set of group names the caller needs; a report missing
    any of them is rejected rather than silently topped up from the
    tabulated areas.
    """
    areas = {}
    saw_table = False
    with open(path) as f:
        for ln in f:
            if HIER_HEAD in ln:
                saw_table = True
                continue
            m = HIER_ROW.match(ln)
            if m:
                areas.setdefault(m.group("design") or m.group("cell"),
                                 float(m.group("area")))
    if not saw_table:
        raise ValueError(
            f"{path}: no '{HIER_HEAD}' table -- report_area was run without "
            f"-hierarchy (see rtl/syn/dc_compile.tcl)")
    missing = [g for g in expected if g not in areas]
    if missing:
        raise ValueError(
            f"{path}: no area reported for {', '.join(missing)}; the report "
            f"holds {', '.join(sorted(areas)) or 'no group'}. Group names "
            f"must match the keys of configs/area/near-bank-pe.yaml.")
    return areas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc-report",
                    help="Design Compiler report_area -hierarchy output")
    args = ap.parse_args()

    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    comp = dict(cfg["pe_components_um2_28nm"])
    sfu = cfg["sfu_um2_28nm"]
    if args.dc_report:
        parsed = parse_dc_report(args.dc_report, comp)
        for name in comp:
            comp[name] = parsed[name]
        sfu = parsed.get("sfu_pipeline", sfu)
        print(f"using synthesis areas from {args.dc_report}")

    k = scale_factor(cfg["process"])
    total_um2 = sum(comp.values())
    pe_mm2 = total_um2 * k / 1e6
    arith = cfg["arithmetic_components"]
    arith_mm2 = sum(comp[c] for c in arith) * k / 1e6
    datapath_mm2 = pe_mm2 - arith_mm2
    sfu_mm2 = sfu * k / 1e6

    die = cfg["die"]
    banks = die["banks"]
    per_bank_die_mm2 = die["area_mm2"] / banks
    pes_mm2 = pe_mm2 * banks
    overhead = pes_mm2 / die["area_mm2"]

    rows = [[c, round(comp[c], 1), round(comp[c] * k / 1e6, 6),
             round(comp[c] / total_um2 * 100, 2)] for c in comp]
    write_csv(os.path.join(RESULTS, "area_pe_breakdown.csv"),
              ["component", "area_um2_28nm", "area_mm2_1z", "share_pct"],
              rows)
    write_csv(os.path.join(RESULTS, "area_summary.csv"),
              ["quantity", "value"],
              [["scale_28nm_to_1z", round(k, 4)],
               ["pe_area_mm2", round(pe_mm2, 4)],
               ["pe_arithmetic_mm2", round(arith_mm2, 4)],
               ["pe_datapath_mm2", round(datapath_mm2, 4)],
               ["pes_per_die_mm2", round(pes_mm2, 4)],
               ["die_area_mm2", die["area_mm2"]],
               ["die_overhead_pct", round(overhead * 100, 3)],
               ["per_bank_die_area_mm2", round(per_bank_die_mm2, 4)],
               ["sfu_per_channel_mm2", round(sfu_mm2, 4)]])

    print(f"\n28 nm -> {cfg['process']['target_node_nm']} nm DRAM process: "
          f"x{k:.3f} "
          f"(({cfg['process']['target_node_nm']}/"
          f"{cfg['process']['synth_node_nm']})^2 x "
          f"{cfg['process']['dram_logic_penalty']:g} logic-DRAM penalty)")
    print(f"\nper-bank PE: {pe_mm2:.3f} mm^2  "
          f"(<< {per_bank_die_mm2:.2f} mm^2 of per-bank die area)")
    for c in sorted(comp, key=lambda x: -comp[x]):
        print(f"  {c:<14} {comp[c]/total_um2*100:5.2f}%  "
              f"{comp[c]*k/1e6:.4f} mm^2")
    print(f"  {'-'*40}")
    print(f"  MAC array + accumulator : {arith_mm2:.3f} mm^2 "
          f"({arith_mm2/pe_mm2*100:.1f}% of the PE)")
    print(f"  FIFO + 4:1 arbiter + mode regs: {datapath_mm2:.3f} mm^2")
    print(f"\n{banks} PEs per die: {pes_mm2:.2f} mm^2 = {overhead*100:.2f}% "
          f"of a {die['area_mm2']:.2f} mm^2 {die['name']} die")
    print(f"per-channel SFU: {sfu_mm2:.2f} mm^2, outside the DRAM arrays, "
          f"amortized across {banks} banks")
    print("\nAll of it is bank-group-local: no cross-BG routing, no "
          "channel-wide crossbar, and no DRAM capacity is sacrificed.")


if __name__ == "__main__":
    main()
