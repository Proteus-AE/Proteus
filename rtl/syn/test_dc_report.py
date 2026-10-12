#!/usr/bin/env python3
"""Checks on the `report_area -hierarchy` reader of experiments/run_area.py.

Runs standalone (`python3 rtl/syn/test_dc_report.py`, or `make -C rtl
syn-check`) and under pytest. The fixture next to this file,
report_area_pe.rpt, is a Design Compiler area report of the six PE groups
dc_compile.tcl emits, so the checks also pin the group names to the keys of
configs/area/near-bank-pe.yaml.
"""
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from run_area import parse_dc_report                        # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "report_area_pe.rpt")
CFG = os.path.join(ROOT, "configs", "area", "near-bank-pe.yaml")


def _groups():
    with open(CFG) as f:
        return yaml.safe_load(f)["pe_components_um2_28nm"]


def test_fixture_matches_the_area_config():
    """Every configured component is reported, to the tabulated area."""
    cfg = _groups()
    areas = parse_dc_report(FIXTURE, cfg)
    for name, tabulated in cfg.items():
        assert round(areas[name], 1) == tabulated, name


def test_instance_rows_are_keyed_by_design():
    """`u_arb (arbiter_4to1)` rows report the design, not the instance."""
    rpt = _write("""
Hierarchical area distribution
--------------------------------------------------------------------------
Hierarchical cell                  Absolute  Percent  Combi-  Noncombi- Bl-
                                   Total              national national  bx
--------------------------------------------------------------------------
pe_cluster                       144798.1211   100.0    0.0000   0.0000 0.0
u_mode (mode_ctrl)                   14.2049     0.0    7.9349   6.2700 0.0
u_arb (arbiter_4to1)                636.0237     0.4  631.8437   4.1800 0.0
g_pe[0].u_fifo (operand_fifo)      2135.9812     1.5 1084.0000 1051.9812 0.0
g_pe[0].u_pe/u_add_tree (add_tree) 10963.9628    7.6    0.0000   0.0000 0.0
""")
    areas = parse_dc_report(rpt, ["mode_ctrl", "arbiter_4to1"])
    assert areas["mode_ctrl"] == 14.2049
    assert areas["arbiter_4to1"] == 636.0237
    assert areas["operand_fifo"] == 2135.9812
    assert areas["add_tree"] == 10963.9628
    assert areas["pe_cluster"] == 144798.1211
    assert "u_arb" not in areas


def test_missing_group_is_rejected():
    """A report without one of the expected groups must not be accepted."""
    try:
        parse_dc_report(FIXTURE, list(_groups()) + ["sfu_pipeline"])
    except ValueError as e:
        assert "sfu_pipeline" in str(e)
    else:
        raise AssertionError("missing group accepted")


def test_non_hierarchical_report_is_rejected():
    """A plain `report_area` carries no group table and must be refused."""
    rpt = _write("""
Number of ports:                          519
Number of cells:                         2114

Combinational area:                 372.481200
Noncombinational area:             1763.500000
Total cell area:                   2135.981200
""")
    try:
        parse_dc_report(rpt, ["operand_fifo"])
    except ValueError as e:
        assert "-hierarchy" in str(e)
    else:
        raise AssertionError("non-hierarchical report accepted")


_TMP = []


def _write(text):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".rpt")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    _TMP.append(path)
    return path


def main():
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    try:
        for check in checks:
            check()
            print(f"  ok  {check.__name__}")
    finally:
        for path in _TMP:
            os.unlink(path)
    print(f"test_dc_report: {len(checks)} checks passed")


if __name__ == "__main__":
    main()
