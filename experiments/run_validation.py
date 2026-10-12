#!/usr/bin/env python3
"""Validation of the machine model against reference points (Sec. V-A).

Three layers of checking, in increasing strength:

1. *Machine-model validation.* The near-bank substrate and the xPU roofline
   are compared against published operating points
   (configs/validation/reference_points.yaml). This is what pins the PIM
   timing model to a real near-bank part and the xPU model to the A100 whose
   throughput the evaluation assumes.

2. *Configuration audit.* Every baseline's aggregate capacity, bandwidth and
   device count are recomputed from its configuration and compared against
   the organization its own work reports -- the "iso-device" setup of
   Sec. V-A. A silent drift in a baseline config is the easiest way to make
   a comparison unfair, so it is checked mechanically.

3. *Implementation cross-validation.* The closed-form analytical layer, the
   event-driven command-level backend, and the independent C++ port must
   agree; run_microbench.py and run_sys_crosscheck.py carry those out and
   this script reports where to find their numbers.
"""
import os

import yaml

from common import RESULTS, write_csv
from proteus_sim import build_system, load_memory
from proteus_sim.config import load_system
from proteus_sim.dram import PimChannel
from proteus_sim.memory import derive
from proteus_sim.xpucore import SystolicConfig, XpuEngine
from trace_gen import gemv_trace

CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "validation",
                   "reference_points.yaml")


def aim_measurements():
    """Peak and sustained all-bank GEMV rates of the AiM-class substrate.

    The same organization model and the same command-level backend are run
    on the GDDR6-AiM timing table, so the published AiM operating point is
    reproduced from the substrate configuration rather than asserted."""
    mem = load_memory("gddr6-aim")
    d = derive(mem)
    ch = PimChannel(mem, mode="direct")
    st = ch.execute(gemv_trace(96, mem, set_mode=False))
    return dict(aim_allbank_peak_tflops=d.pe_flops_peak / 1e12,
                aim_allbank_internal_tbps=d.internal_peak / 1e12,
                _eff=st.sustained_bw() / ch.peak_bw("direct"),
                _cross={"aim_direct_sustained_tbps":
                        (d.internal_bw / 1e12,
                         st.sustained_bw() * d.channels / 1e12)})


def lpddr_cross_layer():
    """Analytical sustained rates vs the command-level backend."""
    from trace_gen import skinny_gemm_trace
    mem = load_memory("lpddr5x-8533")
    d = derive(mem)
    channels = mem["packages_per_device"] * mem["channels_per_package"]
    out = {}
    ch = PimChannel(mem, "direct")
    st = ch.execute(gemv_trace(96, mem, set_mode=False))
    out["lpddr5x_direct_sustained_tbps"] = (
        d.internal_bw / 1e12, st.sustained_bw() * channels / 1e12)
    cb = PimChannel(mem, "broadcast")
    sb = cb.execute(skinny_gemm_trace(96, 4, mem, mode="broadcast",
                                      set_mode=False))
    out["lpddr5x_broadcast_sustained_tbps"] = (
        d.broadcast_bw / 1e12, sb.sustained_bw() * channels / 1e12)
    return out


def xpu_measurements():
    eng = XpuEngine(SystolicConfig())
    gpu = build_system("dgx-a100").cfg
    bw = gpu["hbm_bw_aggregate"] / gpu["devices"] / 1e12
    peak = eng.cfg.peak_flops / 1e12
    return dict(a100_peak_tflops_fp16=peak,
                a100_hbm_tbps=bw,
                a100_ridge_flops_per_byte=peak / bw)


def organization(name):
    sys_ = build_system(name)
    cfg = sys_.cfg
    devices = cfg["devices"]
    if name == "proteus":
        d = sys_.dmem
        return dict(capacity_gb=d.capacity * devices / 1e9,
                    bandwidth_tbps=d.internal_peak * devices / 1e12,
                    devices=devices)
    cap = sys_.total_capacity() / cfg.get("usable_fraction", 0.90)
    # The reported organization is the device's *external* memory bandwidth
    # (aggregate over the deployment); near-bank all-bank rates are internal
    # multipliers on top of it and are audited by the substrate study.
    for key in ("hbm_bw_aggregate", "xpu_bw_aggregate"):
        if key in cfg:
            bw = cfg[key]
            break
    else:
        bw = cfg.get("internal_bw_per_device",
                     cfg.get("bw_per_device", 0.0)) * devices
    return dict(capacity_gb=cap / 1e9, bandwidth_tbps=bw / 1e12,
                devices=devices)


def main():
    ref = yaml.safe_load(open(CFG))
    measured = {}
    aim = aim_measurements()
    cross = dict(aim.pop("_cross"))
    eff = aim.pop("_eff")
    measured.update(aim)
    measured.update(xpu_measurements())
    cross.update(lpddr_cross_layer())

    rows, failures = [], 0
    print("machine-model validation")
    for group in ("near_bank_pim", "xpu"):
        for e in ref[group]:
            got = measured[e["name"]]
            dev = abs(got / e["reported"] - 1)
            ok = dev <= e["tolerance"]
            failures += not ok
            rows.append([group, e["name"], e["reported"], round(got, 4),
                         round(dev * 100, 2), "PASS" if ok else "FAIL"])
            print(f"  {e['name']:<32} reported {e['reported']:>8.3f}  "
                  f"simulated {got:>8.3f}  deviation {dev*100:5.2f}%  "
                  f"{'PASS' if ok else 'FAIL'}")
    print(f"  (all-bank GEMV streaming efficiency of the AiM substrate: "
          f"{eff:.3f})")

    print("\ncross-layer closure "
          "(analytical closed form vs command-level backend)")
    for e in ref["cross_layer"]:
        ana, meas = cross[e["name"]]
        dev = abs(meas / ana - 1)
        ok = dev <= e["tolerance"]
        failures += not ok
        rows.append(["cross_layer", e["name"], round(ana, 4),
                     round(meas, 4), round(dev * 100, 2),
                     "PASS" if ok else "FAIL"])
        print(f"  {e['name']:<32} analytical {ana:>8.3f}  "
              f"backend {meas:>8.3f}  deviation {dev*100:5.2f}%  "
              f"{'PASS' if ok else 'FAIL'}")

    print("\nbaseline organization audit (iso-device setup of Sec. V-A)")
    for name, want in ref["baseline_organization"].items():
        got = organization(name)
        worst = 0.0
        for k, v in want.items():
            dev = abs(got[k] / v - 1) if v else 0.0
            worst = max(worst, dev)
            rows.append(["organization", f"{name}.{k}", v, round(got[k], 3),
                         round(dev * 100, 2), "PASS" if dev <= 0.02 else "FAIL"])
        failures += worst > 0.02
        print(f"  {name:<10} capacity {got['capacity_gb']:>7.0f} GB "
              f"(want {want['capacity_gb']})  bandwidth "
              f"{got['bandwidth_tbps']:>7.2f} TB/s "
              f"(want {want['bandwidth_tbps']})  "
              f"{'PASS' if worst <= 0.02 else 'FAIL'}")

    # Sec. V-A: heterogeneous systems match Proteus's aggregate xPU
    # throughput. Enforced here rather than asserted in prose.
    print("\nheterogeneous xPU parity (Sec. V-A)")
    want_fl = build_system("proteus").xpu_flops * \
        load_system("proteus")["devices"]
    for name in ["dgx-a100", "neupims", "papi", "pimphony"]:
        c = load_system(name)
        got = c.get("flops_fp16_aggregate") or c.get("xpu_flops_aggregate")
        dev = abs(got / want_fl - 1) if got else 1.0
        ok = got is not None and dev <= 0.02
        failures += not ok
        rows.append(["xpu_parity", name, want_fl / 1e12,
                     round((got or 0) / 1e12, 1), round(dev * 100, 2),
                     "PASS" if ok else "FAIL"])
        print(f"  {name:<10} aggregate xPU {(got or 0)/1e12:>7.1f} TFLOPS "
              f"(want {want_fl/1e12:.1f})  {'PASS' if ok else 'FAIL'}")

    write_csv(os.path.join(RESULTS, "validation.csv"),
              ["group", "quantity", "reported", "simulated", "dev_pct",
               "status"], rows)
    print("\nimplementation cross-validation")
    print("  analytical layer vs command-level backend : "
          "experiments/run_microbench.py -> results/microbench_*.csv")
    print("  Python engine vs independent C++ port     : "
          "experiments/run_sys_crosscheck.py -> results/sys_crossvalidation.csv")
    print("  built-in DRAM model vs Ramulator 2.0      : "
          "experiments/run_ramulator_xcheck.py (needs `make deps`)")
    print("  xPU tile model vs ONNXim                  : "
          "experiments/run_onnxim_xcheck.py (needs `make deps`)")
    print(f"\n{'VALIDATION PASS' if not failures else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
