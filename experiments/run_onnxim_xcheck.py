#!/usr/bin/env python3
"""xPU-side cross-check against ONNXim.

The decode-layer weight GEMMs of Llama-3.1-70B (batch 32) are run through

  * the tile-level xpucore engine (the model used by run_integrated), and
  * ONNXim's cycle-level NPU simulator (ext/onnxim), on the same shapes
    and an equivalent single-core systolic configuration.

Requires an ONNXim build and the `onnx` package (pip install
.[integration]); without them the experiment reports the xpucore side only
and marks the external column SKIPPED.
"""
import os
import sys

from common import RESULTS, write_csv
from proteus_sim import load_model
from proteus_sim.xpucore import SystolicConfig, XpuEngine
from proteus_sim.xpucore import onnxim_bridge as ox

BATCH = 32
FREQ_MHZ = 1410.0


def decode_gemms(model):
    """(name, m, k, n) for the per-layer weight GEMMs at decode."""
    d = model["d_model"]
    inter = model.get("d_ffn", 4 * d)
    return [
        ("qkv_proj", BATCH, d, 3 * d),
        ("out_proj", BATCH, d, d),
        ("ffn_up", BATCH, d, inter),
        ("ffn_down", BATCH, inter, d),
    ]


def main():
    model = load_model("llama3-70b")
    # Single-core configuration so both sides time one 32x32 array bank.
    eng = XpuEngine(SystolicConfig(n_arrays=1, freq_ghz=FREQ_MHZ / 1e3))

    try:
        ox.find_binary()
        have = True
    except ox.OnnximUnavailable as e:
        have = False
        print(f"note: {e}")
        print("      reporting the xpucore side only\n")

    rows = []
    for name, m, k, n in decode_gemms(model):
        cyc_local = int(round(eng.tile(name, m, k, n).compute_cycles))
        if have:
            try:
                rep = ox.run_gemm(m, k, n)
                cyc_ext = rep["cycles"]
                dev = f"{abs(cyc_ext - cyc_local) / cyc_ext * 100:.1f}%"
            except ox.OnnximUnavailable as e:
                cyc_ext, dev = f"FAILED: {str(e)[:50]}", "n/a"
        else:
            cyc_ext, dev = "SKIPPED", "n/a"
        rows.append([name, m, k, n, cyc_local, cyc_ext, dev])
        print(f"{name:<10} ({m}x{k}x{n}): xpucore {cyc_local:>10,} cycles"
              f" | ONNXim {cyc_ext} | deviation {dev}")

    write_csv(os.path.join(RESULTS, "onnxim_xcheck.csv"),
              ["gemm", "M", "K", "N", "xpucore_cycles", "onnxim_cycles",
               "deviation"], rows)
    print(f"\nwrote {os.path.join(RESULTS, 'onnxim_xcheck.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
