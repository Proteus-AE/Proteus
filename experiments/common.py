"""Shared experiment definitions (Sec. V evaluation setup)."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proteus_sim import load_model, build_system          # noqa: E402
from proteus_sim.workload import build_workload           # noqa: E402
from proteus_sim.system import VARIANTS                   # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
os.makedirs(RESULTS, exist_ok=True)

# Evaluation setup (Sec. V-A): input 2K / output 6K, batch 16-64.
CTX_IN, CTX_OUT = 2048, 6144
BATCHES = [16, 32, 64]
MODELS = ["deepseek-v2-lite", "switch-26b", "mixtral-8x7b", "llama3-70b"]
SYSTEMS = ["dgx-a100", "cxl-pnm", "cent", "neupims", "papi", "pimphony", "proteus"]
SYSTEM_LABELS = {"dgx-a100": "DGX-A100", "cxl-pnm": "CXL-PNM", "cent": "CENT",
                 "neupims": "NeuPIMs", "papi": "PAPI", "pimphony": "PIMphony",
                 "proteus": "Proteus"}
NORM_BASE = "cxl-pnm"          # normalization anchor (runs in every config)


def run_cell(system, model_name, batch, ctx=None, devices=None, dp=1,
             variant="full", routing="expected", seed=7):
    model = load_model(model_name)
    sys_ = build_system(system, features=VARIANTS[variant])
    w = build_workload(model, batch, CTX_IN, CTX_OUT, routing=routing,
                       seed=seed, ctx_override=ctx)
    return sys_.simulate(w, devices=devices, dp=dp)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(header)
        cw.writerows(rows)
    print(f"wrote {os.path.relpath(path, os.path.dirname(RESULTS))}")


def geomean(vals):
    vals = np.array([v for v in vals if v], dtype=float)
    return float(np.exp(np.log(vals).mean())) if len(vals) else 0.0
