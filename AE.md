# Artifact Evaluation Appendix

## Abstract

This artifact provides ProteusSim, the simulator used for the evaluation of
Proteus (Sec. V), including the operator-level system timing/energy engine,
the command-level LPDDR5X-PIM backend with its trace generator, the six
baseline system models, and scripts reproducing every figure of the paper.

## Requirements

* **Hardware**: any x86/ARM machine, 1 core, < 1 GB RAM, < 100 MB disk.
* **Software**: Python >= 3.9 with `pyyaml`, `matplotlib`, `numpy`
  (`pip install -r requirements.txt`), CMake >= 3.16, and a C++17 compiler
  (`make core` builds the pimcore binaries the cross-check experiments
  run). Optional: `make deps` fetches and builds Ramulator 2.0 and ONNXim
  (`ext/`) for the external cross-checks (network + g++-12), `onnx` for
  the ONNXim path (`pip install .[integration]`), Icarus Verilog for the
  RTL testbenches.
* **Time**: full reproduction `bash scripts/run_all.sh` ~= 3 minutes;
  per-figure scripts run in seconds (the command-level microbenchmarks
  dominate the runtime).

## Claims mapped to experiments

| Paper claim | Experiment | Check |
|---|---|---|
| Throughput gains over 6 baselines, 4 models x b16-64 (Fig. Overall a) | `experiments/run_overall.py` | monotonicity + bound tests |
| Energy-efficiency gains (Fig. Overall b) | `experiments/run_overall.py` | monotonicity + bound tests |
| Mechanism attribution +AS/+RD/+OF/+EC (Fig. Breakdown) | `experiments/run_breakdown.py` | chain monotonicity test |
| Context-length & batch sensitivity (Fig. Sensitivity) | `experiments/run_sensitivity.py` | monotonicity tests |
| Near-linear device scaling; PP > DP (Fig. Scalability) | `experiments/run_scalability.py` | scaling test |
| Broadcasting = min(n,4)-fold operand reuse (Sec. IV-C) | `experiments/run_microbench.py` (2)(5) | `tests/test_dram_backend.py` |
| Broadcasting frees memory-service slots for the xPU (Sec. IV-B/C) | `experiments/run_microbench.py` (3) | idem |
| Sustained-rate constants of the analytical layer | `experiments/run_microbench.py` (1)(4) | idem |
| Substrate choice (Table "DRAM technologies") | `experiments/run_substrate.py` | C++/Python backend agreement |
| Host/PIM arbitration trade-off (Sec. IV-B) | `experiments/run_coexec.py` | `pimcore_tests` |
| Implementation correctness (two independent engines) | `experiments/run_sys_crosscheck.py` | <1% on all 198 cells |
| Serving dynamics (independent C++ engine) | `experiments/run_serving.py` | <3% steady-state agreement |
| Host-path DRAM model vs Ramulator 2.0 | `experiments/run_ramulator_xcheck.py` | external anchor (needs `make deps`) |
| xPU tile model vs ONNXim | `experiments/run_onnxim_xcheck.py` | external anchor (needs `make deps`) |

## Workflow

```bash
pip install -r requirements.txt
make core                        # C++ binaries (required)
make all                         # all experiments, figures, tests
make deps                        # optional: external cross-check simulators
```

Single configurations, operator-graph dumps, pipeline timelines, and PIM
trace replay are available through `python main.py --help`.
`integration/ramulator2/` carries optional patches hooking the PIM device
model into Ramulator 2.0; no result depends on them.

## Expected results

Generated CSVs under `results/` correspond one-to-one to the paper's
figures.
