# ProteusSim

Simulation infrastructure for **Proteus**, a heterogeneous xPU +
near-bank-PIM system for LLM inference, and the six baseline systems
evaluated in the paper (DGX-A100, CXL-PNM, CENT, NeuPIMs, PAPI, PIMphony).

Two layers:

* an **operator-level system engine** resolves one steady-state decode
  iteration: workload lowering, crossover scheduling, xPU-PIM co-execution,
  pipeline parallelism, capacity-bounded in-flight batching, and
  activity-based energy;
* a **command-level LPDDR5X-PIM backend** executes all-bank command traces
  under the DRAM timing state (row buffers, tCCD_L bank-group buses,
  tRRD/tFAW, refresh, PE operand FIFOs, direct/broadcast connectivity).
  The sustained rates the system layer consumes are derived from and
  cross-checked against this backend.

Both layers exist twice: in Python (`proteus_sim/`) and in dependency-free
C++17 (`pimcore/`). Every evaluation cell is cross-checked between the two
(`experiments/run_sys_crosscheck.py`, `experiments/run_substrate.py`).

## Building

Requirements: Python >= 3.9, CMake >= 3.16, a C++17 compiler.

```bash
pip install -r requirements.txt        # pyyaml, matplotlib, numpy
make core                              # C++ core (pimcore binaries)
```

The C++ core is required: the substrate/co-execution studies, the
system-layer cross-check, and the serving cross-check run its binaries.

Optional components:

```bash
make deps                  # ext/: Ramulator 2.0 + ONNXim (cross-checks;
                           #   clones + builds, needs network and g++-12)
pip install .[integration] # onnx package for the ONNXim cross-check
make rtl                   # Verilog testbenches (needs iverilog)
```

## Quick start

```bash
# One configuration, with derived machine parameters and placements:
python main.py --system proteus --model mixtral-8x7b --batch 32 --verbose

# A baseline on the identical workload:
python main.py --system neupims --model mixtral-8x7b --batch 32

# Effectiveness-analysis variants (Sec. V-C):
python main.py --system proteus --model mixtral-8x7b --batch 32 --variant rd

# Sampled MoE routing; pipeline timeline:
python main.py --system proteus --model mixtral-8x7b --batch 32 \
    --routing sampled --iters 200
python main.py --system proteus --model llama3-70b --batch 32 --timeline

# Operator graph after the fusion/annotation passes:
python main.py --model mixtral-8x7b --dump-graph graph.json

# Generate and replay a PIM command trace on the command-level backend:
python trace_gen/gen_trace.py --kernel skinny-gemm --rows 64 --vectors 8 \
    --mode broadcast -o results/gemm.trace
python main.py --replay-trace results/gemm.trace --pim-mode broadcast
```

## Reproducing the paper's evaluation

| Paper figure / claim | Command | Output (results/) |
|---|---|---|
| Fig. Overall (a)(b): throughput & energy efficiency | `python experiments/run_overall.py` | `throughput_*.csv`, `energyeff_*.csv` |
| Fig. Breakdown: Base/+AS/+RD/+OF/+EC | `python experiments/run_breakdown.py` | `effectiveness_breakdown.csv` |
| Fig. Sensitivity: context & batch sweeps | `python experiments/run_sensitivity.py` | `sensitivity_*.csv` |
| Fig. Scalability: devices & [PP,DP] | `python experiments/run_scalability.py` | `scalability_*.csv` |
| Sec. IV-B/C mechanisms at command level | `python experiments/run_microbench.py` | `microbench_*.csv` |
| Runtime adaptation under continuous batching | `python experiments/run_serving.py` | `serving_dynamics_*.csv` |
| Substrate comparison (C1/C2/C3) at command level | `python experiments/run_substrate.py` | `substrate_comparison.csv` |
| Host/PIM arbitration policy study | `python experiments/run_coexec.py` | `coexec_policies.csv` |
| C++ vs Python cross-check (all tables) | `python experiments/run_sys_crosscheck.py` | `sys_crossvalidation.csv` |
| Integrated xPU+PIM co-simulation | `python experiments/run_integrated.py` | `integrated_*.csv` |
| Host path vs Ramulator 2.0 (ext/) | `python experiments/run_ramulator_xcheck.py` | `ramulator_xcheck.csv` |
| xPU tiles vs ONNXim (ext/) | `python experiments/run_onnxim_xcheck.py` | `onnxim_xcheck.csv` |
| All figures (PNG) | `python experiments/plot_all.py` | `figures/*.png` |

Everything at once (about three minutes), including both test suites:

```bash
make all        # = make core + bash scripts/run_all.sh
```

The two external cross-checks report SKIPPED unless `make deps` has been
run (they compare the built-in backends against independent simulators;
no paper figure depends on them).

## Repository layout

```
configs/
  models/        llama3-70b, mixtral-8x7b, switch-26b, deepseek-v2-lite
  systems/       proteus + six baselines (all mechanism parameters)
  memory/        LPDDR5X-8533 organization, DRAM timing, energy/bit
proteus_sim/
  workload.py    model -> per-layer operator list (traffic/FLOP accounting)
  compiler/      operator-graph IR, lowering, fusion & AI-annotation passes
  scheduler.py   crossover model, placement, PIM-mode selection
  memory.py      analytical sustained-rate derivation
  system.py      Proteus timing engine, variants, pipeline, energy
  serving.py     closed-loop continuous-batching request simulation
  xpucore/       tile-level systolic xPU engine (+ optional ONNX front-end)
  fabric.py      CXL 3.0 fabric + iteration timeline
  dram/          command-level PIM backend (banks, buses, refresh, PEs,
                 mode switching, xPU slot stealing, command energy)
  baselines/     gpu, cxl_pnm, cent, neupims, papi, pimphony
trace_gen/       kernel/request trace generation CLIs (Python; the C++
                 counterpart is pimcore_tracegen)
request_traces/  replayable request-arrival traces for the serving layer
pimcore/         C++17 core: memory backend, system layer, serving engine,
                 trace/host-stream generation (see its README)
ext/             external simulators (Ramulator 2.0, ONNXim; make deps)
integration/     Ramulator 2.0 patches (device model, controller plugin,
                 trace frontend); optional, see its README
rtl/             synthesizable Verilog + testbenches + DC synthesis scripts
experiments/     per-figure scripts, microbenchmarks, plotting
tests/           system-level and backend test suites
docs/            DRAM model notes, API reference
```

`tests/` covers physical bounds, monotonicity, OOM accounting, crossover
closure, streaming efficiency, broadcast reuse, trace round-trip, and the
compiler passes. `AE.md` maps paper claims to experiments.

## Documentation

* `docs/dram-model.md` -- the command-level backend and its cross-validation
  against the analytical layer.
* `docs/api.md` -- Python API reference.

## License

MIT -- see `LICENSE`.
