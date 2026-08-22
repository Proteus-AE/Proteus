# ProteusSim

Simulation infrastructure for **Proteus**, a scalable and adaptive
heterogeneous xPU + near-bank-PIM system for LLM inference, and the six
baseline systems it is compared against (DGX-A100, CXL-PNM, CENT, NeuPIMs,
PAPI, PIMphony).

Three layers, each validated against the one below it:

* an **operator-level system engine** resolves one steady-state decode
  iteration -- workload lowering into per-matrix and per-expert operator
  groups, the analytical crossover estimate and its runtime adaptation,
  xPU/PIM co-execution, memory-hierarchy-aligned parallelism (tensor-parallel
  groups inside a CXL switch domain, pipeline parallelism across them) with
  its ring collectives, capacity-bounded in-flight batching, and
  activity-based energy;
* a **command-level LPDDR5X-PIM backend** executes all-bank command streams
  under the DRAM timing state: per-bank row buffers, the bank-local near-bank
  read path and its two column cadences, bank-group broadcast distribution,
  tRRD/tFAW, refresh, PE operand FIFOs, mode switching, and command energy.
  Every sustained rate the system layer consumes is derived in closed form
  *and* measured here;
* a **request-level serving driver** replays an arrival trace under KV
  admission control and continuous batching, and reports per-token latency
  and SLO attainment against offered load.

Both the system layer and the memory backend exist twice -- in Python
(`proteus_sim/`) and in dependency-free C++17 (`pimcore/`). The two read the
same configuration tree, and every cell of the evaluation grid together with
the derived variant, scalability and crossover tables is cross-checked
between them.

## Building

Requirements: Python >= 3.9, CMake >= 3.16, a C++17 compiler.

```bash
pip install -r requirements.txt        # pyyaml, matplotlib, numpy
make core                              # C++ core (pimcore binaries)
make traces                            # replayable kernel + request traces
make test                              # all four test suites
```

Both steps are prerequisites, not conveniences. The C++ core carries the
substrate studies, the co-execution study and both cross-checks, and the
experiment scripts abort naming the build step if its binaries are absent.
The replayed traces are not shipped: `make traces` regenerates them from the
seeded generators in `trace_gen/`, so what the serving experiments consume is
produced on the machine that runs them.

Further components, each with its own toolchain requirement:

```bash
make deps                  # ext/: Ramulator 2.0 + ONNXim at the revisions
                           #   pinned in ext/VERSIONS (clones + builds,
                           #   needs network); integration/ramulator2/
                           #   carries the Proteus device model, the
                           #   PIM-priority controller plugin and the
                           #   trace-replay frontend
pip install .[integration] # onnx package for the ONNXim cross-check
make rtl                   # Verilog testbenches (needs iverilog)
make -C rtl syn            # Design Compiler synthesis -> area report
```

## Quick start

```bash
# One configuration, with derived machine parameters and placements:
python main.py --system proteus --model mixtral-8x7b --batch 32 --verbose

# A baseline on the identical workload:
python main.py --system neupims --model mixtral-8x7b --batch 32

# Effectiveness-analysis variants (Sec. V-D) and Proteus-Static (Sec. V-C):
python main.py --system proteus --model mixtral-8x7b --batch 32 --variant rd
python main.py --system proteus --model mixtral-8x7b --batch 32 --static

# Perturb the analytical crossover threshold (Sec. V-E):
python main.py --system proteus --model llama3-70b --batch 64 --theta 2

# Multi-device: 64 devices = 8 tensor-parallel groups pipelined over layers
python main.py --system proteus --model llama3-405b --batch 16 --ctx 32768 \
    --devices 64 --timeline

# Sampled MoE routing; per-layer substrate occupancy:
python main.py --system proteus --model mixtral-8x7b --batch 32 \
    --routing sampled --iters 200
python main.py --system proteus --model llama3-70b --batch 32 --engine detailed

# Operator graph after the fusion/annotation passes:
python main.py --model mixtral-8x7b --dump-graph graph.json

# Generate and replay a PIM command trace on the command-level backend:
python trace_gen/gen_trace.py --kernel skinny-gemm --rows 64 --vectors 8 \
    --mode broadcast -o results/gemm.trace
python main.py --replay-trace results/gemm.trace --pim-mode broadcast
```


## Repository layout

```
configs/
  models/        llama3-70b, llama3-405b, mixtral-8x7b, switch-26b,
                 deepseek-v2-lite
  systems/       proteus + six baselines (all mechanism parameters)
  memory/        LPDDR5X-8533 organization, DRAM timing, energy/bit
  area/          post-synthesis cell areas and the process scaling
  validation/    reference points the machine model is checked against
proteus_sim/
  workload.py    model -> per-matrix / per-expert operator groups
  compiler/      operator-graph IR, lowering, fusion & AI-annotation passes
  scheduler.py   crossover estimate, placement, connectivity, instrumentation
  memory.py      LPDDR5X-PIM machine model (organization -> sustained rates)
  system.py      Proteus timing engine: topology, scheduling, energy
  fabric.py      CXL 3.0 fabric: ring collectives and pipeline timeline
  serving.py     open-loop SLO driver + closed-loop adaptation driver
  engine.py      per-layer substrate timeline of one pipeline stage
  xpucore/       tile-level systolic xPU engine (+ optional ONNX front-end)
  dram/          command-level PIM backend (banks, near-bank datapath,
                 broadcast distribution, refresh, PEs, command energy)
  baselines/     gpu, cxl_pnm, cent, neupims, papi, pimphony
trace_gen/       kernel and request trace generation CLIs
request_traces/  replayable request-arrival traces for the serving layer
pimcore/         C++17 core: memory backend, system layer, serving engine,
                 trace/host-stream generation (see its README)
ext/             external simulators (Ramulator 2.0, ONNXim; make deps)
integration/     Ramulator 2.0 patches (device model, controller plugin,
                 trace frontend); see its README
rtl/             synthesizable Verilog + testbenches + DC synthesis scripts
experiments/     per-figure scripts, microbenchmarks, cross-checks, plotting
tests/           system-level and backend test suites
docs/            DRAM model notes, API reference
```

`tests/` covers the derived organization against Table III, physical bounds,
monotonicity, OOM accounting, crossover closure and threshold sensitivity,
collective volume, frozen-mapping regression, streaming efficiency, broadcast
reuse, co-execution headroom, trace round-trip, scheduler overhead and the
compiler passes. `AE.md` maps paper claims to experiments.

## Documentation

* `docs/dram-model.md` -- the command-level backend, the near-bank datapath,
  and its cross-validation against the analytical layer.
* `docs/api.md` -- Python API reference.

## License

MIT -- see `LICENSE`.
