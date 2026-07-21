# API Reference

Programmatic entry points of the Python layer (the C++ core mirrors these;
see `pimcore/README.md`).

## Configuration (`proteus_sim.config`)

```python
load_model(name)    # configs/models/<name>.yaml   -> dict-like Cfg
load_system(name)   # configs/systems/<name>.yaml  -> Cfg
load_memory(name)   # configs/memory/<name>.yaml   -> Cfg
list_models(); list_systems()
```

## Workload (`proteus_sim.workload`)

```python
w = build_workload(model_cfg, batch, ctx_in, ctx_out,
                   routing="expected"|"sampled", seed=7, ctx_override=None)
# aggregates: w.weight_bytes, w.weight_flops, w.kv_bytes, w.attn_flops,
#             w.tokens_per_expert, w.active_experts, w.peak_mem
# per-layer operator list: w.operators  (Operator: kind, bytes, flops, AI)
```

## Systems (`proteus_sim.system`)

```python
sys_ = build_system("proteus" | "dgx-a100" | ... , features=VARIANTS["rd"])
r = sys_.simulate(w, devices=None, dp=1)     # -> Result
# r.alive, r.throughput (tokens/s), r.t_iter_ms, r.tokens_per_joule,
# r.power_w, r.counters, r.placements, r.notes ("x*=..., m=...")
VARIANTS  # {"base","as","rd","of","ec","full"} -> feature frozensets
```

## Scheduler (`proteus_sim.scheduler`)

```python
sched = CrossoverScheduler(sys_cfg, derived_memory, xpu_flops, xpu_bw)
sched.place(op)              # -> Placement(substrate, pim_mode, ai, region)
sched.broadcast_passes(g)    # ceil(g / fanout)
```

## Compiler (`proteus_sim.compiler`)

```python
g = lower_model(model_cfg, n_layers=None)    # -> OpGraph
run_default_pipeline(g)      # fuse element-wise chains + annotate AI exprs
g.to_json(path); g.stats()
```

## Command-level backend (`proteus_sim.dram`)

```python
ch = PimChannel(mem_cfg, mode="direct"|"broadcast")
ch.attach_xpu_stream()                       # opportunistic host slots
stats = ch.execute(commands)                 # ChannelStats
CommandEnergy(mem_cfg).account(stats, external=False)   # EnergyReport
# traces: proteus_sim.dram.trace.read_trace / write_trace
```

Kernel generators (`trace_gen`): `gemv_trace(rows, mem)`,
`skinny_gemm_trace(rows, n_vectors, mem, mode=...)`,
`attention_trace(ctx, kv_bytes_tok_layer, group, mem, mode=...,
kv_append=...)`, `layout_rows_per_bank(bytes, mem, channels=None)`.

## C++ bridge (`proteus_sim.dram.pimcore_bridge`)

```python
build()                       # cmake+make on demand -> binary path
run_kernel(config="lpddr5x-8533", kernel="gemv", rows=64, vectors=1,
           group=1, mode="direct", host_gbps=None, external_energy=False)
run_tests()                   # executes the pimcore C++ suite
```

## xPU core (`proteus_sim.xpucore`)

```python
eng = XpuEngine(SystolicConfig())            # 312 TFLOPS A100-class default
t = eng.run_op(name, M, K, N, dram_bw)       # OpTiming: tile, compute/memory ns
eng.run_graph(op_graph, batch, tpe, dram_bw)
load_onnx_graph(path)                        # optional `onnx` dependency
```

## Serving (`proteus_sim.serving`)

```python
sim = ServingSimulator(system, model_cfg, max_batch=32,
                       prompt_mean=2048, out_mean=256, seed=13)
sim = ServingSimulator.from_trace(system, model_cfg,
                                  "request_traces/pool256.txt")
recs = sim.run(600)          # IterationRecord: batch, mean_ctx, throughput,
                             # x_split, expert placements, mode switches
```

## External simulator bridges

```python
from proteus_sim.dram import ramulator_bridge
ramulator_bridge.find_binary()               # ext/ramulator2 or RAMULATOR2_BIN
ramulator_bridge.run_stream({"rw": path})    # -> bw_gbps, cycles, stats

from proteus_sim.xpucore import onnxim_bridge
onnxim_bridge.run_gemm(m, k, n)              # -> ONNXim cycles (needs onnx)
```

Both raise `*Unavailable` with fetch/build instructions when the external
simulator is absent; the cross-check experiments catch this and report
SKIPPED. The C++ serving engine (`pimcore_serve`) and trace generator
(`pimcore_tracegen`) are driven via subprocess (see `pimcore/README.md`).

## Fabric and detailed engine

```python
CxlFabric(ic_cfg).iteration_timeline(stages, t_stage_ns, act_bytes)
DetailedEngine(system).build(w, result)      # per-layer substrate timeline
```
