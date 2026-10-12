# Ramulator 2.0 integration

Reference patches hooking the ProteusSim PIM device model into
[Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2). They document
how the backend's command set and arbitration map onto Ramulator's
interfaces:

| Patch | Adds |
|---|---|
| `0001-add-lpddr5x-pim-device-model.patch` | `src/dram/impl/LPDDR5X-PIM.cpp` -- the LPDDR5X-8533 device with the all-bank PIM commands (ACTab/PREab, RDMAC/RDMACb, WRab, MODE) and the per-channel datapath mode register |
| `0002-add-pim-priority-controller-plugin.patch` | `src/dram_controller/impl/plugin/proteus_pim_sched.cpp` -- PIM-priority scheduling, host gap-stealing, FIFO-drain gating of MODE |
| `0003-add-proteus-trace-replay-frontend.patch` | `src/frontend/impl/proteus_trace_replay.cpp` + `example_config/proteus-lpddr5x-pim.yaml` -- replays `trace_gen/gen_trace.py` traces |

## Applying

The patches only add new files, so they apply cleanly to any Ramulator 2.0
checkout:

```bash
git clone https://github.com/CMU-SAFARI/ramulator2
cd ramulator2
git apply /path/to/proteus-sim/integration/ramulator2/patches/*.patch
# then add the three .cpp files to the build (they are picked up
# automatically if the tree globs src/**/*.cpp; otherwise list them in
# CMakeLists.txt) and rebuild.
python3 /path/to/proteus-sim/trace_gen/gen_trace.py --kernel skinny-gemm \
    --rows 64 --vectors 8 --mode broadcast -o gemm.trace
./ramulator2 -f example_config/proteus-lpddr5x-pim.yaml
```

`patch -p1 < patches/....patch` works equally.

## Trace replay

`ProteusTraceReplay` reads the command traces of `proteus_sim/dram/trace.py`
as `trace_gen/gen_trace.py` writes them -- `MODE <direct|broadcast>`,
`ACT_AB <row>`, `RDMAC_AB <row> <col>`, `WR_AB <row> <col>`, `PRE_AB`, the
single-bank host commands `ACT`/`RD`/`PRE <row> <col> <bank>`, and
`BARRIER`, with `#` comment lines -- and reports the offending line number
on anything else. Column commands become the `pim-mac`, `pim-mac-broadcast`,
`pim-write`, `pim-mode` and `read` request types of the device model;
RDMAC_AB picks its broadcast variant from the mode register the preceding
MODE line set. Row control is left to the device model, whose prerequisite
chain issues ACTab/PREab on its own, so the row commands of the trace are
consumed as bookkeeping. The frontend's `bankgroups` and `banks_per_group`
parameters split the channel-global bank id of a host command and must match
`MemorySystem.DRAM.org`.

## Scope and caveats

None of the paper's results depend on this integration: every figure is
produced by the self-contained backends in `proteus_sim/dram/` and
`pimcore/`, which model the same command set, timing state, and arbitration
and are cross-validated against each other (`experiments/run_substrate.py`,
`docs/dram-model.md`). The patches are provided for users who want the PIM
device inside Ramulator's controller/frontend ecosystem.

The code targets the Ramulator 2.0 implementation-registry API
(`RAMULATOR_REGISTER_IMPLEMENTATION`, `ImplDef`/`ImplLUT` tables,
controller plugins). Ramulator evolves; if the interface has drifted since
the patches were written, the adjustments needed are mechanical (the tables
and the plugin logic are self-contained).
