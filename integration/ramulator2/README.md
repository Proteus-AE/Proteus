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
./ramulator2 -f example_config/proteus-lpddr5x-pim.yaml
```

`patch -p1 < patches/....patch` works equally.

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
