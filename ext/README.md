# External simulators

Third-party simulators the cross-validation experiments link against. They
are not vendored and not git submodules -- nothing under `ext/` is tracked
except this file and `VERSIONS`. Fetch and build them with:

```bash
make deps                           # = bash scripts/fetch_deps.sh
```

`scripts/fetch_deps.sh` clones each upstream at the revision pinned in
`ext/VERSIONS` into the directory below and builds what it can. It is
idempotent: an already-cloned tree is left in place, so it is safe to re-run
after fixing a toolchain problem.

| Directory | Upstream | Used by |
|---|---|---|
| `ext/ramulator2` | CMU-SAFARI/ramulator2 | `experiments/run_ramulator_xcheck.py` -- replays the host-path (xPU) DRAM request stream through Ramulator 2.0's LPDDR5X model and compares sustained bandwidth against the built-in backends; `integration/ramulator2/patches/` additionally adds the PIM device model to the Ramulator tree |
| `ext/onnxim` | PSAL-POSTECH/ONNXim | `experiments/run_onnxim_xcheck.py` -- runs the decode-layer GEMMs through ONNXim's cycle-level NPU model and compares against the tile-level xpucore engine (requires the `onnx` Python package: `pip install .[integration]`) |

The paper's figures are produced by the built-in backends (`proteus_sim/`,
`pimcore/`); the two cross-checks above anchor those backends against
independent simulators.

Build notes:

* **ramulator2**: CMake >= 3.14, C++20 compiler (g++-12 or newer).
  `fetch_deps.sh` builds it and applies `integration/ramulator2/patches/`
  into a separate `ext/ramulator2-pim` tree (the stock tree stays pristine
  for the host-path cross-check).
* **ONNXim**: see its README (CMake, protobuf; a Docker image is provided
  upstream). Set `ONNXIM_HOME` if you build it outside `ext/onnxim`.
