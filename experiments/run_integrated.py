#!/usr/bin/env python3
"""Integrated xPU + PIM co-simulation at the shared memory interface
(Sec. V-A: "xPU accesses and PIM commands share the LPDDR channels").

For one decode layer of Llama-3.1-70B at batch 32 on one Proteus device:

  1. the compiler lowers the layer; the crossover scheduler assigns
     attention to PIM (broadcast) and splits the weight GEMMs by x*.
     Every device of the tensor-parallel group owns a 1/N shard of the
     layer, its KV heads included (Sec. IV-B);
  2. XpuCore tiles the xPU-side share on the systolic arrays and lowers its
     DRAM traffic to a host-request stream;
  3. the command-level backend executes the PIM attention stream while the
     host controller schedules the xPU stream onto the same channels
     (PIM-priority arbitration);
  4. the co-simulated layer time is compared against the analytical
     stage-time share of the aggregate engine.
"""
import math
import os

from common import RESULTS, write_csv
from proteus_sim import load_model, build_system
from proteus_sim.workload import build_workload
from proteus_sim.xpucore import SystolicConfig, XpuEngine
from proteus_sim.config import load_memory
from proteus_sim.dram import PimChannel
from trace_gen import layout_rows_per_bank, skinny_gemm_trace

MODEL = "llama3-70b"
BATCH = 32


def main():
    model = load_model(MODEL)
    sys_ = build_system("proteus")
    w = build_workload(model, BATCH, 2048, 6144)
    r = sys_.simulate(w)
    # Fraction of the layer's single-pass weight operands the scheduler put
    # on the xPU; the complement is streamed by the near-bank PEs.
    x = r.counters["xpu_weight_share"]
    n_tp = r.counters["tp_width"]
    layers = r.counters["layers_per_stage"]

    # ---- xPU side: tile the x* weight share of one layer ------------- #
    eng = XpuEngine(SystolicConfig())
    print("xPU core:", eng.cfg.describe())
    d = model["d_model"]
    dff = model["d_ffn"]
    # one layer's weight GEMMs (qkv, out, up/gate, down), x* share on xPU
    gemms = [("qkv_proj", BATCH, d, d + 2 * model["n_kv_heads"]
              * model["d_head"]),
             ("out_proj", BATCH, d, d),
             ("ffn_up", BATCH, d, 2 * dff),
             ("ffn_down", BATCH, dff, d)]
    xbw = sys_.xpu_bw
    xpu_ns = 0.0
    xpu_bytes = 0.0
    rows_t = [["op", "tile", "compute_us", "memory_us", "bound", "util"]]
    for name, m, k, n in gemms:
        # every device of the tensor-parallel group owns a 1/n_tp shard
        t = eng.run_op(name, m, k, max(n // n_tp, 1), dram_bw=xbw)
        xpu_ns += t.time_ns * x          # x* work share of the operator
        xpu_bytes += t.schedule.dram_bytes * x
        rows_t.append([name,
                       f"{t.schedule.tm}x{t.schedule.tk}x{t.schedule.tn}",
                       round(t.compute_ns / 1e3, 2),
                       round(t.memory_ns / 1e3, 2),
                       "memory" if t.memory_bound else "compute",
                       round(t.utilization, 3)])
    write_csv(os.path.join(RESULTS, "integrated_xpu_tiles.csv"),
              rows_t[0], rows_t[1:])

    # ---- PIM side: one layer's attention + (1-x*) weights, command level
    # The layer's KV slice (batch x ctx x per-layer KV bytes) and its
    # (1-x*) weight share are striped over the owning device's 64 channels.
    mem = load_memory("lpddr5x-8533")
    from proteus_sim.dram.commands import Command
    from proteus_sim.dram.commands import MODE
    kv_layer_bytes = BATCH * w.ctx_avg * model["kv_bytes_per_token"] \
        / model["n_layers"] / n_tp
    kv_rows = max(1, layout_rows_per_bank(kv_layer_bytes, mem))
    import trace_gen.kernels as tk
    passes = math.ceil(w.attn_reuse / 4)
    cmds = [Command(MODE, arg="broadcast")]
    for _ in range(passes):
        cmds += tk._stream_rows(kv_rows, mem["row_bytes"] // 32, 0)
    from proteus_sim.dram.commands import WR_AB, PRE_AB
    cmds.append(Command(WR_AB, row=kv_rows, col=0))     # in-place KV append
    cmds.append(Command(PRE_AB))
    wt_layer = (1 - x) * w.weight_bytes / model["n_layers"] / n_tp
    wt_rows = max(1, layout_rows_per_bank(wt_layer, mem))
    cmds += skinny_gemm_trace(wt_rows, min(BATCH, 32), mem,
                              mode="broadcast", set_mode=False)
    ch = PimChannel(mem, "broadcast")
    ch.attach_xpu_stream()
    st = ch.execute(cmds)

    # ---- closure ----------------------------------------------------- #
    t_stage_ms = r.t_iter_ms / r.counters["pipeline_groups"]
    t_layer_analytical = t_stage_ms / layers * 1e6      # ns
    t_layer_cosim = max(st.time_ns, xpu_ns)
    dev = abs(t_layer_cosim / t_layer_analytical - 1)

    print(f"\nintegrated co-simulation, one decode layer "
          f"({MODEL}, b={BATCH}, x*={x:.2f}):")
    print(f"  xPU tile schedule     : {xpu_ns/1e3:8.1f} us "
          f"({xpu_bytes/1e6:.1f} MB DRAM traffic, x* work share)")
    print(f"  PIM command stream    : {st.time_ns/1e3:8.1f} us "
          f"({st.n_rd_burst} bursts, {st.n_mode_switch} mode switches, "
          f"KV append {st.n_wr_burst} writes)")
    print(f"  co-simulated layer    : {t_layer_cosim/1e3:8.1f} us")
    print(f"  analytical layer share: {t_layer_analytical/1e3:8.1f} us "
          f"(deviation {dev*100:.1f}%)")
    write_csv(os.path.join(RESULTS, "integrated_closure.csv"),
              ["quantity", "value"],
              [["x_split", round(x, 3)],
               ["xpu_us", round(xpu_ns / 1e3, 2)],
               ["pim_us", round(st.time_ns / 1e3, 2)],
               ["cosim_layer_us", round(t_layer_cosim / 1e3, 2)],
               ["analytical_layer_us", round(t_layer_analytical / 1e3, 2)],
               ["deviation_pct", round(dev * 100, 2)]])
    if dev > 0.25:
        print("  WARNING: closure deviation above 25%")


if __name__ == "__main__":
    main()
