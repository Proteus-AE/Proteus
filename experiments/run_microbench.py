#!/usr/bin/env python3
"""Command-level microbenchmarks (docs/dram-model.md).

Cross-validates the sustained rates used by the analytical layer against the
command-level LPDDR5X-PIM backend, and quantifies the mechanisms of Sec. IV-C
at command granularity:

  (1) sustained internal bandwidth & row-buffer locality of all-bank
      streaming, against the closed form of proteus_sim/memory.py;
  (2) effective operand reuse of broadcasting vs. direct connectivity for
      skinny-GEMMs of 1..32 concurrent vectors (the ceil(n/4) pass model);
  (3) xPU co-execution headroom: channel-busy fraction for an iso-work
      kernel under each mode, i.e. the memory-service slots broadcasting
      returns to the xPU (Sec. IV-B unified memory path);
  (4) command-based energy per byte, near-bank vs. external termination
      (the 2.2 vs 4.5 pJ/bit split used by the energy model).
"""
import os

from common import RESULTS, write_csv
from proteus_sim.config import load_memory
from proteus_sim.dram import PimChannel, CommandEnergy
from proteus_sim.memory import derive
from trace_gen import gemv_trace, skinny_gemm_trace, attention_trace

MEM = "lpddr5x-8533"
ROWS = 96          # rows per bank streamed per pass (steady state)


def channel_peak(mem, mode="direct"):
    """All-bank peak read bandwidth of one channel in a connectivity mode."""
    return PimChannel(mem, mode=mode).peak_bw(mode)


def bench_streaming(mem):
    print("== (1) all-bank streaming: sustained bandwidth =================")
    ch = PimChannel(mem, mode="direct")
    stats = ch.execute(gemv_trace(ROWS, mem))
    peak = channel_peak(mem)
    eff = stats.sustained_bw() / peak
    dmem = derive(mem)
    print(f"  peak {peak/1e9:.1f} GB/s/ch | sustained {stats.sustained_bw()/1e9:.1f}"
          f" GB/s/ch | efficiency {eff:.3f}")
    print(f"  row-hit rate {stats.row_hits/stats.n_rd_burst:.3f} | "
          f"refresh events {stats.n_refresh}")
    print(f"  analytical layer uses {dmem.internal_eff:.2f} "
          f"(closed form) -> deviation "
          f"{abs(eff/dmem.internal_eff-1)*100:.1f}%")
    return [["allbank_stream", f"{stats.sustained_bw()/1e9:.2f}",
             f"{eff:.4f}", f"{stats.row_hits/stats.n_rd_burst:.4f}"]], eff


def bench_broadcast(mem):
    print("\n== (2) operand reuse: direct vs broadcasting ==================")
    rows_hdr = ["n_vectors", "direct_ns", "broadcast_ns", "speedup",
                "effective_reuse"]
    out = []
    for n in [1, 2, 4, 8, 16, 32]:
        td = PimChannel(mem, "direct").execute(
            skinny_gemm_trace(ROWS, n, mem, mode="direct")).time_ns
        tb = PimChannel(mem, "broadcast").execute(
            skinny_gemm_trace(ROWS, n, mem, mode="broadcast")).time_ns
        reuse = td / tb
        out.append([n, round(td), round(tb), round(td / tb, 2),
                    round(reuse, 2)])
        print(f"  n={n:<3} direct {td/1e3:8.1f} us | broadcast {tb/1e3:8.1f} us"
              f" | speedup {td/tb:5.2f}x")
    return [rows_hdr] + out


def bench_coexec(mem):
    """Concurrent host bandwidth each connectivity mode leaves free.

    The analytical layer derives this from the column slots the all-bank
    stream does not use while it is streaming (`memory.coexec_bw`). Here the
    event-driven backend counts the slots an attached host stream actually
    claims over the whole kernel, which additionally includes the
    command-bus and refresh time during which the bank column ports are idle
    as well -- so the closed form is the conservative bracket of the two.
    Both agree exactly that direct mode returns nothing."""
    print("\n== (3) co-execution headroom (iso-work, n=8 vectors) ==========")
    hdr = ["mode", "kernel_ns", "host_bursts_per_pim_burst",
           "closed_form", "dev_pct"]
    out = [hdr]
    for mode in ["direct", "broadcast"]:
        ch = PimChannel(mem, mode)
        ch.attach_xpu_stream()
        st = ch.execute(skinny_gemm_trace(ROWS, 8, mem, mode=mode))
        measured = st.xpu_bursts / max(st.n_rd_burst, 1)
        analytical = ch.host_slot_bursts(mode) / ch.n_banks
        dev = abs(measured / analytical - 1) * 100 if analytical else 0.0
        out.append([mode, round(st.time_ns), round(measured, 4),
                    round(analytical, 4), round(dev, 1)])
        print(f"  {mode:<9} kernel {st.time_ns/1e3:8.1f} us | host bursts per "
              f"PIM burst {measured:.4f} measured vs {analytical:.4f} in "
              f"closed form (+{dev:.0f}% of command-bus and refresh time)")
    dmem = derive(mem)
    print(f"  -> direct mode returns no memory-service slots; broadcasting "
          f"returns {dmem.coexec_broadcast/1e12:.2f} TB/s per device, more "
          f"than the {dmem.external_bw/1e12:.2f} TB/s external interface can "
          f"absorb")
    return out


def bench_energy(mem):
    print("\n== (4) command-based energy =====================================")
    en = CommandEnergy(mem)
    ch = PimChannel(mem, "direct")
    st = ch.execute(gemv_trace(ROWS, mem))
    near = en.account(st, external=False)
    ext = en.account(st, external=True)
    pnb = near.pj_per_byte(st.bytes_read) / 8
    pex = ext.pj_per_byte(st.bytes_read) / 8
    print(f"  near-bank termination : {pnb:.2f} pJ/bit")
    print(f"  external termination  : {pex:.2f} pJ/bit")
    print(f"  energy model uses 2.2 / 4.5 pJ/bit "
          f"(dev {abs(pnb/2.2-1)*100:.0f}% / {abs(pex/4.5-1)*100:.0f}%)")
    print(near.describe())
    return [["termination", "pj_per_bit"],
            ["near_bank", round(pnb, 3)], ["external", round(pex, 3)]]


def bench_attention(mem):
    print("\n== (5) GQA attention: per-mode KV re-streaming =================")
    hdr = ["group_size", "direct_ns", "broadcast_ns", "speedup"]
    out = [hdr]
    for g in [1, 4, 8, 16]:
        td = PimChannel(mem, "direct").execute(
            attention_trace(4096, 4096, g, mem, mode="direct")).time_ns
        tb = PimChannel(mem, "broadcast").execute(
            attention_trace(4096, 4096, g, mem, mode="broadcast")).time_ns
        out.append([g, round(td), round(tb), round(td / tb, 2)])
        print(f"  g={g:<3} direct {td/1e3:8.1f} us | broadcast {tb/1e3:8.1f} us"
              f" | speedup {td/tb:5.2f}x")
    return out


def main():
    mem = load_memory(MEM)
    stream_rows, eff = bench_streaming(mem)
    bc = bench_broadcast(mem)
    co = bench_coexec(mem)
    enr = bench_energy(mem)
    att = bench_attention(mem)
    write_csv(os.path.join(RESULTS, "microbench_streaming.csv"),
              ["kernel", "sustained_gbps_per_ch", "efficiency", "row_hit"],
              stream_rows)
    write_csv(os.path.join(RESULTS, "microbench_broadcast.csv"), bc[0], bc[1:])
    write_csv(os.path.join(RESULTS, "microbench_coexec.csv"), co[0], co[1:])
    write_csv(os.path.join(RESULTS, "microbench_energy.csv"), enr[0], enr[1:])
    write_csv(os.path.join(RESULTS, "microbench_attention.csv"), att[0], att[1:])


if __name__ == "__main__":
    main()
