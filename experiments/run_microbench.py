#!/usr/bin/env python3
"""Command-level microbenchmarks (docs/dram-model.md).

Cross-validates the sustained rates used by the analytical layer against the
command-level LPDDR5X-PIM backend, and quantifies the mechanisms of Sec. IV-C
at command granularity:

  (1) sustained internal bandwidth & row-buffer locality of all-bank
      streaming (calibrates the 0.80 streaming-efficiency constant);
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


def channel_peak(mem):
    bgs = mem["dies_per_channel"] * mem["bankgroups_per_die"]
    return bgs * 32 / (mem["tCCD_L_ns"] / 2.0) * 1e9   # B/s per channel


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
          f"(configs stream_efficiency) -> deviation "
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
              f" | speedup {td/tb:5.2f}x (model: min(n,4) up to pass rounding)")
    return [rows_hdr] + out


def bench_coexec(mem):
    print("\n== (3) co-execution headroom (iso-work, n=8 vectors) ==========")
    hdr = ["mode", "kernel_ns", "busy_frac_vs_direct", "xpu_slots_frac"]
    out = [hdr]
    t_ref = None
    for mode in ["direct", "broadcast"]:
        ch = PimChannel(mem, mode)
        ch.attach_xpu_stream()
        st = ch.execute(skinny_gemm_trace(ROWS, 8, mem, mode=mode))
        t_ref = t_ref or st.time_ns
        busy = st.time_ns / t_ref
        xpu_frac = st.xpu_bursts * 32 / max(st.bytes_read, 1)
        out.append([mode, round(st.time_ns), round(busy, 3),
                    round(xpu_frac, 4)])
        print(f"  {mode:<9} kernel {st.time_ns/1e3:8.1f} us | busy vs direct "
              f"{busy:５.2f} | in-kernel xPU slot share {xpu_frac:.3f}")
    freed = 1 - out[2][1] / out[1][1]
    print(f"  -> broadcasting frees {freed*100:.0f}% of channel time for "
          f"concurrent xPU access (iso-work)")
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
