"""Tile-level systolic-array timing model.

Models an A100-class matrix engine as ``n_arrays`` weight-stationary systolic
arrays of ``rows x cols`` MACs with double-buffered SRAM operand staging:

  * a GEMM (M x K) . (K x N) is tiled into (Tm, Tk, Tn) blocks sized to the
    array and the SRAM budget;
  * each tile costs ``Tk + rows + cols`` cycles to stream through the array
    (fill + drain), overlapped across tiles by the double buffer;
  * weight tiles are fetched from DRAM once per (Tk, Tn) block and reused
    across the M dimension; activation tiles stream per (Tm, Tk) block;
  * per-op time is max(compute, DRAM traffic / available bandwidth) --
    the available bandwidth is what the shared LPDDR channels grant the
    xPU, which is where the PIM co-execution couples in (Sec. IV-B).

The engine reports per-op compute cycles, SRAM traffic, DRAM traffic, and
utilization, and can emit its DRAM accesses as a host-request stream for
the command-level backend (``experiments/run_integrated.py``).
"""
import math
from dataclasses import dataclass


@dataclass
class SystolicConfig:
    rows: int = 32                   # per-SM fused tensor-core tile
    cols: int = 32
    n_arrays: int = 108              # SM count (A100-class -> 312 TFLOPS)
    freq_ghz: float = 1.41
    sram_bytes: int = 192 * 1024     # per-array operand SRAM (double-buffered)
    dtype_bytes: int = 2             # FP16

    @property
    def peak_flops(self):
        return self.rows * self.cols * 2.0 * self.n_arrays * self.freq_ghz * 1e9

    def describe(self):
        return (f"{self.n_arrays} x {self.rows}x{self.cols} systolic arrays "
                f"@ {self.freq_ghz} GHz -> {self.peak_flops/1e12:.0f} TFLOPS "
                f"(FP16)")


@dataclass
class TileSchedule:
    """Tiling decision and cost for one GEMM operator."""
    op_name: str
    m: int
    k: int
    n: int
    tm: int
    tk: int
    tn: int
    n_tiles: int = 0
    compute_cycles: float = 0.0
    weight_dram_bytes: float = 0.0
    act_dram_bytes: float = 0.0
    out_dram_bytes: float = 0.0

    @property
    def dram_bytes(self):
        return self.weight_dram_bytes + self.act_dram_bytes + self.out_dram_bytes

    @property
    def flops(self):
        return 2.0 * self.m * self.k * self.n


@dataclass
class OpTiming:
    schedule: TileSchedule
    compute_ns: float
    memory_ns: float

    @property
    def time_ns(self):
        return max(self.compute_ns, self.memory_ns)

    @property
    def utilization(self):
        return self.compute_ns / self.time_ns if self.time_ns else 0.0

    @property
    def memory_bound(self):
        return self.memory_ns > self.compute_ns


class XpuEngine:
    """Schedules operator-graph GEMMs onto the systolic arrays."""

    def __init__(self, cfg: SystolicConfig = None):
        self.cfg = cfg or SystolicConfig()

    # ------------------------------------------------------------------ #
    def tile(self, name, m, k, n) -> TileSchedule:
        """Choose a (Tm, Tk, Tn) tiling under the SRAM budget.

        Weight-stationary: a (Tk x Tn) weight tile is pinned in the array
        (Tk <= rows, Tn <= cols); activation (Tm x Tk) and output (Tm x Tn)
        tiles double-buffer through the SRAM.
        """
        c = self.cfg
        tk = min(k, c.rows)
        tn = min(n, c.cols)
        # activation tile height bounded by half the SRAM (double buffer)
        budget = c.sram_bytes // 2
        tm = max(1, min(m, budget // max((tk + tn) * c.dtype_bytes, 1)))
        s = TileSchedule(op_name=name, m=m, k=k, n=n, tm=tm, tk=tk, tn=tn)

        n_k = math.ceil(k / tk)
        n_n = math.ceil(n / tn)
        n_m = math.ceil(m / tm)
        s.n_tiles = n_k * n_n * n_m
        # fill+drain amortized once per (k,n) block thanks to double buffering
        s.compute_cycles = (n_k * n_n *
                            (n_m * (tm + tk) + c.rows + c.cols))
        s.weight_dram_bytes = float(k * n * c.dtype_bytes)         # once
        s.act_dram_bytes = float(n_k * m * tk * c.dtype_bytes)     # per k-block
        s.out_dram_bytes = float(m * n * c.dtype_bytes)
        return s

    # ------------------------------------------------------------------ #
    def run_op(self, name, m, k, n, dram_bw, parallel_arrays=None) -> OpTiming:
        """Time one GEMM given the DRAM bandwidth currently granted to the
        xPU by the shared memory interface."""
        c = self.cfg
        s = self.tile(name, m, k, n)
        arrays = parallel_arrays or c.n_arrays
        compute_ns = s.compute_cycles / arrays / c.freq_ghz
        memory_ns = s.dram_bytes / dram_bw * 1e9 if dram_bw > 0 else float("inf")
        return OpTiming(schedule=s, compute_ns=compute_ns, memory_ns=memory_ns)

    # ------------------------------------------------------------------ #
    def run_graph(self, graph, batch, tokens_per_expert, dram_bw):
        """Time every weight-GEMM node of a compiled operator graph.

        Symbolic dims are instantiated from the runtime values, mirroring the
        per-iteration expression instantiation of Sec. IV-E.
        """
        timings = []
        for node in graph.iter_nodes():
            if node.kind != "weight_gemm":
                continue
            m = node.shape.get("m")
            m = batch if m == "b" else (
                max(int(round(tokens_per_expert)), 1) if m == "tpe" else int(m))
            k = int(node.shape["k"])
            n = int(node.shape["n"])
            timings.append(self.run_op(node.name, m, k, n, dram_bw))
        return timings

    # ------------------------------------------------------------------ #
    def emit_host_requests(self, timing: OpTiming, bursts_per_req=2):
        """Lower one op's DRAM traffic to a host-request count for the
        command-level backend (64 B requests by default)."""
        req_bytes = bursts_per_req * 32
        return int(math.ceil(timing.schedule.dram_bytes / req_bytes))
