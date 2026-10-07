"""Detailed per-layer execution engine.

Expands the block-aggregate schedule of `system.py` into an explicit
per-layer discrete-event timeline: every transformer block a pipeline group
owns contributes its operator groups on the substrate the crossover
scheduler selected, its tensor-parallel reduction, and its fused-SFU
interval, laid out on the three substrate tracks of one device. The engine
replays the same placement decisions the aggregate engine made, so its
intervals reproduce the stage time; what it exposes is what the aggregate
view hides: per-layer occupancy, substrate utilization, and the interleaving
pattern of xPU and PIM work within one iteration.

Used by `main.py --engine detailed` and by the timeline JSON export.
"""
import json
from dataclasses import dataclass, field

from .scheduler import SchedulerCounters, small_op_efficiency
from .system import ProteusSystem
from .workload import Workload


@dataclass
class Interval:
    track: str        # 'xpu' | 'pim' | 'sfu'
    layer: int
    op: str
    start_ms: float
    end_ms: float

    @property
    def dur_ms(self):
        return self.end_ms - self.start_ms


@dataclass
class LayerTimeline:
    intervals: list = field(default_factory=list)
    t_stage_ms: float = 0.0
    utilization: dict = field(default_factory=dict)

    def to_json(self, path=None):
        doc = {
            "t_stage_ms": self.t_stage_ms,
            "utilization": self.utilization,
            "intervals": [{
                "track": iv.track, "layer": iv.layer, "op": iv.op,
                "start_ms": round(iv.start_ms, 6),
                "end_ms": round(iv.end_ms, 6),
            } for iv in self.intervals],
        }
        s = json.dumps(doc, indent=1)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    def render(self, width=72):
        """ASCII per-track occupancy over the stage window."""
        if not self.intervals:
            return "(empty timeline)"
        t_end = max(self.t_stage_ms, max(iv.end_ms for iv in self.intervals))
        lines = []
        for track in ("xpu", "pim", "sfu"):
            row = [" "] * width
            for iv in self.intervals:
                if iv.track != track:
                    continue
                i0 = int(iv.start_ms / t_end * (width - 1))
                i1 = max(i0 + 1, int(iv.end_ms / t_end * (width - 1)))
                ch = "#" if track == "xpu" else ("=" if track == "pim" else "~")
                for i in range(i0, min(i1, width)):
                    row[i] = ch
            util = self.utilization.get(track, 0.0)
            lines.append(f"  {track:<4}|{''.join(row)}| {util*100:5.1f}%")
        lines.append(f"      0{' ' * (width - 10)}{t_end:8.3f} ms")
        return "\n".join(lines)


class DetailedEngine:
    """Builds the per-layer timeline consistent with a simulated result."""

    def __init__(self, system: ProteusSystem):
        self.sys = system

    def build(self, w: Workload, result) -> LayerTimeline:
        c = result.counters
        sys_ = self.sys
        groups = c["pipeline_groups"]
        n_tp = c["tp_width"]
        layers = c["layers_per_stage"]
        t_stage = result.t_iter_ms / groups

        f = sys_.features
        shard = 1.0 / n_tp
        smallf = small_op_efficiency(
            sys_.cfg["scheduler"]["small_op_efficiency"],
            max(w.tokens_per_expert, 1.0))
        compute = [op for op in w.block if op.kind != "elementwise"]
        costs = [sys_._op_cost(op, shard, sys_.dmem.internal_bw,
                               sys_.dmem.broadcast_bw, sys_.dmem.pe_flops,
                               sys_.xpu_bw, smallf, f) for op in compute]
        _, _, _, chosen = sys_._schedule_block(costs, f, SchedulerCounters())

        # The block's tensor-parallel reduction serializes with the layer it
        # reduces and is charged on the PIM/fabric track.
        t_coll_l = sys_.fabric.tp_allreduce_ns(
            w.activation_bytes(), n_tp,
            int(sys_.cfg["parallelism"]["tp_collectives_per_layer"])) / 1e6

        tl = LayerTimeline(t_stage_ms=t_stage)
        t_x = 0.0
        t_p = 0.0
        for layer in range(layers):
            for cost, sub, frac in chosen:
                dur = cost.t_on(sub) * frac * 1e3
                if dur <= 0:
                    continue
                if sub == "xpu":
                    tl.intervals.append(Interval("xpu", layer, cost.name,
                                                 t_x, t_x + dur))
                    t_x += dur
                else:
                    tl.intervals.append(Interval("pim", layer, cost.name,
                                                 t_p, t_p + dur))
                    t_p += dur
            if t_coll_l:
                tl.intervals.append(Interval("pim", layer, "tp-allreduce",
                                             t_p, t_p + t_coll_l))
                t_p += t_coll_l
            # Fused nonlinear tails stream on the SFUs behind the PIM pipeline.
            t_sfu = 0.08 * t_coll_l if t_coll_l else 0.0
            tl.intervals.append(Interval("sfu", layer, "fused-nonlinear",
                                         max(t_p - t_sfu, 0.0), t_p))

        for track in ("xpu", "pim", "sfu"):
            busy = sum(iv.dur_ms for iv in tl.intervals if iv.track == track)
            tl.utilization[track] = min(busy / t_stage, 1.0) if t_stage else 0.0
        return tl
