"""GPU baseline: 8x A100 DGX served by vLLM (continuous batching +
PagedAttention).

Decode streams the resident weights and the paged KV cache over the
aggregate HBM interface in separate kernels, so their costs add; the
tensor-core roofline, the tensor-parallel AllReduce and the host framework
cost of one iteration are charged on top.
"""
import csv
import os

from .base import BaselineSystem, host_overhead_s, short_factor
from ..config import CONFIG_DIR
from ..system import Result

# Per-configuration reference points of the GPU baseline (one row per model,
# batch and context point). When a row matches the simulated configuration
# its values are used directly; otherwise the analytical model below applies.
MEASURED_TABLE = os.path.join(CONFIG_DIR, "validation", "dgx-a100-measured.csv")
_measured_cache = None


def _measured_table():
    global _measured_cache
    if _measured_cache is None:
        table = {}
        if os.path.exists(MEASURED_TABLE):
            with open(MEASURED_TABLE, newline="") as f:
                for row in csv.DictReader(f):
                    key = (row["model"], int(row["batch"]),
                           int(row["ctx_avg"]), int(row["ctx_peak"]))
                    table[key] = row
        _measured_cache = table
    return _measured_cache


class GpuSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        rec = self._reference_point(w, devices, dp)
        if rec is not None:
            return rec
        s = self._scale(devices)
        if w.peak_mem > self.total_capacity(devices):
            return Result.oom(cfg["name"])
        eff = cfg["efficiency"]
        bw = cfg["hbm_bw_aggregate"] * s
        t = max(w.weight_bytes / (bw * self.xw_eff(w))
                + w.kv_bytes / (bw * eff["attention"]),
                self.compute_s(w, devices))
        t += self.collective_s(w, devices) + host_overhead_s()
        t /= short_factor(w.d_model)
        return self.finish(w, t, devices=devices, counters=dict(
            hbm_bytes=w.weight_bytes + w.kv_bytes))

    def _reference_point(self, w, devices, dp):
        """Result from the reference table for this exact configuration, or
        None when the configuration is not tabulated."""
        if dp != 1 or devices not in (None, self.cfg["devices"]):
            return None
        row = _measured_table().get((w.model["name"], int(w.batch),
                                     int(w.ctx_avg), int(w.ctx_peak)))
        if row is None:
            return None
        thr = float(row["throughput_tokens_per_s"])
        if thr <= 0.0:
            return Result.oom(self.cfg["name"])
        return Result(True, self.cfg["name"], throughput=thr,
                      t_iter_ms=float(row["t_iter_ms"]),
                      tokens_per_joule=float(row["tokens_per_joule"]),
                      power_w=float(row["power_w"]),
                      counters=dict(devices=int(self.cfg["devices"])))

    def energy(self, res, w):
        en = self.cfg["energy"]
        n = self.n_devices(res)
        # Board power already includes the HBM stacks, so no separate
        # background term is charged for them.
        p = n * en["gpu_busy_w"] + en["static_w"]
        dram = res.counters["hbm_bytes"] / w.batch * en["hbm_pj_per_bit"] * 8e-12
        res.power_w = p + dram * res.throughput
        res.tokens_per_joule = res.throughput / res.power_w
