"""GPU baseline: 8x A100 DGX served by vLLM (continuous batching +
PagedAttention). Weight streaming and paged attention both run over the
aggregate HBM bandwidth; tensor-parallel AllReduce and kernel-launch
overheads are charged per iteration."""
from .base import BaselineSystem, short_factor
from ..system import Result


class GpuSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        if w.peak_mem > cfg["hbm_capacity"] * s * cfg["usable_fraction"]:
            return Result.oom(cfg["name"])
        eff = cfg["efficiency"]
        bw = cfg["hbm_bw_aggregate"] * s
        t = (w.weight_bytes / (bw * self.xw_eff(w))
             + w.kv_bytes / (bw * eff["attention"]))
        t = t * (1.0 + eff["allreduce_overhead"]) + eff["launch_overhead_ms"] * 1e-3
        t /= short_factor(w.d_model)
        return self.finish(w, t, counters=dict(
            hbm_bytes=w.weight_bytes + w.kv_bytes))

    def energy(self, res, w):
        en = self.cfg["energy"]
        bg = self.cfg["hbm_capacity"] / 1e9 * en["background_w_per_gb"] \
            * en["background_idle_factor"]
        p = self.cfg["devices"] * en["gpu_busy_w"] + en["static_w"] + bg
        dram = res.counters["hbm_bytes"] / w.batch * en["hbm_pj_per_bit"] * 8e-12
        res.power_w = p
        res.tokens_per_joule = 1.0 / (p / res.throughput + dram)
