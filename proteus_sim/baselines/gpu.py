"""GPU baseline: 8x A100 DGX served by vLLM (continuous batching +
PagedAttention).

Decode streams the resident weights and the paged KV cache over the
aggregate HBM interface in separate kernels, so their costs add; the
tensor-core roofline, the tensor-parallel AllReduce and the host framework
cost of one iteration are charged on top.
"""
from .base import BaselineSystem, host_overhead_s, short_factor
from ..system import Result


class GpuSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
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

    def energy(self, res, w):
        en = self.cfg["energy"]
        n = self.n_devices(res)
        # Board power already includes the HBM stacks, so no separate
        # background term is charged for them.
        p = n * en["gpu_busy_w"] + en["static_w"]
        dram = res.counters["hbm_bytes"] / w.batch * en["hbm_pj_per_bit"] * 8e-12
        res.power_w = p + dram * res.throughput
        res.tokens_per_joule = res.throughput / res.power_w
