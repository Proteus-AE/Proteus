"""CENT baseline: GPU-free system scaling CXL-attached GDDR6-AiM devices.

Near-bank GEMV execution without inter-PE operand sharing:
  * shared-operand GEMMs decompose into per-vector GEMVs, so active weights
    are re-streamed once per routed token (traffic x tokens/expert);
  * GQA/MLA attention re-reads the shared KV once per query group (x g);
  * the host-driven command stream adds a per-iteration overhead, and very
    short per-expert operand chains reduce streaming efficiency."""
from .base import BaselineSystem, short_factor
from ..system import Result


class CentSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        if w.peak_mem > cfg["devices"] * s * cfg["capacity_per_device"] * 0.90:
            return Result.oom(cfg["name"])
        bw = cfg["devices"] * s * cfg["internal_bw_per_device"] \
            * cfg["stream_efficiency"] * self.smallf(w)
        tpe = max(w.tokens_per_expert, 1.0)
        traffic = w.weight_bytes * tpe + w.kv_bytes * w.attn_reuse
        t = traffic / bw + cfg["host_cmd_overhead_ms"] * 1e-3
        t /= short_factor(w.d_model)
        return self.finish(w, t, counters=dict(gddr_int_bytes=traffic))

    def energy(self, res, w):
        en = self.cfg["energy"]
        cap_gb = self.cfg["devices"] * self.cfg["capacity_per_device"] / 1e9
        bg = cap_gb * en["background_w_per_gb"] * en["background_idle_factor"]
        p = self.cfg["devices"] * en["device_w"] + en["static_w"] + bg
        dram = res.counters["gddr_int_bytes"] / w.batch * en["gddr_int_pj_per_bit"] * 8e-12
        res.power_w = p
        res.tokens_per_joule = 1.0 / (p / res.throughput + dram)
