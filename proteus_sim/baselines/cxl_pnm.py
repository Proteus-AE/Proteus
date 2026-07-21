"""CXL-PNM baseline: LPDDR-based CXL memory expander with channel-level
near-memory processing. All operators stream through the channel-level PNM
engines at external-class bandwidth (no bank-level operand visibility), with
the conservatively strengthened controller-side compute ceiling."""
from .base import BaselineSystem, short_factor


class CxlPnmSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        # 4 TB LPDDR capacity: never the binding constraint in our sweeps.
        bw = cfg["devices"] * s * cfg["bw_per_device"] * cfg["stream_efficiency"]
        fl = cfg["devices"] * s * cfg["flops_per_device"] * cfg["compute_efficiency"]
        bytes_total = w.weight_bytes + w.kv_bytes
        flops_total = w.weight_flops + w.attn_flops
        t = max(bytes_total / bw, flops_total / fl) + cfg["launch_overhead_ms"] * 1e-3
        t /= short_factor(w.d_model)
        return self.finish(w, t, counters=dict(
            lpddr_bytes=bytes_total, engine_duty=min((flops_total / fl) / t, 1.0)))

    def energy(self, res, w):
        en = self.cfg["energy"]
        cap_gb = self.cfg["devices"] * self.cfg["capacity_per_device"] / 1e9
        bg = cap_gb * en["background_w_per_gb"] * en["background_idle_factor"]
        p = self.cfg["devices"] * (en["controller_w_per_device"]
                                   + en["engine_full_load_w"] * res.counters["engine_duty"]) \
            + en["static_w"] + bg
        dram = res.counters["lpddr_bytes"] / w.batch * en["lpddr_ext_pj_per_bit"] * 8e-12
        res.power_w = p
        res.tokens_per_joule = 1.0 / (p / res.throughput + dram)
