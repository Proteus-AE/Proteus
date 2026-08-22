"""PIMphony baseline: orchestration system for PIM-based long-context
inference on AiMX-class devices.

Modelled as NeuPIMs-style heterogeneous execution plus PIMphony's three
techniques: Token-Centric PIM Partitioning restores channel utilization
regardless of batch size, Dynamic PIM Command Scheduling overlaps data
movement with computation, and the Dynamic PIM Access controller removes
static memory waste. The latter two are credited as one aggregate
orchestration gain on the weight-streaming path. Attention executes over the
AiMX all-bank internal bandwidth and is hidden behind weight streaming at
every evaluated point.
"""
from .base import BaselineSystem, host_overhead_s, short_factor
from ..system import Result


class PimphonySystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        if w.peak_mem > self.total_capacity(devices):
            return Result.oom(cfg["name"])
        t_fc = w.weight_bytes / (cfg["xpu_bw_aggregate"] * s
                                 * cfg["weight_stream"]
                                 * cfg["orchestration_gain"])
        att_bw = cfg["devices"] * s * cfg["aim_internal_per_device"] \
            * cfg["pim_stream_efficiency"] * cfg["pim_util"]
        t_att = w.kv_bytes * w.attn_reuse / att_bw
        t = max(max(t_fc, t_att) * cfg["overlap_overhead"],
                self.compute_s(w, devices))
        t += self.collective_s(w, devices) + host_overhead_s()
        t /= short_factor(w.d_model)
        return self.finish(w, t, devices=devices, counters=dict(
            t_fc=t_fc, t_att=t_att,
            hbm_bytes=w.weight_bytes,
            pim_int_bytes=w.kv_bytes * w.attn_reuse))

    def energy(self, res, w):
        # DCS chunking keeps the streaming pipeline continuously busy; KV
        # operands are served from the AiMX near-bank internal datapath.
        en = self.cfg["energy"]
        c = res.counters
        n = self.n_devices(res)
        t_busy = w.batch / res.throughput * short_factor(w.d_model)
        duty = min(c["t_fc"] / t_busy, 1.0)
        cap = self.cfg.get("hbm_capacity") or self.cfg["capacity"]
        bg = cap * self.dev_scale(res) / 1e9 * en["background_w_per_gb"] \
            * en["background_idle_factor"]
        p = n * (en["xpu_busy_w"] * duty + en["xpu_idle_w"] * (1 - duty)
                 + en["pim_pe_w_per_device"]) + en["static_w"] + bg
        dram = (c["hbm_bytes"] * en["hbm_pj_per_bit"]
                + c["pim_int_bytes"] * en["aim_int_pj_per_bit"]) \
            * 8e-12 / w.batch
        res.power_w = p + dram * res.throughput
        res.tokens_per_joule = res.throughput / res.power_w
