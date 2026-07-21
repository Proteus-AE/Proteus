"""PIMphony baseline: orchestration system for PIM-based inference on
AiMX-class devices.

Modeled as the NeuPIMs-style heterogeneous execution enhanced by PIMphony's
three techniques (credited as an aggregate orchestration gain): TCP restores
channel utilization, kernel orchestration removes xPU GEMM fragmentation,
and DCS chunking overlaps the fallback path. Attention executes over the
AiMX all-bank internal bandwidth and is effectively hidden behind weight
streaming for all evaluated points."""
from .neupims import stream_eff
from .base import BaselineSystem, short_factor
from ..system import Result


class PimphonySystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        if w.peak_mem > cfg["capacity"] * s * 0.90:
            return Result.oom(cfg["name"])
        e_w = stream_eff(cfg["weight_stream_curve"], w.weight_bytes)
        t_fc = w.weight_bytes / (cfg["xpu_bw_aggregate"] * s * e_w
                                 * cfg["orchestration_gain"])
        att_bw = cfg["devices"] * s * cfg["aim_internal_per_device"] \
            * cfg["pim_stream_efficiency"] * cfg["pim_util"]
        t_att = w.kv_bytes * w.attn_reuse / att_bw
        t = max(t_fc, t_att) * cfg["overlap_overhead"]
        t /= short_factor(w.d_model)
        return self.finish(w, t, counters=dict(
            t_fc=t_fc, t_att=t_att,
            hbm_bytes=w.weight_bytes,
            pim_int_bytes=w.kv_bytes * w.attn_reuse))

    def energy(self, res, w):
        # DCS chunking keeps the xPU streaming pipeline continuously busy;
        # KV operands are served from the AiMX near-bank internal datapath.
        en = self.cfg["energy"]
        c = res.counters
        bg = self.cfg["capacity"] / 1e9 * en["background_w_per_gb"] \
            * en["background_idle_factor"]
        p = self.cfg["devices"] * (en["xpu_busy_w"]
                                   + en["pim_pe_w_per_device"]) \
            + en["static_w"] + bg
        dram = (c["hbm_bytes"] * en["hbm_pj_per_bit"]
                + c["pim_int_bytes"] * en["aim_int_pj_per_bit"]) * 8e-12 / w.batch
        res.power_w = p
        res.tokens_per_joule = 1.0 / (p / res.throughput + dram)
