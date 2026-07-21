"""PAPI baseline: heterogeneous Attn-PIM / FC-PIM device pools (2:1) + xPU.

Mechanisms:
  * KV caches are resident in the Attn-PIM pool; attention always executes
    there at the all-bank internal rate. MLA's short latent-KV rows sustain
    a lower streaming efficiency than GQA/MHA rows (att_eff per type).
  * FC layers: PAPI's logic-die FC-PIM provides GEMV-grade execution that
    excels for small (fragmenting) experts; large-operand skinny-GEMMs run
    on the xPU with the MoE-fragmentation efficiency curve.
  * Attn-PIM and FC phases overlap (original PAPI pipelining), with a
    coordination overhead.
"""
from .base import BaselineSystem, short_factor
from ..system import Result


class PapiSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        kv_cap = cfg["attn_pool_fraction"] * cfg["hbm_capacity"] * s \
            * cfg["usable_fraction"]
        if (w.peak_mem - w.model["weight_bytes"]) > kv_cap:
            return Result.oom(cfg["name"], "KV cache exceeds Attn-PIM pool")
        att_eff = cfg["attention_eff"][w.model["attention"]]
        att_bw = cfg["attn_pool_fraction"] * cfg["attn_allbank_mult"] \
            * cfg["xpu_bw_aggregate"] * s * att_eff
        t_att = w.kv_bytes * w.attn_reuse / att_bw

        moe = w.model["moe"]
        small_experts = moe["enabled"] and \
            moe["expert_bytes"] < cfg["fc_pim_expert_bytes_max"]
        if small_experts:     # GEMV-grade FC-PIM pool
            fc_bw = (1.0 - cfg["attn_pool_fraction"]) * cfg["fc_pim_allbank_mult"] \
                * cfg["xpu_bw_aggregate"] * s * cfg["fc_pim_efficiency"]
            t_fc = w.weight_bytes / fc_bw
            fc_on_pim = True
        else:                 # xPU skinny-GEMM
            t_fc = w.weight_bytes / (cfg["xpu_bw_aggregate"] * s * self.xw_eff(w))
            fc_on_pim = False
        t = max(t_fc, t_att) * cfg["overlap_overhead"] \
            + cfg["launch_overhead_ms"] * 1e-3
        t /= short_factor(w.d_model)
        return self.finish(w, t, counters=dict(
            t_fc=t_fc, t_att=t_att, fc_on_pim=fc_on_pim,
            hbm_bytes=0.0 if fc_on_pim else w.weight_bytes,
            pim_int_bytes=w.kv_bytes * w.attn_reuse
            + (w.weight_bytes if fc_on_pim else 0.0)))

    def energy(self, res, w):
        # xPU units drive command generation and gather/scatter even when FC
        # executes in the FC-PIM pool, keeping them near full occupancy; KV
        # and FC-PIM operands are charged at the PIM-internal access energy.
        en = self.cfg["energy"]
        c = res.counters
        bg = self.cfg["hbm_capacity"] / 1e9 * en["background_w_per_gb"] \
            * en["background_idle_factor"]
        t_busy = w.batch / res.throughput * short_factor(w.d_model)
        duty = 1.0 if c["fc_on_pim"] else min(c["t_fc"] / t_busy, 1.0)
        p = self.cfg["devices"] * (en["xpu_busy_w"] * duty
                                   + en["xpu_idle_w"] * (1 - duty)
                                   + en["pim_pe_w_per_device"]) \
            + en["static_w"] + bg
        dram = (c["hbm_bytes"] * en["hbm_pj_per_bit"]
                + c["pim_int_bytes"] * en["hbm_pim_int_pj_per_bit"]) * 8e-12 / w.batch
        res.power_w = p
        res.tokens_per_joule = 1.0 / (p / res.throughput + dram)
