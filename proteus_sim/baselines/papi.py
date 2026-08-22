"""PAPI baseline: heterogeneous Attn-PIM / FC-PIM device pools (2:1) + xPU.

Mechanisms:
  * KV caches are resident in the Attn-PIM pool; attention always executes
    there, at the all-bank internal rate, re-reading the shared KV once per
    query group because the datapath is one-to-one.
  * FC layers run wherever they are cheaper: the logic-die FC-PIM pool gives
    GEMV-grade execution and wins for small (xPU-fragmenting) experts, while
    large-operand skinny-GEMMs run on the xPU under the MoE-fragmentation
    efficiency curve. FC-PIM is only eligible when the resident weights fit
    in the FC pool.
  * Attn-PIM and FC phases overlap (PAPI's own pipelining), with a
    coordination overhead.
"""
from .base import BaselineSystem, host_overhead_s, short_factor
from ..system import Result


class PapiSystem(BaselineSystem):
    def _pools(self, devices=None):
        """(Attn-PIM KV capacity, FC-PIM weight capacity) in bytes."""
        c = self.cfg
        cap = c["hbm_capacity"] * self._scale(devices) * c["usable_fraction"]
        return c["attn_pool_fraction"] * cap, \
            (1.0 - c["attn_pool_fraction"]) * cap

    def kv_capacity(self, model, devices=None):
        """KV caches are resident in the Attn-PIM pool only (Sec. V-A)."""
        return self._pools(devices)[0]

    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        kv_cap, fc_cap = self._pools(devices)
        if (w.peak_mem - w.model["weight_bytes"]) > kv_cap:
            return Result.oom(cfg["name"], "KV cache exceeds Attn-PIM pool")
        att_bw = cfg["attn_pool_fraction"] * cfg["attn_allbank_mult"] \
            * cfg["xpu_bw_aggregate"] * s * cfg["attention_eff"]
        t_att = w.kv_bytes * w.attn_reuse / att_bw

        t_xpu = w.weight_bytes / (cfg["xpu_bw_aggregate"] * s * self.xw_eff(w))
        if w.model["weight_bytes"] <= fc_cap:
            fc_bw = (1.0 - cfg["attn_pool_fraction"]) \
                * cfg["fc_pim_allbank_mult"] * cfg["xpu_bw_aggregate"] * s \
                * cfg["fc_pim_efficiency"]
            t_fc_pim = w.weight_bytes / fc_bw
        else:                 # weights do not fit the FC-PIM pool
            t_fc_pim = float("inf")
        fc_on_pim = t_fc_pim <= t_xpu
        t_fc = min(t_fc_pim, t_xpu)

        t = max(max(t_fc, t_att) * cfg["overlap_overhead"],
                self.compute_s(w, devices))
        t += self.collective_s(w, devices) + host_overhead_s()
        t /= short_factor(w.d_model)
        return self.finish(w, t, devices=devices, counters=dict(
            t_fc=t_fc, t_att=t_att, fc_on_pim=fc_on_pim,
            hbm_bytes=0.0 if fc_on_pim else w.weight_bytes,
            pim_int_bytes=w.kv_bytes * w.attn_reuse
            + (w.weight_bytes if fc_on_pim else 0.0)))

    def energy(self, res, w):
        # The xPU units are busy only for the FC work they actually run; KV
        # and FC-PIM operands are charged at the PIM-internal access energy.
        en = self.cfg["energy"]
        c = res.counters
        n = self.n_devices(res)
        t_busy = w.batch / res.throughput * short_factor(w.d_model)
        duty = 0.0 if c["fc_on_pim"] else min(c["t_fc"] / t_busy, 1.0)
        cap = self.cfg.get("hbm_capacity") or self.cfg["capacity"]
        bg = cap * self.dev_scale(res) / 1e9 * en["background_w_per_gb"] \
            * en["background_idle_factor"]
        p = n * (en["xpu_busy_w"] * duty + en["xpu_idle_w"] * (1 - duty)
                 + en["pim_pe_w_per_device"]) + en["static_w"] + bg
        dram = (c["hbm_bytes"] * en["hbm_pj_per_bit"]
                + c["pim_int_bytes"] * en["hbm_pim_int_pj_per_bit"]) \
            * 8e-12 / w.batch
        res.power_w = p + dram * res.throughput
        res.tokens_per_joule = res.throughput / res.power_w
