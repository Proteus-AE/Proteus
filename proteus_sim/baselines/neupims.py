"""NeuPIMs baseline: heterogeneous NPU + HBM-PIM with dual row buffers and
sub-batch interleaving (concurrent NPU GEMM and PIM attention).

Mechanisms:
  * Weight GEMMs stream on the NPU at the memory-bandwidth utilization the
    work reports for its interleaved schedule.
  * Attention executes in HBM-PIM at the dual-row-buffer internal rate. The
    fixed one-to-one datapath has no inter-PE operand reuse, so GQA/MLA
    attention re-reads the shared KV once per query group; the scheduler
    takes whichever of the PIM and NPU paths is faster, overlapped with
    weight streaming through sub-batch interleaving.
"""
from .base import BaselineSystem, host_overhead_s, short_factor
from ..system import Result


class NeuPimsSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        if w.peak_mem > self.total_capacity(devices):
            return Result.oom(cfg["name"])
        bw = cfg["hbm_bw_aggregate"] * s
        t_fc = w.weight_bytes / (bw * cfg["weight_stream"])
        t_pim = w.kv_bytes * w.attn_reuse / \
            (bw * cfg["pim_internal_mult"] * cfg["pim_stream_efficiency"])
        t_npu = w.kv_bytes / (bw * cfg["attention_xpu_eff"])
        att_on_pim = t_pim <= t_npu
        t_att = min(t_pim, t_npu)
        t = max(max(t_fc, t_att) * cfg["overlap_overhead"],
                self.compute_s(w, devices))
        t += self.collective_s(w, devices) + host_overhead_s()
        t /= short_factor(w.d_model)
        return self.finish(w, t, devices=devices, counters=dict(
            att_on_pim=att_on_pim, t_fc=t_fc, t_att=t_att,
            hbm_bytes=w.weight_bytes + (0.0 if att_on_pim else w.kv_bytes),
            pim_int_bytes=w.kv_bytes * w.attn_reuse if att_on_pim else 0.0))

    def energy(self, res, w):
        # Weights stream through the external HBM interface; KV operands are
        # served from the PIM stacks' internal datapath (with the g-fold
        # re-read of the fixed one-to-one connectivity). NPU busy fraction is
        # measured against the command-limited schedule.
        en = self.cfg["energy"]
        c = res.counters
        n = self.n_devices(res)
        t_busy = w.batch / res.throughput * short_factor(w.d_model)
        duty = min((c["t_fc"] + (0.0 if c["att_on_pim"] else c["t_att"]))
                   / t_busy, 1.0)
        bg = self.cfg["hbm_capacity"] * self.dev_scale(res) / 1e9 \
            * en["background_w_per_gb"] * en["background_idle_factor"]
        p = n * (en["npu_busy_w"] * duty + en["npu_idle_w"] * (1 - duty)
                 + en["pim_pe_w_per_device"]) + en["static_w"] + bg
        dram = (c["hbm_bytes"] * en["hbm_pj_per_bit"]
                + c["pim_int_bytes"] * en["hbm_pim_int_pj_per_bit"]) \
            * 8e-12 / w.batch
        res.power_w = p + dram * res.throughput
        res.tokens_per_joule = res.throughput / res.power_w
