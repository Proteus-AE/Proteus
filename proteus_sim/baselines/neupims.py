"""NeuPIMs baseline: heterogeneous NPU + HBM-PIM with dual row buffers and
sub-batch interleaving (concurrent NPU GEMM and PIM attention).

Mechanisms:
  * Weight GEMMs stream on the NPU; sustained efficiency grows with the
    active weight working set (longer DMA bursts amortize per-kernel
    overheads) and saturates -- see ``weight_stream_curve`` (A-NPU).
  * MHA attention executes in HBM-PIM at the dual-row-buffer internal rate.
    GQA/MLA attention has no inter-PE operand reuse in the fixed one-to-one
    datapath (it would re-read the shared KV once per query group), so the
    scheduler falls back to the NPU, overlapped with weight streaming via
    sub-batch interleaving.
"""
import math
from .base import BaselineSystem, short_factor
from ..system import Result


def stream_eff(curve, wt_bytes):
    return min(curve["cap"],
               curve["base"] + curve["log2_slope"] * math.log2(wt_bytes / 1e9))


class NeuPimsSystem(BaselineSystem):
    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        if w.peak_mem > cfg["hbm_capacity"] * s * cfg["usable_fraction"]:
            return Result.oom(cfg["name"])
        bw = cfg["hbm_bw_aggregate"] * s
        e_w = stream_eff(cfg["weight_stream_curve"], w.weight_bytes)
        t_fc = w.weight_bytes / (bw * e_w)
        if w.attn_reuse > 1:      # GQA/MLA: NPU fallback (no PIM operand reuse)
            t_att = w.kv_bytes / (bw * cfg["attention_xpu_eff"])
            att_on_pim = False
        else:                     # MHA: PIM attention at dual-row-buffer rate
            t_att = w.kv_bytes * w.attn_reuse / \
                (bw * cfg["pim_internal_mult"] * cfg["pim_stream_efficiency"])
            att_on_pim = True
        t = max(t_fc, t_att) * cfg["overlap_overhead"]
        t /= short_factor(w.d_model)
        return self.finish(w, t, counters=dict(
            att_on_pim=att_on_pim, t_fc=t_fc, t_att=t_att,
            hbm_bytes=w.weight_bytes,
            pim_int_bytes=w.kv_bytes * w.attn_reuse))

    def energy(self, res, w):
        # Weights stream through the external HBM interface; KV operands are
        # served from the PIM stacks' internal datapath (with the g-fold
        # re-read of the fixed one-to-one connectivity). NPU busy fraction is
        # measured against the command-limited schedule.
        en = self.cfg["energy"]
        c = res.counters
        bg = self.cfg["hbm_capacity"] / 1e9 * en["background_w_per_gb"] \
            * en["background_idle_factor"]
        t_busy = w.batch / res.throughput * short_factor(w.d_model)
        duty = min((c["t_fc"] + c["t_att"]) / t_busy, 1.0)
        p = self.cfg["devices"] * (en["npu_busy_w"] * duty
                                   + en["npu_idle_w"] * (1 - duty)
                                   + en["pim_pe_w_per_device"]) \
            + en["static_w"] + bg
        dram = (c["hbm_bytes"] * en["hbm_pj_per_bit"]
                + c["pim_int_bytes"] * en["hbm_pim_int_pj_per_bit"]) * 8e-12 / w.batch
        res.power_w = p
        res.tokens_per_joule = 1.0 / (p / res.throughput + dram)
