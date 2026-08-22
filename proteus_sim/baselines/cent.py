"""CENT baseline: GPU-free system scaling CXL-attached GDDR6-AiM devices.

Near-bank GEMV execution without inter-PE operand sharing:
  * the work states that a pipeline stage processes one token at a time, so
    shared-operand GEMMs decompose into per-vector GEMVs and the active
    weights are re-streamed once per routed token (traffic x tokens/expert);
  * GQA/MLA attention re-reads the shared KV once per query group (x g);
  * very short per-expert operand chains reduce streaming efficiency;
  * whole layers are pipelined across devices rather than sharded, so there
    is no tensor-parallel reduction -- only one activation vector crosses the
    CXL fabric per pipeline-stage boundary.
"""
from .base import BaselineSystem, host_overhead_s, short_factor
from ..fabric import CxlFabric
from ..system import Result


class CentSystem(BaselineSystem):
    def collective_s(self, w, devices=None):
        """Pipeline-boundary activation transfers, not a TP collective."""
        n = int(devices or self.cfg["devices"])
        boundaries = min(int(self.cfg["pipeline_boundaries_per_token"]), n)
        if boundaries <= 0:
            return 0.0
        f = CxlFabric(self.cfg["interconnect"])
        return boundaries * f.transfer_ns(w.activation_bytes()) / 1e9

    def simulate(self, w, devices=None, dp=1):
        cfg = self.cfg
        s = self._scale(devices)
        if w.peak_mem > self.total_capacity(devices):
            return Result.oom(cfg["name"])
        bw = cfg["devices"] * s * cfg["internal_bw_per_device"] \
            * cfg["stream_efficiency"]
        tpe = max(w.tokens_per_expert, 1.0)
        # Only the weight stream is fragmented into per-token GEMVs; the KV
        # stream is one long per-request chain and keeps the full rate.
        traffic = w.weight_bytes * tpe + w.kv_bytes * w.attn_reuse
        t = max(w.weight_bytes * tpe / (bw * self.smallf(w))
                + w.kv_bytes * w.attn_reuse / bw,
                self.compute_s(w, devices))
        t += self.collective_s(w, devices) + host_overhead_s()
        t /= short_factor(w.d_model)
        return self.finish(w, t, devices=devices,
                           counters=dict(gddr_int_bytes=traffic))

    def energy(self, res, w):
        en = self.cfg["energy"]
        n = self.n_devices(res)
        p = n * en["device_w"] + en["static_w"]
        dram = res.counters["gddr_int_bytes"] / w.batch \
            * en["gddr_int_pj_per_bit"] * 8e-12
        res.power_w = p + dram * res.throughput
        res.tokens_per_joule = res.throughput / res.power_w
