"""Shared helpers for baseline system models."""
from ..config import load_common
from ..system import Result
from ..scheduler import moe_frag_efficiency, small_op_efficiency


def short_factor(d_model):
    sp = load_common()["short_payload"]
    return sp["system_efficiency"] if d_model <= sp["d_model_threshold"] else 1.0


def host_overhead_s():
    """Per-iteration host framework cost, identical for every system
    (configs/common.yaml)."""
    return load_common()["host_iteration_overhead_ms"] * 1e-3


class BaselineSystem:
    """Base class: capacity check + throughput/energy plumbing."""

    def __init__(self, cfg):
        self.cfg = cfg

    # -- collectives --------------------------------------------------- #
    def collective_s(self, w, devices=None):
        """Seconds a decode iteration spends in tensor-parallel AllReduces.

        Every evaluated system shards each layer across its devices, so each
        transformer block ends with the same two reductions Proteus pays
        (Sec. IV-B); only the fabric differs. Modeled as a chunk-pipelined
        ring: ``2 (N-1)/N`` of the activation tensor per device and
        collective, plus one port-to-port latency per collective."""
        ic = self.cfg.get("interconnect")
        if not ic:
            return 0.0
        n = int(devices or self.cfg["devices"])
        if n <= 1:
            return 0.0
        from ..fabric import CxlFabric
        per_layer = int(self.cfg.get("tp_collectives_per_layer", 2))
        f = CxlFabric(ic)
        return f.tp_allreduce_ns(w.activation_bytes(), n, per_layer) \
            * w.n_layers / 1e9

    # -- compute roof --------------------------------------------------- #
    def compute_s(self, w, devices=None):
        """Seconds the declared compute engines need for one iteration.

        Every system that publishes an arithmetic throughput is given the
        corresponding roofline, so no system is modelled as having infinite
        compute (Sec. V-A: heterogeneous systems match Proteus's aggregate
        xPU throughput)."""
        for key in ("flops_fp16_aggregate", "xpu_flops_aggregate",
                    "pim_flops_aggregate"):
            if key in self.cfg:
                fl = self.cfg[key] * self._scale(devices)
                return (w.weight_flops + w.attn_flops) / fl
        return 0.0

    # -- capacity ------------------------------------------------------ #
    def total_capacity(self, devices=None):
        """Usable memory across the deployment (bytes)."""
        c = self.cfg
        s = self._scale(devices)
        for key in ("hbm_capacity", "capacity"):
            if key in c:
                return c[key] * s * c.get("usable_fraction", 0.90)
        return c["devices"] * s * c["capacity_per_device"] * \
            c.get("usable_fraction", 0.90)

    def kv_capacity(self, model, devices=None):
        """Bytes available for KV caches once the weights are resident."""
        return max(self.total_capacity(devices) - model["weight_bytes"], 0.0)

    # -- helpers ------------------------------------------------------- #
    def _scale(self, devices):
        """Linear resource scaling when a non-default device count is used."""
        return (devices or self.cfg["devices"]) / self.cfg["devices"]

    #: Sustained weight-streaming efficiency of a well-formed dense GEMM.
    DENSE_STREAM_EFF = 0.75

    def xw_eff(self, w):
        """xPU weight-streaming efficiency.

        A dense layer presents one well-formed GEMM per matrix and streams at
        the engine's sustained rate. MoE routing instead splits the same work
        across experts, and with few tokens per expert the resulting tiles
        underutilize the array and each launch amortizes over less work, so
        the efficiency follows the fragmentation curve until it saturates."""
        c = self.cfg["efficiency"]["moe_frag"] if "efficiency" in self.cfg \
            else self.cfg["weight_eff"]
        dense = self.cfg["efficiency"]["weight_stream"] \
            if "efficiency" in self.cfg and "weight_stream" in self.cfg["efficiency"] \
            else c.get("dense", self.DENSE_STREAM_EFF)
        if not w.model["moe"]["enabled"]:
            return dense
        tpe = max(w.tokens_per_expert, 1.0)
        if tpe >= c["saturate_tpe"]:
            return dense
        return min(dense, moe_frag_efficiency(c, tpe))

    def smallf(self, w):
        return small_op_efficiency(self.cfg["small_op_efficiency"],
                                   max(w.tokens_per_expert, 1.0))

    def finish(self, w, t_iter, counters=None, notes="", devices=None):
        c = dict(counters or {})
        c["devices"] = int(devices or self.cfg["devices"])
        res = Result(True, self.cfg["name"], throughput=w.batch / t_iter,
                     t_iter_ms=t_iter * 1e3, counters=c, notes=notes)
        self.energy(res, w)
        return res

    def n_devices(self, res):
        """Device count of the deployment a result was produced for."""
        return res.counters.get("devices", self.cfg["devices"])

    def dev_scale(self, res):
        return self.n_devices(res) / self.cfg["devices"]

    def energy(self, res, w):   # overridden per system
        pass
