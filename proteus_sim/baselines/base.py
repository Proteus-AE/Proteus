"""Shared helpers for baseline system models."""
import os
import yaml

from ..system import Result
from ..scheduler import moe_frag_efficiency, small_op_efficiency

_COMMON = None


def common():
    global _COMMON
    if _COMMON is None:
        p = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "common.yaml")
        with open(p) as f:
            _COMMON = yaml.safe_load(f)
    return _COMMON


def short_factor(d_model):
    sp = common()["short_payload"]
    return sp["system_efficiency"] if d_model <= sp["d_model_threshold"] else 1.0


class BaselineSystem:
    """Base class: capacity check + throughput/energy plumbing."""

    def __init__(self, cfg):
        self.cfg = cfg

    # -- helpers ------------------------------------------------------- #
    def _scale(self, devices):
        """Linear resource scaling when a non-default device count is used."""
        return (devices or self.cfg["devices"]) / self.cfg["devices"]

    def xw_eff(self, w):
        """xPU GEMM streaming efficiency under MoE fragmentation."""
        c = self.cfg["efficiency"]["moe_frag"] if "efficiency" in self.cfg \
            else self.cfg["weight_eff"]
        tpe = max(w.tokens_per_expert, 1.0)
        if tpe >= 32:
            return 0.75
        return moe_frag_efficiency(c, tpe)

    def smallf(self, w):
        return small_op_efficiency(self.cfg["small_op_efficiency"],
                                   max(w.tokens_per_expert, 1.0))

    def finish(self, w, t_iter, counters=None, notes=""):
        res = Result(True, self.cfg["name"], throughput=w.batch / t_iter,
                     t_iter_ms=t_iter * 1e3, counters=counters or {}, notes=notes)
        self.energy(res, w)
        return res

    def energy(self, res, w):   # overridden per system
        pass
