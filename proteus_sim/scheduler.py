"""Adaptive operation scheduling (Sec. IV-D, Sec. IV-E).

The scheduler owns three decisions, all re-evaluated once per decode
iteration from quantities that are only known at runtime (token counts,
expert routing, processor availability):

1. *Analytical crossover estimate.* For an operator with ``N_ops`` floating
   point operations and ``N_acc`` bytes of memory traffic the first-order
   costs are

       T_xPU = max(N_ops/F_xPU, N_acc/BW_out)
       T_PIM = max(N_ops/F_PIM, N_acc/BW_in)                        -- Eq. (1)

   which places compute-intensive operators on the xPU and strongly
   memory-bound operators in PIM. Between the two machine balance points
   ``AI_PIM = F_PIM/BW_in`` and ``AI_xPU = F_xPU/BW_out`` the decisive
   quantity is ``theta = F_PIM/BW_out`` (Eq. (2)), which is 32.768 TFLOPS
   over a 1 TB/s external interface for the evaluated configuration.

2. *PIM connectivity.* Reuse-free operators use direct mode; reuse-bearing
   skinny-GEMMs use broadcasting, whose BG-local fan-out is bounded by the
   operand-FIFO depth (Sec. IV-C).

3. *Runtime adaptation.* The crossover estimate is a coarse first-order
   default. For operators inside the crossover region the scheduler adds the
   *estimated* queueing delay -- the sum of the analytical costs of
   dispatched but unfinished operators -- to the operator's own estimated
   cost on each substrate and greedily selects the lower total. Compute- and
   memory-dominated operators retain their default mapping, which limits
   remapping-induced loss.

   Both the default and the revision are driven by the same first-order cost
   model, so ``theta`` is not merely a threshold: it *is* the estimator's
   view of the machine, ``theta = F_PIM/BW_out``. Perturbing it perturbs
   every estimate the scheduler makes, in the same direction. A scheduler
   configured with theta = AI_PIM believes the xPU has ``theta_nominal/2``
   times more operand bandwidth than it has, places everything there, and
   never revises the decision -- its queueing estimates are wrong by the same
   factor. This is what Fig. 15 measures, and it is why a mis-estimated
   threshold cannot be repaired at runtime while a slightly imprecise one
   costs almost nothing.

Expert-centric processing (Sec. IV-E) enters through the ``n_vectors`` of an
operator: with token-centric execution every routed token forms its own
reuse-free GEMV (``n = 1``), while expert-centric grouping turns each expert
into one skinny-GEMM whose intensity is set by its runtime token count.
"""
import math
from dataclasses import dataclass


def shared_operand_ai(n, d):
    """Arithmetic intensity of a shared-operand skinny-GEMM (Eq. (3)).

    ``n x d`` input activations against a shared ``d x d`` FP16 matrix:
    ``N_ops = 2 n d^2`` and ``N_acc = 2 (2 n d + d^2)`` bytes, hence
    ``AI_w = n d / (2 n + d)``.
    """
    return shared_operand_ai_rect(n, d, d)


def shared_operand_ai_rect(n, k, m):
    """Eq. (3) for a rectangular ``k x m`` shared operand.

    ``n x k`` activations against a shared ``k x m`` FP16 matrix:
    ``N_ops = 2 n k m``, ``N_acc = 2 (k m + n k + n m)``, hence
    ``AI_w = n k m / (k m + n k + n m)``; ``k = m = d`` recovers Eq. (3).
    """
    n = max(float(n), 1.0)
    return n * k * m / (k * m + n * k + n * m)


@dataclass
class Placement:
    op: object
    substrate: str        # 'xpu' | 'pim' | 'sfu'
    pim_mode: str         # 'direct' | 'broadcast' | '-'
    ai: float             # workload arithmetic intensity
    region: str           # 'compute' | 'crossover' | 'memory'
    remapped: bool = False   # moved off its default by runtime adaptation


@dataclass
class SchedulerCounters:
    """Instrumentation of the runtime scheduler itself (Sec. V-B).

    One invocation evaluates a closed-form intensity expression per operator
    group, reuses the result across identically shaped transformer blocks,
    and rebuilds one routing histogram per MoE layer.
    """
    invocations: int = 0
    flops: int = 0            # floating-point operations in the estimator
    int_ops: int = 0          # integer increments (routing histograms)
    remaps: int = 0
    mode_switches: int = 0

    FLOPS_PER_ESTIMATE = 11   # AI_w, T_xPU, T_PIM, queue add, 2 compares

    def account_estimate(self, n_groups):
        self.invocations += 1
        self.flops += n_groups * self.FLOPS_PER_ESTIMATE

    def account_histogram(self, n_tokens, n_layers):
        self.int_ops += int(n_tokens) * int(n_layers)

    def overhead_ns(self, fp_ns=1.0, int_ns=0.3):
        """Wall-clock estimate on a host core (1 GHz-equivalent scalar issue)."""
        return self.flops * fp_ns + self.int_ops * int_ns

    def as_dict(self):
        return dict(sched_invocations=self.invocations,
                    sched_flops=self.flops, sched_int_ops=self.int_ops,
                    sched_remaps=self.remaps,
                    sched_mode_switches=self.mode_switches,
                    sched_overhead_us=self.overhead_ns() / 1e3)


class CrossoverScheduler:
    """Analytical placement + PIM-connectivity selection for one iteration."""

    def __init__(self, sys_cfg, dmem, xpu_flops, xpu_bw_out):
        sc = sys_cfg["scheduler"]
        self.theta = float(sc["crossover_ai"])          # F_PIM / BW_out
        self.runtime_adaptation = bool(sc.get("runtime_adaptation", True))
        # Multiplicative half-width of the band around theta the runtime is
        # allowed to revise (Sec. IV-D "Runtime Adaptation").
        # The runtime keeps an operator on its default substrate unless the
        # alternative is estimated to be better by this margin: without the
        # hysteresis the placement of a near-balanced operator would oscillate
        # between consecutive decode iterations and each flip would cost a
        # datapath reconfiguration.
        self.adapt_hysteresis = float(sc.get("adaptation_hysteresis", 1.0))
        self.dmem = dmem
        self.fanout = dmem.fanout
        self.f_pim = dmem.pe_flops_peak
        self.bw_in = dmem.internal_peak
        self.f_xpu = xpu_flops
        self.bw_out = xpu_bw_out
        self.ai_pim = self.f_pim / self.bw_in           # 2.0
        self.ai_xpu = self.f_xpu / self.bw_out          # 312.0
        self.counters = SchedulerCounters()

    # -- crossover model ------------------------------------------------ #
    def region(self, ai):
        if ai > self.ai_xpu:
            return "compute"
        if ai > self.ai_pim:
            return "crossover"
        return "memory"

    def default_substrate(self, ai):
        """Eq. (2): the lower-estimated-latency substrate for intensity ``ai``."""
        reg = self.region(ai)
        if reg == "compute":
            return "xpu", reg
        if reg == "memory":
            return "pim", reg
        return ("xpu" if ai > self.theta else "pim"), reg

    @property
    def bw_out_estimated(self):
        """The operand bandwidth the estimator ascribes to the xPU.

        ``theta = F_PIM / BW_out`` by definition (Sec. IV-D), so a configured
        theta is exactly a claim about ``BW_out``; at the nominal setting it
        reproduces the true external interface."""
        return self.f_pim / self.theta if self.theta > 0 else self.bw_out

    def revisable(self, ai):
        """True if the operator lies in the crossover region, where the
        first-order estimate cannot decide the substrate on its own."""
        return self.runtime_adaptation and self.ai_pim < ai <= self.ai_xpu

    def pim_mode(self, n_vectors, enable_broadcast=True):
        """Reuse-free GEMVs use direct mode; reuse-bearing skinny-GEMMs use
        broadcasting (Sec. IV-C)."""
        return "broadcast" if (enable_broadcast and n_vectors > 1) else "direct"

    def broadcast_passes(self, reuse):
        """Streaming passes over a shared operand given the available reuse
        and the BG-local fan-out."""
        return max(1, math.ceil(max(reuse, 1.0) / self.fanout))

    # -- reporting ------------------------------------------------------ #
    def place(self, op, enable_broadcast=True):
        """Discrete default placement of one operator group (reporting path)."""
        if op.kind == "elementwise":
            return Placement(op, "sfu", "-", op.arithmetic_intensity, "memory")
        ai = op.intensity
        sub, reg = self.default_substrate(ai)
        mode = self.pim_mode(op.n_vectors, enable_broadcast) if sub == "pim" else "-"
        return Placement(op, sub, mode, ai, reg)


def small_op_efficiency(table, tpe):
    """PIM streaming efficiency for very short skinny-GEMM operand chains."""
    if tpe < 2:
        return table["tpe_lt2"]
    if tpe < 4:
        return table["tpe_lt4"]
    return table["else"]


def moe_frag_efficiency(curve, tpe):
    """xPU GEMM efficiency under MoE expert fragmentation."""
    return curve["base"] + min(tpe, curve["saturate_tpe"]) / curve["saturate_tpe"] \
        * curve["span"]
