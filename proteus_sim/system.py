"""Proteus system timing engine (Sec. IV) and the incremental-variant
machinery used by the effectiveness analysis (Sec. V-D).

Topology
--------
Proteus matches each form of parallelism to the bandwidth available at its
level of the memory hierarchy (Sec. IV-B, Fig. 6):

  * up to ``parallelism.group_size`` devices inside one CXL switch domain
    form a **tensor-parallel group**. Every device of a group holds a shard
    of every layer together with its resident KV shard, so a group executes
    the whole batch in one forward pass and pays two ring AllReduces per
    transformer block;
  * groups are **pipeline stages** holding whole layers; when the layer
    count does not divide evenly the trailing groups hold one fewer layer,
    and the stage time is set by the fullest group. Only a single
    ``b x d_model`` activation crosses a switch domain per boundary.

With ``G`` groups the runtime keeps ``m <= G`` independent in-flight
micro-batches (autoregressive decode cannot refill its own pipeline),
bounded by aggregate KV-cache capacity, so

    tokens/s = batch * m / (G * t_stage),      per-token latency = G * t_stage.

Execution
---------
Each transformer block is scheduled onto the two substrates by the
analytical crossover estimate of Sec. IV-D, with crossover-region operators
greedily rebalanced against the queueing delay already dispatched to each
substrate. Concurrency between the xPU and PIM exists only once the
reconfigurable datapath runs shared-operand operators in broadcasting mode,
which is what frees memory-service slots for concurrent xPU requests.

Feature flags (cumulative variants of Sec. V-D):
  as  adaptive operation scheduling: analytical-crossover placement, batched
      PIM command issue (restores row-buffer locality), asynchronous weight
      streaming, coarse-grained xPU/PIM time-slicing.
  rd  reconfigurable datapath: broadcasting mode (ceil(g/4) attention passes,
      4-way weight-operand reuse) and true xPU-PIM co-execution enabled by
      the freed memory-service slots.
  of  structural operator fusion on the SFUs; unfused nonlinear operators
      stall the PIM streaming pipeline by a fraction of the PIM-side time.
  ec  expert-centric MoE processing: per-expert skinny-GEMMs instead of
      token-centric per-vector GEMVs (restores weight-operand reuse).
"""
from dataclasses import dataclass, field

from . import memory as memmod
from .fabric import CxlFabric
from .scheduler import (CrossoverScheduler, SchedulerCounters,
                        small_op_efficiency)
from .workload import Workload, build_workload

FULL = frozenset({"as", "rd", "of", "ec"})
VARIANTS = {          # cumulative variants of Sec. V-D
    "base": frozenset(),
    "as": frozenset({"as"}),
    "rd": frozenset({"as", "rd"}),
    "of": frozenset({"as", "rd", "of"}),
    "ec": FULL,
    "full": FULL,
}


@dataclass
class Result:
    alive: bool
    system: str
    throughput: float = 0.0        # tokens/s
    t_iter_ms: float = 0.0         # per-token latency of one decode iteration
    tokens_per_joule: float = 0.0
    power_w: float = 0.0
    counters: dict = field(default_factory=dict)
    placements: list = field(default_factory=list)
    notes: str = ""

    @staticmethod
    def oom(system, note="memory footprint exceeds system capacity"):
        return Result(False, system, notes="OOM: " + note)


@dataclass
class OpCost:
    """Analytical cost of one operator group on each candidate substrate
    (Eq. (1)), evaluated for one device and one transformer block."""
    name: str
    kind: str
    intensity: float
    region: str
    t_xpu: float                  # true cost on the xPU
    t_xpu_est: float              # cost the scheduler's estimator predicts
    bytes_xpu: float
    op_flops: float
    pim_mode: str                 # connectivity the datapath would select
    t_pim_by_mode: dict           # 'direct' / 'broadcast' -> seconds
    bytes_pim_by_mode: dict       # 'direct' / 'broadcast' -> bytes
    passes_by_mode: dict

    @property
    def t_pim(self):
        return self.t_pim_by_mode[self.pim_mode]

    @property
    def bytes_pim(self):
        return self.bytes_pim_by_mode[self.pim_mode]

    def t_on(self, substrate, mode=None):
        if substrate == "xpu":
            return self.t_xpu
        return self.t_pim_by_mode[mode or self.pim_mode]

    def t_est(self, substrate, mode=None):
        """Cost as the runtime scheduler's first-order model predicts it."""
        if substrate == "xpu":
            return self.t_xpu_est
        return self.t_pim_by_mode[mode or self.pim_mode]

    def bytes_on(self, substrate, mode=None):
        if substrate == "xpu":
            return self.bytes_xpu
        return self.bytes_pim_by_mode[mode or self.pim_mode]


class ProteusSystem:
    def __init__(self, sys_cfg, mem_cfg, features=FULL):
        self.cfg = sys_cfg
        self.mem = mem_cfg
        self.dmem = memmod.derive(mem_cfg)
        self.features = frozenset(features)
        self.fabric = CxlFabric(sys_cfg["interconnect"])
        xpu = sys_cfg["xpu"]
        # The xPU streams over the device's external LPDDR interface; its peak
        # and sustained rates come from the memory configuration so that the
        # ridge point AI_xPU = F_xPU / BW_out matches Sec. III-B (312).
        self.xpu_flops = xpu["flops_fp16"]
        self.xpu_bw_peak = self.dmem.external_peak
        self.xpu_bw = self.dmem.external_bw
        self.sched = CrossoverScheduler(sys_cfg, self.dmem,
                                        self.xpu_flops, self.xpu_bw_peak)
        v = sys_cfg.get("variant_penalties", {})
        self.k_cmd = v.get("per_op_command_issue", 0.80)   # row-buffer locality loss
        self.k_sync = v.get("sync_weight_stream", 0.95)    # no async prefetch
        self.f_unfused = v.get("unfused_stall", 0.20)      # nonlinear-op PIM stall
        self.k_slice = v.get("coarse_slicing", 1.0)        # phase-switch cost
        self.k_token_centric = v.get("token_centric_issue", 1.0)
        # External bandwidth a concurrent xPU stream can sustain while the
        # PIM datapath streams in each connectivity mode. This is derived
        # from the column slots each mode leaves free (memory.coexec_bw), not
        # configured: direct mode returns none, broadcasting returns more
        # than the external interface can absorb.
        self.coexec = {m: min(self.xpu_bw, self.dmem.coexec_bw(m))
                       for m in ("direct", "broadcast")}

    # ---------------- topology ---------------------------------------- #
    def topology(self, devices):
        """(tensor-parallel width N, pipeline groups G) for `devices`."""
        gs = int(self.cfg["parallelism"]["group_size"])
        n = min(gs, devices)
        return n, max(1, devices // n)

    @staticmethod
    def layer_partition(n_layers, groups):
        """Whole-layer partition across pipeline groups; trailing groups hold
        one fewer layer. Returns (layers in the fullest group, imbalance)."""
        base, rem = divmod(n_layers, groups)
        full = base + (1 if rem else 0)
        return max(full, 1), (rem / groups if rem else 0.0)

    def total_capacity(self, devices=None):
        """Usable memory across the deployment (bytes)."""
        devices = devices or self.cfg["devices"]
        return self.dmem.capacity * devices * \
            self.cfg["capacity"]["usable_fraction"]

    def kv_capacity(self, model, devices=None):
        """Bytes available for KV caches once the weights are resident."""
        return max(self.total_capacity(devices) - model["weight_bytes"], 0.0)

    def _inflight(self, w: Workload, devices, groups):
        cap = self.total_capacity(devices)
        kv_peak = w.peak_mem - w.model["weight_bytes"]
        kv_budget = cap - w.model["weight_bytes"]
        if kv_budget <= 0 or w.peak_mem > cap:
            return 0
        return max(0, min(groups, int(kv_budget // max(kv_peak, 1.0))))

    # ---------------- per-operator analytical cost --------------------- #
    def _op_cost(self, op, shard, pbw_direct, pbw_bcast, pf, xbw, smallf,
                 f, mode_pref="auto"):
        """Cost of one operator group on each substrate and, for PIM, on each
        connectivity mode (Eq. (1)).

        ``direct`` mode has no inter-PE operand reuse: the same dataflow
        repeatedly streams the matrix columns for different vectors, so the
        resident operand is re-streamed once per consuming vector (per query
        group, for attention) and traffic grows linearly with batch size.
        ``broadcast`` mode reuses each readout across the ``fanout`` PEs of a
        bank group, so ``ceil(reuse/fanout)`` passes suffice, each at the
        broadcasting-mode stream rate.

        Expert-centric processing (Sec. IV-E) is what turns an expert's routed
        tokens into one skinny-GEMM. Without it every routed token is an
        independent reuse-free GEMV, and the near-bank PEs -- which hold no
        operand storage beyond the FIFO -- must re-stream the expert's weights
        once per token in either connectivity mode. The xPU is unaffected in
        occupancy terms: it is driven by the same unified runtime as the PIM
        command path, with no per-token dispatch boundary, and its matrix
        units are fed from a shared operand buffer, so fragmentation costs it
        reuse rather than utilization.
        """
        token_centric = op.is_expert and "ec" not in f
        eff = smallf if op.kind == "weight_gemm" else 1.0
        xeff = 1.0
        if token_centric:
            # Fragmentation is an execution property, not a workload one: the
            # operator still presents `tokens` vectors against a shared
            # operand, and Eq. (3) reads its intensity off that shape. What
            # token-centric dispatch changes is how many times the datapath
            # must fetch the operand to serve them.
            tokens = max(op.tokens, 1.0)
            passes = {"direct": tokens, "broadcast": tokens}
            eff *= self.k_token_centric
            ai = op.intensity
        else:
            reuse = max(float(op.reuse if op.kind == "attention"
                              else op.n_vectors), 1.0)
            passes = {"direct": reuse,
                      "broadcast": float(self.sched.broadcast_passes(reuse))}
            ai = op.intensity
        bw = {"direct": pbw_direct, "broadcast": pbw_bcast}
        b_pim, t_pim = {}, {}
        for mode in ("direct", "broadcast"):
            b_pim[mode] = op.bytes * shard * passes[mode]
            t_pim[mode] = max(op.flops * shard / pf,
                              b_pim[mode] / (bw[mode] * eff))
        if "rd" not in f:
            mode = "direct"
        elif mode_pref in ("direct", "broadcast"):
            mode = mode_pref
        else:
            mode = min(("direct", "broadcast"), key=lambda m: t_pim[m])
        # The xPU reads the resident operand once over the external interface
        # and keeps the activations in its local SRAM.
        b_xpu = op.bytes * shard
        t_xpu = max(op.flops * shard / self.xpu_flops, b_xpu / (xbw * xeff))
        # The scheduler predicts the xPU cost from its own view of the machine
        # balance (theta = F_PIM/BW_out); at the nominal theta this reproduces
        # the true cost, and a mis-estimated theta biases every decision.
        xbw_est = self.sched.bw_out_estimated * (xbw / self.xpu_bw_peak)
        t_xpu_est = max(op.flops * shard / self.xpu_flops,
                        b_xpu / (max(xbw_est, 1.0) * xeff))
        _, region = self.sched.default_substrate(ai)
        return OpCost(op.name, op.kind, ai, region, t_xpu, t_xpu_est, b_xpu,
                      op.flops, mode, t_pim, b_pim, passes)

    # ---------------- one transformer block ---------------------------- #
    def _schedule_block(self, costs, f, counters):
        """Place one block's operator groups on the two substrate queues.

        Sec. IV-D "Runtime Adaptation": the analytical crossover supplies the
        default mapping; operators inside the crossover region may be revised
        against the queueing delay already dispatched to each substrate. The
        groups are visited in decreasing cost (longest-processing-time first),
        which is what a work-conserving list scheduler does and what bounds
        the imbalance of the greedy pass. A final work-conserving step splits
        the largest crossover-region operator still on the critical substrate
        along its output columns -- an exact refinement, since output columns
        are independent -- so that both substrates finish together.
        """
        q = {"xpu": 0.0, "pim": 0.0}         # true occupancy (timing)
        qe = {"xpu": 0.0, "pim": 0.0}        # occupancy the estimator predicts
        bytes_ = {"xpu": 0.0, "pim": 0.0}
        modes = {"direct": 0, "broadcast": 0}
        chosen = []
        adapt = ("as" in f) and self.sched.runtime_adaptation
        overlap = "rd" in f          # substrates can run concurrently
        counters.account_estimate(len(costs))

        order = sorted(costs, key=lambda c: -max(c.t_xpu_est, c.t_pim)) \
            if (adapt and overlap) else list(costs)
        for c in order:
            if "as" not in f:
                # Proteus-Base: static deployment-time mapping -- attention in
                # PIM direct mode, shared-operand operators on the xPU.
                sub = "pim" if c.kind == "attention" else "xpu"
            else:
                sub, _ = self.sched.default_substrate(c.intensity)
                if adapt and self.sched.revisable(c.intensity):
                    alt = "pim" if sub == "xpu" else "xpu"
                    # Without concurrent execution the substrates serialize,
                    # so the work-conserving objective is the operator's own
                    # estimated cost; with co-execution it is the estimated
                    # completion time of the substrate queue it would join.
                    here = c.t_est(sub) + (qe[sub] if overlap else 0.0)
                    there = c.t_est(alt) + (qe[alt] if overlap else 0.0)
                    if there * self.sched.adapt_hysteresis < here:
                        sub = alt
                        counters.remaps += 1
            qe[sub] += c.t_est(sub)
            q[sub] += c.t_on(sub)
            bytes_[sub] += c.bytes_on(sub)
            if sub == "pim":
                modes[c.pim_mode] += 1
            chosen.append([c, sub, 1.0])

        if adapt and "rd" in f:
            self._qe = qe
            # Lend the idle substrate column blocks of the shared-operand
            # GEMMs on the critical path until the two finish together.
            for _ in range(len(chosen)):
                if not self._work_conserving_split(qe, q, bytes_, chosen,
                                                    counters):
                    break
        return q, bytes_, modes, chosen

    def _work_conserving_split(self, qe, q, bytes_, chosen, counters):
        """Equalize the two substrate queues by splitting one crossover-region
        GEMM along its output columns (Sec. IV-D).

        This refinement never changes an operator's *mapping*: it only lends
        the idle substrate a column block of the single largest operator on
        the critical path, so it recovers otherwise-wasted capacity without
        undoing a placement the crossover estimate made. Because it moves at
        most one operator, it cannot repair a systematically mis-estimated
        threshold -- which is what Fig. 15 measures."""
        # The refinement is planned on the estimator's view of the queues and
        # executed on the real machine, so a mis-estimated theta produces a
        # mis-sized split.
        slow = "xpu" if qe["xpu"] > qe["pim"] else "pim"
        fast = "pim" if slow == "xpu" else "xpu"
        gap = qe[slow] - qe[fast]
        if gap <= 1e-12:
            return False
        cand = None
        for entry in chosen:
            c, sub, frac = entry
            if sub != slow or c.region != "crossover" or c.kind != "weight_gemm":
                continue
            if not self.sched.revisable(c.intensity):
                continue
            if frac <= 1e-9:
                continue
            if cand is None or c.t_est(slow) * frac > cand[0].t_est(slow) * cand[2]:
                cand = entry
        if cand is None:
            return False
        c, _, avail = cand
        t_slow, t_fast = c.t_est(slow) * avail, c.t_est(fast) * avail
        if t_slow <= 0 or t_fast <= 0:
            return False
        # Move a fraction y of the operator to the faster substrate such that
        # (q_slow - y t_slow) == (q_fast + y t_fast).
        y = min(1.0, max(0.0, gap / (t_slow + t_fast)))
        if y <= 1e-9:
            return False
        qe[slow] -= y * t_slow
        qe[fast] += y * t_fast
        q[slow] -= y * avail * c.t_on(slow)
        q[fast] += y * avail * c.t_on(fast)
        bytes_[slow] -= y * avail * c.bytes_on(slow)
        bytes_[fast] += y * avail * c.bytes_on(fast)
        cand[2] = avail * (1.0 - y)
        chosen.append([c, fast, avail * y])
        counters.remaps += 1
        return True

    def _compose(self, q, modes, stall, f):
        """Compose the two substrate queues into one block time.

        With the reconfigurable datapath in broadcasting mode the PIM stream
        leaves memory-service slots free, so the xPU runs concurrently and the
        block time is the larger of the two queues. Direct mode drives every
        bank at its minimum column cycle and returns no slots, so the phases
        serialize whatever the placement, and each switch between them costs a
        command-queue drain and an xPU DMA restart.
        """
        mode = "broadcast" if modes["broadcast"] >= modes["direct"] else "direct"
        t_p = q["pim"] * (1.0 + stall)
        if "rd" in f and (self.coexec[mode] > 0.0 or q["pim"] <= 0.0):
            return max(q["xpu"], t_p)
        return (q["xpu"] + t_p) * self.k_slice

    # ---------------- main entry point --------------------------------- #
    def simulate(self, w: Workload, devices=None, dp=1, frozen_plan=None):
        cfg = self.cfg
        devices = devices or cfg["devices"]
        if dp > 1:
            return self._simulate_dp(w, devices, dp)

        n_tp, groups = self.topology(devices)
        m = self._inflight(w, devices, groups)
        if m < 1:
            return Result.oom(cfg["name"])
        layers, imbalance = self.layer_partition(w.n_layers, groups)

        f = self.features
        shard = 1.0 / n_tp                       # tensor-parallel shard fraction
        short_sys = memmod.short_payload_factor(self.mem, w.d_model, "system")
        cmd_eff = memmod.short_payload_factor(self.mem, w.d_model, "pim")

        pbw_d = self.dmem.internal_bw * cmd_eff
        pbw_b = self.dmem.broadcast_bw * cmd_eff
        pf = self.dmem.pe_flops
        # Under co-execution the xPU sustains the smaller of its own streaming
        # efficiency and the memory-service headroom the PIM connectivity mode
        # leaves free (both measured as fractions of the external interface).
        # Without co-execution the substrates serialize and the xPU has the
        # whole external interface while it runs; with co-execution it is
        # capped by the headroom the chosen connectivity mode leaves.
        def xpu_bw_in(mode):
            bw = self.coexec[mode] if ("rd" in f and self.coexec[mode] > 0) \
                else self.xpu_bw
            return bw * (self.k_sync if "as" not in f else 1.0)

        if "as" not in f:                        # Base: per-operator command
            pbw_d *= self.k_cmd                  # issue + synchronous weight
            pbw_b *= self.k_cmd                  # streaming
        tpe = max(w.tokens_per_expert, 1.0)
        smallf = small_op_efficiency(cfg["scheduler"]["small_op_efficiency"], tpe)

        counters = SchedulerCounters()
        if w.model["moe"]["enabled"]:
            counters.account_histogram(w.batch * w.model["moe"]["top_k"], layers)

        compute = [op for op in w.block if op.kind != "elementwise"]
        elem = [op for op in w.block if op.kind == "elementwise"]
        stall = 0.0 if "of" in f else self.f_unfused
        unfused = (sum(o.bytes * shard * 2.0 for o in elem) / pbw_d
                   if ("of" not in f and elem) else 0.0)

        def plan_block(mode_pref):
            """Schedule and compose one block under a connectivity mode."""
            xbw = xpu_bw_in("direct" if mode_pref == "auto" else mode_pref)
            cost = [self._op_cost(op, shard, pbw_d, pbw_b, pf, xbw, smallf,
                                  f, mode_pref) for op in compute]
            cnt = SchedulerCounters()
            if frozen_plan is not None:
                q, by, modes, chosen = self._replay_plan(cost, frozen_plan)
            else:
                q, by, modes, chosen = self._schedule_block(cost, f, cnt)
            return (self._compose(q, modes, stall, f) + unfused,
                    q, by, modes, chosen, cnt)

        # At each decode iteration the scheduler jointly selects the execution
        # substrate and, for PIM-mapped operators, the connectivity mode
        # (Sec. IV-C "Lightweight Reconfiguration"). The mode register is
        # per channel, so the choice is evaluated on the whole block: taking
        # the broadcasting cadence can be worth a slower PIM stream when it
        # buys the concurrent xPU execution the freed slots allow.
        options = ["auto"] if "rd" not in f else ["direct", "broadcast"]
        t_block, q, by, modes, chosen, counters_used = min(
            (plan_block(m) for m in options), key=lambda r: r[0])
        counters.flops += counters_used.flops
        counters.remaps += counters_used.remaps
        counters.invocations += max(counters_used.invocations, 1)

        # ---- communication (Sec. IV-B) -------------------------------- #
        act_bytes = w.activation_bytes()
        t_coll = self.fabric.tp_allreduce_ns(
            act_bytes, n_tp,
            int(cfg["parallelism"]["tp_collectives_per_layer"])) * 1e-9
        t_xfer = (self.fabric.transfer_ns(act_bytes) * 1e-9) if groups > 1 else 0.0

        pl = cfg["pipeline"]
        ic = cfg["interconnect"]
        t_stage = (t_block + t_coll) * layers + t_xfer
        if groups > 1:
            # Pipeline bubbles and stage imbalance only exist across groups.
            t_stage /= pl["efficiency"]
        t_stage += ic["scheduling_overhead_ms"] * 1e-3
        t_stage /= short_sys
        t_iter = t_stage * groups
        thr = w.batch * m / (t_stage * groups)

        # xPU duty is measured against the command-limited (pre-derate)
        # schedule: short-payload command stalls occupy, not idle, the engine.
        t_ref = t_stage * short_sys
        counters.mode_switches = modes["broadcast"] and 1 or 0
        cnt = self._counters(w, q, by, modes, layers, n_tp, groups, m, devices,
                             t_ref, t_coll, t_xfer, imbalance, counters,
                             chosen, elem, shard)
        x = by["xpu"] / max(by["xpu"] + by["pim"], 1.0)
        res = Result(True, cfg["name"], throughput=thr, t_iter_ms=t_iter * 1e3,
                     counters=cnt,
                     notes=f"TP={n_tp} x PP={groups}, m={m} in-flight, "
                           f"x*={x:.2f} weight bytes to xPU")
        self._energy(res, w, cnt)
        res.placements = [self.sched.place(op, enable_broadcast="rd" in f)
                          for op in w.block]
        return res

    # ---------------- Proteus-Static ----------------------------------- #
    def deployment_plan(self, w: Workload, devices=None):
        """Freeze the whole operator schedule at deployment time.

        This is the Proteus-Static configuration of Sec. V-C: identical
        hardware -- reconfigurable datapath, memory-side fusion, SFUs -- but
        the substrate assignment, the connectivity mode and the co-execution
        split are all decided once for a deployment-time reference workload
        and never re-derived. Serving then drifts away from that reference as
        continuous batching changes the batch, the context and the routing.

        The plan is derived exactly as the runtime would derive it for the
        reference workload, including the block-level choice of connectivity
        mode, so the two configurations differ only in *when* the decision is
        taken.
        """
        devices = devices or self.cfg["devices"]
        n_tp, _ = self.topology(devices)
        shard = 1.0 / n_tp
        f = self.features
        smallf = small_op_efficiency(self.cfg["scheduler"]["small_op_efficiency"],
                                     max(w.tokens_per_expert, 1.0))
        pbw_d = self.dmem.internal_bw
        pbw_b = self.dmem.broadcast_bw
        if "as" not in f:
            pbw_d *= self.k_cmd
            pbw_b *= self.k_cmd
        compute = [op for op in w.block if op.kind != "elementwise"]
        elem = [op for op in w.block if op.kind == "elementwise"]
        stall = 0.0 if "of" in f else self.f_unfused
        unfused = (sum(o.bytes * shard * 2.0 for o in elem) / pbw_d
                   if ("of" not in f and elem) else 0.0)

        best = None
        for pref in (["auto"] if "rd" not in f else ["direct", "broadcast"]):
            mode = "direct" if pref == "auto" else pref
            xbw = self.coexec[mode] if ("rd" in f and self.coexec[mode] > 0) \
                else self.xpu_bw
            if "as" not in f:
                xbw *= self.k_sync
            costs = [self._op_cost(op, shard, pbw_d, pbw_b, self.dmem.pe_flops,
                                   xbw, smallf, f, pref) for op in compute]
            q, _, modes, chosen = self._schedule_block(costs, f,
                                                       SchedulerCounters())
            t = self._compose(q, modes, stall, f) + unfused
            if best is None or t < best[0]:
                best = (t, chosen)
        plan = {}
        for c, sub, frac in best[1]:
            plan.setdefault(c.name, []).append((sub, c.pim_mode, frac))
        return plan

    def _replay_plan(self, costs, plan):
        """Execute a frozen deployment-time schedule (Proteus-Static).

        The plan pins the substrate, the connectivity mode and the fraction of
        every operator assigned to each, so an operator whose runtime reuse no
        longer matches the frozen decision pays for it: direct mode re-streams
        the operand once per vector, broadcasting streams at the fan-in
        cadence without the reuse to amortize it, and a frozen split leaves
        one substrate idle while the other runs long."""
        q = {"xpu": 0.0, "pim": 0.0}
        bytes_ = {"xpu": 0.0, "pim": 0.0}
        modes = {"direct": 0, "broadcast": 0}
        chosen = []
        for c in costs:
            for sub, mode, frac in plan.get(c.name,
                                            [("pim", c.pim_mode, 1.0)]):
                if frac <= 0.0:
                    continue
                q[sub] += c.t_on(sub, mode) * frac
                bytes_[sub] += c.bytes_on(sub, mode) * frac
                if sub == "pim":
                    modes[mode] += 1
                chosen.append([c, sub, frac])
        return q, bytes_, modes, chosen

    # ---------------- [PP, DP] hybrid ---------------------------------- #
    def _simulate_dp(self, w: Workload, devices, dp):
        """[PP, DP] hybrid at a fixed total batch (Sec. V-F): each data-parallel
        replica owns devices/dp devices and serves batch/dp requests, but
        independently holds a full copy of the model."""
        sub_devices = max(devices // dp, 1)
        sub = build_workload(w.model, max(w.batch // dp, 1), 0, 0,
                             ctx_override=w.ctx_avg)
        sub.ctx_peak = w.ctx_peak
        sub.peak_mem = w.model["weight_bytes"] + \
            sub.batch * w.ctx_peak * w.model["kv_bytes_per_token"]
        r = self.simulate(sub, devices=sub_devices)
        if not r.alive:
            return r
        r.throughput *= dp
        r.notes += f" [replicas={dp} x {sub_devices} devices]"
        r.counters["devices"] = devices
        r.counters["replicas"] = dp
        return r

    # ---------------- bookkeeping -------------------------------------- #
    def _counters(self, w, q, by, modes, layers, n_tp, groups, m, devices,
                  t_ref, t_coll, t_xfer, imbalance, sched, chosen, elem,
                  shard):
        scale = layers * n_tp * groups        # per-device-block -> per system
        pim_flops = sum(c.op_flops * frac for c, sub, frac in chosen
                        if sub == "pim") * shard * scale
        bcast_bytes = sum(c.bytes_on("pim") * frac for c, sub, frac in chosen
                          if sub == "pim" and c.pim_mode == "broadcast") * scale
        sfu_flops = sum(o.flops for o in elem) * shard * scale
        # Share of the *single-pass* shared-operand traffic placed on the xPU.
        # Unlike x*, which is measured on the re-read-inflated PIM traffic,
        # this is the fraction of the actual weight operands the xPU streams,
        # and it is what the integrated co-simulation replays.
        gemm_bytes = sum(c.bytes_xpu * frac for c, _, frac in chosen
                         if c.kind == "weight_gemm")
        xpu_gemm_bytes = sum(c.bytes_xpu * frac for c, sub, frac in chosen
                             if sub == "xpu" and c.kind == "weight_gemm")
        d = dict(
            pim_flops=pim_flops,
            xpu_weight_share=xpu_gemm_bytes / max(gemm_bytes, 1e-30),
            pim_broadcast_bytes=bcast_bytes,
            sfu_flops=sfu_flops,
            link_bytes=(self.fabric.tp_allreduce_bytes(
                w.activation_bytes(), n_tp,
                int(self.cfg["parallelism"]["tp_collectives_per_layer"]))
                * layers + (w.activation_bytes() if groups > 1 else 0.0))
                * devices,
            xpu_ext_bytes=by["xpu"] * layers * n_tp * groups,
            pim_bytes=by["pim"] * layers * n_tp * groups,
            tp_width=n_tp, pipeline_groups=groups, layers_per_stage=layers,
            layer_imbalance=imbalance,
            inflight=m, devices=devices,
            direct_ops=modes["direct"], broadcast_ops=modes["broadcast"],
            xpu_duty=min(q["xpu"] * layers / t_ref, 1.0) if t_ref else 0.0,
            pim_duty=min(q["pim"] * layers / t_ref, 1.0) if t_ref else 0.0,
            collective_bytes_per_device=self.fabric.tp_allreduce_bytes(
                w.activation_bytes(), n_tp,
                int(self.cfg["parallelism"]["tp_collectives_per_layer"])) * layers,
            collective_ms=t_coll * layers * 1e3,
            stage_transfer_ms=t_xfer * 1e3,
            activation_bytes_per_stage=w.activation_bytes(),
        )
        d.update(sched.as_dict())
        return d

    def _energy(self, res, w, c):
        """Activity-based system energy (Sec. V-B).

        DRAM traffic is charged per byte at the near-bank or external
        pJ/bit of the substrate; near-bank reads terminate at the bank-local
        PE and therefore exclude the I/O and PHY component. Near-bank compute
        is charged per 16-lane MAC issue at the synthesis-derived energy of
        the PE, and broadcasting additionally pays the BG-local distribution
        of every reused burst. The xPU is charged a utilization-interpolated
        engine power, and each device's CXL port a link power scaled by the
        activation traffic it actually carries.
        """
        en = self.cfg["energy"]
        mem = self.mem
        e_ext = mem["energy_pj_per_bit"]["external"] * 8e-12
        e_int = mem["energy_pj_per_bit"]["near_bank"] * 8e-12
        cmd = mem["command_energy_pj"]
        devices = c["devices"]
        duty = c["xpu_duty"]

        # --- static and utilization-scaled power ------------------------ #
        bg = self.dmem.capacity * devices / 1e9 * en["background_w_per_gb"] \
            * en["background_idle_factor"]
        link = en.get("link_w_per_device", 0.0) * devices
        p = devices * (en["xpu_busy_w"] * duty
                       + en["xpu_idle_w"] * (1.0 - duty)) \
            + en["static_w"] + bg + link

        # --- per-token energy of the memory and near-bank datapath ------ #
        burst = self.dmem.burst_bytes
        mac_issue_flops = 2.0 * mem["pe_lanes"]        # one 16-lane FP16 issue
        pim_mac_j = c["pim_flops"] / mac_issue_flops * cmd["mac_op"] * 1e-12
        bcast_j = c["pim_broadcast_bytes"] / burst \
            * cmd.get("bg_broadcast", 0.0) * 1e-12
        sfu_j = c["sfu_flops"] / mac_issue_flops * cmd["mac_op"] * 1e-12
        dram_j = c["xpu_ext_bytes"] * e_ext + c["pim_bytes"] * e_int
        link_j = c["link_bytes"] * en.get("link_pj_per_bit", 2.0) * 8e-12

        j_per_token = (dram_j + pim_mac_j + bcast_j + sfu_j + link_j) / w.batch
        res.power_w = p + (pim_mac_j + bcast_j + sfu_j + link_j) \
            / max(res.t_iter_ms * 1e-3, 1e-12)
        res.tokens_per_joule = 1.0 / (p / res.throughput + j_per_token)


def build_system(name, features=FULL):
    """Factory: build a system model by config name."""
    from .config import load_system, load_memory
    cfg = load_system(name)
    kind = cfg["kind"]
    if kind == "proteus":
        return ProteusSystem(cfg, load_memory(cfg["memory"]), features=features)
    from . import baselines
    return baselines.build(kind, cfg)
