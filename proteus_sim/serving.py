"""Request-level serving simulation (Sec. V-C, Fig. 13).

Two drivers share one iteration-level timing model.

``SloServingSimulator`` -- *open loop*. Requests arrive from a trace. Every
request reserves peak KV capacity for its full input plus output length
before admission and waits in FIFO order until that capacity is free;
admitted requests decode together under continuous batching, up to a
concurrency limit. The metric is a request's **average per-token latency**,
completion time minus *arrival* time divided by output length, so queueing
delay is charged to every token; SLO attainment is the fraction of requests
whose per-token latency does not exceed the SLO. Sweeping the offered load
(by scaling inter-arrival times uniformly) traces the latency and attainment
curves of Fig. 13 and locates each system's sustainable load.

The simulation is decode-only: the trace's input lengths set the resident KV
footprint and the admission reservation, and the reported latency covers the
autoregressive decode phase, matching the output-dominated workload of
Sec. V-A.

``ServingSimulator`` -- *closed loop*. A fixed pool of concurrent requests
decodes one token per iteration and completions are replaced immediately, so
batch composition, aggregate context and per-expert token counts drift every
iteration. Before each iteration the runtime re-samples MoE routing,
recomputes the workload intensities and re-derives the placement: the driver
records how the co-execution split, the per-expert crossover decision and
the PIM connectivity respond to that drift.

Both drivers work with any system model in this package -- Proteus, the
frozen-mapping Proteus-Static, and the six baselines -- because they only
require ``simulate()`` and ``kv_capacity()``.
"""
import math
import random
from collections import deque
from dataclasses import dataclass

from .system import ProteusSystem
from .workload import build_workload


# --------------------------------------------------------------------- #
# Open-loop, SLO-oriented driver
# --------------------------------------------------------------------- #
@dataclass
class Request:
    rid: int
    arrival_s: float
    prompt: int
    target_out: int
    admit_s: float = -1.0
    finish_s: float = -1.0
    generated: int = 0

    @property
    def ctx(self):
        return self.prompt + self.generated

    @property
    def reservation_tokens(self):
        """Peak context the admission controller must reserve."""
        return self.prompt + self.target_out

    def per_token_latency_ms(self):
        return (self.finish_s - self.arrival_s) / self.target_out * 1e3


@dataclass
class ServeReport:
    system: str
    offered_tokens_s: float = 0.0
    achieved_tokens_s: float = 0.0
    mean_per_token_ms: float = 0.0
    p99_per_token_ms: float = 0.0
    slo_attainment: float = 0.0
    mean_batch: float = 0.0
    mean_queue_s: float = 0.0
    completed: int = 0
    admitted: int = 0
    saturated: bool = False

    def row(self):
        return [self.system, round(self.offered_tokens_s),
                round(self.achieved_tokens_s),
                round(self.mean_per_token_ms, 3),
                round(self.p99_per_token_ms, 3),
                round(self.slo_attainment, 4), round(self.mean_batch, 2),
                round(self.mean_queue_s, 3), self.completed]

    HEADER = ["system", "offered_tokens_s", "achieved_tokens_s",
              "mean_per_token_ms", "p99_per_token_ms", "slo_attainment",
              "mean_batch", "mean_queue_s", "completed"]


class SloServingSimulator:
    """Open-loop continuous-batching driver with KV admission control."""

    #: context quantization used by the iteration-time memo (tokens). The
    #: iteration time is smooth in the mean context, so quantizing it bounds
    #: the model error while collapsing millions of identical evaluations.
    CTX_QUANTUM = 256
    MAX_JUMP = 64                 # iterations advanced per scheduling decision

    def __init__(self, system, model, trace, max_concurrent=64,
                 slo_ms=30.0, load_scale=1.0, frozen_plan=None,
                 devices=None, horizon_s=None):
        self.sys = system
        self.model = model
        self.max_concurrent = max_concurrent
        self.slo_ms = slo_ms
        self.frozen_plan = frozen_plan
        self.devices = devices
        self.horizon_s = horizon_s
        self.kv_per_token = model["kv_bytes_per_token"]
        self.kv_budget = system.kv_capacity(model, devices)
        self.pending = deque(
            Request(rid=i, arrival_s=a / 1e3 / load_scale, prompt=p,
                    target_out=o)
            for i, (a, p, o) in enumerate(trace))
        self.offered_tokens_s = 0.0
        if self.pending:
            span = self.pending[-1].arrival_s - self.pending[0].arrival_s
            if span > 0:
                self.offered_tokens_s = \
                    sum(r.target_out for r in self.pending) / span
        self._memo = {}

    # -- iteration timing ------------------------------------------------ #
    def _iteration_ms(self, batch, mean_ctx, peak_tokens):
        key = (batch,
               int(mean_ctx // self.CTX_QUANTUM),
               int(peak_tokens // (self.CTX_QUANTUM * 8)))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        w = build_workload(self.model, batch, 0, 0,
                           ctx_override=max(int(mean_ctx), 1))
        w.ctx_peak = max(int(peak_tokens // max(batch, 1)), 1)
        w.peak_mem = self.model["weight_bytes"] + peak_tokens * self.kv_per_token
        kw = {}
        if self.devices:
            kw["devices"] = self.devices
        if self.frozen_plan is not None:
            kw["frozen_plan"] = self.frozen_plan
        r = self.sys.simulate(w, **kw)
        val = r.t_iter_ms / 1e3 if r.alive else None
        self._memo[key] = val
        return val

    # -- main loop -------------------------------------------------------- #
    def run(self):
        t = 0.0
        running = []
        reserved = 0.0
        done = []
        batch_time = 0.0
        batch_weight = 0.0
        stall = 0

        while self.pending or running:
            if self.horizon_s and t > self.horizon_s:
                break
            # --- admission: FIFO, reserving peak KV for the full length --- #
            while self.pending and len(running) < self.max_concurrent:
                r = self.pending[0]
                if r.arrival_s > t:
                    break
                need = r.reservation_tokens * self.kv_per_token
                if reserved + need > self.kv_budget:
                    break                      # head-of-line: wait for capacity
                self.pending.popleft()
                r.admit_s = t
                reserved += need
                running.append(r)

            if not running:
                if not self.pending:
                    break
                t = max(t, self.pending[0].arrival_s)
                continue

            batch = len(running)
            mean_ctx = sum(r.ctx for r in running) / batch
            peak_tokens = sum(r.reservation_tokens for r in running)
            dt = self._iteration_ms(batch, mean_ctx, peak_tokens)
            if dt is None:                     # configuration cannot host it
                return ServeReport(self.sys.cfg["name"], saturated=True,
                                   offered_tokens_s=self.offered_tokens_s)

            # --- advance to the next scheduling event -------------------- #
            steps_to_finish = min(r.target_out - r.generated for r in running)
            steps = min(steps_to_finish, self.MAX_JUMP)
            if self.pending and len(running) < self.max_concurrent:
                nxt = self.pending[0].arrival_s
                if nxt > t:
                    steps = max(1, min(steps, int(math.ceil((nxt - t) / dt))))
            # keep the quantized iteration time valid over the jump
            steps = max(1, min(steps, max(1, int(self.CTX_QUANTUM / 2))))

            t += dt * steps
            batch_time += dt * steps
            batch_weight += batch * dt * steps
            for r in running:
                r.generated += steps
            still = []
            for r in running:
                if r.generated >= r.target_out:
                    r.generated = r.target_out
                    r.finish_s = t
                    reserved -= r.reservation_tokens * self.kv_per_token
                    done.append(r)
                else:
                    still.append(r)
            if len(still) == len(running):
                stall += 1
                if stall > 5_000_000:
                    break
            else:
                stall = 0
            running = still

        return self._report(done, t, batch_weight, batch_time)

    def _report(self, done, t_end, batch_weight, batch_time):
        rep = ServeReport(self.sys.cfg["name"],
                          offered_tokens_s=self.offered_tokens_s)
        rep.completed = len(done)
        rep.admitted = len(done)
        if not done:
            rep.saturated = True
            return rep
        lat = sorted(r.per_token_latency_ms() for r in done)
        rep.mean_per_token_ms = sum(lat) / len(lat)
        rep.p99_per_token_ms = lat[min(len(lat) - 1, int(0.99 * len(lat)))]
        rep.slo_attainment = sum(1 for x in lat if x <= self.slo_ms) / len(lat)
        rep.achieved_tokens_s = sum(r.target_out for r in done) / max(t_end, 1e-9)
        rep.mean_batch = batch_weight / max(batch_time, 1e-12)
        rep.mean_queue_s = sum(r.admit_s - r.arrival_s for r in done) / len(done)
        rep.saturated = rep.slo_attainment < 0.90
        return rep


def knee_load(reports, attainment=0.90):
    """Largest offered load a system sustains at the SLO, read off its
    attainment curve.

    `reports` must be in sweep order (increasing offered load). The sweep
    only samples the load axis, so the last point that still meets the target
    attainment underestimates the knee; interpolating between it and the
    first failing point recovers the crossing, which is what "sustainable
    load" means: the delivered load at which SLO attainment falls to the
    target."""
    pts = [(r.achieved_tokens_s, r.slo_attainment) for r in reports
           if r.completed]
    best = 0.0
    for (l0, a0), (l1, a1) in zip(pts, pts[1:]):
        if a0 >= attainment > a1 and a0 > a1:
            best = max(best, l0 + (l1 - l0) * (a0 - attainment) / (a0 - a1))
    if not best:
        best = max((l for l, a in pts if a >= attainment), default=0.0)
    return best


def sustainable_load(system, model, trace, max_concurrent=64, slo_ms=30.0,
                     attainment=0.90, scales=None, frozen_plan=None,
                     devices=None, horizon_s=None):
    """Largest offered load at which the system still meets `attainment` of
    the SLO, found by sweeping the inter-arrival scaling factor.

    Returns ``(sustainable_tokens_s, [ServeReport, ...])``."""
    scales = scales or [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    reports = [SloServingSimulator(system, model, trace,
                                   max_concurrent=max_concurrent,
                                   slo_ms=slo_ms, load_scale=s,
                                   frozen_plan=frozen_plan, devices=devices,
                                   horizon_s=horizon_s).run()
               for s in scales]
    return knee_load(reports, attainment), reports


# --------------------------------------------------------------------- #
# Closed-loop driver: runtime-adaptation dynamics
# --------------------------------------------------------------------- #
@dataclass
class IterationRecord:
    it: int
    batch: int
    mean_ctx: float
    t_iter_ms: float
    throughput: float
    tokens_per_expert: float
    x_split: float
    experts_on_xpu: int
    experts_on_pim: int
    placement_switches: int
    remaps: int
    completed: int


class ServingSimulator:
    """Closed-loop continuous-batching driver for a ProteusSystem."""

    def __init__(self, system: ProteusSystem, model, max_batch=32,
                 prompt_mean=2048, out_mean=6144, seed=13,
                 request_source=None):
        assert isinstance(system, ProteusSystem)
        self.sys = system
        self.model = model
        self.max_batch = max_batch
        self.rng = random.Random(seed)
        self.prompt_mean = prompt_mean
        self.out_mean = out_mean
        self._next_rid = 0
        self._source = list(request_source) if request_source else None
        self.pool = []
        while len(self.pool) < max_batch:
            r = self._new_request()
            if r is None:
                break
            self.pool.append(r)
        self._prev_placement = None
        self._w = None

    @classmethod
    def from_trace(cls, system, model, path, max_batch=32, seed=13):
        """Build a simulator whose request pool replays a request trace."""
        from trace_gen.gen_requests import read_trace
        rows = read_trace(path)
        return cls(system, model, max_batch=max_batch, seed=seed,
                   request_source=[(p, o) for _, p, o in rows])

    def _new_request(self):
        if self._source is not None:
            if not self._source:
                return None                      # trace exhausted
            prompt, out = self._source.pop(0)
        else:
            prompt = max(64, int(self.rng.lognormvariate(0, 0.35)
                                 * self.prompt_mean))
            out = max(32, int(self.rng.lognormvariate(0, 0.45)
                              * self.out_mean))
        r = Request(rid=self._next_rid, arrival_s=0.0, prompt=prompt,
                    target_out=out)
        self._next_rid += 1
        return r

    def _placement_signature(self):
        """Per-expert crossover decision under this iteration's routing."""
        moe = self.model["moe"]
        if not moe["enabled"] or not self._w.routing_hist:
            return ()
        from .scheduler import shared_operand_ai
        d = self.model["d_model"]
        theta = self.sys.sched.theta
        return tuple(1 if shared_operand_ai(n, d) > theta else 0
                     for n in self._w.routing_hist if n > 0)

    def step(self, it):
        batch = [r for r in self.pool if r.generated < r.target_out][:self.max_batch]
        if not batch:
            return None                          # request trace fully drained
        mean_ctx = sum(r.ctx for r in batch) / len(batch)
        peak_ctx = max(r.ctx for r in batch)
        self._w = build_workload(self.model, len(batch), 0, 0,
                                 routing="sampled",
                                 seed=self.rng.randrange(1 << 30),
                                 ctx_override=int(mean_ctx))
        self._w.ctx_peak = peak_ctx
        self._w.peak_mem = self.model["weight_bytes"] + \
            len(batch) * peak_ctx * self.model["kv_bytes_per_token"]
        res = self.sys.simulate(self._w)
        sig = self._placement_signature()
        switches = 0
        if self._prev_placement is not None:
            switches = sum(1 for a, b in zip(sig, self._prev_placement)
                           if a != b) + abs(len(sig) - len(self._prev_placement))
        self._prev_placement = sig
        on_xpu = sum(sig) if sig else 0

        completed = 0
        for r in batch:
            r.generated += 1
            if r.generated >= r.target_out:
                completed += 1
        self.pool = [r for r in self.pool if r.generated < r.target_out]
        while len(self.pool) < self.max_batch:
            nr = self._new_request()
            if nr is None:
                break
            self.pool.append(nr)

        x = float(res.notes.split("x*=")[1].split()[0]) \
            if "x*=" in res.notes else 1.0
        return IterationRecord(
            it=it, batch=len(batch), mean_ctx=mean_ctx,
            t_iter_ms=res.t_iter_ms, throughput=res.throughput,
            tokens_per_expert=self._w.tokens_per_expert, x_split=x,
            experts_on_xpu=on_xpu, experts_on_pim=len(sig) - on_xpu,
            placement_switches=switches,
            remaps=res.counters.get("sched_remaps", 0), completed=completed)

    def run(self, iterations=300):
        out = []
        for i in range(iterations):
            rec = self.step(i)
            if rec is None:
                break
            out.append(rec)
        return out
