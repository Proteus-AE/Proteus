"""Sanity and self-consistency checks (mirrors the validation layers of
Sec. V-A: physical bounds, monotonicity, OOM accounting, crossover closure).

Runs under pytest (`pytest tests/`) or standalone (`python tests/test_sanity.py`).
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proteus_sim import load_model, build_system            # noqa: E402
from proteus_sim.workload import build_workload             # noqa: E402
from proteus_sim.system import VARIANTS                     # noqa: E402

MODELS = ["deepseek-v2-lite", "switch-26b", "mixtral-8x7b", "llama3-70b"]
SYSTEMS = ["dgx-a100", "cxl-pnm", "cent", "neupims", "papi", "pimphony", "proteus"]
CTX = (2048, 6144)


def run(system, model, batch, **kw):
    w = build_workload(load_model(model), batch, *CTX,
                       routing=kw.pop("routing", "expected"),
                       seed=kw.pop("seed", 7))
    return build_system(system, features=VARIANTS[kw.pop("variant", "full")]) \
        .simulate(w, **kw)


def test_batch_monotonicity():
    """Throughput non-decreasing in batch (among runnable cells); OOM only
    appears once batch grows, and never reverses."""
    for system in SYSTEMS:
        for model in MODELS:
            prev, seen_oom = 0.0, False
            for b in [16, 32, 64]:
                r = run(system, model, b)
                if not r.alive:
                    seen_oom = True
                    continue
                assert not seen_oom, f"{system}/{model}: revives after OOM"
                assert r.throughput >= prev * 0.999, f"{system}/{model} b={b}"
                prev = r.throughput


def test_derived_machine_matches_table_iii():
    """The derived organization must reproduce Table III exactly."""
    sys_ = build_system("proteus")
    d = sys_.dmem
    assert d.channels == 64 and d.dies == 64
    assert d.bankgroups == 256 and d.banks == 1024
    assert abs(d.capacity / 256e9 - 1) < 1e-9            # 256 GB/device
    assert abs(d.internal_peak / 16e12 - 1) < 0.03       # 16 TB/s internal
    assert abs(d.pe_flops_peak / 32e12 - 1) < 0.03       # 32 TFLOPS PE
    assert abs(d.external_peak / 1e12 - 1) < 1e-9        # 1 TB/s external
    # Ridge points of Sec. III-B / IV-D.
    assert abs(d.ridge_pim - 2.0) < 0.02
    assert abs(sys_.sched.ai_xpu - 312.0) < 0.5
    # theta is the estimator's view of the same machine, so the configured
    # value must be F_PIM/BW_out and not an independent tuning knob.
    assert abs(sys_.sched.theta - d.pe_flops_peak / d.external_peak) < 1e-9
    assert abs(sys_.sched.theta / 32.0 - 1) < 0.03
    # Broadcasting trades raw bandwidth for a fanout-fold reuse and lands
    # exactly on the PE peak.
    assert abs(d.broadcast_peak * d.fanout / d.pe_flops_peak - 1) < 1e-9


def test_proteus_physical_bounds():
    """Sustained rates must not exceed the derived machine parameters."""
    sys_ = build_system("proteus")
    for model in MODELS:
        w = build_workload(load_model(model), 32, *CTX)
        r = sys_.simulate(w)
        t_iter = r.t_iter_ms * 1e-3
        devices = r.counters["devices"]
        pim_bytes_dev = r.counters["pim_bytes"] / devices
        assert pim_bytes_dev / t_iter <= sys_.dmem.internal_bw * 1.001, model
        xpu_bytes_dev = r.counters["xpu_ext_bytes"] / devices
        assert xpu_bytes_dev / t_iter <= sys_.xpu_bw * 1.001, model


def test_memory_hierarchy_aligned_topology():
    """Eight devices form one tensor-parallel group; more devices pipeline
    whole layers across groups (Sec. IV-B, Fig. 6)."""
    sys_ = build_system("proteus")
    w = build_workload(load_model("llama3-405b"), 16, 0, 0, ctx_override=32768)
    for devices, tp, groups in [(8, 8, 1), (16, 8, 2), (32, 8, 4), (64, 8, 8)]:
        c = sys_.simulate(w, devices=devices).counters
        assert (c["tp_width"], c["pipeline_groups"]) == (tp, groups)
        # whole layers, trailing groups one short, fullest group sets the pace
        assert c["layers_per_stage"] * groups >= w.n_layers
        assert (c["layers_per_stage"] - 1) * groups < w.n_layers


def test_tensor_parallel_collective_cost():
    """The ring AllReduce volume and time must match the analysis of
    Sec. IV-B: 147 MB per device per decode iteration for Llama-3.1-70B at
    d = 8192, b = 32 over eight devices, 1.15 ms of payload on one x16 port
    plus the 2(N-1) hop latencies of each collective."""
    sys_ = build_system("proteus")
    w = build_workload(load_model("llama3-70b"), 32, *CTX)
    c = sys_.simulate(w).counters
    assert abs(c["collective_bytes_per_device"] / 147e6 - 1) < 0.05
    payload_ms = 147e6 / sys_.fabric.link_bw * 1e3
    hops_ms = 80 * 2 * 2 * 7 * sys_.fabric.lat_ns * 1e-6
    assert abs(c["collective_ms"] / (payload_ms + hops_ms) - 1) < 0.05
    # a cross-group activation transfer is a single 512 KiB message, ~4 us
    r = sys_.simulate(w, devices=16)
    assert abs(r.counters["stage_transfer_ms"] * 1e3 / 4.1 - 1) < 0.25


def test_oom_boundaries():
    """Capacity accounting reproduces the OOM pattern of Sec. V-B."""
    assert not run("cent", "llama3-70b", 16).alive       # weights > 8x16 GB
    assert run("neupims", "llama3-70b", 32).alive
    assert not run("neupims", "llama3-70b", 64).alive    # KV growth
    assert not run("papi", "llama3-70b", 64).alive       # KV > Attn-PIM pool
    assert run("dgx-a100", "llama3-70b", 64).alive
    assert run("proteus", "llama3-70b", 64).alive        # 4 TB LPDDR headroom


def test_crossover_self_consistency():
    """At the analytical crossover (AI_w ~ F_PIM/BW_out) the two substrates
    must independently produce similar execution times (Sec. IV-D)."""
    from proteus_sim.scheduler import shared_operand_ai
    model = load_model("mixtral-8x7b")
    w = build_workload(model, 128, *CTX)     # ~32 tokens/expert -> AI_w ~ 32
    n = w.tokens_per_expert
    sys_ = build_system("proteus")
    ai = shared_operand_ai(n, model["d_model"])
    assert abs(ai / sys_.sched.theta - 1) < 0.05
    # theta = F_PIM / BW_out is a ratio of peak rates, so the balance point it
    # defines is stated on peak rates as well; the sustained rates of the two
    # substrates differ by their own streaming efficiencies.
    t_x = w.weight_bytes / sys_.xpu_bw_peak / 8
    t_p = max(w.weight_flops / sys_.dmem.pe_flops_peak,
              w.weight_bytes * math.ceil(n / 4) /
              sys_.dmem.broadcast_peak) / 8
    assert abs(t_x / t_p - 1) < 0.20


def test_crossover_threshold_drives_placement():
    """theta is the estimator's view of the machine (F_PIM/BW_out), so a
    badly mis-estimated threshold must cost throughput and cannot be
    repaired at runtime, while a slightly imprecise one costs little
    (Fig. 15)."""
    w = build_workload(load_model("llama3-70b"), 64, *CTX)

    def thr(theta):
        s = build_system("proteus")
        s.sched.theta = theta
        return s.simulate(w).throughput

    nominal = thr(32.0)
    assert thr(0.9 * 32.0) / nominal > 0.90
    assert thr(1.1 * 32.0) / nominal > 0.90
    assert thr(2.0) / nominal < 0.75          # xPU-centric misplacement
    assert thr(312.0) / nominal < 0.75        # PIM-centric misplacement


def test_static_mapping_is_worse():
    """Proteus-Static keeps the hardware and freezes only the mapping, so it
    must never beat the adaptive runtime and must lose ground as the runtime
    state drifts from the deployment-time reference (Sec. V-C)."""
    model = load_model("mixtral-8x7b")
    sys_ = build_system("proteus")
    plan = sys_.deployment_plan(build_workload(model, 64, 0, 0,
                                               ctx_override=2048))
    worse = 0
    for b in [1, 4, 16, 64]:
        w = build_workload(model, b, *CTX)
        adaptive = sys_.simulate(w).throughput
        static = sys_.simulate(w, frozen_plan=plan).throughput
        assert static <= adaptive * 1.001, b
        worse += static < adaptive * 0.999
    assert worse, "a frozen mapping never lost anything -- check the plan"


def test_variant_chain_monotone():
    """Each cumulative mechanism must not reduce throughput (Sec. V-C)."""
    for model in ["mixtral-8x7b", "llama3-70b"]:
        prev = 0.0
        for v in ["base", "as", "rd", "of", "ec"]:
            thr = run("proteus", model, 32, variant=v).throughput
            assert thr >= prev * 0.999, f"{model}: variant +{v} regressed"
            prev = thr


def test_sampled_routing_is_pessimistic():
    """Per-iteration sampled routing must sit *below* the uniform-routing
    expectation, and by a bounded amount.

    The number of broadcasting passes an expert needs is ceil(n/fanout) in
    its own token count n. That function is convex, so by Jensen's inequality
    the mean over a drawn routing histogram exceeds the value at the mean
    token count: the closed-form expectation is optimistic, and the runtime's
    per-iteration histogram (Sec. IV-E) is what exposes the real cost. The
    expectation is still the reproducible default for the sweeps; this test
    pins how much it flatters them."""
    thr_exp = run("proteus", "mixtral-8x7b", 32).throughput
    vals = [run("proteus", "mixtral-8x7b", 32, routing="sampled",
                seed=s).throughput for s in range(60)]
    thr_smp = sum(vals) / len(vals)
    assert thr_smp < thr_exp
    assert thr_smp / thr_exp > 0.80
    assert max(vals) <= thr_exp * 1.001


def test_energy_sane():
    """tokens/J positive; tokens/J non-decreasing with batch (amortization)."""
    prev = 0.0
    for b in [16, 32, 64]:
        r = run("proteus", "mixtral-8x7b", b)
        assert r.tokens_per_joule > 0 and r.power_w > 500
        assert r.tokens_per_joule >= prev * 0.999
        prev = r.tokens_per_joule


def test_serving_simulation():
    """Closed-loop continuous batching must track the steady-state operating
    point and produce well-formed per-iteration records (Sec. IV-D)."""
    from proteus_sim.serving import ServingSimulator
    model = load_model("mixtral-8x7b")
    sim = ServingSimulator(build_system("proteus"), model, max_batch=8,
                           prompt_mean=1024, out_mean=128, seed=3)
    recs = sim.run(80)
    assert all(r.throughput > 0 and r.batch <= 8 for r in recs)
    warm = recs[20:]
    thr = sum(r.throughput for r in warm) / len(warm)
    w = build_workload(model, 8, 0, 0,
                       ctx_override=int(sum(r.mean_ctx for r in warm)
                                        / len(warm)))
    ref = build_system("proteus").simulate(w).throughput
    assert abs(thr / ref - 1) < 0.30
    assert sum(r.completed for r in recs) > 0


def test_xpucore_tile_model():
    """Tile schedules must respect the array/SRAM bounds, conserve FLOPs,
    and classify decode GEMMs as memory-bound (the premise of Sec. III)."""
    from proteus_sim.xpucore import SystolicConfig, XpuEngine
    eng = XpuEngine(SystolicConfig())
    assert abs(eng.cfg.peak_flops / 312e12 - 1) < 0.02   # A100-class
    t = eng.run_op("ffn_up", 32, 8192, 57344, dram_bw=0.85e12)
    s = t.schedule
    assert s.tk <= eng.cfg.rows and s.tn <= eng.cfg.cols
    assert (s.tm * (s.tk + s.tn) * eng.cfg.dtype_bytes
            <= eng.cfg.sram_bytes // 2)
    assert s.flops == 2.0 * 32 * 8192 * 57344
    assert t.memory_bound                    # decode GEMV/skinny is BW-bound
    assert s.weight_dram_bytes >= 8192 * 57344 * 2
    # a large-M GEMM at full bandwidth must become compute-bound
    big = eng.run_op("prefill", 8192, 8192, 8192, dram_bw=2.0e12)
    assert not big.memory_bound


def test_serving_from_trace():
    """Trace-driven serving must replay deterministically and drain."""
    from proteus_sim.serving import ServingSimulator
    model = load_model("mixtral-8x7b")
    src = [(1024, 40)] * 12                  # 12 short requests
    sim = ServingSimulator(build_system("proteus"), model, max_batch=4,
                           request_source=src, seed=5)
    recs = sim.run(400)
    assert sum(r.completed for r in recs) == 12   # all requests finish
    assert recs[-1].batch >= 1
    assert len(recs) < 400                   # drained before the cap


def test_device_scaling_near_linear():
    r1 = run("proteus", "llama3-70b", 32, devices=1)
    r16 = run("proteus", "llama3-70b", 32, devices=16)
    speedup = r16.throughput / r1.throughput
    assert 12.0 < speedup <= 16.0


def test_slo_serving_admission_and_ordering():
    """The open-loop driver must respect KV admission, charge queueing delay
    to every token, and rank the systems the way Sec. V-C reports."""
    from proteus_sim.serving import SloServingSimulator
    from trace_gen.gen_requests import generate
    model = load_model("mixtral-8x7b")
    trace = generate("azure", 0, 1.2, 2048, 6144, 5, 600.0)
    reps = {}
    for name in ["dgx-a100", "cent", "pimphony", "proteus"]:
        reps[name] = SloServingSimulator(build_system(name), model, trace,
                                         max_concurrent=64, slo_ms=30.0,
                                         load_scale=0.5).run()
    for name, rep in reps.items():
        assert rep.completed > 0, name
        assert 0.0 <= rep.slo_attainment <= 1.0
        assert rep.achieved_tokens_s <= rep.offered_tokens_s * 1.02, name
        assert rep.mean_batch <= 64.0 + 1e-9, name
    # Proteus must sustain the highest achieved load of the four.
    best = max(reps, key=lambda k: reps[k].achieved_tokens_s)
    assert best == "proteus", best


def test_scheduler_overhead_is_microsecond_scale():
    """The runtime scheduler must cost tens of FLOPs per operator group and a
    few thousand integer increments per MoE layer, i.e. microseconds against
    millisecond-scale decode iterations (Sec. V-B)."""
    for model, moe in [("llama3-70b", False), ("mixtral-8x7b", True)]:
        c = run("proteus", model, 32).counters
        assert c["sched_flops"] < 5000
        assert (c["sched_int_ops"] > 0) == moe
        assert c["sched_overhead_us"] < 50.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURES'}")
    sys.exit(1 if fails else 0)
