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


def test_proteus_physical_bounds():
    """Sustained rates must not exceed the derived machine parameters."""
    sys_ = build_system("proteus")
    for model in MODELS:
        w = build_workload(load_model(model), 32, *CTX)
        r = sys_.simulate(w)
        t_iter = r.t_iter_ms * 1e-3
        pim_bytes_dev = (r.counters["pim_weight_bytes"]
                         + r.counters["pim_kv_bytes"]) / 8
        assert pim_bytes_dev / t_iter <= sys_.dmem.internal_bw * 1.001, model
        xpu_bytes_dev = r.counters["xpu_ext_bytes"] / 8
        assert xpu_bytes_dev / t_iter <= sys_.cfg["xpu"]["mem_bw_per_device"] \
            * 1.001, model


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
    model = load_model("mixtral-8x7b")
    w = build_workload(model, 128, *CTX)     # ~32 tokens/expert -> AI_w ~ 32
    d = model["d_model"]
    n = w.tokens_per_expert
    ai = n * d / (2 * n + d)
    sys_ = build_system("proteus")
    assert abs(ai / sys_.sched.crossover_ai - 1) < 0.05
    t_x = w.weight_bytes / sys_.cfg["xpu"]["mem_bw_per_device"] / 8
    t_p = max(w.weight_flops / sys_.dmem.pe_flops,
              w.weight_bytes * math.ceil(n / 4) / sys_.dmem.internal_bw) / 8
    assert abs(t_x / t_p - 1) < 0.15


def test_variant_chain_monotone():
    """Each cumulative mechanism must not reduce throughput (Sec. V-C)."""
    for model in ["mixtral-8x7b", "llama3-70b"]:
        prev = 0.0
        for v in ["base", "as", "rd", "of", "ec"]:
            thr = run("proteus", model, 32, variant=v).throughput
            assert thr >= prev * 0.999, f"{model}: variant +{v} regressed"
            prev = thr


def test_sampled_routing_consistency():
    """Sampled multinomial routing must agree with the uniform-routing
    expectation within a few percent when averaged over iterations."""
    thr_exp = run("proteus", "mixtral-8x7b", 32).throughput
    vals = [run("proteus", "mixtral-8x7b", 32, routing="sampled", seed=s).throughput
            for s in range(60)]
    thr_smp = sum(vals) / len(vals)
    assert abs(thr_smp / thr_exp - 1) < 0.03


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
    assert 14.0 < speedup <= 16.0


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
