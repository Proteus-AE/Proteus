"""Command-level backend checks: analytical-layer cross-validation,
broadcast operand reuse, co-execution headroom, energy split, trace I/O,
and the compilation passes.

Runs under pytest or standalone (`python tests/test_dram_backend.py`).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proteus_sim.config import load_memory, load_model            # noqa: E402
from proteus_sim.dram import PimChannel, CommandEnergy            # noqa: E402
from proteus_sim.dram.trace import write_trace, read_trace        # noqa: E402
from proteus_sim.memory import derive                             # noqa: E402
from trace_gen import gemv_trace, skinny_gemm_trace               # noqa: E402
from proteus_sim.compiler import lower_model, run_default_pipeline  # noqa: E402

MEM = load_memory("lpddr5x-8533")
ROWS = 32


def _channel(mode):
    return PimChannel(MEM, mode)


def test_organization_matches_table_iii():
    """The command-level channel must carry the organization of Sec. V-A:
    one 32Gb die per x16 channel, four bank groups of four banks, one PE per
    bank, and the two column cadences of Fig. 9."""
    ch = _channel("direct")
    assert ch.n_banks == 16 and ch.n_bg == 4
    assert len(ch.pes) == ch.n_banks
    assert abs(ch.column_cadence("direct") - MEM["tCCD_L_ns"] / 2.0) < 1e-9
    assert abs(ch.column_cadence("broadcast") - MEM["tCCD_L_ns"]) < 1e-9
    channels = MEM["packages_per_device"] * MEM["channels_per_package"]
    assert abs(ch.peak_bw("direct") * channels / 16.384e12 - 1) < 1e-6
    assert abs(ch.peak_bw("broadcast") * channels / 8.192e12 - 1) < 1e-6


def test_streaming_matches_analytical_layer():
    """Sustained all-bank streaming efficiency measured at command level must
    agree with the closed form the analytical layer uses, in both
    connectivity modes (within 5%)."""
    d = derive(MEM)
    st = _channel("direct").execute(gemv_trace(ROWS, MEM))
    eff = st.sustained_bw() / _channel("direct").peak_bw("direct")
    assert abs(eff - d.internal_eff) / d.internal_eff < 0.05, \
        f"direct: measured {eff:.3f} vs {d.internal_eff:.3f}"
    assert st.row_hits == st.n_rd_burst          # streaming = all row hits

    sb = _channel("broadcast").execute(
        skinny_gemm_trace(ROWS, 4, MEM, mode="broadcast"))
    eff_b = sb.sustained_bw() / _channel("broadcast").peak_bw("broadcast")
    assert abs(eff_b - d.broadcast_eff) / d.broadcast_eff < 0.05, \
        f"broadcast: measured {eff_b:.3f} vs {d.broadcast_eff:.3f}"


def test_broadcast_operand_reuse():
    """Four-way inter-PE reuse and the cadence that comes with it.

    Every PE assembles one burst from every bank of its group into a FIFO
    whose depth matches the BG fan-in, and a 4:1 selector drains them into the
    MAC array one issue at a time; the bank column cadence therefore matches
    the fan-in. A skinny-GEMM of n vectors costs n passes in direct mode
    against ceil(n/fanout) passes at the broadcasting cadence, and the reuse
    width bounds the benefit: two vectors gain nothing (Sec. IV-C)."""
    import math
    d = derive(MEM)
    ratio_eff = d.broadcast_eff / d.internal_eff
    for n in (2, 4, 8, 16):
        td = _channel("direct").execute(
            skinny_gemm_trace(ROWS, n, MEM, mode="direct")).time_ns
        tb = _channel("broadcast").execute(
            skinny_gemm_trace(ROWS, n, MEM, mode="broadcast")).time_ns
        expect = n / (2.0 * math.ceil(n / d.fanout)) * ratio_eff
        assert abs(td / tb - expect) / expect < 0.06, (n, td / tb, expect)


def test_coexecution_headroom():
    """A bank serves either its local PE or the channel's global I/O in a
    column cycle. Direct mode drives every bank at its minimum column cycle
    and returns nothing to the xPU; the broadcasting cadence frees enough
    slots to cover the channel's share of the 1 TB/s external interface --
    the memory-service opportunity of Sec. IV-C. These measurements are the
    co-execution constants of configs/systems/proteus.yaml."""
    channels = MEM["packages_per_device"] * MEM["channels_per_package"]
    ext_per_channel = MEM["external_bw_per_device_tbps"] * 1e12 / channels
    out = {}
    for mode, n in (("direct", 8), ("broadcast", 8)):
        ch = _channel(mode)
        ch.attach_xpu_stream()
        st = ch.execute(skinny_gemm_trace(ROWS, n, MEM, mode=mode))
        out[mode] = (st, st.xpu_bw(32))
    assert out["direct"][1] == 0.0
    assert out["broadcast"][1] > 0.5 * ext_per_channel
    # iso-work, so the freed time is real: broadcasting also finishes sooner
    assert out["broadcast"][0].time_ns < out["direct"][0].time_ns


def test_mac_counts():
    """MAC issue counts must equal bursts x fan-out in broadcasting mode."""
    st_d = PimChannel(MEM, "direct").execute(gemv_trace(ROWS, MEM))
    assert st_d.n_mac == st_d.n_rd_burst
    st_b = PimChannel(MEM, "broadcast").execute(
        skinny_gemm_trace(ROWS, 4, MEM, mode="broadcast"))
    assert st_b.n_mac == st_b.n_rd_burst * MEM["banks_per_bankgroup"]


def test_energy_split():
    """Near-bank termination must land near 2.2 pJ/bit and external near
    4.5 pJ/bit (the constants of the system-level energy model)."""
    en = CommandEnergy(MEM)
    st = PimChannel(MEM, "direct").execute(gemv_trace(ROWS, MEM))
    near = en.account(st, external=False).pj_per_byte(st.bytes_read) / 8
    ext = en.account(st, external=True).pj_per_byte(st.bytes_read) / 8
    assert abs(near - 2.2) / 2.2 < 0.20
    assert abs(ext - 4.5) / 4.5 < 0.10
    assert ext > near


def test_trace_roundtrip():
    cmds = skinny_gemm_trace(4, 8, MEM, mode="broadcast")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.trace")
        write_trace(p, cmds)
        back = read_trace(p)
    assert len(back) == len(cmds)
    assert [c.kind for c in back] == [c.kind for c in cmds]
    t1 = PimChannel(MEM, "direct").execute(cmds).time_ns
    t2 = PimChannel(MEM, "direct").execute(back).time_ns
    assert abs(t1 - t2) < 1e-6


def test_compiler_fusion_and_annotation():
    """Fusion must fold all interior element-wise chains into SFU pipelines
    and annotate runtime-parameterized AI expressions (Sec. IV-E)."""
    g = run_default_pipeline(lower_model(load_model("mixtral-8x7b"),
                                         n_layers=2))
    st = g.stats()
    # only the boundary norm (embedding input) may survive per graph
    assert st["by_kind"].get("elementwise", 0) <= 1
    assert st["fused_away"] >= 20
    expert_ops = [n for n in g.iter_nodes() if "e0.up" in n.name]
    assert expert_ops and "tpe" in expert_ops[0].ai_expr
    att = [n for n in g.iter_nodes() if n.kind == "attention"]
    assert att and att[0].ai_expr == "g"
    dense = run_default_pipeline(lower_model(load_model("llama3-70b"),
                                             n_layers=1))
    assert dense.stats()["by_kind"].get("reduce", 0) == 0


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
