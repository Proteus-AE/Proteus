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


def _peak():
    bgs = MEM["dies_per_channel"] * MEM["bankgroups_per_die"]
    return bgs * 32 / (MEM["tCCD_L_ns"] / 2.0) * 1e9


def test_streaming_matches_analytical_layer():
    """Sustained all-bank streaming efficiency measured at command level must
    agree with the 0.80 constant used by the analytical layer (within 5%)."""
    st = PimChannel(MEM, "direct").execute(gemv_trace(ROWS, MEM))
    eff = st.sustained_bw() / _peak()
    ana = derive(MEM).internal_eff
    assert abs(eff - ana) / ana < 0.05, f"measured {eff:.3f} vs {ana:.2f}"
    assert st.row_hits == st.n_rd_burst          # streaming = all row hits


def test_broadcast_operand_reuse():
    """Broadcasting must deliver ~min(n,4)-fold reuse (Sec. IV-C)."""
    for n, expect in [(2, 2.0), (4, 4.0), (8, 4.0), (16, 4.0)]:
        td = PimChannel(MEM, "direct").execute(
            skinny_gemm_trace(ROWS, n, MEM, mode="direct")).time_ns
        tb = PimChannel(MEM, "broadcast").execute(
            skinny_gemm_trace(ROWS, n, MEM, mode="broadcast")).time_ns
        assert abs(td / tb - expect) / expect < 0.10, (n, td / tb)


def test_coexecution_headroom():
    """Iso-work: broadcasting must free ~3/4 of the channel time that direct
    mode occupies, exposing memory-service slots to the xPU (Sec. IV-B/C)."""
    td = PimChannel(MEM, "direct").execute(
        skinny_gemm_trace(ROWS, 8, MEM, mode="direct")).time_ns
    tb = PimChannel(MEM, "broadcast").execute(
        skinny_gemm_trace(ROWS, 8, MEM, mode="broadcast")).time_ns
    assert 0.70 < 1 - tb / td < 0.80


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
