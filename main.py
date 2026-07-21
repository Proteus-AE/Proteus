#!/usr/bin/env python3
"""ProteusSim command-line driver.

Examples:
  python main.py --system proteus --model mixtral-8x7b --batch 32
  python main.py --system dgx-a100 --model llama3-70b --batch 64 --verbose
  python main.py --system proteus --model mixtral-8x7b --batch 32 --variant rd
  python main.py --system proteus --model llama3-70b --batch 32 --devices 16
  python main.py --system proteus --model mixtral-8x7b --batch 32 --routing sampled --iters 200
"""
import argparse
import sys

from proteus_sim import load_model, build_system
from proteus_sim.workload import build_workload
from proteus_sim.system import VARIANTS, ProteusSystem


def main(argv=None):
    ap = argparse.ArgumentParser(description="ProteusSim")
    ap.add_argument("--system", default="proteus",
                    help="system config name (configs/systems/*.yaml)")
    ap.add_argument("--model", default="llama3-70b",
                    help="model config name (configs/models/*.yaml)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx-in", type=int, default=2048)
    ap.add_argument("--ctx-out", type=int, default=6144)
    ap.add_argument("--ctx", type=int, default=None,
                    help="sustained context override (sets peak=avg=ctx)")
    ap.add_argument("--devices", type=int, default=None)
    ap.add_argument("--dp", type=int, default=1, help="data-parallel replicas")
    ap.add_argument("--variant", default="full", choices=sorted(VARIANTS),
                    help="Proteus incremental variant (Sec. V-C)")
    ap.add_argument("--routing", default="expected", choices=["expected", "sampled"],
                    help="MoE routing: expectation under uniform gating, or "
                         "sampled per-iteration routing averaged over --iters")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--timeline", action="store_true",
                    help="render the per-stage pipeline timeline (Proteus)")
    ap.add_argument("--engine", default="aggregate",
                    choices=["aggregate", "detailed"],
                    help="'detailed' additionally builds the per-layer "
                         "substrate timeline of one stage (Proteus)")
    ap.add_argument("--dump-timeline", metavar="PATH",
                    help="write the per-layer timeline as JSON (implies "
                         "--engine detailed)")
    ap.add_argument("--dump-graph", metavar="PATH",
                    help="lower the model, run the fusion/annotation passes, "
                         "and write the operator graph as JSON")
    ap.add_argument("--graph-layers", type=int, default=1,
                    help="layers to include in --dump-graph (default 1)")
    ap.add_argument("--replay-trace", metavar="FILE",
                    help="replay a PIM command trace on the command-level "
                         "backend and report timing/energy")
    ap.add_argument("--pim-mode", default=None,
                    choices=["direct", "broadcast"],
                    help="initial connectivity mode for --replay-trace")
    args = ap.parse_args(argv)

    # argument validation with friendly diagnostics
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.ctx is not None and args.ctx < 1:
        ap.error("--ctx must be >= 1")
    if args.devices is not None and args.devices < 1:
        ap.error("--devices must be >= 1")
    if args.dp < 1:
        ap.error("--dp must be >= 1")

    if args.replay_trace:
        import os
        if not os.path.exists(args.replay_trace):
            print(f"error: trace file not found: {args.replay_trace}")
            return 2
        return replay_trace(args)

    from proteus_sim.config import list_models, list_systems
    try:
        model = load_model(args.model)
    except FileNotFoundError:
        print(f"error: unknown model '{args.model}' "
              f"(available: {', '.join(list_models())})")
        return 2
    if args.dump_graph:
        from proteus_sim.compiler import lower_model, run_default_pipeline
        g = run_default_pipeline(lower_model(model, n_layers=args.graph_layers))
        g.to_json(args.dump_graph)
        st = g.stats()
        print(f"operator graph -> {args.dump_graph}")
        print(f"  {st['nodes']} nodes after fusion "
              f"({st['fused_away']} element-wise ops fused into SFU chains)")
        print(f"  by kind: {st['by_kind']}")
        return 0

    try:
        sys_ = build_system(args.system, features=VARIANTS[args.variant])
    except FileNotFoundError:
        print(f"error: unknown system '{args.system}' "
              f"(available: {', '.join(list_systems())})")
        return 2
    if args.dp > 1 and not isinstance(sys_, ProteusSystem):
        print("note: [PP,DP] hybrid parallelism is modeled for Proteus only; "
              "--dp is ignored for baseline systems")

    def one(seed):
        w = build_workload(model, args.batch, args.ctx_in, args.ctx_out,
                           routing=args.routing, seed=seed, ctx_override=args.ctx)
        return w, sys_.simulate(w, devices=args.devices, dp=args.dp)

    if args.routing == "sampled" and model["moe"]["enabled"]:
        results = [one(args.seed + i) for i in range(args.iters)]
        alive = [r for _, r in results if r.alive]
        if not alive:
            print(f"{args.system}/{args.model} b={args.batch}: OOM"); return 1
        thr = sum(r.throughput for r in alive) / len(alive)
        tpj = sum(r.tokens_per_joule for r in alive) / len(alive)
        w, r = results[0]
        print(f"[sampled routing, {args.iters} iterations]")
    else:
        w, r = one(args.seed)
        if not r.alive:
            print(f"{args.system}/{args.model} b={args.batch}: {r.notes}"); return 1
        thr, tpj = r.throughput, r.tokens_per_joule

    print(f"system            : {r.system} ({args.variant})")
    print(f"model             : {model['name']}  batch={args.batch}  "
          f"ctx={w.ctx_avg} (peak {w.ctx_peak})")
    print(f"throughput        : {thr:,.0f} tokens/s")
    print(f"iteration latency : {r.t_iter_ms:.3f} ms")
    print(f"energy efficiency : {tpj:.2f} tokens/J   (avg power {r.power_w:,.0f} W)")
    if r.notes:
        print(f"notes             : {r.notes}")

    if args.verbose:
        if isinstance(sys_, ProteusSystem):
            print("\nderived machine parameters:")
            print(sys_.dmem.describe())
            n, modes = sys_.sched.summary(r.placements)
            print(f"\noperator placement ({len(r.placements)} operators): "
                  f"xPU {n['xpu']}, PIM {n['pim']} "
                  f"(direct {modes['direct']}, broadcast {modes['broadcast']}), "
                  f"SFU {n['sfu']}")
            print(f"crossover AI      : {sys_.sched.crossover_ai} "
                  f"(AI_PIM={sys_.sched.ai_pim:.1f}, AI_xPU={sys_.sched.ai_xpu:.1f})")
        if r.counters:
            print("\ncounters:")
            for k, v in r.counters.items():
                if isinstance(v, float) and abs(v) > 1e6:
                    print(f"  {k:<28}: {v/1e9:.2f} GB")
                else:
                    print(f"  {k:<28}: {v}")

    if (args.engine == "detailed" or args.dump_timeline) and \
            isinstance(sys_, ProteusSystem) and r.alive:
        from proteus_sim.engine import DetailedEngine
        tl = DetailedEngine(sys_).build(w, r)
        print("\nper-layer substrate occupancy (one pipeline stage):")
        print(tl.render())
        if args.dump_timeline:
            tl.to_json(args.dump_timeline)
            print(f"timeline -> {args.dump_timeline}")

    if args.timeline and isinstance(sys_, ProteusSystem) and r.alive:
        from proteus_sim.fabric import CxlFabric
        c = r.counters
        t_stage_ns = r.t_iter_ms * 1e6 * c["inflight"] / c["devices"]
        fab = CxlFabric(sys_.cfg["interconnect"])
        ev = fab.iteration_timeline(c["devices"], t_stage_ns,
                                    c["activation_bytes_per_stage"])
        print("\npipeline timeline (one decode iteration):")
        print(CxlFabric.render(ev))
    return 0


def replay_trace(args):
    from proteus_sim.config import load_memory
    from proteus_sim.dram import PimChannel, CommandEnergy
    from proteus_sim.dram.trace import read_trace
    mem = load_memory("lpddr5x-8533")
    ch = PimChannel(mem, mode=args.pim_mode or "direct")
    cmds = read_trace(args.replay_trace)
    st = ch.execute(cmds)
    en = CommandEnergy(mem).account(st)
    peak = ch.n_bg * 32 / (mem["tCCD_L_ns"] / 2.0) * 1e9
    print(f"trace              : {args.replay_trace} ({len(cmds)} commands)")
    print(f"execution time     : {st.time_ns/1e3:,.1f} us")
    print(f"sustained internal : {st.sustained_bw()/1e9:.1f} GB/s/channel "
          f"(eff {st.sustained_bw()/peak:.3f})")
    print(f"row-buffer hit rate: {st.row_hits/max(st.n_rd_burst,1):.3f} | "
          f"ACT {st.n_act} | RD bursts {st.n_rd_burst} | MAC {st.n_mac}")
    print(f"refresh events     : {st.n_refresh} | mode switches "
          f"{st.n_mode_switch} | FIFO stall {st.fifo_stall_ns/1e3:.1f} us")
    print(f"energy             : {en.total_nj/1e3:.1f} uJ "
          f"({en.pj_per_byte(st.bytes_read)/8:.2f} pJ/bit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
