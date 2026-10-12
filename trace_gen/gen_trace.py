#!/usr/bin/env python3
"""CLI: generate PIM command traces for the command-level backend.

Examples:
  python trace_gen/gen_trace.py --kernel gemv --rows 64 -o results/gemv.trace
  python trace_gen/gen_trace.py --kernel skinny-gemm --rows 64 --vectors 8 \
      --mode broadcast -o results/gemm_b.trace
  python trace_gen/gen_trace.py --kernel attention --ctx 8192 --group 8 \
      --mode broadcast -o results/attn.trace
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from proteus_sim.config import load_memory                     # noqa: E402
from proteus_sim.dram.trace import write_trace                 # noqa: E402
from trace_gen.kernels import (gemv_trace, skinny_gemm_trace,  # noqa: E402
                               attention_trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="lpddr5x-8533")
    ap.add_argument("--kernel", required=True,
                    choices=["gemv", "skinny-gemm", "attention"])
    ap.add_argument("--rows", type=int, default=64,
                    help="rows per bank of the striped operand")
    ap.add_argument("--vectors", type=int, default=8,
                    help="concurrent input vectors (skinny-gemm)")
    ap.add_argument("--ctx", type=int, default=8192, help="context tokens")
    ap.add_argument("--kv-bytes", type=float, default=4096,
                    help="KV bytes/token/layer of the traced head-group slice")
    ap.add_argument("--group", type=int, default=8, help="GQA group size")
    ap.add_argument("--mode", default="broadcast",
                    choices=["direct", "broadcast"])
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    mem = load_memory(args.memory)
    if args.kernel == "gemv":
        cmds = gemv_trace(args.rows, mem)
    elif args.kernel == "skinny-gemm":
        cmds = skinny_gemm_trace(args.rows, args.vectors, mem, mode=args.mode)
    else:
        cmds = attention_trace(args.ctx, args.kv_bytes, args.group, mem,
                               mode=args.mode)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_trace(args.out, cmds)
    print(f"{len(cmds)} commands -> {args.out}")


if __name__ == "__main__":
    main()
