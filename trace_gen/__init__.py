"""PIM command-trace generation for the command-level backend.

Lowers the decode-phase operator kernels of Sec. IV (reuse-free GEMV,
reuse-bearing skinny-GEMM, GQA/MLA attention over the column/row-striped KV
layout of Fig. 7) into all-bank command streams executable by
``proteus_sim.dram.PimChannel``.
"""
from .kernels import (gemv_trace, skinny_gemm_trace, attention_trace,
                      layout_rows_per_bank)

__all__ = ["gemv_trace", "skinny_gemm_trace", "attention_trace",
           "layout_rows_per_bank"]
