"""XpuCore: tile-level xPU (matrix-engine) simulator.

Fills the role the paper's methodology assigns to the NPU-side simulator:
a systolic-array core model that consumes the operator graphs produced by
the compilation layer (or an ONNX graph when the optional loader is
available), schedules each GEMM as SRAM-resident tiles with double
buffering, and emits its DRAM traffic as host-request streams.

Integration with the PIM side happens at the memory interface, exactly as
described in Sec. V-A: the xPU tile-fetch stream and the all-bank PIM
command stream share the LPDDR channels. `experiments/run_integrated.py`
co-simulates both streams on the command-level backend and closes the loop
against the analytical co-execution split x*.
"""
from .systolic import SystolicConfig, TileSchedule, XpuEngine
from .onnx_loader import load_onnx_graph

__all__ = ["SystolicConfig", "TileSchedule", "XpuEngine", "load_onnx_graph"]
