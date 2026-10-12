"""Command-level LPDDR5X near-bank-PIM backend.

This package models one PIM channel at DRAM-command granularity: per-bank row
buffers and timing state, bank-group data buses, per-die ACT windows (tFAW /
tRRD) and all-bank refresh, the per-bank PE pipelines with their operand
FIFOs, and the direct / broadcasting bank-to-PE connectivity of Sec. IV-C.

It executes command traces produced by ``trace_gen`` (or generated in memory)
under an earliest-issue event-driven scheduler, and reports sustained
bandwidth, row-buffer locality, PE utilization, and DRAMPower-style
command-based energy.

The analytical layer (``proteus_sim.memory``) uses closed-form sustained
rates; ``experiments/run_microbench.py`` cross-validates those rates against
this backend (see docs/dram-model.md).
"""
from .timing import TimingParams
from .channel import PimChannel
from .power import CommandEnergy

__all__ = ["TimingParams", "PimChannel", "CommandEnergy"]
