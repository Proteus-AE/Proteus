"""DRAM / PIM command types for the command-level backend.

All-bank PIM commands (``ACT_AB``, ``RDMAC_AB``, ``PRE_AB``) are issued once
on the command/address bus and executed by every bank, subject to each bank's
local timing state -- the execution model of Sec. IV-C. ``MODE`` performs the
per-channel connectivity reconfiguration (direct <-> broadcasting) and takes
effect once the per-bank FIFOs drain.
"""
from dataclasses import dataclass

# command kinds
ACT_AB = "ACT_AB"        # all-bank row activate (row id attached)
RDMAC_AB = "RDMAC_AB"    # all-bank 32B column read feeding the PE MAC path
PRE_AB = "PRE_AB"        # all-bank precharge
WR_AB = "WR_AB"          # all-bank 32B column write (in-place KV append)
ACT = "ACT"              # single-bank variants (host xPU traffic)
RD = "RD"
PRE = "PRE"
MODE = "MODE"            # connectivity mode register update
BARRIER = "BARRIER"      # trace-level synchronization marker


@dataclass
class Command:
    kind: str
    row: int = 0
    col: int = 0
    bank: int = -1        # target bank for single-bank commands (-1 = all)
    arg: str = ""         # MODE target: 'direct' | 'broadcast'

    def __repr__(self):
        t = f"{self.kind} r{self.row} c{self.col}"
        if self.bank >= 0:
            t += f" b{self.bank}"
        if self.arg:
            t += f" {self.arg}"
        return t
