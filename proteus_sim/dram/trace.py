"""PIM command-trace serialization.

Text format, one command per line (AiM-style command streams):

    MODE broadcast
    ACT_AB   <row>
    RDMAC_AB <row> <col>
    PRE_AB
    RD       <row> <col> <bank>      # host (xPU) single-bank traffic
    BARRIER

Traces produced by ``trace_gen`` can be replayed with
``python main.py --replay-trace <file>`` or consumed programmatically by
``PimChannel.execute(read_trace(path))``.
"""
from .commands import (Command, ACT_AB, RDMAC_AB, WR_AB, PRE_AB, ACT, RD,
                       PRE, MODE, BARRIER)


def write_trace(path, commands):
    with open(path, "w") as f:
        for c in commands:
            if c.kind == MODE:
                f.write(f"MODE {c.arg}\n")
            elif c.kind == ACT_AB:
                f.write(f"ACT_AB {c.row}\n")
            elif c.kind == RDMAC_AB:
                f.write(f"RDMAC_AB {c.row} {c.col}\n")
            elif c.kind == WR_AB:
                f.write(f"WR_AB {c.row} {c.col}\n")
            elif c.kind == PRE_AB:
                f.write("PRE_AB\n")
            elif c.kind in (ACT, RD, PRE):
                f.write(f"{c.kind} {c.row} {c.col} {c.bank}\n")
            else:
                f.write("BARRIER\n")


def read_trace(path):
    out = []
    with open(path) as f:
        for ln in f:
            tok = ln.split()
            if not tok or tok[0].startswith("#"):
                continue
            k = tok[0]
            if k == "MODE":
                out.append(Command(MODE, arg=tok[1]))
            elif k == "ACT_AB":
                out.append(Command(ACT_AB, row=int(tok[1])))
            elif k == "RDMAC_AB":
                out.append(Command(RDMAC_AB, row=int(tok[1]), col=int(tok[2])))
            elif k == "WR_AB":
                out.append(Command(WR_AB, row=int(tok[1]), col=int(tok[2])))
            elif k == "PRE_AB":
                out.append(Command(PRE_AB))
            elif k in (ACT, RD, PRE):
                out.append(Command(k, row=int(tok[1]), col=int(tok[2]),
                                   bank=int(tok[3])))
            elif k == "BARRIER":
                out.append(Command(BARRIER))
            else:
                raise ValueError(f"bad trace line: {ln!r}")
    return out
