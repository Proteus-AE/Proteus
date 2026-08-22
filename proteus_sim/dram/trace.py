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
``PimChannel.execute(read_trace(path))``. A line the reader cannot parse
raises ``TraceError`` naming the file, the line number and what was expected.
"""
from .commands import (Command, ACT_AB, RDMAC_AB, WR_AB, PRE_AB, ACT, RD,
                       PRE, MODE, BARRIER)

#: operand count of every command kind the text format accepts
OPERANDS = {MODE: 1, ACT_AB: 1, RDMAC_AB: 2, WR_AB: 2, PRE_AB: 0,
            ACT: 3, RD: 3, PRE: 3, BARRIER: 0}


class TraceError(ValueError):
    """A trace line is malformed; the message names the file and the line."""


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


def _parse_line(tok, where):
    """One trace line, already split into tokens, as a Command."""
    k = tok[0]
    if k not in OPERANDS:
        raise TraceError(f"{where}: unknown command {k!r} "
                         f"(expected one of {', '.join(sorted(OPERANDS))})")
    args = tok[1:]
    if len(args) != OPERANDS[k]:
        raise TraceError(f"{where}: {k} takes {OPERANDS[k]} operand(s), "
                         f"got {len(args)}")
    if k == MODE:
        try:
            return Command(MODE, arg=args[0])
        except ValueError as e:
            raise TraceError(f"{where}: {e}") from None
    try:
        nums = [int(a) for a in args]
    except ValueError:
        raise TraceError(f"{where}: {k} operands must be integers, got "
                         f"{' '.join(args)!r}") from None
    if k in (ACT, RD, PRE):
        return Command(k, row=nums[0], col=nums[1], bank=nums[2])
    if k in (RDMAC_AB, WR_AB):
        return Command(k, row=nums[0], col=nums[1])
    if k == ACT_AB:
        return Command(ACT_AB, row=nums[0])
    return Command(k)


def read_trace(path):
    out = []
    with open(path) as f:
        for lineno, ln in enumerate(f, start=1):
            tok = ln.split()
            if not tok or tok[0].startswith("#"):
                continue
            out.append(_parse_line(tok, f"{path}:{lineno}"))
    return out
