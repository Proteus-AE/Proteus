"""Per-bank DRAM timing state and the bank-local PE (Fig. 8(c))."""
from dataclasses import dataclass


@dataclass
class Bank:
    die: int
    bg: int                  # global bank-group id within the channel
    idx: int                 # index within the bank group
    open_row: int = -1
    # earliest-next-allowed times (ns) per command class
    next_act: float = 0.0
    next_col: float = 0.0    # bank-local column cycle (near-bank read path)
    next_pre: float = 0.0
    act_time: float = -1e18  # time of the last ACT (for tRAS/tRCD)
    # statistics
    n_act: int = 0
    n_rd: int = 0
    n_wr: int = 0
    row_hits: int = 0

    def can_read_row(self, row):
        return self.open_row == row


def activity_spread(banks):
    """Range of per-bank activity over an executed command stream.

    An all-bank command is issued once and retired by every bank against its
    own timing state, so a layout that stripes its operand evenly leaves every
    bank with the same activation, read and write counts; a layout that does
    not shows up here as a spread, and the widest bank sets the stream time.
    """
    if not banks:
        return "no banks"
    act = [b.n_act for b in banks]
    rd = [b.n_rd for b in banks]
    wr = [b.n_wr for b in banks]
    hits = sum(b.row_hits for b in banks)
    return (f"ACT {min(act)}-{max(act)} | RD {min(rd)}-{max(rd)} | "
            f"WR {min(wr)}-{max(wr)} | row hits {hits/max(sum(rd), 1):.3f}")


@dataclass
class PE:
    """16-lane FP16 MAC pipeline with a 4-entry (128 B) operand FIFO.

    One issue consumes one 32 B burst of the shared matrix operand against
    one resident input vector. In *direct* mode the PE consumes only its own
    bank's bursts and therefore idles between column cycles; in *broadcast*
    mode every burst produced inside the bank group is pushed into the FIFO
    and the 4:1 selector feeds them into the MAC array back to back, so the
    pipeline saturates. The FIFO decouples the DRAM burst cadence from MAC
    issue: a full FIFO back-pressures the bank readout.
    """
    mac_ns: float = 1.0      # 32 B = 16 FP16 lanes -> one MAC issue @ 1 GHz
    fifo_depth: int = 4
    free_at: float = 0.0     # when the pipeline finishes the current drain
    occupancy: int = 0
    last_push: float = 0.0
    n_mac: int = 0
    busy_ns: float = 0.0

    def push(self, t):
        """Push one operand burst at time t; returns the stall (ns) imposed on
        the producer if the FIFO is full."""
        drained = int(max(0.0, (t - self.last_push)) / self.mac_ns)
        self.occupancy = max(0, self.occupancy - drained)
        stall = 0.0
        if self.occupancy >= self.fifo_depth:
            stall = self.free_at - t if self.free_at > t else self.mac_ns
            self.occupancy -= 1
        self.occupancy += 1
        self.last_push = t + stall
        self.free_at = max(self.free_at, t + stall) + self.mac_ns
        self.n_mac += 1
        self.busy_ns += self.mac_ns
        return stall
