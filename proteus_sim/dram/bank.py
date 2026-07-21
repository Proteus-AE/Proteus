"""Per-bank DRAM timing state and the bank-local PE."""
from dataclasses import dataclass, field


@dataclass
class Bank:
    die: int
    bg: int                  # global bank-group id
    idx: int                 # index within the bank group
    open_row: int = -1
    # earliest-next-allowed times (ns) per command class
    next_act: float = 0.0
    next_rd: float = 0.0
    next_pre: float = 0.0
    act_time: float = -1e18  # time of the last ACT (for tRAS/tRCD)
    # statistics
    n_act: int = 0
    n_rd: int = 0
    row_hits: int = 0

    def can_read_row(self, row):
        return self.open_row == row


@dataclass
class PE:
    """16-lane FP16 MAC pipeline with a 4-entry operand FIFO (Fig. microarch(c)).

    In direct mode the PE consumes only its local bank's bursts; in
    broadcasting mode every burst on the BG bus is pushed to all four PEs of
    the group. The FIFO decouples the DRAM burst cadence from MAC issue: a
    burst is dropped into the FIFO at data-return time and drained at
    ``mac_ns`` per entry; a full FIFO back-pressures the bank readout.
    """
    mac_ns: float = 1.0      # 32 B = 16 FP16 lanes -> one MAC issue @ 1 GHz
    fifo_depth: int = 4
    free_at: float = 0.0     # when the pipeline finishes the current drain
    occupancy: int = 0
    last_push: float = 0.0
    n_mac: int = 0

    def push(self, t):
        """Push one operand burst at time t; returns the stall (ns) imposed on
        the producer if the FIFO is full."""
        # drain entries that completed since the last push
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
        return stall
