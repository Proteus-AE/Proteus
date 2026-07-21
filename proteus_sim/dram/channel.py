"""Event-driven command-level model of one LPDDR5X PIM channel.

Organization (Fig. microarch): 4 dies x 4 bank groups x 4 banks, one 16-lane
PE per bank, a time-multiplexed 32 B data bus per bank group, a shared
command/address bus per channel.

The engine issues a command stream under an earliest-issue policy subject to:
  * per-bank row state and tRCD/tRP/tRAS/tRC/tCCD_L,
  * per-BG data-bus occupancy (one 32 B burst per ``burst_ns``),
  * per-die tRRD and the rolling four-ACT tFAW window,
  * per-die all-bank refresh every tREFI (blocking tRFCab),
  * PE operand-FIFO back-pressure (Sec. IV-C),
  * command/address bus occupancy (all-bank commands issue once).

Connectivity modes:
  * ``direct``     one-to-one bank-to-PE: every burst feeds its local PE.
  * ``broadcast``  one-to-many within a BG: every burst feeds all four PEs
                   (4-way inter-PE operand reuse); mode switches drain the
                   per-bank FIFOs and update the mode register.

Concurrent xPU traffic: `attach_xpu_stream()` registers a host read stream
that opportunistically claims free BG-bus slots, modeling the unified-memory
co-execution of Sec. IV-B/IV-C. Its achieved bandwidth quantifies how much
memory-service headroom each PIM mode leaves to the xPU.
"""
from collections import deque
from dataclasses import dataclass, field

from .bank import Bank, PE
from .commands import (ACT_AB, RDMAC_AB, WR_AB, PRE_AB, ACT, RD, PRE, MODE,
                       BARRIER)
from .timing import TimingParams


@dataclass
class ChannelStats:
    time_ns: float = 0.0
    n_cmds: int = 0
    n_act: int = 0
    n_rd_burst: int = 0
    n_wr_burst: int = 0
    n_mac: int = 0
    n_mode_switch: int = 0
    n_refresh: int = 0
    row_hits: int = 0
    fifo_stall_ns: float = 0.0
    xpu_bursts: int = 0
    bytes_read: int = 0

    def sustained_bw(self):
        return self.bytes_read / (self.time_ns * 1e-9) if self.time_ns else 0.0

    def xpu_bw(self, burst_bytes):
        return self.xpu_bursts * burst_bytes / (self.time_ns * 1e-9) \
            if self.time_ns else 0.0


class PimChannel:
    def __init__(self, mem_cfg, mode="direct"):
        self.p = TimingParams.from_config(mem_cfg)
        self.dies = mem_cfg["dies_per_channel"]
        self.bgs_per_die = mem_cfg["bankgroups_per_die"]
        self.banks_per_bg = mem_cfg["banks_per_bankgroup"]
        self.n_bg = self.dies * self.bgs_per_die
        self.mode = mode
        self.banks = []
        for d in range(self.dies):
            for g in range(self.bgs_per_die):
                bg = d * self.bgs_per_die + g
                for b in range(self.banks_per_bg):
                    self.banks.append(Bank(die=d, bg=bg, idx=b))
        self.pes = [PE(fifo_depth=mem_cfg["pe_operand_fifo"])
                    for _ in self.banks]
        # resources
        self.bg_bus_free = [0.0] * self.n_bg
        self.ca_free = 0.0
        self.die_acts = [deque() for _ in range(self.dies)]   # tFAW windows
        self.die_last_act = [-1e18] * self.dies               # tRRD_S
        # All-bank refresh is aligned across the channel's dies: lockstep
        # all-bank execution would otherwise stall once per die per tREFI.
        self.next_refresh = [self.p.tREFI] * self.dies
        self.stats = ChannelStats()
        self._xpu = None

    # -------------------------------------------------------------- #
    def attach_xpu_stream(self, min_gap_bursts=1):
        """Enable an opportunistic host read stream (unified memory path)."""
        self._xpu = dict(min_gap=min_gap_bursts * self.p.burst_ns, next_ok=0.0)

    # -------------------------------------------------------------- #
    def _refresh_block(self, die, t):
        """Advance t past any all-bank refresh window of `die`; account it."""
        while t >= self.next_refresh[die]:
            start = self.next_refresh[die]
            if t < start + self.p.tRFCab:
                t = start + self.p.tRFCab
            self.next_refresh[die] += self.p.tREFI
            self.stats.n_refresh += 1
        return t

    def _issue_ca(self, t):
        t = max(t, self.ca_free)
        self.ca_free = t + self.p.ca_cmd_ns
        self.stats.n_cmds += 1
        return t

    def _act_constraints(self, bank, t):
        p = self.p
        t = max(t, bank.next_act)
        t = max(t, self.die_last_act[bank.die] + p.tRRD_S)
        w = self.die_acts[bank.die]
        while w and w[0] <= t - p.tFAW:
            w.popleft()
        if len(w) >= 4:
            t = max(t, w[0] + p.tFAW)
            while w and w[0] <= t - p.tFAW:
                w.popleft()
        return self._refresh_block(bank.die, t)

    def _do_act(self, bank, row, t):
        p = self.p
        t = self._act_constraints(bank, t)
        bank.open_row = row
        bank.act_time = t
        bank.next_act = t + p.tRC
        bank.next_rd = t + p.tRCD
        bank.next_pre = t + p.tRAS
        self.die_last_act[bank.die] = t
        self.die_acts[bank.die].append(t)
        bank.n_act += 1
        self.stats.n_act += 1
        return t

    def _do_rd_burst(self, bank, pe_targets, t):
        """One 32 B column burst from `bank`, delivered to `pe_targets`."""
        p = self.p
        t = max(t, bank.next_rd)
        t = self._refresh_block(bank.die, t)
        prev_free = self.bg_bus_free[bank.bg]
        t = max(t, prev_free)                          # BG bus slot
        stall = 0.0
        for pe in pe_targets:                          # FIFO back-pressure
            stall = max(stall, pe.push(t))
        t += stall
        self.stats.fifo_stall_ns += stall
        self._maybe_xpu(prev_free, t)                  # host steals the gap
        self.bg_bus_free[bank.bg] = t + p.burst_ns
        bank.next_rd = max(bank.next_rd, t + p.tCCD_L)
        bank.next_pre = max(bank.next_pre, t + p.tRTP)
        bank.n_rd += 1
        self.stats.n_rd_burst += 1
        self.stats.n_mac += len(pe_targets)
        self.stats.bytes_read += p.burst_bytes
        return t + p.burst_ns

    def _maybe_xpu(self, gap_start, gap_end):
        """Let the host stream claim BG-bus slots inside a PIM scheduling gap
        (unified-memory co-execution, Sec. IV-B)."""
        if self._xpu is None:
            return
        gap_start = max(self._xpu["next_ok"], gap_start)
        while gap_start + self._xpu["min_gap"] <= gap_end:
            self.stats.xpu_bursts += 1
            gap_start += self.p.burst_ns
        self._xpu["next_ok"] = gap_start

    def _do_wr_burst(self, bank, t):
        """One 32 B column write (decode-appended KV entry, updated in place;
        Sec. IV-A 'Execution Flow'). Writes share the BG bus and extend the
        precharge constraint by tWR."""
        p = self.p
        t = max(t, bank.next_rd)
        t = self._refresh_block(bank.die, t)
        prev_free = self.bg_bus_free[bank.bg]
        t = max(t, prev_free)
        self._maybe_xpu(prev_free, t)
        self.bg_bus_free[bank.bg] = t + p.burst_ns
        bank.next_rd = max(bank.next_rd, t + p.tCCD_L)
        bank.next_pre = max(bank.next_pre, t + p.tWR)
        self.stats.n_wr_burst += 1
        return t + p.burst_ns

    # -------------------------------------------------------------- #
    def bg_peers(self, bank_id):
        bank = self.banks[bank_id]
        base = bank.bg * self.banks_per_bg
        return list(range(base, base + self.banks_per_bg))

    def execute(self, commands):
        """Run a command stream; returns ChannelStats."""
        t_all = 0.0
        for cmd in commands:
            if cmd.kind == BARRIER:
                continue
            if cmd.kind == MODE:
                # takes effect when in-flight FIFOs drain (Sec. IV-C)
                drain = max(pe.free_at for pe in self.pes)
                t = self._issue_ca(max(t_all, drain))
                self.mode = cmd.arg
                self.stats.n_mode_switch += 1
                t_all = max(t_all, t + self.p.ca_cmd_ns)
            elif cmd.kind == ACT_AB:
                t = self._issue_ca(t_all)
                done = t
                for b, bank in enumerate(self.banks):
                    done = max(done, self._do_act(bank, cmd.row, t))
                t_all = done
            elif cmd.kind == RDMAC_AB:
                t = self._issue_ca(t_all)
                done = t
                for b, bank in enumerate(self.banks):
                    if bank.can_read_row(cmd.row):
                        bank.row_hits += 1
                        self.stats.row_hits += 1
                    if self.mode == "broadcast":
                        targets = [self.pes[i] for i in self.bg_peers(b)]
                    else:
                        targets = [self.pes[b]]
                    done = max(done, self._do_rd_burst(bank, targets, t))
                t_all = done
            elif cmd.kind == WR_AB:
                t = self._issue_ca(t_all)
                done = t
                for bank in self.banks:
                    if bank.open_row != cmd.row:
                        done = max(done, self._do_act(bank, cmd.row, t))
                    done = max(done, self._do_wr_burst(bank, max(t, done)))
                t_all = done
            elif cmd.kind == PRE_AB:
                t = self._issue_ca(t_all)
                for bank in self.banks:
                    bank.open_row = -1
                    bank.next_act = max(bank.next_act,
                                        max(t, bank.next_pre) + self.p.tRP)
                t_all = max(t_all, t + self.p.ca_cmd_ns)
            elif cmd.kind in (ACT, RD, PRE):
                bank = self.banks[cmd.bank]
                t = self._issue_ca(t_all)
                if cmd.kind == ACT:
                    t_all = max(t_all, self._do_act(bank, cmd.row, t))
                elif cmd.kind == RD:
                    t_all = max(t_all,
                                self._do_rd_burst(bank, [self.pes[cmd.bank]], t))
                else:
                    bank.open_row = -1
                    bank.next_act = max(bank.next_act,
                                        max(t, bank.next_pre) + self.p.tRP)
            else:
                raise ValueError(f"unknown command {cmd.kind}")
        self.stats.time_ns = t_all
        return self.stats
