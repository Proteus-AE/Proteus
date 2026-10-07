"""Event-driven command-level model of one LPDDR5X PIM channel.

Organization (Sec. V-A, Fig. 8(a)): one 32Gb die per x16 channel, four bank
groups of four banks, one 16-lane FP16 PE per bank, a BG-local distribution
bus used only by broadcasting, the channel's global I/O used only by host
traffic leaving the die, and a shared command/address bus.

Datapath
--------
The near-bank read path runs from a bank's I/O sense amplifiers straight into
its co-located PE (Fig. 8(c)), bypassing the shared BG bus and the global
I/O. Its cadence is the bank's own column cycle ``tCCD_L/2``; the tCCD_L and
tCCD_S constraints of the standard bound the *shared* bus, which this path
does not use, and the PE cannot accept a burst faster than ``mac_ns``.

  * ``direct``     one-to-one: every burst feeds the bank-local PE, and every
                   fetched matrix element contributes to a single vector. All
                   banks read concurrently, so the channel sustains
                   ``banks * 32 B / (tCCD_L/2)``.
  * ``broadcast``  one-to-many: each burst is shared with all PEs of its bank
                   group, and every PE assembles one burst from every bank of
                   the group into its four-entry operand FIFO, whose depth
                   matches the BG fan-in. A 4:1 selector drains them into the
                   MAC array one issue at a time, so each matrix fetch is
                   reused across four concurrent input vectors and the column
                   cadence matches the fan-in,
                   ``max(tCCD_L, fanout * mac_ns)``.

All-bank commands (``ACT_AB``, ``RDMAC_AB``, ``PRE_AB``, ``WR_AB``) are
issued once on the command/address bus and executed by every bank. Because
the device provisions an all-bank activation as a single operation, they are
not subject to tRRD/tFAW, which bound the controller's ability to stagger
independent single-bank activations; single-bank host commands are.

Concurrent xPU traffic: ``attach_xpu_stream()`` registers a host read stream
that claims the column slots the PIM datapath leaves free. Its achieved
bandwidth quantifies how much memory-service headroom each connectivity mode
returns to the xPU (Sec. IV-B/IV-C), and is the measurement behind the
co-execution constant of the analytical layer.
"""
from collections import deque
from dataclasses import dataclass

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
    n_broadcast: int = 0        # bursts distributed over a BG-local bus
    n_mode_switch: int = 0
    n_refresh: int = 0
    row_hits: int = 0
    fifo_stall_ns: float = 0.0
    xpu_bursts: int = 0
    bytes_read: int = 0

    def sustained_bw(self):
        return self.bytes_read / (self.time_ns * 1e-9) if self.time_ns else 0.0

    def mac_rate(self, lanes=16):
        """Sustained near-bank FLOP/s of the channel."""
        return self.n_mac * lanes * 2.0 / (self.time_ns * 1e-9) \
            if self.time_ns else 0.0

    def xpu_bw(self, burst_bytes):
        return self.xpu_bursts * burst_bytes / (self.time_ns * 1e-9) \
            if self.time_ns else 0.0


class PimChannel:
    def __init__(self, mem_cfg, mode="direct"):
        self.cfg = mem_cfg
        self.p = TimingParams.from_config(mem_cfg)
        self.dies = mem_cfg["dies_per_channel"]
        self.bgs_per_die = mem_cfg["bankgroups_per_die"]
        self.banks_per_bg = mem_cfg["banks_per_bankgroup"]
        self.n_bg = self.dies * self.bgs_per_die
        self.fanout = mem_cfg.get("broadcast_fanout", self.banks_per_bg)
        self.mode = mode
        self.banks = []
        for d in range(self.dies):
            for g in range(self.bgs_per_die):
                bg = d * self.bgs_per_die + g
                for b in range(self.banks_per_bg):
                    self.banks.append(Bank(die=d, bg=bg, idx=b))
        self.pes = [PE(mac_ns=self.p.mac_ns,
                       fifo_depth=mem_cfg["pe_operand_fifo"])
                    for _ in self.banks]
        # resources
        self.bg_bus_free = [0.0] * self.n_bg    # BG-local broadcast distribution
        self.global_io_free = 0.0               # channel global I/O (host path)
        self.ca_free = 0.0
        self.die_acts = [deque() for _ in range(self.dies)]   # tFAW windows
        self.die_last_act = [-1e18] * self.dies               # tRRD_S
        # All-bank refresh is aligned across the channel's dies: lockstep
        # all-bank execution would otherwise stall once per die per tREFI.
        self.next_refresh = [self.p.tREFI] * self.dies
        self.stats = ChannelStats()
        self._xpu = None

    # -------------------------------------------------------------- #
    @property
    def n_banks(self):
        return len(self.banks)

    def column_cadence(self, mode=None):
        """Bank column cycle of a connectivity mode (ns)."""
        mode = mode or self.mode
        if mode == "broadcast":
            return max(self.p.tCCD_L, self.fanout * self.p.mac_ns)
        return max(self.p.tCCD_PIM, self.p.mac_ns)

    def peak_bw(self, mode=None):
        """All-bank peak read bandwidth of this channel in a mode (B/s)."""
        return self.n_banks * self.p.burst_bytes / (self.column_cadence(mode)
                                                    * 1e-9)

    def host_slot_bursts(self, mode=None):
        """Column bursts per cadence this channel can still deliver to the
        external interface while the all-bank PIM stream runs (Sec. IV-B).

        A bank serves one column access per near-bank column cycle; the PIM
        stream claims one of those per cadence, leaving
        ``cadence/tCCD_PIM - 1`` per bank. A host burst, unlike a near-bank
        read, does traverse the bank-group I/O and the channel's global I/O,
        so it is additionally bounded by tCCD_L per bank group and by the DQ
        occupancy of the channel. Direct mode runs at the minimum column
        cycle and therefore leaves nothing; broadcasting runs at the
        bank-group-bus cadence and leaves a full slot per bank."""
        p = self.p
        cadence = self.column_cadence(mode)
        free_per_bank = max(0.0, cadence / p.tCCD_PIM - 1.0)
        return min(self.n_banks * free_per_bank,          # bank column ports
                   self.n_bg * cadence / p.tCCD_L,        # BG I/O gating
                   cadence / p.burst_ns)                  # channel DQ

    def host_slot_bw(self, mode=None):
        """Concurrent external bandwidth of one channel (B/s)."""
        cadence = self.column_cadence(mode)
        return self.host_slot_bursts(mode) * self.p.burst_bytes / (cadence
                                                                   * 1e-9)

    # -------------------------------------------------------------- #
    def attach_xpu_stream(self):
        """Enable an opportunistic host read stream (unified memory path)."""
        self._xpu = {"frac": 0.0}

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

    def _act_constraints(self, bank, t, all_bank):
        p = self.p
        t = max(t, bank.next_act)
        if not (all_bank and p.allbank_faw_exempt):
            t = max(t, self.die_last_act[bank.die] + p.tRRD_S)
            w = self.die_acts[bank.die]
            while w and w[0] <= t - p.tFAW:
                w.popleft()
            if len(w) >= 4:
                t = max(t, w[0] + p.tFAW)
                while w and w[0] <= t - p.tFAW:
                    w.popleft()
        return self._refresh_block(bank.die, t)

    def _do_act(self, bank, row, t, all_bank=False):
        p = self.p
        t = self._act_constraints(bank, t, all_bank)
        rcd = p.allbank_rcd if all_bank else p.tRCD
        ras = p.allbank_ras if all_bank else p.tRAS
        bank.open_row = row
        bank.act_time = t
        bank.next_act = t + p.tRC
        bank.next_col = t + rcd
        bank.next_pre = t + ras
        if not (all_bank and p.allbank_faw_exempt):
            self.die_last_act[bank.die] = t
            self.die_acts[bank.die].append(t)
        bank.n_act += 1
        self.stats.n_act += 1
        return t

    # -------------------------------------------------------------- #
    def _near_bank_read(self, bank, pe_targets, t, cadence):
        """One 32 B column burst read over the bank-local path and delivered
        to `pe_targets` (its own PE, or every PE of its bank group)."""
        p = self.p
        t = max(t, bank.next_col)
        t = self._refresh_block(bank.die, t)
        if len(pe_targets) > 1:
            # broadcasting occupies the BG-local distribution bus
            t = max(t, self.bg_bus_free[bank.bg])
        stall = 0.0
        for pe in pe_targets:
            stall = max(stall, pe.push(t))
        t += stall
        self.stats.fifo_stall_ns += stall
        if len(pe_targets) > 1:
            self.bg_bus_free[bank.bg] = t + p.mac_ns
            self.stats.n_broadcast += 1
        bank.next_col = t + cadence
        bank.next_pre = max(bank.next_pre, t + p.tRTP)
        bank.n_rd += 1
        self.stats.n_rd_burst += 1
        self.stats.n_mac += len(pe_targets)
        self.stats.bytes_read += p.burst_bytes
        return t

    def _host_read(self, bank, t):
        """One 32 B single-bank read leaving the die over the global I/O."""
        p = self.p
        t = max(t, bank.next_col, self.global_io_free)
        t = self._refresh_block(bank.die, t)
        self.global_io_free = t + p.burst_ns
        bank.next_col = max(bank.next_col, t + p.tCCD_L)
        bank.next_pre = max(bank.next_pre, t + p.tRTP)
        bank.n_rd += 1
        self.stats.n_rd_burst += 1
        self.stats.bytes_read += p.burst_bytes
        return t + p.burst_ns

    def _do_wr_burst(self, bank, t, cadence):
        """One 32 B column write (decode-appended KV entry, updated in place;
        Sec. IV-A "Execution Flow"). Writes extend the precharge constraint by
        tWR."""
        p = self.p
        t = max(t, bank.next_col)
        t = self._refresh_block(bank.die, t)
        bank.next_col = t + cadence
        bank.next_pre = max(bank.next_pre, t + p.tWR)
        bank.n_wr += 1
        self.stats.n_wr_burst += 1
        return t

    def _steal_slots(self, t0, t1, cadence):
        """Let the concurrent host stream claim what the PIM datapath leaves
        free between t0 and t1 (Sec. IV-B unified memory path).

        The accounting is `host_slot_bursts`: the column slots each bank has
        left over the cadence, bounded by the bank-group I/O gating and by
        the channel's DQ occupancy. Direct mode drives every bank at its
        minimum column cycle and therefore leaves nothing; broadcasting runs
        at twice that period and frees enough to saturate the external
        interface."""
        if self._xpu is None or t1 <= t0:
            return
        per_cadence = self.host_slot_bursts()
        if per_cadence <= 0:
            return
        credit = (t1 - t0) / cadence * per_cadence + self._xpu.get("frac", 0.0)
        whole = int(credit)
        self._xpu["frac"] = credit - whole
        self.stats.xpu_bursts += max(0, whole)

    # -------------------------------------------------------------- #
    def bg_peers(self, bank_id):
        base = (bank_id // self.banks_per_bg) * self.banks_per_bg
        return list(range(base, base + self.banks_per_bg))

    def execute(self, commands):
        """Run a command stream; returns ChannelStats."""
        t_all = 0.0
        for cmd in commands:
            if cmd.kind == BARRIER:
                continue
            cadence = self.column_cadence()
            if cmd.kind == MODE:
                # takes effect when in-flight FIFOs drain (Sec. IV-C)
                drain = max((pe.free_at for pe in self.pes), default=0.0)
                t = self._issue_ca(max(t_all, drain))
                self.mode = cmd.arg
                self.stats.n_mode_switch += 1
                t_all = max(t_all, t + self.p.ca_cmd_ns)
            elif cmd.kind == ACT_AB:
                t = self._issue_ca(t_all)
                done = t
                for bank in self.banks:
                    done = max(done, self._do_act(bank, cmd.row, t,
                                                  all_bank=True))
                t_all = done
            elif cmd.kind == RDMAC_AB:
                t = self._issue_ca(t_all)
                done = t
                t0 = t_all
                for b, bank in enumerate(self.banks):
                    if bank.can_read_row(cmd.row):
                        bank.row_hits += 1
                        self.stats.row_hits += 1
                    if self.mode == "broadcast":
                        targets = [self.pes[i] for i in self.bg_peers(b)]
                    else:
                        targets = [self.pes[b]]
                    done = max(done, self._near_bank_read(bank, targets, t,
                                                          cadence))
                self._steal_slots(t0, done, cadence)
                t_all = done
            elif cmd.kind == WR_AB:
                t = self._issue_ca(t_all)
                done = t
                for bank in self.banks:
                    if bank.open_row != cmd.row:
                        done = max(done, self._do_act(bank, cmd.row, t,
                                                      all_bank=True))
                    done = max(done, self._do_wr_burst(bank, max(t, done),
                                                       cadence))
                t_all = done
            elif cmd.kind == PRE_AB:
                t = self._issue_ca(t_all)
                for bank in self.banks:
                    bank.open_row = -1
                    bank.next_act = max(bank.next_act,
                                        max(t, bank.next_pre)
                                        + self.p.allbank_rp)
                t_all = max(t_all, t + self.p.ca_cmd_ns)
            elif cmd.kind in (ACT, RD, PRE):
                bank = self.banks[cmd.bank]
                t = self._issue_ca(t_all)
                if cmd.kind == ACT:
                    t_all = max(t_all, self._do_act(bank, cmd.row, t))
                elif cmd.kind == RD:
                    t_all = max(t_all, self._host_read(bank, t))
                else:
                    bank.open_row = -1
                    bank.next_act = max(bank.next_act,
                                        max(t, bank.next_pre) + self.p.tRP)
            else:
                raise ValueError(f"unknown command {cmd.kind}")
        self.stats.time_ns = t_all
        return self.stats
