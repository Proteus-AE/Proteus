"""LPDDR5X near-bank PIM machine model (Sec. IV-C, Table III).

Derives the sustained rates consumed by the timing engine from the DRAM
organization and core timing of ``configs/memory/*.yaml``. Everything below
is a closed form of the all-bank execution model; the event-driven backend
in ``proteus_sim.dram`` executes the same organization command by command
and ``experiments/run_microbench.py`` checks the two against each other.

Organization
------------
A device is ``packages x channels x dies x bank groups x banks``. Every bank
carries a 16-lane FP16 MAC PE fed through the bank's own I/O sense
amplifiers (Fig. 8(c)), so the near-bank read path never traverses the
shared bank-group bus, the channel's global I/O, or any cross-BG crossbar.

Connectivity modes
------------------
All banks execute concurrently through all-bank commands in both modes; what
changes is where a bank readout goes.

*direct*    One-to-one: each bank drives its own PE over the bank-private
            path. Every fetched matrix element contributes to one vector, so
            the column cadence is the bank's own near-bank column cycle
            ``tCCD_PIM`` -- tCCD_L constrains column commands whose data
            traverses the shared BG I/O, which this path bypasses -- floored
            by the interval at which the PE can accept a burst.

*broadcast* One-to-many: each bank readout is shared with all PEs of its bank
            group, and every PE assembles one burst from every bank into its
            four-entry operand FIFO, whose depth matches the BG fan-in. A 4:1
            selector then feeds those bursts into the MAC array one issue at a
            time, so each matrix fetch is reused across ``fanout`` concurrent
            input vectors. The bank column cadence consequently matches the
            fan-in, ``max(tCCD_L, fanout * mac_ns)``, and the compute per byte
            of DRAM access rises ``fanout``-fold: from one MAC per 32 B burst
            to four.

Because the reuse width is bounded by the fan-in, beyond ``fanout`` vectors
the matrix must be streamed again. At peak rates the two modes bracket the
near-bank PE array exactly: direct mode delivers 16.384 TB/s at one MAC per
byte-pair and leaves the PEs half idle, while broadcasting delivers half
that with four-fold reuse and saturates them.

Concurrent host traffic
-----------------------
Both modes leave the bank column ports occupied once per cadence. Direct
mode runs at the minimum cycle and returns no slots, so an external stream
can only be interleaved between PIM phases. Broadcasting runs at twice that
period and leaves one slot per bank, which -- once the bank-group I/O gating
and the channel's DQ occupancy are accounted for -- exceeds the external
interface, so an xPU stream runs concurrently at its full sustained rate.
``coexec_bw`` derives that headroom; the command-level backend measures the
same quantity by counting the column slots the PIM stream leaves free
(``experiments/run_microbench.py`` (3)).

For the evaluated LPDDR5X-8533 organization the derivation yields the Table
III aggregates: 1024 banks, 16.384 TB/s of all-bank internal read bandwidth,
and 32.768 TFLOPS of near-bank FP16 throughput, which broadcasting saturates
at 8.192 TB/s of DRAM access.
"""
from dataclasses import dataclass


@dataclass
class DerivedMemory:
    # organization
    channels: int            # per device
    dies: int                # per device
    bankgroups: int          # per device
    banks: int               # per device
    burst_bytes: int         # 32 B (x16, BL16)
    # column cadences of the two connectivity modes (ns)
    t_col_direct_ns: float
    t_col_broadcast_ns: float
    mac_ns: float
    fanout: int
    # peak / sustained rates
    internal_peak: float     # B/s, direct-mode all-bank read bandwidth
    broadcast_peak: float    # B/s, broadcasting-mode all-bank read bandwidth
    internal_eff: float      # row-activation + refresh efficiency, direct mode
    broadcast_eff: float     # ... and in broadcasting mode
    refresh_avail: float     # 1 - tRFCab/tREFI
    internal_bw: float       # B/s sustained, direct mode
    broadcast_bw: float      # B/s sustained, broadcasting mode
    pe_flops_peak: float     # FLOP/s aggregate near-bank MAC issue
    pe_flops: float          # FLOP/s sustained
    external_peak: float     # B/s external interface (Table III: 1 TB/s)
    external_bw: float       # B/s sustained external interface
    # external bandwidth available concurrently with an all-bank PIM stream
    coexec_direct: float     # B/s, direct mode
    coexec_broadcast: float  # B/s, broadcasting mode
    capacity: float          # bytes per device
    cfg: dict = None

    @property
    def ridge_pim(self):
        """AI_PIM = F_PIM / BW_in (Sec. IV-D); 2.0 for this configuration."""
        return self.pe_flops_peak / self.internal_peak

    def mode_rates(self, mode):
        """(sustained bandwidth, operand reuse) of a connectivity mode."""
        if mode == "broadcast":
            return self.broadcast_bw, self.fanout
        return self.internal_bw, 1

    def coexec_bw(self, mode):
        """External bandwidth available while this mode streams (B/s)."""
        return self.coexec_broadcast if mode == "broadcast" \
            else self.coexec_direct

    def describe(self):
        return (
            f"  channels / dies        : {self.channels} / {self.dies}\n"
            f"  bank groups / banks    : {self.bankgroups} / {self.banks}\n"
            f"  column cadence dir/bc  : {self.t_col_direct_ns:.2f} / "
            f"{self.t_col_broadcast_ns:.2f} ns (MAC issue {self.mac_ns:.2f} ns)\n"
            f"  internal BW peak/sust. : {self.internal_peak/1e12:.2f} / "
            f"{self.internal_bw/1e12:.2f} TB/s (eff {self.internal_eff:.3f})\n"
            f"  broadcast BW peak/sust.: {self.broadcast_peak/1e12:.2f} / "
            f"{self.broadcast_bw/1e12:.2f} TB/s (eff {self.broadcast_eff:.3f}, "
            f"x{self.fanout} operand reuse)\n"
            f"  near-bank PE peak/sust.: {self.pe_flops_peak/1e12:.2f} / "
            f"{self.pe_flops/1e12:.2f} TFLOPS\n"
            f"  PIM ridge point AI_PIM : {self.ridge_pim:.2f} FLOP/B\n"
            f"  external BW peak/sust. : {self.external_peak/1e12:.2f} / "
            f"{self.external_bw/1e12:.2f} TB/s\n"
            f"  capacity               : {self.capacity/1e9:.0f} GB")


def _check(name, got, want, tol=0.05):
    if want and abs(got - want) / want > tol:
        raise ValueError(
            f"derived {name} = {got:.4g} deviates from the declared "
            f"configuration value {want:.4g} by more than {tol:.0%}; the "
            f"memory config and the organization model disagree")


def derive(mem):
    """Compute the sustained machine parameters of a memory configuration."""
    channels = mem["packages_per_device"] * mem["channels_per_package"]
    dies = channels * mem["dies_per_channel"]
    bgs = dies * mem["bankgroups_per_die"]
    banks = bgs * mem["banks_per_bankgroup"]

    burst_bytes = mem["io_width"] * mem["burst_length"] // 8        # 32 B
    fanout = mem.get("broadcast_fanout", mem["banks_per_bankgroup"])

    # One MAC issue consumes one burst of matrix operand: 16 lanes x 2 B.
    #
    # Direct mode: every bank drives its own PE, so the cycle is the bank's
    # near-bank column cycle tCCD_PIM (see the memory config for why that is
    # not tCCD_L), unless the PE is the slower of the two.
    # Broadcast mode: one readout is distributed to `fanout` PEs over the
    # bank-group bus, so the cycle is bounded below by the shared-bus
    # constraint tCCD_L *and* by the time each PE needs to retire the fanout
    # bursts it now receives per cycle.
    mac_ns = burst_bytes / (mem["pe_lanes"] * 2.0) / mem["pe_freq_ghz"]
    t_col_pim = mem.get("tCCD_PIM_ns", mem["tCCD_L_ns"] / 2.0)
    t_col_direct = max(t_col_pim, mac_ns)
    t_col_bcast = max(mem["tCCD_L_ns"], fanout * mac_ns)

    internal_peak = banks * burst_bytes / (t_col_direct * 1e-9)
    broadcast_peak = banks * burst_bytes / (t_col_bcast * 1e-9)
    pe_peak = banks * mem["pe_lanes"] * 2.0 * mem["pe_freq_ghz"] * 1e9

    # Row-activation amortization under all-bank execution. One row cycle is
    #
    #   ACT_AB -> tRCD -> (row_bytes/burst) column cycles -> tRTP -> PRE_AB
    #          -> tRP -> next ACT_AB,
    #
    # and the streaming window is the column cycles alone. Activation is
    # fully exposed once per row because all banks run in lockstep and cannot
    # hide each other's tRCD/tRP -- the price the all-bank model pays for its
    # bandwidth. tRTP is measured from the *last* column command, which is
    # issued one cycle before the window ends, so only the part of tRTP that
    # outlasts a column cycle is exposed. All-bank refresh additionally
    # removes tRFCab out of every tREFI. tRAS is never binding here: the
    # streaming window alone exceeds it by more than 3x.
    #
    # Both values are reproduced within 2% by the event-driven command-level
    # backend, which enforces the same constraints per command
    # (experiments/run_microbench.py, docs/dram-model.md).
    ab = mem.get("allbank", {})
    rcd_ns = ab.get("act_rcd_ns", mem["tRCD_ns"])
    rp_ns = ab.get("rp_ns", mem["tRP_ns"])
    refresh_avail = 1.0 - mem["tRFCab_ns"] / mem["tREFI_ns"]
    bursts_per_row = mem["row_bytes"] / burst_bytes

    def _eff(t_col):
        stream_ns = bursts_per_row * t_col
        tail_ns = max(0.0, mem["tRTP_ns"] - t_col)
        row_ns = rcd_ns + stream_ns + tail_ns + rp_ns
        return stream_ns / row_ns * refresh_avail

    eff = _eff(t_col_direct)
    eff_b = _eff(t_col_bcast)

    _check("internal bandwidth (TB/s)", internal_peak / 1e12,
           mem.get("nominal_internal_bw_tbps"))
    _check("near-bank PE throughput (TFLOPS)", pe_peak / 1e12,
           mem.get("nominal_pe_tflops"))

    # The external path interleaves across the banks of a channel, so row
    # activation is hidden; only refresh and bus overhead remain.
    ext_peak = mem["external_bw_per_device_tbps"] * 1e12
    ext_eff = refresh_avail * mem.get("ext_bus_efficiency", 1.0)

    # Column slots the all-bank stream leaves free for a concurrent host
    # stream, per device. A bank serves one column access per tCCD_PIM and
    # the PIM stream claims one per cadence; a host burst additionally passes
    # the bank-group I/O (tCCD_L per BG) and the channel DQ (burst_ns).
    burst_ns = mem["burst_length"] / mem["data_rate_mtps"] * 1e3

    def _coexec(t_col):
        free_per_bank = max(0.0, t_col / t_col_pim - 1.0)
        bursts = min(banks * free_per_bank,
                     bgs * t_col / mem["tCCD_L_ns"],
                     channels * t_col / burst_ns)
        return bursts * burst_bytes / (t_col * 1e-9)
    return DerivedMemory(
        channels=channels, dies=dies, bankgroups=bgs, banks=banks,
        burst_bytes=burst_bytes,
        t_col_direct_ns=t_col_direct, t_col_broadcast_ns=t_col_bcast,
        mac_ns=mac_ns, fanout=int(fanout),
        internal_peak=internal_peak, broadcast_peak=broadcast_peak,
        internal_eff=eff, broadcast_eff=eff_b, refresh_avail=refresh_avail,
        internal_bw=internal_peak * eff,
        broadcast_bw=broadcast_peak * eff_b,
        pe_flops_peak=pe_peak, pe_flops=pe_peak * mem["pe_pipeline_eff"],
        external_peak=ext_peak,
        external_bw=ext_peak * ext_eff,
        coexec_direct=min(_coexec(t_col_direct), ext_peak * ext_eff),
        coexec_broadcast=min(_coexec(t_col_bcast), ext_peak * ext_eff),
        capacity=mem["capacity_per_package_gb"] * 1e9 * mem["packages_per_device"],
        cfg=mem)


def short_payload_factor(mem, d_model, what="system"):
    """Uniform derate for small-hidden-dimension models whose per-command
    column payloads are too short to amortize command/row-activation overhead
    (applies to every evaluated system)."""
    sp = mem["short_payload"]
    if d_model > sp["d_model_threshold"]:
        return 1.0
    return sp["system_efficiency"] if what == "system" else sp["pim_cmd_efficiency"]
