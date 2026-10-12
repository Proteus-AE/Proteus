"""DRAMPower-style command-based energy accounting for the PIM backend.

Energy is attributed per issued command from the counters produced by
``PimChannel.execute`` (ACT/PRE pairs, 32 B read bursts, bank-group
broadcast distribution, MAC issues, mode switches, refresh events), using
per-command energies derived from Micron
LPDDR5X current profiles. Near-bank reads terminate at the bank-local PE and
therefore exclude the I/O + PHY component; the external path adds it.
"""
from dataclasses import dataclass


@dataclass
class EnergyReport:
    act_pre_nj: float
    rd_array_nj: float
    wr_array_nj: float
    rd_io_nj: float
    bcast_nj: float
    mac_nj: float
    mode_nj: float
    refresh_nj: float

    @property
    def total_nj(self):
        return (self.act_pre_nj + self.rd_array_nj + self.wr_array_nj
                + self.rd_io_nj + self.bcast_nj + self.mac_nj + self.mode_nj
                + self.refresh_nj)

    def pj_per_byte(self, bytes_read):
        return self.total_nj * 1e3 / bytes_read if bytes_read else 0.0

    def describe(self):
        rows = [("ACT/PRE", self.act_pre_nj), ("RD array", self.rd_array_nj),
                ("WR array", self.wr_array_nj),
                ("RD I/O+PHY", self.rd_io_nj),
                ("BG broadcast", self.bcast_nj), ("PE MAC", self.mac_nj),
                ("mode switch", self.mode_nj), ("refresh", self.refresh_nj),
                ("total", self.total_nj)]
        return "\n".join(f"  {k:<12}: {v/1e3:10.2f} uJ" for k, v in rows)


class CommandEnergy:
    def __init__(self, mem_cfg):
        self.e = mem_cfg["command_energy_pj"]

    def account(self, stats, external=False):
        """Energy of an executed command stream (nJ). ``external=True`` adds
        the I/O+PHY component to every burst (host-path reads)."""
        e = self.e
        return EnergyReport(
            act_pre_nj=stats.n_act * e["act_pre"] * 1e-3,
            rd_array_nj=stats.n_rd_burst * e["rd_burst_array"] * 1e-3,
            wr_array_nj=stats.n_wr_burst * e.get("wr_burst_array", e["rd_burst_array"]) * 1e-3,
            rd_io_nj=(stats.n_rd_burst * e["rd_burst_io"] * 1e-3
                      if external else 0.0),
            bcast_nj=stats.n_broadcast * e.get("bg_broadcast", 0.0) * 1e-3,
            mac_nj=stats.n_mac * e["mac_op"] * 1e-3,
            mode_nj=stats.n_mode_switch * e["mode_switch"] * 1e-3,
            refresh_nj=stats.n_refresh * e["refresh_ab"] * 1e-3,
        )
