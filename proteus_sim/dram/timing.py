"""DRAM timing parameters (ns domain) for the command-level backend."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingParams:
    tRCD: float
    tRP: float
    tRAS: float
    tRC: float
    tCCD_L: float          # same-BG column-to-column
    tCCD_S: float          # cross-BG column-to-column
    tRRD_L: float          # same-BG ACT-to-ACT
    tRRD_S: float          # cross-BG ACT-to-ACT
    tFAW: float            # rolling four-ACT window per die
    tWR: float
    tRTP: float
    tREFI: float           # all-bank refresh interval per die
    tRFCab: float          # all-bank refresh cycle time
    burst_bytes: int       # 32 B (x16 BL16)
    burst_ns: float        # BG data-bus occupancy of one burst (DDR cadence)
    ca_cmd_ns: float       # command/address bus occupancy per command
    row_bytes: int

    @classmethod
    def from_config(cls, mem):
        return cls(
            tRCD=mem["tRCD_ns"], tRP=mem["tRP_ns"], tRAS=mem["tRAS_ns"],
            tRC=mem["tRC_ns"], tCCD_L=mem["tCCD_L_ns"], tCCD_S=mem["tCCD_S_ns"],
            tRRD_L=mem["tRRD_L_ns"], tRRD_S=mem["tRRD_S_ns"], tFAW=mem["tFAW_ns"],
            tWR=mem["tWR_ns"], tRTP=mem["tRTP_ns"], tREFI=mem["tREFI_ns"],
            tRFCab=mem["tRFCab_ns"],
            burst_bytes=mem["io_width"] * mem["burst_length"] // 8,
            burst_ns=mem["tCCD_L_ns"] / 2.0,
            ca_cmd_ns=1.0,
            row_bytes=mem["row_bytes"],
        )
