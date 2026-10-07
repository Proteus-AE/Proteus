"""DRAM timing parameters (ns domain) for the command-level backend."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingParams:
    tRCD: float
    tRP: float
    tRAS: float
    tRC: float
    tCCD_L: float          # same-BG column-to-column on the shared bus
    tCCD_S: float          # cross-BG column-to-column on the shared bus
    tCCD_PIM: float        # bank-local column-to-column (near-bank read path)
    tRRD_L: float          # same-BG ACT-to-ACT
    tRRD_S: float          # cross-BG ACT-to-ACT
    tFAW: float            # rolling four-ACT window per die
    tWR: float
    tWTR: float            # write-to-read turnaround within a bank
    tRTP: float
    tREFI: float           # all-bank refresh interval per die
    tRFCab: float          # all-bank refresh cycle time
    burst_bytes: int       # 32 B (x16 BL16)
    burst_ns: float        # DQ occupancy of one burst (burst_length / data rate)
    mac_ns: float          # PE issue interval for one 32 B operand burst
    ca_cmd_ns: float       # command/address bus occupancy per command
    row_bytes: int
    # all-bank command constraints (see configs/memory/*.yaml "allbank")
    allbank_rcd: float
    allbank_ras: float
    allbank_rp: float
    allbank_faw_exempt: bool

    @classmethod
    def from_config(cls, mem):
        burst = mem["io_width"] * mem["burst_length"] // 8
        ab = mem.get("allbank", {})
        return cls(
            tRCD=mem["tRCD_ns"], tRP=mem["tRP_ns"], tRAS=mem["tRAS_ns"],
            tRC=mem["tRC_ns"], tCCD_L=mem["tCCD_L_ns"], tCCD_S=mem["tCCD_S_ns"],
            tCCD_PIM=mem.get("tCCD_PIM_ns", mem["tCCD_L_ns"] / 2.0),
            tRRD_L=mem["tRRD_L_ns"], tRRD_S=mem["tRRD_S_ns"], tFAW=mem["tFAW_ns"],
            tWR=mem["tWR_ns"], tWTR=mem.get("tWTR_ns", 0.0),
            tRTP=mem["tRTP_ns"], tREFI=mem["tREFI_ns"],
            tRFCab=mem["tRFCab_ns"],
            burst_bytes=burst,
            burst_ns=mem["burst_length"] / mem["data_rate_mtps"] * 1e3,
            mac_ns=burst / (mem["pe_lanes"] * 2.0) / mem["pe_freq_ghz"],
            ca_cmd_ns=mem.get("ca_cmd_ns", 1.0),
            row_bytes=mem["row_bytes"],
            allbank_rcd=ab.get("act_rcd_ns", mem["tRCD_ns"]),
            allbank_ras=ab.get("ras_ns", mem["tRAS_ns"]),
            allbank_rp=ab.get("rp_ns", mem["tRP_ns"]),
            allbank_faw_exempt=bool(ab.get("faw_exempt", True)),
        )
