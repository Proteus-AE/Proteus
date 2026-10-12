// DRAM timing, organization and per-command energy of one memory substrate.
//
// Every value is read from the substrate's memory configuration
// (configs/memory/<name>.yaml), the same file the Python layer consumes, so
// a JEDEC parameter has exactly one definition in the artifact. All times
// are held in nanoseconds.
#pragma once

#include "pimcore/config.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

struct TimingParams {
  // Row timings
  ns_t tRCD = 0.0;       // ACT -> column command
  ns_t tRP = 0.0;        // PRE -> ACT
  ns_t tRAS = 0.0;       // ACT -> PRE
  ns_t tRC = 0.0;        // ACT -> ACT (same bank)
  // Column timings
  ns_t tCCD_L = 0.0;     // same-bank-group column-to-column (shared BG bus)
  ns_t tCCD_S = 0.0;     // cross-bank-group column-to-column
  ns_t tCCD_PIM = 0.0;   // bank-local column-to-column (near-bank read path)
  ns_t tRTP = 0.0;       // RD -> PRE
  ns_t tWR = 0.0;        // WR data -> PRE
  ns_t tWTR = 0.0;       // WR -> RD turnaround within a bank
  // Activation windows
  ns_t tRRD_L = 0.0;     // same-BG ACT-to-ACT
  ns_t tRRD_S = 0.0;     // cross-BG ACT-to-ACT
  ns_t tFAW = 0.0;       // rolling four-ACT window per die
  // Refresh
  ns_t tREFI = 0.0;      // all-bank refresh interval per die
  ns_t tRFCab = 0.0;     // all-bank refresh cycle time
  // Buses
  ns_t burst_ns = 0.0;   // DQ occupancy of one burst (burst_length / data rate)
  ns_t ca_cmd_ns = 1.0;  // command/address bus occupancy per command
  // All-bank command constraints. A single all-bank command is provisioned by
  // the device as one operation, so it is not bound by tRRD/tFAW -- those
  // limit the controller's ability to stagger *independent* activations,
  // which all-bank execution does not do.
  ns_t allbank_rcd = 0.0;
  ns_t allbank_ras = 0.0;
  ns_t allbank_rp = 0.0;
  bool allbank_faw_exempt = true;
  // Geometry-coupled
  int burst_bytes = 0;   // payload per column burst (io_width x BL / 8)
  int row_bytes = 0;     // per-bank row buffer

  static TimingParams from_config(const ConfigNode& mem);
  void validate() const;   // basic JEDEC-consistency assertions
};

struct Geometry {
  int channels = 1;        // per device (SPMD scaling factor)
  int dies_per_channel = 1;
  int bankgroups_per_die = 4;
  int banks_per_bankgroup = 4;
  int pe_lanes = 16;
  double pe_freq_ghz = 1.0;
  int pe_fifo_depth = 4;
  int broadcast_fanout = 4;

  int bankgroups() const { return dies_per_channel * bankgroups_per_die; }
  int banks() const { return bankgroups() * banks_per_bankgroup; }

  // One 16-lane FP16 MAC issue consumes one 32 B burst of matrix operand.
  double mac_ns(int burst_bytes) const {
    return burst_bytes / (pe_lanes * 2.0) / pe_freq_ghz;
  }

  static Geometry from_config(const ConfigNode& mem);
};

// Per-command energy table (pJ), DRAMPower-style attribution.
struct EnergyTable {
  double act_pre_pj = 0.0;         // one ACT+PRE pair per bank
  double rd_burst_array_pj = 0.0;  // array read terminated at the local PE
  double wr_burst_array_pj = 0.0;  // array write (KV append)
  double rd_burst_io_pj = 0.0;     // added I/O + PHY when leaving the die
  double bg_broadcast_pj = 0.0;    // BG-local distribution of one burst
  double mac_op_pj = 0.0;          // one MAC issue
  double mode_switch_pj = 0.0;     // mode-register update per channel
  double refresh_ab_pj = 0.0;      // all-bank refresh per die

  static EnergyTable from_config(const ConfigNode& mem);
};

}  // namespace pimcore
