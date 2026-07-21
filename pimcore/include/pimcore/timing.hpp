// DRAM timing parameter tables per memory substrate.
//
// All values are held in nanoseconds. The tables are populated either from a
// configuration file (`timing:` section) or from the built-in per-substrate
// defaults in src/substrates/{lpddr5x_pim,hbm_pim,gddr6_aim}.cpp, which
// follow JEDEC-class datasheet values.
#pragma once

#include "pimcore/config.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

struct TimingParams {
  // Row timings
  ns_t tRCD = 18.0;      // ACT -> column command
  ns_t tRP = 21.0;       // PRE -> ACT
  ns_t tRAS = 42.0;      // ACT -> PRE
  ns_t tRC = 63.0;       // ACT -> ACT (same bank)
  // Column timings
  ns_t tCCD_L = 4.0;     // same-bank-group column-to-column
  ns_t tCCD_S = 2.0;     // cross-bank-group column-to-column
  ns_t tRTP = 7.5;       // RD -> PRE
  ns_t tWR = 34.0;       // WR data -> PRE
  ns_t tWTR = 10.0;      // WR -> RD turnaround
  // Activation windows
  ns_t tRRD_L = 7.5;     // same-BG ACT-to-ACT
  ns_t tRRD_S = 3.75;    // cross-BG ACT-to-ACT
  ns_t tFAW = 20.0;      // rolling four-ACT window per die
  // Refresh
  ns_t tREFI = 3906.0;   // refresh interval
  ns_t tRFCab = 280.0;   // all-bank refresh cycle
  // Buses
  ns_t burst_ns = 2.0;   // BG data-bus occupancy per burst (DDR cadence)
  ns_t ca_cmd_ns = 1.0;  // command/address bus occupancy per command
  // Geometry-coupled
  int burst_bytes = 32;  // payload per column burst
  int row_bytes = 2048;  // per-bank row buffer

  static TimingParams defaults_for(Substrate s);
  static TimingParams from_config(const ConfigNode& root, Substrate s);
  void validate() const;   // basic JEDEC-consistency assertions
};

struct Geometry {
  int channels = 1;        // simulated channels (SPMD scaling factor)
  int dies_per_channel = 4;
  int bankgroups_per_die = 4;
  int banks_per_bankgroup = 4;
  int pe_lanes = 16;
  double pe_freq_ghz = 1.0;
  int pe_fifo_depth = 4;
  int broadcast_fanout = 4;

  int bankgroups() const { return dies_per_channel * bankgroups_per_die; }
  int banks() const { return bankgroups() * banks_per_bankgroup; }

  static Geometry defaults_for(Substrate s);
  static Geometry from_config(const ConfigNode& root, Substrate s);
};

// Per-command energy table (pJ), DRAMPower-style attribution.
struct EnergyTable {
  double act_pre_pj = 900.0;       // one ACT+PRE pair per bank
  double rd_burst_array_pj = 450;  // array read terminated at the local PE
  double wr_burst_array_pj = 480;  // array write (KV append)
  double rd_burst_io_pj = 700;     // added I/O + PHY when leaving the die
  double mac_op_pj = 18.0;         // one MAC issue
  double mode_switch_pj = 40.0;    // mode-register update per channel
  double refresh_ab_pj = 28000.0;  // all-bank refresh per die

  static EnergyTable defaults_for(Substrate s);
  static EnergyTable from_config(const ConfigNode& root, Substrate s);
};

}  // namespace pimcore
