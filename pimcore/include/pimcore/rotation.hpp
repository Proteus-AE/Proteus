// Concurrent PIM streaming and host (xPU) service on one LPDDR5X-PIM channel,
// at DRAM-command granularity.
//
// Broadcasting feeds the four PEs of a bank group from one 32 B transfer per
// PE cycle (1 ns) on the bank group's broadcast line, while a bank supplies
// one burst per tCCD_PIM (2 ns). Two banks of a bank group therefore suffice
// to keep the PEs busy. The per-channel PIM command queue streams them in
// rotation (`Schedule::ROTATE`): each bank streams one row at tCCD_PIM
// (64 bursts, 128 ns), offset by half a row (64 ns) and one PE cycle from its
// neighbour, and then rests for 128 ns. During its rest a bank is precharged,
// may serve host requests (ACT, RD, PRE through the normal I/O path and DQ),
// and re-activates its next PIM row before its next turn. PIM and host thus
// never hold rows in the same bank at the same time. The PIM schedule is a
// deterministic function of the kernel start, which the memory controller
// mirrors; host commands are placed into the capacity it leaves, so the PIM
// stream is never delayed.
//
// `Schedule::LOCKSTEP` is the all-bank alternative (every bank of a group
// streams at the fan-in cadence, 4 ns) and `Schedule::DIRECT` the one-to-one
// mode (every bank streams at tCCD_PIM to its own PE).
//
// Every command is logged and re-verified by an independent checker:
//   bank      tRCD, tRAS, tRP, tRC, tRTP, one open row, tCCD_PIM (PIM reads)
//             and tCCD_L (host reads)
//   BG        one broadcast-line transfer per PE cycle; tCCD_L between host
//             reads of a bank group
//   die       tRRD_S/tRRD_L and the rolling tFAW window over ALL activations
//             (the staggered PIM activations are not exempt); refresh
//   channel   tCCD_S between host reads, I/O (GIO/DQ) occupancy, C/A
//             occupancy, DQ read/write turnaround
//   PE        4-entry operand FIFO, one MAC issue per PE cycle, concurrent
//             partial sums (accumulators) per PE
#pragma once

#include <string>
#include <vector>

#include "pimcore/timing.hpp"

namespace pimcore {

enum class Schedule { ROTATE, LOCKSTEP, DIRECT };

struct RotationConfig {
  Schedule schedule = Schedule::ROTATE;
  double bg_offset_ns = 13.0;      // schedule phase offset between bank groups
  std::vector<double> bg_offsets;  // explicit per-group offsets (overrides)
  std::vector<double> bank_phase;  // per-bank phase within the rotation (ns)
  int rows_per_bank = 32;          // PIM kernel length (rows streamed per bank)
  bool pim = true;                 // run the PIM kernel
  bool refresh = true;             // all-bank refresh every tREFI
  // PIM I/O on the channel's normal I/O path: input vectors written over DQ
  // into the PE vector registers and partial results / scores drained to the
  // channel SFU. Expressed as bytes per byte streamed from the banks.
  double pim_in_ratio = 0.0;       // vector bytes / operand bytes
  double pim_out_ratio = 0.0;      // result bytes / operand bytes
  bool pim_in_external = true;     // vectors arrive over DQ (bus turnaround)
  int pim_io_batch = 2;            // bursts per PIM I/O transfer job
  double pim_prefetch_ns = 512.0;  // vector prefetch window (double buffer)
  double pim_out_slack_ns = 256.0; // SFU input-buffer slack for results
  double pim_io_guard_ns = 8.0;    // issue a job this early before its deadline
  // host stream
  bool host = true;
  bool saturate = true;            // closed loop (else Poisson open loop)
  double offered_gbps = 0.0;       // open-loop offered load per channel
  int granule_bursts = 8;          // contiguous host bursts per bank row
  int window_requests = 256;       // controller reorder window (requests)
  double ext_bw_gbps = 15.625;     // per-channel share of the 1 TB/s interface
  double ext_bus_eff = 0.98;       // command/turnaround efficiency of DQ
  double turnaround_ns = 4.0;      // DQ read<->write turnaround
  double read_latency_ns = 20.0;   // RD command -> first data on DQ (RL)
  double data_latency_ns = 2.0;    // bank readout -> PE FIFO
  double duration_ns = 0.0;        // simulated span (0: until PIM finishes)
  unsigned seed = 1;
  bool trace = false;
  double trace_from_ns = 0.0;
  double trace_until_ns = 600.0;
};

struct TraceEvent {
  double t = 0.0;
  double dur = 0.0;
  std::string unit;   // "bg0.bank1", "bg0.bus", "bg0.fifo", "io", "ca"
  std::string kind;   // PIM_ACT, PIM_RD, PIM_PRE, HOST_ACT, HOST_RD, ...
  int value = 0;      // bank id, occupancy, ...
};

struct RotationReport {
  double span_ns = 0.0;            // simulated span
  double pim_time_ns = 0.0;        // kernel time (first ACT to last MAC)
  double pim_ideal_ns = 0.0;       // operand bytes / peak broadcast rate
  double pim_bursts = 0.0;         // bank reads
  double pim_gbps = 0.0;           // array read bandwidth of the kernel
  double pe_util = 0.0;            // MAC-issue cycles / kernel cycles
  double pim_io_bytes = 0.0;
  double host_bytes = 0.0;
  double host_gbps = 0.0;
  double host_frac = 0.0;          // of the sustained external bandwidth
  double host_lat_mean = 0.0, host_lat_p50 = 0.0, host_lat_p99 = 0.0,
         host_lat_max = 0.0;
  long host_acts = 0, pim_acts = 0, host_reads = 0;
  double bursts_per_host_act = 0.0;
  // checker results (all expected 0)
  int viol_act = 0;                // tRRD / tFAW
  int viol_bank = 0;               // tRCD/tRAS/tRP/tRC/tRTP/tCCD, row state
  int viol_conflict = 0;           // host and PIM rows in one bank at once
  int viol_bus = 0;                // broadcast-line double booking
  int viol_io = 0;                 // I/O or DQ double booking
  int viol_ca = 0;                 // C/A double booking
  int viol_refresh = 0;            // activity during refresh
  int pim_io_missed = 0;           // PIM I/O jobs past their deadline (stall)
  double pim_io_late_max = 0.0;
  int fifo_max = 0;
  int acc_max = 0;                 // concurrent partial sums per PE
  double ca_util = 0.0;
  double io_util = 0.0;
  double act_rate_per_us = 0.0;    // all activations
  std::vector<TraceEvent> trace;
};

RotationReport run_rotation(const TimingParams& t, const RotationConfig& c);

const char* schedule_name(Schedule s);

}  // namespace pimcore
