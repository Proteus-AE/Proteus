// Event-driven engine for one PIM channel.
//
// Executes a command stream under an earliest-issue policy subject to:
//   * per-bank row state and tRCD/tRP/tRAS/tRC/tCCD/tWTR/tWR,
//   * per-bank-group data-bus occupancy (one burst per `burst_ns`),
//   * per-die tRRD spacing and the rolling four-ACT tFAW window,
//   * aligned all-bank refresh every tREFI (blocking tRFCab),
//   * PE operand-FIFO back-pressure,
//   * command/address bus occupancy (all-bank commands issue once).
//
// A host (xPU) read stream can be attached to the channel; depending on the
// arbitration policy it either fills scheduling gaps (PIM priority), claims
// the next slot (host priority), or round-robins with the PIM stream.
#pragma once

#include <memory>
#include <vector>

#include "pimcore/bank.hpp"
#include "pimcore/pim.hpp"
#include "pimcore/stats.hpp"
#include "pimcore/timing.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

struct HostStreamConfig {
  bool enabled = false;
  double demand_gbps = 0.0;            // offered load; 0 = greedy
  ArbitrationPolicy policy = ArbitrationPolicy::PIM_PRIORITY;
  int burst_per_req = 2;               // 64 B host requests by default
};

class Channel {
 public:
  Channel(const TimingParams& timing, const Geometry& geom,
          ConnectivityMode initial_mode);

  void attach_host_stream(const HostStreamConfig& cfg) { host_ = cfg; }

  // Execute a full command stream; returns aggregate statistics.
  const ChannelStats& execute(const std::vector<Command>& stream);

  // Single-command issue (used by the co-execution engine): applies `c` at
  // or after `t_all` and returns the new stream front time.
  ns_t step(const Command& c, ns_t t_all);

  // Issue one host request (decoded coordinates) onto this channel's
  // resources: activates the row if needed and claims `bursts` bus slots.
  // Returns the completion time; row-hit status is reported via *row_hit.
  ns_t host_issue(int flat_bank, int row, int bursts, ns_t t, bool* row_hit);

  bool bank_row_open(int flat_bank, int row) const {
    return banks_[flat_bank].row_open(row);
  }
  void finalize(ns_t t) { stats_.time_ns = t; }

  const ChannelStats& stats() const { return stats_; }
  const std::vector<Bank>& banks() const { return banks_; }
  ConnectivityMode mode() const { return mode_ctrl_.mode(); }

  double peak_internal_bw() const {    // B/s, all BGs streaming
    return geom_.bankgroups() * timing_.burst_bytes / (timing_.burst_ns * 1e-9);
  }

 private:
  // command handlers; each returns the completion time of the command
  ns_t do_act_ab(int row, ns_t t);
  ns_t do_rdmac_ab(int row, ns_t t);
  ns_t do_wr_ab(int row, ns_t t);
  ns_t do_pre_ab(ns_t t);
  ns_t do_mode(ConnectivityMode m, ns_t t);
  ns_t do_single(const Command& c, ns_t t);
  ns_t do_refresh_ab(ns_t t);

  ns_t issue_ca(ns_t t);
  ns_t max_fifo_drain() const;
  void host_fill_gap(int bg, ns_t gap_start, ns_t gap_end);
  void host_generate(ns_t now);

  TimingParams timing_;
  Geometry geom_;
  std::vector<Bank> banks_;
  std::vector<DieState> dies_;
  std::vector<BankGroupBus> buses_;
  std::vector<ProcessingElement> pes_;
  ModeController mode_ctrl_;
  ns_t ca_free_ = 0.0;
  ChannelStats stats_;

  HostStreamConfig host_;
  ns_t host_next_arrival_ = 0.0;
  uint64_t host_backlog_ = 0;
  std::vector<ns_t> host_arrival_times_;
};

}  // namespace pimcore
