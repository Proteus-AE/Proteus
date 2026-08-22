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

#include <algorithm>
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

  // Bank column cycle of a connectivity mode. The near-bank read path is
  // bank-private, so its cadence is the bank's own column cycle tCCD_PIM in
  // direct mode, floored by the interval at which the PE accepts a burst;
  // broadcasting distributes one readout to `broadcast_fanout` PEs over the
  // shared bank-group wires, and each of them must then issue that many MACs,
  // so the cadence relaxes to max(tCCD_L, fanout * mac_ns).
  ns_t column_cadence(ConnectivityMode m) const {
    ns_t mac = geom_.mac_ns(timing_.burst_bytes);
    if (m == ConnectivityMode::BROADCAST)
      return std::max(timing_.tCCD_L, geom_.broadcast_fanout * mac);
    return std::max(timing_.tCCD_PIM, mac);
  }
  ns_t column_cadence() const { return column_cadence(mode_ctrl_.mode()); }

  // B/s, all banks streaming into their local PEs.
  double peak_internal_bw(ConnectivityMode m) const {
    return geom_.banks() * timing_.burst_bytes /
           (column_cadence(m) * 1e-9);
  }
  double peak_internal_bw() const {
    return peak_internal_bw(mode_ctrl_.mode());
  }

  // Column bursts per cadence this channel can still deliver to the external
  // interface while the all-bank PIM stream runs (Sec. IV-B).
  //
  // A bank serves one column access per near-bank column cycle and the PIM
  // stream claims one of those per cadence, leaving cadence/tCCD_PIM - 1 per
  // bank. A host burst, unlike a near-bank read, does traverse the bank-group
  // I/O and the channel's global I/O, so it is additionally bounded by tCCD_L
  // per bank group and by the DQ occupancy of the channel. Direct mode runs
  // at the minimum column cycle and therefore leaves nothing; broadcasting
  // runs at the bank-group-bus cadence and leaves a full slot per bank.
  double host_slot_bursts(ConnectivityMode m) const {
    const ns_t cadence = column_cadence(m);
    double free_per_bank = std::max(0.0, cadence / timing_.tCCD_PIM - 1.0);
    return std::min({banks_.size() * free_per_bank,
                     geom_.bankgroups() * cadence / timing_.tCCD_L,
                     cadence / timing_.burst_ns});
  }
  double host_slot_bursts() const {
    return host_slot_bursts(mode_ctrl_.mode());
  }

  // Concurrent external bandwidth of one channel (B/s).
  double host_slot_bw(ConnectivityMode m) const {
    return host_slot_bursts(m) * timing_.burst_bytes /
           (column_cadence(m) * 1e-9);
  }
  double host_slot_bw() const { return host_slot_bw(mode_ctrl_.mode()); }

 private:
  // One row activation on one bank, under the bank's own tRC, the die's
  // tRRD/tFAW window (which all-bank commands are exempt from) and the
  // refresh schedule; returns the effective activation time.
  ns_t do_act(Bank& bank, int row, ns_t t, bool all_bank);

  // command handlers; each returns the completion time of the command
  ns_t do_act_ab(int row, ns_t t);
  ns_t do_rdmac_ab(int row, ns_t t);
  ns_t do_wr_ab(int row, ns_t t);
  ns_t do_pre_ab(ns_t t);
  ns_t do_mode(ConnectivityMode m, ns_t t);
  ns_t do_single(const Command& c, ns_t t);

  ns_t issue_ca(ns_t t);
  ns_t max_fifo_drain() const;
  void host_fill_gap(int bg, ns_t gap_start, ns_t gap_end);
  void host_steal_slots(ns_t t0, ns_t t1, ns_t cadence);
  void host_generate(ns_t now);

  TimingParams timing_;
  Geometry geom_;
  std::vector<Bank> banks_;
  std::vector<DieState> dies_;
  std::vector<BankGroupBus> buses_;
  std::vector<ProcessingElement> pes_;
  ModeController mode_ctrl_;
  ns_t ca_free_ = 0.0;
  ns_t global_io_free_ = 0.0;             // channel DQ, host path only
  ChannelStats stats_;

  HostStreamConfig host_;
  ns_t host_next_arrival_ = 0.0;
  uint64_t host_backlog_ = 0;
  double host_credit_ = 0.0;
  std::vector<ns_t> host_arrival_times_;
};

}  // namespace pimcore
