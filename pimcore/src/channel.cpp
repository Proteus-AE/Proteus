#include "pimcore/channel.hpp"

#include <algorithm>
#include <stdexcept>

namespace pimcore {

Channel::Channel(const TimingParams& timing, const Geometry& geom,
                 ConnectivityMode initial_mode)
    : timing_(timing), geom_(geom) {
  timing_.validate();
  for (int d = 0; d < geom_.dies_per_channel; ++d) {
    for (int g = 0; g < geom_.bankgroups_per_die; ++g) {
      int bg = d * geom_.bankgroups_per_die + g;
      for (int b = 0; b < geom_.banks_per_bankgroup; ++b)
        banks_.emplace_back(d, bg, b);
    }
    dies_.emplace_back(timing_);
  }
  buses_.resize(geom_.bankgroups());
  pes_.resize(geom_.banks());
  const ns_t mac_ns = geom_.mac_ns(timing_.burst_bytes);
  for (auto& pe : pes_) pe.configure(geom_.pe_fifo_depth, mac_ns);
  mode_ctrl_.switch_mode(initial_mode);
}

ns_t Channel::issue_ca(ns_t t) {
  t = std::max(t, ca_free_);
  ca_free_ = t + timing_.ca_cmd_ns;
  ++stats_.n_cmds;
  return t;
}

ns_t Channel::max_fifo_drain() const {
  ns_t d = 0.0;
  for (const auto& pe : pes_) d = std::max(d, pe.drain_time());
  return d;
}

// ---------------------------------------------------------------- //
// Host stream: opportunistic single-bank reads sharing the channel.
// ---------------------------------------------------------------- //

void Channel::host_generate(ns_t now) {
  if (!host_.enabled) return;
  if (host_.demand_gbps <= 0.0) {
    // Greedy stream: always one request backlogged.
    if (host_backlog_ == 0) {
      host_backlog_ = 1;
      host_arrival_times_.push_back(now);
      ++stats_.host_reqs_issued;
    }
    return;
  }
  double bytes_per_req =
      static_cast<double>(host_.burst_per_req) * timing_.burst_bytes;
  ns_t inter_arrival = bytes_per_req / (host_.demand_gbps * 1e9) * 1e9;
  while (host_next_arrival_ <= now) {
    ++host_backlog_;
    ++stats_.host_reqs_issued;
    host_arrival_times_.push_back(host_next_arrival_);
    host_next_arrival_ += inter_arrival;
  }
}

void Channel::host_steal_slots(ns_t t0, ns_t t1, ns_t cadence) {
  // The host stream claims the column slots the all-bank PIM datapath leaves
  // free: `host_slot_bursts` counts them per cadence, bounded by the bank
  // column ports, the bank-group I/O gating and the channel's DQ occupancy.
  // Direct mode drives every bank at its minimum column cycle and therefore
  // leaves nothing; broadcasting runs at twice that period and frees enough
  // to saturate the external interface.
  if (!host_.enabled || t1 <= t0) return;
  double per_cadence = host_slot_bursts();
  if (per_cadence <= 0.0) return;
  host_generate(t1);
  host_credit_ += (t1 - t0) / cadence * per_cadence;
  while (host_credit_ >= host_.burst_per_req && host_backlog_ > 0) {
    host_credit_ -= host_.burst_per_req;
    --host_backlog_;
    ++stats_.host_reqs_served;
    stats_.host_bursts += host_.burst_per_req;
    stats_.host_bytes +=
        static_cast<uint64_t>(host_.burst_per_req) * timing_.burst_bytes;
    if (!host_arrival_times_.empty()) {
      stats_.host_latency.record(t1 - host_arrival_times_.front());
      host_arrival_times_.erase(host_arrival_times_.begin());
    }
    host_generate(t1);
  }
}

void Channel::host_fill_gap(int /*bg*/, ns_t gap_start, ns_t gap_end) {
  if (!host_.enabled) return;
  host_generate(gap_end);
  ns_t req_ns = host_.burst_per_req * timing_.burst_ns;
  while (host_backlog_ > 0 && gap_start + req_ns <= gap_end) {
    ns_t done = gap_start + req_ns;
    --host_backlog_;
    ++stats_.host_reqs_served;
    stats_.host_bursts += host_.burst_per_req;
    stats_.host_bytes +=
        static_cast<uint64_t>(host_.burst_per_req) * timing_.burst_bytes;
    if (!host_arrival_times_.empty()) {
      stats_.host_latency.record(done - host_arrival_times_.front());
      host_arrival_times_.erase(host_arrival_times_.begin());
    }
    gap_start = done;
  }
}

// ---------------------------------------------------------------- //
// All-bank command handlers.
// ---------------------------------------------------------------- //

ns_t Channel::do_act(Bank& bank, int row, ns_t t, bool all_bank) {
  DieState& die = dies_[bank.die()];
  const bool exempt = all_bank && timing_.allbank_faw_exempt;
  ns_t bt = std::max(t, bank.next_act());
  if (!exempt) bt = die.constrain_act(bt);
  bt = die.constrain_refresh(bt, &stats_.refresh_stall_ns, &stats_.n_refresh);
  bt = bank.apply_act(bt, row, timing_, all_bank);
  if (!exempt) die.record_act(bt);
  ++stats_.n_act;
  return bt;
}

ns_t Channel::do_act_ab(int row, ns_t t) {
  ns_t done = t;
  for (auto& bank : banks_) done = std::max(done, do_act(bank, row, t, true));
  return done;
}

ns_t Channel::do_rdmac_ab(int row, ns_t t) {
  // All-bank near-bank read: every bank drives one 32 B burst out of its own
  // I/O sense amplifiers into its co-located PE. In direct mode that path is
  // bank-private and never touches a shared bus; in broadcasting mode the
  // readout is distributed over the BG-local wires to every PE of the group,
  // which is the only case where the bank-group bus is occupied.
  const ns_t cadence = column_cadence();
  const bool bcast = mode_ctrl_.mode() == ConnectivityMode::BROADCAST;
  const ns_t mac_ns = geom_.mac_ns(timing_.burst_bytes);
  ns_t done = t;
  const ns_t t0 = t;
  for (size_t b = 0; b < banks_.size(); ++b) {
    Bank& bank = banks_[b];
    if (bank.row_open(row)) {
      ++bank.hits;
      ++stats_.row_hits;
    } else {
      ++stats_.row_misses;
    }
    DieState& die = dies_[bank.die()];
    ns_t bt = die.constrain_refresh(std::max(t, bank.next_col()),
                                    &stats_.refresh_stall_ns,
                                    &stats_.n_refresh);
    if (bcast) bt = std::max(bt, buses_[bank.bank_group()].next_free());
    ns_t stall = 0.0;
    if (bcast) {
      // Every PE of the group assembles one burst from every bank of the
      // group, so a readout is pushed into all of them.
      int base = bank.bank_group() * geom_.banks_per_bankgroup;
      for (int k = 0; k < geom_.banks_per_bankgroup; ++k)
        stall = std::max(stall, pes_[base + k].push(bt));
      stats_.n_mac += geom_.banks_per_bankgroup;
      ++stats_.n_broadcast;
    } else {
      stall = pes_[b].push(bt);
      ++stats_.n_mac;
    }
    stats_.fifo_stall_ns += stall;
    bt += stall;
    if (bcast) buses_[bank.bank_group()].reserve(bt + mac_ns);
    bt = bank.apply_read(bt, timing_, cadence);
    ++stats_.n_rd_burst;
    stats_.pim_bytes_read += timing_.burst_bytes;
    done = std::max(done, bt);
  }
  host_steal_slots(t0, done, cadence);
  return done;
}

ns_t Channel::do_wr_ab(int row, ns_t t) {
  // All-bank column write (the in-place KV append of Sec. IV-A). Like every
  // other all-bank command it is issued once and executed by every bank, so
  // a row that is not already open is opened by an all-bank activation --
  // not by a staggered sequence of single-bank ACTs -- and the write runs at
  // the column cadence of the mode in force. The data never leaves the die,
  // so the bank-group bus stays free.
  const ns_t cadence = column_cadence();
  ns_t done = t;
  for (auto& bank : banks_) {
    if (!bank.row_open(row)) done = std::max(done, do_act(bank, row, t, true));
    DieState& die = dies_[bank.die()];
    ns_t bt = die.constrain_refresh(std::max(std::max(t, done),
                                             bank.next_col()),
                                    &stats_.refresh_stall_ns,
                                    &stats_.n_refresh);
    bt = bank.apply_write(bt, timing_, cadence);
    ++stats_.n_wr_burst;
    stats_.pim_bytes_written += timing_.burst_bytes;
    done = std::max(done, bt);
  }
  return done;
}

ns_t Channel::do_pre_ab(ns_t t) {
  for (auto& bank : banks_) bank.apply_pre(t, timing_, /*all_bank=*/true);
  ++stats_.n_pre;
  return t + timing_.ca_cmd_ns;
}

ns_t Channel::do_mode(ConnectivityMode m, ns_t t_all) {
  // The mode register takes effect once the in-flight operand FIFOs drain
  // (Sec. IV-C); the command itself occupies the command/address bus. Only a
  // command that actually changes the connectivity is a reconfiguration.
  ns_t t = issue_ca(std::max(t_all, max_fifo_drain()));
  if (mode_ctrl_.switch_mode(m)) ++stats_.n_mode_switch;
  return std::max(t_all, t + timing_.ca_cmd_ns);
}

ns_t Channel::do_single(const Command& c, ns_t t) {
  if (c.bank < 0 || c.bank >= static_cast<int>(banks_.size()))
    throw std::runtime_error("single-bank command with invalid bank id");
  Bank& bank = banks_[c.bank];
  DieState& die = dies_[bank.die()];
  switch (c.kind) {
    case CommandKind::ACT:
      return do_act(bank, c.row, t, /*all_bank=*/false);
    case CommandKind::RD: {
      // Single-bank host read: an array access like any other, plus the
      // channel's global I/O on the way off the die. Its column-to-column
      // cycle is the full tCCD_L -- it does traverse the bank-group I/O the
      // near-bank path bypasses -- and it is charged to both the array and
      // the host counters.
      ns_t bt = std::max(std::max(t, bank.next_col()), global_io_free_);
      bt = die.constrain_refresh(bt, &stats_.refresh_stall_ns,
                                 &stats_.n_refresh);
      global_io_free_ = bt + timing_.burst_ns;
      bank.apply_read(bt, timing_);
      ++stats_.n_rd_burst;
      stats_.pim_bytes_read += timing_.burst_bytes;
      stats_.host_bytes += timing_.burst_bytes;
      ++stats_.host_bursts;
      return bt + timing_.burst_ns;
    }
    case CommandKind::PRE:
      bank.apply_pre(t, timing_);
      ++stats_.n_pre;
      return t + timing_.ca_cmd_ns;
    default:
      throw std::runtime_error("unhandled single-bank command");
  }
}

ns_t Channel::host_issue(int flat_bank, int row, int bursts, ns_t t,
                         bool* row_hit) {
  Bank& bank = banks_[flat_bank];
  DieState& die = dies_[bank.die()];
  if (row_hit) *row_hit = bank.row_open(row);
  ns_t bt = t;
  if (!bank.row_open(row)) bt = do_act(bank, row, bt, /*all_bank=*/false);
  ns_t done = bt;
  for (int i = 0; i < bursts; ++i) {
    ns_t ct = std::max(std::max(done, bank.next_col()), global_io_free_);
    ct = die.constrain_refresh(ct, &stats_.refresh_stall_ns,
                               &stats_.n_refresh);
    global_io_free_ = ct + timing_.burst_ns;
    bank.apply_read(ct, timing_);
    ++stats_.n_rd_burst;
    stats_.pim_bytes_read += timing_.burst_bytes;
    stats_.host_bytes += timing_.burst_bytes;
    ++stats_.host_bursts;
    done = ct + timing_.burst_ns;
  }
  ++stats_.host_reqs_served;
  return done;
}

ns_t Channel::step(const Command& c, ns_t t_all) {
  switch (c.kind) {
    case CommandKind::BARRIER:
      return t_all;
    case CommandKind::MODE:
      return do_mode(c.mode, t_all);
    case CommandKind::ACT_AB:
      return std::max(t_all, do_act_ab(c.row, issue_ca(t_all)));
    case CommandKind::RDMAC_AB:
      return std::max(t_all, do_rdmac_ab(c.row, issue_ca(t_all)));
    case CommandKind::WR_AB:
      return std::max(t_all, do_wr_ab(c.row, issue_ca(t_all)));
    case CommandKind::PRE_AB:
      return std::max(t_all, do_pre_ab(issue_ca(t_all)));
    case CommandKind::ACT:
    case CommandKind::RD:
    case CommandKind::PRE:
      return std::max(t_all, do_single(c, issue_ca(t_all)));
    default:
      throw std::runtime_error("invalid command in stream");
  }
}

const ChannelStats& Channel::execute(const std::vector<Command>& stream) {
  ns_t t_all = 0.0;
  for (const Command& c : stream) t_all = step(c, t_all);
  stats_.time_ns = t_all;
  return stats_;
}

}  // namespace pimcore
