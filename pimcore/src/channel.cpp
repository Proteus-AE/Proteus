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
  for (auto& pe : pes_) pe.configure(geom_.pe_fifo_depth, geom_.pe_freq_ghz);
  if (initial_mode == ConnectivityMode::BROADCAST)
    (void)mode_ctrl_.switch_mode(initial_mode, 0.0, 0.0, 0.0);
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

ns_t Channel::do_act_ab(int row, ns_t t) {
  ns_t done = t;
  for (auto& bank : banks_) {
    DieState& die = dies_[bank.die()];
    ns_t bt = die.constrain_refresh(std::max(t, bank.next_act()),
                                    &stats_.refresh_stall_ns,
                                    &stats_.n_refresh);
    bt = die.constrain_act(bt);
    bt = bank.apply_act(bt, row, timing_);
    die.record_act(bt);
    ++stats_.n_act;
    done = std::max(done, bt);
  }
  return done;
}

ns_t Channel::do_rdmac_ab(int row, ns_t t) {
  ns_t done = t;
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
    BankGroupBus& bus = buses_[bank.bank_group()];
    ns_t prev_free = bus.next_free();
    ns_t slot = bus.claim(bt, timing_.burst_ns);
    // Deliver the burst to the connected PEs (FIFO back-pressure applies).
    ns_t stall = 0.0;
    if (mode_ctrl_.mode() == ConnectivityMode::BROADCAST) {
      int base = bank.bank_group() * geom_.banks_per_bankgroup;
      int fan = std::min(geom_.broadcast_fanout, geom_.banks_per_bankgroup);
      for (int k = 0; k < fan; ++k)
        stall = std::max(stall, pes_[base + k].push(slot));
      stats_.n_mac += fan;
    } else {
      stall = pes_[b].push(slot);
      ++stats_.n_mac;
    }
    stats_.fifo_stall_ns += stall;
    // Host stream may claim the scheduling gap ahead of this burst.
    if (slot > prev_free)
      host_fill_gap(bank.bank_group(), prev_free, slot);
    ns_t bt2 = bank.apply_read(slot + stall, timing_) + timing_.burst_ns;
    ++stats_.n_rd_burst;
    stats_.pim_bytes_read += timing_.burst_bytes;
    done = std::max(done, bt2);
  }
  return done;
}

ns_t Channel::do_wr_ab(int row, ns_t t) {
  ns_t done = t;
  for (auto& bank : banks_) {
    DieState& die = dies_[bank.die()];
    ns_t bt = std::max(t, bank.next_col());
    if (!bank.row_open(row)) {
      bt = die.constrain_refresh(std::max(bt, bank.next_act()),
                                 &stats_.refresh_stall_ns, &stats_.n_refresh);
      bt = die.constrain_act(bt);
      bt = bank.apply_act(bt, row, timing_);
      die.record_act(bt);
      ++stats_.n_act;
      bt = std::max(bt, bank.next_col());
    }
    bt = die.constrain_refresh(bt, &stats_.refresh_stall_ns,
                               &stats_.n_refresh);
    BankGroupBus& bus = buses_[bank.bank_group()];
    ns_t prev_free = bus.next_free();
    ns_t slot = bus.claim(bt, timing_.burst_ns);
    if (slot > prev_free)
      host_fill_gap(bank.bank_group(), prev_free, slot);
    bank.apply_write(slot, timing_);
    ++stats_.n_wr_burst;
    stats_.pim_bytes_written += timing_.burst_bytes;
    done = std::max(done, slot + timing_.burst_ns);
  }
  return done;
}

ns_t Channel::do_pre_ab(ns_t t) {
  for (auto& bank : banks_) bank.apply_pre(t, timing_);
  ++stats_.n_pre;
  return t + timing_.ca_cmd_ns;
}

ns_t Channel::do_mode(ConnectivityMode m, ns_t t) {
  ns_t eff = mode_ctrl_.switch_mode(m, t, max_fifo_drain(),
                                    timing_.ca_cmd_ns);
  if (eff != t) ++stats_.n_mode_switch;
  return eff;
}

ns_t Channel::do_refresh_ab(ns_t t) {
  // Explicit refresh: block every die for tRFCab from `t`.
  for (auto& bank : banks_) bank.apply_pre(t, timing_);
  ++stats_.n_refresh;
  return t + timing_.tRFCab;
}

ns_t Channel::do_single(const Command& c, ns_t t) {
  if (c.bank < 0 || c.bank >= static_cast<int>(banks_.size()))
    throw std::runtime_error("single-bank command with invalid bank id");
  Bank& bank = banks_[c.bank];
  DieState& die = dies_[bank.die()];
  switch (c.kind) {
    case CommandKind::ACT: {
      ns_t bt = die.constrain_refresh(std::max(t, bank.next_act()),
                                      &stats_.refresh_stall_ns,
                                      &stats_.n_refresh);
      bt = die.constrain_act(bt);
      bt = bank.apply_act(bt, c.row, timing_);
      die.record_act(bt);
      ++stats_.n_act;
      return bt;
    }
    case CommandKind::RD:
    case CommandKind::WR: {
      ns_t bt = die.constrain_refresh(std::max(t, bank.next_col()),
                                      &stats_.refresh_stall_ns,
                                      &stats_.n_refresh);
      ns_t slot = buses_[bank.bank_group()].claim(bt, timing_.burst_ns);
      if (c.kind == CommandKind::RD) {
        bank.apply_read(slot, timing_);
        stats_.host_bytes += timing_.burst_bytes;
        ++stats_.host_bursts;
      } else {
        bank.apply_write(slot, timing_);
        stats_.pim_bytes_written += timing_.burst_bytes;
        ++stats_.n_wr_burst;
      }
      return slot + timing_.burst_ns;
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
  if (!bank.row_open(row)) {
    bt = die.constrain_refresh(std::max(bt, bank.next_act()),
                               &stats_.refresh_stall_ns, &stats_.n_refresh);
    bt = die.constrain_act(bt);
    bt = bank.apply_act(bt, row, timing_);
    die.record_act(bt);
    ++stats_.n_act;
  }
  ns_t done = bt;
  for (int i = 0; i < bursts; ++i) {
    ns_t ct = die.constrain_refresh(std::max(done, bank.next_col()),
                                    &stats_.refresh_stall_ns,
                                    &stats_.n_refresh);
    ns_t slot = buses_[bank.bank_group()].claim(ct, timing_.burst_ns);
    bank.apply_read(slot, timing_);
    stats_.host_bytes += timing_.burst_bytes;
    ++stats_.host_bursts;
    done = slot + timing_.burst_ns;
  }
  ++stats_.host_reqs_served;
  return done;
}

ns_t Channel::step(const Command& c, ns_t t_all) {
  switch (c.kind) {
    case CommandKind::BARRIER:
      return t_all;
    case CommandKind::MODE:
      return std::max(t_all, do_mode(c.mode, issue_ca(t_all)));
    case CommandKind::ACT_AB:
      return std::max(t_all, do_act_ab(c.row, issue_ca(t_all)));
    case CommandKind::RDMAC_AB:
      return std::max(t_all, do_rdmac_ab(c.row, issue_ca(t_all)));
    case CommandKind::WR_AB:
      return std::max(t_all, do_wr_ab(c.row, issue_ca(t_all)));
    case CommandKind::PRE_AB:
      return std::max(t_all, do_pre_ab(issue_ca(t_all)));
    case CommandKind::REF_AB:
      return std::max(t_all, do_refresh_ab(issue_ca(t_all)));
    case CommandKind::ACT:
    case CommandKind::RD:
    case CommandKind::WR:
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
