// Command-based energy accounting from executed-stream statistics: ACT/PRE
// pairs, array bursts, the I/O + PHY component of bursts that leave the die,
// bank-group broadcast distribution, MAC issues, mode-register updates and
// refresh events.
#pragma once

#include "pimcore/stats.hpp"
#include "pimcore/timing.hpp"

namespace pimcore {

class PowerModel {
 public:
  explicit PowerModel(const EnergyTable& table) : e_(table) {}

  // `external` charges the I/O + PHY component on every read burst
  // (host-path reads leaving the die); the near-bank path omits it.
  EnergyBreakdown account(const ChannelStats& s, bool external = false) const {
    EnergyBreakdown b;
    b.act_pre_nj = s.n_act * e_.act_pre_pj * 1e-3;
    b.rd_array_nj = s.n_rd_burst * e_.rd_burst_array_pj * 1e-3;
    b.wr_array_nj = s.n_wr_burst * e_.wr_burst_array_pj * 1e-3;
    b.rd_io_nj = external ? s.n_rd_burst * e_.rd_burst_io_pj * 1e-3 : 0.0;
    b.bcast_nj = s.n_broadcast * e_.bg_broadcast_pj * 1e-3;
    b.mac_nj = s.n_mac * e_.mac_op_pj * 1e-3;
    b.mode_nj = s.n_mode_switch * e_.mode_switch_pj * 1e-3;
    b.refresh_nj = s.n_refresh * e_.refresh_ab_pj * 1e-3;
    return b;
  }

 private:
  EnergyTable e_;
};

}  // namespace pimcore
