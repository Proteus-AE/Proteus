// Near-bank processing elements and the bank-group broadcast datapath.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "pimcore/timing.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

// One 16-lane FP16 MAC pipeline with an operand FIFO whose depth matches the
// bank-group fan-in. The FIFO decouples the DRAM burst cadence from MAC
// issue: a burst enters at data-return time and drains at `mac_ns` per
// entry; a full FIFO back-pressures the producing bank readout.
class ProcessingElement {
 public:
  void configure(int fifo_depth, double freq_ghz) {
    fifo_depth_ = fifo_depth;
    mac_ns_ = 1.0 / freq_ghz;
  }

  // Push one operand burst at absolute time `t`; returns the stall imposed
  // on the producer when the FIFO is full.
  ns_t push(ns_t t) {
    int drained = static_cast<int>(std::max(0.0, t - last_push_) / mac_ns_);
    occupancy_ = std::max(0, occupancy_ - drained);
    ns_t stall = 0.0;
    if (occupancy_ >= fifo_depth_) {
      stall = free_at_ > t ? free_at_ - t : mac_ns_;
      --occupancy_;
    }
    ++occupancy_;
    last_push_ = t + stall;
    free_at_ = std::max(free_at_, t + stall) + mac_ns_;
    ++macs_;
    return stall;
  }

  ns_t drain_time() const { return free_at_; }
  uint64_t macs() const { return macs_; }

 private:
  int fifo_depth_ = 4;
  ns_t mac_ns_ = 1.0;
  int occupancy_ = 0;
  ns_t last_push_ = 0.0;
  ns_t free_at_ = 0.0;
  uint64_t macs_ = 0;
};

// Time-multiplexed 32 B data bus of one bank group, with the 4:1 arbiter
// that implements the broadcast fan-out.
class BankGroupBus {
 public:
  ns_t next_free() const { return next_free_; }

  // Claim one burst slot at or after `t`; returns the granted slot start.
  ns_t claim(ns_t t, ns_t burst_ns) {
    ns_t slot = std::max(t, next_free_);
    next_free_ = slot + burst_ns;
    return slot;
  }

  // Gap available before `deadline` (for opportunistic host slots).
  ns_t gap_before(ns_t deadline) const {
    return deadline > next_free_ ? deadline - next_free_ : 0.0;
  }

 private:
  ns_t next_free_ = 0.0;
};

// Channel-level mode register controlling bank-to-PE connectivity. A switch
// is a lightweight register update that takes effect once the in-flight
// per-bank FIFOs drain (tens of ns).
class ModeController {
 public:
  ConnectivityMode mode() const { return mode_; }

  ns_t switch_mode(ConnectivityMode m, ns_t t, ns_t fifo_drain,
                   ns_t ca_cmd_ns) {
    if (m == mode_) return t;
    ns_t eff = std::max(t, fifo_drain) + ca_cmd_ns;
    mode_ = m;
    ++switches_;
    return eff;
  }

  uint64_t switches() const { return switches_; }

 private:
  ConnectivityMode mode_ = ConnectivityMode::DIRECT;
  uint64_t switches_ = 0;
};

}  // namespace pimcore
