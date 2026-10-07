// Host-path memory controller sharing the channel with the PIM stream.
//
// Models the xPU side of the unified memory path (Sec. IV-B): a bounded
// request queue with FR-FCFS scheduling (row-hit-first, then oldest), read/
// write turnaround management, and one of three arbitration policies
// against the all-bank PIM stream:
//
//   pim-priority   host commands issue only into PIM scheduling gaps
//                  (the Proteus default: PIM row service is never delayed);
//   host-priority  a backlogged host request claims the next bus slot ahead
//                  of the PIM stream (bounds host tail latency, stretches
//                  the PIM kernel);
//   interleave     round-robin between backlogged host and PIM slots.
//
// The controller is driven by the traffic generators in traffic.hpp and
// reports per-request latency into the channel's histogram.
#pragma once

#include <deque>
#include <vector>

#include "pimcore/address.hpp"
#include "pimcore/stats.hpp"
#include "pimcore/timing.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

struct HostRequest {
  addr_t addr = 0;
  bool is_write = false;
  ns_t arrival = 0.0;
  int bursts = 2;                // 64 B request by default
};

// Outcome of scheduling one host request onto concrete bank/bus resources.
struct HostIssue {
  bool ok = false;
  ns_t start = 0.0;
  ns_t finish = 0.0;
  bool row_hit = false;
};

class HostController {
 public:
  HostController(const TimingParams& timing, const Geometry& geom,
                 ArbitrationPolicy policy, size_t queue_depth = 32);

  void push(const HostRequest& req);
  bool backlogged() const { return !queue_.empty(); }
  size_t queue_len() const { return queue_.size(); }
  ArbitrationPolicy policy() const { return policy_; }

  // Pick the next request under FR-FCFS given the current open rows; the
  // caller (channel/device) supplies a row-lookup callback.
  template <typename RowOpenFn>
  int select(RowOpenFn row_open) const {
    // row-hit-first
    for (size_t i = 0; i < queue_.size(); ++i) {
      Coordinates c = mapper_.decode(queue_[i].addr);
      if (row_open(c)) return static_cast<int>(i);
    }
    return queue_.empty() ? -1 : 0;    // oldest otherwise
  }

  HostRequest pop(int idx);
  const AddressMapper& mapper() const { return mapper_; }

  // aggregate accounting
  uint64_t issued = 0;
  uint64_t dropped = 0;                 // queue-full arrivals

 private:
  TimingParams timing_;
  Geometry geom_;
  ArbitrationPolicy policy_;
  size_t depth_;
  AddressMapper mapper_;
  std::deque<HostRequest> queue_;
};

}  // namespace pimcore
