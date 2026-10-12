// Co-execution engine: one channel shared between an all-bank PIM stream
// and controller-scheduled host traffic (the unified memory path of
// Sec. IV-B at command granularity).
#pragma once

#include <memory>
#include <vector>

#include "pimcore/channel.hpp"
#include "pimcore/controller.hpp"
#include "pimcore/traffic.hpp"

namespace pimcore {

struct CoExecConfig {
  ArbitrationPolicy policy = ArbitrationPolicy::PIM_PRIORITY;
  TrafficPattern pattern = TrafficPattern::STREAM;
  double offered_gbps = 0.0;       // 0 = closed loop
  int bursts_per_req = 2;          // 64 B host requests
  int interleave_period = 8;       // host slot every N PIM commands
  uint64_t seed = 1;
};

struct CoExecReport {
  ChannelStats stats;
  ns_t pim_only_time = 0.0;        // reference kernel time without host load
  double pim_slowdown = 0.0;       // co-executed / reference
  uint64_t host_served = 0;
  double host_bw = 0.0;            // B/s achieved
  ns_t host_latency_mean = 0.0;
  ns_t host_latency_p95 = 0.0;
};

class CoExecEngine {
 public:
  CoExecEngine(const TimingParams& timing, const Geometry& geom,
               ConnectivityMode mode, const CoExecConfig& cfg);

  // Run the PIM stream to completion while serving host traffic under the
  // configured arbitration policy; then drain the remaining host queue.
  CoExecReport run(const std::vector<Command>& pim_stream);

  // Host traffic only, no PIM stream: drive the controller for `duration`
  // ns and report the served bandwidth/latency. This is the configuration
  // the external-simulator cross-check compares against (the same request
  // stream replayed through Ramulator 2.0's LPDDR5X model).
  CoExecReport host_only(ns_t duration);

 private:
  void serve_host(ns_t now, int max_reqs);

  TimingParams timing_;
  Geometry geom_;
  CoExecConfig cfg_;
  Channel channel_;
  HostController ctrl_;
  TrafficGenerator gen_;
  ns_t host_front_ = 0.0;          // host scheduling frontier
};

}  // namespace pimcore
