#include "pimcore/device.hpp"

#include <algorithm>

namespace pimcore {

namespace {
constexpr uint64_t kHostAddrSpace = 1ull << 24;   // burst-granular addresses
}

CoExecEngine::CoExecEngine(const TimingParams& timing, const Geometry& geom,
                           ConnectivityMode mode, const CoExecConfig& cfg)
    : timing_(timing), geom_(geom), cfg_(cfg),
      channel_(timing, geom, mode),
      ctrl_(timing, geom, cfg.policy),
      gen_(cfg.pattern, cfg.offered_gbps, cfg.bursts_per_req, kHostAddrSpace,
           cfg.seed) {}

void CoExecEngine::serve_host(ns_t now, int max_reqs) {
  gen_.advance(now, ctrl_);
  ChannelStats& st = const_cast<ChannelStats&>(channel_.stats());
  for (int served = 0; served < max_reqs && ctrl_.backlogged(); ++served) {
    int idx = ctrl_.select([&](const Coordinates& c) {
      return channel_.bank_row_open(c.flat_bank(geom_), c.row);
    });
    if (idx < 0) break;
    HostRequest r = ctrl_.pop(idx);
    Coordinates c = ctrl_.mapper().decode(r.addr);
    bool hit = false;
    ns_t done = channel_.host_issue(c.flat_bank(geom_), c.row, r.bursts,
                                    std::max(now, host_front_), &hit);
    host_front_ = done;
    st.host_latency.record(done - r.arrival);
    ++st.host_reqs_issued;
  }
}

CoExecReport CoExecEngine::run(const std::vector<Command>& pim_stream) {
  // Reference: identical kernel with no host interference.
  Channel ref(timing_, geom_, channel_.mode());
  ns_t t_ref = 0.0;
  for (const Command& c : pim_stream) t_ref = ref.step(c, t_ref);

  ns_t t = 0.0;
  uint64_t cmd_index = 0;
  for (const Command& c : pim_stream) {
    switch (cfg_.policy) {
      case ArbitrationPolicy::HOST_PRIORITY:
        // Backlogged host requests claim resources ahead of every all-bank
        // command (bounded to the queue per command to preserve progress).
        serve_host(t, 4);
        break;
      case ArbitrationPolicy::INTERLEAVE:
        if (cmd_index % static_cast<uint64_t>(cfg_.interleave_period) == 0)
          serve_host(t, 1);
        break;
      case ArbitrationPolicy::PIM_PRIORITY:
        // Host requests ride only in scheduling gaps: serve at row
        // boundaries (PRE_AB), where the ACT ramp leaves the buses idle.
        if (c.kind == CommandKind::PRE_AB) serve_host(t, 2);
        break;
    }
    t = channel_.step(c, t);
    ++cmd_index;
  }
  // Drain whatever the generator still owes and the queue still holds.
  serve_host(t, static_cast<int>(ctrl_.queue_len()) + 1);
  ns_t end = std::max(t, host_front_);
  channel_.finalize(end);

  CoExecReport rep;
  rep.stats = channel_.stats();
  rep.pim_only_time = t_ref;
  rep.pim_slowdown = t_ref > 0 ? t / t_ref : 0.0;
  rep.host_served = rep.stats.host_reqs_served;
  rep.host_bw = rep.stats.sustained_host_bw();
  rep.host_latency_mean = rep.stats.host_latency.mean();
  rep.host_latency_p95 = rep.stats.host_latency.percentile(0.95);
  return rep;
}

CoExecReport CoExecEngine::host_only(ns_t duration) {
  ns_t t = 0.0;
  while (std::max(t, host_front_) < duration) {
    ns_t now = std::max(t, host_front_);
    serve_host(now, 256);
    if (host_front_ > now)
      t = host_front_;
    else
      t = now + timing_.tCCD_S;   // open-loop idle gap: advance to poll
  }
  ns_t end = std::max(t, host_front_);
  channel_.finalize(end);

  CoExecReport rep;
  rep.stats = channel_.stats();
  rep.host_served = rep.stats.host_reqs_served;
  rep.host_bw = rep.stats.sustained_host_bw();
  rep.host_latency_mean = rep.stats.host_latency.mean();
  rep.host_latency_p95 = rep.stats.host_latency.percentile(0.95);
  return rep;
}

}  // namespace pimcore
