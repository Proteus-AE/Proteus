#include "pimcore/traffic.hpp"

#include <stdexcept>

namespace pimcore {

TrafficPattern pattern_from_string(const std::string& s) {
  if (s == "stream") return TrafficPattern::STREAM;
  if (s == "strided") return TrafficPattern::STRIDED;
  if (s == "random") return TrafficPattern::RANDOM;
  throw std::runtime_error("unknown traffic pattern: " + s);
}

TrafficGenerator::TrafficGenerator(TrafficPattern pattern, double offered_gbps,
                                   int bursts_per_req, uint64_t addr_space,
                                   uint64_t seed)
    : pattern_(pattern), offered_gbps_(offered_gbps),
      bursts_per_req_(bursts_per_req), addr_space_(addr_space), rng_(seed) {}

addr_t TrafficGenerator::next_addr() {
  switch (pattern_) {
    case TrafficPattern::STREAM:
      cursor_ = (cursor_ + static_cast<uint64_t>(bursts_per_req_)) %
                addr_space_;
      return cursor_;
    case TrafficPattern::STRIDED:
      cursor_ = (cursor_ + stride_) % addr_space_;
      return cursor_;
    default:
      return rng_() % addr_space_;
  }
}

void TrafficGenerator::advance(ns_t now, HostController& ctrl) {
  double bytes_per_req = bursts_per_req_ * 32.0;
  ns_t inter = offered_gbps_ > 0.0
                   ? bytes_per_req / (offered_gbps_ * 1e9) * 1e9
                   : 0.0;
  if (offered_gbps_ <= 0.0) {
    // closed loop: keep exactly one request outstanding
    if (!ctrl.backlogged()) {
      HostRequest r;
      r.addr = next_addr();
      r.arrival = now;
      r.bursts = bursts_per_req_;
      ctrl.push(r);
      ++generated_;
    }
    return;
  }
  while (next_arrival_ <= now) {
    HostRequest r;
    r.addr = next_addr();
    r.arrival = next_arrival_;
    r.bursts = bursts_per_req_;
    ctrl.push(r);
    ++generated_;
    next_arrival_ += inter;
  }
}

}  // namespace pimcore
