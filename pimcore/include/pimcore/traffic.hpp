// Host (xPU) traffic generators for co-execution studies.
//
// Three access patterns cover the xPU-side streams of a decode iteration:
//   stream   sequential bursts (weight/activation streaming reads)
//   strided  fixed-stride bursts (tensor-slice gathers)
//   random   uniform random bursts (page-table / KV-index walks)
// Arrivals follow either a fixed rate (offered GB/s) or a closed loop
// (next request after the previous completes).
#pragma once

#include <cstdint>
#include <random>
#include <string>

#include "pimcore/controller.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

enum class TrafficPattern : uint8_t { STREAM, STRIDED, RANDOM };

TrafficPattern pattern_from_string(const std::string& s);

class TrafficGenerator {
 public:
  TrafficGenerator(TrafficPattern pattern, double offered_gbps,
                   int bursts_per_req, uint64_t addr_space, uint64_t seed);

  // Emit every request with arrival <= now into the controller.
  void advance(ns_t now, HostController& ctrl);

  uint64_t generated() const { return generated_; }

 private:
  addr_t next_addr();

  TrafficPattern pattern_;
  double offered_gbps_;
  int bursts_per_req_;
  uint64_t addr_space_;
  ns_t next_arrival_ = 0.0;
  addr_t cursor_ = 0;
  uint64_t stride_ = 64;
  std::mt19937_64 rng_;
  uint64_t generated_ = 0;
};

}  // namespace pimcore
