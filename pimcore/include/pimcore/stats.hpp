// Simulation statistics: counters, sustained rates, and latency histograms.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pimcore/types.hpp"

namespace pimcore {

// Fixed-boundary latency histogram (ns buckets) for host requests.
class LatencyHistogram {
 public:
  LatencyHistogram();
  void record(ns_t latency);
  uint64_t samples() const { return samples_; }
  ns_t mean() const { return samples_ ? sum_ / samples_ : 0.0; }
  ns_t max() const { return max_; }
  ns_t percentile(double p) const;      // approximate, bucket upper bound
  std::string format() const;

 private:
  std::vector<ns_t> bounds_;
  std::vector<uint64_t> counts_;
  uint64_t samples_ = 0;
  ns_t sum_ = 0.0;
  ns_t max_ = 0.0;
};

struct ChannelStats {
  ns_t time_ns = 0.0;

  // command counters
  uint64_t n_cmds = 0;
  uint64_t n_act = 0;
  uint64_t n_pre = 0;
  uint64_t n_rd_burst = 0;
  uint64_t n_wr_burst = 0;
  uint64_t n_mac = 0;
  uint64_t n_broadcast = 0;   // bursts distributed over a BG-local bus
  uint64_t n_mode_switch = 0;
  uint64_t n_refresh = 0;

  // locality / stalls
  uint64_t row_hits = 0;
  uint64_t row_misses = 0;
  ns_t fifo_stall_ns = 0.0;
  ns_t refresh_stall_ns = 0.0;

  // host (xPU) co-execution
  uint64_t host_reqs_issued = 0;
  uint64_t host_reqs_served = 0;
  uint64_t host_bursts = 0;
  LatencyHistogram host_latency;

  // traffic
  uint64_t pim_bytes_read = 0;
  uint64_t pim_bytes_written = 0;
  uint64_t host_bytes = 0;

  double sustained_pim_bw() const {     // B/s
    return time_ns > 0.0
        ? (pim_bytes_read + pim_bytes_written) / (time_ns * 1e-9) : 0.0;
  }
  double sustained_host_bw() const {
    return time_ns > 0.0 ? host_bytes / (time_ns * 1e-9) : 0.0;
  }
  double row_hit_rate() const {
    uint64_t total = row_hits + row_misses;
    return total ? static_cast<double>(row_hits) / total : 0.0;
  }

  std::string format(double peak_bw) const;
  std::string to_json(double peak_bw) const;
};

struct EnergyBreakdown {
  double act_pre_nj = 0.0;
  double rd_array_nj = 0.0;
  double wr_array_nj = 0.0;
  double rd_io_nj = 0.0;
  double bcast_nj = 0.0;
  double mac_nj = 0.0;
  double mode_nj = 0.0;
  double refresh_nj = 0.0;

  double total_nj() const {
    return act_pre_nj + rd_array_nj + wr_array_nj + rd_io_nj + bcast_nj +
           mac_nj + mode_nj + refresh_nj;
  }
  double pj_per_bit(uint64_t bytes) const {
    return bytes ? total_nj() * 1e3 / (bytes * 8.0) : 0.0;
  }
  std::string format(uint64_t bytes) const;
  std::string to_json(uint64_t bytes) const;
};

}  // namespace pimcore
