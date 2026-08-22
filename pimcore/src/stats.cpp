#include "pimcore/stats.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace pimcore {

LatencyHistogram::LatencyHistogram() {
  // ns buckets: 0-100 in tens, 100-1000 in hundreds, then coarse tail.
  for (int i = 10; i <= 100; i += 10) bounds_.push_back(i);
  for (int i = 200; i <= 1000; i += 100) bounds_.push_back(i);
  bounds_.push_back(2000);
  bounds_.push_back(5000);
  bounds_.push_back(10000);
  counts_.assign(bounds_.size() + 1, 0);
}

void LatencyHistogram::record(ns_t latency) {
  ++samples_;
  sum_ += latency;
  max_ = std::max(max_, latency);
  size_t i = 0;
  while (i < bounds_.size() && latency > bounds_[i]) ++i;
  ++counts_[i];
}

ns_t LatencyHistogram::percentile(double p) const {
  if (!samples_) return 0.0;
  uint64_t target = static_cast<uint64_t>(p * samples_);
  uint64_t acc = 0;
  for (size_t i = 0; i < counts_.size(); ++i) {
    acc += counts_[i];
    if (acc >= target) return i < bounds_.size() ? bounds_[i] : max_;
  }
  return max_;
}

std::string LatencyHistogram::format() const {
  char buf[160];
  std::snprintf(buf, sizeof(buf),
                "n=%llu mean=%.1fns p50=%.0fns p95=%.0fns p99=%.0fns "
                "max=%.0fns",
                static_cast<unsigned long long>(samples_), mean(),
                percentile(0.50), percentile(0.95), percentile(0.99), max_);
  return buf;
}

std::string ChannelStats::format(double peak_bw) const {
  std::ostringstream os;
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "time            : %.1f us\n"
                "commands        : %llu (ACT %llu, RD %llu, WR %llu, "
                "MODE %llu, REF %llu)\n",
                time_ns / 1e3, static_cast<unsigned long long>(n_cmds),
                static_cast<unsigned long long>(n_act),
                static_cast<unsigned long long>(n_rd_burst),
                static_cast<unsigned long long>(n_wr_burst),
                static_cast<unsigned long long>(n_mode_switch),
                static_cast<unsigned long long>(n_refresh));
  os << buf;
  std::snprintf(buf, sizeof(buf),
                "PIM sustained   : %.1f GB/s (eff %.3f) | row hit %.3f | "
                "MACs %llu\n",
                sustained_pim_bw() / 1e9,
                peak_bw > 0 ? sustained_pim_bw() / peak_bw : 0.0,
                row_hit_rate(), static_cast<unsigned long long>(n_mac));
  os << buf;
  std::snprintf(buf, sizeof(buf),
                "stalls          : FIFO %.1f us | refresh %.1f us\n",
                fifo_stall_ns / 1e3, refresh_stall_ns / 1e3);
  os << buf;
  if (host_reqs_issued) {
    std::snprintf(buf, sizeof(buf),
                  "host stream     : served %llu/%llu reqs, %.1f GB/s\n",
                  static_cast<unsigned long long>(host_reqs_served),
                  static_cast<unsigned long long>(host_reqs_issued),
                  sustained_host_bw() / 1e9);
    os << buf;
    os << "host latency    : " << host_latency.format() << "\n";
  }
  return os.str();
}

std::string ChannelStats::to_json(double peak_bw) const {
  std::ostringstream os;
  os.precision(6);
  os << "{"
     << "\"time_ns\":" << time_ns
     << ",\"n_cmds\":" << n_cmds
     << ",\"n_act\":" << n_act
     << ",\"n_rd_burst\":" << n_rd_burst
     << ",\"n_wr_burst\":" << n_wr_burst
     << ",\"n_mac\":" << n_mac
     << ",\"n_mode_switch\":" << n_mode_switch
     << ",\"n_refresh\":" << n_refresh
     << ",\"row_hit_rate\":" << row_hit_rate()
     << ",\"fifo_stall_ns\":" << fifo_stall_ns
     << ",\"refresh_stall_ns\":" << refresh_stall_ns
     << ",\"pim_bytes\":" << (pim_bytes_read + pim_bytes_written)
     << ",\"sustained_pim_bw\":" << sustained_pim_bw()
     << ",\"efficiency\":" << (peak_bw > 0 ? sustained_pim_bw() / peak_bw : 0)
     << ",\"host_reqs_served\":" << host_reqs_served
     << ",\"host_bytes\":" << host_bytes
     << ",\"sustained_host_bw\":" << sustained_host_bw()
     << ",\"host_latency_mean_ns\":" << host_latency.mean()
     << ",\"host_latency_p95_ns\":" << host_latency.percentile(0.95)
     << "}";
  return os.str();
}

std::string EnergyBreakdown::format(uint64_t bytes) const {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "energy          : %.1f uJ (%.2f pJ/bit)\n"
                "  ACT/PRE %.1f | RD %.1f | WR %.1f | IO %.1f | BC %.1f | "
                "MAC %.1f | REF %.1f uJ",
                total_nj() / 1e3, pj_per_bit(bytes), act_pre_nj / 1e3,
                rd_array_nj / 1e3, wr_array_nj / 1e3, rd_io_nj / 1e3,
                bcast_nj / 1e3, mac_nj / 1e3, refresh_nj / 1e3);
  return buf;
}

std::string EnergyBreakdown::to_json(uint64_t bytes) const {
  std::ostringstream os;
  os.precision(6);
  os << "{"
     << "\"total_nj\":" << total_nj()
     << ",\"pj_per_bit\":" << pj_per_bit(bytes)
     << ",\"act_pre_nj\":" << act_pre_nj
     << ",\"rd_array_nj\":" << rd_array_nj
     << ",\"wr_array_nj\":" << wr_array_nj
     << ",\"rd_io_nj\":" << rd_io_nj
     << ",\"bcast_nj\":" << bcast_nj
     << ",\"mac_nj\":" << mac_nj
     << ",\"refresh_nj\":" << refresh_nj
     << "}";
  return os.str();
}

}  // namespace pimcore
