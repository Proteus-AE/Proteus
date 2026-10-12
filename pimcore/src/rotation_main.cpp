// pimcore_rotation: command-level study of concurrent PIM streaming and host
// (xPU) service on one LPDDR5X-PIM channel.
//
//   pimcore_rotation [--memory lpddr5x-8533] [--configs DIR]
//                    [--schedule rotate|lockstep|direct] [--rows N]
//                    [--no-pim] [--no-host] [--no-refresh]
//                    [--offered GBPS] [--window N] [--granule N]
//                    [--in-ratio R] [--out-ratio R] [--duration NS]
//                    [--bg-offset NS] [--trace FILE --trace-from NS
//                     --trace-until NS] [--header]
//
// Prints one CSV row per run (see --header).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "pimcore/config.hpp"
#include "pimcore/rotation.hpp"
#include "pimcore/timing.hpp"

using namespace pimcore;

int main(int argc, char** argv) {
  std::string configs = "../../configs", memory = "lpddr5x-8533", trace_file;
  std::string label = "run";
  RotationConfig c;
  bool header = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
      return argv[++i];
    };
    if (a == "--memory") memory = nxt();
    else if (a == "--configs") configs = nxt();
    else if (a == "--label") label = nxt();
    else if (a == "--schedule") {
      std::string s = nxt();
      c.schedule = s == "direct" ? Schedule::DIRECT
                   : s == "lockstep" ? Schedule::LOCKSTEP : Schedule::ROTATE;
    }
    else if (a == "--rows") c.rows_per_bank = std::atoi(nxt().c_str());
    else if (a == "--no-pim") c.pim = false;
    else if (a == "--no-host") c.host = false;
    else if (a == "--no-refresh") c.refresh = false;
    else if (a == "--offered") { c.saturate = false; c.offered_gbps = std::atof(nxt().c_str()); }
    else if (a == "--window") c.window_requests = std::atoi(nxt().c_str());
    else if (a == "--granule") c.granule_bursts = std::atoi(nxt().c_str());
    else if (a == "--in-ratio") c.pim_in_ratio = std::atof(nxt().c_str());
    else if (a == "--out-ratio") c.pim_out_ratio = std::atof(nxt().c_str());
    else if (a == "--onchip-in") c.pim_in_external = false;
    else if (a == "--io-batch") c.pim_io_batch = std::atoi(nxt().c_str());
    else if (a == "--prefetch") c.pim_prefetch_ns = std::atof(nxt().c_str());
    else if (a == "--out-slack") c.pim_out_slack_ns = std::atof(nxt().c_str());
    else if (a == "--duration") c.duration_ns = std::atof(nxt().c_str());
    else if (a == "--bg-offset") c.bg_offset_ns = std::atof(nxt().c_str());
    else if (a == "--bg-offsets" || a == "--bank-phase") {
      std::vector<double>& v = (a == "--bg-offsets") ? c.bg_offsets : c.bank_phase;
      std::string s = nxt();
      size_t pos = 0;
      while (pos < s.size()) {
        size_t q = s.find(',', pos);
        if (q == std::string::npos) q = s.size();
        v.push_back(std::atof(s.substr(pos, q - pos).c_str()));
        pos = q + 1;
      }
    }
    else if (a == "--seed") c.seed = static_cast<unsigned>(std::atoi(nxt().c_str()));
    else if (a == "--trace") { c.trace = true; trace_file = nxt(); }
    else if (a == "--trace-from") c.trace_from_ns = std::atof(nxt().c_str());
    else if (a == "--trace-until") c.trace_until_ns = std::atof(nxt().c_str());
    else if (a == "--header") header = true;
    else { std::cerr << "unknown option " << a << "\n"; return 2; }
  }
  ConfigNode cfg = load_config(configs, "memory", memory, "--memory");
  TimingParams t = TimingParams::from_config(cfg);
  RotationReport r = run_rotation(t, c);
  if (header)
    std::printf("label,schedule,rows,pim,host,offered,in_ratio,out_ratio,"
                "span_ns,pim_time_ns,pim_ideal_ns,pim_gbps,pe_util,"
                "host_gbps,host_frac,lat_mean,lat_p50,lat_p99,lat_max,"
                "host_reads,host_acts,bursts_per_act,pim_acts,act_rate_us,"
                "io_util,ca_util,fifo_max,acc_max,viol_act,viol_bank,"
                "viol_conflict,viol_bus,viol_io,viol_ca,viol_refresh,pim_io_bytes,"
                "pim_io_missed,pim_io_late_max\n");
  std::printf("%s,%s,%d,%d,%d,%.3f,%.4f,%.4f,%.1f,%.1f,%.1f,%.2f,%.4f,"
              "%.3f,%.4f,%.1f,%.1f,%.1f,%.1f,%ld,%ld,%.2f,%ld,%.1f,"
              "%.4f,%.4f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%d,%.1f\n",
              label.c_str(), schedule_name(c.schedule), c.rows_per_bank,
              c.pim ? 1 : 0, c.host ? 1 : 0, c.saturate ? -1.0 : c.offered_gbps,
              c.pim_in_ratio, c.pim_out_ratio, r.span_ns, r.pim_time_ns,
              r.pim_ideal_ns, r.pim_gbps, r.pe_util, r.host_gbps, r.host_frac,
              r.host_lat_mean, r.host_lat_p50, r.host_lat_p99, r.host_lat_max,
              r.host_reads, r.host_acts, r.bursts_per_host_act, r.pim_acts,
              r.act_rate_per_us, r.io_util, r.ca_util, r.fifo_max, r.acc_max,
              r.viol_act, r.viol_bank, r.viol_conflict, r.viol_bus, r.viol_io,
              r.viol_ca, r.viol_refresh, r.pim_io_bytes, r.pim_io_missed,
              r.pim_io_late_max);
  if (c.trace && !trace_file.empty()) {
    FILE* f = std::fopen(trace_file.c_str(), "w");
    std::fprintf(f, "t_ns,dur_ns,unit,kind,value\n");
    for (const TraceEvent& e : r.trace)
      std::fprintf(f, "%.3f,%.3f,%s,%s,%d\n", e.t, e.dur, e.unit.c_str(),
                   e.kind.c_str(), e.value);
    std::fclose(f);
  }
  return 0;
}
