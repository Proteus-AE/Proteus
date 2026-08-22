// pimcore_coexec: co-execution study driver.
//
// Sweeps host arbitration policies and offered loads against a PIM kernel
// and emits one CSV row per configuration:
//   mode,policy,offered_gbps,pim_slowdown,host_gbps,host_lat_mean,host_lat_p95
//
// Example:
//   pimcore_coexec --memory lpddr5x-8533 --rows 48 --vectors 8
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "pimcore/config.hpp"
#include "pimcore/device.hpp"
#include "pimcore/kernels.hpp"

using namespace pimcore;

namespace {

struct Options {
  std::string cfg_file;
  std::string configs = "../../configs";
  std::string memory = "lpddr5x-8533";
  int rows = 48;
  int vectors = 8;
  std::string pattern = "stream";
  bool host_only = false;
  double duration_us = 100.0;
  double offered = 0.0;               // 0 = closed loop
  int bursts_per_req = 2;
};

int run(const Options& o) {
  ConfigNode cfg =
      o.cfg_file.empty()
          ? load_config(o.configs, "memory", o.memory, "--memory")
          : ConfigNode::load_file(o.cfg_file);
  TimingParams timing = TimingParams::from_config(cfg);
  Geometry geom = Geometry::from_config(cfg);

  if (o.host_only) {
    // Host traffic only (no PIM stream): the configuration compared
    // against external memory simulators (experiments/run_ramulator_xcheck).
    CoExecConfig cc;
    cc.pattern = pattern_from_string(o.pattern);
    cc.offered_gbps = o.offered;
    cc.bursts_per_req = o.bursts_per_req;
    CoExecEngine eng(timing, geom, ConnectivityMode::DIRECT, cc);
    CoExecReport rep = eng.host_only(o.duration_us * 1e3);
    std::printf("pattern,duration_us,host_gbps,host_lat_mean_ns,"
                "host_lat_p95_ns,host_served\n");
    std::printf("%s,%.1f,%.3f,%.1f,%.1f,%llu\n", o.pattern.c_str(),
                o.duration_us, rep.host_bw / 1e9, rep.host_latency_mean,
                rep.host_latency_p95,
                static_cast<unsigned long long>(rep.host_served));
    return 0;
  }

  std::printf("mode,policy,offered_gbps,pim_slowdown,host_gbps,"
              "host_lat_mean_ns,host_lat_p95_ns,host_served\n");

  const std::vector<ConnectivityMode> modes = {ConnectivityMode::DIRECT,
                                               ConnectivityMode::BROADCAST};
  const std::vector<ArbitrationPolicy> policies = {
      ArbitrationPolicy::PIM_PRIORITY, ArbitrationPolicy::INTERLEAVE,
      ArbitrationPolicy::HOST_PRIORITY};
  const std::vector<double> loads = {2.0, 4.0, 8.0};   // GB/s per channel

  for (ConnectivityMode m : modes) {
    KernelParams kp;
    kp.rows_per_bank = o.rows;
    kp.n_vectors = o.vectors;
    kp.mode = m;
    kp.broadcast_fanout = geom.broadcast_fanout;
    auto stream = skinny_gemm_kernel(kp, timing);
    for (ArbitrationPolicy p : policies) {
      for (double load : loads) {
        CoExecConfig cc;
        cc.policy = p;
        cc.pattern = pattern_from_string(o.pattern);
        cc.offered_gbps = load;
        CoExecEngine eng(timing, geom, m, cc);
        CoExecReport rep = eng.run(stream);
        const char* pol =
            p == ArbitrationPolicy::PIM_PRIORITY ? "pim-priority" :
            p == ArbitrationPolicy::INTERLEAVE ? "interleave" :
            "host-priority";
        std::printf("%s,%s,%.1f,%.4f,%.3f,%.1f,%.1f,%llu\n",
                    m == ConnectivityMode::DIRECT ? "direct" : "broadcast",
                    pol, load, rep.pim_slowdown, rep.host_bw / 1e9,
                    rep.host_latency_mean, rep.host_latency_p95,
                    static_cast<unsigned long long>(rep.host_served));
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--config") o.cfg_file = next();
    else if (a == "--configs") o.configs = next();
    else if (a == "--memory") o.memory = next();
    else if (a == "--rows") o.rows = std::stoi(next());
    else if (a == "--vectors") o.vectors = std::stoi(next());
    else if (a == "--pattern") o.pattern = next();
    else if (a == "--host-only") o.host_only = true;
    else if (a == "--duration-us") o.duration_us = std::stod(next());
    else if (a == "--offered") o.offered = std::stod(next());
    else if (a == "--bursts-per-req") o.bursts_per_req = std::stoi(next());
    else {
      std::cerr << "unknown option " << a << "\n";
      return 2;
    }
  }

  try {
    return run(o);
  } catch (const std::exception& e) {
    std::cerr << "pimcore_coexec: " << e.what() << "\n";
    return 1;
  }
}
