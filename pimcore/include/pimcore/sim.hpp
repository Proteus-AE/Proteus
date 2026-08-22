// Simulator facade: configuration + kernel/trace -> report.
#pragma once

#include <string>
#include <vector>

#include "pimcore/channel.hpp"
#include "pimcore/kernels.hpp"
#include "pimcore/power.hpp"

namespace pimcore {

struct SimOptions {
  std::string config_path;       // configs/memory/<substrate>.yaml
  std::string kernel;            // gemv | skinny-gemm | attention | "" (trace)
  std::string trace_path;
  KernelParams kernel_params;
  HostStreamConfig host;
  bool external_energy = false;
  bool json = false;
  std::string json_path;
};

struct SimReport {
  ChannelStats stats;
  EnergyBreakdown energy;
  double peak_bw = 0.0;
  int device_channels = 1;       // SPMD scale factor for device-level rates
  std::string substrate;         // `name:` of the memory configuration

  std::string format() const;
  std::string to_json() const;
};

class Simulator {
 public:
  explicit Simulator(const SimOptions& opt);
  SimReport run();

  const TimingParams& timing() const { return timing_; }
  const Geometry& geometry() const { return geom_; }

 private:
  SimOptions opt_;
  ConfigNode mem_;
  TimingParams timing_;
  Geometry geom_;
  EnergyTable energy_;
};

}  // namespace pimcore
