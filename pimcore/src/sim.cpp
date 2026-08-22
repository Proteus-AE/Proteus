#include "pimcore/sim.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "pimcore/trace.hpp"

namespace pimcore {

Simulator::Simulator(const SimOptions& opt) : opt_(opt) {
  if (opt_.config_path.empty())
    throw std::runtime_error("no memory configuration given; expected a file "
                             "from configs/memory/");
  mem_ = ConfigNode::load_file(opt_.config_path);
  timing_ = TimingParams::from_config(mem_);
  geom_ = Geometry::from_config(mem_);
  energy_ = EnergyTable::from_config(mem_);
}

SimReport Simulator::run() {
  Channel ch(timing_, geom_, opt_.kernel_params.mode);
  if (opt_.host.enabled) ch.attach_host_stream(opt_.host);

  std::vector<Command> stream;
  if (!opt_.trace_path.empty()) {
    stream = read_trace(opt_.trace_path);
  } else if (opt_.kernel == "gemv") {
    stream = gemv_kernel(opt_.kernel_params, timing_);
  } else if (opt_.kernel == "skinny-gemm") {
    stream = skinny_gemm_kernel(opt_.kernel_params, timing_);
  } else if (opt_.kernel == "attention") {
    stream = attention_kernel(opt_.kernel_params, timing_);
  } else {
    throw std::runtime_error("no kernel or trace specified");
  }

  SimReport rep;
  rep.stats = ch.execute(stream);
  rep.energy = PowerModel(energy_).account(rep.stats, opt_.external_energy);
  rep.peak_bw = ch.peak_internal_bw();
  rep.device_channels = geom_.channels;
  rep.substrate = mem_.get_string("name", "");
  return rep;
}

std::string SimReport::format() const {
  std::ostringstream os;
  os << "substrate       : " << substrate << "\n";
  os << stats.format(peak_bw);
  os << energy.format(stats.pim_bytes_read + stats.pim_bytes_written) << "\n";
  if (device_channels > 1) {
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "device-level    : %.1f TB/s over %d channels (SPMD)\n",
                  stats.sustained_pim_bw() * device_channels / 1e12,
                  device_channels);
    os << buf;
  }
  return os.str();
}

std::string SimReport::to_json() const {
  std::ostringstream os;
  os << "{\"stats\":" << stats.to_json(peak_bw)
     << ",\"energy\":"
     << energy.to_json(stats.pim_bytes_read + stats.pim_bytes_written)
     << ",\"peak_bw\":" << peak_bw
     << ",\"device_channels\":" << device_channels
     << ",\"substrate\":\"" << substrate << "\"}";
  return os.str();
}

}  // namespace pimcore
