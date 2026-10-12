// PimCore command-line driver.
//
// The substrate is one of the repository memory configurations,
// configs/memory/<name>.yaml -- the same files the Python layer reads.
//
// Examples:
//   pimcore --kernel gemv --rows 96
//   pimcore --kernel skinny-gemm --rows 64 --vectors 8 --mode broadcast
//   pimcore --trace gemm.trace
//   pimcore --memory hbm-pim --kernel attention --group 8
//   pimcore --kernel skinny-gemm --vectors 8 --mode broadcast
//           --host-policy pim-priority --host-gbps 8 --json report.json
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "pimcore/sim.hpp"

namespace {

void usage() {
  std::cout <<
      "pimcore -- command-level near-bank PIM memory simulator\n\n"
      "  --configs <dir>       repository configs/ directory\n"
      "  --memory <name>       substrate from <configs>/memory/<name>.yaml\n"
      "  --config <file>       substrate configuration by path\n"
      "  --kernel <name>       gemv | skinny-gemm | attention\n"
      "  --trace <file>        replay a command trace instead of a kernel\n"
      "  --rows <n>            striped-operand rows per bank (default 64)\n"
      "  --vectors <n>         concurrent input vectors (skinny-gemm)\n"
      "  --group <g>           query-group size (attention)\n"
      "  --mode <m>            direct | broadcast (default direct)\n"
      "  --kv-append           emit the in-place KV write tail (attention)\n"
      "  --host-gbps <x>       attach a host read stream (0 = greedy)\n"
      "  --host-policy <p>     pim-priority | host-priority | interleave\n"
      "  --external-energy     charge I/O+PHY on read bursts\n"
      "  --json [file]         machine-readable report (stdout or file)\n";
}

}  // namespace

int main(int argc, char** argv) {
  using namespace pimcore;
  SimOptions opt;
  opt.kernel_params.rows_per_bank = 64;
  std::string configs = "../../configs";
  std::string memory = "lpddr5x-8533";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--config") opt.config_path = next();
    else if (a == "--configs") configs = next();
    else if (a == "--memory") memory = next();
    else if (a == "--kernel") opt.kernel = next();
    else if (a == "--trace") opt.trace_path = next();
    else if (a == "--rows") opt.kernel_params.rows_per_bank = std::stoi(next());
    else if (a == "--vectors") opt.kernel_params.n_vectors = std::stoi(next());
    else if (a == "--group") opt.kernel_params.group_size = std::stoi(next());
    else if (a == "--mode")
      opt.kernel_params.mode = (next() == "broadcast")
                                   ? ConnectivityMode::BROADCAST
                                   : ConnectivityMode::DIRECT;
    else if (a == "--kv-append") opt.kernel_params.kv_append = true;
    else if (a == "--host-gbps") {
      opt.host.enabled = true;
      opt.host.demand_gbps = std::stod(next());
    } else if (a == "--host-policy") {
      opt.host.enabled = true;
      opt.host.policy = arbitration_from_string(next());
    } else if (a == "--external-energy") opt.external_energy = true;
    else if (a == "--json") {
      opt.json = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') opt.json_path = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage();
      return 0;
    } else {
      std::cerr << "unknown option: " << a << "\n";
      usage();
      return 2;
    }
  }

  try {
    if (opt.config_path.empty())
      opt.config_path = config_path(configs, "memory", memory, "--memory");
    Simulator sim(opt);
    SimReport rep = sim.run();
    if (opt.json) {
      if (opt.json_path.empty()) {
        std::cout << rep.to_json() << "\n";
      } else {
        std::ofstream f(opt.json_path);
        f << rep.to_json() << "\n";
        std::cout << "report -> " << opt.json_path << "\n";
      }
    } else {
      std::cout << rep.format();
    }
  } catch (const std::exception& e) {
    std::cerr << "pimcore: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
