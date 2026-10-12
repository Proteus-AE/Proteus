// pimcore_tracegen: command-trace and host-stream generation.
//
// Two output classes:
//
//   * PIM kernel traces (shared textual format, replayable by both the
//     Python backend and pimcore --trace):
//       pimcore_tracegen --kernel skinny-gemm --rows 64 --vectors 8
//           --mode broadcast -o gemm.trace
//
//   * host-path address streams (physical addresses under the unified
//     xPU/PIM mapping; consumed by external memory simulators, e.g. the
//     Ramulator 2.0 cross-check):
//       pimcore_tracegen --host-stream --pattern stream --reqs 200000
//           --style rw -o host.trace
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "pimcore/address.hpp"
#include "pimcore/config.hpp"
#include "pimcore/kernels.hpp"
#include "pimcore/trace.hpp"

using namespace pimcore;

namespace {

int host_stream(const TimingParams& tp, const Geometry& geom,
                       const std::string& pattern, long reqs, int stride,
                       double write_frac, uint64_t seed,
                       const std::string& style, const std::string& out) {
  AddressMapper map(geom, tp);
  FILE* f = out.empty() ? stdout : std::fopen(out.c_str(), "w");
  if (!f) {
    std::cerr << "cannot write " << out << "\n";
    return 1;
  }
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const int bursts_per_row = map.bursts_per_row();
  const long total_bursts =
      (long)geom.banks() * bursts_per_row * 64;   // 64-row working set
  long burst = 0;
  const char* rd = (style == "ldst") ? "LD" : "R";
  const char* wr = (style == "ldst") ? "ST" : "W";
  for (long i = 0; i < reqs; ++i) {
    if (pattern == "random")
      burst = (long)(uni(rng) * total_bursts);
    else if (pattern == "strided")
      burst = (burst + stride) % total_bursts;
    else
      burst = (burst + 1) % total_bursts;
    Coordinates c;
    long b = burst;
    c.col = b % bursts_per_row;             b /= bursts_per_row;
    c.bank = b % geom.banks_per_bankgroup;  b /= geom.banks_per_bankgroup;
    c.bankgroup = b % geom.bankgroups_per_die; b /= geom.bankgroups_per_die;
    c.die = b % geom.dies_per_channel;      b /= geom.dies_per_channel;
    c.row = (int)b;
    addr_t addr = map.encode(c);
    std::fprintf(f, "%s 0x%llx\n", uni(rng) < write_frac ? wr : rd,
                 (unsigned long long)addr);
  }
  if (f != stdout) std::fclose(f);
  return 0;
}

int generate(int argc, char** argv) {
  std::string cfg_file, kernel, out, pattern = "stream", style = "rw";
  std::string configs = "../../configs";
  std::string memory = "lpddr5x-8533";
  KernelParams kp;
  bool host = false;
  long reqs = 100000;
  int stride = 8;
  double write_frac = 0.0;
  uint64_t seed = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--config") cfg_file = next();
    else if (a == "--configs") configs = next();
    else if (a == "--memory") memory = next();
    else if (a == "--kernel") kernel = next();
    else if (a == "--rows") kp.rows_per_bank = std::stoi(next());
    else if (a == "--vectors") kp.n_vectors = std::stoi(next());
    else if (a == "--group") kp.group_size = std::stoi(next());
    else if (a == "--mode")
      kp.mode = next() == "broadcast" ? ConnectivityMode::BROADCAST
                                      : ConnectivityMode::DIRECT;
    else if (a == "--kv-append") kp.kv_append = true;
    else if (a == "--host-stream") host = true;
    else if (a == "--pattern") pattern = next();
    else if (a == "--reqs") reqs = std::stol(next());
    else if (a == "--stride") stride = std::stoi(next());
    else if (a == "--write-frac") write_frac = std::stod(next());
    else if (a == "--seed") seed = std::stoull(next());
    else if (a == "--style") style = next();
    else if (a == "-o" || a == "--out") out = next();
    else {
      std::cerr << "unknown option " << a << "\n";
      return 2;
    }
  }

  ConfigNode cfg = cfg_file.empty()
                       ? load_config(configs, "memory", memory, "--memory")
                       : ConfigNode::load_file(cfg_file);
  TimingParams tp = TimingParams::from_config(cfg);
  Geometry geom = Geometry::from_config(cfg);
  kp.broadcast_fanout = geom.broadcast_fanout;

  if (host)
    return host_stream(tp, geom, pattern, reqs, stride, write_frac, seed,
                       style, out);

  std::vector<Command> cmds;
  if (kernel == "gemv") cmds = gemv_kernel(kp, tp);
  else if (kernel == "skinny-gemm") cmds = skinny_gemm_kernel(kp, tp);
  else if (kernel == "attention") cmds = attention_kernel(kp, tp);
  else {
    std::cerr << "--kernel gemv|skinny-gemm|attention or --host-stream "
                 "required\n";
    return 2;
  }
  if (out.empty()) {
    std::cerr << "-o <file> required for kernel traces\n";
    return 2;
  }
  write_trace(out, cmds);
  std::fprintf(stderr, "%s: %zu commands -> %s\n", kernel.c_str(),
               cmds.size(), out.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return generate(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "pimcore_tracegen: " << e.what() << "\n";
    return 1;
  }
}
