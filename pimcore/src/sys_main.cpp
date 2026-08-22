// pimcore_sys: operator-level system simulator (C++ implementation).
//
// Regenerates the overall evaluation tables (throughput and energy
// efficiency across the four models, three batch sizes, and seven systems)
// from the shared YAML configurations, independently of the Python layer.
//
// Usage:
//   pimcore_sys --configs ../../configs                 # overall CSV tables
//   pimcore_sys --configs ../../configs --table breakdown   # Sec. V-C
//   pimcore_sys --configs ../../configs --table sensitivity # Sec. V-D
//   pimcore_sys --configs ../../configs --table scalability # Sec. V-E
//   pimcore_sys --configs ../../configs --system proteus
//               --model mixtral-8x7b --batch 32 --variant rd  # one cell
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "pimcore/syscore.hpp"

using namespace pimcore::syscore;

namespace {

const std::vector<std::string> kModels = {
    "deepseek-v2-lite", "switch-26b", "mixtral-8x7b", "llama3-70b"};
const std::vector<std::string> kSystems = {
    "dgx-a100", "cxl-pnm", "cent", "neupims", "papi", "pimphony", "proteus"};
const std::vector<int> kBatches = {16, 32, 64};
constexpr int kCtxIn = 2048;
constexpr int kCtxOut = 6144;

}  // namespace

namespace {

// Sec. V-C: incremental variants on the two breakdown models.
void table_breakdown(const std::string& configs) {
  const std::vector<std::string> models = {"mixtral-8x7b", "llama3-70b"};
  const std::vector<std::string> variants = {"base", "as", "rd", "of", "ec"};
  auto sys = build_system(configs, "proteus");
  std::printf("batch,Proteus-Base,+AS,+RD,+OF,+EC\n");
  for (const auto& mn : models) {
    ModelSpec m = ModelSpec::load(configs, mn);
    for (int b : kBatches) {
      Workload w = Workload::build(m, b, kCtxIn, kCtxOut);
      double base = 0.0;
      std::printf("%d", b);
      for (const auto& v : variants) {
        SysResult r = sys->simulate_ex(w, VariantFlags::named(v), 0, 1);
        if (base == 0.0) base = r.throughput;
        std::printf(",%.3f", r.alive ? r.throughput / base : 0.0);
      }
      std::printf("\n");
    }
  }
}

// Sec. V-D: context-length and batch sweeps (5 systems, norm. CXL-PNM).
void table_sensitivity(const std::string& configs) {
  const std::vector<std::string> systems = {"dgx-a100", "cxl-pnm", "cent",
                                            "pimphony", "proteus"};
  std::vector<std::unique_ptr<SystemModel>> sys;
  for (const auto& s : systems) sys.push_back(build_system(configs, s));
  ModelSpec m = ModelSpec::load(configs, "mixtral-8x7b");

  std::printf("# context sweep (b=32, sustained ctx, norm. CXL-PNM)\n");
  std::printf("ctx,DGX-A100,CXL-PNM,CENT,PIMphony,Proteus\n");
  for (int ctx : {1024, 4096, 8192, 32768, 65536, 131072}) {
    Workload w = Workload::build(m, 32, 0, 0, ctx);
    std::vector<double> thr;
    for (const auto& s : sys) {
      SysResult r = s->simulate(w);
      thr.push_back(r.alive ? r.throughput : 0.0);
    }
    std::printf("%dK", ctx / 1024);
    for (double t : thr) std::printf(",%.4f", thr[1] ? t / thr[1] : 0.0);
    std::printf("\n");
  }

  std::printf("# batch sweep (2K/6K ctx, norm. CXL-PNM)\n");
  std::printf("batch,DGX-A100,CXL-PNM,CENT,PIMphony,Proteus\n");
  for (int b : {1, 8, 16, 32, 64, 128}) {
    Workload w = Workload::build(m, b, kCtxIn, kCtxOut);
    std::vector<double> thr;
    for (const auto& s : sys) {
      SysResult r = s->simulate(w);
      thr.push_back(r.alive ? r.throughput : 0.0);
    }
    std::printf("%d", b);
    for (double t : thr) std::printf(",%.4f", thr[1] ? t / thr[1] : 0.0);
    std::printf("\n");
  }
}

// Sec. V-F: multi-device scalability and [PP, DP] hybrids (Fig. 17/18).
void table_scalability(const std::string& configs) {
  auto sys = build_system(configs, "proteus");
  ModelSpec m = ModelSpec::load(configs, "llama3-405b");
  const int kGroup = 8, kBatchPerGroup = 16, kCtx = 32768;
  (void)kGroup;
  std::printf("# device scaling (Llama-3.1-405B, %d requests per group, "
              "%dK ctx)\n", kBatchPerGroup, kCtx / 1024);
  std::printf("devices,groups,layers_per_stage,tokens_s,normalized\n");
  double base = 0.0;
  Workload w = Workload::build(m, kBatchPerGroup, 0, 0, kCtx);
  for (int d : {8, 16, 32, 64}) {
    SysResult r = sys->simulate_ex(w, VariantFlags{}, d, 1);
    double thr = r.alive ? r.throughput : 0.0;
    if (base == 0.0) base = thr;
    std::printf("%d,%d,%d,%.0f,%.3f\n", d, r.pipeline_groups,
                r.layers_per_stage, thr, base ? thr / base : 0.0);
  }

  std::printf("# [PP,DP] on 64 devices at %dK ctx, each at the largest "
              "batch its capacity permits\n", kCtx / 1024);
  std::printf("config,batch,tokens_s,normalized\n");
  base = 0.0;
  const std::vector<std::pair<int, int>> combos = {{8, 1}, {4, 2},
                                                   {2, 4}, {1, 8}};
  for (auto [pp, dp] : combos) {
    int best_b = 0;
    double best_t = 0.0;
    for (int b = 1; b <= 2048; b = b < 16 ? b * 2 : b + 16) {
      Workload wb = Workload::build(m, b, 0, 0, kCtx);
      SysResult r = sys->simulate_ex(wb, VariantFlags{}, 64, dp);
      if (r.alive && r.throughput > best_t) {
        best_t = r.throughput;
        best_b = b;
      }
    }
    if (base == 0.0) base = best_t;
    std::printf("PP%dxDP%d,%d,%.0f,%.3f\n", pp, dp, best_b, best_t,
                base ? best_t / base : 0.0);
  }
}

// Sec. V-E: crossover robustness (Fig. 15).
void table_crossover(const std::string& configs) {
  std::printf("# throughput under a perturbed crossover threshold\n");
  std::printf("model,batch,AI_PIM,0.9*theta,theta,1.1*theta,AI_xPU\n");
  const std::vector<std::pair<std::string, std::vector<int>>> grid = {
      {"mixtral-8x7b", {32, 64}}, {"llama3-70b", {64, 128}}};
  // The sweep is expressed relative to the machine's own balance points, so
  // it follows the configuration rather than a hard-coded threshold.
  auto nominal_sys = build_system(configs, "proteus");
  const double nominal = nominal_sys->theta();
  const double ridge_pim = nominal_sys->ridge_pim();
  const double ridge_xpu = nominal_sys->ridge_xpu();
  for (const auto& g : grid) {
    ModelSpec m = ModelSpec::load(configs, g.first);
    for (int b : g.second) {
      Workload w = Workload::build(m, b, kCtxIn, kCtxOut);
      std::vector<double> thr;
      for (double th : {ridge_pim, 0.9 * nominal, nominal, 1.1 * nominal,
                        ridge_xpu}) {
        auto sys = build_system(configs, "proteus", th);
        SysResult r = sys->simulate(w);
        thr.push_back(r.alive ? r.throughput : 0.0);
      }
      std::printf("%s,%d", g.first.c_str(), b);
      for (double t : thr) std::printf(",%.4f", thr[2] ? t / thr[2] : 0.0);
      std::printf("\n");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string configs = "configs";
  std::string one_system, one_model, table, variant = "full";
  int one_batch = 0, devices = 0, dp = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--configs") configs = next();
    else if (a == "--system") one_system = next();
    else if (a == "--model") one_model = next();
    else if (a == "--batch") one_batch = std::stoi(next());
    else if (a == "--table") table = next();
    else if (a == "--variant") variant = next();
    else if (a == "--devices") devices = std::stoi(next());
    else if (a == "--dp") dp = std::stoi(next());
    else {
      std::cerr << "unknown option " << a << "\n";
      return 2;
    }
  }

  if (table == "breakdown") { table_breakdown(configs); return 0; }
  if (table == "sensitivity") { table_sensitivity(configs); return 0; }
  if (table == "scalability") { table_scalability(configs); return 0; }
  if (table == "crossover") { table_crossover(configs); return 0; }

  try {
    if (!one_system.empty()) {
      ModelSpec m = ModelSpec::load(configs, one_model);
      Workload w = Workload::build(m, one_batch, kCtxIn, kCtxOut);
      auto sys = build_system(configs, one_system);
      SysResult r = sys->simulate_ex(w, VariantFlags::named(variant),
                                     devices, dp);
      if (!r.alive) {
        std::printf("%s / %s b=%d: OOM\n", sys->name().c_str(),
                    m.name.c_str(), one_batch);
        return 1;
      }
      std::printf("%s / %s b=%d\n", sys->name().c_str(), m.name.c_str(),
                  one_batch);
      std::printf("  throughput : %.0f tokens/s\n", r.throughput);
      std::printf("  iteration  : %.3f ms\n", r.t_iter_ms);
      std::printf("  energy eff : %.2f tokens/J (%.0f W)\n",
                  r.tokens_per_joule, r.power_w);
      if (r.inflight)
        std::printf("  x* = %.2f, m = %d in-flight batches\n", r.x_split,
                    r.inflight);
      return 0;
    }

    // full overall tables
    std::vector<std::unique_ptr<SystemModel>> systems;
    for (const auto& s : kSystems) systems.push_back(build_system(configs, s));

    std::printf("# throughput (tokens/s)\nbatch");
    for (const auto& s : systems) std::printf(",%s", s->name().c_str());
    std::printf("\n");
    std::vector<std::string> energy_rows;
    for (const auto& mn : kModels) {
      ModelSpec m = ModelSpec::load(configs, mn);
      for (int b : kBatches) {
        Workload w = Workload::build(m, b, kCtxIn, kCtxOut);
        std::printf("%d", b);
        std::string erow = std::to_string(b);
        for (const auto& s : systems) {
          SysResult r = s->simulate(w);
          std::printf(",%.0f", r.alive ? r.throughput : 0.0);
          char buf[32];
          std::snprintf(buf, sizeof(buf), ",%.4f",
                        r.alive ? r.tokens_per_joule : 0.0);
          erow += buf;
        }
        std::printf("\n");
        energy_rows.push_back(erow);
      }
    }
    std::printf("# energy efficiency (tokens/J)\nbatch");
    for (const auto& s : systems) std::printf(",%s", s->name().c_str());
    std::printf("\n");
    for (const auto& row : energy_rows) std::printf("%s\n", row.c_str());
  } catch (const std::exception& e) {
    std::cerr << "pimcore_sys: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
