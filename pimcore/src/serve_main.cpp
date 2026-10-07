// pimcore_serve: closed-loop continuous-batching serving driver.
//
// Emits one CSV row per decode iteration (same schema as the Python
// serving layer) plus a steady-state summary on stderr.
//
// Example:
//   pimcore_serve --configs ../configs --model mixtral-8x7b \
//       --batch 32 --prompt-mean 2048 --out-mean 256 --iters 600
#include <cstdio>
#include <iostream>
#include <string>

#include "pimcore/serving.hpp"

using namespace pimcore;

int main(int argc, char** argv) {
  std::string configs = "../../configs";
  std::string model_name = "mixtral-8x7b";
  std::string trace_path;
  std::string csv_path;
  ServingConfig scfg;
  int iters = 600;
  int warmup = 50;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--configs") configs = next();
    else if (a == "--model") model_name = next();
    else if (a == "--batch") scfg.max_batch = std::stoi(next());
    else if (a == "--prompt-mean") scfg.prompt_mean = std::stod(next());
    else if (a == "--out-mean") scfg.out_mean = std::stod(next());
    else if (a == "--seed") scfg.seed = std::stoull(next());
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--warmup") warmup = std::stoi(next());
    else if (a == "--trace") trace_path = next();
    else if (a == "--csv") csv_path = next();
    else {
      std::cerr << "unknown option " << a << "\n";
      return 2;
    }
  }

  syscore::ModelSpec model = syscore::ModelSpec::load(configs, model_name);
  auto sys = syscore::build_system(configs, "proteus");
  ConfigNode sys_cfg =
      ConfigNode::load_file(configs + "/systems/proteus.yaml");

  std::vector<std::pair<int, int>> source;
  if (!trace_path.empty())
    source = ServingSim::read_request_trace(trace_path);

  ServingSim sim(*sys, model, sys_cfg, scfg, source);
  auto recs = sim.run(iters);

  FILE* out = stdout;
  if (!csv_path.empty()) {
    out = std::fopen(csv_path.c_str(), "w");
    if (!out) {
      std::cerr << "cannot write " << csv_path << "\n";
      return 1;
    }
  }
  std::fprintf(out, "iter,batch,mean_ctx,t_iter_ms,tokens_s,"
                    "tokens_per_expert,x_split,experts_xpu,experts_pim,"
                    "placement_switches,completed\n");
  for (const auto& r : recs)
    std::fprintf(out, "%d,%d,%.0f,%.3f,%.0f,%.2f,%.3f,%d,%d,%d,%d\n",
                 r.it, r.batch, r.mean_ctx, r.t_iter_ms, r.throughput,
                 r.tokens_per_expert, r.x_split, r.experts_xpu,
                 r.experts_pim, r.switches, r.completed);
  if (out != stdout) std::fclose(out);

  if ((int)recs.size() > warmup) {
    double thr = 0.0, xmin = 1e9, xmax = -1e9;
    int sw = 0, completed = 0;
    for (size_t i = warmup; i < recs.size(); ++i) {
      thr += recs[i].throughput;
      xmin = std::min(xmin, recs[i].x_split);
      xmax = std::max(xmax, recs[i].x_split);
      sw += recs[i].switches;
    }
    for (const auto& r : recs) completed += r.completed;
    thr /= (recs.size() - warmup);
    std::fprintf(stderr,
                 "steady state over %zu iterations: %.0f tokens/s, "
                 "x* %.2f..%.2f, %d placement switches, %d completed\n",
                 recs.size() - warmup, thr, xmin, xmax, sw, completed);
  }
  return 0;
}
