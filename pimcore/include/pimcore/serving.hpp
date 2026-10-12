// Closed-loop continuous-batching serving simulation on the system layer.
//
// C++ counterpart of proteus_sim/serving.py: a request pool decodes one
// token per iteration; completed requests leave and the pool refills
// immediately, so batch composition, aggregate context, and per-expert
// routing drift continuously. Each iteration re-derives the co-execution
// split and the per-expert crossover placement, which is what the
// run_serving experiment measures.
#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "pimcore/config.hpp"
#include "pimcore/syscore.hpp"

namespace pimcore {

struct ServeRequest {
  int rid = 0;
  int prompt = 0;
  int target_out = 0;
  int generated = 0;

  int ctx() const { return prompt + generated; }
  bool done() const { return generated >= target_out; }
};

struct ServeRecord {
  int it = 0;
  int batch = 0;
  double mean_ctx = 0.0;
  double t_iter_ms = 0.0;
  double throughput = 0.0;
  double tokens_per_expert = 0.0;
  double x_split = 1.0;
  int experts_xpu = 0;
  int experts_pim = 0;
  int switches = 0;
  int completed = 0;
};

struct ServingConfig {
  int max_batch = 32;
  double prompt_mean = 2048.0;
  double out_mean = 6144.0;
  uint64_t seed = 13;
};

class ServingSim {
 public:
  // `sys` must be the Proteus model; `sys_cfg` is configs/systems/proteus.yaml
  // (for the crossover constant). An optional request source replays a
  // (prompt, output) list instead of the lognormal generator.
  ServingSim(const syscore::SystemModel& sys, const syscore::ModelSpec& model,
             const ConfigNode& sys_cfg, const ServingConfig& cfg,
             std::vector<std::pair<int, int>> source = {});

  // trace_gen/gen_requests.py format: "arrival_ms prompt output" per line.
  static std::vector<std::pair<int, int>> read_request_trace(
      const std::string& path);

  // One decode iteration; empty when a replayed trace is fully drained.
  std::optional<ServeRecord> step(int it);
  std::vector<ServeRecord> run(int iterations);

 private:
  std::optional<ServeRequest> new_request();
  std::vector<int> routing_histogram(int batch);
  std::vector<int> placement_signature(const std::vector<int>& hist) const;

  const syscore::SystemModel& sys_;
  const syscore::ModelSpec& model_;
  ServingConfig cfg_;
  double crossover_ai_ = 0.0;
  std::mt19937_64 rng_;
  std::vector<ServeRequest> pool_;
  std::vector<std::pair<int, int>> source_;
  size_t source_pos_ = 0;
  bool replay_ = false;
  int next_rid_ = 0;
  std::vector<int> prev_sig_;
  bool have_prev_ = false;
};

}  // namespace pimcore
