#include "pimcore/serving.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace pimcore {

ServingSim::ServingSim(const syscore::SystemModel& sys, const syscore::ModelSpec& model,
                       const ConfigNode& sys_cfg, const ServingConfig& cfg,
                       std::vector<std::pair<int, int>> source)
    : sys_(sys), model_(model), cfg_(cfg), rng_(cfg.seed),
      source_(std::move(source)) {
  replay_ = !source_.empty();
  if (sys_cfg.has("scheduler"))
    crossover_ai_ = sys_cfg.at("scheduler").get_double("crossover_ai");
  while ((int)pool_.size() < cfg_.max_batch) {
    auto r = new_request();
    if (!r) break;
    pool_.push_back(*r);
  }
}

std::vector<std::pair<int, int>> ServingSim::read_request_trace(
    const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("request trace not found: " + path);
  std::vector<std::pair<int, int>> rows;
  std::string line;
  while (std::getline(f, line)) {
    size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    std::istringstream ss(line);
    double arrival;
    int prompt, out;
    if (ss >> arrival >> prompt >> out) rows.emplace_back(prompt, out);
  }
  return rows;
}

std::optional<ServeRequest> ServingSim::new_request() {
  ServeRequest r;
  if (replay_) {
    if (source_pos_ >= source_.size()) return std::nullopt;
    r.prompt = source_[source_pos_].first;
    r.target_out = source_[source_pos_].second;
    ++source_pos_;
  } else {
    std::lognormal_distribution<double> lp(0.0, 0.35), lo(0.0, 0.45);
    r.prompt = std::max(64, (int)(lp(rng_) * cfg_.prompt_mean));
    r.target_out = std::max(32, (int)(lo(rng_) * cfg_.out_mean));
  }
  r.rid = next_rid_++;
  return r;
}

std::vector<int> ServingSim::routing_histogram(int batch) {
  // Per-token top-k routing: k distinct experts sampled uniformly
  // (partial Fisher-Yates), accumulated over the batch.
  int n = model_.n_experts, k = model_.top_k;
  std::vector<int> hist(n, 0), idx(n);
  for (int t = 0; t < batch; ++t) {
    std::iota(idx.begin(), idx.end(), 0);
    for (int j = 0; j < k; ++j) {
      std::uniform_int_distribution<int> pick(j, n - 1);
      std::swap(idx[j], idx[pick(rng_)]);
      hist[idx[j]] += 1;
    }
  }
  return hist;
}

std::vector<int> ServingSim::placement_signature(
    const std::vector<int>& hist) const {
  // Arithmetic intensity of one expert GEMM with n routed tokens:
  // 2*n*d flops over (n + d)-ish traffic -> n*d/(2n+d) per the crossover
  // model; the expert runs on the xPU iff that exceeds the crossover AI.
  std::vector<int> sig;
  double d = model_.d_model;
  for (int nt : hist)
    if (nt > 0)
      sig.push_back(nt * d / (2.0 * nt + d) > crossover_ai_ ? 1 : 0);
  return sig;
}

std::optional<ServeRecord> ServingSim::step(int it) {
  std::vector<ServeRequest*> batch;
  for (auto& r : pool_)
    if (!r.done() && (int)batch.size() < cfg_.max_batch)
      batch.push_back(&r);
  if (batch.empty()) return std::nullopt;   // replayed trace drained

  double mean_ctx = 0.0;
  int peak_ctx = 0;
  for (auto* r : batch) {
    mean_ctx += r->ctx();
    peak_ctx = std::max(peak_ctx, r->ctx());
  }
  mean_ctx /= batch.size();

  syscore::Workload w =
      syscore::Workload::build(model_, (int)batch.size(), 0, 0, (int)mean_ctx);
  w.ctx_peak = peak_ctx;
  w.peak_mem = model_.weight_bytes +
               batch.size() * (double)peak_ctx * model_.kv_bytes_per_token;

  std::vector<int> sig;
  if (model_.moe) {
    auto hist = routing_histogram((int)batch.size());
    int n_act = 0;
    for (int h : hist) n_act += (h > 0);
    n_act = std::max(n_act, 1);
    w.active_experts = n_act;
    w.weight_bytes =
        model_.dense_bytes + (double)n_act * model_.expert_bytes *
                                 model_.n_layers;
    w.tokens_per_expert = batch.size() * (double)model_.top_k / n_act;
    sig = placement_signature(hist);
  }

  syscore::SysResult res = sys_.simulate(w);

  int switches = 0;
  if (have_prev_) {
    size_t common = std::min(sig.size(), prev_sig_.size());
    for (size_t i = 0; i < common; ++i)
      switches += (sig[i] != prev_sig_[i]);
    switches += (int)(std::max(sig.size(), prev_sig_.size()) - common);
  }
  prev_sig_ = sig;
  have_prev_ = true;

  int completed = 0;
  for (auto* r : batch) {
    r->generated += 1;
    completed += r->done();
  }
  pool_.erase(std::remove_if(pool_.begin(), pool_.end(),
                             [](const ServeRequest& r) { return r.done(); }),
              pool_.end());
  while ((int)pool_.size() < cfg_.max_batch) {
    auto nr = new_request();
    if (!nr) break;                          // trace exhausted: drain
    pool_.push_back(*nr);
  }

  ServeRecord rec;
  rec.it = it;
  rec.batch = (int)batch.size();
  rec.mean_ctx = mean_ctx;
  rec.t_iter_ms = res.t_iter_ms;
  rec.throughput = res.throughput;
  rec.tokens_per_expert = w.tokens_per_expert;
  rec.x_split = res.x_split;
  rec.experts_xpu = std::accumulate(sig.begin(), sig.end(), 0);
  rec.experts_pim = (int)sig.size() - rec.experts_xpu;
  rec.switches = switches;
  rec.completed = completed;
  return rec;
}

std::vector<ServeRecord> ServingSim::run(int iterations) {
  std::vector<ServeRecord> out;
  for (int i = 0; i < iterations; ++i) {
    auto rec = step(i);
    if (!rec) break;
    out.push_back(*rec);
  }
  return out;
}

}  // namespace pimcore
