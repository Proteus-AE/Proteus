// SysCore: C++ implementation of the operator-level system layer.
//
// An independent port of the decode-iteration timing and energy models --
// the Proteus engine with its co-execution split and in-flight batching,
// and the six baseline systems -- consuming the same YAML configurations as
// the Python layer (configs/models, configs/systems, configs/memory). Used
// by `pimcore_sys` to regenerate the overall evaluation tables and by the
// cross-validation harness that checks the two implementations against
// each other cell by cell.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "pimcore/config.hpp"

namespace pimcore {
namespace syscore {

// ---------------------------------------------------------------- //
// Model description and one steady-state decode-iteration workload.
// ---------------------------------------------------------------- //
struct ModelSpec {
  std::string name;
  bool moe = false;
  int n_layers = 0;
  int d_model = 0;
  int attn_reuse = 1;              // GQA group size / MLA head count
  double weight_bytes = 0.0;       // total parameter footprint
  double active_params = 0.0;      // parameters touched per token
  double kv_bytes_per_token = 0.0;
  int n_experts = 0;
  int top_k = 0;
  double expert_bytes = 0.0;
  double dense_bytes = 0.0;

  static ModelSpec load(const std::string& config_dir,
                        const std::string& name);
};

// Cumulative Proteus variants of the effectiveness analysis (Sec. V-C).
struct VariantFlags {
  bool adaptive_sched = true;    // +AS
  bool reconfig_datapath = true; // +RD
  bool operator_fusion = true;   // +OF
  bool expert_centric = true;    // +EC
  static VariantFlags named(const std::string& v);
};

struct Workload {
  const ModelSpec* model = nullptr;
  int batch = 0;
  int ctx_avg = 0;
  int ctx_peak = 0;
  double weight_bytes = 0.0;       // streamed (active experts only)
  double weight_flops = 0.0;
  double kv_bytes = 0.0;
  double attn_flops = 0.0;
  double peak_mem = 0.0;
  double tokens_per_expert = 0.0;
  double active_experts = 0.0;
  int attn_reuse = 1;
  int d_model = 0;

  static Workload build(const ModelSpec& m, int batch, int ctx_in,
                        int ctx_out, std::optional<int> ctx_override = {});
};

// ---------------------------------------------------------------- //
// System models.
// ---------------------------------------------------------------- //
struct SysResult {
  bool alive = false;
  double throughput = 0.0;         // tokens/s
  double t_iter_ms = 0.0;
  double tokens_per_joule = 0.0;
  double power_w = 0.0;
  double x_split = 1.0;            // Proteus co-execution split
  int inflight = 0;
};

class SystemModel {
 public:
  virtual ~SystemModel() = default;
  virtual SysResult simulate(const Workload& w) const = 0;
  virtual std::string name() const = 0;

  // Optional extended entry point (Proteus): variant flags, device-count
  // override, and [PP, DP] hybrid parallelism. Defaults to simulate().
  virtual SysResult simulate_ex(const Workload& w, const VariantFlags&,
                                int /*devices*/, int /*dp*/) const {
    return simulate(w);
  }
};

// Factory: build a system model from configs/systems/<name>.yaml.
// `config_root` is the repository configs/ directory.
std::unique_ptr<SystemModel> build_system(const std::string& config_root,
                                          const std::string& name);

// Shared efficiency helpers (identical semantics to the Python layer).
double small_op_efficiency(const ConfigNode& table, double tpe);
double moe_frag_efficiency(const ConfigNode& curve, double tpe);
double weight_stream_curve(const ConfigNode& curve, double wt_bytes);
double short_payload_factor(int d_model);

}  // namespace syscore
}  // namespace pimcore
