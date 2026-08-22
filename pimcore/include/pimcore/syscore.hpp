// SysCore: C++ implementation of the operator-level system layer.
//
// An independent port of the decode-iteration timing and energy models --
// the Proteus engine with its memory-hierarchy-aligned parallelism, its
// crossover scheduler and its co-execution split, and the six baseline
// systems -- consuming the same YAML configurations as the Python layer
// (configs/models, configs/systems, configs/memory). Used by `pimcore_sys`
// to regenerate the overall evaluation tables and by the cross-validation
// harness that checks the two implementations against each other cell by
// cell.
#pragma once

#include <memory>
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
  std::string attention;           // "mha" | "gqa" | "mla"
  bool moe = false;
  int n_layers = 0;
  int d_model = 0;
  int n_heads = 0;
  int n_kv_heads = 1;
  int d_head = 0;
  int d_ffn = 0;
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

// One operator group of a transformer block. Emitting every weight matrix
// separately, and one skinny-GEMM per routed expert, is what lets the
// crossover estimate be applied at the granularity the runtime schedules.
struct OperatorGroup {
  std::string name;
  int kind = 0;                    // 0 = weight GEMM, 1 = attention, 2 = elementwise
  double bytes = 0.0;              // resident-operand traffic, one pass
  double flops = 0.0;
  double n_vectors = 1.0;          // concurrent vectors sharing the operand
  double k_dim = 0.0;              // shared-operand rows
  double n_out = 0.0;              // shared-operand columns
  double reuse = 1.0;
  double tokens = 0.0;             // routed tokens (>0 marks an MoE expert)

  bool is_expert() const { return tokens > 0.0; }
  double intensity() const;        // Eq. (3), rectangular form
};

// Cumulative Proteus variants of the effectiveness analysis (Sec. V-D).
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
  int n_layers = 0;
  std::vector<OperatorGroup> block;

  double activation_bytes() const {
    return static_cast<double>(batch) * d_model * 2.0;
  }

  static Workload build(const ModelSpec& m, int batch, int ctx_in,
                        int ctx_out, std::optional<int> ctx_override = {},
                        bool expert_centric = true);
};

// ---------------------------------------------------------------- //
// Interconnect: chunk-pipelined ring collectives (Sec. IV-B).
// ---------------------------------------------------------------- //
struct Fabric {
  double link_bw = 0.0;            // B/s per direction on one port
  double latency_ns = 0.0;
  double doorbell_ns = 0.0;

  static Fabric from_config(const ConfigNode& ic);
  double allreduce_bytes(double act_bytes, int n, int per_layer) const;
  double allreduce_ns(double act_bytes, int n, int per_layer) const;
  double transfer_ns(double nbytes) const;
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
  double x_split = 1.0;            // fraction of weight bytes on the xPU
  int inflight = 0;
  int tp_width = 1;
  int pipeline_groups = 1;
  int layers_per_stage = 0;
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

  // Machine balance points of the analytical crossover model (Sec. IV-D):
  // theta = F_PIM/BW_out and the two ridge points AI_PIM, AI_xPU. Zero on a
  // system that has no such model.
  virtual double theta() const { return 0.0; }
  virtual double ridge_pim() const { return 0.0; }
  virtual double ridge_xpu() const { return 0.0; }
};

// Factory: build a system model from configs/systems/<name>.yaml.
// `config_root` is the repository configs/ directory.
// `theta_override` (> 0) replaces the analytical crossover threshold the
// scheduler is configured with, which is what the Fig. 15 sweep perturbs.
std::unique_ptr<SystemModel> build_system(const std::string& config_root,
                                          const std::string& name,
                                          double theta_override = 0.0);

// Shared efficiency helpers (identical semantics to the Python layer).
double small_op_efficiency(const ConfigNode& table, double tpe);
double moe_frag_efficiency(const ConfigNode& curve, double tpe);
double weight_stream_curve(const ConfigNode& curve, double wt_bytes);
double short_payload_factor(int d_model);
double shared_operand_ai(double n, double k, double m);

}  // namespace syscore
}  // namespace pimcore
