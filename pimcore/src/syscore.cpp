#include "pimcore/syscore.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace pimcore {
namespace syscore {

namespace {
constexpr double TB = 1e12;
constexpr double GB = 1e9;

double pj_bit_to_j_byte(double pj_per_bit) { return pj_per_bit * 8e-12; }

enum Kind { WEIGHT_GEMM = 0, ATTENTION = 1, ELEMENTWISE = 2 };
}  // namespace

// ---------------------------------------------------------------- //
// Shared helpers (semantics identical to the Python layer).
// ---------------------------------------------------------------- //

double shared_operand_ai(double n, double k, double m) {
  // Eq. (3), rectangular form: n x k activations against a shared k x m FP16
  // matrix. N_ops = 2 n k m, N_acc = 2 (k m + n k + n m).
  n = std::max(n, 1.0);
  return n * k * m / (k * m + n * k + n * m);
}

double OperatorGroup::intensity() const {
  if (kind == ATTENTION) return reuse;
  if (kind == WEIGHT_GEMM) return shared_operand_ai(n_vectors, k_dim, n_out);
  return flops / std::max(bytes, 1.0);
}

double small_op_efficiency(const ConfigNode& table, double tpe) {
  if (tpe < 2.0) return table.get_double("tpe_lt2");
  if (tpe < 4.0) return table.get_double("tpe_lt4");
  return table.get_double("else");
}

double moe_frag_efficiency(const ConfigNode& curve, double tpe) {
  double sat = curve.get_double("saturate_tpe");
  return curve.get_double("base") +
         std::min(tpe, sat) / sat * curve.get_double("span");
}

double weight_stream_curve(const ConfigNode& curve, double wt_bytes) {
  return std::min(curve.get_double("cap"),
                  curve.get_double("base") +
                      curve.get_double("log2_slope") *
                          std::log2(wt_bytes / GB));
}

// Cross-system constants (configs/common.yaml), loaded once per process so
// that a single edit to that file reaches every system model.
const ConfigNode& common_config(const std::string& config_root) {
  static std::map<std::string, ConfigNode> cache;
  auto it = cache.find(config_root);
  if (it == cache.end())
    it = cache.emplace(config_root,
                       ConfigNode::load_file(config_root + "/common.yaml"))
             .first;
  return it->second;
}

double short_payload_factor(const ConfigNode& common, int d_model) {
  const ConfigNode& sp = common.at("short_payload");
  return d_model <= sp.get_int("d_model_threshold")
             ? sp.get_double("system_efficiency")
             : 1.0;
}

double host_overhead_s(const ConfigNode& common) {
  return common.get_double("host_iteration_overhead_ms") * 1e-3;
}

// ---------------------------------------------------------------- //
// Interconnect.
// ---------------------------------------------------------------- //

Fabric Fabric::from_config(const ConfigNode& ic) {
  Fabric f;
  f.link_bw = ic.get_double("link_bytes_per_s_per_dir", 0.0);
  f.latency_ns = ic.get_double("end_to_end_latency_ns", 0.0);
  f.doorbell_ns = ic.get_double("doorbell_ns", 0.0);
  return f;
}

double Fabric::allreduce_bytes(double act_bytes, int n, int per_layer) const {
  if (n <= 1) return 0.0;
  return per_layer * 2.0 * (n - 1) / n * act_bytes;
}

double Fabric::allreduce_ns(double act_bytes, int n, int per_layer) const {
  if (n <= 1 || link_bw <= 0.0) return 0.0;
  // Chunk-pipelined ring: the payload streams at link rate, but every chunk
  // still traverses the 2(N-1) reduce-scatter and all-gather hops.
  double hops = 2.0 * (n - 1) * per_layer;
  return allreduce_bytes(act_bytes, n, per_layer) / link_bw * 1e9 +
         hops * latency_ns;
}

double Fabric::transfer_ns(double nbytes) const {
  if (link_bw <= 0.0) return 0.0;
  return latency_ns + doorbell_ns + nbytes / link_bw * 1e9;
}

// ---------------------------------------------------------------- //
// Model + workload.
// ---------------------------------------------------------------- //

VariantFlags VariantFlags::named(const std::string& v) {
  VariantFlags f;
  if (v == "base") { f.adaptive_sched = f.reconfig_datapath =
                     f.operator_fusion = f.expert_centric = false; }
  else if (v == "as") { f.reconfig_datapath = f.operator_fusion =
                        f.expert_centric = false; }
  else if (v == "rd") { f.operator_fusion = f.expert_centric = false; }
  else if (v == "of") { f.expert_centric = false; }
  else if (v != "ec" && v != "full")
    throw std::runtime_error("unknown variant: " + v);
  return f;
}

ModelSpec ModelSpec::load(const std::string& config_dir,
                          const std::string& name) {
  ConfigNode c = ConfigNode::load_file(config_dir + "/models/" + name +
                                       ".yaml");
  ModelSpec m;
  m.name = c.get_string("name");
  m.attention = c.get_string("attention", "mha");
  m.n_layers = static_cast<int>(c.get_int("n_layers"));
  m.d_model = static_cast<int>(c.get_int("d_model"));
  m.n_heads = static_cast<int>(c.get_int("n_heads"));
  m.n_kv_heads = static_cast<int>(c.get_int("n_kv_heads"));
  m.d_head = static_cast<int>(c.get_int("d_head"));
  m.d_ffn = static_cast<int>(c.get_int("d_ffn"));
  m.weight_bytes = c.get_double("weight_bytes");
  m.active_params = c.get_double("active_params");
  m.kv_bytes_per_token = c.get_double("kv_bytes_per_token");
  if (c.has("attention_reuse")) {
    m.attn_reuse = static_cast<int>(c.get_int("attention_reuse"));
  } else {
    m.attn_reuse = std::max(1, m.n_heads / std::max(m.n_kv_heads, 1));
  }
  const ConfigNode& moe = c.at("moe");
  m.moe = moe.at("enabled").as_bool();
  if (m.moe) {
    m.n_experts = static_cast<int>(moe.get_int("n_experts"));
    m.top_k = static_cast<int>(moe.get_int("top_k"));
    m.expert_bytes = moe.get_double("expert_bytes");
    m.dense_bytes = moe.get_double("dense_bytes");
  }
  return m;
}

namespace {

struct Shape { const char* name; double k; double m; };

// Per-layer weight matrices of one transformer block (Fig. 1).
void block_matrices(const ModelSpec& m, std::vector<Shape>* dense,
                    std::vector<Shape>* expert) {
  double d = m.d_model;
  double q_out = static_cast<double>(m.n_heads) * m.d_head;
  double kv_out = static_cast<double>(m.n_kv_heads) * m.d_head;
  double ff = m.d_ffn;
  *dense = {{"q_proj", d, q_out}, {"k_proj", d, kv_out},
            {"v_proj", d, kv_out}, {"o_proj", q_out, d}};
  *expert = {{"gate", d, ff}, {"up", d, ff}, {"down", ff, d}};
  if (!m.moe) {
    dense->push_back({"ffn_gate", d, ff});
    dense->push_back({"ffn_up", d, ff});
    dense->push_back({"ffn_down", ff, d});
    expert->clear();
  }
}

double params(const std::vector<Shape>& s) {
  double t = 0.0;
  for (const auto& x : s) t += x.k * x.m;
  return t;
}

}  // namespace

Workload Workload::build(const ModelSpec& m, int batch, int ctx_in,
                         int ctx_out, std::optional<int> ctx_override,
                         bool expert_centric) {
  Workload w;
  w.model = &m;
  w.batch = batch;
  if (ctx_override) {
    w.ctx_avg = *ctx_override;
    w.ctx_peak = *ctx_override;
  } else {
    w.ctx_avg = ctx_in + ctx_out / 2;
    w.ctx_peak = ctx_in + ctx_out;
  }
  w.attn_reuse = m.attn_reuse;
  w.d_model = m.d_model;
  w.n_layers = m.n_layers;
  w.kv_bytes = static_cast<double>(batch) * w.ctx_avg * m.kv_bytes_per_token;
  w.attn_flops = w.kv_bytes * m.attn_reuse;
  w.weight_flops = 2.0 * m.active_params * batch;
  w.peak_mem = m.weight_bytes +
               static_cast<double>(batch) * w.ctx_peak * m.kv_bytes_per_token;
  if (m.moe) {
    double n = m.n_experts;
    double de = n * (1.0 - std::pow(1.0 - 1.0 / n,
                                    static_cast<double>(batch) * m.top_k));
    w.active_experts = de;
    w.weight_bytes = m.dense_bytes + de * m.expert_bytes * m.n_layers;
    w.tokens_per_expert = static_cast<double>(batch) * m.top_k / de;
  } else {
    w.weight_bytes = m.weight_bytes;
    w.tokens_per_expert = batch;
  }

  // ---- operator groups of one transformer block --------------------- //
  double L = m.n_layers;
  double d = m.d_model;
  OperatorGroup att;
  att.name = "attention";
  att.kind = ATTENTION;
  att.bytes = w.kv_bytes / L;
  att.flops = att.bytes * w.attn_reuse;
  att.n_vectors = batch;
  att.k_dim = att.n_out = d;
  att.reuse = w.attn_reuse;
  w.block.push_back(att);

  std::vector<Shape> dense_sh, expert_sh;
  block_matrices(m, &dense_sh, &expert_sh);
  double wb_l = w.weight_bytes / L;
  double scale_d, scale_e = 0.0;
  int n_act = 1;
  double tpe = std::max(w.tokens_per_expert, 1.0);
  if (m.moe) {
    n_act = std::max(1, static_cast<int>(std::lround(w.active_experts)));
    double dense_l = m.dense_bytes / L;
    double expert_l = std::max(wb_l - dense_l, 0.0);
    scale_d = dense_l / (2.0 * params(dense_sh));
    scale_e = params(expert_sh) > 0
                  ? expert_l / (2.0 * params(expert_sh) * n_act) : 0.0;
  } else {
    scale_d = wb_l / (2.0 * params(dense_sh));
  }
  for (const auto& sh : dense_sh) {
    OperatorGroup g;
    g.name = sh.name;
    g.kind = WEIGHT_GEMM;
    g.bytes = 2.0 * sh.k * sh.m * scale_d;
    g.flops = g.bytes * batch;
    g.n_vectors = batch;
    g.k_dim = sh.k;
    g.n_out = sh.m;
    g.reuse = batch;
    w.block.push_back(g);
  }
  if (m.moe) {
    double n_vec = expert_centric ? std::max(std::round(tpe), 1.0) : 1.0;
    for (int i = 0; i < n_act; ++i) {
      for (const auto& sh : expert_sh) {
        OperatorGroup g;
        g.name = "expert" + std::to_string(i) + "_" + sh.name;
        g.kind = WEIGHT_GEMM;
        g.bytes = 2.0 * sh.k * sh.m * scale_e;
        g.flops = g.bytes * tpe;
        g.n_vectors = n_vec;
        g.k_dim = sh.k;
        g.n_out = sh.m;
        g.reuse = n_vec;
        g.tokens = tpe;
        w.block.push_back(g);
      }
    }
  }
  OperatorGroup el;
  el.name = "nonlinear";
  el.kind = ELEMENTWISE;
  el.bytes = 2.0 * batch * d * 2.0;
  el.flops = 8.0 * batch * d;
  el.n_vectors = batch;
  el.k_dim = el.n_out = d;
  w.block.push_back(el);

  // Renormalize so the emitted groups reproduce the model-level aggregates.
  double sb = 0.0, sf = 0.0;
  for (const auto& g : w.block)
    if (g.kind == WEIGHT_GEMM) { sb += g.bytes; sf += g.flops; }
  double kb = sb > 0 ? (w.weight_bytes / L) / sb : 1.0;
  double kf = sf > 0 ? (w.weight_flops / L) / sf : 1.0;
  for (auto& g : w.block)
    if (g.kind == WEIGHT_GEMM) { g.bytes *= kb; g.flops *= kf; }
  return w;
}

// ---------------------------------------------------------------- //
// Base class with config plumbing.
// ---------------------------------------------------------------- //

class ConfiguredSystem : public SystemModel {
 public:
  ConfiguredSystem(const std::string& config_root, const std::string& name)
      : cfg_(load_config(config_root, "systems", name, "--system")),
        common_(common_config(config_root)),
        config_root_(config_root) {}

  std::string name() const override { return cfg_.get_string("name"); }

  SysResult simulate(const Workload& w) const override {
    return simulate_ex(w, VariantFlags{}, 0, 1);
  }

 protected:
  // Sustained weight-streaming efficiency of a well-formed dense GEMM.
  static constexpr double kDenseStreamEff = 0.75;

  double xw_eff(const Workload& w) const {
    const ConfigNode& c = cfg_.has("efficiency")
                              ? cfg_.at("efficiency").at("moe_frag")
                              : cfg_.at("weight_eff");
    double dense = kDenseStreamEff;
    if (cfg_.has("efficiency") && cfg_.at("efficiency").has("weight_stream"))
      dense = cfg_.at("efficiency").get_double("weight_stream");
    else
      dense = c.get_double("dense", kDenseStreamEff);
    if (!w.model->moe) return dense;
    double tpe = std::max(w.tokens_per_expert, 1.0);
    if (tpe >= c.get_double("saturate_tpe")) return dense;
    return std::min(dense, moe_frag_efficiency(c, tpe));
  }

  // Linear resource scaling when a non-default device count is used.
  int n_devices(int devices) const {
    return devices > 0 ? devices
                       : static_cast<int>(cfg_.get_double("devices"));
  }
  double dev_scale(int devices) const {
    return n_devices(devices) / cfg_.get_double("devices");
  }

  double total_capacity(int devices) const {
    double frac = cfg_.get_double("usable_fraction", 0.90);
    double s = dev_scale(devices);
    if (cfg_.has("hbm_capacity"))
      return cfg_.get_double("hbm_capacity") * s * frac;
    if (cfg_.has("capacity")) return cfg_.get_double("capacity") * s * frac;
    return cfg_.get_double("devices") * s *
           cfg_.get_double("capacity_per_device") * frac;
  }

  // Seconds the declared compute engines need for one iteration. Every
  // system that publishes an arithmetic throughput is given the
  // corresponding roofline (Sec. V-A).
  double compute_s(const Workload& w, int devices) const {
    for (const char* key : {"flops_fp16_aggregate", "xpu_flops_aggregate",
                            "pim_flops_aggregate"}) {
      if (!cfg_.has(key)) continue;
      double fl = cfg_.get_double(key) * dev_scale(devices);
      return (w.weight_flops + w.attn_flops) / fl;
    }
    return 0.0;
  }

  // Seconds per decode iteration spent in tensor-parallel AllReduces.
  double collective_s(const Workload& w, int devices) const {
    if (!cfg_.has("interconnect")) return 0.0;
    int n = n_devices(devices);
    int per_layer =
        static_cast<int>(cfg_.get_int("tp_collectives_per_layer", 2));
    if (n <= 1 || per_layer <= 0) return 0.0;
    Fabric f = Fabric::from_config(cfg_.at("interconnect"));
    return f.allreduce_ns(w.activation_bytes(), n, per_layer) * w.n_layers /
           1e9;
  }

  double short_f(const Workload& w) const {
    return short_payload_factor(common_, w.d_model);
  }
  double host_s() const { return host_overhead_s(common_); }

  // Fill the shared part of a live result.
  void finish(SysResult* r, const Workload& w, double t) const {
    r->alive = true;
    r->throughput = w.batch / t;
    r->t_iter_ms = t * 1e3;
  }

  ConfigNode cfg_;
  ConfigNode common_;
  std::string config_root_;
};

// ---------------------------------------------------------------- //
// DGX-A100 (vLLM serving).
// ---------------------------------------------------------------- //

class GpuSystem : public ConfiguredSystem {
 public:
  using ConfiguredSystem::ConfiguredSystem;

  SysResult simulate_ex(const Workload& w, const VariantFlags&, int devices,
                        int) const override {
    SysResult r;
    if (w.peak_mem > total_capacity(devices)) return r;
    const ConfigNode& eff = cfg_.at("efficiency");
    double bw = cfg_.get_double("hbm_bw_aggregate") * dev_scale(devices);
    double t = std::max(w.weight_bytes / (bw * xw_eff(w)) +
                            w.kv_bytes / (bw * eff.get_double("attention")),
                        compute_s(w, devices));
    t += collective_s(w, devices) + host_s();
    t /= short_f(w);
    finish(&r, w, t);
    const ConfigNode& en = cfg_.at("energy");
    // Board power already includes the HBM stacks, so no separate
    // background term is charged for them.
    double p = n_devices(devices) * en.get_double("gpu_busy_w") +
               en.get_double("static_w");
    double dram = (w.weight_bytes + w.kv_bytes) / w.batch *
                  pj_bit_to_j_byte(en.get_double("hbm_pj_per_bit"));
    r.power_w = p + dram * r.throughput;
    r.tokens_per_joule = r.throughput / r.power_w;
    return r;
  }
};

// ---------------------------------------------------------------- //
// CXL-PNM (channel-level near-memory processing).
// ---------------------------------------------------------------- //

class CxlPnmSystem : public ConfiguredSystem {
 public:
  using ConfiguredSystem::ConfiguredSystem;

  SysResult simulate_ex(const Workload& w, const VariantFlags&, int devices,
                        int) const override {
    SysResult r;
    if (w.peak_mem > total_capacity(devices)) return r;
    double dev = cfg_.get_double("devices") * dev_scale(devices);
    double bw = dev * cfg_.get_double("bw_per_device") *
                cfg_.get_double("stream_efficiency");
    double fl = dev * cfg_.get_double("flops_per_device") *
                cfg_.get_double("compute_efficiency");
    double bytes = w.weight_bytes + w.kv_bytes;
    double flops = w.weight_flops + w.attn_flops;
    double t = std::max(bytes / bw, flops / fl) + collective_s(w, devices) +
               host_s();
    t /= short_f(w);
    finish(&r, w, t);
    const ConfigNode& en = cfg_.at("energy");
    double duty = std::min((flops / fl) / t, 1.0);
    double cap_gb = dev * cfg_.get_double("capacity_per_device") / GB;
    double bg = cap_gb * en.get_double("background_w_per_gb") *
                en.get_double("background_idle_factor");
    double p = dev * (en.get_double("controller_w_per_device") +
                      en.get_double("engine_full_load_w") * duty) +
               en.get_double("static_w") + bg;
    double dram = bytes / w.batch *
                  pj_bit_to_j_byte(en.get_double("lpddr_ext_pj_per_bit"));
    r.power_w = p + dram * r.throughput;
    r.tokens_per_joule = r.throughput / r.power_w;
    return r;
  }
};

// ---------------------------------------------------------------- //
// CENT (GPU-free GDDR6-AiM scale-out; GEMV decomposition).
// ---------------------------------------------------------------- //

class CentSystem : public ConfiguredSystem {
 public:
  using ConfiguredSystem::ConfiguredSystem;

  // Whole layers are pipelined across devices rather than sharded, so the
  // only per-token communication is one activation per stage boundary.
  double boundary_s(const Workload& w, int devices) const {
    int n = std::min(static_cast<int>(
                         cfg_.get_int("pipeline_boundaries_per_token", 0)),
                     n_devices(devices));
    if (n <= 0) return 0.0;
    Fabric f = Fabric::from_config(cfg_.at("interconnect"));
    return n * f.transfer_ns(w.activation_bytes()) / 1e9;
  }

  SysResult simulate_ex(const Workload& w, const VariantFlags&, int devices,
                        int) const override {
    SysResult r;
    if (w.peak_mem > total_capacity(devices)) return r;
    double dev = cfg_.get_double("devices") * dev_scale(devices);
    double tpe = std::max(w.tokens_per_expert, 1.0);
    double bw = dev * cfg_.get_double("internal_bw_per_device") *
                cfg_.get_double("stream_efficiency");
    double smallf = small_op_efficiency(cfg_.at("small_op_efficiency"), tpe);
    double traffic = w.weight_bytes * tpe + w.kv_bytes * w.attn_reuse;
    // Only the weight stream is fragmented into per-token GEMVs; the KV
    // stream is one long per-request chain and keeps the full rate.
    double t = std::max(w.weight_bytes * tpe / (bw * smallf) +
                            w.kv_bytes * w.attn_reuse / bw,
                        compute_s(w, devices));
    t += boundary_s(w, devices) + host_s();
    t /= short_f(w);
    finish(&r, w, t);
    const ConfigNode& en = cfg_.at("energy");
    double p = dev * en.get_double("device_w") + en.get_double("static_w");
    double dram = traffic / w.batch *
                  pj_bit_to_j_byte(en.get_double("gddr_int_pj_per_bit"));
    r.power_w = p + dram * r.throughput;
    r.tokens_per_joule = r.throughput / r.power_w;
    return r;
  }
};

// ---------------------------------------------------------------- //
// NeuPIMs (dual-row-buffer NPU + HBM-PIM concurrency).
// ---------------------------------------------------------------- //

class NeuPimsSystem : public ConfiguredSystem {
 public:
  using ConfiguredSystem::ConfiguredSystem;

  SysResult simulate_ex(const Workload& w, const VariantFlags&, int devices,
                        int) const override {
    SysResult r;
    if (w.peak_mem > total_capacity(devices)) return r;
    double bw = cfg_.get_double("hbm_bw_aggregate") * dev_scale(devices);
    double t_fc = w.weight_bytes / (bw * cfg_.get_double("weight_stream"));
    double t_pim = w.kv_bytes * w.attn_reuse /
                   (bw * cfg_.get_double("pim_internal_mult") *
                    cfg_.get_double("pim_stream_efficiency"));
    double t_npu = w.kv_bytes / (bw * cfg_.get_double("attention_xpu_eff"));
    bool att_on_pim = t_pim <= t_npu;
    double t_att = std::min(t_pim, t_npu);
    double t = std::max(std::max(t_fc, t_att) *
                            cfg_.get_double("overlap_overhead"),
                        compute_s(w, devices));
    t += collective_s(w, devices) + host_s();
    double sf = short_f(w);
    t /= sf;
    finish(&r, w, t);
    const ConfigNode& en = cfg_.at("energy");
    double duty =
        std::min((t_fc + (att_on_pim ? 0.0 : t_att)) / (t * sf), 1.0);
    double dev = n_devices(devices);
    double bg = cfg_.get_double("hbm_capacity") * dev_scale(devices) / GB *
                en.get_double("background_w_per_gb") *
                en.get_double("background_idle_factor");
    double p = dev * (en.get_double("npu_busy_w") * duty +
                      en.get_double("npu_idle_w") * (1.0 - duty) +
                      en.get_double("pim_pe_w_per_device")) +
               en.get_double("static_w") + bg;
    double hbm_bytes = w.weight_bytes + (att_on_pim ? 0.0 : w.kv_bytes);
    double pim_bytes = att_on_pim ? w.kv_bytes * w.attn_reuse : 0.0;
    double dram =
        (hbm_bytes * pj_bit_to_j_byte(en.get_double("hbm_pj_per_bit")) +
         pim_bytes *
             pj_bit_to_j_byte(en.get_double("hbm_pim_int_pj_per_bit"))) /
        w.batch;
    r.power_w = p + dram * r.throughput;
    r.tokens_per_joule = r.throughput / r.power_w;
    return r;
  }
};

// ---------------------------------------------------------------- //
// PAPI (Attn-PIM / FC-PIM pools + xPU).
// ---------------------------------------------------------------- //

class PapiSystem : public ConfiguredSystem {
 public:
  using ConfiguredSystem::ConfiguredSystem;

  SysResult simulate_ex(const Workload& w, const VariantFlags&, int devices,
                        int) const override {
    SysResult r;
    double cap = cfg_.get_double("hbm_capacity") * dev_scale(devices) *
                 cfg_.get_double("usable_fraction");
    double frac = cfg_.get_double("attn_pool_fraction");
    double kv_cap = frac * cap, fc_cap = (1.0 - frac) * cap;
    if ((w.peak_mem - w.model->weight_bytes) > kv_cap) return r;

    double s = dev_scale(devices);
    double att_bw = frac * cfg_.get_double("attn_allbank_mult") *
                    cfg_.get_double("xpu_bw_aggregate") * s *
                    cfg_.get_double("attention_eff");
    double t_att = w.kv_bytes * w.attn_reuse / att_bw;

    double t_xpu =
        w.weight_bytes / (cfg_.get_double("xpu_bw_aggregate") * s * xw_eff(w));
    double t_fc_pim = std::numeric_limits<double>::infinity();
    if (w.model->weight_bytes <= fc_cap) {
      double fc_bw = (1.0 - frac) * cfg_.get_double("fc_pim_allbank_mult") *
                     cfg_.get_double("xpu_bw_aggregate") * s *
                     cfg_.get_double("fc_pim_efficiency");
      t_fc_pim = w.weight_bytes / fc_bw;
    }
    bool fc_on_pim = t_fc_pim <= t_xpu;
    double t_fc = std::min(t_fc_pim, t_xpu);

    double t = std::max(std::max(t_fc, t_att) *
                            cfg_.get_double("overlap_overhead"),
                        compute_s(w, devices));
    t += collective_s(w, devices) + host_s();
    double sf = short_f(w);
    t /= sf;
    finish(&r, w, t);

    const ConfigNode& en = cfg_.at("energy");
    double duty = fc_on_pim ? 0.0 : std::min(t_fc / (t * sf), 1.0);
    double dev = n_devices(devices);
    double bg = cfg_.get_double("hbm_capacity") * dev_scale(devices) / GB *
                en.get_double("background_w_per_gb") *
                en.get_double("background_idle_factor");
    double p = dev * (en.get_double("xpu_busy_w") * duty +
                      en.get_double("xpu_idle_w") * (1.0 - duty) +
                      en.get_double("pim_pe_w_per_device")) +
               en.get_double("static_w") + bg;
    double hbm_bytes = fc_on_pim ? 0.0 : w.weight_bytes;
    double pim_bytes =
        w.kv_bytes * w.attn_reuse + (fc_on_pim ? w.weight_bytes : 0.0);
    double dram =
        (hbm_bytes * pj_bit_to_j_byte(en.get_double("hbm_pj_per_bit")) +
         pim_bytes *
             pj_bit_to_j_byte(en.get_double("hbm_pim_int_pj_per_bit"))) /
        w.batch;
    r.power_w = p + dram * r.throughput;
    r.tokens_per_joule = r.throughput / r.power_w;
    return r;
  }
};

// ---------------------------------------------------------------- //
// PIMphony (orchestrated AiMX-class execution).
// ---------------------------------------------------------------- //

class PimphonySystem : public ConfiguredSystem {
 public:
  using ConfiguredSystem::ConfiguredSystem;

  SysResult simulate_ex(const Workload& w, const VariantFlags&, int devices,
                        int) const override {
    SysResult r;
    if (w.peak_mem > total_capacity(devices)) return r;
    double s = dev_scale(devices);
    double t_fc = w.weight_bytes / (cfg_.get_double("xpu_bw_aggregate") * s *
                                    cfg_.get_double("weight_stream") *
                                    cfg_.get_double("orchestration_gain"));
    double att_bw = cfg_.get_double("devices") * s *
                    cfg_.get_double("aim_internal_per_device") *
                    cfg_.get_double("pim_stream_efficiency") *
                    cfg_.get_double("pim_util");
    double t_att = w.kv_bytes * w.attn_reuse / att_bw;
    double t = std::max(std::max(t_fc, t_att) *
                            cfg_.get_double("overlap_overhead"),
                        compute_s(w, devices));
    t += collective_s(w, devices) + host_s();
    double sf = short_f(w);
    t /= sf;
    finish(&r, w, t);
    const ConfigNode& en = cfg_.at("energy");
    double duty = std::min(t_fc / (t * sf), 1.0);
    double dev = n_devices(devices);
    double cap = cfg_.has("hbm_capacity") ? cfg_.get_double("hbm_capacity")
                                         : cfg_.get_double("capacity");
    double bg = cap * dev_scale(devices) / GB *
                en.get_double("background_w_per_gb") *
                en.get_double("background_idle_factor");
    double p = dev * (en.get_double("xpu_busy_w") * duty +
                      en.get_double("xpu_idle_w") * (1.0 - duty) +
                      en.get_double("pim_pe_w_per_device")) +
               en.get_double("static_w") + bg;
    double dram =
        (w.weight_bytes * pj_bit_to_j_byte(en.get_double("hbm_pj_per_bit")) +
         w.kv_bytes * w.attn_reuse *
             pj_bit_to_j_byte(en.get_double("aim_int_pj_per_bit"))) /
        w.batch;
    r.power_w = p + dram * r.throughput;
    r.tokens_per_joule = r.throughput / r.power_w;
    return r;
  }
};

// ---------------------------------------------------------------- //
// Proteus.
// ---------------------------------------------------------------- //

namespace {

// Analytical cost of one operator group on each substrate (Eq. (1)).
struct OpCost {
  const OperatorGroup* op = nullptr;
  double ai = 0.0;
  bool crossover = false;
  double t_xpu = 0.0;
  double t_xpu_est = 0.0;
  double b_xpu = 0.0;
  double t_pim[2] = {0.0, 0.0};    // 0 = direct, 1 = broadcast
  double b_pim[2] = {0.0, 0.0};
  int mode = 0;

  double t_on(int sub) const { return sub == 0 ? t_xpu : t_pim[mode]; }
  double t_est(int sub) const { return sub == 0 ? t_xpu_est : t_pim[mode]; }
  double b_on(int sub) const { return sub == 0 ? b_xpu : b_pim[mode]; }
};

struct Assign { OpCost* c; int sub; double frac; };

}  // namespace

class ProteusSystem : public ConfiguredSystem {
 public:
  ProteusSystem(const std::string& root, const std::string& name,
                double theta_override = 0.0)
      : ConfiguredSystem(root, name),
        mem_(load_config(root, "memory", cfg_.get_string("memory"),
                         "memory")) {
    // Derived machine parameters -- identical closed form to
    // proteus_sim/memory.py; see that module for the derivation.
    int channels = static_cast<int>(mem_.get_int("packages_per_device") *
                                    mem_.get_int("channels_per_package"));
    int dies = channels * static_cast<int>(mem_.get_int("dies_per_channel"));
    banks_ = dies * static_cast<int>(mem_.get_int("bankgroups_per_die") *
                                     mem_.get_int("banks_per_bankgroup"));
    double burst = mem_.get_double("io_width") *
                   mem_.get_double("burst_length") / 8.0;
    fanout_ = mem_.get_double("broadcast_fanout",
                              mem_.get_double("banks_per_bankgroup"));
    double mac_ns = burst / (mem_.get_double("pe_lanes") * 2.0) /
                    mem_.get_double("pe_freq_ghz");
    double t_col_pim =
        mem_.get_double("tCCD_PIM_ns", mem_.get_double("tCCD_L_ns") / 2.0);
    double t_col_d = std::max(t_col_pim, mac_ns);
    double t_col_b = std::max(mem_.get_double("tCCD_L_ns"), fanout_ * mac_ns);
    internal_peak_ = banks_ * burst / (t_col_d * 1e-9);
    double bcast_peak = banks_ * burst / (t_col_b * 1e-9);
    pe_peak_ = banks_ * mem_.get_double("pe_lanes") * 2.0 *
               mem_.get_double("pe_freq_ghz") * 1e9;

    double rcd_ns = mem_.get_double("tRCD_ns"), rp_ns = mem_.get_double("tRP_ns");
    if (mem_.has("allbank")) {
      rcd_ns = mem_.at("allbank").get_double("act_rcd_ns", rcd_ns);
      rp_ns = mem_.at("allbank").get_double("rp_ns", rp_ns);
    }
    double refresh = 1.0 - mem_.get_double("tRFCab_ns") /
                               mem_.get_double("tREFI_ns");
    double bursts_per_row = mem_.get_double("row_bytes") / burst;
    double rtp = mem_.get_double("tRTP_ns");
    auto eff_of = [&](double t_col) {
      double stream = bursts_per_row * t_col;
      double tail = std::max(0.0, rtp - t_col);
      return stream / (rcd_ns + stream + tail + rp_ns) * refresh;
    };
    internal_bw_ = internal_peak_ * eff_of(t_col_d);
    broadcast_bw_ = bcast_peak * eff_of(t_col_b);
    pe_flops_ = pe_peak_ * mem_.get_double("pe_pipeline_eff");
    capacity_ = mem_.get_double("capacity_per_package_gb") * GB *
                mem_.get_double("packages_per_device");
    ext_peak_ = mem_.get_double("external_bw_per_device_tbps") * TB;
    xpu_bw_ = ext_peak_ * refresh * mem_.get_double("ext_bus_efficiency", 1.0);
    xpu_flops_ = cfg_.at("xpu").get_double("flops_fp16");

    // Column slots the all-bank stream leaves free for a concurrent host
    // stream. A bank serves one column access per tCCD_PIM and the PIM
    // stream claims one per cadence; a host burst additionally passes the
    // bank-group I/O (tCCD_L per BG) and the channel DQ (burst_ns).
    double burst_ns = mem_.get_double("burst_length") /
                      mem_.get_double("data_rate_mtps") * 1e3;
    int bgs = dies * static_cast<int>(mem_.get_int("bankgroups_per_die"));
    auto coexec_of = [&](double t_col) {
      double free_per_bank = std::max(0.0, t_col / t_col_pim - 1.0);
      double bursts = std::min({banks_ * free_per_bank,
                                bgs * t_col / mem_.get_double("tCCD_L_ns"),
                                channels * t_col / burst_ns});
      return std::min(bursts * burst / (t_col * 1e-9), xpu_bw_);
    };
    coexec_[0] = coexec_of(t_col_d);
    coexec_[1] = coexec_of(t_col_b);

    const ConfigNode& sc = cfg_.at("scheduler");
    theta_ = theta_override > 0.0 ? theta_override
                                  : sc.get_double("crossover_ai");
    hysteresis_ = sc.get_double("adaptation_hysteresis", 1.0);
    adaptive_ = !sc.has("runtime_adaptation") ||
                sc.at("runtime_adaptation").as_bool();
    ai_pim_ = pe_peak_ / internal_peak_;
    ai_xpu_ = xpu_flops_ / ext_peak_;
    if (cfg_.has("variant_penalties")) {
      const ConfigNode& v = cfg_.at("variant_penalties");
      k_cmd_ = v.get_double("per_op_command_issue", k_cmd_);
      k_sync_ = v.get_double("sync_weight_stream", k_sync_);
      f_unfused_ = v.get_double("unfused_stall", f_unfused_);
      k_slice_ = v.get_double("coarse_slicing", k_slice_);
      k_token_ = v.get_double("token_centric_issue", k_token_);
    }
    group_size_ =
        static_cast<int>(cfg_.at("parallelism").get_int("group_size", 8));
    tp_coll_ = static_cast<int>(
        cfg_.at("parallelism").get_int("tp_collectives_per_layer", 2));
    fabric_ = Fabric::from_config(cfg_.at("interconnect"));
  }

  double theta() const override { return theta_; }
  double ridge_pim() const override { return ai_pim_; }
  double ridge_xpu() const override { return ai_xpu_; }

  SysResult simulate_ex(const Workload& w, const VariantFlags& f,
                        int devices_override, int dp) const override {
    SysResult r;
    int devices = devices_override > 0
                      ? devices_override
                      : static_cast<int>(cfg_.get_double("devices"));
    if (dp > 1) {
      int sub_devices = std::max(devices / dp, 1);
      Workload sub = Workload::build(*w.model, std::max(w.batch / dp, 1), 0,
                                     0, w.ctx_avg, f.expert_centric);
      sub.ctx_peak = w.ctx_peak;
      sub.peak_mem = w.model->weight_bytes +
                     static_cast<double>(sub.batch) * w.ctx_peak *
                         w.model->kv_bytes_per_token;
      SysResult rr = simulate_ex(sub, f, sub_devices, 1);
      if (rr.alive) rr.throughput *= dp;
      return rr;
    }

    int n_tp = std::min(group_size_, devices);
    int groups = std::max(1, devices / n_tp);
    double cap = capacity_ * devices *
                 cfg_.at("capacity").get_double("usable_fraction");
    double kv_peak = w.peak_mem - w.model->weight_bytes;
    double kv_budget = cap - w.model->weight_bytes;
    if (kv_budget <= 0.0 || w.peak_mem > cap) return r;
    int m = static_cast<int>(std::min(
        static_cast<double>(groups), std::floor(kv_budget /
                                                std::max(kv_peak, 1.0))));
    if (m < 1) return r;
    int base = w.n_layers / groups, rem = w.n_layers % groups;
    int layers = std::max(base + (rem ? 1 : 0), 1);

    double short_sys = short_f(w);
    double cmd_eff = 1.0;
    if (mem_.has("short_payload") &&
        w.d_model <= mem_.at("short_payload").get_int("d_model_threshold"))
      cmd_eff = mem_.at("short_payload").get_double("pim_cmd_efficiency", 1.0);

    double pbw_d = internal_bw_ * cmd_eff;
    double pbw_b = broadcast_bw_ * cmd_eff;
    if (!f.adaptive_sched) { pbw_d *= k_cmd_; pbw_b *= k_cmd_; }
    // Without co-execution the substrates serialize and the xPU has the
    // whole external interface while it runs; with co-execution it is capped
    // by the headroom the chosen connectivity mode leaves.
    auto xpu_bw_in = [&](int mode) {
      double bw = (f.reconfig_datapath && coexec_[mode] > 0.0) ? coexec_[mode]
                                                               : xpu_bw_;
      return f.adaptive_sched ? bw : bw * k_sync_;
    };

    double shard = 1.0 / n_tp;
    double tpe = std::max(w.tokens_per_expert, 1.0);
    double smallf =
        small_op_efficiency(cfg_.at("scheduler").at("small_op_efficiency"),
                            tpe);

    // ---- per-block plan ------------------------------------------- //
    // At each decode iteration the scheduler jointly selects the execution
    // substrate and, for PIM-mapped operators, the connectivity mode
    // (Sec. IV-C "Lightweight Reconfiguration"). The mode register is per
    // channel, so the choice is evaluated on the whole block: taking the
    // broadcasting cadence can be worth a slower PIM stream when it buys the
    // concurrent xPU execution the freed memory-service slots allow.
    double elem_bytes = 0.0, elem_flops = 0.0;
    for (const auto& op : w.block) {
      if (op.kind != ELEMENTWISE) continue;
      elem_bytes += op.bytes;
      elem_flops += op.flops;
    }
    double stall = f.operator_fusion ? 0.0 : f_unfused_;
    double unfused = (!f.operator_fusion && elem_bytes > 0.0)
                         ? elem_bytes * shard * 2.0 / pbw_d
                         : 0.0;

    struct Plan {
      std::vector<OpCost> costs;
      std::vector<Assign> chosen;
      double q[2] = {0.0, 0.0};
      double by[2] = {0.0, 0.0};
      int modes[2] = {0, 0};
      double t_block = 0.0;
    };

    auto plan_block = [&](int mode_pref) {
      Plan pl;
      double xbw = xpu_bw_in(mode_pref < 0 ? 0 : mode_pref);
      pl.costs.reserve(w.block.size());
      for (const auto& op : w.block) {
        if (op.kind == ELEMENTWISE) continue;
        pl.costs.push_back(
            op_cost(op, shard, pbw_d, pbw_b, xbw, smallf, f, mode_pref));
      }
      double qe[2] = {0.0, 0.0};
      schedule_block(&pl.costs, f, pl.q, qe, pl.by, pl.modes, &pl.chosen);
      int mode = pl.modes[1] >= pl.modes[0] ? 1 : 0;
      double t_p = pl.q[1] * (1.0 + stall);
      if (f.reconfig_datapath && (coexec_[mode] > 0.0 || pl.q[1] <= 0.0)) {
        // Broadcasting lowers the memory-service demand of the PIM stream,
        // which is what leaves the xPU room to run against it.
        pl.t_block = std::max(pl.q[0], t_p);
      } else {
        // Direct mode drives every bank at its minimum column cycle and
        // returns no service slots, so the phases serialize whatever the
        // placement, and each switch between them costs a command-queue
        // drain and an xPU DMA restart.
        pl.t_block = (pl.q[0] + t_p) * k_slice_;
      }
      pl.t_block += unfused;
      return pl;
    };

    Plan best = plan_block(f.reconfig_datapath ? 0 : -1);
    if (f.reconfig_datapath) {
      Plan alt = plan_block(1);
      if (alt.t_block < best.t_block) best = std::move(alt);
    }
    // `costs` is reserved up front and both vectors are moved (never copied),
    // so the OpCost pointers held by `chosen` stay valid across the choice.
    double* q = best.q;
    double* by = best.by;
    int* modes = best.modes;
    double t_block = best.t_block;

    // ---- communication -------------------------------------------- //
    double t_coll =
        fabric_.allreduce_ns(w.activation_bytes(), n_tp, tp_coll_) * 1e-9;
    double t_xfer =
        groups > 1 ? fabric_.transfer_ns(w.activation_bytes()) * 1e-9 : 0.0;

    double t_stage = (t_block + t_coll) * layers + t_xfer;
    if (groups > 1)
      t_stage /= cfg_.at("pipeline").get_double("efficiency");
    t_stage += cfg_.at("interconnect").get_double("scheduling_overhead_ms") *
               1e-3;
    t_stage /= short_sys;
    double t_iter = t_stage * groups;

    r.alive = true;
    r.throughput = static_cast<double>(w.batch) * m / (t_stage * groups);
    r.t_iter_ms = t_iter * 1e3;
    r.inflight = m;
    r.tp_width = n_tp;
    r.pipeline_groups = groups;
    r.layers_per_stage = layers;
    r.x_split = by[0] / std::max(by[0] + by[1], 1.0);

    energy(&r, w, q, by, best.chosen, modes, layers, n_tp, groups, devices,
           shard, elem_flops, t_stage * short_sys);
    return r;
  }

 private:
  // ``mode_pref``: -1 selects the faster connectivity mode per operator,
  // 0 pins direct mode, 1 pins broadcasting mode. The mode register is per
  // channel, so a block-level plan pins one mode for every PIM operator.
  OpCost op_cost(const OperatorGroup& op, double shard, double pbw_d,
                 double pbw_b, double xbw, double smallf,
                 const VariantFlags& f, int mode_pref) const {
    OpCost c;
    c.op = &op;
    bool token_centric = op.is_expert() && !f.expert_centric;
    double eff = op.kind == WEIGHT_GEMM ? smallf : 1.0;
    double xeff = 1.0;
    double passes[2];
    if (token_centric) {
      // Fragmentation is an execution property, not a workload one: the
      // operator still presents `tokens` vectors against a shared operand and
      // Eq. (3) reads its intensity off that shape. Token-centric dispatch
      // changes how many times the datapath must fetch the operand, and the
      // near-bank PEs hold no operand storage beyond the FIFO, so the weights
      // are re-streamed once per token in either connectivity mode.
      double tokens = std::max(op.tokens, 1.0);
      passes[0] = passes[1] = tokens;
      eff *= k_token_;
      c.ai = op.intensity();
    } else {
      double reuse = std::max(op.kind == ATTENTION ? op.reuse : op.n_vectors,
                              1.0);
      passes[0] = reuse;
      passes[1] = std::max(1.0, std::ceil(reuse / fanout_));
      c.ai = op.intensity();
    }
    double bw[2] = {pbw_d, pbw_b};
    for (int mo = 0; mo < 2; ++mo) {
      c.b_pim[mo] = op.bytes * shard * passes[mo];
      c.t_pim[mo] = std::max(op.flops * shard / pe_flops_,
                             c.b_pim[mo] / (bw[mo] * eff));
    }
    if (!f.reconfig_datapath)
      c.mode = 0;
    else if (mode_pref == 0 || mode_pref == 1)
      c.mode = mode_pref;
    else
      c.mode = c.t_pim[1] < c.t_pim[0] ? 1 : 0;
    c.b_xpu = op.bytes * shard;
    c.t_xpu = std::max(op.flops * shard / xpu_flops_, c.b_xpu / (xbw * xeff));
    double xbw_est = (pe_peak_ / theta_) * (xbw / ext_peak_);
    c.t_xpu_est = std::max(op.flops * shard / xpu_flops_,
                           c.b_xpu / (std::max(xbw_est, 1.0) * xeff));
    c.crossover = c.ai > ai_pim_ && c.ai <= ai_xpu_;
    return c;
  }

  int default_substrate(double ai) const {
    if (ai > ai_xpu_) return 0;              // compute-dominated -> xPU
    if (ai <= ai_pim_) return 1;             // memory-dominated  -> PIM
    return ai > theta_ ? 0 : 1;              // crossover region  -> Eq. (2)
  }

  void schedule_block(std::vector<OpCost>* costs, const VariantFlags& f,
                      double q[2], double qe[2], double by[2], int modes[2],
                      std::vector<Assign>* chosen) const {
    bool adapt = f.adaptive_sched && adaptive_;
    bool overlap = f.reconfig_datapath;
    std::vector<OpCost*> order;
    order.reserve(costs->size());
    for (auto& c : *costs) order.push_back(&c);
    if (adapt && overlap)
      std::sort(order.begin(), order.end(), [](const OpCost* a,
                                               const OpCost* b) {
        return std::max(a->t_xpu_est, a->t_pim[a->mode]) >
               std::max(b->t_xpu_est, b->t_pim[b->mode]);
      });
    for (OpCost* c : order) {
      int sub;
      if (!f.adaptive_sched) {
        sub = c->op->kind == ATTENTION ? 1 : 0;
      } else {
        sub = default_substrate(c->ai);
        if (adapt && c->crossover) {
          int alt = 1 - sub;
          double here = c->t_est(sub) + (overlap ? qe[sub] : 0.0);
          double there = c->t_est(alt) + (overlap ? qe[alt] : 0.0);
          if (there * hysteresis_ < here) sub = alt;
        }
      }
      qe[sub] += c->t_est(sub);
      q[sub] += c->t_on(sub);
      by[sub] += c->b_on(sub);
      if (sub == 1) modes[c->mode] += 1;
      chosen->push_back({c, sub, 1.0});
    }
    if (adapt && overlap)
      for (size_t i = 0; i < chosen->size() + 1; ++i)
        if (!split_step(qe, q, by, chosen)) break;
  }

  // Work-conserving refinement: lend the idle substrate a column block of the
  // largest crossover-region GEMM on the critical path (Sec. IV-D). Planned on
  // the estimator's view of the queues, executed on the real machine.
  bool split_step(double qe[2], double q[2], double by[2],
                  std::vector<Assign>* chosen) const {
    int slow = qe[0] > qe[1] ? 0 : 1, fast = qe[0] > qe[1] ? 1 : 0;
    double gap = qe[slow] - qe[fast];
    if (gap <= 1e-12) return false;
    Assign* cand = nullptr;
    for (auto& a : *chosen) {
      if (a.sub != slow || !a.c->crossover ||
          a.c->op->kind != WEIGHT_GEMM || a.frac <= 1e-9)
        continue;
      if (!cand || a.c->t_est(slow) * a.frac >
                       cand->c->t_est(slow) * cand->frac)
        cand = &a;
    }
    if (!cand) return false;
    double avail = cand->frac;
    double t_slow = cand->c->t_est(slow) * avail;
    double t_fast = cand->c->t_est(fast) * avail;
    if (t_slow <= 0.0 || t_fast <= 0.0) return false;
    double y = std::min(1.0, std::max(0.0, gap / (t_slow + t_fast)));
    if (y <= 1e-9) return false;
    qe[slow] -= y * t_slow;
    qe[fast] += y * t_fast;
    q[slow] -= y * avail * cand->c->t_on(slow);
    q[fast] += y * avail * cand->c->t_on(fast);
    by[slow] -= y * avail * cand->c->b_on(slow);
    by[fast] += y * avail * cand->c->b_on(fast);
    OpCost* c = cand->c;
    cand->frac = avail * (1.0 - y);
    chosen->push_back({c, fast, avail * y});
    return true;
  }

  void energy(SysResult* r, const Workload& w, const double q[2],
              const double by[2], const std::vector<Assign>& chosen,
              const int modes[2], int layers, int n_tp, int groups,
              int devices, double shard, double elem_flops,
              double t_ref) const {
    (void)modes;
    const ConfigNode& en = cfg_.at("energy");
    double e_ext = mem_.at("energy_pj_per_bit").get_double("external") * 8e-12;
    double e_int = mem_.at("energy_pj_per_bit").get_double("near_bank") *
                   8e-12;
    const ConfigNode& cmd = mem_.at("command_energy_pj");
    double scale = static_cast<double>(layers) * n_tp * groups;
    double pim_flops = 0.0, bcast_bytes = 0.0;
    for (const auto& a : chosen) {
      if (a.sub != 1) continue;
      pim_flops += a.c->op->flops * a.frac * shard * scale;
      if (a.c->mode == 1) bcast_bytes += a.c->b_on(1) * a.frac * scale;
    }
    double xpu_bytes = by[0] * scale;
    double pim_bytes = by[1] * scale;
    double sfu_flops = elem_flops * shard * scale;
    double burst = mem_.get_double("io_width") *
                   mem_.get_double("burst_length") / 8.0;
    double mac_issue_flops = 2.0 * mem_.get_double("pe_lanes");
    double link_bytes =
        (fabric_.allreduce_bytes(w.activation_bytes(), n_tp, tp_coll_) *
             layers +
         (groups > 1 ? w.activation_bytes() : 0.0)) * devices;

    double duty = t_ref > 0 ? std::min(q[0] * layers / t_ref, 1.0) : 0.0;
    double bg = capacity_ * devices / GB *
                en.get_double("background_w_per_gb") *
                en.get_double("background_idle_factor");
    double p = devices * (en.get_double("xpu_busy_w") * duty +
                          en.get_double("xpu_idle_w") * (1.0 - duty)) +
               en.get_double("static_w") + bg +
               en.get_double("link_w_per_device", 0.0) * devices;

    double pim_mac_j = pim_flops / mac_issue_flops * cmd.get_double("mac_op") *
                       1e-12;
    double bcast_j = bcast_bytes / burst *
                     cmd.get_double("bg_broadcast", 0.0) * 1e-12;
    double sfu_j = sfu_flops / mac_issue_flops * cmd.get_double("mac_op") *
                   1e-12;
    double dram_j = xpu_bytes * e_ext + pim_bytes * e_int;
    double link_j = link_bytes * en.get_double("link_pj_per_bit", 2.0) *
                    8e-12;
    double act_j = pim_mac_j + bcast_j + sfu_j + link_j;
    r->power_w = p + act_j / std::max(r->t_iter_ms * 1e-3, 1e-12);
    r->tokens_per_joule =
        1.0 / (p / r->throughput + (dram_j + act_j) / w.batch);
  }

  ConfigNode mem_;
  Fabric fabric_;
  int banks_ = 0;
  double fanout_ = 4.0;
  double internal_peak_ = 0.0, internal_bw_ = 0.0, broadcast_bw_ = 0.0;
  double pe_peak_ = 0.0, pe_flops_ = 0.0;
  double capacity_ = 0.0, ext_peak_ = 0.0, xpu_bw_ = 0.0, xpu_flops_ = 0.0;
  double theta_ = 32.0, ai_pim_ = 2.0, ai_xpu_ = 312.0;
  double hysteresis_ = 1.0;
  bool adaptive_ = true;
  double k_cmd_ = 0.80, k_sync_ = 0.95, f_unfused_ = 0.26;
  double k_slice_ = 1.0, k_token_ = 1.0;
  double coexec_[2] = {0.0, 0.0};   // 0 = direct, 1 = broadcast
  int group_size_ = 8, tp_coll_ = 2;
};

// ---------------------------------------------------------------- //

std::unique_ptr<SystemModel> build_system(const std::string& root,
                                          const std::string& name,
                                          double theta_override) {
  ConfigNode c = ConfigNode::load_file(root + "/systems/" + name + ".yaml");
  std::string kind = c.get_string("kind");
  if (kind == "gpu") return std::make_unique<GpuSystem>(root, name);
  if (kind == "cxl_pnm") return std::make_unique<CxlPnmSystem>(root, name);
  if (kind == "cent") return std::make_unique<CentSystem>(root, name);
  if (kind == "neupims") return std::make_unique<NeuPimsSystem>(root, name);
  if (kind == "papi") return std::make_unique<PapiSystem>(root, name);
  if (kind == "pimphony")
    return std::make_unique<PimphonySystem>(root, name);
  if (kind == "proteus")
    return std::make_unique<ProteusSystem>(root, name, theta_override);
  throw std::runtime_error("unknown system kind: " + kind);
}

}  // namespace syscore
}  // namespace pimcore
