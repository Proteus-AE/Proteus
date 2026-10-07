// PimCore unit and behavior tests (dependency-free assert harness).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#include "pimcore/channel.hpp"
#include "pimcore/config.hpp"
#include "pimcore/kernels.hpp"
#include "pimcore/power.hpp"
#include "pimcore/sim.hpp"
#include "pimcore/trace.hpp"
#include "pimcore/address.hpp"
#include "pimcore/device.hpp"
#include "pimcore/serving.hpp"
#include "pimcore/syscore.hpp"

using namespace pimcore;

static int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                              \
  do {                                                                     \
    double _a = (a), _b = (b);                                             \
    if (std::fabs(_a - _b) > (tol) * std::max(std::fabs(_b), 1e-12)) {     \
      std::printf("FAIL %s:%d: %s=%.4g vs %s=%.4g\n", __FILE__, __LINE__,  \
                  #a, _a, #b, _b);                                         \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

// The tests run from the build directory, so the repository configuration
// tree sits two levels up. Every substrate is read from the same files the
// Python layer consumes.
static const char* kConfigRoot = "../../configs";

static ConfigNode memory_config(const std::string& name) {
  return load_config(kConfigRoot, "memory", name, "--config");
}
static TimingParams lp_timing() {
  return TimingParams::from_config(memory_config("lpddr5x-8533"));
}
static Geometry lp_geom() {
  return Geometry::from_config(memory_config("lpddr5x-8533"));
}

// ---------------------------------------------------------------- //
static void test_config_parser() {
  ConfigNode n = ConfigNode::parse(
      "a: 1\n"
      "b:\n"
      "  c: 2.5   # comment\n"
      "  d: hello\n"
      "  e:\n"
      "    f: true\n");
  CHECK(n.get_int("a") == 1);
  CHECK_NEAR(n.at("b").get_double("c"), 2.5, 1e-12);
  CHECK(n.at("b").get_string("d") == "hello");
  CHECK(n.at("b").at("e").at("f").as_bool());
  CHECK(n.at("b").get_double("missing", 7.0) == 7.0);
}

// ---------------------------------------------------------------- //
static void test_timing_validation() {
  TimingParams t = lp_timing();
  t.validate();                       // must not throw
  bool threw = false;
  t.tRC = 1.0;                        // violates tRC >= tRAS + tRP
  try {
    t.validate();
  } catch (...) {
    threw = true;
  }
  CHECK(threw);
}

// ---------------------------------------------------------------- //
static void test_streaming_efficiency() {
  // Sustained all-bank streaming efficiency. Under lockstep all-bank
  // execution the row activation is fully exposed once per row -- the banks
  // cannot hide each other's tRCD/tRP -- only the part of tRTP that outlasts
  // a column cycle is exposed, and all-bank refresh removes tRFCab out of
  // every tREFI, giving
  //   stream / (tRCD + stream + max(0, tRTP - t_col) + tRP) * (1 - tRFC/tREFI)
  // where stream = row_bytes/burst * t_col. Broadcasting streams the same
  // row at twice the cadence and amortizes the activation over twice the
  // window, so it reaches a higher fraction of a lower peak. The analytical
  // layer evaluates the same closed form (proteus_sim/memory.py).
  KernelParams kp;
  kp.rows_per_bank = 96;
  Channel ch(lp_timing(), lp_geom(), ConnectivityMode::DIRECT);
  const ChannelStats& s = ch.execute(gemv_kernel(kp, lp_timing()));
  double eff = s.sustained_pim_bw() / ch.peak_internal_bw();
  CHECK_NEAR(eff, 0.67, 0.04);
  CHECK(s.row_hit_rate() > 0.999);
  CHECK(s.n_refresh > 0);

  KernelParams kb;
  kb.rows_per_bank = 96;
  kb.n_vectors = 4;
  kb.mode = ConnectivityMode::BROADCAST;
  Channel cb(lp_timing(), lp_geom(), ConnectivityMode::BROADCAST);
  const ChannelStats& sb = cb.execute(skinny_gemm_kernel(kb, lp_timing()));
  CHECK_NEAR(sb.sustained_pim_bw() / cb.peak_internal_bw(), 0.77, 0.04);
}

// ---------------------------------------------------------------- //
static void test_broadcast_reuse() {
  // Every PE assembles one burst from every bank of its group into a FIFO
  // whose depth matches the BG fan-in, and a 4:1 selector drains them into
  // the MAC array one issue at a time, so the bank column cadence matches the
  // fan-in. A skinny-GEMM of n vectors costs n passes in direct mode against
  // ceil(n/fanout) at the broadcasting cadence; the reuse width bounds the
  // benefit, so two vectors gain nothing.
  TimingParams tp = lp_timing();
  for (int n : {2, 4, 8, 16}) {
    KernelParams kd;
    kd.rows_per_bank = 32;
    kd.n_vectors = n;
    kd.mode = ConnectivityMode::DIRECT;
    Channel cd(tp, lp_geom(), ConnectivityMode::DIRECT);
    const ChannelStats sd = cd.execute(skinny_gemm_kernel(kd, tp));

    KernelParams kb = kd;
    kb.mode = ConnectivityMode::BROADCAST;
    Channel cb(tp, lp_geom(), ConnectivityMode::BROADCAST);
    const ChannelStats sb = cb.execute(skinny_gemm_kernel(kb, tp));

    // direct: n passes at the direct column rate; broadcast: ceil(n/4)
    // passes at the fan-in cadence, at the higher streaming efficiency that
    // cadence buys by amortizing the same row activation over a longer
    // window.
    double eff_d = sd.sustained_pim_bw() / cd.peak_internal_bw();
    double eff_b = sb.sustained_pim_bw() / cb.peak_internal_bw();
    double passes_b = std::ceil(n / 4.0);
    double expect = n / (2.0 * passes_b) * (eff_b / eff_d);
    CHECK_NEAR(sd.time_ns / sb.time_ns, expect, 0.05);
  }
}

// ---------------------------------------------------------------- //
static void test_mac_accounting() {
  TimingParams tp = lp_timing();
  KernelParams kp;
  kp.rows_per_bank = 16;
  kp.n_vectors = 4;
  kp.mode = ConnectivityMode::BROADCAST;
  Channel ch(tp, lp_geom(), ConnectivityMode::BROADCAST);
  const ChannelStats& s = ch.execute(skinny_gemm_kernel(kp, tp));
  CHECK(s.n_mac == s.n_rd_burst * 4);   // 4-way fan-out
}

// ---------------------------------------------------------------- //
static void test_host_coexecution() {
  // A bank serves either its local PE or the channel's global I/O in a given
  // column cycle. Direct mode drives every bank at its minimum column cycle
  // and therefore returns no memory-service slots to the host; the
  // broadcasting cadence frees enough of them to saturate the external
  // interface. This is the measurement behind the co-execution constant of
  // the analytical layer (configs/systems/proteus.yaml, `coexec`).
  TimingParams tp = lp_timing();
  KernelParams kp;
  kp.rows_per_bank = 48;
  kp.n_vectors = 8;

  kp.mode = ConnectivityMode::DIRECT;
  Channel cd(tp, lp_geom(), ConnectivityMode::DIRECT);
  HostStreamConfig host;
  host.enabled = true;
  cd.attach_host_stream(host);
  const ChannelStats sd = cd.execute(skinny_gemm_kernel(kp, tp));

  kp.mode = ConnectivityMode::BROADCAST;
  Channel cb(tp, lp_geom(), ConnectivityMode::BROADCAST);
  cb.attach_host_stream(host);
  const ChannelStats sb = cb.execute(skinny_gemm_kernel(kp, tp));

  CHECK(sd.host_bytes == 0);            // direct mode leaves no free slot
  CHECK(sb.host_bytes > 0);             // broadcasting does
  CHECK(sb.host_latency.samples() > 0);
  // iso-work (8 vectors): direct needs 8 passes at the direct column rate,
  // broadcasting 2 passes at the fan-in cadence and a higher streaming
  // efficiency
  double eff_d = sd.sustained_pim_bw() / cd.peak_internal_bw();
  double eff_b = sb.sustained_pim_bw() / cb.peak_internal_bw();
  CHECK_NEAR(sd.time_ns / sb.time_ns, 2.0 * (eff_b / eff_d), 0.05);
  // the freed slots must cover the channel's share of a 1 TB/s external
  // interface (1 TB/s / 64 channels = 15.6 GB/s)
  double host_gbps = sb.host_bytes / (sb.time_ns * 1e-9) / 1e9;
  CHECK(host_gbps > 6.0);
}

// ---------------------------------------------------------------- //
static void test_kv_append() {
  TimingParams tp = lp_timing();
  KernelParams kp;
  kp.rows_per_bank = 8;
  kp.group_size = 8;
  kp.mode = ConnectivityMode::BROADCAST;
  kp.kv_append = true;
  Channel ch(tp, lp_geom(), ConnectivityMode::BROADCAST);
  const ChannelStats& s = ch.execute(attention_kernel(kp, tp));
  const uint64_t banks = static_cast<uint64_t>(lp_geom().banks());
  CHECK(s.n_wr_burst == banks);         // one all-bank write burst set
  CHECK(s.pim_bytes_written == banks * 32u);
}

// ---------------------------------------------------------------- //
static void test_mode_switch_semantics() {
  TimingParams tp = lp_timing();
  std::vector<Command> stream;
  Command m1; m1.kind = CommandKind::MODE; m1.mode = ConnectivityMode::BROADCAST;
  Command m2; m2.kind = CommandKind::MODE; m2.mode = ConnectivityMode::DIRECT;
  stream.push_back(m1);
  auto pass = stream_pass(2, tp.row_bytes / tp.burst_bytes, 0);
  stream.insert(stream.end(), pass.begin(), pass.end());
  stream.push_back(m2);
  pass = stream_pass(2, tp.row_bytes / tp.burst_bytes, 0);
  stream.insert(stream.end(), pass.begin(), pass.end());

  Channel ch(tp, lp_geom(), ConnectivityMode::DIRECT);
  const ChannelStats& s = ch.execute(stream);
  CHECK(s.n_mode_switch == 2);
  CHECK(ch.mode() == ConnectivityMode::DIRECT);
}

// ---------------------------------------------------------------- //
static void test_trace_roundtrip() {
  TimingParams tp = lp_timing();
  KernelParams kp;
  kp.rows_per_bank = 4;
  kp.n_vectors = 8;
  kp.mode = ConnectivityMode::BROADCAST;
  auto cmds = skinny_gemm_kernel(kp, tp);
  const std::string path = "/tmp/pimcore_roundtrip.trace";
  write_trace(path, cmds);
  auto back = read_trace(path);
  CHECK(back.size() == cmds.size());
  Channel c1(tp, lp_geom(), ConnectivityMode::DIRECT);
  Channel c2(tp, lp_geom(), ConnectivityMode::DIRECT);
  double t1 = c1.execute(cmds).time_ns;
  double t2 = c2.execute(back).time_ns;
  CHECK_NEAR(t1, t2, 1e-9);
}

// ---------------------------------------------------------------- //
static void test_energy_split() {
  TimingParams tp = lp_timing();
  EnergyTable et = EnergyTable::from_config(memory_config("lpddr5x-8533"));
  KernelParams kp;
  kp.rows_per_bank = 64;
  Channel ch(tp, lp_geom(), ConnectivityMode::DIRECT);
  const ChannelStats& s = ch.execute(gemv_kernel(kp, tp));
  PowerModel pm(et);
  uint64_t bytes = s.pim_bytes_read + s.pim_bytes_written;
  double near = pm.account(s, false).pj_per_bit(bytes);
  double ext = pm.account(s, true).pj_per_bit(bytes);
  CHECK(near > 1.5 && near < 2.6);      // near-bank ~2.2 pJ/bit
  CHECK(ext > 4.0 && ext < 5.2);        // external ~4.5 pJ/bit
  CHECK(ext > near);
}

// ---------------------------------------------------------------- //
static void test_substrate_tables() {
  for (const char* name : {"lpddr5x-8533", "hbm-pim", "gddr6-aim"}) {
    ConfigNode mem = memory_config(name);
    TimingParams t = TimingParams::from_config(mem);
    t.validate();
    Geometry g = Geometry::from_config(mem);
    CHECK(g.banks() > 0);
    Channel ch(t, g, ConnectivityMode::DIRECT);
    KernelParams kp;
    kp.rows_per_bank = 16;
    const ChannelStats& st = ch.execute(gemv_kernel(kp, t));
    CHECK(st.time_ns > 0.0);
    CHECK(st.sustained_pim_bw() < ch.peak_internal_bw() * 1.0001);
  }
}

// ---------------------------------------------------------------- //
static void test_address_mapper() {
  Geometry g = lp_geom();
  TimingParams t = lp_timing();
  AddressMapper m(g, t);
  // encode/decode round trip over a spread of addresses
  for (addr_t a = 0; a < 100000; a += 7919) {
    Coordinates c = m.decode(a);
    CHECK(m.encode(c) == a);
    CHECK(c.flat_bank(g) >= 0 && c.flat_bank(g) < g.banks());
    CHECK(c.col < m.bursts_per_row());
  }
  // consecutive requests spread across bank groups before rows change
  Coordinates c0 = m.decode(0), c1 = m.decode(2);
  CHECK(c0.row == c1.row);
}

// ---------------------------------------------------------------- //
static void test_host_controller_frfcfs() {
  Geometry g = lp_geom();
  TimingParams t = lp_timing();
  HostController ctrl(t, g, ArbitrationPolicy::PIM_PRIORITY);
  AddressMapper m(g, t);
  Coordinates hot;   // a request whose row we will declare open
  hot.row = 5;
  Coordinates cold = hot;
  cold.row = 9;
  HostRequest r1; r1.addr = m.encode(cold); r1.arrival = 0.0;
  HostRequest r2; r2.addr = m.encode(hot);  r2.arrival = 1.0;
  ctrl.push(r1);
  ctrl.push(r2);
  // FR-FCFS: the row-hit request is selected even though it is younger.
  int idx = ctrl.select([&](const Coordinates& c) { return c.row == 5; });
  CHECK(idx == 1);
  // Without any open row, the oldest wins.
  idx = ctrl.select([&](const Coordinates&) { return false; });
  CHECK(idx == 0);
}

// ---------------------------------------------------------------- //
static void test_coexec_policies() {
  TimingParams t = lp_timing();
  Geometry g = lp_geom();
  KernelParams kp;
  kp.rows_per_bank = 24;
  kp.n_vectors = 4;
  kp.mode = ConnectivityMode::DIRECT;
  auto stream = skinny_gemm_kernel(kp, t);

  CoExecConfig base;
  base.offered_gbps = 4.0;

  base.policy = ArbitrationPolicy::PIM_PRIORITY;
  CoExecReport pimp = CoExecEngine(t, g, kp.mode, base).run(stream);
  base.policy = ArbitrationPolicy::HOST_PRIORITY;
  CoExecReport hostp = CoExecEngine(t, g, kp.mode, base).run(stream);

  CHECK(pimp.host_served > 0 && hostp.host_served > 0);
  // PIM priority protects the kernel; host priority stretches it.
  CHECK(pimp.pim_slowdown < hostp.pim_slowdown + 1e-9);
  // Host priority must not give worse mean latency than gap-stealing.
  CHECK(hostp.host_latency_mean <= pimp.host_latency_mean + 1e-9);
}

// ---------------------------------------------------------------- //
static void test_syscore_sanity() {
  using namespace pimcore::syscore;
  const std::string root = kConfigRoot;
  // batch monotonicity + OOM ordering across every system/model
  const std::vector<std::string> systems = {
      "dgx-a100", "cxl-pnm", "cent", "neupims", "papi", "pimphony",
      "proteus"};
  const std::vector<std::string> models = {
      "deepseek-v2-lite", "switch-26b", "mixtral-8x7b", "llama3-70b"};
  for (const auto& sn : systems) {
    auto sys = build_system(root, sn);
    for (const auto& mn : models) {
      ModelSpec m = ModelSpec::load(root, mn);
      double prev = 0.0;
      bool seen_oom = false;
      for (int b : {16, 32, 64}) {
        SysResult r = sys->simulate(Workload::build(m, b, 2048, 6144));
        if (!r.alive) { seen_oom = true; continue; }
        CHECK(!seen_oom);
        CHECK(r.throughput >= prev * 0.999);
        CHECK(r.tokens_per_joule > 0.0);
        prev = r.throughput;
      }
    }
  }
}

// ---------------------------------------------------------------- //
static void test_syscore_variants_monotone() {
  using namespace pimcore::syscore;
  const std::string root = kConfigRoot;
  auto sys = build_system(root, "proteus");
  for (const auto& mn : {std::string("mixtral-8x7b"),
                         std::string("llama3-70b")}) {
    ModelSpec m = ModelSpec::load(root, mn);
    Workload w = Workload::build(m, 32, 2048, 6144);
    double prev = 0.0;
    for (const auto& v : {"base", "as", "rd", "of", "ec"}) {
      SysResult r = sys->simulate_ex(w, VariantFlags::named(v), 0, 1);
      CHECK(r.alive && r.throughput >= prev * 0.999);
      prev = r.throughput;
    }
  }
}

// ---------------------------------------------------------------- //
static void test_syscore_scaling() {
  using namespace pimcore::syscore;
  const std::string root = kConfigRoot;
  auto sys = build_system(root, "proteus");
  ModelSpec m = ModelSpec::load(root, "llama3-70b");
  Workload w = Workload::build(m, 32, 2048, 6144);
  SysResult r1 = sys->simulate_ex(w, VariantFlags{}, 1, 1);
  SysResult r16 = sys->simulate_ex(w, VariantFlags{}, 16, 1);
  // 1 device is a degenerate group (no tensor parallelism, no collective);
  // 16 devices form two tensor-parallel groups of eight pipelined over the
  // layers, so the per-device work drops 16-fold and the collectives and
  // stage transfers claim a few percent back.
  double s = r16.throughput / r1.throughput;
  CHECK(s > 12.0 && s <= 16.0);
  CHECK(r1.tp_width == 1 && r1.pipeline_groups == 1);
  CHECK(r16.tp_width == 8 && r16.pipeline_groups == 2);
  CHECK(r16.layers_per_stage * r16.pipeline_groups >= m.n_layers);
  // DP monotone degradation at fixed total batch
  double prev = 1e18;
  for (int dp : {1, 2, 4, 8}) {
    SysResult r = sys->simulate_ex(w, VariantFlags{}, 8, dp);
    CHECK(r.alive && r.throughput <= prev * 1.001);
    prev = r.throughput;
  }
}

static void test_host_only_mode() {
  // Host-only traffic: sequential beats random (row locality), both under
  // the bus ceiling of 32 B / tCCD_S.
  TimingParams tp = lp_timing();
  Geometry g = lp_geom();
  double peak = 32.0 / tp.tCCD_S * 1e9;
  double bw[2];
  TrafficPattern pats[2] = {TrafficPattern::STREAM, TrafficPattern::RANDOM};
  for (int i = 0; i < 2; ++i) {
    CoExecConfig cc;
    cc.pattern = pats[i];
    cc.offered_gbps = 20.0;
    CoExecEngine eng(tp, g, ConnectivityMode::DIRECT, cc);
    CoExecReport rep = eng.host_only(200e3);
    bw[i] = rep.host_bw;
    CHECK(rep.host_served > 0);
    CHECK(rep.host_bw < peak);
    CHECK(rep.host_latency_p95 >= rep.host_latency_mean);
  }
  CHECK(bw[0] > bw[1]);   // sequential > random
}

static void test_serving_closed_loop() {
  using syscore::ModelSpec;
  ModelSpec model = ModelSpec::load(kConfigRoot, "mixtral-8x7b");
  auto sys = syscore::build_system(kConfigRoot, "proteus");
  ConfigNode cfg =
      load_config(kConfigRoot, "systems", "proteus", "--system");
  ServingConfig sc;
  sc.max_batch = 8;
  sc.prompt_mean = 1024;
  sc.out_mean = 40;
  ServingSim sim(*sys, model, cfg, sc);
  auto recs = sim.run(150);
  CHECK(recs.size() == 150);             // generator never drains
  int completed = 0;
  for (const auto& r : recs) {
    CHECK(r.batch == 8);                 // closed loop keeps the pool full
    CHECK(r.throughput > 0.0);
    CHECK(r.x_split >= 0.0 && r.x_split <= 1.0);
    CHECK(r.experts_xpu + r.experts_pim <= model.n_experts);
    completed += r.completed;
  }
  CHECK(completed > 0);                  // short outputs finish in-run

  // Replay drain: a 4-request trace with 2-token outputs ends the run.
  std::vector<std::pair<int, int>> src = {{128, 2}, {128, 2},
                                          {128, 2}, {128, 2}};
  ServingSim replay(*sys, model, cfg, sc, src);
  auto rr = replay.run(60);
  CHECK(rr.size() == 2);                 // all four decode two tokens
  CHECK(rr.back().completed == 4);
}

static void test_serving_switches_smallbatch() {
  // Small batches drop experts between iterations, so placements churn.
  using syscore::ModelSpec;
  ModelSpec model = ModelSpec::load(kConfigRoot, "mixtral-8x7b");
  auto sys = syscore::build_system(kConfigRoot, "proteus");
  ConfigNode cfg =
      load_config(kConfigRoot, "systems", "proteus", "--system");
  ServingConfig sc;
  sc.max_batch = 4;
  sc.prompt_mean = 1024;
  sc.out_mean = 512;
  ServingSim sim(*sys, model, cfg, sc);
  auto recs = sim.run(100);
  int sw = 0;
  for (const auto& r : recs) sw += r.switches;
  CHECK(sw > 0);
}

int main() {
  test_config_parser();
  test_timing_validation();
  test_streaming_efficiency();
  test_broadcast_reuse();
  test_mac_accounting();
  test_host_coexecution();
  test_kv_append();
  test_mode_switch_semantics();
  test_trace_roundtrip();
  test_energy_split();
  test_substrate_tables();
  test_address_mapper();
  test_host_controller_frfcfs();
  test_coexec_policies();
  test_syscore_sanity();
  test_syscore_variants_monotone();
  test_syscore_scaling();
  test_host_only_mode();
  test_serving_closed_loop();
  test_serving_switches_smallbatch();
  if (g_failures == 0) {
    std::printf("ALL PASS (20 test groups)\n");
    return 0;
  }
  std::printf("%d FAILURES\n", g_failures);
  return 1;
}
