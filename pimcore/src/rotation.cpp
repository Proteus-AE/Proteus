#include "pimcore/rotation.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <deque>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pimcore {

const char* schedule_name(Schedule s) {
  switch (s) {
    case Schedule::ROTATE: return "rotate";
    case Schedule::LOCKSTEP: return "lockstep";
    case Schedule::DIRECT: return "direct";
  }
  return "?";
}

namespace {

constexpr int NBG = 4;
constexpr int NBK = 4;            // banks per bank group
constexpr int NB = NBG * NBK;
constexpr double EPS = 1e-6;
constexpr double PE_CYCLE = 1.0;  // ns, PE clock 1 GHz
constexpr long HOST_ROW_BASE = 1L << 20;

inline int bank_id(int g, int k) { return g * NBK + k; }
inline int bg_of(int b) { return b / NBK; }

enum Kind { K_ACT, K_RD, K_PRE };

struct Cmd {
  double t;
  int bank;
  Kind kind;
  bool pim;
  long row;
  bool exempt;   // all-bank activation (LOCKSTEP / DIRECT reference schedules)
};

struct Turn {        // one PIM row turn of one bank
  double act = 0, first = 0, last = 0, pre = 0;
  int n = 0;         // bursts
  long row = 0;
};

struct Window {      // host may ACT at >= w0 and must PRE at <= w1
  double w0, w1;
};

struct Interval {
  double a, b;
};

// A time-slotted resource (C/A bus, channel I/O). PIM reservations are fixed
// in advance; host allocations are placed into the remaining time.
struct Timeline {
  std::vector<Interval> fixed;        // PIM reservations
  std::vector<Interval> host;         // host allocations
  std::map<double, double> busy;      // start -> end, non-overlapping

  void finalize() {
    std::sort(fixed.begin(), fixed.end(),
              [](const Interval& x, const Interval& y) { return x.a < y.a; });
    for (const Interval& iv : fixed) {
      double a = earliest(iv.a, iv.b - iv.a);   // shift a colliding PIM slot
      busy[a] = a + (iv.b - iv.a);
    }
    fixed.clear();
    for (auto& kv : busy) fixed.push_back({kv.first, kv.second});
  }
  // earliest x >= t with [x, x+d) free
  double earliest(double t, double d) const {
    double x = t;
    auto it = busy.upper_bound(x);
    if (it != busy.begin()) {
      auto p = std::prev(it);
      if (p->second > x + EPS) x = p->second;
    }
    while (it != busy.end() && it->first < x + d - EPS) {
      x = std::max(x, it->second);
      ++it;
    }
    return x;
  }
  // latest x in [lo, t] with [x, x+d) free; -1 if none
  double latest(double t, double d, double lo) const {
    double x = t;
    for (int guard = 0; guard < 4096; ++guard) {
      if (x < lo - EPS) return -1.0;
      auto it = busy.lower_bound(x + d - EPS);
      if (it == busy.begin()) return x;
      auto p = std::prev(it);
      if (p->second <= x + EPS) return x;
      x = p->first - d;
    }
    return -1.0;
  }
  void take(double a, double d) {
    busy[a] = a + d;
    host.push_back({a, a + d});
  }
  void release(double a) {
    busy.erase(a);
    for (size_t i = host.size(); i-- > 0;)
      if (std::fabs(host[i].a - a) < 1e-9) { host.erase(host.begin() + i); break; }
  }
};

// tRRD_S / tRRD_L / tFAW over all activations of the die.
struct ActTracker {
  std::multimap<double, int> acts;   // time -> bank group
  double tRRD_S, tRRD_L, tFAW;

  bool ok(double t, int bg) const {
    auto lo = acts.lower_bound(t - tFAW + EPS);
    auto hi = acts.upper_bound(t + tFAW - EPS);
    std::vector<double> win;
    for (auto it = lo; it != hi; ++it) {
      double d = std::fabs(t - it->first);
      double need = (it->second == bg) ? tRRD_L : tRRD_S;
      if (d < need - EPS) return false;
      win.push_back(it->first);
    }
    win.push_back(t);
    std::sort(win.begin(), win.end());
    // any 5 activations must span at least tFAW
    for (size_t i = 0; i + 4 < win.size(); ++i)
      if (win[i + 4] - win[i] < tFAW - EPS) return false;
    return true;
  }
  // earliest legal time in [t, limit], scanning at 0.125 ns
  double earliest(double t, int bg, double limit) const {
    for (double x = t; x <= limit + EPS; x += 0.125)
      if (ok(x, bg)) return x;
    return -1.0;
  }
  void add(double t, int bg) { acts.emplace(t, bg); }
};

struct Read {
  double t;
  int bank;
  int turn;
};

// ---------------------------------------------------------------------------
// PIM schedule generation
// ---------------------------------------------------------------------------

struct Segment {
  double s, e;   // activity allowed in [s, e); e = next refresh start
};

std::vector<Segment> make_segments(const TimingParams& t, bool refresh,
                                   double horizon, std::vector<double>* refs) {
  std::vector<Segment> segs;
  if (!refresh) {
    segs.push_back({0.0, horizon});
    return segs;
  }
  double s = 0.0;
  double r = 0.5 * t.tREFI;   // refresh phase relative to the kernel start
  while (s < horizon) {
    segs.push_back({s, r});
    refs->push_back(r);
    s = r + t.tRFCab;
    r += t.tREFI;
  }
  return segs;
}

struct PimSchedule {
  std::vector<std::vector<Turn>> turns;   // per bank
  std::vector<std::vector<double>> reads; // per bank, read times
  std::vector<Cmd> cmds;
  double first_act = 0, last_read = 0;
  long bursts = 0;
};

// Rotating broadcast schedule. Per segment and bank group: bank 0 starts a
// full turn, bank 3 a half turn (fill), banks 1 and 2 follow at +64 / +128 ns
// (+1 ns parity for odd banks); afterwards each bank alternates 128 ns turns
// and 128 ns rests. Reads are truncated so that every row is precharged
// before the next refresh, and each bank group stops once it has streamed
// its share of the kernel.
PimSchedule rotate_schedule(const TimingParams& t, const RotationConfig& c,
                            const std::vector<Segment>& segs) {
  const int row_bursts = t.row_bytes / t.burst_bytes;   // 64
  const double half = row_bursts * t.tCCD_PIM / 2.0;    // 64 ns
  const double period = NBK * half;                      // 256 ns
  const long target = static_cast<long>(c.rows_per_bank) * NBK * row_bursts;
  PimSchedule ps;
  ps.turns.assign(NB, {});
  ps.reads.assign(NB, {});

  for (int g = 0; g < NBG; ++g) {
    std::vector<Read> cand;
    int turn_id = 0;
    std::vector<int> turn_bank;
    for (const Segment& sg : segs) {
      double goff = c.bg_offsets.size() == NBG ? c.bg_offsets[g] : g * c.bg_offset_ns;
      double p = sg.s + goff;                          // first ACT of bank 0
      double r0 = p + t.tRCD;                          // first read of bank 0
      double last_ok = sg.e - t.tRP - t.tRTP;          // last read in segment
      if (r0 > last_ok) continue;
      // turn start offsets of the regular rotation
      double off[NBK];
      for (int k = 0; k < NBK; ++k)
        off[k] = k * half + (c.bank_phase.size() == NBK ? c.bank_phase[k]
                                                         : (k % 2) * PE_CYCLE);
      // fill: half turn of bank 3 so two banks stream from the start; its
      // activation follows bank 0's by tRRD_L, its reads keep odd parity
      {
        double a3 = p + std::ceil(t.tRRD_L);
        double f3 = a3 + t.tRCD;
        if (std::fmod(f3 - r0, 2.0) < 0.5) f3 += 1.0;
        int id = turn_id++;
        turn_bank.push_back(3);
        for (double x = f3; x < r0 + off[1] - 1.0 + EPS && x <= last_ok + EPS;
             x += t.tCCD_PIM)
          cand.push_back({x, bank_id(g, 3), id});
      }
      for (int i = 0;; ++i) {
        bool any = false;
        for (int k = 0; k < NBK; ++k) {
          double f = r0 + off[k] + i * period;
          if (f > last_ok) continue;
          any = true;
          int id = turn_id++;
          turn_bank.push_back(k);
          for (int j = 0; j < row_bursts; ++j) {
            double x = f + j * t.tCCD_PIM;
            if (x > last_ok + EPS) break;
            cand.push_back({x, bank_id(g, k), id});
          }
        }
        if (!any) break;
        // stop generating once far beyond what the kernel needs
        if (static_cast<long>(cand.size()) > 3 * target + 4096) break;
      }
      if (static_cast<long>(cand.size()) > 3 * target + 4096) break;
    }
    std::sort(cand.begin(), cand.end(),
              [](const Read& a, const Read& b) { return a.t < b.t; });
    if (static_cast<long>(cand.size()) < target)
      throw std::runtime_error("rotate_schedule: horizon too short");
    cand.resize(target);
    // rebuild turns from the kept reads
    std::map<int, Turn> by_turn;
    for (const Read& r : cand) {
      auto it = by_turn.find(r.turn);
      if (it == by_turn.end()) {
        Turn tr;
        tr.first = tr.last = r.t;
        tr.n = 1;
        by_turn[r.turn] = tr;
      } else {
        it->second.last = std::max(it->second.last, r.t);
        it->second.first = std::min(it->second.first, r.t);
        it->second.n += 1;
      }
      ps.reads[r.bank].push_back(r.t);
    }
    for (auto& kv : by_turn) {
      int b = bank_id(g, turn_bank[kv.first]);
      Turn tr = kv.second;
      tr.act = tr.first - t.tRCD;
      tr.pre = std::max(tr.last + t.tRTP, tr.act + t.tRAS);
      tr.row = static_cast<long>(ps.turns[b].size());
      ps.turns[b].push_back(tr);
    }
  }
  return ps;
}

// All-bank reference schedules. LOCKSTEP: all four banks of a group stream
// at the fan-in cadence (4 ns), staggered by one PE cycle on the broadcast
// line. DIRECT: every bank streams at tCCD_PIM to its own PE.
PimSchedule allbank_schedule(const TimingParams& t, const RotationConfig& c,
                             const std::vector<Segment>& segs) {
  const int row_bursts = t.row_bytes / t.burst_bytes;
  const bool lock = c.schedule == Schedule::LOCKSTEP;
  const double cad = lock ? std::max(t.tCCD_L, 4.0 * PE_CYCLE) : t.tCCD_PIM;
  PimSchedule ps;
  ps.turns.assign(NB, {});
  ps.reads.assign(NB, {});
  for (int b = 0; b < NB; ++b) {
    int k = b % NBK;
    int rows = 0;
    for (const Segment& sg : segs) {
      double act = sg.s;
      while (rows < c.rows_per_bank) {
        double first = act + t.tRCD + (lock ? k * PE_CYCLE : 0.0);
        double last = first + (row_bursts - 1) * cad;
        double pre = last + t.tRTP;
        if (pre + t.tRP > sg.e + EPS) break;
        Turn tr{act, first, last, pre, row_bursts, rows};
        ps.turns[b].push_back(tr);
        for (int j = 0; j < row_bursts; ++j)
          ps.reads[b].push_back(first + j * cad);
        ++rows;
        // the next all-bank ACT waits for the slowest bank of the group
        double slow_pre = act + t.tRCD + (lock ? (NBK - 1) * PE_CYCLE : 0.0) +
                          (row_bursts - 1) * cad + t.tRTP;
        act = std::max(slow_pre + t.tRP, act + t.tRC);
      }
      if (rows >= c.rows_per_bank) break;
    }
    if (rows < c.rows_per_bank)
      throw std::runtime_error("allbank_schedule: horizon too short");
  }
  return ps;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double idx = p * (v.size() - 1);
  size_t i = static_cast<size_t>(idx);
  double f = idx - i;
  if (i + 1 < v.size()) return v[i] * (1 - f) + v[i + 1] * f;
  return v[i];
}

}  // namespace

// ---------------------------------------------------------------------------

RotationReport run_rotation(const TimingParams& t, const RotationConfig& c) {
  RotationReport rep;
  const int row_bursts = t.row_bytes / t.burst_bytes;
  // A BL16 burst occupies the DQ (and the channel's global I/O) for
  // burst_ns at the LPDDR5X pin rate; the device's external interface caps
  // the average rate per channel at ext_bw x bus efficiency (token bucket).
  const double dq_ns = t.burst_ns;
  double ext_rate = c.ext_bw_gbps * c.ext_bus_eff;       // B/ns
  const double bucket_cap = (std::getenv("PIMCORE_BUCKET") ? std::atof(std::getenv("PIMCORE_BUCKET")) : 4.0) * t.burst_bytes;
  const bool noca = std::getenv("PIMCORE_NOCA") != nullptr;   // analysis only
  // one command per CK cycle on C/A (CK = WCK/4; two CK per BL16 burst)
  const double CMD_CA = noca ? 1e-3 : t.burst_ns / 2.0;

  // ---- refresh segments and PIM schedule ----------------------------------
  double horizon = c.duration_ns > 0 ? c.duration_ns
                                     : 2.0 * c.rows_per_bank * 300.0 + 20000.0;
  std::vector<double> refs;
  std::vector<Segment> segs = make_segments(t, c.refresh, horizon * 3, &refs);

  PimSchedule ps;
  ps.turns.assign(NB, {});
  ps.reads.assign(NB, {});
  if (c.pim) {
    ps = (c.schedule == Schedule::ROTATE) ? rotate_schedule(t, c, segs)
                                          : allbank_schedule(t, c, segs);
  }
  const bool exempt = c.schedule != Schedule::ROTATE;

  // PIM span
  double pim_first = 1e18, pim_last = 0.0;
  long pim_bursts = 0;
  for (int b = 0; b < NB; ++b) {
    for (const Turn& tr : ps.turns[b]) {
      pim_first = std::min(pim_first, tr.act);
      pim_last = std::max(pim_last, tr.last);
    }
    pim_bursts += static_cast<long>(ps.reads[b].size());
  }
  if (!c.pim) { pim_first = 0.0; pim_last = horizon; }
  // the kernel ends when the last burst has been consumed by the PEs
  double span_end = c.pim ? pim_last + c.data_latency_ns + PE_CYCLE : horizon;
  if (c.pim && c.duration_ns > 0) span_end = std::max(span_end, c.duration_ns);
  const double span0 = c.pim ? pim_first : 0.0;

  // ---- fixed (PIM) reservations: activations, C/A, I/O --------------------
  ActTracker acts{{}, t.tRRD_S, t.tRRD_L, t.tFAW};
  Timeline ca, io;
  std::vector<Cmd> log;
  for (int b = 0; b < NB; ++b) {
    for (const Turn& tr : ps.turns[b]) {
      acts.add(tr.act, bg_of(b));
      log.push_back({tr.act, b, K_ACT, true, tr.row, exempt});
      for (double x : ps.reads[b])
        if (x >= tr.first - EPS && x <= tr.last + EPS)
          log.push_back({x, b, K_RD, true, tr.row, exempt});
      log.push_back({tr.pre, b, K_PRE, true, tr.row, exempt});
    }
  }
  if (c.pim && c.schedule == Schedule::ROTATE) {
    // The controller enqueues streaming commands into the per-channel PIM
    // command queue, which sequences the per-bank ACT/RD/PRE of the rotation
    // on die. Conservatively, one C/A command per rotation period (256 ns).
    double period = NBK * (row_bursts * t.tCCD_PIM / 2.0);
    for (double x = pim_first - 4.0; x <= pim_last; x += period)
      ca.fixed.push_back({x, x + CMD_CA});
  }
  if (c.pim && c.schedule != Schedule::ROTATE) {
    // all-bank ACT / column / PRE commands issued over C/A once per channel
    std::vector<double> ab;
    for (const Turn& tr : ps.turns[0]) {
      ab.push_back(tr.act);
      ab.push_back(tr.pre);
    }
    for (double x : ab) ca.fixed.push_back({x, x + CMD_CA});
    double cad = (c.schedule == Schedule::LOCKSTEP) ? 4.0 : t.tCCD_PIM;
    for (const Turn& tr : ps.turns[0])
      for (double x = tr.first; x <= tr.last + EPS; x += cad)
        ca.fixed.push_back({x - 1.0, x});
  }
  // merge overlapping C/A reservations (two PIM commands cannot share a slot;
  // shift the later one)
  std::sort(ca.fixed.begin(), ca.fixed.end(),
            [](const Interval& x, const Interval& y) { return x.a < y.a; });
  for (size_t i = 1; i < ca.fixed.size(); ++i)
    if (ca.fixed[i].a < ca.fixed[i - 1].b - EPS) {
      double d = ca.fixed[i - 1].b - ca.fixed[i].a;
      ca.fixed[i].a += d;
      ca.fixed[i].b += d;
    }

  // PIM I/O on the normal I/O path: input vectors written over DQ into the
  // PE vector registers (double-buffered, prefetched ahead of use) and
  // results/scores drained on die to the channel SFU (through its input
  // buffer). Both are transfer jobs with a release time and a deadline that
  // the controller schedules alongside host requests (earliest deadline
  // first when urgent); a missed deadline would stall the PIM stream.
  struct IoJob {
    double rel, dl;
    int bursts;
    bool ext;      // over DQ (vector load) or on die (to/from the SFU)
    bool done;
  };
  std::vector<IoJob> jobs;
  if (c.pim && (c.pim_in_ratio > 0 || c.pim_out_ratio > 0)) {
    std::vector<double> all_reads;
    for (int b = 0; b < NB; ++b)
      all_reads.insert(all_reads.end(), ps.reads[b].begin(), ps.reads[b].end());
    std::sort(all_reads.begin(), all_reads.end());
    const double batch_bytes = c.pim_io_batch * t.burst_bytes;
    double acc_in = 0, acc_out = 0;   // the first batch is preloaded
    for (double x : all_reads) {
      acc_in += c.pim_in_ratio * t.burst_bytes;
      acc_out += c.pim_out_ratio * t.burst_bytes;
      while (acc_in >= batch_bytes && c.pim_in_ratio > 0) {
        jobs.push_back({x - c.pim_prefetch_ns, x, c.pim_io_batch,
                        c.pim_in_external, false});
        acc_in -= batch_bytes;
      }
      while (acc_out >= batch_bytes) {
        double r = x + c.data_latency_ns;
        jobs.push_back({r, r + c.pim_out_slack_ns, c.pim_io_batch, false, false});
        acc_out -= batch_bytes;
      }
    }
    std::sort(jobs.begin(), jobs.end(),
              [](const IoJob& x, const IoJob& y) { return x.rel < y.rel; });
    for (const IoJob& j : jobs) rep.pim_io_bytes += j.bursts * t.burst_bytes;
  }
  ca.finalize();
  io.finalize();
  double vec_bytes = 0.0;
  if (c.pim && c.pim_in_external && c.pim_in_ratio > 0)
    vec_bytes = c.pim_in_ratio * pim_bursts * t.burst_bytes;
  const double kernel_ns = std::max(span_end - span0, 1.0);
  const double host_rate = std::max(ext_rate - vec_bytes / kernel_ns, 0.0);

  // ---- host windows per bank ---------------------------------------------
  // A bank may hold a host row between the precharge of one PIM turn and the
  // activation of its next, and never across a refresh.
  std::vector<std::vector<Window>> win(NB);
  for (int b = 0; b < NB; ++b) {
    std::vector<Interval> busy;   // [ACT, PRE + tRP) of PIM turns
    for (const Turn& tr : ps.turns[b]) busy.push_back({tr.act, tr.pre + t.tRP});
    for (double r : refs) busy.push_back({r - t.tRP, r + t.tRFCab});
    std::sort(busy.begin(), busy.end(),
              [](const Interval& x, const Interval& y) { return x.a < y.a; });
    double cur = 0.0;
    for (const Interval& iv : busy) {
      // host ACT >= cur, host PRE + tRP <= iv.a  ->  PRE <= iv.a - tRP
      if (iv.a - t.tRP - cur >= t.tRAS + EPS) win[b].push_back({cur, iv.a - t.tRP});
      cur = std::max(cur, iv.b);
    }
    if (span_end + 2000 > cur) win[b].push_back({cur, span_end + 4000});
    if (std::getenv("PIMCORE_WIN") && std::atoi(std::getenv("PIMCORE_WIN")) == b)
      for (const Window& w : win[b]) std::fprintf(stderr, "win b%d [%.1f, %.1f]\n", b, w.w0, w.w1);
  }

  // ---- host request stream -------------------------------------------------
  struct Req {
    double arr;
    int bank;
    long row;
    bool done;
  };
  std::deque<Req> pending;   // in arrival order
  long next_seq = 0;
  std::mt19937_64 rng(c.seed);
  double next_arrival = 0.0;
  const double lambda = c.offered_gbps / t.burst_bytes;   // bursts per ns
  std::exponential_distribution<double> expo(lambda > 0 ? lambda : 1.0);
  auto make_req = [&](double arr) {
    long n = next_seq++;
    // ch-bg-bank-col-row interleaving within the channel, in granules of
    // `granule_bursts` bursts per bank row visit
    long gnum = n / c.granule_bursts;
    int g = static_cast<int>(gnum % NBG);
    int k = static_cast<int>((gnum / NBG) % NBK);
    long row = HOST_ROW_BASE + n / (static_cast<long>(NB) * row_bursts);
    pending.push_back({arr, bank_id(g, k), row, false});
  };
  auto refill = [&](double now) {
    if (!c.host) return;
    if (c.saturate) {
      while (static_cast<int>(pending.size()) < c.window_requests)
        make_req(now);
    } else {
      while (next_arrival <= now + EPS) {
        make_req(next_arrival);
        next_arrival += expo(rng);
      }
    }
  };

  struct HostBank {
    bool open = false;
    long row = -1;
    double act = -1e18, last_rd = -1e18, last_act = -1e18, last_pre = -1e18;
    double pre_at = -1.0;   // reserved closing PRE (short windows)
    size_t w = 0;           // current window index
    double w1 = 0;          // PRE deadline of the open row
  };
  std::vector<HostBank> hb(NB);
  double bg_last_rd[NBG];
  for (double& x : bg_last_rd) x = -1e18;
  double last_rd_any = -1e18;
  std::vector<double> lat;
  long host_reads = 0, host_acts = 0;
  int late_pre = 0;
  double host_bytes_in_span = 0.0;
  const double tAAD = 8.0;          // ACT-1 -> ACT-2 (LPDDR5 two-part ACT)
  double tok = bucket_cap, tok_t = 0.0;
  auto tok_ready = [&](double x) {   // earliest time >= x with a burst's tokens
    double have = std::min(bucket_cap, tok + host_rate * (x - tok_t));
    if (have >= t.burst_bytes - 1e-9) return x;
    return x + (t.burst_bytes - have) / std::max(host_rate, 1e-9);
  };
  const double PRE_MARGIN = 2.0 * CMD_CA;
  // DQ transfers with direction (true = write) for bus turnaround
  std::map<double, std::pair<double, bool>> dq;   // start -> (end, write)
  auto dq_ok = [&](double s0, double e0, bool wr) {
    auto it = dq.lower_bound(s0);
    if (it != dq.end()) {
      double gap = (it->second.second != wr) ? c.turnaround_ns : 0.0;
      if (it->first < e0 + gap - EPS) return false;
    }
    if (it != dq.begin()) {
      auto p = std::prev(it);
      double gap = (p->second.second != wr) ? c.turnaround_ns : 0.0;
      if (p->second.first + gap > s0 + EPS) return false;
    }
    return true;
  };
  size_t job_lo = 0;
  int io_missed = 0;
  double io_late_max = 0.0;
  // earliest slot for a PIM I/O job at or after x
  auto job_slot = [&](const IoJob& j, double x) {
    double len = j.bursts * dq_ns;
    for (int it = 0; it < 4096; ++it) {
      double y = io.earliest(x, len);
      if (j.ext) {
        if (!dq_ok(y, y + len, true)) { x = y + 0.125; continue; }
        double z = ca.earliest(y - c.read_latency_ns, CMD_CA);   // WR command
        if (z > y - c.read_latency_ns + EPS) { x = z + c.read_latency_ns; continue; }
      }
      return y;
    }
    return x;
  };
  auto issue_job = [&](IoJob& j, double y) {
    double len = j.bursts * dq_ns;
    io.take(y, len);
    io.host.pop_back();
    io.fixed.push_back({y, y + len});
    if (j.ext) {
      dq[y] = {y + len, true};
      ca.take(y - c.read_latency_ns, CMD_CA);
      ca.host.pop_back();
      ca.fixed.push_back({y - c.read_latency_ns, y - c.read_latency_ns + CMD_CA});
    }
    if (y + len > j.dl + EPS) {
      ++io_missed;
      io_late_max = std::max(io_late_max, y + len - j.dl);
    }
    j.done = true;
  };

  const double stop = span_end;
  double now = 0.0;
  if (c.host) refill(now);
  int guard = 0;
  while (c.host && now < stop) {
    if (++guard > 50000000) throw std::runtime_error("run_rotation: no progress");
    // rows whose reserved PRE has passed are closed
    for (int b = 0; b < NB; ++b) {
      HostBank& h = hb[b];
      if (h.open && h.pre_at >= 0 && h.pre_at <= now + EPS) {
        log.push_back({h.pre_at, b, K_PRE, false, h.row, false});
        h.open = false;
        h.last_pre = h.pre_at;
        h.pre_at = -1.0;
        h.row = -1;
      }
    }
    // PIM I/O: reserve every released job whose slack is exhausted (EDF)
    while (job_lo < jobs.size() && jobs[job_lo].done) ++job_lo;
    for (size_t i = job_lo; i < jobs.size() && jobs[i].rel <= now + 2000.0; ++i) {
      IoJob& j = jobs[i];
      if (j.done) continue;
      double y = job_slot(j, std::max(now, j.rel));
      double extra = j.ext ? 2.0 * c.turnaround_ns + c.read_latency_ns : 0.0;
      if (y + j.bursts * dq_ns + c.pim_io_guard_ns + extra >= j.dl) issue_job(j, y);
    }
    // earliest-deadline released job competes with host commands
    long job_pick = -1;
    double job_t = 1e18;
    for (size_t i = job_lo; i < jobs.size() && jobs[i].rel <= now + EPS; ++i) {
      if (jobs[i].done) continue;
      if (job_pick < 0 || jobs[i].dl < jobs[job_pick].dl) job_pick = static_cast<long>(i);
    }
    if (job_pick >= 0) job_t = job_slot(jobs[job_pick], now);
    struct Cand {
      double t;        // commit time (ACT-1 for activations)
      int bank;
      Kind kind;
      size_t req;
      double t2;       // ACT-2 time
      size_t win = 0;  // host window of the activation
    };
    Cand best{1e18, -1, K_PRE, 0, 0, 0};
    auto better = [&](const Cand& x) {
      if (x.t < best.t - EPS) return true;
      if (x.t > best.t + EPS) return false;
      if (x.kind != best.kind) return x.kind == K_PRE ||
                                      (x.kind == K_RD && best.kind == K_ACT);
      return x.req < best.req;
    };
    std::vector<long> oldest(NB, -1), oldest_hit(NB, -1);
    size_t lim = std::min(pending.size(), static_cast<size_t>(c.window_requests));
    for (size_t i = 0; i < lim; ++i) {
      const Req& r = pending[i];
      if (r.done) continue;
      if (oldest[r.bank] < 0) oldest[r.bank] = static_cast<long>(i);
      if (hb[r.bank].open && hb[r.bank].row == r.row && oldest_hit[r.bank] < 0)
        oldest_hit[r.bank] = static_cast<long>(i);
    }
    for (int b = 0; b < NB; ++b) {
      HostBank& h = hb[b];
      int g = bg_of(b);
      if (h.open) {
        double rd_deadline = (h.pre_at >= 0) ? h.pre_at - t.tRTP : 1e18;
        if (oldest_hit[b] >= 0) {
          const Req& r = pending[oldest_hit[b]];
          double x = std::max({now, r.arr, h.act + t.tRCD, h.last_rd + t.tCCD_L,
                               bg_last_rd[g] + t.tCCD_L,
                               last_rd_any + t.tCCD_S});
          for (int it = 0; it < 4096; ++it) {
            double y = ca.earliest(tok_ready(x), CMD_CA);
            double z = io.earliest(y + c.read_latency_ns, dq_ns) - c.read_latency_ns;
            if (z > y + EPS) { x = z; continue; }
            if (!dq_ok(y + c.read_latency_ns, y + c.read_latency_ns + dq_ns, false)) {
              x = y + 0.125;
              continue;
            }
            x = y;
            break;
          }
          (void)rd_deadline;
          bool fits = x + t.tRTP <= h.pre_at + EPS;
          if (fits) {
            Cand cd{x, b, K_RD, static_cast<size_t>(oldest_hit[b]), 0, 0};
            if (better(cd)) best = cd;
            continue;
          }
        }
        if (oldest_hit[b] < 0) {
          // no hit pending: close early (the reserved deadline PRE is
          // released), unless the reserved slot comes first anyway
          double x = std::max({now, h.last_rd + t.tRTP, h.act + t.tRAS});
          x = ca.earliest(x, CMD_CA);
          if (x < h.pre_at - EPS) {
            Cand cd{x, b, K_PRE, 0, 0, 0};
            if (better(cd)) best = cd;
          }
        }
        continue;
      }
      if (oldest[b] < 0) continue;
      // activate in the earliest window that can hold a host row
      const Req& r = pending[oldest[b]];
      double from = std::max({r.arr, h.last_pre + t.tRP, h.last_act + t.tRC});
      while (h.w < win[b].size() &&
             win[b][h.w].w1 - t.tRAS < std::max(from, now) - EPS)
        ++h.w;
      double f1 = -1, f2 = -1;
      size_t fw = h.w;
      for (size_t w = h.w; w < win[b].size() && w < h.w + 4; ++w) {
        double a0 = std::max({from, now + CMD_CA, win[b][w].w0});
        double latest = std::min(win[b][w].w1 - t.tRAS,
                                 win[b][w].w1 - t.tRTP - t.tRCD) - PRE_MARGIN;
        double x = a0;
        for (int it = 0; it < 512 && x <= latest + EPS; ++it) {
          double z = acts.earliest(x, g, latest);
          if (z < 0) { x = 1e18; break; }
          double z2 = ca.earliest(z, CMD_CA);                  // ACT-2 slot
          if (z2 > z + EPS) { x = z2; continue; }
          double y = ca.latest(z - CMD_CA, CMD_CA, std::max(now, z - tAAD));
          if (y < 0) { x = z + 0.125; continue; }
          // the closing PRE must find a slot in [ACT + tRAS, w1]
          if (ca.latest(win[b][w].w1, CMD_CA, z + t.tRAS) < 0) { x = 1e18; break; }
          f1 = y; f2 = z; fw = w;
          break;
        }
        if (f2 >= 0) break;
      }
      if (f2 >= 0) {
        Cand cd{f1, b, K_ACT, static_cast<size_t>(oldest[b]), f2, fw};
        if (better(cd)) best = cd;
      }
    }

    if (job_pick >= 0 && job_t < best.t - EPS) {
      issue_job(jobs[job_pick], job_t);
      continue;
    }
    // before committing a host command at best.t, reserve every job that
    // could otherwise no longer meet its deadline
    {
      bool any = false;
      double horizon_t = std::min(best.t, 1e17);
      for (size_t i = job_lo; i < jobs.size() && jobs[i].rel <= horizon_t + EPS; ++i) {
        IoJob& j = jobs[i];
        if (j.done) continue;
        double len = j.bursts * dq_ns;
        double slackneed = len + c.pim_io_guard_ns +
                           (j.ext ? 2.0 * c.turnaround_ns + c.read_latency_ns : 0.0);
        if (j.dl - slackneed <= horizon_t + 4.0 * dq_ns) {
          issue_job(j, job_slot(j, std::max(now, j.rel)));
          any = true;
        }
      }
      if (any) continue;
    }
    if (best.bank < 0) {
      double nxt = 1e18;
      if (!c.saturate) nxt = std::min(nxt, next_arrival);
      if (job_lo < jobs.size()) nxt = std::min(nxt, std::max(jobs[job_lo].rel, now + 0.125));
      for (int b = 0; b < NB; ++b) {
        if (hb[b].open && hb[b].pre_at > now) nxt = std::min(nxt, hb[b].pre_at);
        for (size_t w = hb[b].w; w < win[b].size() && w < hb[b].w + 2; ++w)
          if (win[b][w].w0 > now + EPS) nxt = std::min(nxt, win[b][w].w0);
      }
      if (nxt >= 1e17) break;
      now = std::max(now + 0.125, nxt);
      refill(now);
      continue;
    }
    if (best.t >= stop) break;
    now = best.t;
    HostBank& h = hb[best.bank];
    int g = bg_of(best.bank);
    if (best.kind == K_ACT) {
      const Req& r = pending[best.req];
      double a2 = best.t2;
      h.open = true;
      h.row = r.row;
      h.act = a2;
      h.last_act = a2;
      h.last_rd = -1e18;
      h.w = best.win;
      h.w1 = win[best.bank][h.w].w1;
      acts.add(a2, g);
      ca.take(best.t, CMD_CA);   // ACT-1
      ca.take(a2, CMD_CA);       // ACT-2
      log.push_back({a2, best.bank, K_ACT, false, r.row, false});
      ++host_acts;
      {
        double p = ca.latest(h.w1, CMD_CA, a2 + t.tRAS);
        if (p < 0) {
          p = ca.earliest(a2 + t.tRAS, CMD_CA);
          ++late_pre;
          if (std::getenv("PIMCORE_DEBUG"))
            std::fprintf(stderr, "late PRE reservation: bank %d act %.3f w1 %.3f -> %.3f\n",
                         best.bank, a2, h.w1, p);
        }
        ca.take(p, CMD_CA);
        h.pre_at = p;
      }
    } else if (best.kind == K_RD) {
      Req& r = pending[best.req];
      r.done = true;
      h.last_rd = now;
      bg_last_rd[g] = now;
      last_rd_any = now;
      ca.take(now, CMD_CA);
      io.take(now + c.read_latency_ns, dq_ns);
      dq[now + c.read_latency_ns] = {now + c.read_latency_ns + dq_ns, false};
      tok = std::min(bucket_cap, tok + host_rate * (now - tok_t)) - t.burst_bytes;
      tok_t = now;
      log.push_back({now, best.bank, K_RD, false, r.row, false});
      double done = now + c.read_latency_ns + dq_ns;
      lat.push_back(done - r.arr);
      ++host_reads;
      if (now >= span0 && now <= stop) host_bytes_in_span += t.burst_bytes;
      while (!pending.empty() && pending.front().done) pending.pop_front();
    } else {
      if (now > h.w1 + EPS) {
        ++late_pre;
        if (std::getenv("PIMCORE_DEBUG"))
          std::fprintf(stderr, "late early-PRE: bank %d t %.3f w1 %.3f\n", best.bank, now, h.w1);
      }
      if (h.pre_at >= 0) { ca.release(h.pre_at); h.pre_at = -1.0; }
      h.open = false;
      h.last_pre = now;
      ca.take(now, CMD_CA);
      log.push_back({now, best.bank, K_PRE, false, h.row, false});
      h.row = -1;
    }
    refill(now);
  }

  // rows still open at the end close at their reserved PRE
  for (int b = 0; b < NB; ++b)
    if (hb[b].open && hb[b].pre_at >= 0)
      log.push_back({hb[b].pre_at, b, K_PRE, false, hb[b].row, false});
  for (IoJob& j : jobs)
    if (!j.done) issue_job(j, job_slot(j, std::max(j.rel, 0.0)));
  rep.pim_io_missed = io_missed;
  rep.pim_io_late_max = io_late_max;

  // ---- independent checker -------------------------------------------------
  std::sort(log.begin(), log.end(), [](const Cmd& a, const Cmd& b) {
    if (std::fabs(a.t - b.t) > EPS) return a.t < b.t;
    return a.kind > b.kind;   // PRE before ACT at the same instant
  });
  // per bank protocol
  const bool dbg = std::getenv("PIMCORE_DEBUG") != nullptr;
  int dbg_left = 20;
  auto bad = [&](int& ctr, const char* what, const Cmd& x) {
    ++ctr;
    if (dbg && dbg_left-- > 0)
      std::fprintf(stderr, "violation %s: t=%.3f bank=%d kind=%d pim=%d row=%ld\n",
                   what, x.t, x.bank, static_cast<int>(x.kind), x.pim ? 1 : 0, x.row);
  };
  for (int b = 0; b < NB; ++b) {
    bool open = false;
    bool owner_pim = false;
    long row = -1;
    double act = -1e18, pre = -1e18, prev_act = -1e18, last_rd = -1e18;
    for (const Cmd& x : log) {
      if (x.bank != b) continue;
      if (x.kind == K_ACT) {
        if (open) {
          bad(rep.viol_bank, "ACT to open bank", x);
          if (owner_pim != x.pim) ++rep.viol_conflict;
        }
        if (x.t < pre + t.tRP - EPS) bad(rep.viol_bank, "tRP", x);
        if (x.t < prev_act + t.tRC - EPS) bad(rep.viol_bank, "tRC", x);
        open = true;
        owner_pim = x.pim;
        row = x.row;
        act = prev_act = x.t;
        last_rd = -1e18;
      } else if (x.kind == K_RD) {
        if (!open || x.pim != owner_pim || x.row != row) {
          bad(rep.viol_bank, "RD state", x);
          if (open && x.pim != owner_pim) ++rep.viol_conflict;
        }
        if (x.t < act + t.tRCD - EPS) bad(rep.viol_bank, "tRCD", x);
        double ccd = x.pim ? t.tCCD_PIM : t.tCCD_L;
        if (x.t < last_rd + ccd - EPS) bad(rep.viol_bank, "tCCD", x);
        last_rd = x.t;
      } else {
        if (!open) bad(rep.viol_bank, "PRE closed", x);
        if (x.t < act + t.tRAS - EPS) bad(rep.viol_bank, "tRAS", x);
        if (x.t < last_rd + t.tRTP - EPS) bad(rep.viol_bank, "tRTP", x);
        open = false;
        pre = x.t;
      }
    }
  }
  // activation windows over all activations (exempt all-bank ACTs of the
  // reference schedules are checked only against host activations)
  {
    std::vector<std::pair<double, int>> a;
    for (const Cmd& x : log)
      if (x.kind == K_ACT && !x.exempt) a.push_back({x.t, bg_of(x.bank)});
    std::sort(a.begin(), a.end());
    for (size_t i = 0; i < a.size(); ++i) {
      for (size_t j = i + 1; j < a.size() && a[j].first - a[i].first < t.tRRD_L; ++j) {
        double need = (a[i].second == a[j].second) ? t.tRRD_L : t.tRRD_S;
        if (a[j].first - a[i].first < need - EPS) {
          ++rep.viol_act;
          if (dbg) std::fprintf(stderr, "violation tRRD: %.3f (bg%d) -> %.3f (bg%d)\n",
                                a[i].first, a[i].second, a[j].first, a[j].second);
        }
      }
      if (i + 4 < a.size() && a[i + 4].first - a[i].first < t.tFAW - EPS) {
        ++rep.viol_act;
        if (dbg) std::fprintf(stderr, "violation tFAW: %.3f .. %.3f\n", a[i].first,
                              a[i + 4].first);
      }
    }
    rep.act_rate_per_us = a.size() / std::max(stop - span0, 1.0) * 1000.0;
  }
  // host column spacing
  {
    double last_any = -1e18, last_bg[NBG];
    for (double& x : last_bg) x = -1e18;
    for (const Cmd& x : log) {
      if (x.pim || x.kind != K_RD) continue;
      int g = bg_of(x.bank);
      if (x.t < last_any + t.tCCD_S - EPS) ++rep.viol_io;
      if (x.t < last_bg[g] + t.tCCD_L - EPS) ++rep.viol_io;
      last_any = x.t;
      last_bg[g] = x.t;
    }
  }
  // I/O and C/A double booking
  auto overlaps = [](std::vector<Interval> v) {
    std::sort(v.begin(), v.end(),
              [](const Interval& x, const Interval& y) { return x.a < y.a; });
    int n = 0;
    for (size_t i = 1; i < v.size(); ++i)
      if (v[i].a < v[i - 1].b - EPS) ++n;
    return n;
  };
  {
    std::vector<Interval> all = io.fixed;
    all.insert(all.end(), io.host.begin(), io.host.end());
    rep.viol_io += overlaps(all);
    double busy = 0;
    for (const Interval& iv : all) {
      double a = std::max(iv.a, span0), b = std::min(iv.b, stop);
      if (b > a) busy += b - a;
    }
    rep.io_util = busy / std::max(stop - span0, 1.0);
    std::vector<Interval> cav = ca.fixed;
    cav.insert(cav.end(), ca.host.begin(), ca.host.end());
    rep.viol_ca = overlaps(cav);
    double cbusy = 0;
    for (const Interval& iv : cav) {
      double a = std::max(iv.a, span0), b = std::min(iv.b, stop);
      if (b > a) cbusy += b - a;
    }
    rep.ca_util = cbusy / std::max(stop - span0, 1.0);
  }
  // refresh: no command inside [REF, REF + tRFCab), all rows closed at REF
  for (double r : refs) {
    if (r > stop) break;
    for (const Cmd& x : log)
      if (x.t > r - EPS && x.t < r + t.tRFCab - EPS) ++rep.viol_refresh;
  }

  // broadcast line, PE FIFO and accumulators (per bank group)
  int fifo_max = 0, acc_max = 0, bus_coll = 0;
  double mac_cycles = 0;
  if (c.pim) {
    for (int g = 0; g < NBG; ++g) {
      std::vector<std::pair<double, int>> arr;   // (arrival at PE, bank)
      for (int k = 0; k < NBK; ++k)
        for (double x : ps.reads[bank_id(g, k)])
          arr.push_back({x + c.data_latency_ns, k});
      std::sort(arr.begin(), arr.end());
      if (c.schedule == Schedule::DIRECT) {
        // one-to-one: each PE drains its own bank, one burst per 2 ns
        fifo_max = std::max(fifo_max, 1);
        acc_max = std::max(acc_max, 1);
        mac_cycles += arr.size() / static_cast<double>(NBK);
        continue;
      }
      // broadcast line: one transfer per PE cycle
      std::map<long, int> slot;
      for (auto& a : arr) {
        long s = static_cast<long>(std::floor(a.first / PE_CYCLE + 1e-9));
        if (++slot[s] > 1) ++bus_coll;
      }
      // FIFO: push on arrival, pop one per PE cycle (MAC issue)
      double pe_free = -1e18;
      std::deque<double> q;
      for (auto& a : arr) {
        while (!q.empty() && q.front() <= a.first + EPS) q.pop_front();
        double issue = std::max(a.first, pe_free);
        pe_free = issue + PE_CYCLE;
        q.push_back(issue);
        fifo_max = std::max(fifo_max, static_cast<int>(q.size()));
      }
      mac_cycles += arr.size();
      // concurrent partial sums: banks whose stream is in flight at once
      std::vector<std::pair<double, int>> ev;
      for (int k = 0; k < NBK; ++k)
        for (const Turn& tr : ps.turns[bank_id(g, k)]) {
          ev.push_back({tr.first + c.data_latency_ns, +1});
          ev.push_back({tr.last + c.data_latency_ns + PE_CYCLE, -1});
        }
      std::sort(ev.begin(), ev.end());
      int cur = 0;
      for (auto& e : ev) { cur += e.second; acc_max = std::max(acc_max, cur); }
    }
    mac_cycles /= NBG;   // per PE (all PEs of a group issue together)
  }
  rep.fifo_max = fifo_max;
  rep.acc_max = acc_max;
  rep.viol_bus = bus_coll;
  rep.viol_bank += late_pre;

  // ---- report ----------------------------------------------------------------
  rep.span_ns = stop - span0;
  rep.pim_bursts = static_cast<double>(pim_bursts);
  if (c.pim) {
    rep.pim_time_ns = stop - span0;
    // peak: four bank groups x one transfer per PE cycle (broadcast) or
    // sixteen banks x one burst per tCCD_PIM (direct)
    double peak_bursts_per_ns = (c.schedule == Schedule::DIRECT)
                                    ? NB / t.tCCD_PIM
                                    : NBG / PE_CYCLE;
    rep.pim_ideal_ns = pim_bursts / peak_bursts_per_ns;
    rep.pim_gbps = pim_bursts * t.burst_bytes / rep.pim_time_ns;
    rep.pe_util = mac_cycles / rep.pim_time_ns;
  }
  rep.host_reads = host_reads;
  rep.host_acts = host_acts;
  rep.host_bytes = host_bytes_in_span;
  rep.host_gbps = host_bytes_in_span / std::max(rep.span_ns, 1.0);
  rep.host_frac = rep.host_gbps / (c.ext_bw_gbps * c.ext_bus_eff);
  rep.bursts_per_host_act = host_reads / std::max<double>(host_acts, 1);
  rep.pim_acts = 0;
  for (int b = 0; b < NB; ++b) rep.pim_acts += ps.turns[b].size();
  if (!lat.empty()) {
    double s = 0;
    for (double x : lat) s += x;
    rep.host_lat_mean = s / lat.size();
    rep.host_lat_p50 = percentile(lat, 0.5);
    rep.host_lat_p99 = percentile(lat, 0.99);
    rep.host_lat_max = *std::max_element(lat.begin(), lat.end());
  }

  // ---- trace -------------------------------------------------------------------
  if (c.trace) {
    auto in = [&](double x) { return x >= c.trace_from_ns && x <= c.trace_until_ns; };
    const char* kn[] = {"ACT", "RD", "PRE"};
    for (const Cmd& x : log) {
      if (!in(x.t)) continue;
      std::string unit = "bg" + std::to_string(bg_of(x.bank)) + ".bank" +
                         std::to_string(x.bank % NBK);
      std::string kind = std::string(x.pim ? "PIM_" : "HOST_") + kn[x.kind];
      double dur = (x.kind == K_RD) ? (x.pim ? t.tCCD_PIM : t.tCCD_L) : 0.0;
      rep.trace.push_back({x.t, dur, unit, kind, x.bank});
    }
    for (const Interval& iv : io.fixed)
      if (in(iv.a)) rep.trace.push_back({iv.a, iv.b - iv.a, "io", "PIM_IO", 0});
    for (const Interval& iv : io.host)
      if (in(iv.a)) rep.trace.push_back({iv.a, iv.b - iv.a, "io", "HOST_DATA", 0});
    for (const Interval& iv : ca.fixed)
      if (in(iv.a)) rep.trace.push_back({iv.a, iv.b - iv.a, "ca", "PIM_CMD", 0});
    for (const Interval& iv : ca.host)
      if (in(iv.a)) rep.trace.push_back({iv.a, iv.b - iv.a, "ca", "HOST_CMD", 0});
    for (double r : refs)
      if (in(r)) rep.trace.push_back({r, t.tRFCab, "die", "REF", 0});
    // broadcast line and FIFO of bank group 0
    if (c.pim && c.schedule != Schedule::DIRECT) {
      std::vector<std::pair<double, int>> arr;
      for (int k = 0; k < NBK; ++k)
        for (double x : ps.reads[bank_id(0, k)])
          arr.push_back({x + c.data_latency_ns, k});
      std::sort(arr.begin(), arr.end());
      double pe_free = -1e18;
      std::deque<double> q;
      for (auto& a : arr) {
        while (!q.empty() && q.front() <= a.first + EPS) q.pop_front();
        double issue = std::max(a.first, pe_free);
        pe_free = issue + PE_CYCLE;
        q.push_back(issue);
        if (in(a.first)) {
          rep.trace.push_back({a.first, PE_CYCLE, "bg0.bus", "XFER", a.second});
          rep.trace.push_back({a.first, 0.0, "bg0.fifo", "OCC",
                               static_cast<int>(q.size())});
          rep.trace.push_back({issue, PE_CYCLE, "bg0.pe", "MAC", a.second});
        }
      }
    }
    std::sort(rep.trace.begin(), rep.trace.end(),
              [](const TraceEvent& a, const TraceEvent& b) { return a.t < b.t; });
  }
  (void)row_bursts;
  return rep;
}

}  // namespace pimcore
