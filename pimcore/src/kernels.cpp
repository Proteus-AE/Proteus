#include "pimcore/kernels.hpp"

namespace pimcore {

namespace {
int ceil_div(int a, int b) { return (a + b - 1) / b; }
}  // namespace

std::vector<Command> stream_pass(int rows, int bursts_per_row, int base_row) {
  std::vector<Command> out;
  out.reserve(static_cast<size_t>(rows) * (bursts_per_row + 2));
  for (int r = 0; r < rows; ++r) {
    Command act;
    act.kind = CommandKind::ACT_AB;
    act.row = base_row + r;
    out.push_back(act);
    for (int c = 0; c < bursts_per_row; ++c) {
      Command rd;
      rd.kind = CommandKind::RDMAC_AB;
      rd.row = base_row + r;
      rd.col = c;
      out.push_back(rd);
    }
    Command pre;
    pre.kind = CommandKind::PRE_AB;
    out.push_back(pre);
  }
  return out;
}

static std::vector<Command> with_mode(const KernelParams& kp) {
  std::vector<Command> out;
  if (kp.emit_mode_cmd) {
    Command m;
    m.kind = CommandKind::MODE;
    m.mode = kp.mode;
    out.push_back(m);
  }
  return out;
}

std::vector<Command> gemv_kernel(const KernelParams& kp,
                                 const TimingParams& tp) {
  KernelParams p = kp;
  p.mode = ConnectivityMode::DIRECT;
  std::vector<Command> out = with_mode(p);
  int bursts = tp.row_bytes / tp.burst_bytes;
  auto pass = stream_pass(p.rows_per_bank, bursts, 0);
  out.insert(out.end(), pass.begin(), pass.end());
  return out;
}

std::vector<Command> skinny_gemm_kernel(const KernelParams& kp,
                                        const TimingParams& tp) {
  std::vector<Command> out = with_mode(kp);
  int bursts = tp.row_bytes / tp.burst_bytes;
  int passes = (kp.mode == ConnectivityMode::BROADCAST)
                   ? ceil_div(kp.n_vectors, kp.broadcast_fanout)
                   : kp.n_vectors;
  for (int p = 0; p < passes; ++p) {
    auto pass = stream_pass(kp.rows_per_bank, bursts, 0);
    out.insert(out.end(), pass.begin(), pass.end());
  }
  return out;
}

std::vector<Command> attention_kernel(const KernelParams& kp,
                                      const TimingParams& tp) {
  std::vector<Command> out = with_mode(kp);
  int bursts = tp.row_bytes / tp.burst_bytes;
  int passes = (kp.mode == ConnectivityMode::BROADCAST)
                   ? ceil_div(kp.group_size, kp.broadcast_fanout)
                   : kp.group_size;
  for (int p = 0; p < passes; ++p) {
    auto pass = stream_pass(kp.rows_per_bank, bursts, 0);
    out.insert(out.end(), pass.begin(), pass.end());
  }
  if (kp.kv_append) {
    // This decode step's K/V entries land at the stripe tail in place; the
    // per-bank share of one token is below one burst.
    Command wr;
    wr.kind = CommandKind::WR_AB;
    wr.row = kp.rows_per_bank;
    wr.col = 0;
    out.push_back(wr);
    Command pre;
    pre.kind = CommandKind::PRE_AB;
    out.push_back(pre);
  }
  return out;
}

}  // namespace pimcore
