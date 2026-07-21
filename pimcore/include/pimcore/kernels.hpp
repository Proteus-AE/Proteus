// Built-in kernel generators: lower the decode-phase operators onto the
// column/row-striped bank layout and emit all-bank command streams.
#pragma once

#include <vector>

#include "pimcore/timing.hpp"
#include "pimcore/types.hpp"

namespace pimcore {

struct KernelParams {
  int rows_per_bank = 64;        // striped-operand rows per bank per pass
  int n_vectors = 1;             // concurrent input vectors (skinny-GEMM)
  int group_size = 1;            // GQA/MLA query-group size (attention)
  ConnectivityMode mode = ConnectivityMode::DIRECT;
  int broadcast_fanout = 4;
  bool kv_append = false;        // trailing in-place KV write (decode step)
  bool emit_mode_cmd = true;
};

// One streaming pass over the striped operand: ACT_AB / RDMAC_AB* / PRE_AB.
std::vector<Command> stream_pass(int rows, int bursts_per_row, int base_row);

// Reuse-free GEMV: single pass, direct connectivity.
std::vector<Command> gemv_kernel(const KernelParams& kp,
                                 const TimingParams& tp);

// Shared-operand skinny-GEMM: `n_vectors` concurrent inputs.
//   direct    -> one pass per vector (no inter-PE reuse)
//   broadcast -> ceil(n / fanout) passes
std::vector<Command> skinny_gemm_kernel(const KernelParams& kp,
                                        const TimingParams& tp);

// Decode attention over the resident KV slice of one head-group.
std::vector<Command> attention_kernel(const KernelParams& kp,
                                      const TimingParams& tp);

}  // namespace pimcore
