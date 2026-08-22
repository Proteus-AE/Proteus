// ============================================================================
// pe_mac16: 16-lane FP16 multiply-accumulate array -- the per-bank
// processing element of Sec. IV-C (Fig. microarch (c)).
//
// Each cycle one 256-bit operand burst (16 FP16 words, one row-slice of the
// resident matrix) is multiplied lane-wise with the vector operand held in
// the vector register and accumulated into the per-lane accumulators.
// `acc_clear` starts a new output element; `acc_out` exposes the reduction
// of the 16 lanes through a 4-stage FP16 adder tree.
//
// The datapath is split into the three blocks the area model of Sec. V-B
// reports separately (configs/area/near-bank-pe.yaml): the multiply-
// accumulate lanes (mul_array), the reduction tree (add_tree), and the
// vector/accumulator register bank (vec_acc_regs).
// ============================================================================
`timescale 1ns / 1ps

module pe_mac16 (
    input  wire         clk,
    input  wire         rst_n,
    // operand burst from the FIFO
    input  wire         op_valid,
    output wire         op_ready,
    input  wire [255:0] op_data,
    // vector operand (broadcast input vector slice), loaded per pass
    input  wire         vec_we,
    input  wire [255:0] vec_data,
    // accumulator control
    input  wire         acc_clear,
    output wire [15:0]  acc_out,
    output wire         acc_valid
);

  assign op_ready = 1'b1;   // single-cycle issue; FIFO provides elasticity

  wire [255:0] vec_q;       // vector operand of the current pass
  wire [255:0] acc_q;       // 16 lane accumulators
  wire [255:0] acc_next;    // accumulators after this burst

  vec_acc_regs u_vec_acc_regs (
      .clk(clk), .rst_n(rst_n),
      .vec_we(vec_we),
      .vec_data(vec_data),
      .acc_clear(acc_clear),
      .acc_we(op_valid),
      .acc_next(acc_next),
      .vec_q(vec_q),
      .acc_q(acc_q),
      .acc_valid(acc_valid)
  );

  mul_array u_mul_array (
      .op_data(op_data),
      .vec_data(vec_q),
      .acc_in(acc_q),
      .acc_out(acc_next)
  );

  add_tree u_add_tree (
      .lane_in(acc_q),
      .sum(acc_out)
  );

endmodule
