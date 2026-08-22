// ============================================================================
// mul_array: the 16 multiply-accumulate lanes of the per-bank PE
// (Fig. microarch (c)).
//
// One lane per FP16 word of the 256-bit operand burst: the resident-matrix
// word is multiplied by the corresponding word of the vector register and
// added into that lane's accumulator. Both units are combinational, so a
// lane's multiply and accumulate share a single 1 GHz cycle; the result is
// captured by vec_acc_regs.
// ============================================================================
`timescale 1ns / 1ps

module mul_array #(
    parameter LANES = 16
) (
    input  wire [LANES*16-1:0] op_data,    // operand burst (matrix slice)
    input  wire [LANES*16-1:0] vec_data,   // vector register contents
    input  wire [LANES*16-1:0] acc_in,     // current accumulators
    output wire [LANES*16-1:0] acc_out     // accumulators + lane products
);

  genvar gi;
  generate
    for (gi = 0; gi < LANES; gi = gi + 1) begin : g_lane
      wire [15:0] prod;

      fp16_mul u_mul (
          .a(op_data[gi*16 +: 16]),
          .b(vec_data[gi*16 +: 16]),
          .y(prod)
      );

      fp16_add u_acc (
          .a(acc_in[gi*16 +: 16]),
          .b(prod),
          .y(acc_out[gi*16 +: 16])
      );
    end
  endgenerate

endmodule
