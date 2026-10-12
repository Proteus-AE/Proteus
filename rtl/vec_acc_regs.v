// ============================================================================
// vec_acc_regs: vector register and lane accumulator bank of the per-bank PE
// (Fig. microarch (c)).
//
// Holds the 16-word vector operand loaded once per pass and the 16 lane
// accumulators updated on every operand burst. `acc_clear` starts a new
// output element; `acc_valid` tracks whether the accumulators hold a
// partial result. Both banks launch the PE's critical paths -- the vector
// register into the multiply-accumulate lanes, the accumulators into both
// the lanes and the reduction tree -- so both synthesize with high-drive
// asynchronously reset flops.
// ============================================================================
`timescale 1ns / 1ps

module vec_acc_regs #(
    parameter LANES = 16
) (
    input  wire                clk,
    input  wire                rst_n,
    // vector operand load
    input  wire                vec_we,
    input  wire [LANES*16-1:0] vec_data,
    // accumulator update
    input  wire                acc_clear,
    input  wire                acc_we,
    input  wire [LANES*16-1:0] acc_next,
    // register contents
    output reg  [LANES*16-1:0] vec_q,
    output reg  [LANES*16-1:0] acc_q,
    output reg                 acc_valid
);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) vec_q <= {LANES*16{1'b0}};
    else if (vec_we) vec_q <= vec_data;
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc_q <= {LANES*16{1'b0}};
      acc_valid <= 1'b0;
    end else if (acc_clear) begin
      acc_q <= {LANES*16{1'b0}};
      acc_valid <= 1'b0;
    end else if (acc_we) begin
      acc_q <= acc_next;
      acc_valid <= 1'b1;
    end
  end

endmodule
