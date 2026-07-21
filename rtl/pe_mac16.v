// ============================================================================
// pe_mac16: 16-lane FP16 multiply-accumulate array -- the per-bank
// processing element of Sec. IV-C (Fig. microarch (c)).
//
// Each cycle one 256-bit operand burst (16 FP16 words, one row-slice of the
// resident matrix) is multiplied lane-wise with the vector operand held in
// the vector register and accumulated into the per-lane accumulators.
// `acc_clear` starts a new output element; `acc_out` exposes the reduction
// of the 16 lanes through a 4-stage FP16 adder tree.
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
    output reg          acc_valid
);

  reg [255:0] vec_q;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) vec_q <= 256'd0;
    else if (vec_we) vec_q <= vec_data;
  end

  assign op_ready = 1'b1;   // single-cycle issue; FIFO provides elasticity

  // lane products
  wire [15:0] prod [0:15];
  genvar gi;
  generate
    for (gi = 0; gi < 16; gi = gi + 1) begin : g_lane
      fp16_mul u_mul (
          .a(op_data[gi*16 +: 16]),
          .b(vec_q[gi*16 +: 16]),
          .y(prod[gi])
      );
    end
  endgenerate

  // per-lane accumulators
  reg  [15:0] acc [0:15];
  wire [15:0] acc_next [0:15];
  generate
    for (gi = 0; gi < 16; gi = gi + 1) begin : g_acc
      fp16_add u_acc (
          .a(acc[gi]),
          .b(prod[gi]),
          .y(acc_next[gi])
      );
    end
  endgenerate

  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i = 0; i < 16; i = i + 1) acc[i] <= 16'd0;
      acc_valid <= 1'b0;
    end else begin
      if (acc_clear) begin
        for (i = 0; i < 16; i = i + 1) acc[i] <= 16'd0;
        acc_valid <= 1'b0;
      end else if (op_valid) begin
        for (i = 0; i < 16; i = i + 1) acc[i] <= acc_next[i];
        acc_valid <= 1'b1;
      end
    end
  end

  // 16 -> 1 reduction tree (4 FP16 adder levels)
  wire [15:0] l1 [0:7];
  wire [15:0] l2 [0:3];
  wire [15:0] l3 [0:1];
  generate
    for (gi = 0; gi < 8; gi = gi + 1) begin : g_red1
      fp16_add u (.a(acc[2*gi]), .b(acc[2*gi+1]), .y(l1[gi]));
    end
    for (gi = 0; gi < 4; gi = gi + 1) begin : g_red2
      fp16_add u (.a(l1[2*gi]), .b(l1[2*gi+1]), .y(l2[gi]));
    end
    for (gi = 0; gi < 2; gi = gi + 1) begin : g_red3
      fp16_add u (.a(l2[2*gi]), .b(l2[2*gi+1]), .y(l3[gi]));
    end
  endgenerate
  fp16_add u_red4 (.a(l3[0]), .b(l3[1]), .y(acc_out));

endmodule
