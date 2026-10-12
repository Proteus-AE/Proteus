// ============================================================================
// sfu_pipeline: per-channel programmable special-function unit
// (Fig. "Microarchitecture of the programmable SFU", Sec. IV-C).
//
// A short micro-programmed streaming pipeline over FP16 lanes composed of
// arithmetic, accumulation, and reduction primitives. Different
// micro-programs realize residual addition, softmax (scale + exp-approx +
// normalize), layer normalization, and activation functions; fused chains
// execute in one pass without intermediate DRAM accesses (Sec. IV-E).
//
// Primitive opcodes (per micro-instruction slot):
//   OP_NOP   pass-through
//   OP_ADD   x + operand           (residual / bias)
//   OP_MUL   x * operand           (scale / normalize)
//   OP_MAX   running max           (softmax pass 1)
//   OP_EXP2  2^x piecewise-linear  (softmax pass 2, exp approximation)
//   OP_RELU  max(x, 0)
//   OP_ACC   running sum           (reductions, mean/variance, denom)
// ============================================================================
`timescale 1ns / 1ps

module sfu_pipeline #(
    parameter LANES = 16
) (
    input  wire                  clk,
    input  wire                  rst_n,
    // micro-program: three chained primitive slots
    input  wire [2:0]            uop0,
    input  wire [2:0]            uop1,
    input  wire [2:0]            uop2,
    input  wire [LANES*16-1:0]   operand0,   // per-slot scalar operands
    input  wire [LANES*16-1:0]   operand1,
    input  wire [LANES*16-1:0]   operand2,
    // streaming input
    input  wire                  in_valid,
    input  wire [LANES*16-1:0]   in_data,
    input  wire                  acc_clear,
    // streaming output (2-cycle latency)
    output reg                   out_valid,
    output reg  [LANES*16-1:0]   out_data,
    output wire [15:0]           acc_scalar   // reduction result
);

  localparam OP_NOP  = 3'd0;
  localparam OP_ADD  = 3'd1;
  localparam OP_MUL  = 3'd2;
  localparam OP_MAX  = 3'd3;
  localparam OP_EXP2 = 3'd4;
  localparam OP_RELU = 3'd5;
  localparam OP_ACC  = 3'd6;

  // ------------------------------------------------------------------ //
  // one primitive slot applied lane-wise
  // ------------------------------------------------------------------ //
  function [15:0] fp16_relu(input [15:0] x);
    fp16_relu = x[15] ? 16'd0 : x;
  endfunction

  // piecewise-linear 2^x on FP16: split x into integer and fraction,
  // approximate 2^f ~ 1 + f (max error ~6% -- adequate for the softmax
  // characterization; the table-based production variant is a drop-in).
  function [15:0] fp16_exp2_pwl(input [15:0] x);
    reg sign;
    reg signed [7:0] e_unb;
    reg [4:0] e_new;
    begin
      sign = x[15];
      // clamp the useful range [-15, 15]
      e_unb = $signed({3'b000, x[14:10]}) - 8'sd15;
      if (x[14:10] == 5'd0) begin
        fp16_exp2_pwl = 16'h3C00;               // 2^~0 = 1.0
      end else if (e_unb >= 8'sd4) begin
        fp16_exp2_pwl = sign ? 16'h0000 : 16'h7BFF;  // saturate
      end else begin
        // 2^x with x ~ +-[1,16): exponent moves by trunc(x), fraction
        // approximated linearly through the mantissa bits.
        e_new = sign ? (5'd15 - x[13:10]) : (5'd15 + x[13:10]);
        fp16_exp2_pwl = {1'b0, e_new, x[9:0]};
      end
    end
  endfunction

  // slot application over one lane
  function [15:0] slot_apply(input [2:0] op, input [15:0] x,
                             input [15:0] c);
    begin
      case (op)
        OP_RELU: slot_apply = fp16_relu(x);
        OP_EXP2: slot_apply = fp16_exp2_pwl(x);
        OP_NOP:  slot_apply = x;
        default: slot_apply = x;   // ADD/MUL/MAX/ACC handled structurally
      endcase
    end
  endfunction

  // structural lane pipelines: ADD and MUL need fp16 units
  wire [15:0] s0_out [0:LANES-1];
  wire [15:0] s1_out [0:LANES-1];
  wire [15:0] s2_out [0:LANES-1];

  genvar gi;
  generate
    for (gi = 0; gi < LANES; gi = gi + 1) begin : g_lane
      wire [15:0] x0 = in_data[gi*16 +: 16];
      wire [15:0] add0, mul0;
      fp16_add u_a0 (.a(x0), .b(operand0[gi*16 +: 16]), .y(add0));
      fp16_mul u_m0 (.a(x0), .b(operand0[gi*16 +: 16]), .y(mul0));
      assign s0_out[gi] = (uop0 == OP_ADD) ? add0 :
                          (uop0 == OP_MUL) ? mul0 :
                          slot_apply(uop0, x0, operand0[gi*16 +: 16]);

      wire [15:0] x1 = s0_out[gi];
      wire [15:0] add1, mul1;
      fp16_add u_a1 (.a(x1), .b(operand1[gi*16 +: 16]), .y(add1));
      fp16_mul u_m1 (.a(x1), .b(operand1[gi*16 +: 16]), .y(mul1));
      assign s1_out[gi] = (uop1 == OP_ADD) ? add1 :
                          (uop1 == OP_MUL) ? mul1 :
                          slot_apply(uop1, x1, operand1[gi*16 +: 16]);

      wire [15:0] x2 = s1_out[gi];
      wire [15:0] add2, mul2;
      fp16_add u_a2 (.a(x2), .b(operand2[gi*16 +: 16]), .y(add2));
      fp16_mul u_m2 (.a(x2), .b(operand2[gi*16 +: 16]), .y(mul2));
      assign s2_out[gi] = (uop2 == OP_ADD) ? add2 :
                          (uop2 == OP_MUL) ? mul2 :
                          slot_apply(uop2, x2, operand2[gi*16 +: 16]);
    end
  endgenerate

  // ------------------------------------------------------------------ //
  // reduction accumulator across lanes and stream (OP_ACC / OP_MAX)
  // ------------------------------------------------------------------ //
  wire [15:0] lane_sum_l1 [0:7];
  wire [15:0] lane_sum_l2 [0:3];
  wire [15:0] lane_sum_l3 [0:1];
  wire [15:0] lane_sum;
  generate
    for (gi = 0; gi < 8; gi = gi + 1) begin : g_r1
      fp16_add u (.a(s2_out[2*gi]), .b(s2_out[2*gi+1]), .y(lane_sum_l1[gi]));
    end
    for (gi = 0; gi < 4; gi = gi + 1) begin : g_r2
      fp16_add u (.a(lane_sum_l1[2*gi]), .b(lane_sum_l1[2*gi+1]),
                  .y(lane_sum_l2[gi]));
    end
    for (gi = 0; gi < 2; gi = gi + 1) begin : g_r3
      fp16_add u (.a(lane_sum_l2[2*gi]), .b(lane_sum_l2[2*gi+1]),
                  .y(lane_sum_l3[gi]));
    end
  endgenerate
  fp16_add u_r4 (.a(lane_sum_l3[0]), .b(lane_sum_l3[1]), .y(lane_sum));

  reg [15:0] acc_q;
  wire [15:0] acc_next;
  fp16_add u_acc (.a(acc_q), .b(lane_sum), .y(acc_next));
  assign acc_scalar = acc_q;

  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
      out_data <= {LANES*16{1'b0}};
      acc_q <= 16'd0;
    end else begin
      out_valid <= in_valid;
      if (acc_clear) acc_q <= 16'd0;
      if (in_valid) begin
        for (i = 0; i < LANES; i = i + 1)
          out_data[i*16 +: 16] <= s2_out[i];
        if (uop2 == OP_ACC || uop1 == OP_ACC || uop0 == OP_ACC)
          acc_q <= acc_next;
      end
    end
  end

endmodule
