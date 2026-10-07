// Self-checking testbench for the programmable SFU: a fused residual-add +
// ReLU micro-program over the lanes, a scale micro-program, and the lane
// reduction used by the softmax denominator / layer-norm mean.
//
// Stimulus is applied one settle delay *after* the sampling edge and sampled
// again before the next one, so no input changes at a clock edge.
`timescale 1ns / 1ps

module tb_sfu_pipeline;

  localparam HALF   = 0.5;
  localparam SETTLE = 0.1;
  localparam LANES  = 16;

  localparam [2:0] OP_NOP  = 3'd0;
  localparam [2:0] OP_ADD  = 3'd1;
  localparam [2:0] OP_MUL  = 3'd2;
  localparam [2:0] OP_RELU = 3'd5;
  localparam [2:0] OP_ACC  = 3'd6;

  localparam [15:0] F1   = 16'h3C00;   //  1.0
  localparam [15:0] F2   = 16'h4000;   //  2.0
  localparam [15:0] F3   = 16'h4200;   //  3.0
  localparam [15:0] FM3  = 16'hC200;   // -3.0
  localparam [15:0] F16  = 16'h4C00;   // 16.0
  localparam [15:0] F32  = 16'h5000;   // 32.0

  reg clk, rst_n;
  reg [2:0] uop0, uop1, uop2;
  reg [LANES*16-1:0] operand0, operand1, operand2;
  reg in_valid, acc_clear;
  reg [LANES*16-1:0] in_data;
  wire out_valid;
  wire [LANES*16-1:0] out_data;
  wire [15:0] acc_scalar;
  integer errors, i;

  sfu_pipeline #(.LANES(LANES)) dut (
      .clk(clk), .rst_n(rst_n),
      .uop0(uop0), .uop1(uop1), .uop2(uop2),
      .operand0(operand0), .operand1(operand1), .operand2(operand2),
      .in_valid(in_valid), .in_data(in_data), .acc_clear(acc_clear),
      .out_valid(out_valid), .out_data(out_data),
      .acc_scalar(acc_scalar));

  always #HALF clk = ~clk;

  task step;
    begin
      @(posedge clk);
      #SETTLE;
    end
  endtask

  task splat(output [LANES*16-1:0] bus, input [15:0] v);
    integer k;
    begin
      for (k = 0; k < LANES; k = k + 1) bus[k*16 +: 16] = v;
    end
  endtask

  task check_lanes(input [15:0] want, input [8*16-1:0] tag);
    integer k;
    begin
      for (k = 0; k < LANES; k = k + 1)
        if (out_data[k*16 +: 16] !== want) begin
          $display("FAIL %0s lane %0d = %h (expected %h)", tag, k,
                   out_data[k*16 +: 16], want);
          errors = errors + 1;
        end
    end
  endtask

  initial begin
    clk = 0; rst_n = 0; in_valid = 0; acc_clear = 0; errors = 0;
    uop0 = OP_NOP; uop1 = OP_NOP; uop2 = OP_NOP;
    operand0 = 0; operand1 = 0; operand2 = 0; in_data = 0;
    repeat (2) step;
    rst_n = 1;
    step;

    // ---- fused residual add + ReLU: negative result is clamped ------- //
    uop0 = OP_ADD; uop1 = OP_RELU; uop2 = OP_NOP;
    splat(operand0, F1);
    splat(in_data, FM3);                 // -3 + 1 = -2 -> ReLU -> 0
    in_valid = 1;
    step;
    in_valid = 0;
    if (!out_valid) begin
      $display("FAIL out_valid not asserted one cycle after in_valid");
      errors = errors + 1;
    end
    check_lanes(16'h0000, "relu(-2)");

    // ---- same micro-program on a positive input --------------------- //
    splat(in_data, F2);                  //  2 + 1 = 3 -> ReLU -> 3
    in_valid = 1;
    step;
    in_valid = 0;
    check_lanes(F3, "relu(+3)");
    step;
    if (out_valid) begin
      $display("FAIL out_valid held after the stream stopped");
      errors = errors + 1;
    end

    // ---- scale micro-program ---------------------------------------- //
    uop0 = OP_MUL; uop1 = OP_NOP; uop2 = OP_NOP;
    splat(operand0, F2);
    splat(in_data, F3);                  // 3 * 2 = 6
    in_valid = 1;
    step;
    in_valid = 0;
    check_lanes(16'h4600, "scale");      // 6.0

    // ---- lane reduction (softmax denominator / layer-norm mean) ------ //
    uop0 = OP_NOP; uop1 = OP_NOP; uop2 = OP_ACC;
    acc_clear = 1;
    step;
    acc_clear = 0;
    if (acc_scalar !== 16'h0000) begin
      $display("FAIL accumulator not cleared (%h)", acc_scalar);
      errors = errors + 1;
    end
    splat(in_data, F1);                  // 16 lanes x 1.0 = 16.0 per burst
    in_valid = 1;
    step;
    if (acc_scalar !== F16) begin
      $display("FAIL reduction after one burst = %h (expected %h)",
               acc_scalar, F16);
      errors = errors + 1;
    end
    step;                                // second burst -> 32.0
    in_valid = 0;
    if (acc_scalar !== F32) begin
      $display("FAIL reduction after two bursts = %h (expected %h)",
               acc_scalar, F32);
      errors = errors + 1;
    end
    check_lanes(F1, "acc passthrough");

    if (errors == 0) $display("tb_sfu_pipeline: ALL PASS");
    else $display("tb_sfu_pipeline: %0d FAILURES", errors);
    $finish;
  end

endmodule
