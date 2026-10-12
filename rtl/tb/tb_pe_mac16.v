// Self-checking testbench for the 16-lane PE: dot-product of exact FP16
// vectors through the MAC array and the reduction tree.
//
// Stimulus is applied one settle delay *after* the sampling edge and sampled
// again before the next one, so no input changes at a clock edge.
`timescale 1ns / 1ps

module tb_pe_mac16;

  localparam HALF   = 0.5;
  localparam SETTLE = 0.1;

  reg clk, rst_n;
  reg op_valid;
  wire op_ready;
  reg [255:0] op_data;
  reg vec_we;
  reg [255:0] vec_data;
  reg acc_clear;
  wire [15:0] acc_out;
  wire acc_valid;
  integer errors, i;

  pe_mac16 dut (
      .clk(clk), .rst_n(rst_n),
      .op_valid(op_valid), .op_ready(op_ready), .op_data(op_data),
      .vec_we(vec_we), .vec_data(vec_data),
      .acc_clear(acc_clear), .acc_out(acc_out), .acc_valid(acc_valid));

  always #HALF clk = ~clk;

  task step;
    begin
      @(posedge clk);
      #SETTLE;
    end
  endtask

  initial begin
    clk = 0; rst_n = 0; op_valid = 0; vec_we = 0; acc_clear = 0;
    op_data = 0; vec_data = 0; errors = 0;
    repeat (2) step;
    rst_n = 1;
    step;

    // vector = all 1.0; operand burst = all 1.0
    // one burst -> lane accs all 1.0 -> reduction = 16.0 (0x4C00)
    for (i = 0; i < 16; i = i + 1) vec_data[i*16 +: 16] = 16'h3C00;
    vec_we = 1;
    step;
    vec_we = 0;

    acc_clear = 1;
    step;
    acc_clear = 0;

    for (i = 0; i < 16; i = i + 1) op_data[i*16 +: 16] = 16'h3C00;
    op_valid = 1;
    step;
    op_valid = 0;
    if (!acc_valid || acc_out !== 16'h4C00) begin
      $display("FAIL dot16(1,1) = %h (expected 4C00 = 16.0)", acc_out);
      errors = errors + 1;
    end

    // second burst of all 2.0 -> accs = 1 + 2 = 3 -> reduction 48.0 (0x5200)
    for (i = 0; i < 16; i = i + 1) op_data[i*16 +: 16] = 16'h4000;
    op_valid = 1;
    step;
    op_valid = 0;
    if (acc_out !== 16'h5200) begin
      $display("FAIL accumulate = %h (expected 5200 = 48.0)", acc_out);
      errors = errors + 1;
    end

    // clear resets the accumulators
    acc_clear = 1;
    step;
    acc_clear = 0;
    if (acc_valid) begin
      $display("FAIL acc_valid after clear");
      errors = errors + 1;
    end
    if (acc_out !== 16'h0000) begin
      $display("FAIL accumulators not cleared (%h)", acc_out);
      errors = errors + 1;
    end

    if (errors == 0) $display("tb_pe_mac16: ALL PASS");
    else $display("tb_pe_mac16: %0d FAILURES", errors);
    $finish;
  end

endmodule
