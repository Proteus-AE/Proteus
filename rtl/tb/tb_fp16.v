// Self-checking testbench for the FP16 arithmetic primitives.
// Run: make tb_fp16 (Icarus Verilog)
`timescale 1ns / 1ps

module tb_fp16;

  reg  [15:0] a, b;
  wire [15:0] ym, ya;
  integer errors;

  fp16_mul u_mul (.a(a), .b(b), .y(ym));
  fp16_add u_add (.a(a), .b(b), .y(ya));

  task check_mul(input [15:0] xa, input [15:0] xb, input [15:0] exp);
    begin
      a = xa; b = xb; #1;
      if (ym !== exp) begin
        $display("FAIL mul %h * %h = %h (expected %h)", xa, xb, ym, exp);
        errors = errors + 1;
      end
    end
  endtask

  task check_add(input [15:0] xa, input [15:0] xb, input [15:0] exp);
    begin
      a = xa; b = xb; #1;
      if (ya !== exp) begin
        $display("FAIL add %h + %h = %h (expected %h)", xa, xb, ya, exp);
        errors = errors + 1;
      end
    end
  endtask

  initial begin
    errors = 0;

    // exact power-of-two and small-integer cases (no rounding involved)
    check_mul(16'h3C00, 16'h3C00, 16'h3C00);  // 1.0 * 1.0 = 1.0
    check_mul(16'h4000, 16'h4000, 16'h4400);  // 2.0 * 2.0 = 4.0
    check_mul(16'h3800, 16'h4000, 16'h3C00);  // 0.5 * 2.0 = 1.0
    check_mul(16'h4200, 16'h4000, 16'h4600);  // 3.0 * 2.0 = 6.0
    check_mul(16'hC000, 16'h4000, 16'hC400);  // -2 * 2 = -4
    check_mul(16'h0000, 16'h5640, 16'h0000);  // 0 * 100 = 0
    check_mul(16'h7C00, 16'h3C00, 16'h7C00);  // inf * 1 = inf
    check_mul(16'h7C00, 16'h0000, 16'h7E00);  // inf * 0 = NaN

    check_add(16'h3C00, 16'h3C00, 16'h4000);  // 1 + 1 = 2
    check_add(16'h4000, 16'h4200, 16'h4500);  // 2 + 3 = 5
    check_add(16'h4400, 16'hC400, 16'h0000);  // 4 + (-4) = 0
    check_add(16'h4200, 16'hBC00, 16'h4000);  // 3 + (-1) = 2
    check_add(16'h0000, 16'h4880, 16'h4880);  // 0 + 9 = 9
    check_add(16'h7C00, 16'hFC00, 16'h7E00);  // inf + -inf = NaN
    check_add(16'h3C00, 16'h3800, 16'h3E00);  // 1 + 0.5 = 1.5

    if (errors == 0) $display("tb_fp16: ALL PASS");
    else $display("tb_fp16: %0d FAILURES", errors);
    $finish;
  end

endmodule
