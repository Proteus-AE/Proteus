// Self-checking testbench for the 4-entry operand FIFO: fill/drain,
// almost-full back-pressure, FIFO ordering.
//
// Stimulus is applied one settle delay *after* the sampling edge and sampled
// again before the next one, so no input changes at a clock edge.
`timescale 1ns / 1ps

module tb_operand_fifo;

  localparam HALF   = 0.5;    // clock half period (ns)
  localparam SETTLE = 0.1;    // post-edge settle window (ns)

  reg clk, rst_n;
  reg in_valid;
  wire in_ready;
  reg [255:0] in_data;
  wire out_valid;
  reg out_ready;
  wire [255:0] out_data;
  wire afull;
  integer errors, i;

  operand_fifo dut (
      .clk(clk), .rst_n(rst_n),
      .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),
      .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data),
      .afull(afull));

  always #HALF clk = ~clk;

  task step;                  // advance one cycle, then let outputs settle
    begin
      @(posedge clk);
      #SETTLE;
    end
  endtask

  initial begin
    clk = 0; rst_n = 0; in_valid = 0; out_ready = 0; in_data = 0;
    errors = 0;
    repeat (4) step;
    rst_n = 1;
    step;

    // fill to capacity; afull must assert once three entries are resident
    for (i = 0; i < 4; i = i + 1) begin
      in_valid = 1;
      in_data = {248'd0, i[7:0]};
      step;
      if (i == 2 && !afull) begin
        $display("FAIL afull not asserted at 3 entries");
        errors = errors + 1;
      end
    end
    in_valid = 0;
    #SETTLE;
    if (in_ready) begin
      $display("FAIL in_ready high on a full FIFO");
      errors = errors + 1;
    end

    // drain and verify order
    out_ready = 1;
    for (i = 0; i < 4; i = i + 1) begin
      if (!out_valid || out_data[7:0] !== i[7:0]) begin
        $display("FAIL drained %h expected %0d", out_data[7:0], i);
        errors = errors + 1;
      end
      step;
    end
    out_ready = 0;
    #SETTLE;
    if (out_valid) begin
      $display("FAIL out_valid high on an empty FIFO");
      errors = errors + 1;
    end

    if (errors == 0) $display("tb_operand_fifo: ALL PASS");
    else $display("tb_operand_fifo: %0d FAILURES", errors);
    $finish;
  end

endmodule
