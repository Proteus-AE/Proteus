// Self-checking testbench for the connectivity mode register: switches must
// wait for the FIFO-drain handshake and ignore redundant writes.
`timescale 1ns / 1ps

module tb_mode_ctrl;

  reg clk, rst_n, mrw_valid, mrw_mode, fifos_empty;
  wire mode_broadcast, switch_busy;
  integer errors;

  mode_ctrl dut (
      .clk(clk), .rst_n(rst_n),
      .mrw_valid(mrw_valid), .mrw_mode(mrw_mode),
      .fifos_empty(fifos_empty),
      .mode_broadcast(mode_broadcast), .switch_busy(switch_busy));

  always #0.5 clk = ~clk;

  initial begin
    clk = 0; rst_n = 0; mrw_valid = 0; mrw_mode = 0; fifos_empty = 0;
    errors = 0;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // request broadcast while FIFOs are busy: pending, not applied
    mrw_valid = 1; mrw_mode = 1;
    @(posedge clk);
    mrw_valid = 0;
    #0.1;
    if (mode_broadcast || !switch_busy) begin
      $display("FAIL switch applied before drain");
      errors = errors + 1;
    end
    repeat (3) @(posedge clk);
    #0.1;
    if (mode_broadcast) begin
      $display("FAIL switch applied while FIFOs busy");
      errors = errors + 1;
    end

    // drain completes -> mode flips, busy clears
    fifos_empty = 1;
    @(posedge clk);
    #0.1;
    if (!mode_broadcast || switch_busy) begin
      $display("FAIL switch not applied after drain");
      errors = errors + 1;
    end

    // redundant write to the same mode: no pending switch
    mrw_valid = 1; mrw_mode = 1;
    @(posedge clk);
    mrw_valid = 0;
    #0.1;
    if (switch_busy) begin
      $display("FAIL redundant MRW raised switch_busy");
      errors = errors + 1;
    end

    if (errors == 0) $display("tb_mode_ctrl: ALL PASS");
    else $display("tb_mode_ctrl: %0d FAILURES", errors);
    $finish;
  end

endmodule
