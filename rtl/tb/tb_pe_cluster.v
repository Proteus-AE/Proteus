// Self-checking testbench for the bank-group cluster: direct-mode dot
// products on independent PEs, then a broadcast-mode pass where one bank's
// burst reaches all four PEs of the group.
//
// Stimulus is applied one settle delay *after* the sampling edge and sampled
// again before the next one, so no input changes at a clock edge.
`timescale 1ns / 1ps

module tb_pe_cluster;

  localparam HALF   = 0.5;
  localparam SETTLE = 0.1;

  reg clk, rst_n, mrw_valid, mrw_mode;
  wire mode_broadcast, switch_busy;
  reg  [3:0] rd_valid;
  wire [3:0] rd_grant;
  reg  [255:0] rd_data0, rd_data1, rd_data2, rd_data3;
  reg  [3:0] vec_we;
  reg  [255:0] vec_data;
  reg  [3:0] acc_clear;
  wire [63:0] acc_out;
  wire [3:0] acc_valid;
  integer errors, i;

  pe_cluster dut (
      .clk(clk), .rst_n(rst_n),
      .mrw_valid(mrw_valid), .mrw_mode(mrw_mode),
      .mode_broadcast(mode_broadcast), .switch_busy(switch_busy),
      .rd_valid(rd_valid), .rd_grant(rd_grant),
      .rd_data0(rd_data0), .rd_data1(rd_data1),
      .rd_data2(rd_data2), .rd_data3(rd_data3),
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
    clk = 0; rst_n = 0; mrw_valid = 0; mrw_mode = 0;
    rd_valid = 0; vec_we = 0; acc_clear = 0; errors = 0;
    rd_data0 = 0; rd_data1 = 0; rd_data2 = 0; rd_data3 = 0;
    vec_data = 0;
    repeat (2) step;
    rst_n = 1;
    step;

    // load all-ones vectors into every PE, then clear the accumulators
    for (i = 0; i < 16; i = i + 1) vec_data[i*16 +: 16] = 16'h3C00;
    vec_we = 4'b1111;
    step;
    vec_we = 0;
    acc_clear = 4'b1111;
    step;
    acc_clear = 0;

    // direct mode: bank 0 streams one all-ones burst to PE 0 only
    for (i = 0; i < 16; i = i + 1) rd_data0[i*16 +: 16] = 16'h3C00;
    rd_valid = 4'b0001;
    step;                          // arbiter grant -> FIFO enqueue
    rd_valid = 0;
    step;                          // FIFO -> PE accumulate
    if (acc_out[15:0] !== 16'h4C00) begin   // dot16(1,1) = 16.0
      $display("FAIL direct PE0 acc = %h (expected 4C00)", acc_out[15:0]);
      errors = errors + 1;
    end
    if (acc_valid[1] || acc_valid[2] || acc_valid[3]) begin
      $display("FAIL direct mode leaked into peer PEs");
      errors = errors + 1;
    end

    // switch to broadcasting; the new mode applies once the FIFOs are drained
    mrw_valid = 1; mrw_mode = 1;
    step;                          // mode_ctrl latches the request
    mrw_valid = 0;
    if (!switch_busy) begin
      $display("FAIL switch_busy not asserted during reconfiguration");
      errors = errors + 1;
    end
    step;                          // FIFOs already empty -> mode applied
    if (!mode_broadcast || switch_busy) begin
      $display("FAIL broadcast mode not applied after drain");
      errors = errors + 1;
    end

    acc_clear = 4'b1111;
    step;
    acc_clear = 0;

    // broadcasting: bank 1 streams one burst; all four PEs must accumulate
    for (i = 0; i < 16; i = i + 1) rd_data1[i*16 +: 16] = 16'h4000; // 2.0
    rd_valid = 4'b0010;
    step;                          // one grant -> four FIFO enqueues
    rd_valid = 0;
    step;                          // FIFOs -> PEs accumulate
    for (i = 0; i < 4; i = i + 1) begin
      if (acc_out[i*16 +: 16] !== 16'h5000) begin  // dot16(2,1) = 32.0
        $display("FAIL broadcast PE%0d acc = %h (expected 5000)", i,
                 acc_out[i*16 +: 16]);
        errors = errors + 1;
      end
    end

    if (errors == 0) $display("tb_pe_cluster: ALL PASS");
    else $display("tb_pe_cluster: %0d FAILURES", errors);
    $finish;
  end

endmodule
