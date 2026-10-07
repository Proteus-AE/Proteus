// Self-checking testbench for the bank-group broadcast arbiter:
// round-robin grants in direct mode, one-to-many forwarding in broadcast
// mode, and almost-full back-pressure in both.
`timescale 1ns / 1ps

module tb_arbiter_4to1;

  reg clk, rst_n, mode_broadcast;
  reg  [3:0] rd_valid;
  wire [3:0] rd_grant;
  reg  [255:0] rd_data0, rd_data1, rd_data2, rd_data3;
  wire [3:0] pe_push;
  wire [255:0] pe_data;
  reg  [3:0] pe_afull;
  integer errors;

  arbiter_4to1 dut (
      .clk(clk), .rst_n(rst_n), .mode_broadcast(mode_broadcast),
      .rd_valid(rd_valid), .rd_grant(rd_grant),
      .rd_data0(rd_data0), .rd_data1(rd_data1),
      .rd_data2(rd_data2), .rd_data3(rd_data3),
      .pe_push(pe_push), .pe_data(pe_data), .pe_afull(pe_afull));

  always #0.5 clk = ~clk;

  initial begin
    clk = 0; rst_n = 0; mode_broadcast = 0;
    rd_valid = 0; pe_afull = 0;
    rd_data0 = 256'd10; rd_data1 = 256'd11;
    rd_data2 = 256'd12; rd_data3 = 256'd13;
    errors = 0;
    repeat (2) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    // direct mode: single grant, push targets the granting bank's PE
    rd_valid = 4'b1111;
    #0.1;
    if (rd_grant !== 4'b0001 || pe_push !== 4'b0001 ||
        pe_data[7:0] !== 8'd10) begin
      $display("FAIL direct grant0: grant=%b push=%b", rd_grant, pe_push);
      errors = errors + 1;
    end
    @(posedge clk); #0.1;              // round-robin advances
    if (rd_grant !== 4'b0010 || pe_push !== 4'b0010) begin
      $display("FAIL direct grant1: grant=%b push=%b", rd_grant, pe_push);
      errors = errors + 1;
    end

    // broadcast mode: one grant fans out to all four PEs
    mode_broadcast = 1;
    @(posedge clk); #0.1;
    if (pe_push !== 4'b1111) begin
      $display("FAIL broadcast fan-out: push=%b", pe_push);
      errors = errors + 1;
    end

    // broadcast back-pressure: any almost-full PE stalls the grant
    pe_afull = 4'b0100;
    #0.1;
    if (rd_grant !== 4'b0000 || pe_push !== 4'b0000) begin
      $display("FAIL broadcast stall: grant=%b push=%b", rd_grant, pe_push);
      errors = errors + 1;
    end
    pe_afull = 4'b0000;

    // direct-mode back-pressure only from the selected PE
    mode_broadcast = 0;
    rd_valid = 4'b0001;
    pe_afull = 4'b1110;                // others full, target free
    #0.1;
    if (rd_grant !== 4'b0001) begin
      $display("FAIL direct selective stall: grant=%b", rd_grant);
      errors = errors + 1;
    end

    if (errors == 0) $display("tb_arbiter_4to1: ALL PASS");
    else $display("tb_arbiter_4to1: %0d FAILURES", errors);
    $finish;
  end

endmodule
