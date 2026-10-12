// Self-checking testbench for the all-bank command decoder FSM.
//
// Stimulus is applied one settle delay *after* the sampling edge and sampled
// again before the next one, so no input changes at a clock edge.
`timescale 1ns / 1ps

module tb_pim_cmd_decoder;

  reg clk, rst_n, cmd_valid, cmd_mode;
  reg [2:0] cmd_op;
  reg [15:0] cmd_row;
  reg [5:0] cmd_col;
  wire bank_act, bank_rdmac, bank_wr, bank_pre, mrw_valid, mrw_mode;
  wire [15:0] bank_row;
  wire [5:0] bank_col;
  wire row_open, illegal_cmd;
  integer errors;

  pim_cmd_decoder dut (
      .clk(clk), .rst_n(rst_n),
      .cmd_valid(cmd_valid), .cmd_op(cmd_op), .cmd_row(cmd_row),
      .cmd_col(cmd_col), .cmd_mode(cmd_mode),
      .bank_act(bank_act), .bank_row(bank_row), .bank_rdmac(bank_rdmac),
      .bank_wr(bank_wr), .bank_col(bank_col), .bank_pre(bank_pre),
      .mrw_valid(mrw_valid), .mrw_mode(mrw_mode),
      .row_open(row_open), .illegal_cmd(illegal_cmd));

  localparam HALF   = 0.5;
  localparam SETTLE = 0.1;

  always #HALF clk = ~clk;

  task step;
    begin
      @(posedge clk);
      #SETTLE;
    end
  endtask

  task issue(input [2:0] op, input [15:0] row, input [5:0] col,
             input mode);
    begin
      cmd_valid = 1; cmd_op = op; cmd_row = row; cmd_col = col;
      cmd_mode = mode;
      step;                    // the decoder registers the command here
      cmd_valid = 0;
    end
  endtask

  initial begin
    clk = 0; rst_n = 0; cmd_valid = 0; cmd_op = 0;
    cmd_row = 0; cmd_col = 0; cmd_mode = 0; errors = 0;
    repeat (2) step;
    rst_n = 1;
    step;

    // column command with no open row is illegal
    issue(3'd2, 16'd0, 6'd0, 1'b0);          // RDMAC
    if (!illegal_cmd || bank_rdmac) begin
      $display("FAIL RDMAC accepted with row closed");
      errors = errors + 1;
    end

    // ACT then RDMAC
    issue(3'd1, 16'd42, 6'd0, 1'b0);         // ACT row 42
    if (!bank_act || bank_row !== 16'd42) begin
      $display("FAIL ACT not decoded");
      errors = errors + 1;
    end
    issue(3'd2, 16'd42, 6'd7, 1'b0);         // RDMAC col 7
    if (!bank_rdmac || bank_col !== 6'd7 || illegal_cmd) begin
      $display("FAIL RDMAC not decoded with open row");
      errors = errors + 1;
    end

    // WR then PRE closes the row
    issue(3'd3, 16'd42, 6'd63, 1'b0);        // WR (KV append)
    if (!bank_wr || bank_col !== 6'd63) begin
      $display("FAIL WR not decoded");
      errors = errors + 1;
    end
    issue(3'd4, 16'd0, 6'd0, 1'b0);          // PRE
    if (!bank_pre) begin
      $display("FAIL PRE not decoded");
      errors = errors + 1;
    end
    issue(3'd2, 16'd42, 6'd0, 1'b0);         // RDMAC after PRE -> illegal
    if (!illegal_cmd) begin
      $display("FAIL RDMAC accepted after PRE");
      errors = errors + 1;
    end

    // MRW forwards to mode_ctrl
    issue(3'd5, 16'd0, 6'd0, 1'b1);
    if (!mrw_valid || !mrw_mode) begin
      $display("FAIL MRW not forwarded");
      errors = errors + 1;
    end

    if (errors == 0) $display("tb_pim_cmd_decoder: ALL PASS");
    else $display("tb_pim_cmd_decoder: %0d FAILURES", errors);
    $finish;
  end

endmodule
