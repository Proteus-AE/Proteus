// ============================================================================
// pim_cmd_decoder: all-bank PIM command decoder (per channel).
//
// Latches encoded commands from the standard command/address path and
// sequences the per-bank control signals of the all-bank execution model
// (Sec. IV-C): ACT_AB opens `row` in every bank (staggered externally by
// the DRAM core's tRRD/tFAW circuitry), RDMAC_AB triggers one column read
// per bank into the PE datapath, WR_AB drives the in-place KV append,
// PRE_AB closes all rows, and MRW forwards connectivity-mode writes to
// mode_ctrl. A simple two-state FSM enforces that column commands are only
// dispatched while a row is open.
// ============================================================================
`timescale 1ns / 1ps

module pim_cmd_decoder #(
    parameter ROW_BITS = 16,
    parameter COL_BITS = 6
) (
    input  wire                clk,
    input  wire                rst_n,
    // encoded command in (valid for one cycle)
    input  wire                cmd_valid,
    input  wire [2:0]          cmd_op,       // see localparams
    input  wire [ROW_BITS-1:0] cmd_row,
    input  wire [COL_BITS-1:0] cmd_col,
    input  wire                cmd_mode,     // MRW payload
    // per-bank control fan-out (all-bank: single wire set, banks stagger)
    output reg                 bank_act,
    output reg  [ROW_BITS-1:0] bank_row,
    output reg                 bank_rdmac,
    output reg                 bank_wr,
    output reg  [COL_BITS-1:0] bank_col,
    output reg                 bank_pre,
    // mode-control interface
    output reg                 mrw_valid,
    output reg                 mrw_mode,
    // status
    output reg                 row_open,
    output reg                 illegal_cmd   // column command with row closed
);

  localparam OP_NOP   = 3'd0;
  localparam OP_ACT   = 3'd1;
  localparam OP_RDMAC = 3'd2;
  localparam OP_WR    = 3'd3;
  localparam OP_PRE   = 3'd4;
  localparam OP_MRW   = 3'd5;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bank_act   <= 1'b0;
      bank_row   <= {ROW_BITS{1'b0}};
      bank_rdmac <= 1'b0;
      bank_wr    <= 1'b0;
      bank_col   <= {COL_BITS{1'b0}};
      bank_pre   <= 1'b0;
      mrw_valid  <= 1'b0;
      mrw_mode   <= 1'b0;
      row_open   <= 1'b0;
      illegal_cmd <= 1'b0;
    end else begin
      // defaults: one-cycle pulses
      bank_act   <= 1'b0;
      bank_rdmac <= 1'b0;
      bank_wr    <= 1'b0;
      bank_pre   <= 1'b0;
      mrw_valid  <= 1'b0;
      illegal_cmd <= 1'b0;
      if (cmd_valid) begin
        case (cmd_op)
          OP_ACT: begin
            bank_act <= 1'b1;
            bank_row <= cmd_row;
            row_open <= 1'b1;
          end
          OP_RDMAC: begin
            if (row_open) begin
              bank_rdmac <= 1'b1;
              bank_col   <= cmd_col;
            end else begin
              illegal_cmd <= 1'b1;
            end
          end
          OP_WR: begin
            if (row_open) begin
              bank_wr  <= 1'b1;
              bank_col <= cmd_col;
            end else begin
              illegal_cmd <= 1'b1;
            end
          end
          OP_PRE: begin
            bank_pre <= 1'b1;
            row_open <= 1'b0;
          end
          OP_MRW: begin
            mrw_valid <= 1'b1;
            mrw_mode  <= cmd_mode;
          end
          default: ;   // NOP
        endcase
      end
    end
  end

endmodule
