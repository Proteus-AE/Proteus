// ============================================================================
// arbiter_4to1: bank-group 4:1 broadcast datapath (Fig. microarch (b)).
//
// Grants the time-multiplexed 32 B bank-group bus to one of the four bank
// readouts per slot (round-robin among requesters). In direct mode the
// granted burst is forwarded only to the requesting bank's own PE FIFO; in
// broadcasting mode it is forwarded to all four PE FIFOs of the group
// (one-to-many connectivity), stalling the grant while any target FIFO is
// almost-full (back-pressure).
// ============================================================================
`timescale 1ns / 1ps

module arbiter_4to1 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         mode_broadcast,   // from mode_ctrl
    // bank readout requests
    input  wire [3:0]   rd_valid,
    output reg  [3:0]   rd_grant,
    input  wire [255:0] rd_data0,
    input  wire [255:0] rd_data1,
    input  wire [255:0] rd_data2,
    input  wire [255:0] rd_data3,
    // per-PE FIFO enqueue ports
    output reg  [3:0]   pe_push,
    output reg  [255:0] pe_data,
    input  wire [3:0]   pe_afull
);

  reg [1:0] rr_ptr;   // round-robin pointer

  // requester selection: first valid at or after rr_ptr
  reg  [1:0] sel;
  reg        sel_valid;
  integer k;
  always @(*) begin
    sel = 2'd0;
    sel_valid = 1'b0;
    for (k = 0; k < 4; k = k + 1) begin
      if (!sel_valid && rd_valid[(rr_ptr + k[1:0]) & 2'd3]) begin
        sel = (rr_ptr + k[1:0]) & 2'd3;
        sel_valid = 1'b1;
      end
    end
  end

  // back-pressure: broadcast requires all targets non-full, direct only one
  wire stall = mode_broadcast ? |pe_afull : pe_afull[sel];

  always @(*) begin
    rd_grant = 4'd0;
    pe_push  = 4'd0;
    pe_data  = 256'd0;
    if (sel_valid && !stall) begin
      rd_grant[sel] = 1'b1;
      pe_data = (sel == 2'd0) ? rd_data0 :
                (sel == 2'd1) ? rd_data1 :
                (sel == 2'd2) ? rd_data2 : rd_data3;
      pe_push = mode_broadcast ? 4'b1111 : (4'b0001 << sel);
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) rr_ptr <= 2'd0;
    else if (sel_valid && !stall) rr_ptr <= sel + 2'd1;
  end

endmodule
