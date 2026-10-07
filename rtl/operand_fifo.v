// ============================================================================
// operand_fifo: 4-entry x 256-bit operand FIFO in front of each per-bank PE
// (Fig. microarch (c)). Depth matches the bank-group fan-in so that in
// broadcasting mode each PE can hold one in-flight operand per peer bank.
// Standard valid/ready handshake on both sides; `afull` drives readout
// back-pressure toward the bank.
// ============================================================================
`timescale 1ns / 1ps

module operand_fifo #(
    parameter WIDTH = 256,        // one 32 B burst
    parameter DEPTH = 4,
    parameter AW    = 2           // log2(DEPTH)
) (
    input  wire             clk,
    input  wire             rst_n,
    // enqueue (from the BG data bus)
    input  wire             in_valid,
    output wire             in_ready,
    input  wire [WIDTH-1:0] in_data,
    // dequeue (to the MAC pipeline)
    output wire             out_valid,
    input  wire             out_ready,
    output wire [WIDTH-1:0] out_data,
    // status
    output wire             afull        // one slot left
);

  reg [WIDTH-1:0] mem [0:DEPTH-1];
  reg [AW:0] wptr, rptr;

  wire [AW:0] count = wptr - rptr;
  wire full  = (count == DEPTH[AW:0]);
  wire empty = (count == 0);

  assign in_ready  = ~full;
  assign out_valid = ~empty;
  assign out_data  = mem[rptr[AW-1:0]];
  assign afull     = (count >= DEPTH[AW:0] - 1);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wptr <= 0;
      rptr <= 0;
    end else begin
      if (in_valid && in_ready) begin
        mem[wptr[AW-1:0]] <= in_data;
        wptr <= wptr + 1'b1;
      end
      if (out_valid && out_ready) begin
        rptr <= rptr + 1'b1;
      end
    end
  end

endmodule
