// ============================================================================
// pe_cluster: one bank group's complete PIM datapath (Fig. microarch (b)).
//
// Integrates the four bank readout ports, the 4:1 time-multiplexed
// broadcast arbiter, the four operand FIFOs, and the four 16-lane PEs of a
// bank group, under the channel mode register. This is the unit whose area
// overhead is reported in Sec. V-B (arbiter + FIFOs + mode logic on top of
// the proven MAC core), and the top level used for the bank-group
// synthesis runs.
// ============================================================================
`timescale 1ns / 1ps

module pe_cluster (
    input  wire         clk,
    input  wire         rst_n,
    // channel mode register
    input  wire         mrw_valid,
    input  wire         mrw_mode,
    output wire         mode_broadcast,
    output wire         switch_busy,
    // bank readout ports (one 32 B burst per grant)
    input  wire [3:0]   rd_valid,
    output wire [3:0]   rd_grant,
    input  wire [255:0] rd_data0,
    input  wire [255:0] rd_data1,
    input  wire [255:0] rd_data2,
    input  wire [255:0] rd_data3,
    // per-PE vector operands (input-vector slices, loaded per pass)
    input  wire [3:0]   vec_we,
    input  wire [255:0] vec_data,
    // accumulator control / results
    input  wire [3:0]   acc_clear,
    output wire [63:0]  acc_out,       // 4 x FP16 reduction results
    output wire [3:0]   acc_valid
);

  // FIFO drain status feeds the mode controller
  wire [3:0] fifo_out_valid;
  wire fifos_empty = ~(|fifo_out_valid);

  mode_ctrl u_mode (
      .clk(clk), .rst_n(rst_n),
      .mrw_valid(mrw_valid), .mrw_mode(mrw_mode),
      .fifos_empty(fifos_empty),
      .mode_broadcast(mode_broadcast), .switch_busy(switch_busy));

  // broadcast arbiter grants the shared bus and pushes into the FIFOs
  wire [3:0]   pe_push;
  wire [255:0] pe_data;
  wire [3:0]   fifo_afull;

  bg_arbiter u_arb (
      .clk(clk), .rst_n(rst_n), .mode_broadcast(mode_broadcast),
      .rd_valid(rd_valid & {4{~switch_busy}}),   // hold during reconfig
      .rd_grant(rd_grant),
      .rd_data0(rd_data0), .rd_data1(rd_data1),
      .rd_data2(rd_data2), .rd_data3(rd_data3),
      .pe_push(pe_push), .pe_data(pe_data), .pe_afull(fifo_afull));

  genvar gi;
  generate
    for (gi = 0; gi < 4; gi = gi + 1) begin : g_pe
      wire        f_out_valid;
      wire [255:0] f_out_data;
      wire        pe_ready;

      operand_fifo u_fifo (
          .clk(clk), .rst_n(rst_n),
          .in_valid(pe_push[gi]),
          .in_ready(),                        // afull provides back-pressure
          .in_data(pe_data),
          .out_valid(f_out_valid),
          .out_ready(pe_ready),
          .out_data(f_out_data),
          .afull(fifo_afull[gi]));

      assign fifo_out_valid[gi] = f_out_valid;

      pe_mac16 u_pe (
          .clk(clk), .rst_n(rst_n),
          .op_valid(f_out_valid),
          .op_ready(pe_ready),
          .op_data(f_out_data),
          .vec_we(vec_we[gi]),
          .vec_data(vec_data),
          .acc_clear(acc_clear[gi]),
          .acc_out(acc_out[gi*16 +: 16]),
          .acc_valid(acc_valid[gi]));
    end
  endgenerate

endmodule
