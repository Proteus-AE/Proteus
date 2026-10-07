// ============================================================================
// add_tree: 16 -> 1 FP16 reduction tree of the per-bank PE
// (Fig. microarch (c)).
//
// Four levels of combinational FP16 adders collapse the 16 lane
// accumulators into the output element. The whole tree is evaluated inside
// one 1 GHz cycle, so each level owns roughly a quarter of the period --
// the tightest path of the PE, and the reason its adders synthesize larger
// than the accumulate adders of mul_array.
// ============================================================================
`timescale 1ns / 1ps

module add_tree #(
    parameter LANES = 16
) (
    input  wire [LANES*16-1:0] lane_in,
    output wire [15:0]         sum
);

  wire [15:0] l1 [0:7];
  wire [15:0] l2 [0:3];
  wire [15:0] l3 [0:1];

  genvar gi;
  generate
    for (gi = 0; gi < 8; gi = gi + 1) begin : g_lvl1
      fp16_add u (.a(lane_in[(2*gi)*16 +: 16]),
                  .b(lane_in[(2*gi+1)*16 +: 16]), .y(l1[gi]));
    end
    for (gi = 0; gi < 4; gi = gi + 1) begin : g_lvl2
      fp16_add u (.a(l1[2*gi]), .b(l1[2*gi+1]), .y(l2[gi]));
    end
    for (gi = 0; gi < 2; gi = gi + 1) begin : g_lvl3
      fp16_add u (.a(l2[2*gi]), .b(l2[2*gi+1]), .y(l3[gi]));
    end
  endgenerate

  fp16_add u_lvl4 (.a(l3[0]), .b(l3[1]), .y(sum));

endmodule
