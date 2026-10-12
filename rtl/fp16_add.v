// ============================================================================
// fp16_add: IEEE-754 binary16 adder (round-to-nearest-even, flush denormals
// to zero). Combinational; used by the MAC accumulate path and by the SFU
// reduction tree.
// ============================================================================
`timescale 1ns / 1ps

module fp16_add (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] y
);

  wire        sa = a[15];
  wire        sb = b[15];
  wire [4:0]  ea = a[14:10];
  wire [4:0]  eb = b[14:10];
  wire [9:0]  fa = a[9:0];
  wire [9:0]  fb = b[9:0];

  wire a_zero = (ea == 5'd0);
  wire b_zero = (eb == 5'd0);
  wire a_inf  = (ea == 5'd31) && (fa == 10'd0);
  wire b_inf  = (eb == 5'd31) && (fb == 10'd0);
  wire a_nan  = (ea == 5'd31) && (fa != 10'd0);
  wire b_nan  = (eb == 5'd31) && (fb != 10'd0);

  // operand swap so |x| >= |y|
  wire swap = ({ea, fa} < {eb, fb});
  wire [4:0] ex = swap ? eb : ea;
  wire [4:0] ey = swap ? ea : eb;
  wire [9:0] fx = swap ? fb : fa;
  wire [9:0] fy = swap ? fa : fb;
  wire       sx = swap ? sb : sa;
  wire       sy_ = swap ? sa : sb;

  // significands with hidden bit, 3 extra alignment bits (G/R/S)
  wire [13:0] mx = {1'b1, fx, 3'b000};
  wire [13:0] my_pre = {1'b1, fy, 3'b000};
  wire [4:0]  shift = ex - ey;
  wire [13:0] my_shifted = (shift > 5'd13) ? 14'd1 : (my_pre >> shift);
  // preserve sticky on large shifts
  wire        sticky_lost = (shift > 5'd13) ? |my_pre :
                            |(my_pre & ((14'd1 << shift) - 14'd1));
  wire [13:0] my = my_shifted | {13'd0, sticky_lost};

  wire sub = sx ^ sy_;
  wire [14:0] sum = sub ? ({1'b0, mx} - {1'b0, my})
                        : ({1'b0, mx} + {1'b0, my});

  // leading-zero count for normalization (15-bit priority encoder)
  reg [3:0] lz;
  integer i;
  always @(*) begin
    lz = 4'd15;
    for (i = 14; i >= 0; i = i - 1)
      if (sum[i] && lz == 4'd15) lz = 4'd14 - i[3:0];
  end

  wire sum_zero = (sum == 15'd0);
  // normalize: align hidden bit to position 13
  wire [14:0] norm = (lz == 4'd0) ? (sum >> 1) : (sum << (lz - 1));
  wire signed [7:0] exp_adj =
      $signed({3'b000, ex}) + 8'sd1 - $signed({4'b0000, lz});

  // round to nearest even from the 3 alignment bits
  wire [9:0] frac_pre = norm[12:3];
  wire guard  = norm[2];
  wire sticky = |norm[1:0];
  wire round_up = guard && (sticky || frac_pre[0]);
  wire [10:0] frac_rnd = {1'b0, frac_pre} + {10'd0, round_up};
  wire        frac_ovf = frac_rnd[10];
  wire signed [7:0] exp_fin = exp_adj + {7'd0, frac_ovf};
  wire [9:0] frac_fin = frac_ovf ? frac_rnd[10:1] : frac_rnd[9:0];

  assign y = (a_nan || b_nan)                    ? 16'h7E00 :
             (a_inf && b_inf && (sa != sb))      ? 16'h7E00 :
             a_inf                               ? a :
             b_inf                               ? b :
             a_zero                              ? (b_zero ? 16'd0 : b) :
             b_zero                              ? a :
             sum_zero                            ? 16'd0 :
             (exp_fin >= 8'sd31)                 ? {sx, 5'd31, 10'd0} :
             (exp_fin <= 8'sd0)                  ? {sx, 15'd0} :
             {sx, exp_fin[4:0], frac_fin};

endmodule
