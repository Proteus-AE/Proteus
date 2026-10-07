// ============================================================================
// fp16_mul: IEEE-754 binary16 multiplier (round-to-nearest-even, flush
// denormals to zero). Single-cycle combinational core; the MAC lane
// registers the result.
//
// Part of the per-bank PE datapath (Fig. microarch (c)). Synthesized with
// the 28 nm flow of Sec. V-A; area/power reports feed the energy model.
// ============================================================================
`timescale 1ns / 1ps

module fp16_mul (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] y
);

  // field extraction
  wire        sa = a[15];
  wire        sb = b[15];
  wire [4:0]  ea = a[14:10];
  wire [4:0]  eb = b[14:10];
  wire [9:0]  fa = a[9:0];
  wire [9:0]  fb = b[9:0];

  wire a_zero = (ea == 5'd0);            // denormals flushed to zero
  wire b_zero = (eb == 5'd0);
  wire a_inf  = (ea == 5'd31) && (fa == 10'd0);
  wire b_inf  = (eb == 5'd31) && (fb == 10'd0);
  wire a_nan  = (ea == 5'd31) && (fa != 10'd0);
  wire b_nan  = (eb == 5'd31) && (fb != 10'd0);

  wire        sy = sa ^ sb;

  // significand multiply: 11x11 -> 22 bits
  wire [10:0] ma = {1'b1, fa};
  wire [10:0] mb = {1'b1, fb};
  wire [21:0] prod = ma * mb;

  // normalization: product in [1,4) -> possible 1-bit shift
  wire        carry = prod[21];
  wire [21:0] norm = carry ? prod : (prod << 1);
  // norm[21] is the hidden bit; norm[20:11] is the fraction field
  wire [9:0]  frac_pre = norm[20:11];
  wire [10:0] rem = {norm[10:0]};        // guard/round/sticky region

  // round to nearest even
  wire guard  = rem[10];
  wire sticky = |rem[9:0];
  wire lsb    = frac_pre[0];
  wire round_up = guard && (sticky || lsb);
  wire [10:0] frac_rnd = {1'b0, frac_pre} + {10'd0, round_up};
  wire        frac_ovf = frac_rnd[10];

  // exponent arithmetic (bias 15)
  wire signed [7:0] exp_sum =
      $signed({3'b000, ea}) + $signed({3'b000, eb}) - 8'sd15 +
      {7'd0, carry} + {7'd0, frac_ovf};

  wire underflow = exp_sum <= 0;
  wire overflow  = exp_sum >= 8'sd31;

  wire [9:0] frac_out = frac_ovf ? frac_rnd[10:1] : frac_rnd[9:0];

  assign y = (a_nan || b_nan)                     ? 16'h7E00 :  // qNaN
             ((a_inf && b_zero) || (b_inf && a_zero)) ? 16'h7E00 :
             (a_inf || b_inf)                     ? {sy, 5'd31, 10'd0} :
             (a_zero || b_zero)                   ? {sy, 15'd0} :
             overflow                             ? {sy, 5'd31, 10'd0} :
             underflow                            ? {sy, 15'd0} :
             {sy, exp_sum[4:0], frac_out};

endmodule
