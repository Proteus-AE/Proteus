// ============================================================================
// mode_ctrl: per-channel connectivity mode register (Sec. IV-C "Lightweight
// Reconfiguration").
//
// A mode-register write over the standard command/address path requests a
// connectivity switch; the new mode takes effect only after every in-flight
// operand FIFO of the channel has drained (`fifos_empty`), so no data path
// is reconfigured under in-flight bursts. `switch_busy` spans the drain
// window (tens of ns in practice).
// ============================================================================
`timescale 1ns / 1ps

module mode_ctrl (
    input  wire clk,
    input  wire rst_n,
    // mode-register write port (from the command decoder)
    input  wire mrw_valid,
    input  wire mrw_mode,          // 0 = direct, 1 = broadcast
    // datapath status
    input  wire fifos_empty,
    // outputs
    output reg  mode_broadcast,
    output wire switch_busy
);

  reg pending;
  reg pending_mode;

  assign switch_busy = pending;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mode_broadcast <= 1'b0;
      pending <= 1'b0;
      pending_mode <= 1'b0;
    end else begin
      if (mrw_valid && (mrw_mode != mode_broadcast) && !pending) begin
        pending <= 1'b1;
        pending_mode <= mrw_mode;
      end else if (pending && fifos_empty) begin
        mode_broadcast <= pending_mode;
        pending <= 1'b0;
      end
    end
  end

endmodule
