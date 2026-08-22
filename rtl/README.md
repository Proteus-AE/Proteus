# Proteus PIM RTL

Synthesizable Verilog for the in-DRAM logic of Sec. IV-C, used by the
area/energy methodology of Sec. V-A:

| Module | Function |
|---|---|
| `fp16_mul.v` / `fp16_add.v` | IEEE binary16 arithmetic primitives (RNE, DAZ) |
| `mul_array.v` | the PE's 16 multiply-accumulate lanes (one multiplier + one accumulate adder each) |
| `add_tree.v` | 16 -> 1 FP16 reduction tree, four adder levels in one cycle |
| `vec_acc_regs.v` | vector register and lane accumulator bank |
| `pe_mac16.v` | per-bank 16-lane FP16 MAC PE assembled from the three blocks above |
| `operand_fifo.v` | 4-entry x 256-bit operand FIFO (BG fan-in depth) |
| `arbiter_4to1.v` | bank-group 4:1 time-multiplexed bus + broadcast fan-out |
| `mode_ctrl.v` | connectivity mode register with FIFO-drain handshake |
| `pim_cmd_decoder.v` | all-bank command decoder (ACT_AB / RDMAC_AB / WR_AB / PRE_AB / MRW) with the row-open interlock |
| `pe_cluster.v` | bank-group top level: four PEs and FIFOs under the arbiter and the mode register |
| `sfu_pipeline.v` | per-channel programmable SFU (3-slot micro-programmed streaming pipeline with lane reduction) |

## Simulation

Self-checking testbenches (Icarus Verilog), one per module, covering FP16
arithmetic against exact reference values, FIFO ordering and back-pressure,
round-robin grant and broadcast fan-out, the MAC array and its reduction
tree, the command FSM's row-open interlock, the mode register's drain
handshake, the SFU micro-programs, and a bank-group cluster test that
streams one burst in each connectivity mode:

```bash
make          # run every testbench
make lint     # elaborate all modules
```

Every testbench drives its stimulus one settle delay after the sampling
edge, so the checks are independent of the simulator's event ordering.

## Synthesis

`syn/dc_compile.tcl` drives Synopsys Design Compiler at a 28 nm
standard-cell node with the 1 GHz constraint set in `syn/constraints.sdc`.
Point `LIB_PATH` and `TARGET_LIB` at your standard-cell kit and run:

```bash
export LIB_PATH=/path/to/28nm/db TARGET_LIB=sc28nm_tt_1p00v_25c.db
make syn      # reports/ and netlist/ for the PE, the bank-group blocks,
              # the command decoder, pe_cluster and the SFU
```

Reported areas are scaled to the 1z-nm DRAM process with the 10x
logic-DRAM process-efficiency factor (Sec. V-A). The area/power reports
feed the per-command MAC/SFU energies used by the simulators
(`configs/memory/*.yaml`, `pimcore/configs/*.yaml`), and
`reports/pe_area_hier.rpt` carries one `report_area -hierarchy` block per
component of `configs/area/near-bank-pe.yaml`, so Fig. 12 comes straight out
of a synthesis run:

```bash
python3 ../experiments/run_area.py --dc-report reports/pe_area_hier.rpt
```

`syn/report_area_pe.rpt` is a report of that shape for the six PE groups,
so the figure path can be exercised without a Design Compiler licence:

```bash
make syn-check   # parser checks + Fig. 12 from syn/report_area_pe.rpt
```

The SFU is synthesized at the channel streaming width (`LANES = 96`); the
`LANES = 16` default of `sfu_pipeline.v` is the characterization instance
the testbench drives.

Note: `fp16_exp2_pwl` in the SFU uses a piecewise-linear exp approximation
for characterization; the table-based production variant is a drop-in with
the same interface.
