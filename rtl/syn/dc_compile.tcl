# Synopsys Design Compiler synthesis script for the Proteus PIM logic
# (Sec. V-A "Area and Energy Modeling"): the per-bank PE (MAC lanes,
# reduction tree, register bank), the bank-group datapath around it
# (operand FIFO, 4:1 broadcast arbiter, mode register, command decoder),
# the pe_cluster bank-group top level, and the per-channel SFU, at a 28 nm
# standard-cell node. Reported cell area is scaled to the 1z-nm DRAM
# process with the 10x logic-DRAM process-efficiency factor in the paper's
# methodology.
#
# Usage: make -C rtl syn
#   (equivalently `dc_shell -f syn/dc_compile.tcl` from rtl/; set
#    LIB_PATH/TARGET_LIB for your 28 nm standard-cell kit and create the
#    reports/ and netlist/ output directories first -- the make target does
#    both)

set LIB_PATH   $env(LIB_PATH)
set TARGET_LIB $env(TARGET_LIB)

set search_path    [list . $LIB_PATH]
set target_library [list $TARGET_LIB]
set link_library   [concat "*" $target_library]

# Same source list as rtl/Makefile's SRCS, leaves first.
set RTL_FILES {
  fp16_mul.v
  fp16_add.v
  mul_array.v
  add_tree.v
  vec_acc_regs.v
  pe_mac16.v
  operand_fifo.v
  arbiter_4to1.v
  mode_ctrl.v
  pim_cmd_decoder.v
  pe_cluster.v
  sfu_pipeline.v
}

# Fig. 12 is fed by one hierarchical area report holding the six PE groups
# plus the SFU. Their design names are exactly the keys of
# configs/area/near-bank-pe.yaml, so
#   python3 experiments/run_area.py --dc-report rtl/reports/pe_area_hier.rpt
# rebuilds the figure straight from a synthesis run.
set AREA_RPT reports/pe_area_hier.rpt
file delete -force $AREA_RPT

analyze -format verilog $RTL_FILES

# ---- per-bank PE ---------------------------------------------------------
# Compiled with the block boundaries preserved, so that mul_array, add_tree
# and vec_acc_regs remain separately reportable.
elaborate pe_mac16
current_design pe_mac16
link
source syn/constraints.sdc
compile_ultra -no_autoungroup
report_area -hierarchy > reports/pe_mac16_area.rpt
report_power           > reports/pe_mac16_power.rpt
report_timing          > reports/pe_mac16_timing.rpt
write -format verilog -hierarchy -output netlist/pe_mac16_syn.v

foreach group {mul_array add_tree vec_acc_regs} {
  current_design $group
  redirect -append $AREA_RPT { report_area -hierarchy }
}

# ---- bank-group datapath (PE extensions) ---------------------------------
elaborate operand_fifo
current_design operand_fifo
link
source syn/constraints.sdc
compile_ultra
report_area -hierarchy > reports/operand_fifo_area.rpt
report_power           > reports/operand_fifo_power.rpt
redirect -append $AREA_RPT { report_area -hierarchy }

elaborate arbiter_4to1
current_design arbiter_4to1
link
source syn/constraints.sdc
compile_ultra
report_area -hierarchy > reports/arbiter_4to1_area.rpt
report_power           > reports/arbiter_4to1_power.rpt
redirect -append $AREA_RPT { report_area -hierarchy }

elaborate mode_ctrl
current_design mode_ctrl
link
source syn/constraints.sdc
compile_ultra
report_area -hierarchy > reports/mode_ctrl_area.rpt
redirect -append $AREA_RPT { report_area -hierarchy }

elaborate pim_cmd_decoder
current_design pim_cmd_decoder
link
source syn/constraints.sdc
compile_ultra
report_area -hierarchy > reports/pim_cmd_decoder_area.rpt
report_timing          > reports/pim_cmd_decoder_timing.rpt

# ---- per-channel SFU -----------------------------------------------------
# The SFU is instantiated at the channel streaming width; the LANES = 16
# default of sfu_pipeline.v is the characterization instance the testbench
# drives. Keep the elaborated design name unparameterized so its group in
# $AREA_RPT matches the `sfu_um2_28nm` entry of the area config.
set template_naming_style "%s"
elaborate sfu_pipeline -parameters "LANES = 96"
current_design sfu_pipeline
link
source syn/constraints.sdc
compile_ultra -no_autoungroup
report_area -hierarchy > reports/sfu_area.rpt
report_power           > reports/sfu_power.rpt
report_timing          > reports/sfu_timing.rpt
write -format verilog -hierarchy -output netlist/sfu_syn.v
redirect -append $AREA_RPT { report_area -hierarchy }

# ---- bank-group top level ------------------------------------------------
# pe_cluster carries the overhead Sec. V-B reports: four PEs, four operand
# FIFOs, the 4:1 broadcast arbiter and the mode register in one block.
elaborate pe_cluster
current_design pe_cluster
link
source syn/constraints.sdc
compile_ultra -no_autoungroup
report_area -hierarchy > reports/pe_cluster_area.rpt
report_power           > reports/pe_cluster_power.rpt
report_timing          > reports/pe_cluster_timing.rpt
write -format verilog -hierarchy -output netlist/pe_cluster_syn.v

quit
