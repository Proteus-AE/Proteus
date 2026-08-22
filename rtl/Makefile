# RTL simulation and synthesis targets.
#   make            run every testbench (Icarus Verilog)
#   make tb_fp16    run one testbench
#   make lint       elaborate all modules (syntax/elaboration check)
#   make syn        Design Compiler run (needs dc_shell + a 28 nm kit)
#   make syn-check  rebuild the Fig. 12 area table from the checked-in
#                   report_area fixture (no dc_shell needed)
IVERILOG ?= iverilog
VVP      ?= vvp
DC_SHELL ?= dc_shell
PYTHON   ?= python3

SRCS = fp16_mul.v fp16_add.v mul_array.v add_tree.v vec_acc_regs.v \
       pe_mac16.v operand_fifo.v arbiter_4to1.v mode_ctrl.v \
       pim_cmd_decoder.v pe_cluster.v sfu_pipeline.v
TBS  = tb_fp16 tb_operand_fifo tb_arbiter_4to1 tb_pe_mac16 tb_sfu_pipeline \
       tb_pim_cmd_decoder tb_mode_ctrl tb_pe_cluster

SYN_TCL     = syn/dc_compile.tcl
SYN_SDC     = syn/constraints.sdc
SYN_FIXTURE = syn/report_area_pe.rpt
RUN_AREA    = ../experiments/run_area.py

all: $(TBS)

tb_%: tb/tb_%.v $(SRCS)
	$(IVERILOG) -g2005 -o build_$@.vvp $^
	$(VVP) build_$@.vvp
	@rm -f build_$@.vvp

lint: $(SRCS)
	$(IVERILOG) -g2005 -t null $(SRCS)
	@echo "lint: all modules elaborate"

# Synthesis run: writes reports/ and netlist/, including the hierarchical
# reports/pe_area_hier.rpt that experiments/run_area.py --dc-report reads.
# LIB_PATH and TARGET_LIB select the 28 nm standard-cell kit; dc_compile.tcl
# picks them up from the environment.
syn: $(SRCS) $(SYN_TCL) $(SYN_SDC)
	@command -v $(DC_SHELL) >/dev/null 2>&1 || { \
	  echo "make syn: '$(DC_SHELL)' is not on PATH."; \
	  echo "  Synthesis needs Synopsys Design Compiler and a 28 nm"; \
	  echo "  standard-cell kit. Without them, 'make syn-check' replays"; \
	  echo "  the checked-in $(SYN_FIXTURE) instead."; \
	  exit 1; }
	@test -n "$$LIB_PATH" || { \
	  echo "make syn: set LIB_PATH to your 28 nm library directory."; \
	  exit 1; }
	@test -n "$$TARGET_LIB" || { \
	  echo "make syn: set TARGET_LIB to the target .db of that library."; \
	  exit 1; }
	mkdir -p reports netlist
	$(DC_SHELL) -f $(SYN_TCL)
	@echo "syn: reports/ and netlist/ written"

# Offline check of the synthesis-to-figure path: the fixture holds the
# report_area -hierarchy block the flow above emits, so the Fig. 12 table
# must come out of it unchanged.
syn-check: $(SYN_FIXTURE)
	$(PYTHON) syn/test_dc_report.py
	$(PYTHON) $(RUN_AREA) --dc-report $(SYN_FIXTURE)

clean:
	rm -f build_*.vvp
	rm -rf reports netlist

# The tb_% targets produce no file of their own, so they always re-run.
.PHONY: all lint syn syn-check clean
