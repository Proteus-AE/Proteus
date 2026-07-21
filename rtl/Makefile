# RTL simulation targets (Icarus Verilog).
#   make            run every testbench
#   make tb_fp16    run one testbench
#   make lint       elaborate all modules (syntax/elaboration check)
IVERILOG ?= iverilog
VVP      ?= vvp

SRCS = fp16_mul.v fp16_add.v operand_fifo.v pe_mac16.v bg_arbiter.v \
       pim_cmd_decoder.v pe_cluster.v \
       mode_ctrl.v sfu_pipeline.v
TBS  = tb_fp16 tb_operand_fifo tb_bg_arbiter tb_pe_mac16 \
       tb_pim_cmd_decoder tb_mode_ctrl tb_pe_cluster

all: $(TBS)

tb_%: tb/tb_%.v $(SRCS)
	$(IVERILOG) -g2005 -o build_$@.vvp $^
	$(VVP) build_$@.vvp
	@rm -f build_$@.vvp

lint: $(SRCS)
	$(IVERILOG) -g2005 -t null $(SRCS)
	@echo "lint: all modules elaborate"

clean:
	rm -f build_*.vvp

.PHONY: all lint clean $(TBS)
