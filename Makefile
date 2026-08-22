# ProteusSim build orchestration.
#
# The evaluation is staged: the C++ core has to be compiled, the replayable
# traces have to be generated from their generators, and each figure is
# produced by its own experiment script. Nothing is shipped pre-computed, so
# every number under results/ comes from a run on the reviewing machine.
#
#   make core       configure + build the C++ core (pimcore binaries)
#   make traces     regenerate the kernel and request traces
#   make rtl        run the Verilog testbenches (needs iverilog)
#   make syn        synthesize the near-bank logic (needs Design Compiler)
#   make test       C++, Python, command-level and Verilog test suites
#   make deps       fetch + build the external simulators (ext/; optional,
#                   needed only for the ramulator2/ONNXim cross-checks)
#   make clean      drop build artifacts, generated traces and results
#
# Once `make core` and `make traces` have succeeded, the per-figure scripts
# under experiments/ can be run individually; see README.md for the mapping
# from figure to script.

CMAKE   ?= cmake
JOBS    ?= 4
PYTHON  ?= python3
CORE     = pimcore/build/pimcore_sys
TRACES   = request_traces/pool256.txt request_traces/poisson40.txt \
           request_traces/azure30min.txt

.PHONY: core traces rtl syn test deps clean

core: $(CORE)

$(CORE):
	$(CMAKE) -S pimcore -B pimcore/build -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build pimcore/build -j$(JOBS)

traces: $(TRACES)

request_traces/pool256.txt:
	$(PYTHON) trace_gen/gen_requests.py --profile closed -n 256 -o $@

request_traces/poisson40.txt:
	$(PYTHON) trace_gen/gen_requests.py --profile poisson -n 512 --rate 40 \
	    --output-mean 256 -o $@

request_traces/azure30min.txt:
	$(PYTHON) trace_gen/gen_requests.py --profile azure --duration 1800 \
	    --rate 1.4 -n 0 --prompt-mean 2048 --output-mean 6144 -o $@

rtl:
	$(MAKE) -C rtl

syn:
	$(MAKE) -C rtl syn

test: core traces
	cd pimcore/build && ./pimcore_tests
	$(PYTHON) tests/test_sanity.py
	$(PYTHON) tests/test_dram_backend.py
	$(PYTHON) rtl/syn/test_dc_report.py
	@command -v iverilog >/dev/null && $(MAKE) -C rtl || \
	  echo "iverilog not found; skipping the Verilog testbenches"

deps:
	bash scripts/fetch_deps.sh

clean:
	rm -rf pimcore/build results $(TRACES)
	$(MAKE) -C rtl clean
