# ProteusSim build orchestration.
#
#   make core       build the C++ core (pimcore binaries; required)
#   make deps       fetch + build the external simulators (ext/; optional,
#                   needed only for the ramulator2/ONNXim cross-checks)
#   make rtl        run the Verilog testbenches (needs iverilog)
#   make test       C++ + Python test suites
#   make all        core + full evaluation (scripts/run_all.sh)
#   make clean      drop build artifacts and generated results

CMAKE ?= cmake
JOBS  ?= 4

.PHONY: all core deps rtl test run clean

all: core run

core: pimcore/build/pimcore_sys

pimcore/build/pimcore_sys:
	$(CMAKE) -S pimcore -B pimcore/build -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build pimcore/build -j$(JOBS)

deps:
	bash scripts/fetch_deps.sh

rtl:
	$(MAKE) -C rtl

test: core
	cd pimcore/build && ./pimcore_tests
	python3 tests/test_sanity.py
	python3 tests/test_dram_backend.py

run: core
	bash scripts/run_all.sh

clean:
	rm -rf pimcore/build results
