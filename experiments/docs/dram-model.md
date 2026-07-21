# Command-Level LPDDR5X-PIM Backend

`proteus_sim/dram/` models one PIM channel at DRAM-command granularity and is
the calibration/validation substrate for the sustained rates used by the
system-level timing engine (two-layer methodology: an operator-level
performance model on top, a command-level memory model below). A standalone
C++ implementation of the same backend, PimCore (`pimcore/`), provides an
independent cross-check and the multi-standard substrate studies.

## What is modeled

* **Organization** (Fig. microarch): 4 dies x 4 bank groups x 4 banks per
  channel, one 16-lane FP16 PE per bank with a 4-entry operand FIFO, a
  time-multiplexed 32 B data bus per bank group, one command/address bus per
  channel.
* **Timing constraints**: tRCD/tRP/tRAS/tRC per bank, tCCD_L on the BG bus
  (DDR cadence: one 32 B burst per tCCD_L/2), tRRD_L/S and the rolling
  four-ACT tFAW window per die, tREFI/tRFCab all-bank refresh (aligned across
  dies -- lockstep all-bank execution would otherwise stall once per die per
  tREFI), and CA-bus occupancy (all-bank commands issue once, Sec. IV-C).
* **PIM execution**: `ACT_AB`/`RDMAC_AB`/`PRE_AB` all-bank commands executed
  by every bank under local constraints; per-bank PE pipelines consume bursts
  at MAC-issue rate with FIFO back-pressure; `MODE` switches connectivity
  after the FIFOs drain (tens of ns, Sec. IV-C "Lightweight Reconfiguration").
* **Connectivity modes**: direct (one-to-one bank-to-PE) vs. broadcasting
  (each burst feeds all four PEs of the BG through the time-multiplexed bus
  and 4:1 arbiter).
* **Unified-memory co-execution**: an opportunistic host read stream claims
  BG-bus slots the PIM schedule leaves free (Sec. IV-B).
* **Energy**: DRAMPower-style per-command accounting (ACT/PRE pairs, array
  bursts, I/O+PHY when leaving the die, MAC issues, mode switches, refresh).

## Trace generation

`trace_gen/` lowers the decode kernels onto the column/row-striped layouts of
Fig. "Data Placement": reuse-free GEMV, shared-operand skinny-GEMM
(direct: one pass per vector; broadcasting: ceil(n/4) passes), and GQA/MLA
attention over the resident KV slice. Traces are plain text
(`ACT_AB r / RDMAC_AB r c / PRE_AB / MODE m`) and replayable with
`python main.py --replay-trace <file>`.

## Cross-validation with the analytical layer

`python experiments/run_microbench.py` produces (results/microbench_*.csv):

1. **Sustained streaming**: ~199 GB/s per channel out of a 256 GB/s peak
   (efficiency ~0.77, all row hits), against the 0.80 constant used by the
   analytical layer -- agreement within 4%. The gap decomposes into the
   per-row ACT ramp under tFAW (~15%) and aligned all-bank refresh (~7%).
2. **Operand reuse**: broadcasting achieves min(n,4)-fold speedup for
   skinny-GEMMs of n concurrent vectors (4.0x at n >= 4), directly validating
   the ceil(n/4)-pass model of the timing engine.
3. **Co-execution headroom**: for iso-work kernels, broadcasting occupies
   ~25% of the channel time of direct mode, freeing ~75% of memory-service
   slots for concurrent xPU access -- the command-level footing of the
   co-execution split x* in the system model.
4. **Energy split**: near-bank termination measures ~1.9-2.2 pJ/bit and
   external ~4.6 pJ/bit, matching the 2.2/4.5 pJ/bit constants of the
   system-level energy model.
5. **Attention**: per-mode KV re-streaming reproduces the g-fold direct-mode
   penalty and its ceil(g/4) broadcasting recovery.

`tests/test_dram_backend.py` asserts all five relationships automatically.

## xPU-side integration

The xPU core engine (`proteus_sim/xpucore/`) is the tile-level model of the
matrix engine; its DRAM traffic lowers to host-request streams that the
backend schedules against the PIM stream through the host controller
(PimCore) or the opportunistic slot mechanism (Python backend) -- the
shared-memory-interface integration described in Sec. V-A.
`experiments/run_integrated.py` closes one decode layer across the three
layers of the methodology.

## Why an in-package backend instead of a Ramulator2 patch set

The command-level backend implements exactly the timing state a DRAM
simulator would exercise for these all-bank streaming kernels (bank/BG/die
constraints, refresh, buses), but remains pip-free, runs in seconds, and is
cross-checked against the analytical layer in CI. The trace format is
deliberately simulator-agnostic so the same traces can be replayed on an
external DRAM simulator.
