# Command-Level LPDDR5X-PIM Backend

`proteus_sim/dram/` models one PIM channel at DRAM-command granularity. It is
the substrate against which the sustained rates used by the operator-level
timing engine are validated (a two-layer methodology: an operator-level
performance model on top, a command-level memory model below). A standalone
C++17 implementation of the same backend, PimCore (`pimcore/`), provides an
independent cross-check and the multi-standard substrate studies.

## Organization

One x16 channel is one 32Gb die: four bank groups of four banks, 16 banks in
all, each with its own 16-lane FP16 PE and a 4-entry (128 B) operand FIFO.
Eight such channels form a package and eight packages a device, giving 64
channels, 256 bank groups, 1024 banks, 256 GB and 1024 PEs per device.

## The near-bank read path

The datapath that matters is *bank-local*. A PE is fed from its own bank's
I/O sense amplifiers (Fig. 8(c)); the read never traverses the shared
bank-group bus, the channel's global I/O, or any cross-BG crossbar. Its
cadence is therefore the bank's own column cycle, not the standard's tCCD_L,
which constrains the shared bus:

All banks execute concurrently through all-bank commands in either mode; what
changes is where a readout goes, and with it the bank column cadence.

| mode | bank column cadence | why |
|---|---|---|
| `direct` | `max(tCCD_L/2, mac_ns)` = 2 ns | DDR-pumped bank column cycle; every fetched element contributes to one vector |
| `broadcast` | `max(tCCD_L, fanout * mac_ns)` = 4 ns | each PE assembles one burst from *every* bank of its BG into the four-entry FIFO, whose depth matches the fan-in, and the 4:1 selector drains them one MAC issue at a time |

Direct mode reads `1024 * 32 B / 2 ns` = **16.38 TB/s** per device with one
MAC issue per burst. Broadcasting reuses every burst across the four PEs of a
bank group, so the compute per byte of DRAM access rises four-fold -- one MAC
per burst becomes four -- and the MAC arrays saturate at **32.77 TFLOPS** on
8.19 TB/s of DRAM access. Those are exactly the Table III peaks.

For a shared-operand GEMM of `n` vectors, direct mode re-streams the matrix
once per vector while broadcasting needs `ceil(n/fanout)` passes; the reuse
width is bounded by the fan-in, so beyond four vectors the matrix must be
streamed again and bank bandwidth and PE throughput become the joint
bottleneck.

Only broadcasting occupies the BG-local distribution bus, which is what
confines the added wiring to a bank group -- the locality the standard already
reflects in its tCCD_S / tCCD_L distinction.

## What else is modeled

* **Timing constraints**: tRCD/tRP/tRAS/tRC per bank, the per-mode column
  cadence above, tRRD_L/S and the rolling four-ACT tFAW window per die for
  *single-bank* commands, tREFI/tRFCab all-bank refresh (aligned across dies
  -- lockstep all-bank execution would otherwise stall once per die per
  tREFI), and CA-bus occupancy.
* **All-bank commands**: `ACT_AB` / `RDMAC_AB` / `WR_AB` / `PRE_AB` issue once
  on the CA bus and execute on every bank. Because the device provisions an
  all-bank activation as one operation, they are exempt from tRRD/tFAW --
  those bound the controller's ability to stagger *independent* activations,
  which all-bank execution does not do. The exemption and the all-bank
  tRCD/tRAS/tRP are explicit config fields (`allbank:`), so the assumption can
  be inspected and changed.
* **PE pipelines**: bursts are dropped into the operand FIFO and drained at
  MAC-issue rate; a full FIFO back-pressures the bank readout. `MODE` switches
  connectivity once the FIFOs drain (tens of ns, Sec. IV-C "Lightweight
  Reconfiguration").
* **Unified-memory co-execution**: a bank serves either its local PE or the
  channel's global I/O in a given column cycle. `attach_xpu_stream()` gives
  the host stream the slots the PIM datapath leaves free -- one per bank per
  cadence beyond the near-bank column cycle, bounded by the bank-group I/O
  gating and by the channel's DQ occupancy. Direct mode runs at the minimum
  column cycle and leaves none; broadcasting runs at twice that period and
  leaves more than the external interface can absorb. The analytical layer
  derives the same headroom in closed form (`memory.coexec_bw`), which is
  what makes xPU/PIM concurrency a consequence of the connectivity mode
  rather than a configured constant.
* **Energy**: DRAMPower-style per-command accounting (ACT/PRE pairs, array
  bursts, I/O+PHY when a burst leaves the die, MAC issues, mode switches,
  refresh).

## Trace generation

`trace_gen/` lowers the decode kernels onto the column/row-striped layouts of
Fig. 7: reuse-free GEMV, shared-operand skinny-GEMM (direct: one pass per
vector; broadcasting: `ceil(n/fanout)` passes), and GQA/MLA attention over the
resident KV slice with its in-place decode-step append. Traces are plain text
(`MODE m / ACT_AB r / RDMAC_AB r c / WR_AB r c / PRE_AB / BARRIER`) and replayable with
`python main.py --replay-trace <file>`.

## Cross-validation

`experiments/run_microbench.py` and `experiments/run_validation.py` compare
the closed-form rates of `proteus_sim/memory.py` against what this backend
actually executes:

| quantity | closed form | backend | deviation |
|---|---|---|---|
| LPDDR5X-PIM sustained, direct | 10.98 TB/s | 11.16 TB/s | 1.7% |
| LPDDR5X-PIM sustained, broadcast | 6.34 TB/s | 6.36 TB/s | 0.3% |
| GDDR6-AiM sustained, direct | 10.34 TB/s | 10.38 TB/s | 0.4% |

The closed form for the streaming efficiency is

```
stream / (tRCD + stream + max(0, tRTP - t_col) + tRP) * (1 - tRFCab/tREFI)
```

with `stream = row_bytes/burst * t_col`: under lockstep all-bank execution the
row activation is fully exposed once per row because the banks cannot hide
each other's tRCD/tRP, only the part of the read-to-precharge window that
outlasts a column cycle is exposed, and all-bank refresh removes tRFCab out of
every tREFI. Broadcasting amortizes the same activation over twice the
streaming window, which is why it sustains a *higher* fraction of a *lower*
peak.

`experiments/run_substrate.py` runs the same kernels on the HBM-PIM and
GDDR6-AiM timing tables, and `experiments/run_sys_crosscheck.py` checks the
Python and C++ implementations of the layer above cell by cell.
