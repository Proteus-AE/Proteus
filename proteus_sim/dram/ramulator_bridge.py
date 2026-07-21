"""Bridge to Ramulator 2.0 (ext/ramulator2).

Replays a host-path address stream (pimcore_tracegen --host-stream) through
Ramulator's cycle-accurate LPDDR5X model and reports the served bandwidth,
so the host path of the built-in backends can be checked against an
independent DRAM simulator.

Ramulator's implementation registry names (device presets, trace frontends)
have shifted across revisions; the bridge therefore tries a small candidate
list and uses the first configuration the installed revision accepts.
"""
import os
import re
import subprocess
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# (dram impl, org preset, timing preset, data rate MT/s)
_DRAM_CANDIDATES = [
    ("LPDDR5X", "LPDDR5X_8Gb_x16", "LPDDR5X_8533", 8533),
    ("LPDDR5X", "LPDDR5X_8Gb_x16", "LPDDR5X_6400", 6400),
    ("LPDDR5", "LPDDR5_8Gb_x16", "LPDDR5_6400", 6400),
    ("DDR5", "DDR5_8Gb_x8", "DDR5_4800AN", 4800),
]
# (frontend impl, tracegen --style)
_FRONTEND_CANDIDATES = [
    ("LoadStoreTrace", "ldst"),
    ("ReadWriteTrace", "rw"),
]

_CONFIG_TEMPLATE = """\
Frontend:
  impl: {frontend}
  path: {trace}
  clock_ratio: 1
MemorySystem:
  impl: GenericDRAM
  clock_ratio: 1
  DRAM:
    impl: {impl}
    org:
      preset: {org}
      channel: 1
      rank: 1
    timing:
      preset: {timing}
  Controller:
    impl: Generic
    Scheduler:
      impl: FRFCFS
    RefreshManager:
      impl: AllBank
    RowPolicy:
      impl: OpenRowPolicy
  AddrMapper:
    impl: RoBaRaCoCh
"""


class RamulatorUnavailable(RuntimeError):
    pass


def find_binary():
    """Locate the stock ramulator2 binary (RAMULATOR2_BIN overrides)."""
    env = os.environ.get("RAMULATOR2_BIN")
    if env and os.path.exists(env):
        return env
    for rel in ("ext/ramulator2/build/ramulator2",
                "ext/ramulator2/ramulator2"):
        p = os.path.join(_ROOT, rel)
        if os.path.exists(p):
            return p
    raise RamulatorUnavailable(
        "ramulator2 binary not found -- run scripts/fetch_deps.sh "
        "(or set RAMULATOR2_BIN)")


def _parse_stats(text):
    stats = {}
    for m in re.finditer(r"^\s*(\w+):\s*(-?[\d.eE+]+)\s*$", text, re.M):
        try:
            stats[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return stats


def _pick(stats, *needles):
    for key, val in stats.items():
        k = key.lower()
        if all(n in k for n in needles):
            return val
    return None


def run_stream(traces, timeout=600):
    """Run a host address stream; returns a report dict.

    `traces` maps tracegen --style ("rw" / "ldst") to a trace path; the
    style matching whichever frontend the installed revision accepts is
    used.
    """
    binary = find_binary()
    last_err = "no candidate configuration accepted"
    for frontend, style in _FRONTEND_CANDIDATES:
        if style not in traces:
            continue
        trace_path = traces[style]
        for impl, org, timing, rate in _DRAM_CANDIDATES:
            cfg = _CONFIG_TEMPLATE.format(frontend=frontend,
                                          trace=os.path.abspath(trace_path),
                                          impl=impl, org=org, timing=timing)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                             delete=False) as f:
                f.write(cfg)
                cfg_path = f.name
            try:
                proc = subprocess.run([binary, "-f", cfg_path],
                                      capture_output=True, text=True,
                                      timeout=timeout)
                out = proc.stdout + proc.stderr
                if proc.returncode != 0:
                    last_err = out.strip().splitlines()[-1] if out.strip() \
                        else f"exit {proc.returncode}"
                    continue
                stats = _parse_stats(out)
                cycles = _pick(stats, "memory_system", "cycle") or \
                    _pick(stats, "cycle")
                reads = _pick(stats, "read", "request") or \
                    _pick(stats, "num", "read")
                if not cycles:
                    last_err = "no cycle counter in output"
                    continue
                tck_ns = 2.0 / (rate * 1e6) * 1e9   # DDR: clock = rate/2
                time_ns = cycles * tck_ns
                reqs = reads if reads else 0.0
                return {
                    "impl": impl, "timing": timing, "frontend": frontend,
                    "cycles": cycles, "tck_ns": tck_ns,
                    "time_ns": time_ns, "read_reqs": reqs,
                    "bw_gbps": (reqs * 64.0) / time_ns if time_ns else 0.0,
                    "stats": stats,
                }
            finally:
                os.unlink(cfg_path)
    raise RamulatorUnavailable(f"ramulator2 run failed: {last_err}")
