"""Bridge to the PimCore C++ backend.

PimCore (pimcore/) is an independent C++ implementation of the command-level
near-bank-PIM backend and of the system timing layer, sharing the trace
format and configuration style of the Python backend. The bridge locates the
binaries produced by the project's build (``make core``), invokes them with a
kernel or trace, and parses the JSON report -- used both by the
substrate-comparison experiments and by the cross-validation that checks the
two implementations against each other.

The bridge deliberately does *not* build anything itself. The C++ core is a
separate compilation unit of the artifact with its own toolchain
requirements, and silently rebuilding it from inside an experiment would hide
which binary a result actually came from.
"""
import json
import os
import subprocess

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIMCORE_DIR = os.path.join(_ROOT, "pimcore")
BUILD_DIR = os.path.join(PIMCORE_DIR, "build")
BINARY = os.path.join(BUILD_DIR, "pimcore")


BUILD_HINT = ("build the C++ core first:\n"
              "    cmake -S pimcore -B pimcore/build "
              "-DCMAKE_BUILD_TYPE=Release\n"
              "    cmake --build pimcore/build -j4\n"
              "  (or simply: make core)")


class PimCoreUnavailable(RuntimeError):
    pass


def available(name="pimcore"):
    """True if the named C++ binary has been built."""
    return os.path.exists(os.path.join(BUILD_DIR, name))


def binary(name="pimcore"):
    """Path of a built pimcore binary, or an error naming the build step."""
    path = os.path.join(BUILD_DIR, name)
    if not os.path.exists(path):
        raise PimCoreUnavailable(f"{os.path.relpath(path, _ROOT)} not found; "
                                 + BUILD_HINT)
    return path


def config_path(name):
    """Path of a memory-substrate configuration, shared with the Python
    backend so that both read the same JEDEC timing tables."""
    return os.path.join(_ROOT, "configs", "memory", f"{name}.yaml")


def run_kernel(config="lpddr5x-8533", kernel="gemv", rows=64, vectors=1,
               group=1, mode="direct", kv_append=False, host_gbps=None,
               external_energy=False, trace=None):
    """Run one PimCore simulation; returns the parsed JSON report."""
    cmd = [binary(), "--config", config_path(config), "--json"]
    if trace:
        cmd += ["--trace", trace]
    else:
        cmd += ["--kernel", kernel, "--rows", str(rows),
                "--vectors", str(vectors), "--group", str(group),
                "--mode", mode]
        if kv_append:
            cmd.append("--kv-append")
    if host_gbps is not None:
        cmd += ["--host-gbps", str(host_gbps)]
    if external_energy:
        cmd.append("--external-energy")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def run_tests():
    """Execute the PimCore C++ test suite; returns its stdout."""
    # run from the build directory so the relative configs path resolves
    out = subprocess.run([binary("pimcore_tests")], check=True,
                         capture_output=True, text=True, cwd=BUILD_DIR)
    return out.stdout
