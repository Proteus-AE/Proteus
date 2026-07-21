"""Bridge to the PimCore C++ backend.

PimCore (pimcore/) is an independent C++ implementation of the command-level
near-bank-PIM backend, sharing the trace format and configuration style of
the Python backend. The bridge builds the binary on demand (CMake), invokes
it with a kernel or trace, and parses its JSON report -- used both by the
substrate-comparison experiments and by the cross-validation test that
checks the two implementations against each other.
"""
import json
import os
import shutil
import subprocess

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIMCORE_DIR = os.path.join(_ROOT, "pimcore")
BUILD_DIR = os.path.join(PIMCORE_DIR, "build")
BINARY = os.path.join(BUILD_DIR, "pimcore")


class PimCoreUnavailable(RuntimeError):
    pass


def available():
    return os.path.exists(BINARY) or shutil.which("cmake") is not None


def build(force=False):
    """Build the pimcore binary if it is missing (requires cmake + a C++17
    compiler); returns the binary path."""
    if os.path.exists(BINARY) and not force:
        return BINARY
    if shutil.which("cmake") is None:
        raise PimCoreUnavailable("cmake not found; install cmake/g++ or use "
                                 "the Python backend")
    os.makedirs(BUILD_DIR, exist_ok=True)
    subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                   cwd=BUILD_DIR, check=True, capture_output=True)
    subprocess.run(["make", "-j4"], cwd=BUILD_DIR, check=True,
                   capture_output=True)
    if not os.path.exists(BINARY):
        raise PimCoreUnavailable("pimcore build produced no binary")
    return BINARY


def config_path(name):
    return os.path.join(PIMCORE_DIR, "configs", f"{name}.yaml")


def run_kernel(config="lpddr5x-8533", kernel="gemv", rows=64, vectors=1,
               group=1, mode="direct", kv_append=False, host_gbps=None,
               external_energy=False, trace=None):
    """Run one PimCore simulation; returns the parsed JSON report."""
    binary = build()
    cmd = [binary, "--config", config_path(config), "--json"]
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
    build()
    test_bin = os.path.join(BUILD_DIR, "pimcore_tests")
    # run from the build directory so the relative configs path resolves
    out = subprocess.run([test_bin], check=True, capture_output=True,
                         text=True, cwd=BUILD_DIR)
    return out.stdout
