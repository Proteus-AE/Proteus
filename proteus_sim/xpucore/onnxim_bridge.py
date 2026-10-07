"""Bridge to ONNXim (ext/onnxim).

Runs the decode-layer GEMMs through ONNXim's cycle-level NPU model and
reports per-model cycles, so the tile-level xpucore engine can be checked
against an independent accelerator simulator. Requires the `onnx` Python
package (to export the operator graph as a model file ONNXim accepts) and
an ONNXim build (`ext/onnxim/build` or $ONNXIM_HOME).
"""
import json
import os
import re
import subprocess
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class OnnximUnavailable(RuntimeError):
    pass


def find_binary():
    home = os.environ.get("ONNXIM_HOME",
                          os.path.join(_ROOT, "ext", "onnxim"))
    for rel in ("build/bin/Simulator", "build/Simulator", "bin/Simulator"):
        p = os.path.join(home, rel)
        if os.path.exists(p):
            return p
    raise OnnximUnavailable(
        "ONNXim binary not found -- build ext/onnxim (see its README) "
        "or set ONNXIM_HOME")


def export_gemm_onnx(path, m, k, n):
    """Write a single-GEMM ONNX model (requires the `onnx` package)."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        raise OnnximUnavailable(
            "the `onnx` package is required: pip install .[integration]")
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [m, k])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT16, [k, n])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [m, n])
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "gemm", [a, b], [y])
    model = helper.make_model(graph, producer_name="proteus-sim")
    onnx.save(model, path)
    return path


def _npu_config(freq_mhz=1410, n_cores=1, sram_kb=24 * 1024):
    """A100-class single-core configuration in ONNXim's config format."""
    return {
        "num_cores": n_cores,
        "core_type": "systolic_ws",
        "core_freq": freq_mhz,
        "core_width": 32,
        "core_height": 32,
        "spad_size": sram_kb,
        "sram_width": 32,
        "vector_process_bit": 65536,
        "add_latency": 1, "mul_latency": 1, "exp_latency": 1,
        "gelu_latency": 1, "add_tree_latency": 1, "scalar_sqrt_latency": 1,
        "scalar_add_latency": 1, "scalar_mul_latency": 1,
        "dram_type": "simple", "dram_freq": 2133, "dram_channels": 8,
        "dram_req_size": 32, "dram_latency": 10, "dram_nbl": 4,
        "icnt_type": "simple", "icnt_latency": 1, "icnt_freq": 8000,
        "precision": 2, "layout": "NHWC", "scheduler": "simple",
    }


def run_gemm(m, k, n, timeout=1200):
    """Run one (m x k) . (k x n) GEMM through ONNXim; returns cycles."""
    binary = find_binary()
    with tempfile.TemporaryDirectory() as td:
        onnx_path = os.path.join(td, "gemm.onnx")
        export_gemm_onnx(onnx_path, m, k, n)
        cfg = _npu_config()
        cfg_path = os.path.join(td, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        models = {"models": [{"name": "gemm", "path": onnx_path,
                              "batch_size": 1}]}
        models_path = os.path.join(td, "models.json")
        with open(models_path, "w") as f:
            json.dump(models, f, indent=2)
        proc = subprocess.run(
            [binary, "--config", cfg_path, "--models_list", models_path],
            capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            tail = out.strip().splitlines()[-1] if out.strip() else ""
            raise OnnximUnavailable(f"ONNXim run failed: {tail}")
        m_cyc = re.search(r"[Tt]otal\s+cycles?\s*[:=]\s*(\d+)", out)
        if not m_cyc:
            m_cyc = re.search(r"(\d+)\s+cycles", out)
        if not m_cyc:
            raise OnnximUnavailable("no cycle count in ONNXim output")
        return {"cycles": int(m_cyc.group(1)), "raw": out[-2000:]}
