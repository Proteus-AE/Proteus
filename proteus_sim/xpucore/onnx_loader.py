"""Optional ONNX front-end for the xPU core engine.

When the `onnx` package is installed, `load_onnx_graph` lowers the MatMul /
Gemm nodes of an exported model graph into the same OpGraph IR the built-in
compiler produces, so real exported models can drive the tile scheduler.
Without the package (the default artifact environment is dependency-free),
the built-in lowering from the model YAML configs covers every evaluated
model; the two paths produce identical shapes for the transformer blocks.
"""
from ..compiler.graph import OpGraph, OpNode


def load_onnx_graph(path):
    """Lower the GEMM-class nodes of an ONNX graph into an OpGraph.

    Raises ImportError with a clear message when the optional `onnx`
    dependency is not installed.
    """
    try:
        import onnx  # noqa: F401  (optional dependency)
    except ImportError as e:
        raise ImportError(
            "the ONNX front-end needs the optional 'onnx' package "
            "(pip install onnx); the built-in YAML lowering covers all "
            "evaluated models without it") from e

    model = onnx.load(path)
    graph = model.graph

    # tensor-shape lookup from value_info / initializers
    shapes = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else "b"
                for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    for init in graph.initializer:
        shapes[init.name] = list(init.dims)

    g = OpGraph(model_name(graph))
    layer = 0
    for node in graph.node:
        if node.op_type not in ("MatMul", "Gemm"):
            continue
        a, b = node.input[0], node.input[1]
        sa = shapes.get(a, ["b", "?"])
        sb = shapes.get(b, ["?", "?"])
        g.add(OpNode(
            name=node.name or f"gemm{layer}",
            kind="weight_gemm",
            layer=layer,
            shape={"m": sa[0] if sa else "b",
                   "k": sb[0] if len(sb) > 0 else "?",
                   "n": sb[-1] if sb else "?"},
            inputs=list(node.input[:1])))
        layer += 1
    return g


def model_name(graph):
    return graph.name or "onnx-model"
