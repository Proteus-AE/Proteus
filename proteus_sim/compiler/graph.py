"""Operator-graph IR for the compilation layer."""
import json
from dataclasses import dataclass, field


@dataclass
class OpNode:
    name: str
    kind: str                  # weight_gemm | attention | elementwise | reduce
    layer: int
    shape: dict                # symbolic dims, e.g. {"m": "b", "k": 4096, ...}
    attrs: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)    # node names
    fused: list = field(default_factory=list)     # names folded into this node
    ai_expr: str = ""          # symbolic arithmetic-intensity expression

    def is_elementwise(self):
        return self.kind == "elementwise"


class OpGraph:
    def __init__(self, model_name):
        self.model = model_name
        self.nodes = {}
        self.order = []

    def add(self, node: OpNode):
        self.nodes[node.name] = node
        self.order.append(node.name)
        return node

    def remove(self, name):
        self.nodes.pop(name)
        self.order.remove(name)

    def iter_nodes(self):
        return [self.nodes[n] for n in self.order]

    def rewire(self, old, new):
        for n in self.iter_nodes():
            n.inputs = [new if i == old else i for i in n.inputs]

    # ------------------------------------------------------------------ #
    def to_json(self, path=None):
        doc = {"model": self.model, "nodes": [{
            "name": n.name, "kind": n.kind, "layer": n.layer,
            "shape": n.shape, "inputs": n.inputs, "fused": n.fused,
            "ai": n.ai_expr, **({"attrs": n.attrs} if n.attrs else {}),
        } for n in self.iter_nodes()]}
        s = json.dumps(doc, indent=1)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    def stats(self):
        kinds = {}
        fused = 0
        for n in self.iter_nodes():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
            fused += len(n.fused)
        return {"nodes": len(self.order), "by_kind": kinds,
                "fused_away": fused}
