"""Graph-level passes (Sec. IV-E).

``fuse_elementwise``   Structural operator fusion: element-wise / nonlinear
                       chains (norm+residual, softmax+scaling,
                       bias+activation) are folded into their producer and
                       executed as one SFU streaming pipeline without
                       intermediate DRAM round-trips.
``annotate_intensity`` Generates the symbolic arithmetic-intensity
                       expressions the runtime instantiates once per decode
                       iteration (crossover scheduling, Sec. IV-D).
"""


FUSABLE_TAILS = {"elementwise", "reduce"}


def fuse_elementwise(g):
    """Fold every element-wise consumer chain into its producing operator."""
    changed = True
    while changed:
        changed = False
        for n in list(g.iter_nodes()):
            if not n.is_elementwise() or n.name not in g.nodes:
                continue
            if not n.inputs:
                continue
            prod_name = n.inputs[0]
            prod = g.nodes.get(prod_name)
            if prod is None or prod.is_elementwise():
                continue
            # fold n into prod: prod streams its output through the SFU chain
            prod.fused.append(n.name)
            g.rewire(n.name, prod.name)
            g.remove(n.name)
            changed = True
    return g


def annotate_intensity(g):
    """Attach the symbolic AI expression of each remaining operator."""
    for n in g.iter_nodes():
        s = n.shape
        if n.kind == "weight_gemm":
            m, k, nn = s.get("m"), s.get("k"), s.get("n")
            # AI_w = m*k*n / (m*k + k*n + m*n)  ->  ~ m*d/(2m+d) for square
            n.ai_expr = f"({m}*{k}*{nn})/({m}*{k}+{k}*{nn}+{m}*{nn})"
        elif n.kind == "attention":
            n.ai_expr = "g" if n.attrs.get("reuse") == "g" else "1"
        else:
            n.ai_expr = "O(1)"
    return g


def run_default_pipeline(g):
    fuse_elementwise(g)
    annotate_intensity(g)
    return g
