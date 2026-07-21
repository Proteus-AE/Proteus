"""Model configuration -> per-layer operator graph.

One transformer block is lowered into its decode-phase operators with
symbolic shapes ("b" = runtime batch, "s" = context, "tpe" = tokens/expert).
Dense FFNs produce a single shared-operand GEMM chain; MoE layers produce a
router plus per-expert GEMM groups whose token counts are runtime values
(expert-centric scheduling regroups them each iteration, Sec. IV-E).
"""
from .graph import OpNode, OpGraph


def _attention_ops(g, m, layer):
    d, hd = m["d_model"], m["d_head"]
    kvh = m["n_kv_heads"]
    ln = g.add(OpNode(f"L{layer}.ln1", "elementwise", layer,
                      {"m": "b", "n": d}, inputs=[f"L{layer-1}.out"
                                                  if layer else "embed"]))
    qkv = g.add(OpNode(f"L{layer}.qkv_proj", "weight_gemm", layer,
                       {"m": "b", "k": d, "n": d + 2 * kvh * hd},
                       inputs=[ln.name]))
    scale = g.add(OpNode(f"L{layer}.scale", "elementwise", layer,
                         {"m": "b", "n": d}, inputs=[qkv.name]))
    score = g.add(OpNode(f"L{layer}.attn_score", "attention", layer,
                         {"m": "b", "k": hd, "n": "s"},
                         attrs={"operand": "K-cache", "reuse": "g"},
                         inputs=[scale.name]))
    smax = g.add(OpNode(f"L{layer}.softmax", "elementwise", layer,
                        {"m": "b", "n": "s"}, inputs=[score.name]))
    av = g.add(OpNode(f"L{layer}.attn_av", "attention", layer,
                      {"m": "b", "k": "s", "n": hd},
                      attrs={"operand": "V-cache", "reuse": "g"},
                      inputs=[smax.name]))
    out = g.add(OpNode(f"L{layer}.out_proj", "weight_gemm", layer,
                       {"m": "b", "k": d, "n": d}, inputs=[av.name]))
    res = g.add(OpNode(f"L{layer}.residual1", "elementwise", layer,
                       {"m": "b", "n": d}, inputs=[out.name]))
    return res


def _ffn_ops(g, m, layer, prev):
    d, dff = m["d_model"], m["d_ffn"]
    ln = g.add(OpNode(f"L{layer}.ln2", "elementwise", layer,
                      {"m": "b", "n": d}, inputs=[prev.name]))
    moe = m["moe"]
    if moe["enabled"]:
        router = g.add(OpNode(f"L{layer}.router", "weight_gemm", layer,
                              {"m": "b", "k": d, "n": moe["n_experts"]},
                              inputs=[ln.name]))
        gate = g.add(OpNode(f"L{layer}.gating", "elementwise", layer,
                            {"m": "b", "n": moe["n_experts"]},
                            inputs=[router.name]))
        ups = []
        for e in range(moe["n_experts"]):
            up = g.add(OpNode(f"L{layer}.e{e}.up", "weight_gemm", layer,
                              {"m": "tpe", "k": d, "n": dff},
                              attrs={"expert": e}, inputs=[gate.name]))
            act = g.add(OpNode(f"L{layer}.e{e}.act", "elementwise", layer,
                               {"m": "tpe", "n": dff}, inputs=[up.name]))
            dn = g.add(OpNode(f"L{layer}.e{e}.down", "weight_gemm", layer,
                              {"m": "tpe", "k": dff, "n": d},
                              attrs={"expert": e}, inputs=[act.name]))
            ups.append(dn)
        red = g.add(OpNode(f"L{layer}.moe_reduce", "reduce", layer,
                           {"m": "b", "n": d}, inputs=[u.name for u in ups]))
        last = red
    else:
        up = g.add(OpNode(f"L{layer}.ffn_up", "weight_gemm", layer,
                          {"m": "b", "k": d, "n": 2 * dff},
                          inputs=[ln.name]))
        act = g.add(OpNode(f"L{layer}.ffn_act", "elementwise", layer,
                           {"m": "b", "n": dff}, inputs=[up.name]))
        dn = g.add(OpNode(f"L{layer}.ffn_down", "weight_gemm", layer,
                          {"m": "b", "k": dff, "n": d}, inputs=[act.name]))
        last = dn
    res = g.add(OpNode(f"L{layer}.out", "elementwise", layer,
                       {"m": "b", "n": d}, inputs=[last.name, prev.name],
                       attrs={"op": "residual"}))
    return res


def lower_model(model, n_layers=None):
    """Lower `n_layers` transformer blocks (default: all) into an OpGraph."""
    g = OpGraph(model["name"])
    L = n_layers or model["n_layers"]
    for layer in range(L):
        att = _attention_ops(g, model, layer)
        _ffn_ops(g, model, layer, att)
    return g
