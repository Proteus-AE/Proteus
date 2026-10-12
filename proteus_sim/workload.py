"""Workload generation: lower a model configuration into one steady-state
decode iteration, described as the operator groups of a single transformer
block plus the aggregate traffic quantities consumed by the timing engines.

Conventions (FP16 end-to-end):
  * ``bytes``  : DRAM traffic of the operator's resident operand (weights or
                 KV) for one pass over the whole batch, counted before any
                 datapath-induced re-reads. Re-read multipliers (GEMV
                 decomposition, per-query-group KV re-fetch, broadcasting
                 passes) are applied by the *system* models, since they are
                 properties of the execution substrate, not of the workload.
  * ``flops``  : floating-point operations (2 per MAC).
  * ``intensity``: the arithmetic intensity the crossover model of Sec. IV-D
                 compares against theta. For shared-operand operators this is
                 Eq. (3), ``n d / (2 n + d)``; for attention it is the
                 query-group size g (GQA) or the latent-sharing head count
                 (MLA).
  * MoE routing: ``expected`` mode uses the uniform-routing expectation
                 E = N (1 - (1 - 1/N)^(b k)); ``sampled`` mode draws a
                 per-iteration multinomial routing histogram with a seeded
                 RNG, exactly as the runtime rebuilds it (Sec. IV-E).

All transformer blocks of a model are identically shaped, so the operator
groups are emitted once per model and scaled by the number of blocks a
pipeline stage owns. ``operators`` expands them per layer for reporting.
"""
import random
from dataclasses import dataclass, field

from .scheduler import shared_operand_ai_rect


@dataclass
class Operator:
    name: str
    kind: str            # 'weight_gemm' | 'attention' | 'elementwise'
    bytes: float         # resident-operand traffic of this group, one pass
    flops: float
    n_vectors: int       # concurrent input vectors sharing the resident operand
    d_model: int
    k_dim: int = 0       # shared-operand rows    (input features)
    n_out: int = 0       # shared-operand columns (output features)
    reuse: float = 1.0   # operand reuse available in the workload
    count: int = 1       # instances collapsed into this group (routed experts)
    tokens: float = 0.0  # routed tokens per instance (MoE)
    layer: int = 0

    @property
    def arithmetic_intensity(self):
        """Raw FLOP/byte of the resident operand stream."""
        return self.flops / max(self.bytes, 1.0)

    @property
    def is_expert(self):
        """True for a routed-expert skinny-GEMM (Sec. IV-E)."""
        return self.tokens > 0.0

    @property
    def intensity(self):
        """Intensity used by the analytical crossover estimate (Sec. IV-D)."""
        if self.kind == "attention":
            return float(self.reuse)
        if self.kind == "weight_gemm":
            return shared_operand_ai_rect(self.n_vectors,
                                          self.k_dim or self.d_model,
                                          self.n_out or self.d_model)
        return self.arithmetic_intensity


@dataclass
class Workload:
    """Aggregate quantities of one decode iteration (whole model, whole batch)."""
    model: dict
    batch: int
    ctx_avg: int             # steady-state context length (tokens)
    ctx_peak: int            # peak context for capacity accounting
    weight_bytes: float = 0  # streamed weight traffic (active experts only)
    weight_flops: float = 0
    kv_bytes: float = 0      # KV-cache traffic of attention
    attn_reuse: int = 1      # query-group size g (GQA) / head count (MLA)
    tokens_per_expert: float = 0
    active_experts: float = 0
    peak_mem: float = 0      # weights + peak KV footprint
    d_model: int = 0
    n_layers: int = 0
    block: list = field(default_factory=list)   # operator groups of one block
    routing_hist: list = field(default_factory=list)

    @property
    def attn_flops(self):
        # 2 FLOPs per FP16 element (2 B) per consuming query -> flops = bytes * g
        return self.kv_bytes * self.attn_reuse

    @property
    def operators(self):
        """Per-layer expansion of the block operator groups (reporting)."""
        out = []
        for layer in range(self.n_layers):
            for op in self.block:
                o = Operator(**{**op.__dict__})
                o.layer = layer
                out.append(o)
        return out

    def activation_bytes(self):
        """Bytes of one b x d_model FP16 activation tensor."""
        return self.batch * self.d_model * 2.0


def _expected_active_experts(n, k, b):
    return n * (1.0 - (1.0 - 1.0 / n) ** (b * k))


def _sampled_routing(n, k, b, rng):
    """Per-iteration top-k routing histogram (uniform gating)."""
    hist = [0] * n
    k = min(k, n)
    for _ in range(b):
        for e in rng.sample(range(n), k):
            hist[e] += 1
    return hist


def build_workload(model, batch, ctx_in, ctx_out, routing="expected", seed=7,
                   ctx_override=None, expert_centric=True):
    """Build one steady-state decode iteration.

    ``ctx_override``: if given, both the steady-state and peak context are set
    to this value (used by the context-length sweep, where the sweep point is
    the sustained context).
    ``expert_centric``: when False the MoE experts are lowered token-centrically
    (one reuse-free GEMV per routed token), the execution model +EC replaces.
    """
    if ctx_override is not None:
        ctx_avg = ctx_peak = int(ctx_override)
    else:
        ctx_avg = ctx_in + ctx_out // 2      # mid-generation steady state
        ctx_peak = ctx_in + ctx_out

    m = model
    g = m.get("attention_reuse")
    if g is None:
        g = max(1, m["n_heads"] // m["n_kv_heads"])

    w = Workload(model=m, batch=batch, ctx_avg=ctx_avg, ctx_peak=ctx_peak,
                 attn_reuse=g, d_model=m["d_model"], n_layers=m["n_layers"])
    w.kv_bytes = batch * ctx_avg * m["kv_bytes_per_token"]
    w.weight_flops = 2.0 * m["active_params"] * batch
    w.peak_mem = m["weight_bytes"] + batch * ctx_peak * m["kv_bytes_per_token"]

    moe = m["moe"]
    if moe["enabled"]:
        n, k = moe["n_experts"], moe["top_k"]
        if routing == "sampled":
            rng = random.Random(seed)
            hist = _sampled_routing(n, k, batch, rng)
            n_act = sum(1 for h in hist if h)
            w.routing_hist = hist
            w.active_experts = n_act
            w.weight_bytes = moe["dense_bytes"] + \
                n_act * moe["expert_bytes"] * m["n_layers"]
            w.tokens_per_expert = batch * k / max(n_act, 1)
        else:
            de = _expected_active_experts(n, k, batch)
            w.active_experts = de
            w.weight_bytes = moe["dense_bytes"] + de * moe["expert_bytes"] * m["n_layers"]
            w.tokens_per_expert = batch * k / de
    else:
        w.weight_bytes = m["weight_bytes"]
        w.tokens_per_expert = float(batch)

    _emit_block(w, expert_centric)
    return w


def block_matrices(m):
    """Per-layer weight matrices of one transformer block (Fig. 1).

    Returns ``(dense, expert)`` lists of ``(name, k, n_out)`` shapes derived
    from the model dimensions: the attention projections Q/K/V/O (K and V are
    narrowed by grouped-query attention) and either a dense gate/up/down FFN
    or the corresponding per-expert matrices of an MoE layer.
    """
    d = m["d_model"]
    dh = m["d_head"]
    q_out = m["n_heads"] * dh
    kv_out = m["n_kv_heads"] * dh
    ff = m["d_ffn"]
    dense = [("q_proj", d, q_out), ("k_proj", d, kv_out),
             ("v_proj", d, kv_out), ("o_proj", q_out, d)]
    expert = [("gate", d, ff), ("up", d, ff), ("down", ff, d)]
    if not m["moe"]["enabled"]:
        dense = dense + [(f"ffn_{n}", k, o) for n, k, o in expert]
        expert = []
    return dense, expert


def _params(shapes):
    return sum(k * o for _, k, o in shapes)


def _emit_block(w, expert_centric=True):
    """Operator groups of one transformer block.

    Every weight matrix of the block becomes its own operator group so that
    the crossover estimate is applied at the granularity the runtime actually
    schedules: individual projections for dense layers, and one skinny-GEMM
    per *routed expert* under expert-centric processing (Sec. IV-E). The
    traffic and FLOP sums of the emitted groups reproduce the model-level
    aggregates exactly -- the matrix shapes fix the relative sizes and a
    single normalization factor absorbs the parameters the block model does
    not name (embeddings, norms, router).
    """
    m = w.model
    L = m["n_layers"]
    d = m["d_model"]
    moe = m["moe"]

    # Attention over the resident KV cache (scores + AV), per layer.
    kv_l = w.kv_bytes / L
    w.block.append(Operator(
        name="attention", kind="attention", bytes=kv_l,
        flops=kv_l * w.attn_reuse, n_vectors=w.batch, d_model=d,
        k_dim=d, n_out=d, reuse=w.attn_reuse))

    dense_sh, expert_sh = block_matrices(m)
    wb_l = w.weight_bytes / L                 # streamed weight bytes per layer
    dense_p = _params(dense_sh)
    exp_p = _params(expert_sh)

    if moe["enabled"]:
        counts = _expert_token_counts(w)
        n_act = max(len(counts), 1)
        dense_bytes_l = moe["dense_bytes"] / L
        expert_bytes_l = max(wb_l - dense_bytes_l, 0.0)
        scale_d = dense_bytes_l / (2.0 * dense_p)
        scale_e = expert_bytes_l / (2.0 * exp_p * n_act) if exp_p else 0.0
    else:
        counts = []
        scale_d = wb_l / (2.0 * dense_p)
        scale_e = 0.0

    for name, k, o in dense_sh:
        b = 2.0 * k * o * scale_d
        w.block.append(Operator(
            name=name, kind="weight_gemm", bytes=b,
            flops=b * w.batch, n_vectors=w.batch, d_model=d,
            k_dim=k, n_out=o, reuse=w.batch))

    for i, tokens in enumerate(counts):
        n_vec = max(int(round(tokens)), 1) if expert_centric else 1
        for name, k, o in expert_sh:
            b = 2.0 * k * o * scale_e
            w.block.append(Operator(
                name=f"expert{i}_{name}", kind="weight_gemm", bytes=b,
                flops=b * max(tokens, 1.0), n_vectors=n_vec, d_model=d,
                k_dim=k, n_out=o,
                reuse=float(n_vec), tokens=float(tokens)))

    # Element-wise / nonlinear tail (layer norm, residual add, softmax,
    # activation): bandwidth-bound activation streams executed on the SFUs
    # when fused (Sec. IV-C "Generalized Operator Support").
    w.block.append(Operator(
        name="nonlinear", kind="elementwise",
        bytes=2.0 * w.batch * d * 2, flops=8.0 * w.batch * d,
        n_vectors=w.batch, d_model=d, k_dim=d, n_out=d))

    _rescale_block(w)


def _expert_token_counts(w):
    """Token count of every routed expert in this iteration (Sec. IV-E).

    Sampled routing uses the drawn histogram directly; the expectation mode
    spreads the batch's assignments evenly over the expected number of active
    experts, which is what the closed-form model assumes."""
    if w.routing_hist:
        return [h for h in w.routing_hist if h]
    n_act = max(int(round(w.active_experts)), 1)
    return [w.tokens_per_expert] * n_act


def _rescale_block(w):
    """Renormalize the emitted groups so that their per-layer traffic and
    FLOPs reproduce the model-level aggregates exactly."""
    L = w.n_layers
    tgt_b = w.weight_bytes / L
    tgt_f = w.weight_flops / L
    gemms = [o for o in w.block if o.kind == "weight_gemm"]
    if not gemms:
        return
    sb = sum(o.bytes for o in gemms)
    sf = sum(o.flops for o in gemms)
    kb = tgt_b / sb if sb else 1.0
    kf = tgt_f / sf if sf else 1.0
    for o in gemms:
        o.bytes *= kb
        o.flops *= kf
