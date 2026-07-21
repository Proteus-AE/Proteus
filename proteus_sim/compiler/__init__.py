"""Compilation layer (Sec. IV-E "Software Stack").

Lowers a model configuration into a per-layer operator graph, applies the
structural operator-fusion pass (normalization+residual, softmax+scaling,
bias+activation chains fused into single SFU streaming pipelines), and
annotates every operator with its symbolic arithmetic-intensity expression,
parameterized by the runtime variables (batch, context, tokens/expert) that
the scheduler instantiates once per decode iteration (Sec. IV-D).
"""
from .graph import OpNode, OpGraph
from .lowering import lower_model
from .passes import fuse_elementwise, annotate_intensity, run_default_pipeline

__all__ = ["OpNode", "OpGraph", "lower_model", "fuse_elementwise",
           "annotate_intensity", "run_default_pipeline"]
