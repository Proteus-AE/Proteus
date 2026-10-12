"""ProteusSim: an operator-level performance and energy simulator for
heterogeneous xPU + near-bank-PIM LLM inference systems.

The simulator models one steady-state decode iteration at operator
granularity: the workload generator lowers a model configuration into a
per-layer operator graph, the scheduler maps every operator onto an
execution substrate (xPU, PIM-direct, PIM-broadcast, SFU) using the
analytical crossover model of Sec. IV-D, and the timing engine resolves
per-device xPU/PIM co-execution, pipeline parallelism across devices, and
capacity-bounded in-flight batching. Energy is accounted from the byte and
FLOP counters produced by the same simulation.
"""

__version__ = "1.1.0"

from .config import load_model, load_system, load_memory
from .workload import Workload
from .system import build_system

__all__ = ["load_model", "load_system", "load_memory", "Workload", "build_system"]
