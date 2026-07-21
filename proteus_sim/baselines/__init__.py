"""Baseline system models (Sec. V-A).

Every baseline consumes the same Workload description as Proteus and returns
the same Result structure. Mechanisms follow each system's published design;
efficiency parameters are calibrated against the published behavior of each
system in its home setting (see each module's header comment).
"""
from .gpu import GpuSystem
from .cxl_pnm import CxlPnmSystem
from .cent import CentSystem
from .neupims import NeuPimsSystem
from .papi import PapiSystem
from .pimphony import PimphonySystem

_KINDS = {
    "gpu": GpuSystem,
    "cxl_pnm": CxlPnmSystem,
    "cent": CentSystem,
    "neupims": NeuPimsSystem,
    "papi": PapiSystem,
    "pimphony": PimphonySystem,
}


def build(kind, cfg):
    return _KINDS[kind](cfg)
