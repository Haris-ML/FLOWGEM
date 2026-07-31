# read version from installed package
from importlib.metadata import version

__version__ = version("flowgem")

# public API
from flowgem.api import FlowGEM
from flowgem.core import sample_flowgem
from flowgem.optim import opt_wb
from flowgem.bandwidth import sigma_heuristic_pairs, evaluate_sigma

__all__ = [
    "FlowGEM",
    "sample_flowgem",
    "opt_wb",
    "sigma_heuristic_pairs",
    "evaluate_sigma",
]