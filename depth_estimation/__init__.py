"""
Monocular metric depth with sparse prior: Depth Anything + calibration (global, local, INR).

Public API is re-exported from subpackages for a stable import path.
"""

from depth_estimation.calibration import get_method, list_methods
from depth_estimation.evaluation import ExperimentConfig, ExperimentRunner, compute_metrics

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "__version__",
    "compute_metrics",
    "get_method",
    "list_methods",
]
