"""Metric depth calibration: global / local / INR and method registry."""

from depth_estimation.calibration.global_baseline import (
    apply_global_calibration,
    fit_global_scale_shift,
    infer_and_calibrate_single,
)
from depth_estimation.calibration.local_calibration import (
    apply_local_calibration,
    calibrate_local,
    compute_superpixels,
    fit_per_superpixel,
    smooth_fields,
    smooth_fields_bilateral,
)
from depth_estimation.calibration.methods import get_method, list_methods

__all__ = [
    "apply_global_calibration",
    "apply_local_calibration",
    "calibrate_local",
    "compute_superpixels",
    "fit_global_scale_shift",
    "fit_per_superpixel",
    "get_method",
    "infer_and_calibrate_single",
    "list_methods",
    "smooth_fields",
    "smooth_fields_bilateral",
]
