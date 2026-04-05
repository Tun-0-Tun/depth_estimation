import numpy as np
from PIL import Image

from depth_estimation.models.da_inference import infer_depth, load_da_model


def fit_global_scale_shift(
    d_rel: np.ndarray,
    d_gt: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Fit global affine calibration D_met = s * d_rel + t to minimize L2 error to GT.

    Closed-form LS solution over all valid pixels:
      minimize || s * d_rel + t - d_gt ||_2^2
    """
    d_rel = d_rel.astype(np.float64)
    d_gt = d_gt.astype(np.float64)

    if valid_mask is None:
        valid_mask = np.isfinite(d_gt) & (d_gt > 0)
    else:
        valid_mask = valid_mask & np.isfinite(d_gt)

    if not np.any(valid_mask):
        raise ValueError("No valid pixels for global scale/shift fitting.")

    x = d_rel[valid_mask].reshape(-1)
    y = d_gt[valid_mask].reshape(-1)

    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = float((x_centered ** 2).sum())
    if denom <= 1e-8:
        # Degenerate case: relative depth is (almost) constant.
        s = 0.0
    else:
        s = float((x_centered * y_centered).sum() / denom)
    t = y_mean - s * x_mean
    return s, t


def apply_global_calibration(d_rel: np.ndarray, s: float, t: float) -> np.ndarray:
    """
    Apply global affine calibration to predicted relative depth.
    """
    return (s * d_rel + t).astype(np.float32)


def infer_and_calibrate_single(
    image: Image.Image,
    gt_depth: np.ndarray,
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Convenience baseline: run Depth Anything V2 on a single image and fit global metric scale/shift.

    Returns:
        d_rel     - raw Depth Anything prediction (relative)
        d_metric  - globally calibrated depth in GT units (meters for NYU)
        s, t      - fitted global affine parameters
    """
    processor, model, device, dtype = load_da_model(model_id=model_id)
    d_rel = infer_depth(processor, model, image, device, dtype)
    s, t = fit_global_scale_shift(d_rel, gt_depth)
    d_metric = apply_global_calibration(d_rel, s, t)
    return d_rel, d_metric, s, t

