"""
Standard depth estimation metrics.

Implements the metrics commonly used in monocular depth estimation benchmarks
(Eigen et al. 2014, Fu et al. 2018, Ranftl et al. 2020):
  - AbsRel, SqRel  (error metrics, lower is better)
  - RMSE, RMSE_log (error metrics, lower is better)
  - delta_1, delta_2, delta_3  (accuracy metrics, higher is better)
"""

import numpy as np


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    Compute the full set of depth estimation metrics over valid GT pixels.

    Only pixels where gt > 0, pred is finite, and gt is finite are used.

    Returns dict with keys:
        abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3
    """
    valid = (gt > 0) & np.isfinite(pred) & np.isfinite(gt)
    if not np.any(valid):
        return {k: float("nan") for k in _METRIC_KEYS}

    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)

    abs_rel = float(np.mean(np.abs(p - g) / g))
    sq_rel = float(np.mean(((p - g) ** 2) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))

    p_log = np.log(np.clip(p, 1e-6, None))
    g_log = np.log(np.clip(g, 1e-6, None))
    rmse_log = float(np.sqrt(np.mean((p_log - g_log) ** 2)))

    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25**2))
    delta3 = float(np.mean(ratio < 1.25**3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


_METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")

ERROR_METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log")
ACCURACY_METRICS = ("delta1", "delta2", "delta3")


def format_metrics(m: dict[str, float], keys: tuple[str, ...] | None = None) -> str:
    """One-line human-readable summary of metrics."""
    keys = keys or _METRIC_KEYS
    parts = []
    for k in keys:
        v = m.get(k, float("nan"))
        if k.startswith("delta"):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)


def aggregate_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean of each metric across a list of per-sample dicts."""
    if not records:
        return {k: float("nan") for k in _METRIC_KEYS}
    agg = {}
    for k in _METRIC_KEYS:
        vals = [r[k] for r in records if not np.isnan(r.get(k, float("nan")))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg
