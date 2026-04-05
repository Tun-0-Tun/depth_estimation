"""
Flexible multi-method depth comparison figures.

Generates adaptive layouts:

- Row 1: RGB | [superpixels] | method_1 | method_2 | ... | GT
- Row 2 (optional): blank under RGB/SP | signed error (pred − GT) per method | blank under GT

The superpixel column appears automatically when any method returns ``labels``
in its extras dict. Error maps use a shared diverging scale (coolwarm) so
methods are comparable within one figure.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from depth_estimation.data.nyu_utils import fill_border


def _depth_range(gt: np.ndarray) -> tuple[float, float]:
    """Robust vmin/vmax from valid GT pixels (2nd–98th percentile)."""
    valid = gt > 0
    if np.any(valid):
        vmin = float(np.percentile(gt[valid], 2.0))
        vmax = float(np.percentile(gt[valid], 98.0))
        if vmax <= vmin:
            vmin, vmax = float(gt[valid].min()), float(gt[valid].max())
    else:
        vmin, vmax = float(gt.min()), float(gt.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _fill_pred(depth: np.ndarray, vmin: float, vmax: float, gt_valid: np.ndarray) -> np.ndarray:
    """Fill border artifacts in a prediction map."""
    pred_valid = np.isfinite(depth) & (depth >= vmin) & (depth <= vmax)
    combined = pred_valid & gt_valid
    return fill_border(depth, combined)


def _signed_error_maps(
    predictions: dict[str, np.ndarray],
    gt: np.ndarray,
    gt_valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    """
    Per-method signed error (pred - gt) in meters; NaN where invalid.
    Returns (diff_maps, lim) where lim is symmetric axis limit (98th pct of |error|).
    """
    gt64 = gt.astype(np.float64)
    diffs: dict[str, np.ndarray] = {}
    all_abs: list[np.ndarray] = []

    for name, pred in predictions.items():
        p = pred.astype(np.float64)
        valid = gt_valid & np.isfinite(p)
        d = np.full(gt.shape, np.nan, dtype=np.float64)
        d[valid] = p[valid] - gt64[valid]
        diffs[name] = d
        ae = np.abs(d[valid])
        if ae.size > 0:
            all_abs.append(ae)

    if not all_abs:
        return diffs, 1.0
    lim = float(np.percentile(np.concatenate(all_abs), 98.0))
    lim = max(lim, 1e-6)
    return diffs, lim


def make_comparison_figure(
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    predictions: dict[str, np.ndarray],
    extras: dict[str, dict[str, Any]],
    title: str | None = None,
    figscale: float = 4.0,
    show_prediction_diff: bool = True,
) -> plt.Figure:
    """
    Build a comparison figure with adaptive number of columns.

    Columns: RGB | [superpixels overlay] | pred_1 | pred_2 | ... | GT

    If ``show_prediction_diff`` is True, adds a second row with signed
    ``pred − GT`` (meters) for each method column (coolwarm, shared symmetric
    limits). RGB, superpixel, and GT columns have blank second rows.

    Args:
        rgb: (H, W, 3) uint8 image.
        gt_depth: (H, W) ground-truth depth.
        predictions: {label: (H, W) metric depth} for each method.
        extras: {label: extras_dict} — if any contains ``"labels"`` key,
                a superpixel overlay column is added.
        title: optional figure suptitle.
        figscale: width per column in inches.
        show_prediction_diff: add row of pred − GT error maps.
    """
    from skimage.segmentation import mark_boundaries

    gt = gt_depth.astype(np.float32)
    gt_valid = gt > 0
    vmin, vmax = _depth_range(gt)

    labels_for_sp = _find_superpixel_labels(extras)
    has_sp = labels_for_sp is not None

    method_names = list(predictions.keys())
    n_methods = len(method_names)
    n_cols = 1 + int(has_sp) + n_methods + 1  # RGB + [SP] + methods + GT

    single_row = not (show_prediction_diff and n_methods > 0)

    if single_row:
        fig, axes = plt.subplots(1, n_cols, figsize=(figscale * n_cols, figscale))
        if n_cols == 1:
            axes_arr = np.array([[axes]])
        else:
            axes_arr = np.asarray(axes, dtype=object).reshape(1, n_cols)
        diffs = {}
        err_lim = 1.0
    else:
        diffs, err_lim = _signed_error_maps(predictions, gt, gt_valid)
        fig, axes = plt.subplots(
            2,
            n_cols,
            figsize=(figscale * n_cols, figscale * 1.9),
            gridspec_kw={"height_ratios": [1.0, 0.88]},
        )
        axes_arr = np.asarray(axes, dtype=object)
        if axes_arr.ndim == 1:
            axes_arr = axes_arr.reshape(2, n_cols)

    def ax_at(r: int, c: int):
        if single_row:
            return axes_arr[0, c]
        return axes_arr[r, c]

    col = 0

    # RGB
    ax_at(0, col).imshow(rgb)
    ax_at(0, col).set_title("RGB")
    ax_at(0, col).axis("off")
    if not single_row:
        ax_at(1, col).axis("off")
    col += 1

    # Superpixels overlay
    if has_sp:
        sp_vis = mark_boundaries(rgb, labels_for_sp, color=(1, 1, 0), mode="thick")
        ax_at(0, col).imshow(sp_vis)
        n_sp = int(labels_for_sp.max()) + 1
        ax_at(0, col).set_title(f"Superpixels ({n_sp})")
        ax_at(0, col).axis("off")
        if not single_row:
            ax_at(1, col).axis("off")
        col += 1

    # Method predictions + error row
    for name in method_names:
        pred = predictions[name]
        filled = _fill_pred(pred, vmin, vmax, gt_valid)
        im = ax_at(0, col).imshow(filled, cmap="magma", vmin=vmin, vmax=vmax)
        ax_at(0, col).set_title(name)
        ax_at(0, col).axis("off")
        fig.colorbar(im, ax=ax_at(0, col), fraction=0.046, pad=0.04).set_label("Depth (m)")

        if not single_row:
            im_e = ax_at(1, col).imshow(
                diffs[name], cmap="coolwarm", vmin=-err_lim, vmax=err_lim
            )
            ax_at(1, col).set_title(f"Δ {name}\n(pred−GT, m)", fontsize=8)
            ax_at(1, col).axis("off")
            fig.colorbar(
                im_e, ax=ax_at(1, col), fraction=0.046, pad=0.04
            ).set_label("Error (m)")
        col += 1

    # GT
    gt_filled = fill_border(gt, gt_valid)
    im_gt = ax_at(0, col).imshow(gt_filled, cmap="magma", vmin=vmin, vmax=vmax)
    ax_at(0, col).set_title("GT depth")
    ax_at(0, col).axis("off")
    fig.colorbar(im_gt, ax=ax_at(0, col), fraction=0.046, pad=0.04).set_label("Depth (m)")
    if not single_row:
        ax_at(1, col).axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _find_superpixel_labels(extras: dict[str, dict[str, Any]]) -> np.ndarray | None:
    """Return the first ``labels`` array found in any method's extras, or None."""
    for method_extras in extras.values():
        if "labels" in method_extras:
            return method_extras["labels"]
    return None
