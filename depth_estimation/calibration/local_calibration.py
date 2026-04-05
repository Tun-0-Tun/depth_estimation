"""
Local affine calibration of relative depth via SLIC superpixels.

Pipeline:
  1. Segment RGB into superpixels (SLIC).
  2. Per-superpixel closed-form LS fit of s_k, t_k.
  3. Optionally smooth s(x,y), t(x,y): Gaussian (``smooth_fields``) or
     edge-preserving joint bilateral (``smooth_fields_bilateral``).
  4. Apply: D_met(x,y) = s(x,y) * d_rel(x,y) + t(x,y).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from skimage.segmentation import slic

from depth_estimation.calibration.global_baseline import fit_global_scale_shift


def compute_superpixels(rgb: np.ndarray, n_segments: int = 200, compactness: float = 10.0) -> np.ndarray:
    """
    Compute SLIC superpixels on an RGB image.

    Args:
        rgb: (H, W, 3) uint8 image.
        n_segments: target number of superpixels.
        compactness: SLIC compactness parameter (higher = more regular shapes).

    Returns:
        labels: (H, W) int array, superpixel id per pixel.
    """
    labels = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=0)
    return labels.astype(np.int32)


def fit_per_superpixel(
    d_rel: np.ndarray,
    d_gt: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray | None = None,
    fallback_s: float = 0.0,
    fallback_t: float = 1.0,
    min_pixels: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit affine s_k, t_k per superpixel by closed-form LS.

    Args:
        d_rel: (H, W) relative depth from DA.
        d_gt:  (H, W) GT metric depth.
        labels: (H, W) superpixel label map.
        valid_mask: (H, W) bool — which GT pixels are usable. None => (d_gt > 0).
        fallback_s, fallback_t: used when a superpixel has too few valid pixels.
        min_pixels: minimum valid pixels per superpixel for a reliable fit.

    Returns:
        s_map: (H, W) float64 — per-pixel scale (constant within each superpixel).
        t_map: (H, W) float64 — per-pixel shift.
    """
    if valid_mask is None:
        valid_mask = np.isfinite(d_gt) & (d_gt > 0)

    d_rel_f = d_rel.astype(np.float64)
    d_gt_f = d_gt.astype(np.float64)

    n_sp = int(labels.max()) + 1
    s_vals = np.full(n_sp, fallback_s, dtype=np.float64)
    t_vals = np.full(n_sp, fallback_t, dtype=np.float64)

    for k in range(n_sp):
        mask_k = (labels == k) & valid_mask
        count = int(mask_k.sum())
        if count < min_pixels:
            continue

        x = d_rel_f[mask_k]
        y = d_gt_f[mask_k]

        x_mean = x.mean()
        y_mean = y.mean()
        x_c = x - x_mean

        denom = float((x_c ** 2).sum())
        if denom <= 1e-8:
            continue

        s_k = float((x_c * (y - y_mean)).sum() / denom)
        t_k = y_mean - s_k * x_mean
        s_vals[k] = s_k
        t_vals[k] = t_k

    s_map = s_vals[labels]
    t_map = t_vals[labels]
    return s_map, t_map


def smooth_fields(
    s_map: np.ndarray,
    t_map: np.ndarray,
    sigma: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-smooth the piecewise-constant s(x,y) and t(x,y) fields.
    """
    s_smooth = gaussian_filter(s_map.astype(np.float64), sigma=sigma)
    t_smooth = gaussian_filter(t_map.astype(np.float64), sigma=sigma)
    return s_smooth, t_smooth


def smooth_fields_bilateral(
    s_map: np.ndarray,
    t_map: np.ndarray,
    sigma_spatial: float = 5.0,
    sigma_range_s: float | None = None,
    sigma_range_t: float | None = None,
    range_scale: float = 0.25,
    max_radius: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Edge-preserving (joint bilateral) smoothing of piecewise-constant s, t maps.

    At each pixel, neighbors are weighted by a spatial Gaussian and a *range*
    term in (s, t): similar calibration parameters blend across superpixel
    boundaries; large jumps in (s, t) get down-weighted, so boundaries stay
    sharp where the affine fit changes strongly.

    Args:
        s_map, t_map: (H, W) float — typically constant within SLIC regions.
        sigma_spatial: Gaussian sigma in pixels for spatial weights.
        sigma_range_s, sigma_range_t: range sigmas in s / t units. If None,
            set to ``range_scale * std(map)`` (with a small epsilon).
        range_scale: Used only when sigma_range_* are None.
        max_radius: Cap on half-window size to bound memory (window is
            ``(2*max_radius+1)^2``; sliding windows are O(H*W*r^2)).

    Returns:
        Smoothed ``s_map``, ``t_map`` as float64.
    """
    s = s_map.astype(np.float32)
    t = t_map.astype(np.float32)
    h, w = s.shape
    r = int(round(2.5 * float(sigma_spatial)))
    r = max(1, min(r, int(max_radius)))
    kh, kw = 2 * r + 1, 2 * r + 1

    if sigma_range_s is None:
        sigma_range_s = max(float(np.std(s)), 1e-9) * float(range_scale)
    if sigma_range_t is None:
        sigma_range_t = max(float(np.std(t)), 1e-9) * float(range_scale)

    pad = ((r, r), (r, r))
    ps = sliding_window_view(np.pad(s, pad, mode="edge"), (kh, kw))
    pt = sliding_window_view(np.pad(t, pad, mode="edge"), (kh, kw))
    if ps.shape[0] != h or ps.shape[1] != w:
        raise RuntimeError(
            f"Bilateral window shape mismatch: got {ps.shape}, expected ({h},{w},...)"
        )

    cs = ps[:, :, r, r][:, :, np.newaxis, np.newaxis]
    ct = pt[:, :, r, r][:, :, np.newaxis, np.newaxis]
    ds = (ps - cs) / float(sigma_range_s)
    dtt = (pt - ct) / float(sigma_range_t)
    range_w = np.exp(-0.5 * (ds * ds + dtt * dtt)).astype(np.float32)

    ii, jj = np.indices((kh, kw), dtype=np.float32)
    spatial_w = np.exp(
        -0.5 * ((ii - r) ** 2 + (jj - r) ** 2) / (float(sigma_spatial) ** 2 + 1e-12)
    ).astype(np.float32)

    wts = spatial_w * range_w
    wsum = wts.sum(axis=(2, 3))
    wsum_safe = np.maximum(wsum, 1e-12)

    out_s = (wts * ps).sum(axis=(2, 3)) / wsum_safe
    out_t = (wts * pt).sum(axis=(2, 3)) / wsum_safe
    return out_s.astype(np.float64), out_t.astype(np.float64)


def apply_local_calibration(
    d_rel: np.ndarray,
    s_map: np.ndarray,
    t_map: np.ndarray,
) -> np.ndarray:
    """
    Apply per-pixel affine calibration: D_met(x,y) = s(x,y) * d_rel(x,y) + t(x,y).
    """
    return (s_map * d_rel.astype(np.float64) + t_map).astype(np.float32)


def calibrate_local(
    d_rel: np.ndarray,
    d_gt: np.ndarray,
    rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
    n_segments: int = 200,
    sigma: float = 15.0,
    min_pixels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full local calibration pipeline.

    Args:
        d_rel: (H, W) relative depth from DA.
        d_gt:  (H, W) GT/prior metric depth (can be sparse — zeros where unknown).
        rgb:   (H, W, 3) uint8 image.
        valid_mask: (H, W) bool — which pixels of d_gt to use for fitting.
                    None => all pixels where d_gt > 0.
        n_segments: SLIC superpixel count.
        sigma: Gaussian smoothing sigma for s/t fields.
        min_pixels: min valid pixels per superpixel.

    Returns:
        d_metric: (H, W) float32 — locally calibrated metric depth.
        s_map:    (H, W) float64 — smoothed scale field.
        t_map:    (H, W) float64 — smoothed shift field.
        labels:   (H, W) int32   — superpixel label map.
    """
    if valid_mask is None:
        valid_mask = np.isfinite(d_gt) & (d_gt > 0)

    fallback_s, fallback_t = fit_global_scale_shift(d_rel, d_gt, valid_mask)

    labels = compute_superpixels(rgb, n_segments=n_segments)

    s_map, t_map = fit_per_superpixel(
        d_rel, d_gt, labels,
        valid_mask=valid_mask,
        fallback_s=fallback_s,
        fallback_t=fallback_t,
        min_pixels=min_pixels,
    )

    #s_map, t_map = smooth_fields(s_map, t_map, sigma=sigma)

    d_metric = apply_local_calibration(d_rel, s_map, t_map)
    return d_metric, s_map, t_map, labels
