"""NYU Depth V2 loading and sparse-prior utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _load_mat_v73(path: str):
    import h5py

    with h5py.File(path, "r") as f:
        images = np.array(f["images"])
        depths = np.array(f["depths"])
    return images, depths


def _load_mat_v7(path: str):
    from scipy.io import loadmat

    data = loadmat(path)
    return data["images"], data["depths"]


def _normalize_images(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected 4D images, got {images.shape}")

    # (3, H, W, N) -> (N, H, W, 3)
    if images.shape[0] == 3 and images.shape[-1] != 3:
        return np.transpose(images, (3, 2, 1, 0))

    # (N, 3, A, B) where A/B can be H/W or W/H
    if images.shape[1] == 3:
        _, _, a, b = images.shape
        if a == 640 and b == 480:  # (N,C,W,H) -> (N,H,W,C)
            return np.transpose(images, (0, 3, 2, 1))
        return np.transpose(images, (0, 2, 3, 1))  # assume (N,C,H,W)

    # (H, W, 3, N) -> (N, H, W, 3)
    if images.shape[2] == 3:
        return np.transpose(images, (3, 0, 1, 2))

    raise ValueError(f"Unsupported images layout: {images.shape}")


def _normalize_depths(depths: np.ndarray, n: int) -> np.ndarray:
    if depths.ndim != 3:
        raise ValueError(f"Expected 3D depths, got {depths.shape}")

    # (N, H, W)
    if depths.shape[0] == n and depths.shape[1] == 480 and depths.shape[2] == 640:
        return depths
    # (N, W, H)
    if depths.shape[0] == n and depths.shape[1] == 640 and depths.shape[2] == 480:
        return np.transpose(depths, (0, 2, 1))
    # (H, W, N)
    if depths.shape[0] == 480 and depths.shape[1] == 640 and depths.shape[2] == n:
        return np.transpose(depths, (2, 0, 1))
    # (W, H, N)
    if depths.shape[0] == 640 and depths.shape[1] == 480 and depths.shape[2] == n:
        return np.transpose(depths, (2, 1, 0))
    if depths.shape[0] == n:
        return depths
    raise ValueError(f"Unsupported depths layout: {depths.shape}")


def load_nyu_mat(mat_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return images (N,H,W,3) uint8 and depths (N,H,W) float32."""
    try:
        images, depths = _load_mat_v73(mat_path)
    except (OSError, KeyError):
        images, depths = _load_mat_v7(mat_path)
    images = _normalize_images(images)
    depths = _normalize_depths(depths, images.shape[0])
    images = np.ascontiguousarray(np.clip(images, 0, 255).astype(np.uint8))
    depths = np.ascontiguousarray(np.asarray(depths, dtype=np.float32))
    return images, depths


def apply_depth_sensor_noise(
    depths: np.ndarray,
    rng: np.random.Generator,
    relative_std: float = 0.0,
    absolute_std_m: float = 0.0,
    outlier_fraction: float = 0.0,
    outlier_amp_m: float = 0.5,
) -> np.ndarray:
    """
    Corrupt depth samples as a coarse RGB-D / ToF prior (not GT).

    Model (applied per sample, iid):
      * Heteroscedastic Gaussian: d' = d + N(0, (absolute_std_m + relative_std * d)^2).
      * Optional outliers (multipath / flying pixels): fraction ``outlier_fraction``
        of values get an extra N(0, outlier_amp_m^2) offset on top.

    ``depths`` must be positive finite; output is clipped to a small positive floor.
    """
    d = np.asarray(depths, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return d.astype(np.float32)

    sigma = absolute_std_m + relative_std * np.maximum(d, 1e-6)
    noisy = d + rng.normal(0.0, sigma)

    p = float(outlier_fraction)
    if p > 0.0:
        u = rng.random(d.shape)
        out = u < p
        if np.any(out):
            n = int(out.sum())
            noisy[out] = noisy[out] + rng.normal(0.0, float(outlier_amp_m), size=n)

    noisy = np.maximum(noisy, 1e-4)
    return noisy.astype(np.float32)


def simulate_grid_stride_prior(
    gt_depth: np.ndarray,
    stride: int,
    seed: int,
    prior_noise: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Depth prior on a regular grid (e.g. low-res sensor upsampled to full frame).

    Pixels at ``(y, x)`` with ``y in range(0, H, stride)``, ``x in range(0, W, stride)``
    that have valid GT depth are used; values optionally corrupted by ``prior_noise``.
    Stride 4 ≈ 4× lower resolution per axis (every 4th pixel; three pixels between samples).
    """
    gt_depth = np.ascontiguousarray(np.asarray(gt_depth, dtype=np.float32))
    if gt_depth.ndim != 2:
        gt_depth = np.squeeze(gt_depth)
    if gt_depth.ndim != 2:
        raise ValueError(f"gt_depth must be (H, W), got {gt_depth.shape}")

    st = int(stride)
    if st < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    h, w = gt_depth.shape
    ys = np.arange(0, h, st, dtype=np.intp)
    xs = np.arange(0, w, st, dtype=np.intp)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.ravel()
    xx = xx.ravel()

    valid = np.isfinite(gt_depth[yy, xx]) & (gt_depth[yy, xx] > 0)
    yy = yy[valid]
    xx = xx[valid]

    sparse_depth = np.zeros_like(gt_depth, dtype=np.float32)
    sparse_mask = np.zeros_like(gt_depth, dtype=bool)
    if yy.size == 0:
        return sparse_depth, sparse_mask

    rng = np.random.default_rng(seed)
    vals = gt_depth[yy, xx].astype(np.float32)
    pn = _prior_noise_from_dict(prior_noise)
    if pn and (
        pn["relative_std"] > 0
        or pn["absolute_std_m"] > 0
        or pn["outlier_fraction"] > 0
    ):
        vals = apply_depth_sensor_noise(vals, rng, **pn)
    sparse_depth[yy, xx] = vals
    sparse_mask[yy, xx] = True
    return sparse_depth, sparse_mask


def _prior_noise_from_dict(cfg: dict[str, Any] | None) -> dict[str, float] | None:
    if not cfg:
        return None
    return {
        "relative_std": float(cfg.get("relative_std", 0.0)),
        "absolute_std_m": float(cfg.get("absolute_std_m", 0.0)),
        "outlier_fraction": float(cfg.get("outlier_fraction", 0.0)),
        "outlier_amp_m": float(cfg.get("outlier_amp_m", 0.5)),
    }


def simulate_sparse_prior(
    gt_depth: np.ndarray,
    density: float,
    seed: int,
    prior_noise: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gt_depth = np.ascontiguousarray(np.asarray(gt_depth, dtype=np.float32))
    if gt_depth.ndim != 2:
        gt_depth = np.squeeze(gt_depth)
    if gt_depth.ndim != 2:
        raise ValueError(f"gt_depth must be (H, W), got {gt_depth.shape}")

    valid = np.isfinite(gt_depth) & (gt_depth > 0)
    flat_valid = np.flatnonzero(valid.ravel())
    n_valid = int(flat_valid.size)
    if n_valid == 0:
        return np.zeros_like(gt_depth, dtype=np.float32), np.zeros_like(gt_depth, dtype=bool)

    rng = np.random.default_rng(seed)
    n_keep = max(1, min(n_valid, int(round(n_valid * float(density)))))
    chosen = rng.choice(flat_valid, size=n_keep, replace=False)

    sparse_depth = np.zeros_like(gt_depth, dtype=np.float32)
    sparse_mask = np.zeros_like(gt_depth, dtype=bool)
    vals = gt_depth.flat[chosen].astype(np.float32)
    pn = _prior_noise_from_dict(prior_noise)
    if pn and (
        pn["relative_std"] > 0
        or pn["absolute_std_m"] > 0
        or pn["outlier_fraction"] > 0
    ):
        vals = apply_depth_sensor_noise(vals, rng, **pn)
    sparse_depth.flat[chosen] = vals
    sparse_mask.flat[chosen] = True
    return sparse_depth, sparse_mask


def fill_border(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid pixels by nearest valid neighbor value."""
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return depth.astype(np.float32)
    out = depth.copy()
    invalid = ~valid
    yv, xv = np.where(valid)
    if yv.size == 0:
        return np.nan_to_num(out, nan=0.0).astype(np.float32)
    yi, xi = np.where(invalid)
    tree = cKDTree(np.column_stack((yv, xv)))
    _, nn = tree.query(np.column_stack((yi, xi)))
    out[yi, xi] = depth[yv[nn], xv[nn]]
    return out.astype(np.float32)
