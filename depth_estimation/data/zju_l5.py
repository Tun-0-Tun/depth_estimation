"""ZJU-L5-style HDF5 layout (rgb, depth, hist_data, fr, mask)."""

from __future__ import annotations

import json
import os

import h5py
import numpy as np


def list_zju_l5_split(root: str, split: str, manifest_rel: str = "data.json") -> list[str]:
    manifest_path = os.path.join(root, manifest_rel)
    with open(manifest_path, "r") as f:
        data = json.load(f)
    if split not in data:
        raise KeyError(f"Split '{split}' not in {manifest_path}; keys: {list(data.keys())}")
    return [os.path.join(root, entry["filename"]) for entry in data[split]]


def load_zju_l5_h5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        rgb = np.asarray(f["rgb"])
        depth = np.asarray(f["depth"], dtype=np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return rgb, depth


def load_zju_l5_l5_fields(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        hist = np.asarray(f["hist_data"])
        fr = np.asarray(f["fr"])
        mask = np.asarray(f["mask"])
    return hist, fr, mask


def sparse_depth_and_mask_from_l5(
    hist_data: np.ndarray,
    fr: np.ndarray,
    zone_mask: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse depth from L5 zone centres (same semantics as Depthor dtof_to_sparse_depth)."""
    mean = hist_data[:, 0].astype(np.float64)
    c = (fr[:, :2].astype(np.float64) + fr[:, 2:].astype(np.float64)) / 2.0
    ri = np.rint(c[:, 0]).astype(np.int64)
    rj = np.rint(c[:, 1]).astype(np.int64)
    ri = np.clip(ri, 0, height - 1)
    rj = np.clip(rj, 0, width - 1)

    sparse = np.zeros((height, width), dtype=np.float32)
    m = np.zeros((height, width), dtype=bool)
    zmask = zone_mask.astype(bool) if zone_mask.ndim == 1 else zone_mask.reshape(-1).astype(bool)
    for i in range(mean.shape[0]):
        if not zmask[i]:
            continue
        mi = mean[i]
        if not np.isfinite(mi) or mi <= 0:
            continue
        sparse[ri[i], rj[i]] = np.float32(mi)
        m[ri[i], rj[i]] = True
    return sparse, m


def build_depth_prior_from_l5_zones(
    hist_data: np.ndarray,
    fr: np.ndarray,
    zone_mask: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense-ish prior by filling each zone bounding box with zone mean depth."""
    mean = hist_data[:, 0].astype(np.float32)
    prior = np.zeros((height, width), dtype=np.float32)
    pm = np.zeros((height, width), dtype=bool)
    zmask = zone_mask.astype(bool) if zone_mask.ndim == 1 else zone_mask.reshape(-1).astype(bool)
    for i in range(mean.shape[0]):
        if not zmask[i]:
            continue
        mi = float(mean[i])
        if not np.isfinite(mi) or mi <= 0:
            continue
        r0, c0, r1, c1 = fr[i].astype(int).tolist()
        r_low, r_high = sorted((max(0, min(height - 1, r0)), max(0, min(height - 1, r1))))
        c_low, c_high = sorted((max(0, min(width - 1, c0)), max(0, min(width - 1, c1))))
        prior[r_low : r_high + 1, c_low : c_high + 1] = mi
        pm[r_low : r_high + 1, c_low : c_high + 1] = True
    return prior, pm
