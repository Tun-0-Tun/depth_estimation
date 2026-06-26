"""Unified multi-dataset access (NYU + KITTI + ZJU-L5) for cross-domain training.

Exposes a flat list of samples, each yielding ``(rgb, gt, l5_extra)`` where
``l5_extra`` is ``(hist, fr, mask)`` for ZJU-L5 frames (so the real L5 zone prior
can be used) and ``None`` otherwise.

A randomized sparse-prior generator (:func:`build_random_prior`) simulates many
different "bad sensors" on top of GT so the learned projection generalizes across
sensor patterns instead of memorizing one fixed layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from depth_estimation.data.kitti_parquet import KittiParquetReader
from depth_estimation.data.nyu_utils import (
    apply_depth_sensor_noise,
    load_nyu_mat,
    simulate_grid_stride_prior,
    simulate_sparse_prior,
)
from depth_estimation.data.zju_l5 import (
    list_zju_l5_split,
    load_zju_l5_h5,
    load_zju_l5_l5_fields,
    sparse_depth_and_mask_from_l5,
)

SampleLoader = Callable[[], tuple[np.ndarray, np.ndarray, tuple | None]]


@dataclass
class Sample:
    source: str  # 'nyu' | 'kitti' | 'zju_l5'
    key: int
    load: SampleLoader


def _ensure_rgb(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


def build_samples(
    spec: dict[str, Any],
    split: str,
    *,
    max_per_dataset: int | None = None,
    seed: int = 0,
) -> list[Sample]:
    """Build a flat sample list from a dataset spec.

    ``spec`` keys (all optional): ``nyu_mat``, ``kitti_root``, ``zju_l5_root``
    (+ ``zju_l5_manifest``). ``split`` selects within each source where applicable.
    For NYU (single .mat) a deterministic train/val partition is derived from ``seed``.
    """
    samples: list[Sample] = []
    rng = np.random.default_rng(seed)

    if spec.get("nyu_mat"):
        images, depths = load_nyu_mat(spec["nyu_mat"])
        n = images.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(round(0.1 * n)))
        val_idx = set(int(i) for i in perm[:n_val])
        want_val = split == "val"
        idxs = [i for i in range(n) if (int(i) in val_idx) == want_val]
        if max_per_dataset:
            idxs = idxs[:max_per_dataset]
        for i in idxs:
            ii = int(i)

            def _load(ii=ii):
                return _ensure_rgb(images[ii]), np.squeeze(depths[ii]).astype(np.float32), None

            samples.append(Sample("nyu", ii, _load))

    if spec.get("kitti_root"):
        ksplit = "val" if split == "val" else "train"
        reader = KittiParquetReader(spec["kitti_root"], ksplit)
        n = len(reader)
        idxs = list(range(n))
        if max_per_dataset:
            idxs = idxs[:max_per_dataset]
        for i in idxs:
            ii = int(i)

            def _load(ii=ii):
                rgb, gt = reader.load(ii)
                return rgb, gt, None

            samples.append(Sample("kitti", ii, _load))

    if spec.get("zju_l5_root"):
        zsplit = "test" if split == "val" else "train"
        paths = list_zju_l5_split(
            spec["zju_l5_root"], zsplit, spec.get("zju_l5_manifest", "data.json")
        )
        idxs = list(range(len(paths)))
        if max_per_dataset:
            idxs = idxs[:max_per_dataset]
        for i in idxs:
            p = paths[i]

            def _load(p=p):
                rgb, gt = load_zju_l5_h5(p)
                extra = load_zju_l5_l5_fields(p)
                return _ensure_rgb(rgb), gt.astype(np.float32), extra

            samples.append(Sample("zju_l5", int(i), _load))

    return samples


def build_random_prior(
    gt: np.ndarray,
    rng: np.random.Generator,
    *,
    l5_extra: tuple | None = None,
    noise: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly emulate a low-quality depth sensor on top of GT.

    Picks one of: real L5 zones (if available), regular low-res grid, or random
    sparse points; then applies heteroscedastic noise + outliers. This variety is
    what makes the learned rel->metric projection robust to the test sensor.
    """
    h, w = gt.shape
    choices = ["sparse", "grid"]
    if l5_extra is not None:
        choices = ["l5", "sparse", "grid"]
    mode = rng.choice(choices)
    seed = int(rng.integers(0, 2**31 - 1))

    if mode == "l5":
        hist, fr, zm = l5_extra
        sparse, mask = sparse_depth_and_mask_from_l5(hist, fr, zm, h, w)
        # randomly thin the zones to vary density
        keep = float(rng.uniform(0.4, 1.0))
        ys, xs = np.where(mask)
        if ys.size and keep < 1.0:
            n_keep = max(1, int(round(ys.size * keep)))
            pick = rng.choice(ys.size, size=n_keep, replace=False)
            m2 = np.zeros_like(mask)
            d2 = np.zeros_like(sparse)
            m2[ys[pick], xs[pick]] = True
            d2[ys[pick], xs[pick]] = sparse[ys[pick], xs[pick]]
            sparse, mask = d2, m2
    elif mode == "grid":
        stride = int(rng.integers(8, 48))
        sparse, mask = simulate_grid_stride_prior(gt, stride, seed, noise)
        return sparse, mask  # grid path already applied noise
    else:
        density = float(10.0 ** rng.uniform(-3.0, -1.3))  # ~0.001..0.05
        sparse, mask = simulate_sparse_prior(gt, density, seed, noise)
        return sparse, mask

    # apply noise to the L5 path
    if noise:
        ys, xs = np.where(mask)
        if ys.size:
            vals = apply_depth_sensor_noise(
                sparse[ys, xs], rng,
                relative_std=float(noise.get("relative_std", 0.0)),
                absolute_std_m=float(noise.get("absolute_std_m", 0.0)),
                outlier_fraction=float(noise.get("outlier_fraction", 0.0)),
                outlier_amp_m=float(noise.get("outlier_amp_m", 0.5)),
            )
            sparse = sparse.copy()
            sparse[ys, xs] = vals
    return sparse, mask
