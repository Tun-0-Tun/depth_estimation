"""KITTI raw-depth parquet layout (image, depth, mask) used for cross-domain training.

The ``WyettZ/kitti-raw-depth`` HF mirror stores each split as parquet shards with
columns: image (PNG bytes), depth (float32 rows), mask (bool rows), plus metadata.
Depth is metric (meters, 0..~80) and ``mask`` marks valid GT pixels.
"""

from __future__ import annotations

import glob
import io
import os

import numpy as np


def _decode_image(cell) -> np.ndarray:
    from PIL import Image

    b = cell["bytes"] if isinstance(cell, dict) else cell
    im = Image.open(io.BytesIO(b)).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def _decode_rows(cell) -> np.ndarray:
    """depth/mask are stored as an object array of per-row 1D arrays."""
    return np.stack([np.asarray(row) for row in cell])


def list_kitti_parquet_split(root: str, split: str) -> list[str]:
    """Return parquet shard paths for a split ('train' or 'val')."""
    pat = os.path.join(root, "data", f"{split}-*.parquet")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"No KITTI parquet shards matching {pat!r}")
    return paths


def build_kitti_index(root: str, split: str) -> list[tuple[str, int]]:
    """Flat index of (shard_path, row) across all shards of a split."""
    import pandas as pd

    index: list[tuple[str, int]] = []
    for path in list_kitti_parquet_split(root, split):
        n = pd.read_parquet(path, columns=["index"]).shape[0]
        index.extend((path, i) for i in range(n))
    return index


class KittiParquetReader:
    """Lazily reads (rgb, gt) from KITTI parquet shards, caching one shard at a time."""

    def __init__(self, root: str, split: str):
        self.index = build_kitti_index(root, split)
        self._cache_path: str | None = None
        self._cache_df = None

    def __len__(self) -> int:
        return len(self.index)

    def _df(self, path: str):
        import pandas as pd

        if path != self._cache_path:
            self._cache_df = pd.read_parquet(path)
            self._cache_path = path
        return self._cache_df

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        path, row = self.index[i]
        r = self._df(path).iloc[row]
        rgb = _decode_image(r["image"])
        depth = _decode_rows(r["depth"]).astype(np.float32)
        mask = _decode_rows(r["mask"]).astype(bool)
        depth[~mask] = 0.0
        depth[~np.isfinite(depth)] = 0.0
        return rgb, depth
