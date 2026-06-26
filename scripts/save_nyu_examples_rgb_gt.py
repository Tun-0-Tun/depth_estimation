#!/usr/bin/env python3
"""
Slide-ready panel: RGB / GT depth (NYU Depth V2 ``.mat`` or ZJU-L5 HDF5).

Layouts:
  * ``--orientation wide``   (default) — samples are columns, rows are RGB / GT.
  * ``--orientation tall``             — samples are rows, columns are RGB / GT.

Examples::

    uv run python scripts/save_nyu_examples_rgb_gt.py --indices 0 1 2 3 4 5
    uv run python scripts/save_nyu_examples_rgb_gt.py --num-samples 6 --orientation wide
    uv run python scripts/save_nyu_examples_rgb_gt.py --dataset zju_l5 --zju-split test --indices 0 2 4 6 8 10
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.data.nyu_utils import load_nyu_mat  # noqa: E402
from depth_estimation.data.zju_l5 import list_zju_l5_split, load_zju_l5_h5  # noqa: E402


def _depth_range(depth: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return 0.0, 1.0
    d = depth[valid]
    return float(np.percentile(d, 2.0)), float(np.percentile(d, 98.0))


def _ensure_rgb_hwc(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


def _default_indices(n_total: int, n: int) -> list[int]:
    k = max(1, min(n, n_total))
    if k == 1:
        return [0]
    return np.linspace(0, n_total - 1, k, dtype=np.int32).tolist()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save RGB / GT depth example panels (NYU or ZJU-L5).")
    p.add_argument(
        "--dataset",
        type=str,
        choices=("nyu", "zju_l5"),
        default="nyu",
        help="Data source: NYU labeled mat, or ZJU-L5 HDF5 via manifest.",
    )
    p.add_argument(
        "--mat-path",
        type=str,
        default="data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    )
    p.add_argument("--zju-root", type=str, default="data/ZJUL5", help="Root for ZJU-L5 (--dataset zju_l5).")
    p.add_argument(
        "--zju-manifest",
        type=str,
        default="data.json",
        help="Manifest path relative to zju-root.",
    )
    p.add_argument("--zju-split", type=str, default="test", help="Split key in manifest (e.g. test, train).")
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Sample indices to visualize. If omitted, --num-samples are sampled uniformly.",
    )
    p.add_argument(
        "--num-samples",
        "--num-rows",
        dest="num_samples",
        type=int,
        default=6,
        help="Number of samples to display when --indices is not provided.",
    )
    p.add_argument(
        "--orientation",
        choices=("wide", "tall"),
        default="wide",
        help="wide: samples are columns (article-friendly). tall: samples are rows.",
    )
    p.add_argument("--cell-size", type=float, default=2.4, help="Per-image axis size (inches).")
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--out-dir", type=str, default="./outputs/figures")
    p.add_argument("--out-name", type=str, default="nyu_examples_rgb_gt.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ds = (args.dataset or "nyu").lower()
    paths: list[str] | None = None
    images = depths = None

    if ds == "nyu":
        images, depths = load_nyu_mat(args.mat_path)
        n_total = int(images.shape[0])
        if n_total == 0:
            raise SystemExit("Empty NYU dataset.")
    else:
        paths = list_zju_l5_split(
            root=args.zju_root,
            split=args.zju_split,
            manifest_rel=args.zju_manifest,
        )
        n_total = len(paths)
        if n_total == 0:
            raise SystemExit("No ZJU-L5 samples for this split.")

    if args.indices is None:
        indices = _default_indices(n_total, args.num_samples)
    else:
        indices = [int(i) for i in args.indices]
    bad = [i for i in indices if i < 0 or i >= n_total]
    if bad:
        raise SystemExit(f"indices out of range [0, {n_total - 1}]: {bad}")

    n = len(indices)
    if args.orientation == "wide":
        rows, cols = 2, n
        fig_w = max(3.0, args.cell_size * cols)
        fig_h = max(3.0, args.cell_size * rows * 0.78)
    else:
        rows, cols = n, 2
        fig_w = max(3.0, args.cell_size * 2 * 1.6)
        fig_h = max(3.0, args.cell_size * rows * 1.0)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for c, idx in enumerate(indices):
        if ds == "nyu":
            assert images is not None and depths is not None
            rgb = _ensure_rgb_hwc(images[idx])
            gt = np.squeeze(np.asarray(depths[idx], dtype=np.float32))
        else:
            assert paths is not None
            rgb, gt = load_zju_l5_h5(paths[idx])
            rgb = _ensure_rgb_hwc(rgb)
            gt = np.squeeze(np.asarray(gt, dtype=np.float32))
        vmin, vmax = _depth_range(gt)

        if args.orientation == "wide":
            ax_rgb = axes[0, c]
            ax_gt = axes[1, c]
        else:
            ax_rgb = axes[c, 0]
            ax_gt = axes[c, 1]

        ax_rgb.imshow(rgb)
        ax_gt.imshow(gt, cmap="magma", vmin=vmin, vmax=vmax)

        if args.orientation == "wide":
            ax_rgb.set_title(f"#{idx}", fontsize=10)
            if c == 0:
                ax_rgb.set_ylabel("RGB", fontsize=11)
                ax_gt.set_ylabel("GT depth", fontsize=11)
        else:
            ax_rgb.set_ylabel(f"#{idx}", fontsize=9)
            if c == 0:
                ax_rgb.set_title("RGB", fontsize=11)
                ax_gt.set_title("GT depth", fontsize=11)

        for ax in (ax_rgb, ax_gt):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.tight_layout(pad=0.5)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
