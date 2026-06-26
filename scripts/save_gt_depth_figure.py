#!/usr/bin/env python3
"""
Save ground-truth depth as a single figure in the same style as experiment
visualizations (magma colormap, 2–98% depth range, fill_border for display).
No title, colorbar, or axis labels — image only, for slides.

Example::

    uv run python scripts/save_gt_depth_figure.py --index 5 --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.data.nyu_utils import fill_border, load_nyu_mat
from depth_estimation.evaluation.visualization import _depth_range


def main() -> None:
    parser = argparse.ArgumentParser(description="Save GT depth map as in experiment figures.")
    parser.add_argument(
        "--mat-path",
        type=str,
        default="data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
        help="Path to nyu_depth_v2_labeled.mat",
    )
    parser.add_argument("--index", type=int, required=True, help="Sample index in the .mat order.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./outputs/figures",
        help="Directory for the output PNG (and optional .npy).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save raw GT depth as float32 .npy (meters).",
    )
    args = parser.parse_args()

    images, depths = load_nyu_mat(args.mat_path)
    n = images.shape[0]
    if not (0 <= args.index < n):
        raise SystemExit(f"index must be in [0, {n - 1}], got {args.index}")

    gt = depths[args.index].astype(np.float32)
    vmin, vmax = _depth_range(gt)
    gt_valid = gt > 0
    gt_filled = fill_border(gt, gt_valid)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.25))
    ax.imshow(gt_filled, cmap="magma", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"nyu_gt_{args.index:04d}.png")
    fig.savefig(out_png, dpi=args.dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")

    if args.save_npy:
        out_npy = os.path.join(args.out_dir, f"nyu_gt_{args.index:04d}.npy")
        np.save(out_npy, gt)
        print(f"Saved {out_npy}")


if __name__ == "__main__":
    main()
