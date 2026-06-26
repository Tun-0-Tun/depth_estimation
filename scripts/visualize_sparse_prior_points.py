#!/usr/bin/env python3
"""
Overlay a fixed number of sparse depth prior samples on the RGB image (same
sampling idea as simulate_sparse_prior: uniform over valid GT pixels).
No title or legend — image only, for slides.

Example (30 points, matches typical slides)::

    uv run python scripts/visualize_sparse_prior_points.py --index 5 --n-points 30 \\
        --seed 42 --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.data.nyu_utils import load_nyu_mat


def _ensure_rgb_hwc(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


def sample_prior_pixels(
    gt: np.ndarray,
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ys, xs, depth_values) for up to n_points valid pixels."""
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.isfinite(gt) & (gt > 0)
    flat = np.flatnonzero(valid.ravel())
    if flat.size == 0:
        raise ValueError("No valid depth pixels.")
    rng = np.random.default_rng(seed)
    k = min(int(n_points), int(flat.size))
    chosen = rng.choice(flat, size=k, replace=False)
    ys, xs = np.unravel_index(chosen, gt.shape)
    vals = gt[ys, xs]
    return ys.astype(np.int32), xs.astype(np.int32), vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw sparse prior points on RGB (for presentations)."
    )
    parser.add_argument(
        "--mat-path",
        type=str,
        default="data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    )
    parser.add_argument("--index", type=int, required=True, help="Sample index.")
    parser.add_argument(
        "--n-points",
        type=int,
        default=30,
        help="Number of prior samples (uniform over valid GT pixels).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for point placement.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./outputs/figures",
        help="Output directory for PNG.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=70.0,
        help="Scatter marker size (matplotlib `s`).",
    )
    args = parser.parse_args()

    images, depths = load_nyu_mat(args.mat_path)
    n = images.shape[0]
    if not (0 <= args.index < n):
        raise SystemExit(f"index must be in [0, {n - 1}], got {args.index}")

    rgb = _ensure_rgb_hwc(images[args.index])
    gt = depths[args.index].astype(np.float32)
    ys, xs, _ = sample_prior_pixels(gt, args.n_points, args.seed)

    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(rgb, aspect="equal", interpolation="nearest")

    # Layered markers: outer dark ring + bright fill for contrast on any scene
    ax.scatter(
        xs,
        ys,
        s=args.marker_size * 1.35,
        facecolors="none",
        edgecolors="#1a1a1a",
        linewidths=2.2,
        zorder=5,
    )
    ax.scatter(
        xs,
        ys,
        s=args.marker_size,
        facecolors="#ff2d55",
        edgecolors="white",
        linewidths=1.6,
        alpha=0.92,
        zorder=6,
    )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"nyu_prior_{args.n_points}pts_{args.index:04d}.png")
    fig.savefig(out_path, dpi=args.dpi, pad_inches=0, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
