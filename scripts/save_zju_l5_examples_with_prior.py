#!/usr/bin/env python3
"""
Create a slide-ready panel with ZJU-L5 examples and L5 priors.

Per selected frame, the script renders 4 columns:
  1) RGB
  2) GT depth
  3) Prior depth built from L5 zones
  4) Prior mask overlay on RGB

Example:
  python scripts/save_zju_l5_examples_with_prior.py \
      --zju-root data/ZJUL5 \
      --split test \
      --indices 0 1 2 3 4 5 \
      --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.data.zju_l5 import (  # noqa: E402
    build_depth_prior_from_l5_zones,
    list_zju_l5_split,
    load_zju_l5_h5,
    load_zju_l5_l5_fields,
)


def _depth_range(depth: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return 0.0, 1.0
    d = depth[valid]
    return float(np.percentile(d, 2.0)), float(np.percentile(d, 98.0))


def _default_indices(n_total: int, n_rows: int) -> list[int]:
    k = max(1, min(n_rows, n_total))
    if k == 1:
        return [0]
    # Uniformly spread indices across the split for diverse scenes.
    return np.linspace(0, n_total - 1, k, dtype=np.int32).tolist()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save ZJU-L5 examples with depth priors.")
    p.add_argument("--zju-root", type=str, default="data/ZJUL5")
    p.add_argument("--manifest-rel", type=str, default="data.json")
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Manifest indices to visualize. If omitted, --num-rows are sampled uniformly.",
    )
    p.add_argument("--num-rows", type=int, default=6)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--out-dir", type=str, default="./outputs/figures")
    p.add_argument("--out-name", type=str, default="zju_l5_examples_with_prior.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = list_zju_l5_split(
        root=args.zju_root,
        split=args.split,
        manifest_rel=args.manifest_rel,
    )
    if not paths:
        raise SystemExit("No samples found for this split.")

    if args.indices is None:
        indices = _default_indices(len(paths), args.num_rows)
    else:
        indices = [int(i) for i in args.indices]
    bad = [i for i in indices if i < 0 or i >= len(paths)]
    if bad:
        raise SystemExit(f"indices out of range [0, {len(paths) - 1}]: {bad}")

    rows = len(indices)
    fig_h = max(3.0, 2.35 * rows)
    fig, axes = plt.subplots(rows, 4, figsize=(14.5, fig_h), squeeze=False)

    col_titles = ["RGB", "GT depth", "L5 prior depth", "Prior mask on RGB"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11)

    for r, idx in enumerate(indices):
        rgb, gt = load_zju_l5_h5(paths[idx])
        hist, fr, zone_mask = load_zju_l5_l5_fields(paths[idx])
        prior_depth, prior_mask = build_depth_prior_from_l5_zones(
            hist_data=hist,
            fr=fr,
            zone_mask=zone_mask,
            height=gt.shape[0],
            width=gt.shape[1],
        )

        vmin, vmax = _depth_range(gt)
        prior_vis = prior_depth.copy()
        prior_vis[~prior_mask] = np.nan

        ax0, ax1, ax2, ax3 = axes[r]
        ax0.imshow(rgb)
        ax1.imshow(gt, cmap="magma", vmin=vmin, vmax=vmax)
        ax2.imshow(prior_vis, cmap="magma", vmin=vmin, vmax=vmax)
        ax3.imshow(rgb)
        ax3.imshow(prior_mask.astype(np.float32), cmap="Reds", alpha=0.35, vmin=0.0, vmax=1.0)

        coverage = float(prior_mask.mean() * 100.0)
        ax0.set_ylabel(f"#{idx}\n{coverage:.1f}% prior", fontsize=9)

        for ax in (ax0, ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.tight_layout(pad=0.7)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

