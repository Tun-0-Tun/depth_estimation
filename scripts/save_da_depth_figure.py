#!/usr/bin/env python3
"""
Save Depth Anything relative depth for one NYU index, in the same visual style
as experiment depth panels (magma, 2–98% range on valid positives, fill_border).

By default the map is **linearly inverted** (``d_max - d`` on valid pixels) so that
near/far matches the usual metric-depth intuition (closer ≈ darker in magma), aligned
with GT figures. Use ``--no-invert`` for raw model polarity.

No title, colorbar, or axis labels — image only, for slides.

Example::

    uv run python scripts/save_da_depth_figure.py --index 5 --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.data.nyu_utils import fill_border, load_nyu_mat
from depth_estimation.evaluation.visualization import _depth_range
from depth_estimation.models.da_inference import infer_depth, load_da_model


def _ensure_rgb_hwc(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


def _invert_for_metric_style(d_rel: np.ndarray) -> np.ndarray:
    """
    Map relative depth so closer objects get smaller values (like metric z in meters),
    for consistent magma appearance vs GT panels.
    """
    valid = np.isfinite(d_rel) & (d_rel > 0)
    if not np.any(valid):
        return d_rel
    d_max = float(np.max(d_rel[valid]))
    return d_max - d_rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Save DA relative depth map (no labels).")
    parser.add_argument(
        "--mat-path",
        type=str,
        default="data/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    )
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out-dir", type=str, default="./outputs/figures")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save raw relative depth float32 .npy.",
    )
    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Do not flip polarity; visualize DA output as returned by the model.",
    )
    args = parser.parse_args()

    images, _ = load_nyu_mat(args.mat_path)
    n = images.shape[0]
    if not (0 <= args.index < n):
        raise SystemExit(f"index must be in [0, {n - 1}], got {args.index}")

    rgb = _ensure_rgb_hwc(images[args.index])
    pil = Image.fromarray(rgb).convert("RGB")

    processor, model, device, dtype = load_da_model(args.model_id)
    d_rel = infer_depth(processor, model, pil, device, dtype).astype(np.float32)

    d_vis = d_rel if args.no_invert else _invert_for_metric_style(d_rel)
    vmin, vmax = _depth_range(d_vis)
    d_valid = np.isfinite(d_vis) & (d_vis > 0)
    d_filled = fill_border(d_vis, d_valid)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.25))
    ax.imshow(d_filled, cmap="magma", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"nyu_da_{args.index:04d}.png")
    fig.savefig(out_png, dpi=args.dpi, pad_inches=0, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")

    if args.save_npy:
        out_npy = os.path.join(args.out_dir, f"nyu_da_{args.index:04d}.npy")
        np.save(out_npy, d_rel)
        print(f"Saved {out_npy}")


if __name__ == "__main__":
    main()
