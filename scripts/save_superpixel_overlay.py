#!/usr/bin/env python3
"""
RGB + SLIC superpixel boundaries (same segmentation as local calibration), no text.

Output is a plain image (numpy + OpenCV): no titles, axes, or colorbars.

Examples::

    uv run python scripts/save_superpixel_overlay.py --index 5 --n-segments 200

    uv run python scripts/save_superpixel_overlay.py \\
        --indices 0 4 8 12 \\
        --from-exp configs/exp_inr_compare.json --method-key local_bilateral \\
        --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
from skimage.segmentation import mark_boundaries

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.calibration.local_calibration import compute_superpixels
from depth_estimation.data.nyu_utils import load_nyu_mat


def _ensure_rgb_hwc(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_") or "sp"


def _resolve_n_segments(args: argparse.Namespace) -> int:
    if args.n_segments is not None:
        return int(args.n_segments)
    if args.from_exp and args.method_key:
        with open(args.from_exp) as f:
            exp = json.load(f)
        methods = exp.get("methods") or {}
        if args.method_key not in methods:
            raise KeyError(f"No key '{args.method_key}' in methods of {args.from_exp}")
        block = methods[args.method_key]
        if isinstance(block, dict) and "n_segments" in block:
            return int(block["n_segments"])
    return 200


def _save_rgb_png(path: str, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def main() -> None:
    p = argparse.ArgumentParser(
        description="NYU RGB with SLIC superpixel boundaries only (no labels).",
    )
    p.add_argument("--mat-path", type=str, default="data/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--indices", type=int, nargs="+", default=None)
    p.add_argument(
        "--n-segments",
        type=int,
        default=None,
        help="SLIC target count (default: from --from-exp method, else 200).",
    )
    p.add_argument(
        "--compactness",
        type=float,
        default=10.0,
        help="SLIC compactness (same as compute_superpixels).",
    )
    p.add_argument(
        "--from-exp",
        type=str,
        default=None,
        help="Experiment JSON: take n_segments from methods[method-key] if --n-segments omitted.",
    )
    p.add_argument("--method-key", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="./outputs/figures")
    p.add_argument("--out-name", type=str, default=None)
    p.add_argument(
        "--boundary-rgb",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 0.0),
        metavar=("R", "G", "B"),
        help="Boundary color in 0–1 (default: yellow, same as experiment superpixel column).",
    )
    p.add_argument(
        "--thin",
        action="store_true",
        help="Use thin boundaries; default matches experiments (mode='thick').",
    )
    args = p.parse_args()

    if args.indices is not None:
        idxs = list(args.indices)
    elif args.index is not None:
        idxs = [args.index]
    else:
        p.error("Provide --index or --indices.")

    if args.from_exp and not args.method_key:
        p.error("--from-exp requires --method-key.")

    n_seg = _resolve_n_segments(args)
    color = tuple(float(x) for x in args.boundary_rgb)
    mode = "inner" if args.thin else "thick"

    images, _depths = load_nyu_mat(args.mat_path)
    n_max = images.shape[0]
    for ix in idxs:
        if not (0 <= ix < n_max):
            raise SystemExit(f"Each index must be in [0, {n_max - 1}], got {ix}")

    row_blocks: list[np.ndarray] = []
    for ix in idxs:
        rgb = _ensure_rgb_hwc(images[ix])
        labels = compute_superpixels(rgb, n_segments=n_seg, compactness=args.compactness)
        overlay = mark_boundaries(rgb, labels, color=color, mode=mode)
        out_u8 = (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)
        row_blocks.append(out_u8)

    grid = np.vstack(row_blocks)
    multi = len(idxs) > 1
    tag = f"n{n_seg}"
    if args.from_exp and args.method_key:
        tag = f"{_slug(args.method_key)}_{tag}"
    if not multi:
        default_name = f"nyu_superpixels_{tag}_idx{idxs[0]:04d}.png"
    else:
        idx_part = "_".join(f"{i:04d}" for i in idxs)
        default_name = f"nyu_superpixels_{tag}_{idx_part}.png"

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name or default_name)
    _save_rgb_png(out_path, grid)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
