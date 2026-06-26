#!/usr/bin/env python3
"""
Two-column figure: calibrated metric depth (chosen method) | GT depth.
Same styling as experiment panels: magma, per-row vmin/vmax from GT, fill_border.

Modes:
  * Single sample: ``--index N``.
  * Several samples: ``--indices ...`` — rows stacked; optional header ``Method | GT``.

Images are built as **pixel arrays** (no matplotlib axes) so there are **no gaps**
between panels. Optional header is a thin white strip with centered text (OpenCV).

Examples::

    uv run python scripts/save_method_vs_gt_figure.py --index 5 \\
        --from-exp configs/exp_inr_compare.json --method-key global

    uv run python scripts/save_method_vs_gt_figure.py \\
        --indices 0 4 8 12 16 20 \\
        --from-exp configs/exp_inr_compare.json --method-key global

    Force Depth Anything on GPU (recommended if CUDA is installed)::

        ... --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import cv2
import matplotlib as mpl
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.calibration.methods import get_method
from depth_estimation.data.nyu_utils import fill_border, load_nyu_mat, simulate_sparse_prior
from depth_estimation.evaluation.visualization import _depth_range, _fill_pred
from depth_estimation.models.da_inference import infer_depth, load_da_model


def _ensure_rgb_hwc(img: np.ndarray) -> np.ndarray:
    x = img
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_") or "method"


def _load_method_params(args: argparse.Namespace) -> dict:
    if args.method_config:
        with open(args.method_config) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("method-config JSON must be a single object, e.g. {\"type\": \"global\"}")
        return dict(data)
    if args.from_exp and args.method_key:
        with open(args.from_exp) as f:
            exp = json.load(f)
        methods = exp.get("methods") or {}
        if args.method_key not in methods:
            raise KeyError(f"No key '{args.method_key}' in methods of {args.from_exp}")
        return dict(methods[args.method_key])
    raise RuntimeError("Provide --method-config PATH or --from-exp PATH --method-key NAME")


def _depth_to_rgb_u8(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Float depth (H,W) -> uint8 RGB (H,W,3) with magma (same as imshow)."""
    d = np.asarray(depth, dtype=np.float64)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = mpl.colormaps["magma"](norm(d))
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def _put_text_centered_row(
    strip: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    font: int,
    font_scale: float,
    color_bgr: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw text centered at (cx, cy); ``strip`` is BGR (OpenCV convention)."""
    (tw, th), _baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(cx - tw / 2)
    y = int(cy + th / 2)
    cv2.putText(strip, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def _render_header_strip(
    width: int,
    height: int,
    left: str,
    right: str,
) -> np.ndarray:
    """White RGB strip with two centered labels (half-width each). Built in BGR then converted."""
    strip_bgr = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(max(0.45, min(0.9, (height - 8) / 44.0)))
    thickness = max(1, int(round(font_scale * 2)))
    color_bgr = (20, 20, 20)
    mid_y = height // 2
    _put_text_centered_row(strip_bgr, left, width // 4, mid_y, font, font_scale, color_bgr, thickness)
    _put_text_centered_row(strip_bgr, right, 3 * width // 4, mid_y, font, font_scale, color_bgr, thickness)
    return cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)


def _save_rgb_png(path: str, rgb: np.ndarray, _dpi: int) -> None:
    # OpenCV imwrite does not set PNG DPI metadata; --dpi kept for CLI compatibility.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def main() -> None:
    p = argparse.ArgumentParser(description="Two-column figure: method metric depth | GT.")
    p.add_argument("--mat-path", type=str, default="data/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    p.add_argument("--index", type=int, default=None, help="Single sample index (if --indices not used).")
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Several sample indices (one row per index, stacked).",
    )
    p.add_argument("--sparse-density", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42, help="Sparse prior RNG base (same as + index in experiments).")
    p.add_argument("--model-id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for Depth Anything: cuda, cuda:0, cpu, mps. "
        "Default: auto (CUDA if available). Use --device cuda to fail fast if GPU is missing.",
    )
    p.add_argument("--method-config", type=str, default=None, help="JSON file: {\"type\": \"...\", ...}")
    p.add_argument("--from-exp", type=str, default=None, help="Experiment JSON (with \"methods\").")
    p.add_argument("--method-key", type=str, default=None, help="Key inside methods, e.g. inr_film")
    p.add_argument("--out-dir", type=str, default="./outputs/figures")
    p.add_argument("--out-name", type=str, default=None, help="Output PNG filename (default: auto).")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--show-labels",
        action="store_true",
        help="(Single-sample mode) Add a header row: Predicted | Ground truth.",
    )
    p.add_argument(
        "--no-header-row",
        action="store_true",
        help="With --indices: do not add the top 'Method | GT' strip.",
    )
    p.add_argument(
        "--header-left",
        type=str,
        default="Method",
        help="Header text above the method column (multi-row mode).",
    )
    p.add_argument(
        "--header-right",
        type=str,
        default="GT",
        help="Header text above the GT column (multi-row mode).",
    )
    p.add_argument(
        "--header-height",
        type=int,
        default=44,
        help="Header strip height in pixels (multi-row or --show-labels).",
    )
    args = p.parse_args()

    if args.indices is not None:
        idxs = list(args.indices)
    elif args.index is not None:
        idxs = [args.index]
    else:
        p.error("Provide --index (one sample) or --indices (several samples).")

    if args.method_config:
        pass
    elif args.from_exp and args.method_key:
        pass
    else:
        p.error("Provide --method-config FILE, or both --from-exp and --method-key.")

    raw = _load_method_params(args)
    params = dict(raw)
    type_name = params.pop("type", "global")
    method = get_method(type_name, **params)
    method.name = args.method_key or type_name

    images, depths = load_nyu_mat(args.mat_path)
    n_max = images.shape[0]
    for ix in idxs:
        if not (0 <= ix < n_max):
            raise SystemExit(f"Each index must be in [0, {n_max - 1}], got {ix}")

    processor, model, device, dtype = load_da_model(args.model_id, device=args.device)

    multi = len(idxs) > 1
    show_header = multi and not args.no_header_row

    row_blocks: list[np.ndarray] = []

    for ix in idxs:
        rgb = _ensure_rgb_hwc(images[ix])
        gt = depths[ix].astype(np.float32)
        sparse_depth, sparse_mask = simulate_sparse_prior(
            gt, density=args.sparse_density, seed=args.seed + ix,
        )
        d_rel = infer_depth(processor, model, rgb, device, dtype)
        pred, _extras = method.calibrate(
            d_rel, sparse_depth, sparse_mask, rgb, sample_index=ix,
        )
        vmin, vmax = _depth_range(gt)
        gt_valid = gt > 0
        pred_filled = _fill_pred(pred.astype(np.float32), vmin, vmax, gt_valid)
        gt_filled = fill_border(gt, gt_valid)

        left_rgb = _depth_to_rgb_u8(pred_filled, vmin, vmax)
        right_rgb = _depth_to_rgb_u8(gt_filled, vmin, vmax)
        row_blocks.append(np.hstack([left_rgb, right_rgb]))

    grid = np.vstack(row_blocks)
    w = grid.shape[1]

    if show_header:
        hdr = _render_header_strip(w, max(28, args.header_height), args.header_left, args.header_right)
        grid = np.vstack([hdr, grid])
    elif (not multi) and args.show_labels:
        hdr = _render_header_strip(
            w,
            max(28, args.header_height),
            "Predicted (metric)",
            "Ground truth (metric)",
        )
        grid = np.vstack([hdr, grid])

    tag = _slug(args.method_key or type_name)
    if not multi:
        default_name = f"nyu_{idxs[0]:04d}_{tag}_vs_gt.png"
    else:
        idx_part = "_".join(f"{i:04d}" for i in idxs)
        default_name = f"nyu_{tag}_vs_gt_{idx_part}.png"

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name or default_name)
    _save_rgb_png(out_path, grid, args.dpi)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
