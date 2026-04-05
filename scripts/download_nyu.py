#!/usr/bin/env python3
"""
Download and extract NYU Depth V2 dataset for monocular metric depth estimation.

Recommended: use --method mat (official .mat file). Hugging Face dataset scripts
are no longer supported in recent `datasets` versions.

Options:
  1) mat (default): download official nyu_depth_v2_labeled.mat (~2.8 GB), extract
     RGB + depth to images. Supports --max-samples.
  2) hf: attempt load via Hugging Face (may fail with "Dataset scripts are no longer supported").
"""

import argparse
import os
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_mat_v73(path: str):
    """Load MATLAB v7.3 (.mat) file via h5py (large files)."""
    import h5py

    with h5py.File(path, "r") as f:
        images = np.array(f["images"])   # shape often (3, 480, 640, N) in HDF5
        depths = np.array(f["depths"])   # (480, 640, N) or (N, 480, 640)
    return images, depths


def _load_mat_v7(path: str):
    """Load MATLAB v7 or earlier via scipy."""
    from scipy.io import loadmat

    data = loadmat(path)
    images = data["images"]   # (480, 640, 3, N) or (3, 640, 480, N)
    depths = data["depths"]  # (480, 640, N)
    return images, depths


def load_and_extract_mat(
    mat_path: str,
    out_dir: str,
    max_samples: Optional[int] = None,
):
    """Read .mat file and save RGB + depth as per-image files."""
    # Try v7.3 (HDF5) first, then v7
    try:
        images, depths = _load_mat_v73(mat_path)
        # h5py: images (3, 480, 640, N) -> need (N, 480, 640, 3)
        if images.ndim == 4:
            if images.shape[0] == 3:
                images = np.transpose(images, (3, 2, 1, 0))  # N, 640, 480, 3 -> N, 480, 640, 3
            else:
                images = np.transpose(images, (3, 0, 1, 2))
        # depths (480, 640, N) -> (N, 480, 640)
        if depths.ndim == 3 and depths.shape[2] != 3:
            depths = np.transpose(depths, (2, 0, 1))
    except (OSError, KeyError):
        images, depths = _load_mat_v7(mat_path)
        # scipy: images (480, 640, 3, N) -> (N, 480, 640, 3)
        if images.ndim == 4:
            images = np.transpose(images, (3, 0, 1, 2))
        if depths.ndim == 3 and depths.shape[0] != 480:
            depths = np.transpose(depths, (2, 0, 1))

    n = images.shape[0]
    if max_samples is not None:
        n = min(n, max_samples)
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    for i in tqdm(range(n), desc="Extracting"):
        rgb = images[i]
        if rgb.shape[2] == 3 and rgb.dtype in (np.uint8, np.int8):
            pass
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        Image.fromarray(rgb).save(os.path.join(img_dir, f"{i:06d}.png"))
        d = depths[i].astype(np.float32)
        # Save depth as 16-bit PNG (scale to mm or keep scale; NYU is in meters)
        d_mm = (np.clip(d, 0, 65535 / 1000.0) * 1000).astype(np.uint16)
        Image.fromarray(d_mm).save(os.path.join(depth_dir, f"{i:06d}.png"))
    print(f"Saved {n} samples to {out_dir}")
    return out_dir


def download_official_mat(save_dir: str, url: Optional[str] = None) -> str:
    """Download official NYU labeled .mat file (~2.8 GB)."""
    import urllib.request

    url = url or "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    path = os.path.join(save_dir, "nyu_depth_v2", "nyu_depth_v2_labeled.mat")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        print(f"File already exists: {path}")
        return path

    print(f"Downloading from {url} (~2.8 GB)...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Download manually from: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html")
        print(f"  and save as: {path}")
    return path


def download_via_huggingface(
    save_dir: str, split: str = "train", max_samples: Optional[int] = None
):
    """Download NYU Depth V2 via Hugging Face. May fail with recent datasets (scripts deprecated)."""
    from datasets import load_dataset

    cache_dir = os.path.join(save_dir, "nyu_depth_v2_hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading NYU Depth V2 from Hugging Face (sayakpaul/nyu_depth_v2)...")
    try:
        dataset = load_dataset(
            "sayakpaul/nyu_depth_v2",
            split=split,
            trust_remote_code=False,
            cache_dir=cache_dir,
        )
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print("This dataset uses deprecated loading scripts. Use --method mat instead.")
            raise SystemExit(1) from e
        raise

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Dataset split '{split}': {len(dataset)} samples (using up to {n})")

    out_dir = os.path.join(save_dir, "nyu_depth_v2", split)
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    for i in tqdm(range(n), desc="Saving"):
        sample = dataset[i]
        image = sample["image"]
        depth = sample["depth_map"]
        image.save(os.path.join(img_dir, f"{i:06d}.png"))
        depth.save(os.path.join(depth_dir, f"{i:06d}.png"))

    print(f"Saved to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Download NYU Depth V2")
    parser.add_argument("--save-dir", type=str, default="./data", help="Root directory for data")
    parser.add_argument(
        "--method",
        type=str,
        choices=["mat", "hf"],
        default="mat",
        help="mat = official .mat file (recommended), hf = Hugging Face (often broken)",
    )
    parser.add_argument("--split", type=str, default="train", help="For HF: train or validation")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--mat-path",
        type=str,
        default=None,
        help="Path to existing nyu_depth_v2_labeled.mat (skip download)",
    )
    args = parser.parse_args()

    if args.method == "mat":
        if args.mat_path and os.path.isfile(args.mat_path):
            mat_path = args.mat_path
        else:
            mat_path = download_official_mat(args.save_dir)
        if not os.path.isfile(mat_path):
            raise SystemExit("Missing .mat file. Download it manually (see message above).")
        out_dir = os.path.join(args.save_dir, "nyu_depth_v2", "labeled")
        load_and_extract_mat(mat_path, out_dir, args.max_samples)
    else:
        download_via_huggingface(args.save_dir, args.split, args.max_samples)


if __name__ == "__main__":
    main()
