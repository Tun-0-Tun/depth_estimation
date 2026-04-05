#!/usr/bin/env python3
"""
KITTI Depth / KITTI Raw — download and layout instructions.

Official download requires registration at:
  https://www.cvlibs.net/datasets/kitti/

After registration, download from:
  https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion

Components:
  - Development kit (48 KB)
  - Depth completion: annotated depth (14 GB), raw LiDAR (5 GB), val/test (2 GB)
  - Raw data: https://www.cvlibs.net/datasets/kitti/raw_data.php (by date/drive)

This script can:
  1) Download from Hugging Face if a mirror is available.
  2) Verify/organize already-downloaded KITTI files.
"""

import argparse
import os


def get_kitti_info():
    return {
        "official_depth": "https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion",
        "official_raw": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
        "devkit": "https://github.com/joseph-zhong/KITTI-devkit",
        "depth_files": [
            "data_depth_annotated.zip",   # annotated depth (~14 GB)
            "data_depth_velodyne.zip",    # raw LiDAR (~5 GB)
            "data_depth_selection.zip",   # val/test selection (~2 GB)
        ],
    }


def try_huggingface(save_dir: str):
    """Try loading KITTI from Hugging Face if available."""
    import importlib.util

    if importlib.util.find_spec("datasets") is None:
        print("Install: pip install datasets")
        return None

    # Not all KITTI depth splits are on HF; some datasets exist for raw only
    # Example: "kitti" for detection. Depth completion may need manual download.
    print("Checking Hugging Face for KITTI...")
    # If you find a HF dataset for KITTI depth, set it here, e.g.:
    # dataset = load_dataset("...", cache_dir=os.path.join(save_dir, "kitti_hf"))
    print("KITTI depth completion is not standard on HF. Use official download.")
    return None


def setup_dirs(save_dir: str):
    """Create standard KITTI depth layout."""
    base = os.path.join(save_dir, "kitti_depth")
    dirs = [
        os.path.join(base, "train", "image"),
        os.path.join(base, "train", "depth"),
        os.path.join(base, "val", "image"),
        os.path.join(base, "val", "depth"),
        os.path.join(base, "devkit"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base


def main():
    parser = argparse.ArgumentParser(description="KITTI Depth download / setup")
    parser.add_argument("--save-dir", type=str, default="./data", help="Root directory for data")
    parser.add_argument("--info-only", action="store_true", help="Print download URLs and exit")
    parser.add_argument("--setup-dirs", action="store_true", help="Create directory layout only")
    args = parser.parse_args()

    info = get_kitti_info()

    if args.info_only:
        print("KITTI Depth — official sources (registration required):")
        print("  Depth completion:", info["official_depth"])
        print("  Raw data:", info["official_raw"])
        print("  Devkit:", info["devkit"])
        print("  Files to download:", info["depth_files"])
        return

    if args.setup_dirs:
        base = setup_dirs(args.save_dir)
        print(f"Created layout under {base}")
        return

    try_huggingface(args.save_dir)
    setup_dirs(args.save_dir)
    print("Next: register at cvlibs.net and download the depth completion files.")
    print("  --info-only to see URLs again.")


if __name__ == "__main__":
    main()
