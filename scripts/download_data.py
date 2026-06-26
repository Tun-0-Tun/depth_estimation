#!/usr/bin/env python3
"""Download the datasets needed to train / evaluate the rel2metric model.

    python scripts/download_data.py --all
    python scripts/download_data.py --nyu --kitti        # auto-downloadable parts
    python scripts/download_data.py --zju-info           # print ZJU-L5 instructions

Layout produced (relative to repo root)::

    data/
      nyu_depth_v2/nyu_depth_v2_labeled.mat      # NYU labeled (~2.8 GB, public)
      kitti_raw_depth/data/*.parquet             # KITTI raw-depth (HF mirror, ~0.5 GB)
      ZJUL5/data.json + <scene>/*.h5             # ZJU-L5 (manual, see --zju-info)

NYU and KITTI download automatically. ZJU-L5 is gated (DELTAR authors) and must be
fetched manually; this script only prints where to put it.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO_ROOT, "data")

NYU_URL = "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_SIZE = 2_972_037_809
KITTI_REPO = "WyettZ/kitti-raw-depth"


def _report(blocks: int, bs: int, total: int) -> None:
    if total <= 0:
        return
    done = min(blocks * bs, total)
    pct = 100.0 * done / total
    sys.stdout.write(f"\r  NYU: {done / 1e9:5.2f} / {total / 1e9:.2f} GB ({pct:5.1f}%)")
    sys.stdout.flush()


def download_nyu() -> None:
    out_dir = os.path.join(DATA, "nyu_depth_v2")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "nyu_depth_v2_labeled.mat")
    if os.path.exists(dst) and abs(os.path.getsize(dst) - NYU_SIZE) < 1_000_000:
        print(f"NYU already present: {dst}")
        return
    tmp = dst + ".part"
    print(f"Downloading NYU labeled -> {dst}")
    urllib.request.urlretrieve(NYU_URL, tmp, reporthook=_report)
    sys.stdout.write("\n")
    if abs(os.path.getsize(tmp) - NYU_SIZE) > 10_000_000:
        raise RuntimeError(f"NYU size mismatch: got {os.path.getsize(tmp)} expected ~{NYU_SIZE}")
    os.replace(tmp, dst)
    print("NYU done.")


def download_kitti() -> None:
    from huggingface_hub import snapshot_download

    out_dir = os.path.join(DATA, "kitti_raw_depth")
    print(f"Downloading KITTI raw-depth ({KITTI_REPO}) -> {out_dir}")
    snapshot_download(
        repo_id=KITTI_REPO,
        repo_type="dataset",
        local_dir=out_dir,
        allow_patterns=["*.parquet", "*.md", "*.json"],
        max_workers=4,
    )
    n = len([f for f in os.listdir(os.path.join(out_dir, "data")) if f.endswith(".parquet")])
    print(f"KITTI done ({n} parquet shards).")


ZJU_INFO = """\
ZJU-L5 (DELTAR) — manual download (gated by the authors)
--------------------------------------------------------
The L5 dToF + RGB + GT dataset comes from DELTAR (Du et al., ECCV 2022).
1. Get it from the DELTAR project: https://github.com/zju3dv/DELTAR
   (follow their "Dataset" section link to the ZJU-L5 archive).
2. Extract it so the repo sees:

     data/ZJUL5/data.json
     data/ZJUL5/<scene>/<timestamp>.h5      # keys: rgb, depth, hist_data, fr, mask

   data.json must contain {"train": [...], "test": [...]} with
   {"filename": "<scene>/<ts>.h5"} entries (483 train / 527 test in the
   reference split).
3. Verify:  python -c "from depth_estimation.data.zju_l5 import list_zju_l5_split as L; \
print(len(L('data/ZJUL5','train')), len(L('data/ZJUL5','test')))"
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Download NYU + KITTI (ZJU is manual).")
    ap.add_argument("--nyu", action="store_true")
    ap.add_argument("--kitti", action="store_true")
    ap.add_argument("--zju-info", action="store_true", help="Print ZJU-L5 download instructions.")
    args = ap.parse_args()

    if not any([args.all, args.nyu, args.kitti, args.zju_info]):
        ap.print_help()
        return
    if args.all or args.nyu:
        download_nyu()
    if args.all or args.kitti:
        download_kitti()
    if args.all or args.zju_info:
        print("\n" + ZJU_INFO)


if __name__ == "__main__":
    main()
