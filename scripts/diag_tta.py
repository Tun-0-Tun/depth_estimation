#!/usr/bin/env python3
"""Diagnose whether rel2metric TTA actually moves the prediction and helps.

Runs the ``rel2metric_tta`` method twice per image (with TTA and with
``tta_steps=0``) on a few samples and prints, per image:

  - anchor_loss before/after TTA  (is TTA reducing the anchor residual at all?)
  - abs_rel for no-TTA vs TTA      (does that translate into a metric gain?)

    python scripts/diag_tta.py --config configs/exp_nyu_rel2metric_eval_sparse.json --num 8

Reads the checkpoint, prior settings and TTA hyper-parameters straight from the
eval config, so it exercises the same setup as ``run_experiment.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.calibration.rel2metric import calibrate_rel2metric_tta
from depth_estimation.data.nyu_utils import load_nyu_mat, simulate_grid_stride_prior, simulate_sparse_prior
from depth_estimation.evaluation.metrics import compute_metrics
from depth_estimation.models.da_inference import infer_depth, load_da_model


def _resolve_indices(cfg: dict, n_total: int, num: int) -> list[int]:
    f = cfg.get("frame_indices_file")
    if f and os.path.exists(f):
        with open(f) as fh:
            idxs = [int(i) for i in json.load(fh)]
    elif cfg.get("frame_indices"):
        idxs = [int(i) for i in cfg["frame_indices"]]
    else:
        idxs = list(range(n_total))
    return idxs[:num]


def _build_prior(cfg: dict, gt: np.ndarray, seed: int):
    ps = (cfg.get("prior_source") or "simulate").lower()
    noise = cfg.get("prior_noise")
    if ps == "grid_stride" and cfg.get("prior_grid_stride") is not None:
        return simulate_grid_stride_prior(gt, int(cfg["prior_grid_stride"]), seed, noise)
    return simulate_sparse_prior(gt, float(cfg.get("sparse_density", 0.3)), seed, noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp_nyu_rel2metric_eval_sparse.json")
    ap.add_argument("--num", type=int, default=8)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    if (cfg.get("dataset") or "nyu").lower() != "nyu":
        raise SystemExit("diag_tta currently supports dataset=nyu only.")

    spec = dict(cfg["methods"]["rel2metric_tta"])
    spec.pop("type", None)
    tta_params = spec  # checkpoint + tta_* + output_refine_*

    proc, model, device, dtype = load_da_model(cfg["model_id"], device=None)
    images, depths = load_nyu_mat(cfg["mat_path"])
    seed0 = int(cfg.get("seed", 42))
    indices = _resolve_indices(cfg, images.shape[0], args.num)

    print(f"{'idx':>5} {'anchorL_before':>15} {'anchorL_after':>14} {'absrel_noTTA':>13} "
          f"{'absrel_TTA':>11} {'Δabsrel':>9}")
    d_no, d_tta = [], []
    for step, i in enumerate(indices):
        rgb = images[i]
        gt = np.squeeze(np.asarray(depths[i], dtype=np.float32))
        d_rel = infer_depth(proc, model, rgb, device, dtype)
        sparse, mask = _build_prior(cfg, gt, seed0 + step)

        pred_tta, info = calibrate_rel2metric_tta(d_rel, sparse, mask, rgb, device=device, **tta_params)
        no_params = {**tta_params, "tta_steps": 0}
        pred_no, _ = calibrate_rel2metric_tta(d_rel, sparse, mask, rgb, device=device, **no_params)

        ar_no = compute_metrics(pred_no, gt)["abs_rel"]
        ar_tta = compute_metrics(pred_tta, gt)["abs_rel"]
        d_no.append(ar_no)
        d_tta.append(ar_tta)
        print(f"{i:>5} {info['anchor_loss_before']:>15.5f} {info['anchor_loss_after']:>14.5f} "
              f"{ar_no:>13.5f} {ar_tta:>11.5f} {ar_tta - ar_no:>+9.5f}")

    m_no, m_tta = float(np.mean(d_no)), float(np.mean(d_tta))
    print(f"\nmean abs_rel  no_tta={m_no:.5f}  tta={m_tta:.5f}  "
          f"Δ={m_tta - m_no:+.5f}  ({'TTA helps' if m_tta < m_no else 'TTA does not help'})")


if __name__ == "__main__":
    main()
