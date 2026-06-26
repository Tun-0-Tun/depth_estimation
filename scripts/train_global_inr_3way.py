#!/usr/bin/env python3
"""Train shared INR and compare two methods on test:
1) INR + test-time adaptation (few steps)
2) local_bilateral
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.calibration.global_baseline import apply_global_calibration, fit_global_scale_shift
from depth_estimation.calibration.inr_calibration import SimpleINRNet, _pixel_features_uv_d
from depth_estimation.calibration.methods import get_method
from depth_estimation.data.nyu_utils import load_nyu_mat, simulate_sparse_prior
from depth_estimation.evaluation.metrics import aggregate_metrics, compute_metrics, format_metrics
from depth_estimation.models.da_inference import get_device, infer_depth, load_da_model


def split_indices(n: int, tr: float, vr: float, seed: int):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    n_tr = max(1, int(round(n * tr)))
    n_v = max(1, int(round(n * vr)))
    if n_tr + n_v >= n:
        n_tr = max(1, n - 2)
        n_v = 1
    return p[:n_tr], p[n_tr:n_tr + n_v], p[n_tr + n_v:]


def ensure_rgb(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x, 0, 255).astype(np.uint8)


def build_sparse(gt: np.ndarray, density: float, seed: int):
    return simulate_sparse_prior(gt.astype(np.float32), density, seed)


def per_image_tensors(device, d_rel, sparse_depth, sparse_mask, num_freq):
    h, w = d_rel.shape
    mask = sparse_mask & (sparse_depth > 0) & np.isfinite(d_rel)
    s, t = fit_global_scale_shift(d_rel, sparse_depth, valid_mask=sparse_mask)
    base = apply_global_calibration(d_rel, s, t)
    feats_all = _pixel_features_uv_d(h, w, d_rel.astype(np.float32), num_freq, device)
    ys, xs = np.where(mask)
    idx = (ys * w + xs).astype(np.int64)
    feats_tr = feats_all[idx]
    gt_tr = torch.from_numpy(sparse_depth[mask].astype(np.float32)).to(device)
    base_tr = torch.from_numpy(base[mask].astype(np.float32)).to(device)
    return feats_all, feats_tr, gt_tr, base, base_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp_inr_3way_compare.json")
    args = ap.parse_args()

    with open(args.config) as f:
        c = json.load(f)

    images, depths = load_nyu_mat(c["mat_path"])
    n = min(int(c["num_samples"]), images.shape[0])
    tr_idx, va_idx, te_idx = split_indices(n, c["train_ratio"], c["val_ratio"], c["seed"])

    device = get_device()
    proc, da, device, dtype = load_da_model(c["model_id"], device=device)
    d_rel_cache = {}

    def drel(i):
        i = int(i)
        if i not in d_rel_cache:
            pil = Image.fromarray(ensure_rgb(images[i])).convert("RGB")
            d_rel_cache[i] = infer_depth(proc, da, pil, device, dtype)
        return d_rel_cache[i]

    inr_cfg = c["inr"]
    # infer feature dim
    ex = drel(int(tr_idx[0]))
    feats_dim = _pixel_features_uv_d(ex.shape[0], ex.shape[1], ex.astype(np.float32), inr_cfg["num_frequencies"], device).shape[1]
    net = SimpleINRNet(feats_dim, inr_cfg["hidden_dim"], inr_cfg["num_layers"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=inr_cfg["lr"])

    best_absrel = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    for ep in range(int(c["epochs"])):
        order = np.random.default_rng(c["seed"] + ep).permutation(tr_idx)
        losses = []
        net.train()
        for i in order:
            i = int(i)
            gt = depths[i].astype(np.float32)
            sparse_depth, sparse_mask = build_sparse(gt, c["sparse_density"], c["seed"] + i)
            feats_all, feats_tr, gt_tr, base, base_tr = per_image_tensors(
                device, drel(i), sparse_depth, sparse_mask, inr_cfg["num_frequencies"]
            )
            if feats_tr.numel() == 0:
                continue
            opt.zero_grad(set_to_none=True)
            pred = net(feats_tr)
            target = gt_tr - base_tr if inr_cfg.get("affine_baseline", True) else gt_tr
            loss = F.l1_loss(pred, target)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        # val
        net.eval()
        val_m = []
        with torch.no_grad():
            for i in va_idx:
                i = int(i)
                gt = depths[i].astype(np.float32)
                sparse_depth, sparse_mask = build_sparse(gt, c["sparse_density"], c["seed"] + i)
                feats_all, _, _, base, _ = per_image_tensors(
                    device, drel(i), sparse_depth, sparse_mask, inr_cfg["num_frequencies"]
                )
                pred = net(feats_all).cpu().numpy().reshape(gt.shape).astype(np.float32)
                if inr_cfg.get("affine_baseline", True):
                    pred = base + pred
                val_m.append(compute_metrics(pred, gt))
        agg = aggregate_metrics(val_m)
        print(f"Epoch {ep+1}/{c['epochs']} trainL1={np.mean(losses) if losses else 0:.5f}  val={format_metrics(agg)}")
        if agg["abs_rel"] < best_absrel:
            best_absrel = agg["abs_rel"]
            best_state = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)

    local_cfg = dict(c["local_bilateral"])
    local = get_method(local_cfg.pop("type", "local"), **local_cfg)
    tta_steps = int(c.get("tta_steps", 50))
    tta_lr = float(c.get("tta_lr", inr_cfg["lr"]))

    rec_tta, rec_loc = [], []
    for i in te_idx:
        i = int(i)
        rgb = ensure_rgb(images[i])
        gt = depths[i].astype(np.float32)
        sparse_depth, sparse_mask = build_sparse(gt, c["sparse_density"], c["seed"] + i)
        d = drel(i)
        feats_all, feats_tr, gt_tr, base, base_tr = per_image_tensors(
            device, d, sparse_depth, sparse_mask, inr_cfg["num_frequencies"]
        )

        # tta
        net_tta = copy.deepcopy(net)
        opt_tta = torch.optim.Adam(net_tta.parameters(), lr=tta_lr)
        net_tta.train()
        for _ in range(tta_steps):
            if feats_tr.numel() == 0:
                break
            opt_tta.zero_grad(set_to_none=True)
            pred = net_tta(feats_tr)
            target = gt_tr - base_tr if inr_cfg.get("affine_baseline", True) else gt_tr
            loss = F.l1_loss(pred, target)
            loss.backward()
            opt_tta.step()
        net_tta.eval()
        with torch.no_grad():
            p1 = net_tta(feats_all).cpu().numpy().reshape(gt.shape).astype(np.float32)
        if inr_cfg.get("affine_baseline", True):
            p1 = base + p1
        m1 = compute_metrics(p1, gt)
        rec_tta.append(m1)

        # local bilateral
        pl, _ = local.calibrate(d, sparse_depth, sparse_mask, rgb, sample_index=i)
        ml = compute_metrics(pl, gt)
        rec_loc.append(ml)

    out = {
        "config": c,
        "split": {"train": tr_idx.tolist(), "val": va_idx.tolist(), "test": te_idx.tolist()},
        "summary": {
            "inr_tta": aggregate_metrics(rec_tta),
            "local_bilateral": aggregate_metrics(rec_loc),
        },
    }
    os.makedirs(c["out_dir"], exist_ok=True)
    out_path = os.path.join(c["out_dir"], "results_3way.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== Test averages ===")
    print("inr_tta:        ", format_metrics(out["summary"]["inr_tta"]))
    print("local_bilateral:", format_metrics(out["summary"]["local_bilateral"]))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
