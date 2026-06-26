#!/usr/bin/env python3
"""Amortized pretraining of the rel->metric projection on a mixed train set.

    python scripts/train_rel2metric.py --config configs/train_rel2metric_mixed.json

Trains :class:`RelToMetricCNN` on NYU + KITTI + ZJU-L5 with randomized sparse
priors, then saves a checkpoint that the ``rel2metric_tta`` calibration method
loads for per-image test-time adaptation at evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimation.calibration.rel2metric import (
    RelToMetricCNN,
    Rel2MetricInputs,
    anchor_loss,
    field_smoothness,
    predict_pred_n,
    supervised_loss,
)
from depth_estimation.data.unified import Sample, build_random_prior, build_samples
from depth_estimation.evaluation.metrics import aggregate_metrics, compute_metrics, format_metrics
from depth_estimation.models.da_inference import infer_depth, load_da_model, resolve_device


def _drel_cached(cache_dir: str | None, s: Sample, infer) -> np.ndarray:
    if not cache_dir:
        return infer()
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{s.source}_{s.key}.npy")
    if os.path.exists(path):
        return np.load(path)
    d = infer()
    np.save(path, d.astype(np.float32))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_rel2metric_mixed.json")
    args = ap.parse_args()
    with open(args.config) as f:
        c = json.load(f)

    device = resolve_device(c.get("device"))
    proc, da, device, dtype = load_da_model(c["model_id"], device=device)

    spec = c["datasets"]
    train = build_samples(spec, "train", max_per_dataset=c.get("max_per_dataset"), seed=c["seed"])
    val = build_samples(spec, "val", max_per_dataset=c.get("val_per_dataset", 40), seed=c["seed"])
    print(f"train samples: {len(train)}  val samples: {len(val)}")

    cache_dir = c.get("drel_cache_dir")
    noise = c.get("prior_noise")
    w_anchor = float(c.get("anchor_lambda", 1.0))
    w_smooth = float(c.get("smooth_lambda", 0.05))
    grad_accum = int(c.get("grad_accum", 4))

    net = RelToMetricCNN(hidden=int(c.get("hidden", 64)), num_layers=int(c.get("num_layers", 10))).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(c.get("lr", 3e-4)))
    meta = {"hidden": int(c.get("hidden", 64)), "num_layers": int(c.get("num_layers", 10))}

    def drel(s: Sample, rgb):
        return _drel_cached(cache_dir, s, lambda: infer_depth(proc, da, rgb, device, dtype))

    best_absrel = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    out_path = c["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for ep in range(int(c["epochs"])):
        t0 = time.perf_counter()
        rng = np.random.default_rng(c["seed"] + ep)
        order = rng.permutation(len(train))
        net.train()
        losses: list[float] = []
        opt.zero_grad(set_to_none=True)
        for step, si in enumerate(order):
            s = train[int(si)]
            rgb, gt, l5_extra = s.load()
            d_rel = drel(s, rgb)
            sparse, mask = build_random_prior(gt, rng, l5_extra=l5_extra, noise=noise)
            x = Rel2MetricInputs(d_rel, sparse, mask, rgb, device)
            if not x.has_anchors:
                continue
            pred_n, a, b = predict_pred_n(net, x)
            loss = (
                supervised_loss(pred_n, gt, x)
                + w_anchor * anchor_loss(pred_n, x)
                + w_smooth * field_smoothness(a, b, x)
            )
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        opt.step()
        opt.zero_grad(set_to_none=True)

        # validation (amortized, no TTA) -> measures the shared prior's quality
        net.eval()
        vrng = np.random.default_rng(c["seed"] + 10_000)
        rows = []
        with torch.no_grad():
            for s in val:
                rgb, gt, l5_extra = s.load()
                d_rel = drel(s, rgb)
                sparse, mask = build_random_prior(gt, vrng, l5_extra=l5_extra, noise=noise)
                x = Rel2MetricInputs(d_rel, sparse, mask, rgb, device)
                if not x.has_anchors:
                    continue
                pred_n, _, _ = predict_pred_n(net, x)
                pred = (pred_n.squeeze() * x.m).clamp(min=0.0).cpu().numpy().astype(np.float32)
                rows.append(compute_metrics(pred, gt))
        agg = aggregate_metrics(rows)
        dt = time.perf_counter() - t0
        print(
            f"epoch {ep + 1}/{c['epochs']}  trainL1={np.mean(losses) if losses else 0:.4f}  "
            f"val[{format_metrics(agg)}]  ({dt:.0f}s)"
        )
        if agg["abs_rel"] < best_absrel:
            best_absrel = agg["abs_rel"]
            best_state = copy.deepcopy(net.state_dict())
            torch.save({"state_dict": best_state, "meta": meta, "val": agg, "epoch": ep + 1}, out_path)
            print(f"  saved best -> {out_path} (abs_rel={best_absrel:.4f})")

    print(f"\nDone. Best val abs_rel={best_absrel:.4f}. Checkpoint: {out_path}")


if __name__ == "__main__":
    main()
