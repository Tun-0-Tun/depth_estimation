"""Learned relative->metric projection (amortized pretraining + per-image TTA).

Idea
----
Instead of a single global affine ``s*d_rel+t`` (one (s,t) per image) or a
piecewise-constant per-superpixel affine (which produces block seams), we predict
a **dense, smooth affine field** that maps normalized relative depth to metric:

    d_metric = m * ( base_n + a(x,y) * d_rel_n + b(x,y) )

where ``base_n`` is the global-affine baseline (normalized), ``d_rel_n`` is the
per-image affine-invariant relative depth, and ``a, b`` are smooth fields produced
by a small CNN from RGB + d_rel + sparse prior. High-frequency structure comes from
``d_rel`` (so edges stay sharp without blocks); the CNN only has to predict smooth
local corrections to where the global affine is wrong.

Everything the CNN sees is **scale-normalized** (unitless), so a model trained on a
mix of NYU (indoor, 0-10 m), KITTI (outdoor, 0-80 m) and ZJU-L5 transfers across
domains. The metric scale ``m`` is recovered per image from the sparse anchors.

Training (amortized): supervise the dense field against GT across the mixed dataset
plus an anchor loss at the sparse points, with edge-aware smoothness on (a, b).
Test time: copy the net and fine-tune a few steps on the sparse anchors only
(no GT), i.e. learn the projection on that single image, then predict densely.
"""

from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_estimation.calibration.global_baseline import (
    apply_global_calibration,
    fit_global_scale_shift,
)
from depth_estimation.calibration.inr_calibration import (
    _edge_aware_smoothness_loss,
    _refine_output_edge_aware,
)
from depth_estimation.models.da_inference import get_device


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

IN_CHANNELS = 7  # rgb(3) + d_rel_n(1) + base_n(1) + sparse_n(1) + mask(1)


class RelToMetricCNN(nn.Module):
    """Dilated residual CNN predicting a dense affine field (a, b) over d_rel."""

    def __init__(self, hidden: int = 64, num_layers: int = 10, in_ch: int = IN_CHANNELS):
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be >= 3")
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        ]
        for i in range(num_layers - 2):
            d = (1, 2, 4, 8)[i % 4]
            layers += [
                nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, 2, 3, padding=1)
        # zero-init: a=0, b=0 -> pred == global-affine baseline at start
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        ab = self.head(f)
        return ab[:, 0:1], ab[:, 1:2]


# ---------------------------------------------------------------------------
# Normalization / input assembly (shared by train + inference)
# ---------------------------------------------------------------------------


def _normalize_rel(d_rel: np.ndarray) -> np.ndarray:
    med = float(np.median(d_rel))
    scale = float(np.mean(np.abs(d_rel - med))) + 1e-6
    return ((d_rel - med) / scale).astype(np.float32)


def _robust_metric_scale(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    return float(max(np.percentile(values, 95), 1e-3))


class Rel2MetricInputs:
    """Holds tensors needed for a forward pass and for the losses, on ``device``."""

    def __init__(self, d_rel, sparse_depth, sparse_mask, rgb, device):
        d_rel = np.asarray(d_rel, dtype=np.float32)
        h, w = d_rel.shape
        mask = sparse_mask & (sparse_depth > 0) & np.isfinite(d_rel)
        self.has_anchors = bool(mask.any())

        if self.has_anchors:
            s_g, t_g = fit_global_scale_shift(d_rel, sparse_depth, valid_mask=mask)
            m = _robust_metric_scale(sparse_depth[mask])
        else:  # degenerate: no usable prior -> identity baseline
            s_g, t_g, m = 1.0, 0.0, _robust_metric_scale(d_rel[d_rel > 0])
        base = apply_global_calibration(d_rel, s_g, t_g).astype(np.float32)

        d_rel_n = _normalize_rel(d_rel)
        base_n = base / m
        sparse_n = np.where(mask, sparse_depth / m, 0.0).astype(np.float32)
        rgb_n = (np.asarray(rgb, dtype=np.float32) / 255.0).transpose(2, 0, 1)

        stack = np.concatenate(
            [
                rgb_n,
                d_rel_n[None],
                base_n[None].astype(np.float32),
                sparse_n[None],
                mask.astype(np.float32)[None],
            ],
            axis=0,
        )
        self.device = device
        self.h, self.w, self.m = h, w, float(m)
        self.s_g, self.t_g = float(s_g), float(t_g)
        self.inp = torch.from_numpy(stack[None]).to(device)
        self.d_rel_n = torch.from_numpy(d_rel_n[None, None]).to(device)
        self.base_n = torch.from_numpy(base_n[None, None].astype(np.float32)).to(device)
        self.sparse_n = torch.from_numpy(sparse_n[None, None]).to(device)
        self.mask_t = torch.from_numpy(mask.astype(np.float32)[None, None]).to(device)
        self.rgb_t = torch.from_numpy(rgb_n[None]).to(device)


def predict_pred_n(net: RelToMetricCNN, x: Rel2MetricInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (pred_n, a, b) in normalized depth units."""
    a, b = net(x.inp)
    pred_n = x.base_n + a * x.d_rel_n + b
    return pred_n, a, b


# ---------------------------------------------------------------------------
# Losses (used by the training script and by TTA)
# ---------------------------------------------------------------------------


def anchor_loss(pred_n: torch.Tensor, x: Rel2MetricInputs) -> torch.Tensor:
    denom = x.mask_t.sum().clamp(min=1.0)
    return ((pred_n - x.sparse_n).abs() * x.mask_t).sum() / denom


def field_smoothness(a: torch.Tensor, b: torch.Tensor, x: Rel2MetricInputs) -> torch.Tensor:
    return _edge_aware_smoothness_loss(a, x.rgb_t, 10.0) + _edge_aware_smoothness_loss(b, x.rgb_t, 10.0)


def supervised_loss(
    pred_n: torch.Tensor, gt: np.ndarray, x: Rel2MetricInputs
) -> torch.Tensor:
    gt_n = torch.from_numpy((np.asarray(gt, np.float32) / x.m)[None, None]).to(x.device)
    valid = torch.from_numpy(((gt > 0) & np.isfinite(gt)).astype(np.float32)[None, None]).to(x.device)
    denom = valid.sum().clamp(min=1.0)
    return ((pred_n - gt_n).abs() * valid).sum() / denom


def _trimmed_anchor_loss(pred_n: torch.Tensor, x: Rel2MetricInputs, trim_frac: float) -> torch.Tensor:
    """L1 over anchors, discarding the worst ``trim_frac`` residuals (robust to prior outliers).

    ``trim_frac == 0`` reduces to plain L1 anchor loss over all anchor pixels.
    """
    r = (pred_n - x.sparse_n).abs()[x.mask_t.bool()]
    if trim_frac and 0.0 < trim_frac < 1.0 and r.numel() > 10:
        keep = max(1, int(r.numel() * (1.0 - trim_frac)))
        r = torch.sort(r).values[:keep]
    return r.mean()


# ---------------------------------------------------------------------------
# Inference entry point (registered method calls this)
# ---------------------------------------------------------------------------


def calibrate_rel2metric_tta(
    d_rel: np.ndarray,
    sparse_depth: np.ndarray,
    sparse_mask: np.ndarray,
    rgb: np.ndarray,
    *,
    checkpoint: str,
    tta_steps: int = 150,
    tta_lr: float = 2e-3,
    tta_smooth_lambda: float = 0.05,
    tta_prox_lambda: float = 0.1,
    tta_trim_frac: float = 0.1,
    output_refine_mode: str = "guided",
    output_refine_radius: int = 8,
    output_refine_eps: float = 1e-3,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, dict]:
    device = device or get_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = ckpt.get("meta", {})
    net = RelToMetricCNN(
        hidden=int(meta.get("hidden", 64)),
        num_layers=int(meta.get("num_layers", 10)),
    ).to(device)
    net.load_state_dict(ckpt["state_dict"])

    x = Rel2MetricInputs(d_rel, sparse_depth, sparse_mask, rgb, device)

    # Amortized (pre-TTA) prediction: proximal anchor for the off-anchor regions,
    # so TTA corrects locally near anchors without drifting where the prior is trusted.
    net.eval()
    with torch.no_grad():
        prior_pred_n = predict_pred_n(net, x)[0].detach()
    anchor_before = float(anchor_loss(prior_pred_n, x)) if x.has_anchors else float("nan")

    # Per-image test-time adaptation on the sparse anchors only.
    did_tta = False
    if x.has_anchors and tta_steps > 0:
        net_tta = copy.deepcopy(net)
        opt = torch.optim.Adam(net_tta.parameters(), lr=tta_lr)
        net_tta.train()
        off_anchor = 1.0 - x.mask_t
        for _ in range(int(tta_steps)):
            opt.zero_grad(set_to_none=True)
            pred_n, a, b = predict_pred_n(net_tta, x)
            loss_anchor = _trimmed_anchor_loss(pred_n, x, tta_trim_frac)
            loss_smooth = field_smoothness(a, b, x)
            loss_prox = (((pred_n - prior_pred_n) ** 2) * off_anchor).mean()
            loss = loss_anchor + tta_smooth_lambda * loss_smooth + tta_prox_lambda * loss_prox
            loss.backward()
            opt.step()
        net = net_tta
        did_tta = True

    net.eval()
    with torch.no_grad():
        pred_n, _, _ = predict_pred_n(net, x)
        anchor_after = float(anchor_loss(pred_n, x)) if x.has_anchors else float("nan")
        pred = (pred_n.squeeze() * x.m).clamp(min=0.0).cpu().numpy().astype(np.float32)

    if (output_refine_mode or "none").lower() == "guided" and output_refine_radius > 0:
        pred = _refine_output_edge_aware(
            pred, rgb, radius=int(output_refine_radius), eps=float(output_refine_eps), device=device
        )

    return pred, {
        "inr_variant": "rel2metric_tta",
        "checkpoint": checkpoint,
        "tta_steps": int(tta_steps) if did_tta else 0,
        "metric_scale": x.m,
        "s_global": x.s_g,
        "t_global": x.t_g,
        "anchor_loss_before": anchor_before,
        "anchor_loss_after": anchor_after,
        "output_refine_mode": output_refine_mode,
    }
