"""
Per-image INR calibration on a sparse depth prior (GPU).

Two variants:
  * ``inr_simple`` — shared MLP on Fourier features of (u, v) + normalized d_rel.
  * ``inr_film`` — SLIC regions → small CNN context per region; shared MLP with
    FiLM conditioning; pixel input is Fourier(u,v) + d_rel.

Training: Adam on L1 at sparse prior pixels only. Inference: dense depth map.
Optional global affine baseline: prediction = s*d_rel + t + delta(INR).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_estimation.calibration.global_baseline import apply_global_calibration, fit_global_scale_shift
from depth_estimation.calibration.local_calibration import compute_superpixels
from depth_estimation.models.da_inference import get_device


def _fourier_features_2d(x: torch.Tensor, y: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """x, y: (N,) in [-1, 1]. Returns (N, 4 * num_freqs)."""
    out = []
    for i in range(num_freqs):
        f = (2.0**i) * math.pi
        out.extend(
            [
                torch.sin(f * x),
                torch.cos(f * x),
                torch.sin(f * y),
                torch.cos(f * y),
            ]
        )
    return torch.stack(out, dim=-1)


class SimpleINRNet(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            d = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RegionEncoder(nn.Module):
    """Lightweight CNN on RGB + d_rel crop → context vector."""

    def __init__(self, d_c: int = 64, in_ch: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, d_c, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class FilmINRNet(nn.Module):
    """Stack of Linear + FiLM(c) + ReLU, then scalar output."""

    def __init__(self, d_pix: int, d_h: int, d_c: int, n_film_layers: int):
        super().__init__()
        self.lin = nn.ModuleList()
        self.gamma = nn.ModuleList()
        self.beta = nn.ModuleList()
        d_cur = d_pix
        for _ in range(n_film_layers):
            self.lin.append(nn.Linear(d_cur, d_h))
            self.gamma.append(nn.Linear(d_c, d_h))
            self.beta.append(nn.Linear(d_c, d_h))
            d_cur = d_h
        self.head = nn.Linear(d_h, 1)

    def forward(self, x_pix: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = x_pix
        for lin, lg, lb in zip(self.lin, self.gamma, self.beta, strict=True):
            z = lin(h)
            z = z * (1.0 + torch.tanh(lg(c))) + lb(c)
            h = F.relu(z)
        return self.head(h).squeeze(-1)


def _norm_d_rel(d_rel: np.ndarray) -> np.ndarray:
    m = float(np.nanmax(np.abs(d_rel)) + 1e-8)
    return (d_rel / m).astype(np.float32)


def _pixel_features_uv_d(
    h: int,
    w: int,
    d_rel_np: np.ndarray,
    num_freqs: int,
    device: torch.device,
) -> torch.Tensor:
    """Full image flattened: Fourier(u,v) + d_norm → (H*W, d_pix)."""
    yy = torch.linspace(-1, 1, h, device=device)
    xx = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    u = gy.reshape(-1)
    v = gx.reshape(-1)
    ff = _fourier_features_2d(u, v, num_freqs)
    d_n = torch.from_numpy(_norm_d_rel(d_rel_np).reshape(-1)).to(device=device, dtype=torch.float32)
    return torch.cat([ff, d_n.unsqueeze(-1)], dim=-1)


def _train_loop(
    net: nn.Module,
    opt: torch.optim.Optimizer,
    feats: torch.Tensor,
    target: torch.Tensor,
    train_steps: int,
    forward_fn,
) -> None:
    net.train()
    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        pred = forward_fn(feats)
        loss = F.l1_loss(pred, target)
        loss.backward()
        opt.step()


def _predict_dense_chunked(
    net: nn.Module,
    feats_full: torch.Tensor,
    forward_fn,
    chunk: int,
    device: torch.device,
) -> torch.Tensor:
    net.eval()
    out = []
    n = feats_full.shape[0]
    with torch.no_grad():
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            x = feats_full[s:e]
            out.append(forward_fn(x))
    return torch.cat(out, dim=0)


def calibrate_inr_simple(
    d_rel: np.ndarray,
    sparse_depth: np.ndarray,
    sparse_mask: np.ndarray,
    rgb: np.ndarray,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_frequencies: int = 6,
    train_steps: int = 800,
    lr: float = 1e-3,
    affine_baseline: bool = True,
    chunk_size: int = 65536,
    train_seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    Train MLP: features = RFF(u,v) + d_rel_norm → scalar (residual or full depth).
    """
    device = get_device()
    torch.manual_seed(train_seed)
    h, w = d_rel.shape
    d_rel_np = d_rel.astype(np.float32)
    mask = sparse_mask & (sparse_depth > 0) & np.isfinite(d_rel_np)

    s_g, t_g = fit_global_scale_shift(d_rel_np, sparse_depth, valid_mask=sparse_mask)
    base_full = apply_global_calibration(d_rel_np, s_g, t_g)

    feats_all = _pixel_features_uv_d(h, w, d_rel_np, num_frequencies, device)
    d_pix = feats_all.shape[1]

    ys, xs = np.where(mask)
    idx = (ys * w + xs).astype(np.int64)
    feats_tr = feats_all[idx]
    gt_tr = torch.from_numpy(sparse_depth[mask].astype(np.float32)).to(device)

    if affine_baseline:
        base_tr = torch.from_numpy(base_full[mask].astype(np.float32)).to(device)
        target = gt_tr - base_tr
    else:
        target = gt_tr

    net = SimpleINRNet(d_pix, hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def fwd(x):
        return net(x)

    _train_loop(net, opt, feats_tr, target, train_steps, fwd)

    pred_delta = _predict_dense_chunked(net, feats_all, fwd, chunk_size, device)
    pred_delta_np = pred_delta.cpu().numpy().reshape(h, w).astype(np.float32)

    if affine_baseline:
        d_metric = base_full + pred_delta_np
    else:
        d_metric = pred_delta_np

    extras = {
        "inr_variant": "simple",
        "affine_baseline": affine_baseline,
        "s_global": s_g,
        "t_global": t_g,
        "train_steps": train_steps,
    }
    return d_metric, extras


def _build_region_crops_tensor(
    rgb: np.ndarray,
    d_rel: np.ndarray,
    labels: np.ndarray,
    n_reg: int,
    crop_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Tensor (n_reg, 4, S, S) RGB [0,1] + d_norm."""
    rgb_f = rgb.astype(np.float32) / 255.0
    d_n = _norm_d_rel(d_rel)
    H, W = d_rel.shape
    crops = []
    for k in range(n_reg):
        m = labels == k
        if not np.any(m):
            crops.append(torch.zeros(4, crop_size, crop_size, device=device))
            continue
        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        cr = rgb_f[y0 : y1 + 1, x0 : x1 + 1]
        cd = d_n[y0 : y1 + 1, x0 : x1 + 1]
        # (H',W',3) + (H',W',1)
        t = np.concatenate([cr, cd[..., None]], axis=-1)
        tt = torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(device)
        tt = F.interpolate(tt, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        crops.append(tt.squeeze(0))
    return torch.stack(crops, dim=0)


def calibrate_inr_film(
    d_rel: np.ndarray,
    sparse_depth: np.ndarray,
    sparse_mask: np.ndarray,
    rgb: np.ndarray,
    n_segments: int = 200,
    crop_size: int = 32,
    d_c: int = 64,
    hidden_dim: int = 128,
    num_film_layers: int = 4,
    num_frequencies: int = 6,
    train_steps: int = 1200,
    lr: float = 1e-3,
    affine_baseline: bool = True,
    chunk_size: int = 65536,
    train_seed: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    Region CNN context + FiLM MLP per pixel; train on sparse prior only.
    """
    device = get_device()
    torch.manual_seed(train_seed)
    h, w = d_rel.shape
    d_rel_np = d_rel.astype(np.float32)
    labels = compute_superpixels(rgb, n_segments=n_segments)
    n_reg = int(labels.max()) + 1

    s_g, t_g = fit_global_scale_shift(d_rel_np, sparse_depth, valid_mask=sparse_mask)
    base_full = apply_global_calibration(d_rel_np, s_g, t_g)

    enc = RegionEncoder(d_c=d_c, in_ch=4).to(device)
    label_t = torch.from_numpy(labels.astype(np.int64)).to(device)

    crops = _build_region_crops_tensor(rgb, d_rel_np, labels, n_reg, crop_size, device)

    feats_all = _pixel_features_uv_d(h, w, d_rel_np, num_frequencies, device)
    d_pix = feats_all.shape[1]

    net = FilmINRNet(d_pix, hidden_dim, d_c, num_film_layers).to(device)
    params = list(net.parameters()) + list(enc.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    mask = sparse_mask & (sparse_depth > 0) & np.isfinite(d_rel_np)
    ys, xs = np.where(mask)
    idx_flat = ys * w + xs

    feats_tr = feats_all[idx_flat]
    rid_tr = label_t[ys, xs]
    gt_tr = torch.from_numpy(sparse_depth[mask].astype(np.float32)).to(device)

    if affine_baseline:
        base_tr = torch.from_numpy(base_full[mask].astype(np.float32)).to(device)
        target = gt_tr - base_tr
    else:
        target = gt_tr

    net.train()
    enc.train()
    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        # refresh context from current encoder
        ctx_all = enc(crops)
        c_b = ctx_all[rid_tr]
        pred = net(feats_tr, c_b)
        loss = F.l1_loss(pred, target)
        loss.backward()
        opt.step()

    enc.eval()
    net.eval()
    with torch.no_grad():
        ctx_all = enc(crops)

    rid_flat = label_t.reshape(-1)
    c_full = ctx_all[rid_flat]

    pred_list = []
    n_pix = feats_all.shape[0]
    with torch.no_grad():
        for s in range(0, n_pix, chunk_size):
            e = min(s + chunk_size, n_pix)
            xb = feats_all[s:e]
            cb = c_full[s:e]
            pred_list.append(net(xb, cb))
    pred_delta = torch.cat(pred_list, dim=0)
    pred_delta_np = pred_delta.cpu().numpy().reshape(h, w).astype(np.float32)

    if affine_baseline:
        d_metric = base_full + pred_delta_np
    else:
        d_metric = pred_delta_np

    extras = {
        "inr_variant": "film",
        "labels": labels,
        "n_superpixels": n_reg,
        "affine_baseline": affine_baseline,
        "s_global": s_g,
        "t_global": t_g,
        "train_steps": train_steps,
    }
    return d_metric, extras
