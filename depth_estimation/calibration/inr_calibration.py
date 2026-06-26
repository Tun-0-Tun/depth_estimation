"""Per-image INR calibration (simple MLP, FiLM + region encoder, direct CNN)."""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as tvf
from torchvision.models import ResNet18_Weights

from depth_estimation.calibration.global_baseline import (
    apply_global_calibration,
    fit_global_scale_shift,
)
from depth_estimation.calibration.local_calibration import compute_superpixels
from depth_estimation.models.da_inference import get_device


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _fourier_features_2d(x: torch.Tensor, y: torch.Tensor, num_freqs: int) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for i in range(num_freqs):
        f = 2.0**i * math.pi
        out.extend([torch.sin(f * x), torch.cos(f * x), torch.sin(f * y), torch.cos(f * y)])
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


def _imagenet_norm_rgb(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


class ResNet18FrozenRegionContext(nn.Module):
    def __init__(self, d_c: int, backbone_image_size: int = 160):
        super().__init__()
        self.backbone_image_size = int(backbone_image_size)
        net = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        for p in net.parameters():
            p.requires_grad = False
        self.backbone = net
        self.adapter = nn.Sequential(nn.Linear(513, d_c), nn.ReLU(inplace=True))

    def forward(self, rgb_crops: torch.Tensor, d_mean: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            rgb_crops,
            size=(self.backbone_image_size, self.backbone_image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = _imagenet_norm_rgb(x)
        with torch.no_grad():
            f = self.backbone(x)
        h = torch.cat([f, d_mean.unsqueeze(1)], dim=1)
        return self.adapter(h)


class FilmINRNet(nn.Module):
    def __init__(self, d_pix: int, d_h: int, d_c: int, n_film_layers: int):
        super().__init__()
        self.lin = nn.ModuleList()
        self.gamma = nn.ModuleList()
        self.beta = nn.ModuleList()
        d_cur = d_pix
        for _ in range(n_film_layers):
            self.lin.append(nn.Linear(d_cur, d_h))
            g = nn.Linear(d_c, d_h)
            b = nn.Linear(d_c, d_h)
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)
            self.gamma.append(g)
            self.beta.append(b)
            d_cur = d_h
        self.head = nn.Linear(d_h, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_pix: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = x_pix
        for lin, lg, lb in zip(self.lin, self.gamma, self.beta, strict=True):
            z = lin(h)
            z = z * (1.0 + torch.tanh(lg(c))) + lb(c)
            h = F.relu(z)
        return self.head(h).squeeze(-1)


class DirectDepthCNN(nn.Module):
    """Small full-resolution residual CNN for direct depth prediction."""

    def __init__(
        self,
        in_ch: int,
        hidden: int,
        num_layers: int,
        with_edge_head: bool = True,
        edge_gain: float = 0.75,
        use_dilated_backbone: bool = False,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be >= 3")
        self.with_edge_head = bool(with_edge_head)
        self.edge_gain = float(edge_gain)
        self.use_dilated_backbone = bool(use_dilated_backbone)
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for i in range(num_layers - 2):
            dilation = (1, 2, 4)[i % 3] if self.use_dilated_backbone else 1
            layers.extend(
                [
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(inplace=True),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.depth_head = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        self.edge_head = nn.Conv2d(hidden, 1, kernel_size=3, padding=1) if self.with_edge_head else None

        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)
        if self.edge_head is not None:
            nn.init.zeros_(self.edge_head.weight)
            nn.init.zeros_(self.edge_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        f = self.backbone(x)
        delta = self.depth_head(f)
        edge_prob: torch.Tensor | None = None
        if self.edge_head is not None:
            edge_prob = torch.sigmoid(self.edge_head(f))
            delta = delta * (1.0 + self.edge_gain * edge_prob)
        return delta, edge_prob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_d_rel(d_rel: np.ndarray) -> np.ndarray:
    m = float(np.nanmax(np.abs(d_rel)) + 1e-8)
    return (d_rel / m).astype(np.float32)


def _pixel_features_uv_d(h: int, w: int, d_rel_np: np.ndarray, num_freqs: int, device: torch.device):
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
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    net.train()
    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        pred = forward_fn(feats)
        loss = F.l1_loss(pred, target)
        loss.backward()
        opt.step()


def _gaussian_blur_hw(ctx_hw: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return ctx_hw
    x = ctx_hw.permute(2, 0, 1).unsqueeze(0).contiguous()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    k = 2 * radius + 1
    if k % 2 == 0:
        k += 1
    blurred = tvf.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    return blurred.squeeze(0).permute(1, 2, 0)


def _box_filter_2d(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = 2 * radius + 1
    orig_shape = x.shape
    x4 = x.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
    pad = (radius, radius, radius, radius)
    xp = F.pad(x4, pad, mode="reflect")
    weight = x4.new_ones((1, 1, k, k)) / float(k * k)
    out = F.conv2d(xp, weight)
    return out.reshape(orig_shape)


def _guided_filter_hw(ctx_hw: torch.Tensor, guide_hw: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    if radius <= 0:
        return ctx_hw
    h, w, c = ctx_hw.shape
    I = guide_hw.unsqueeze(-1)
    P = ctx_hw
    mean_I = _box_filter_2d(I.permute(2, 0, 1), radius).permute(1, 2, 0)
    mean_P = _box_filter_2d(P.permute(2, 0, 1), radius).permute(1, 2, 0)
    mean_II = _box_filter_2d((I * I).permute(2, 0, 1), radius).permute(1, 2, 0)
    mean_IP = _box_filter_2d((I * P).permute(2, 0, 1), radius).permute(1, 2, 0)
    var_I = mean_II - mean_I * mean_I
    cov_IP = mean_IP - mean_I * mean_P
    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I
    mean_a = _box_filter_2d(a.permute(2, 0, 1), radius).permute(1, 2, 0)
    mean_b = _box_filter_2d(b.permute(2, 0, 1), radius).permute(1, 2, 0)
    return mean_a * I + mean_b


def _smooth_film_context_hw(
    ctx_hw: torch.Tensor,
    *,
    mode: str,
    sigma: float,
    guide_hw: torch.Tensor | None,
    radius: int,
    eps: float,
) -> torch.Tensor:
    m = (mode or "none").lower()
    if m == "none":
        return ctx_hw
    if m == "gaussian":
        return _gaussian_blur_hw(ctx_hw, sigma)
    if m == "guided":
        if guide_hw is None:
            raise ValueError("guided smoothing requires a guide image (H, W).")
        return _guided_filter_hw(ctx_hw, guide_hw, radius=radius, eps=eps)
    raise ValueError(f"Unknown film_context_blur_mode: {mode!r}")


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return (0.2989 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)


def _region_geom_features(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    n_reg = int(labels.max()) + 1
    cy = np.zeros(n_reg, dtype=np.float64)
    cx = np.zeros(n_reg, dtype=np.float64)
    area = np.zeros(n_reg, dtype=np.float64)
    flat = labels.reshape(-1)
    yy, xx = np.mgrid[0:h, 0:w]
    yy_f = yy.reshape(-1).astype(np.float64)
    xx_f = xx.reshape(-1).astype(np.float64)
    np.add.at(cy, flat, yy_f)
    np.add.at(cx, flat, xx_f)
    np.add.at(area, flat, 1.0)
    safe = np.maximum(area, 1.0)
    cy = cy / safe / max(h - 1, 1) * 2.0 - 1.0
    cx = cx / safe / max(w - 1, 1) * 2.0 - 1.0
    la = np.log(area + 1.0) / np.log(float(h * w) + 1.0)
    return np.stack([cy, cx, la], axis=1).astype(np.float32)


def _region_centroids_px(labels: np.ndarray, n_reg: int) -> np.ndarray:
    h, w = labels.shape
    cy = np.zeros(n_reg, dtype=np.float64)
    cx = np.zeros(n_reg, dtype=np.float64)
    cnt = np.zeros(n_reg, dtype=np.float64)
    flat = labels.reshape(-1)
    yy, xx = np.mgrid[0:h, 0:w]
    np.add.at(cy, flat, yy.ravel())
    np.add.at(cx, flat, xx.ravel())
    np.add.at(cnt, flat, 1.0)
    cnt = np.maximum(cnt, 1.0)
    return np.stack([cy / cnt, cx / cnt], axis=1).astype(np.float32)


def _build_region_crops_tensor(
    rgb: np.ndarray, d_rel: np.ndarray, labels: np.ndarray, n_reg: int, crop_size: int, device
) -> torch.Tensor:
    rgb_f = rgb.astype(np.float32) / 255.0
    d_n = _norm_d_rel(d_rel)
    crops: list[torch.Tensor] = []
    for k in range(n_reg):
        m = labels == k
        if not np.any(m):
            crops.append(torch.zeros(4, crop_size, crop_size, device=device))
            continue
        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        cr = rgb_f[y0 : y1 + 1, x0 : x1 + 1]
        cd = d_n[y0 : y1 + 1, x0 : x1 + 1]
        t = np.concatenate([cr, cd[..., None]], axis=-1)
        tt = torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(device)
        tt = F.interpolate(tt, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        crops.append(tt.squeeze(0))
    return torch.stack(crops, dim=0)


def _build_region_rgb_crops_and_dmean(
    rgb: np.ndarray, d_rel: np.ndarray, labels: np.ndarray, n_reg: int, crop_size: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_f = rgb.astype(np.float32) / 255.0
    d_n = _norm_d_rel(d_rel)
    crops: list[torch.Tensor] = []
    means: list[float] = []
    for k in range(n_reg):
        m = labels == k
        if not np.any(m):
            means.append(0.0)
            crops.append(torch.zeros(3, crop_size, crop_size, device=device))
            continue
        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        cr = rgb_f[y0 : y1 + 1, x0 : x1 + 1]
        dn = d_n[y0 : y1 + 1, x0 : x1 + 1]
        means.append(float(np.mean(dn)))
        tt = torch.from_numpy(cr).permute(2, 0, 1).unsqueeze(0).float().to(device)
        tt = F.interpolate(tt, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        crops.append(tt.squeeze(0))
    return torch.stack(crops, dim=0), torch.tensor(means, device=device, dtype=torch.float32)


def _soft_assign_ctx_hw(
    ctx_regions: torch.Tensor,
    centroids: torch.Tensor,
    height: int,
    width: int,
    k_soft: int,
    sigma_px: float,
    coord_chunk: int = 8192,
    *,
    assign_mode: str = "topk",
    radius_softmax_sigma_mult: float = 3.0,
) -> torch.Tensor:
    """
    Spatially blend per-region context vectors onto an (H, W, D) map.

    assign_mode:
      * ``topk`` — softmax over the ``k_soft`` nearest segment centroids (original).
      * ``radius_softmax`` — softmax over all centroids within
        ``(radius_softmax_sigma_mult * sigma_px)`` pixels; rows with no such centroid
        fall back to the same top-k rule as ``topk``.
    """
    device = ctx_regions.device
    R, D = ctx_regions.shape
    kk = min(int(k_soft), R)
    mode = (assign_mode or "topk").lower()
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
    out_list: list[torch.Tensor] = []
    cent = centroids.to(device=device, dtype=torch.float32)
    sig2 = float(sigma_px) ** 2 + 1e-8
    radius_sq = (float(radius_softmax_sigma_mult) * float(sigma_px)) ** 2 + 1e-8

    for s in range(0, coords.shape[0], coord_chunk):
        e = min(s + coord_chunk, coords.shape[0])
        cc = coords[s:e]
        d2 = torch.sum((cc[:, None, :] - cent[None, :, :]) ** 2, dim=2)

        if mode == "radius_softmax":
            logits = -d2 / sig2
            logits = logits.masked_fill(d2 > radius_sq, float("-inf"))
            bad = ~torch.isfinite(logits).any(dim=1)
            if bad.any():
                small_d2, idx = torch.topk(d2[bad], k=kk, dim=1, largest=False)
                lb = torch.full_like(d2[bad], float("-inf"))
                valid_logits = -small_d2 / sig2
                lb.scatter_(1, idx, valid_logits)
                logits = logits.clone()
                logits[bad] = lb
            w = torch.softmax(logits, dim=1)
            blended = w @ ctx_regions
        else:
            small_d2, idx = torch.topk(d2, k=kk, dim=1, largest=False)
            w = torch.softmax(-small_d2 / sig2, dim=1)
            gathered = ctx_regions[idx]
            blended = (gathered * w.unsqueeze(-1)).sum(dim=1)
        out_list.append(blended)
    return torch.cat(out_list, dim=0).view(height, width, D)


def _refine_output_edge_aware(
    depth_np: np.ndarray, rgb: np.ndarray, *, radius: int, eps: float, device
) -> np.ndarray:
    if radius <= 0:
        return depth_np
    guide_np = _rgb_to_luma(rgb)
    dep_t = torch.from_numpy(depth_np.astype(np.float32)).to(device).unsqueeze(-1)
    g_t = torch.from_numpy(guide_np).to(device)
    out = _guided_filter_hw(dep_t, g_t, radius=int(radius), eps=float(eps))
    return out.squeeze(-1).cpu().numpy().astype(np.float32)


def _edge_aware_smoothness_loss(
    depth_hw: torch.Tensor,
    rgb_hw: torch.Tensor,
    color_kappa: float,
    edge_prob_hw: torch.Tensor | None = None,
) -> torch.Tensor:
    dz_dx = (depth_hw[..., 1:] - depth_hw[..., :-1]).abs()
    dz_dy = (depth_hw[..., 1:, :] - depth_hw[..., :-1, :]).abs()
    dI_dx = (rgb_hw[..., 1:] - rgb_hw[..., :-1]).abs().mean(dim=1, keepdim=True)
    dI_dy = (rgb_hw[..., 1:, :] - rgb_hw[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    wx = torch.exp(-color_kappa * dI_dx)
    wy = torch.exp(-color_kappa * dI_dy)
    if edge_prob_hw is not None:
        ex = 0.5 * (edge_prob_hw[..., 1:] + edge_prob_hw[..., :-1])
        ey = 0.5 * (edge_prob_hw[..., 1:, :] + edge_prob_hw[..., :-1, :])
        wx = wx * (1.0 - ex).clamp(min=0.05)
        wy = wy * (1.0 - ey).clamp(min=0.05)
    return (dz_dx * wx).mean() + (dz_dy * wy).mean()


def _rgb_gradient_map(rgb_hw: torch.Tensor) -> torch.Tensor:
    dx = (rgb_hw[..., 1:] - rgb_hw[..., :-1]).abs().mean(dim=1, keepdim=True)
    dy = (rgb_hw[..., 1:, :] - rgb_hw[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    n, _, h, w = rgb_hw.shape
    gx = rgb_hw.new_zeros((n, 1, h, w))
    gy = rgb_hw.new_zeros((n, 1, h, w))
    gx[..., :-1] = dx
    gy[..., :-1, :] = dy
    g = (gx + gy).clamp(min=0.0)
    g_max = g.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return g / g_max


def _single_channel_gradient_map(x_hw: torch.Tensor) -> torch.Tensor:
    dx = (x_hw[..., 1:] - x_hw[..., :-1]).abs()
    dy = (x_hw[..., 1:, :] - x_hw[..., :-1, :]).abs()
    n, _, h, w = x_hw.shape
    gx = x_hw.new_zeros((n, 1, h, w))
    gy = x_hw.new_zeros((n, 1, h, w))
    gx[..., :-1] = dx
    gy[..., :-1, :] = dy
    g = (gx + gy).clamp(min=0.0)
    g_max = g.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return g / g_max


# ---------------------------------------------------------------------------
# Public calibration entry points
# ---------------------------------------------------------------------------


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
    device = get_device()
    torch.manual_seed(train_seed)
    h, w = d_rel.shape
    d_rel_np = d_rel.astype(np.float32)
    mask = sparse_mask & (sparse_depth > 0) & np.isfinite(d_rel_np)
    s_g, t_g = fit_global_scale_shift(d_rel_np, sparse_depth, valid_mask=sparse_mask)
    base_full = apply_global_calibration(d_rel_np, s_g, t_g)
    feats_all = _pixel_features_uv_d(h, w, d_rel_np, num_frequencies, device)
    ys, xs = np.where(mask)
    idx_flat = (ys * w + xs).astype(np.int64)
    feats_tr = feats_all[idx_flat]
    gt_tr = torch.from_numpy(sparse_depth[mask].astype(np.float32)).to(device)
    target = gt_tr - torch.from_numpy(base_full[mask].astype(np.float32)).to(device) if affine_baseline else gt_tr
    net = SimpleINRNet(feats_all.shape[1], hidden_dim, num_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    _train_loop(net, opt, feats_tr, target, train_steps, lambda x: net(x))
    net.eval()
    pred_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, feats_all.shape[0], chunk_size):
            e = min(s + chunk_size, feats_all.shape[0])
            pred_chunks.append(net(feats_all[s:e]))
    pred_delta_np = torch.cat(pred_chunks, dim=0).cpu().numpy().reshape(h, w).astype(np.float32)
    d_metric = base_full + pred_delta_np if affine_baseline else pred_delta_np
    return d_metric, {
        "inr_variant": "simple",
        "affine_baseline": affine_baseline,
        "s_global": s_g,
        "t_global": t_g,
        "train_steps": train_steps,
    }


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
    region_encoder: str = "cnn",
    backbone_image_size: int = 160,
    film_context_blur_mode: str = "guided",
    film_context_blur_sigma: float = 6.0,
    film_context_guide_radius: int = 8,
    film_context_guide_eps: float = 1e-2,
    film_context_soft_k: int = 6,
    film_context_soft_sigma_px: float = 30.0,
    film_context_soft_assign_mode: str = "topk",
    film_context_radius_softmax_sigma_mult: float = 3.0,
    film_context_post_gaussian_sigma: float = 0.0,
    slic_compactness: float = 10.0,
    use_region_geom: bool = True,
    tv_lambda: float = 0.0,
    tv_pairs_per_step: int = 4096,
    tv_color_kappa: float = 10.0,
    tv_on_metric_depth: bool = False,
    output_refine_mode: str = "none",
    output_refine_radius: int = 8,
    output_refine_eps: float = 1e-3,
) -> Tuple[np.ndarray, dict]:
    if region_encoder not in ("cnn", "resnet18_frozen"):
        raise ValueError(f"region_encoder must be 'cnn' or 'resnet18_frozen', got {region_encoder!r}")

    device = get_device()
    torch.manual_seed(train_seed)
    h, w = d_rel.shape
    d_rel_np = d_rel.astype(np.float32)
    labels = compute_superpixels(rgb, n_segments=n_segments, compactness=float(slic_compactness))
    n_reg = int(labels.max()) + 1

    s_g, t_g = fit_global_scale_shift(d_rel_np, sparse_depth, valid_mask=sparse_mask)
    base_full = apply_global_calibration(d_rel_np, s_g, t_g)

    label_t = torch.from_numpy(labels.astype(np.int64)).to(device)
    if region_encoder == "cnn":
        enc = RegionEncoder(d_c=d_c, in_ch=4).to(device)
        crops = _build_region_crops_tensor(rgb, d_rel_np, labels, n_reg, crop_size, device)
        region_mod: nn.Module = enc

        def forward_regions():
            return region_mod(crops)
    else:
        rgb_crops, d_mean = _build_region_rgb_crops_and_dmean(
            rgb, d_rel_np, labels, n_reg, crop_size, device,
        )
        region_mod = ResNet18FrozenRegionContext(d_c, backbone_image_size).to(device)

        def forward_regions():
            return region_mod(rgb_crops, d_mean)

    if use_region_geom:
        geom_np = _region_geom_features(labels)
        geom_t = torch.from_numpy(geom_np).to(device)
        d_c_total = d_c + geom_np.shape[1]
    else:
        geom_t = None
        d_c_total = d_c

    feats_all = _pixel_features_uv_d(h, w, d_rel_np, num_frequencies, device)
    d_pix = feats_all.shape[1]
    net = FilmINRNet(d_pix, hidden_dim, d_c_total, num_film_layers).to(device)

    if region_encoder == "cnn":
        params = list(net.parameters()) + list(region_mod.parameters())
    else:
        params = list(net.parameters()) + list(region_mod.adapter.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    mask = sparse_mask & (sparse_depth > 0) & np.isfinite(d_rel_np)
    ys, xs = np.where(mask)
    idx_flat = (ys * w + xs).astype(np.int64)
    feats_tr = feats_all[idx_flat]
    gt_tr = torch.from_numpy(sparse_depth[mask].astype(np.float32)).to(device)
    if affine_baseline:
        base_tr = torch.from_numpy(base_full[mask].astype(np.float32)).to(device)
        target = gt_tr - base_tr
    else:
        target = gt_tr

    mode_l = (film_context_blur_mode or "none").lower()
    guide_t = None
    if mode_l == "guided":
        guide_t = torch.from_numpy(_rgb_to_luma(rgb)).to(device)

    centroids_t = None
    if mode_l == "soft_assign":
        centroids_np = _region_centroids_px(labels, n_reg)
        centroids_t = torch.from_numpy(centroids_np).to(device)

    use_tv = tv_lambda > 0.0 and tv_pairs_per_step > 0
    rgb_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).to(device) if use_tv else None
    base_hw_t = (
        torch.from_numpy(base_full.astype(np.float32)).to(device)
        if (use_tv and tv_on_metric_depth)
        else None
    )

    def dense_ctx_map():
        raw = forward_regions()
        if geom_t is not None:
            raw = torch.cat([raw, geom_t], dim=1)
        if mode_l == "soft_assign":
            assert centroids_t is not None
            out_sa = _soft_assign_ctx_hw(
                raw,
                centroids_t,
                h,
                w,
                film_context_soft_k,
                film_context_soft_sigma_px,
                assign_mode=film_context_soft_assign_mode,
                radius_softmax_sigma_mult=film_context_radius_softmax_sigma_mult,
            )
            if film_context_post_gaussian_sigma > 0.0:
                out_sa = _gaussian_blur_hw(out_sa, float(film_context_post_gaussian_sigma))
            return out_sa
        ctx_hw = raw[label_t]
        return _smooth_film_context_hw(
            ctx_hw,
            mode=film_context_blur_mode,
            sigma=film_context_blur_sigma,
            guide_hw=guide_t,
            radius=film_context_guide_radius,
            eps=film_context_guide_eps,
        )

    if region_encoder == "cnn":
        region_mod.train()
    else:
        region_mod.backbone.eval()
        region_mod.adapter.train()
    net.train()

    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        ctx_hw = dense_ctx_map()
        c_b = ctx_hw[ys, xs]
        pred = net(feats_tr, c_b)
        loss = F.l1_loss(pred, target)
        if use_tv and rgb_t is not None:
            n_pairs = int(tv_pairs_per_step)
            yy_p = torch.randint(0, h - 1, (n_pairs,), device=device)
            xx_p = torch.randint(0, w - 1, (n_pairs,), device=device)
            ax = torch.randint(0, 2, (n_pairs,), device=device)
            yy_q = yy_p + ax
            xx_q = xx_p + (1 - ax)
            idx_p = yy_p * w + xx_p
            idx_q = yy_q * w + xx_q
            f_p = feats_all[idx_p]
            f_q = feats_all[idx_q]
            c_p = ctx_hw[yy_p, xx_p]
            c_q = ctx_hw[yy_q, xx_q]
            pred_p = net(f_p, c_p)
            pred_q = net(f_q, c_q)
            if base_hw_t is not None:
                dp = pred_p + base_hw_t[yy_p, xx_p]
                dq = pred_q + base_hw_t[yy_q, xx_q]
            else:
                dp, dq = pred_p, pred_q
            color_p = rgb_t[yy_p, xx_p]
            color_q = rgb_t[yy_q, xx_q]
            color_diff = (color_p - color_q).abs().mean(dim=-1)
            edge_w = torch.exp(-tv_color_kappa * color_diff)
            tv = (edge_w * (dp - dq).abs()).mean()
            loss = loss + tv_lambda * tv
        loss.backward()
        opt.step()

    region_mod.eval()
    net.eval()
    with torch.no_grad():
        ctx_hw = dense_ctx_map()
        c_full = ctx_hw.reshape(-1, d_c_total)
        pred_list = []
        for s in range(0, feats_all.shape[0], chunk_size):
            e = min(s + chunk_size, feats_all.shape[0])
            pred_list.append(net(feats_all[s:e], c_full[s:e]))
        pred_delta = torch.cat(pred_list, dim=0)
        pred_delta_np = pred_delta.cpu().numpy().reshape(h, w).astype(np.float32)

    d_metric = base_full + pred_delta_np if affine_baseline else pred_delta_np

    if (output_refine_mode or "none").lower() == "guided" and output_refine_radius > 0:
        d_metric = _refine_output_edge_aware(
            d_metric, rgb, radius=output_refine_radius, eps=output_refine_eps, device=device
        )

    extras = {
        "inr_variant": "film",
        "region_encoder": region_encoder,
        "labels": labels,
        "n_superpixels": n_reg,
        "affine_baseline": affine_baseline,
        "s_global": s_g,
        "t_global": t_g,
        "train_steps": train_steps,
        "film_context_blur_mode": film_context_blur_mode,
        "film_context_blur_sigma": film_context_blur_sigma,
        "film_context_guide_radius": film_context_guide_radius,
        "film_context_guide_eps": film_context_guide_eps,
        "use_region_geom": use_region_geom,
        "tv_lambda": tv_lambda,
        "d_c_total": d_c_total,
        "film_context_soft_k": film_context_soft_k,
        "film_context_soft_sigma_px": film_context_soft_sigma_px,
        "film_context_soft_assign_mode": film_context_soft_assign_mode,
        "film_context_radius_softmax_sigma_mult": film_context_radius_softmax_sigma_mult,
        "film_context_post_gaussian_sigma": film_context_post_gaussian_sigma,
        "slic_compactness": slic_compactness,
        "tv_on_metric_depth": tv_on_metric_depth,
        "output_refine_mode": output_refine_mode,
    }
    if region_encoder == "resnet18_frozen":
        extras["backbone_image_size"] = backbone_image_size
    return d_metric, extras


def calibrate_direct_depth_cnn(
    d_rel: np.ndarray,
    sparse_depth: np.ndarray,
    sparse_mask: np.ndarray,
    rgb: np.ndarray,
    hidden_dim: int = 64,
    num_layers: int = 8,
    train_steps: int = 800,
    lr: float = 3e-4,
    train_seed: int = 0,
    smooth_lambda: float = 0.08,
    smooth_color_kappa: float = 12.0,
    prior_lambda: float = 0.02,
    use_sparse_depth_channel: bool = True,
    with_edge_head: bool = True,
    edge_loss_lambda: float = 0.03,
    edge_gain: float = 0.75,
    use_dilated_backbone: bool = False,
    rel_edge_lambda: float = 0.0,
    output_refine_mode: str = "guided",
    output_refine_radius: int = 8,
    output_refine_eps: float = 1e-3,
) -> Tuple[np.ndarray, dict]:
    """Predict metric depth at full resolution from RGB + d_rel + sparse prior."""
    device = get_device()
    torch.manual_seed(train_seed)

    h, w = d_rel.shape
    d_rel_np = d_rel.astype(np.float32)
    mask_np = sparse_mask & (sparse_depth > 0) & np.isfinite(d_rel_np)
    if int(mask_np.sum()) == 0:
        s_g, t_g = fit_global_scale_shift(d_rel_np, sparse_depth, valid_mask=sparse_mask)
        return apply_global_calibration(d_rel_np, s_g, t_g), {
            "inr_variant": "direct_depth_cnn",
            "fallback": "global_affine_no_sparse",
            "s_global": float(s_g),
            "t_global": float(t_g),
        }

    s_g, t_g = fit_global_scale_shift(d_rel_np, sparse_depth, valid_mask=sparse_mask)
    base_full = apply_global_calibration(d_rel_np, s_g, t_g).astype(np.float32)

    depth_scale = float(np.percentile(sparse_depth[mask_np], 95))
    depth_scale = max(depth_scale, 1e-3)
    sparse_norm = sparse_depth.astype(np.float32) / depth_scale
    base_norm = (base_full / depth_scale).astype(np.float32)
    d_rel_norm = _norm_d_rel(d_rel_np)
    rgb_norm = rgb.astype(np.float32) / 255.0
    mask_f = mask_np.astype(np.float32)

    channels = [rgb_norm, d_rel_norm[..., None], base_norm[..., None]]
    if use_sparse_depth_channel:
        channels.append(sparse_norm[..., None])
    channels.append(mask_f[..., None])
    inp_np = np.concatenate(channels, axis=2)

    inp = torch.from_numpy(inp_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    rgb_t = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    sparse_t = torch.from_numpy(sparse_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    base_t = torch.from_numpy(base_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    net = DirectDepthCNN(
        in_ch=inp.shape[1],
        hidden=hidden_dim,
        num_layers=num_layers,
        with_edge_head=with_edge_head,
        edge_gain=edge_gain,
        use_dilated_backbone=use_dilated_backbone,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    rgb_edge_t = _rgb_gradient_map(rgb_t)
    rel_t = torch.from_numpy(d_rel_norm).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    rel_edge_t = _single_channel_gradient_map(rel_t)

    net.train()
    for _ in range(train_steps):
        opt.zero_grad(set_to_none=True)
        delta, edge_prob = net(inp)
        pred = base_t + delta

        sparse_l1 = (((pred - sparse_t).abs()) * mask_t).sum() / mask_t.sum().clamp(min=1.0)
        prior_l1 = (pred - base_t).abs().mean()
        smooth = _edge_aware_smoothness_loss(
            pred, rgb_t, color_kappa=smooth_color_kappa, edge_prob_hw=edge_prob,
        )

        loss = sparse_l1 + float(prior_lambda) * prior_l1 + float(smooth_lambda) * smooth
        if edge_prob is not None and edge_loss_lambda > 0.0:
            edge_l1 = F.l1_loss(edge_prob, rgb_edge_t)
            loss = loss + float(edge_loss_lambda) * edge_l1
        if rel_edge_lambda > 0.0:
            pred_edge = _single_channel_gradient_map(pred)
            loss = loss + float(rel_edge_lambda) * F.l1_loss(pred_edge, rel_edge_t)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        delta, _ = net(inp)
        pred_norm = (base_t + delta).squeeze(0).squeeze(0)
        pred_metric = (pred_norm * depth_scale).clamp(min=0.0).cpu().numpy().astype(np.float32)

    if (output_refine_mode or "none").lower() == "guided":
        pred_metric = _refine_output_edge_aware(
            pred_metric, rgb, radius=int(output_refine_radius), eps=float(output_refine_eps), device=device,
        )

    extras = {
        "inr_variant": "direct_depth_cnn",
        "s_global": float(s_g),
        "t_global": float(t_g),
        "depth_scale": float(depth_scale),
        "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers),
        "train_steps": int(train_steps),
        "smooth_lambda": float(smooth_lambda),
        "smooth_color_kappa": float(smooth_color_kappa),
        "prior_lambda": float(prior_lambda),
        "use_sparse_depth_channel": bool(use_sparse_depth_channel),
        "with_edge_head": bool(with_edge_head),
        "edge_loss_lambda": float(edge_loss_lambda),
        "edge_gain": float(edge_gain),
        "use_dilated_backbone": bool(use_dilated_backbone),
        "rel_edge_lambda": float(rel_edge_lambda),
        "output_refine_mode": output_refine_mode,
        "output_refine_radius": int(output_refine_radius),
        "output_refine_eps": float(output_refine_eps),
    }
    return pred_metric, extras
