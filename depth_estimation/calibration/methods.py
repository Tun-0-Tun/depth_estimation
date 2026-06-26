"""Registry of calibration methods (global / local / INR / direct CNN)."""

from __future__ import annotations

from dataclasses import dataclass

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_method(type_name: str, **kwargs):
    if type_name not in _REGISTRY:
        raise ValueError(
            f"Unknown method type '{type_name}'. Available: {', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[type_name](**kwargs)


def list_methods() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class CalibrationMethodBase:
    name: str = ""

    def calibrate(self, d_rel, sparse_depth, sparse_mask, rgb, sample_index=0):
        raise NotImplementedError


@register("global")
@dataclass
class GlobalCalibration(CalibrationMethodBase):
    name: str = "global"

    def calibrate(self, d_rel, sparse_depth, sparse_mask, rgb, sample_index: int = 0):
        from depth_estimation.calibration.global_baseline import (
            apply_global_calibration,
            fit_global_scale_shift,
        )

        s, t = fit_global_scale_shift(d_rel, sparse_depth, valid_mask=sparse_mask)
        d_metric = apply_global_calibration(d_rel, s, t)
        return d_metric, {"s": s, "t": t}


@register("local")
@dataclass
class LocalCalibration(CalibrationMethodBase):
    name: str = "local"
    n_segments: int = 200
    sigma: float = 15.0
    smooth: bool = False
    smooth_mode: str = "none"
    min_pixels: int = 10
    sigma_spatial: float = 5.0
    sigma_range_s: float | None = None
    sigma_range_t: float | None = None
    range_scale: float = 0.25
    bilateral_max_radius: int = 10

    def calibrate(self, d_rel, sparse_depth, sparse_mask, rgb, sample_index: int = 0):
        from depth_estimation.calibration.global_baseline import fit_global_scale_shift
        from depth_estimation.calibration.local_calibration import (
            apply_local_calibration,
            compute_superpixels,
            fit_per_superpixel,
            smooth_fields,
            smooth_fields_bilateral,
        )

        fallback_s, fallback_t = fit_global_scale_shift(d_rel, sparse_depth, sparse_mask)
        labels = compute_superpixels(rgb, n_segments=self.n_segments)
        s_map, t_map = fit_per_superpixel(
            d_rel,
            sparse_depth,
            labels,
            valid_mask=sparse_mask,
            fallback_s=fallback_s,
            fallback_t=fallback_t,
            min_pixels=self.min_pixels,
        )
        mode = (self.smooth_mode or "none").lower()
        if self.smooth and mode == "none":
            mode = "gaussian"
        if mode == "gaussian":
            s_map, t_map = smooth_fields(s_map, t_map, sigma=self.sigma)
        elif mode == "bilateral":
            s_map, t_map = smooth_fields_bilateral(
                s_map,
                t_map,
                sigma_spatial=self.sigma_spatial,
                sigma_range_s=self.sigma_range_s,
                sigma_range_t=self.sigma_range_t,
                range_scale=self.range_scale,
                max_radius=self.bilateral_max_radius,
            )
        elif mode != "none":
            raise ValueError(
                f"Unknown smooth_mode '{self.smooth_mode}'. Use 'none', 'gaussian', or 'bilateral'."
            )

        d_metric = apply_local_calibration(d_rel, s_map, t_map)
        n_sp = int(labels.max()) + 1
        return d_metric, {
            "labels": labels,
            "s_map": s_map,
            "t_map": t_map,
            "n_superpixels": n_sp,
            "smooth_mode": mode,
        }


@register("inr_simple")
@dataclass
class INRSimpleCalibration(CalibrationMethodBase):
    name: str = "inr_simple"
    hidden_dim: int = 128
    num_layers: int = 4
    num_frequencies: int = 6
    train_steps: int = 800
    lr: float = 1e-3
    affine_baseline: bool = True
    chunk_size: int = 65536
    train_seed: int = 42

    def calibrate(self, d_rel, sparse_depth, sparse_mask, rgb, sample_index: int = 0):
        from depth_estimation.calibration.inr_calibration import calibrate_inr_simple

        return calibrate_inr_simple(
            d_rel,
            sparse_depth,
            sparse_mask,
            rgb,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_frequencies=self.num_frequencies,
            train_steps=self.train_steps,
            lr=self.lr,
            affine_baseline=self.affine_baseline,
            chunk_size=self.chunk_size,
            train_seed=self.train_seed + int(sample_index),
        )


@register("inr_film")
@dataclass
class INRFilmCalibration(CalibrationMethodBase):
    name: str = "inr_film"
    n_segments: int = 200
    crop_size: int = 32
    region_encoder: str = "cnn"
    backbone_image_size: int = 160
    d_c: int = 64
    hidden_dim: int = 128
    num_film_layers: int = 4
    num_frequencies: int = 6
    train_steps: int = 1200
    lr: float = 1e-3
    affine_baseline: bool = True
    chunk_size: int = 65536
    train_seed: int = 42
    film_context_blur_mode: str = "guided"
    film_context_blur_sigma: float = 6.0
    film_context_guide_radius: int = 8
    film_context_guide_eps: float = 1e-2
    film_context_soft_k: int = 6
    film_context_soft_sigma_px: float = 30.0
    film_context_soft_assign_mode: str = "topk"
    film_context_radius_softmax_sigma_mult: float = 3.0
    film_context_post_gaussian_sigma: float = 0.0
    slic_compactness: float = 10.0
    use_region_geom: bool = True
    tv_lambda: float = 0.0
    tv_pairs_per_step: int = 8192
    tv_color_kappa: float = 10.0
    tv_on_metric_depth: bool = False
    output_refine_mode: str = "none"
    output_refine_radius: int = 8
    output_refine_eps: float = 1e-3

    def calibrate(self, d_rel, sparse_depth, sparse_mask, rgb, sample_index: int = 0):
        from depth_estimation.calibration.inr_calibration import calibrate_inr_film

        return calibrate_inr_film(
            d_rel,
            sparse_depth,
            sparse_mask,
            rgb,
            n_segments=self.n_segments,
            crop_size=self.crop_size,
            d_c=self.d_c,
            hidden_dim=self.hidden_dim,
            num_film_layers=self.num_film_layers,
            num_frequencies=self.num_frequencies,
            train_steps=self.train_steps,
            lr=self.lr,
            affine_baseline=self.affine_baseline,
            chunk_size=self.chunk_size,
            train_seed=self.train_seed + int(sample_index),
            region_encoder=self.region_encoder,
            backbone_image_size=self.backbone_image_size,
            film_context_blur_mode=self.film_context_blur_mode,
            film_context_blur_sigma=self.film_context_blur_sigma,
            film_context_guide_radius=self.film_context_guide_radius,
            film_context_guide_eps=self.film_context_guide_eps,
            film_context_soft_k=self.film_context_soft_k,
            film_context_soft_sigma_px=self.film_context_soft_sigma_px,
            film_context_soft_assign_mode=self.film_context_soft_assign_mode,
            film_context_radius_softmax_sigma_mult=self.film_context_radius_softmax_sigma_mult,
            film_context_post_gaussian_sigma=self.film_context_post_gaussian_sigma,
            slic_compactness=self.slic_compactness,
            use_region_geom=self.use_region_geom,
            tv_lambda=self.tv_lambda,
            tv_pairs_per_step=self.tv_pairs_per_step,
            tv_color_kappa=self.tv_color_kappa,
            tv_on_metric_depth=self.tv_on_metric_depth,
            output_refine_mode=self.output_refine_mode,
            output_refine_radius=self.output_refine_radius,
            output_refine_eps=self.output_refine_eps,
        )


@register("rel2metric_tta")
@dataclass
class Rel2MetricTTACalibration(CalibrationMethodBase):
    """Amortized dense rel->metric affine field + per-image test-time adaptation.

    Loads a checkpoint pretrained on the mixed train set, then fine-tunes a few
    steps on this image's sparse anchors before predicting dense metric depth.
    """

    name: str = "rel2metric_tta"
    checkpoint: str = "outputs/rel2metric/rel2metric_mixed.pt"
    tta_steps: int = 40
    tta_lr: float = 1e-4
    tta_smooth_lambda: float = 0.05
    output_refine_mode: str = "guided"
    output_refine_radius: int = 8
    output_refine_eps: float = 1e-3

    def calibrate(self, d_rel, sparse_depth, sparse_mask, rgb, sample_index: int = 0):
        from depth_estimation.calibration.rel2metric import calibrate_rel2metric_tta

        return calibrate_rel2metric_tta(
            d_rel,
            sparse_depth,
            sparse_mask,
            rgb,
            checkpoint=self.checkpoint,
            tta_steps=self.tta_steps,
            tta_lr=self.tta_lr,
            tta_smooth_lambda=self.tta_smooth_lambda,
            output_refine_mode=self.output_refine_mode,
            output_refine_radius=self.output_refine_radius,
            output_refine_eps=self.output_refine_eps,
        )


@register("direct_depth_cnn")
@dataclass
class DirectDepthCNNCalibration(CalibrationMethodBase):
    """Full-resolution CNN that predicts metric depth directly (no SLIC blocks)."""

    name: str = "direct_depth_cnn"
    hidden_dim: int = 64
    num_layers: int = 8
    train_steps: int = 800
    lr: float = 3e-4
    train_seed: int = 42
    smooth_lambda: float = 0.08
    smooth_color_kappa: float = 12.0
    prior_lambda: float = 0.02
    use_sparse_depth_channel: bool = True
    with_edge_head: bool = True
    edge_loss_lambda: float = 0.03
    edge_gain: float = 0.75
    use_dilated_backbone: bool = False
    rel_edge_lambda: float = 0.0
    output_refine_mode: str = "guided"
    output_refine_radius: int = 8
    output_refine_eps: float = 1e-3

    def calibrate(self, d_rel, sparse_depth, sparse_mask, rgb, sample_index: int = 0):
        from depth_estimation.calibration.inr_calibration import calibrate_direct_depth_cnn

        return calibrate_direct_depth_cnn(
            d_rel,
            sparse_depth,
            sparse_mask,
            rgb,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            train_steps=self.train_steps,
            lr=self.lr,
            train_seed=self.train_seed + int(sample_index),
            smooth_lambda=self.smooth_lambda,
            smooth_color_kappa=self.smooth_color_kappa,
            prior_lambda=self.prior_lambda,
            use_sparse_depth_channel=self.use_sparse_depth_channel,
            with_edge_head=self.with_edge_head,
            edge_loss_lambda=self.edge_loss_lambda,
            edge_gain=self.edge_gain,
            use_dilated_backbone=self.use_dilated_backbone,
            rel_edge_lambda=self.rel_edge_lambda,
            output_refine_mode=self.output_refine_mode,
            output_refine_radius=self.output_refine_radius,
            output_refine_eps=self.output_refine_eps,
        )
