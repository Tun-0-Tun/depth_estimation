"""
Registry of calibration methods for depth estimation experiments.

Each method is a dataclass with a ``calibrate()`` method that transforms
relative depth into metric depth using a sparse depth prior.

Usage::

    from depth_estimation.calibration.methods import get_method, list_methods

    method = get_method("inr_simple", train_steps=500)
    d_metric, extras = method.calibrate(d_rel, sparse_depth, sparse_mask, rgb)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator that registers a calibration method under *name*."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_method(type_name: str, **kwargs) -> "CalibrationMethodBase":
    """Instantiate a registered method by its type name, passing **kwargs to its constructor."""
    if type_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown method type '{type_name}'. Available: {available}")
    return _REGISTRY[type_name](**kwargs)


def list_methods() -> list[str]:
    """Return sorted list of registered method type names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class CalibrationMethodBase:
    """
    Base class for calibration methods.

    Subclasses must implement ``calibrate()``, which returns
    ``(d_metric, extras)`` where *extras* is a dict of additional outputs
    (e.g. superpixel labels, scale/shift maps) that may be used for
    visualization or analysis.
    """

    name: str = ""

    def calibrate(
        self,
        d_rel: np.ndarray,
        sparse_depth: np.ndarray,
        sparse_mask: np.ndarray,
        rgb: np.ndarray,
        sample_index: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Built-in methods
# ---------------------------------------------------------------------------

@register("global")
@dataclass
class GlobalCalibration(CalibrationMethodBase):
    """Global affine calibration: D_met = s * d_rel + t (single s, t per image)."""

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
    """
    Per-superpixel affine calibration with optional post-smoothing of (s, t).

    ``smooth_mode``:
      - ``none``: piecewise-constant s, t inside superpixels.
      - ``gaussian``: Gaussian blur on s, t (uniform smoothing).
      - ``bilateral``: joint bilateral on (s, t) — similar (s, t) blend across
        boundaries; large jumps stay sharp (edge-preserving between regions).

    Legacy: ``smooth=True`` forces ``gaussian`` when ``smooth_mode`` is ``none``.
    """

    name: str = "local"
    n_segments: int = 200
    sigma: float = 15.0
    smooth: bool = False
    smooth_mode: str = "none"
    min_pixels: int = 10
    # Bilateral (used when smooth_mode == "bilateral")
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
                f"Unknown smooth_mode '{self.smooth_mode}'. "
                "Use 'none', 'gaussian', or 'bilateral'."
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
    """
    Shared MLP on Fourier(u, v) + normalized d_rel; trained on sparse prior (L1).
    Optional residual on top of global affine s·d_rel+t.
    """

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
    """
    SLIC regions → CNN context; shared FiLM MLP per pixel on Fourier(u,v)+d_rel.
    Trained on sparse prior (L1); optional residual on global affine.
    """

    name: str = "inr_film"
    n_segments: int = 200
    crop_size: int = 32
    d_c: int = 64
    hidden_dim: int = 128
    num_film_layers: int = 4
    num_frequencies: int = 6
    train_steps: int = 1200
    lr: float = 1e-3
    affine_baseline: bool = True
    chunk_size: int = 65536
    train_seed: int = 42

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
        )
