"""Experiment runner: load dataset, Depth Anything, calibration methods, metrics, figures."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from depth_estimation.calibration.methods import get_method
from depth_estimation.data.nyu_utils import load_nyu_mat, simulate_grid_stride_prior, simulate_sparse_prior
from depth_estimation.data.zju_l5 import (
    list_zju_l5_split,
    load_zju_l5_h5,
    load_zju_l5_l5_fields,
    sparse_depth_and_mask_from_l5,
)
from depth_estimation.evaluation.metrics import _METRIC_KEYS, aggregate_metrics, compute_metrics
from depth_estimation.evaluation.visualization import make_comparison_figure
from depth_estimation.models.da_inference import infer_depth, load_da_model


def _subsample_sparse_mask(
    sparse_depth: np.ndarray,
    sparse_mask: np.ndarray,
    density: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a random fraction of True entries in sparse_mask (at least one if any)."""
    ys, xs = np.where(sparse_mask)
    if ys.size == 0:
        return sparse_depth, sparse_mask
    d = float(density)
    if d >= 1.0:
        return sparse_depth, sparse_mask
    rng = np.random.default_rng(seed)
    n_keep = max(1, min(ys.size, int(round(ys.size * d))))
    pick = rng.choice(ys.size, size=n_keep, replace=False)
    out_d = np.zeros_like(sparse_depth)
    out_m = np.zeros_like(sparse_mask)
    out_d[ys[pick], xs[pick]] = sparse_depth[ys[pick], xs[pick]]
    out_m[ys[pick], xs[pick]] = True
    return out_d, out_m


@dataclass
class ExperimentConfig:
    name: str
    mat_path: str = "data/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    num_samples: int = 20
    sparse_density: float = 0.3
    seed: int = 42
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    out_dir: str = "./outputs/experiments"
    methods: dict[str, dict[str, Any]] = field(default_factory=dict)
    show_prediction_diff: bool = True
    dataset: str = "nyu"
    zju_l5_root: str | None = None
    zju_l5_manifest: str = "data.json"
    zju_l5_split: str = "test"
    prior_source: str = "simulate"
    frame_indices: list[int] | None = None
    frame_indices_file: str | None = None
    prior_noise: dict[str, Any] | None = None
    prior_grid_stride: int | None = None
    figure_suptitle: bool = True
    method_column_titles: dict[str, str] | None = None

    @classmethod
    def from_json(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("Experiment JSON must be an object.")
        methods = raw.get("methods") or {}
        if not isinstance(methods, dict):
            raise ValueError("'methods' must be an object.")
        return cls(
            name=str(raw.get("name", "experiment")),
            mat_path=str(raw.get("mat_path", "data/nyu_depth_v2/nyu_depth_v2_labeled.mat")),
            num_samples=int(raw.get("num_samples", 20)),
            sparse_density=float(raw.get("sparse_density", 0.3)),
            seed=int(raw.get("seed", 42)),
            model_id=str(raw.get("model_id", "depth-anything/Depth-Anything-V2-Small-hf")),
            out_dir=str(raw.get("out_dir", "./outputs/experiments")),
            methods={str(k): dict(v) for k, v in methods.items()},
            show_prediction_diff=bool(raw.get("show_prediction_diff", True)),
            dataset=str(raw.get("dataset", "nyu")),
            zju_l5_root=raw.get("zju_l5_root"),
            zju_l5_manifest=str(raw.get("zju_l5_manifest", "data.json")),
            zju_l5_split=str(raw.get("zju_l5_split", "test")),
            prior_source=str(raw.get("prior_source", "simulate")),
            frame_indices=raw.get("frame_indices"),
            frame_indices_file=raw.get("frame_indices_file"),
            prior_noise=raw.get("prior_noise"),
            prior_grid_stride=(
                int(raw["prior_grid_stride"]) if raw.get("prior_grid_stride") is not None else None
            ),
            figure_suptitle=bool(raw.get("figure_suptitle", True)),
            method_column_titles=(
                {str(k): str(v) for k, v in raw["method_column_titles"].items()}
                if isinstance(raw.get("method_column_titles"), dict)
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _resolve_indices(self, n_total: int) -> list[int]:
        c = self.config
        if c.frame_indices is not None:
            return [int(i) for i in c.frame_indices]
        if c.frame_indices_file:
            with open(c.frame_indices_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                return [int(i) for i in data]
            raise ValueError("frame_indices_file must contain a JSON list of integers.")
        n = min(int(c.num_samples), n_total)
        return list(range(n))

    def _build_sparse(
        self,
        gt_depth: np.ndarray,
        sample_index: int,
        *,
        hist_fr_mask: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        seed = int(c.seed) + int(sample_index)
        ps = (c.prior_source or "simulate").lower()

        if ps == "l5_zones" and hist_fr_mask is not None:
            hist, fr, zm = hist_fr_mask
            h, w = gt_depth.shape
            sparse, mask = sparse_depth_and_mask_from_l5(hist, fr, zm, h, w)
            return _subsample_sparse_mask(sparse, mask, c.sparse_density, seed)

        if ps == "grid_stride" and c.prior_grid_stride is not None:
            return simulate_grid_stride_prior(
                gt_depth, int(c.prior_grid_stride), seed, c.prior_noise,
            )

        return simulate_sparse_prior(gt_depth, c.sparse_density, seed, c.prior_noise)

    def run(self) -> dict[str, Any]:
        c = self.config
        t0 = time.perf_counter()

        out_root = os.path.join(c.out_dir, c.name)
        os.makedirs(out_root, exist_ok=True)

        processor, model, device, dtype = load_da_model(c.model_id, device=None)

        ds = (c.dataset or "nyu").lower()
        if ds == "zju_l5":
            if not c.zju_l5_root:
                raise ValueError("dataset=zju_l5 requires zju_l5_root in config.")
            paths = list_zju_l5_split(
                root=c.zju_l5_root,
                split=c.zju_l5_split,
                manifest_rel=c.zju_l5_manifest,
            )
            indices = self._resolve_indices(len(paths))

            def load_sample(i: int) -> tuple[np.ndarray, np.ndarray, str | None, tuple | None]:
                p = paths[i]
                rgb, gt = load_zju_l5_h5(p)
                extra = load_zju_l5_l5_fields(p)
                return rgb, gt, p, extra

        elif ds == "nyu":
            images, depths = load_nyu_mat(c.mat_path)
            indices = self._resolve_indices(images.shape[0])

            def load_sample(i: int) -> tuple[np.ndarray, np.ndarray, str | None, tuple | None]:
                gt_i = np.squeeze(np.asarray(depths[i], dtype=np.float32))
                return images[i], gt_i, None, None

        else:
            raise ValueError(f"Unknown dataset '{c.dataset}'. Use 'nyu' or 'zju_l5'.")

        per_sample: list[dict[str, Any]] = []
        metric_rows: dict[str, list[dict[str, float]]] = {k: [] for k in c.methods}

        for step, ix in enumerate(indices):
            rgb, gt, h5_path, l5_extra = load_sample(ix)
            d_rel = infer_depth(processor, model, rgb, device, dtype)

            hist_fr_mask = l5_extra if l5_extra is not None else None
            sparse_depth, sparse_mask = self._build_sparse(gt, step, hist_fr_mask=hist_fr_mask)

            predictions: dict[str, np.ndarray] = {}
            extras: dict[str, dict[str, Any]] = {}
            row_metrics: dict[str, dict[str, float]] = {}

            for method_name, spec in c.methods.items():
                mtype = spec.get("type")
                if not mtype:
                    raise ValueError(f"Method '{method_name}' missing 'type'.")
                params = {k: v for k, v in spec.items() if k != "type"}
                method = get_method(str(mtype), **params)
                pred, ex = method.calibrate(d_rel, sparse_depth, sparse_mask, rgb, sample_index=step)
                predictions[method_name] = pred
                extras[method_name] = ex
                m = compute_metrics(pred, gt)
                row_metrics[method_name] = m
                metric_rows[method_name].append(m)

            fig = make_comparison_figure(
                rgb,
                gt,
                predictions,
                extras,
                title=(f"{c.name}  sample={step}" if c.figure_suptitle else None),
                show_prediction_diff=c.show_prediction_diff,
                column_titles=c.method_column_titles,
            )
            fig_path = os.path.join(out_root, f"{c.name}_{step:04d}.png")
            fig.savefig(fig_path, dpi=140)
            plt.close(fig)

            rec: dict[str, Any] = {
                "index": int(ix),
                "step": int(step),
                "metrics": row_metrics,
            }
            if h5_path is not None:
                rec["h5_path"] = h5_path
            per_sample.append(rec)

        summary = {name: aggregate_metrics(rows) for name, rows in metric_rows.items()}
        elapsed = time.perf_counter() - t0

        results = {
            "config": c.to_dict(),
            "elapsed_seconds": float(elapsed),
            "per_sample": per_sample,
            "summary": summary,
        }
        results_path = os.path.join(out_root, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {results_path}")
        for mk, mv in summary.items():
            parts = "  ".join(f"{k}={mv.get(k, float('nan')):.4f}" for k in _METRIC_KEYS)
            print(f"  {mk}: {parts}")
        return results
