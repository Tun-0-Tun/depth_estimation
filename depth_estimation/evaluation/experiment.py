"""
Unified experiment runner.

Orchestrates: data loading -> DA inference -> calibration methods -> metrics -> saving.
DA inference is performed once per sample and shared across all methods.

Results are saved as a structured JSON for later comparison.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from depth_estimation.calibration.methods import get_method
from depth_estimation.data.nyu_utils import load_nyu_mat, simulate_sparse_prior
from depth_estimation.evaluation.metrics import (
    _METRIC_KEYS,
    aggregate_metrics,
    compute_metrics,
    format_metrics,
)
from depth_estimation.evaluation.visualization import make_comparison_figure
from depth_estimation.models.da_inference import infer_depth, load_da_model


@dataclass
class ExperimentConfig:
    """All parameters needed to define a reproducible experiment."""

    name: str = "experiment"
    mat_path: str = "data/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    num_samples: int = 20
    sparse_density: float = 0.3
    seed: int = 42
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    out_dir: str = "./outputs/experiments"
    methods: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "global": {"type": "global"},
            "local": {"type": "local"},
        }
    )
    show_prediction_diff: bool = True

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mat_path": self.mat_path,
            "num_samples": self.num_samples,
            "sparse_density": self.sparse_density,
            "seed": self.seed,
            "model_id": self.model_id,
            "out_dir": self.out_dir,
            "methods": self.methods,
            "show_prediction_diff": self.show_prediction_diff,
        }


class ExperimentRunner:
    """
    Run calibration methods on NYU Depth V2 samples and collect metrics.

    Typical usage::

        config = ExperimentConfig.from_json("configs/baseline.json")
        runner = ExperimentRunner(config)
        runner.run()
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.per_sample: list[dict[str, Any]] = []
        self.summary: dict[str, dict[str, float]] = {}

    def run(self) -> dict[str, Any]:
        cfg = self.config
        exp_dir = os.path.join(cfg.out_dir, cfg.name)
        os.makedirs(exp_dir, exist_ok=True)

        self._save_config(exp_dir)

        print(f"=== Experiment: {cfg.name} ===")
        print(f"Loading NYU .mat from {cfg.mat_path} ...")
        images, depths = load_nyu_mat(cfg.mat_path)
        n_total = images.shape[0]
        n = min(cfg.num_samples, n_total)
        print(f"Loaded {n_total} samples, will use {n}.")
        print(f"Sparse prior density: {cfg.sparse_density:.0%}")

        processor, model, device, dtype = load_da_model(cfg.model_id)

        methods = self._build_methods()
        method_names = list(methods.keys())
        print(f"Methods: {method_names}")

        per_method_records: dict[str, list[dict[str, float]]] = {
            name: [] for name in method_names
        }

        t_start = time.time()

        for i in range(n):
            rgb = images[i]
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            pil_rgb = Image.fromarray(rgb).convert("RGB")
            gt_depth = depths[i].astype(np.float32)

            sparse_depth, sparse_mask = simulate_sparse_prior(
                gt_depth, density=cfg.sparse_density, seed=cfg.seed + i,
            )
            n_sparse = int(sparse_mask.sum())
            n_valid = int((gt_depth > 0).sum())
            print(
                f"\n[{i + 1}/{n}] Prior: {n_sparse}/{n_valid} "
                f"pixels ({n_sparse / max(n_valid, 1):.1%})"
            )

            d_rel = infer_depth(processor, model, pil_rgb, device, dtype)

            sample_preds: dict[str, np.ndarray] = {}
            sample_extras: dict[str, dict[str, Any]] = {}
            sample_metrics: dict[str, dict[str, float]] = {}

            for name, method in methods.items():
                d_metric, extras = method.calibrate(
                    d_rel, sparse_depth, sparse_mask, rgb, sample_index=i,
                )
                m = compute_metrics(d_metric, gt_depth)
                sample_preds[name] = d_metric
                sample_extras[name] = extras
                sample_metrics[name] = m
                per_method_records[name].append(m)
                print(f"  {name}: {format_metrics(m)}")

            self.per_sample.append({"index": i, "metrics": sample_metrics})

            title = self._make_title(i, sample_metrics)
            fig = make_comparison_figure(
                rgb,
                gt_depth,
                sample_preds,
                sample_extras,
                title=title,
                show_prediction_diff=cfg.show_prediction_diff,
            )
            fig_path = os.path.join(exp_dir, f"nyu_{i:03d}.png")
            fig.savefig(fig_path, dpi=120)
            plt.close(fig)
            print(f"  -> {fig_path}")

        elapsed = time.time() - t_start

        self.summary = {
            name: aggregate_metrics(records)
            for name, records in per_method_records.items()
        }

        self._print_summary(elapsed)
        results = self._save_results(exp_dir, elapsed)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_methods(self) -> dict[str, Any]:
        methods = {}
        for label, params in self.config.methods.items():
            params = dict(params)
            type_name = params.pop("type", label)
            method = get_method(type_name, **params)
            method.name = label
            methods[label] = method
        return methods

    def _make_title(
        self, idx: int, sample_metrics: dict[str, dict[str, float]]
    ) -> str:
        parts = [f"NYU #{idx} (prior {self.config.sparse_density:.0%})"]
        for name, m in sample_metrics.items():
            parts.append(f"{name}: AbsRel={m['abs_rel']:.3f} RMSE={m['rmse']:.3f}")
        return "  |  ".join(parts)

    def _print_summary(self, elapsed: float) -> None:
        cfg = self.config
        print(f"\n{'=' * 70}")
        print(f"Experiment: {cfg.name}  |  {cfg.num_samples} samples  |  "
              f"density={cfg.sparse_density:.0%}  |  {elapsed:.1f}s")
        print(f"{'=' * 70}")

        header = f"{'Method':<20s}"
        for k in _METRIC_KEYS:
            w = 8 if k.startswith("delta") else 10
            header += f" {k:>{w}s}"
        print(header)
        print("-" * len(header))

        for name, agg in self.summary.items():
            row = f"{name:<20s}"
            for k in _METRIC_KEYS:
                v = agg[k]
                w = 8 if k.startswith("delta") else 10
                row += f" {v:>{w}.4f}"
            print(row)
        print()

    def _save_config(self, exp_dir: str) -> None:
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

    def _save_results(self, exp_dir: str, elapsed: float) -> dict[str, Any]:
        results = {
            "config": self.config.to_dict(),
            "elapsed_seconds": elapsed,
            "per_sample": self.per_sample,
            "summary": self.summary,
        }
        results_path = os.path.join(exp_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {results_path}")
        return results
