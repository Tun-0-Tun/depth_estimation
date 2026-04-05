#!/usr/bin/env python3
"""
Unified experiment runner for depth estimation calibration methods.

Two modes of operation:

1. Config mode (recommended for reproducible experiments):
   python scripts/run_experiment.py --config configs/baseline.json

2. Quick CLI mode (for one-off tests):
   python scripts/run_experiment.py \\
       --mat-path data/nyu_depth_v2/nyu_depth_v2_labeled.mat \\
       --methods global local \\
       --num-samples 20 \\
       --name quick_test
"""

import argparse
import json

from depth_estimation.calibration.methods import list_methods
from depth_estimation.evaluation.experiment import ExperimentConfig, ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run depth estimation experiment with multiple calibration methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available method types: {', '.join(list_methods())}",
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file. If provided, other CLI args are ignored.",
    )

    parser.add_argument("--name", type=str, default="experiment",
                        help="Experiment name (used as output subdirectory).")
    parser.add_argument("--mat-path", type=str,
                        default="data/nyu_depth_v2/nyu_depth_v2_labeled.mat")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sparse-density", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--out-dir", type=str, default="./outputs/experiments")

    parser.add_argument(
        "--methods", type=str, nargs="+", default=["global", "local"],
        help="Method type names to run (from registry).",
    )
    parser.add_argument("--n-segments", type=int, default=200,
                        help="SLIC superpixel count (for 'local' method).")
    parser.add_argument("--sigma", type=float, default=15.0,
                        help="Gaussian smoothing sigma (for local smooth_mode=gaussian or --smooth).")
    parser.add_argument("--smooth", action="store_true",
                        help="Legacy: enable Gaussian smoothing (same as --smooth-mode gaussian).")
    parser.add_argument(
        "--smooth-mode", type=str, default="none",
        choices=("none", "gaussian", "bilateral"),
        help="Post-smoothing of local s/t fields: none | gaussian | bilateral.",
    )
    parser.add_argument("--sigma-spatial", type=float, default=5.0,
                        help="Bilateral spatial sigma in pixels (local smooth_mode=bilateral).")
    parser.add_argument("--range-scale", type=float, default=0.25,
                        help="Bilateral range scale: sigma_range *= range_scale * std(s|t) when auto.")
    parser.add_argument("--bilateral-max-radius", type=int, default=10,
                        help="Max half-window radius for bilateral (memory cap).")
    parser.add_argument("--min-pixels", type=int, default=10,
                        help="Min valid pixels per superpixel (for 'local' method).")
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Do not add second row with pred−GT error maps in saved figures.",
    )

    return parser.parse_args()


def config_from_cli(args) -> ExperimentConfig:
    """Build ExperimentConfig from CLI arguments (quick mode)."""
    methods = {}
    for method_type in args.methods:
        if method_type == "local":
            methods[method_type] = {
                "type": "local",
                "n_segments": args.n_segments,
                "sigma": args.sigma,
                "smooth": args.smooth,
                "smooth_mode": args.smooth_mode,
                "sigma_spatial": args.sigma_spatial,
                "range_scale": args.range_scale,
                "bilateral_max_radius": args.bilateral_max_radius,
                "min_pixels": args.min_pixels,
            }
        else:
            methods[method_type] = {"type": method_type}

    return ExperimentConfig(
        name=args.name,
        mat_path=args.mat_path,
        num_samples=args.num_samples,
        sparse_density=args.sparse_density,
        seed=args.seed,
        model_id=args.model_id,
        out_dir=args.out_dir,
        methods=methods,
        show_prediction_diff=not args.no_diff,
    )


def main():
    args = parse_args()

    if args.config:
        print(f"Loading config from {args.config}")
        config = ExperimentConfig.from_json(args.config)
    else:
        config = config_from_cli(args)

    print(f"Experiment: {config.name}")
    print(f"Methods: {list(config.methods.keys())}")
    print(f"Config: {json.dumps(config.to_dict(), indent=2)}")
    print()

    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
