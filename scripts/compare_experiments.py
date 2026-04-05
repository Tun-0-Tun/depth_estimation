#!/usr/bin/env python3
"""
Compare metrics across multiple experiments.

Usage:
    python scripts/compare_experiments.py \\
        outputs/experiments/baseline/results.json \\
        outputs/experiments/smooth_sigma/results.json

    python scripts/compare_experiments.py outputs/experiments/*/results.json

Prints a consolidated comparison table and optionally exports to CSV.
"""

import argparse
import csv
import json
from pathlib import Path

METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")
ERROR_METRICS = {"abs_rel", "sq_rel", "rmse", "rmse_log"}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_comparison_table(experiments: list[tuple[str, dict]]):
    """Print a formatted comparison table to stdout."""
    rows = []
    for exp_name, data in experiments:
        summary = data.get("summary", {})
        for method_name, metrics in summary.items():
            rows.append((exp_name, method_name, metrics))

    if not rows:
        print("No results to compare.")
        return

    col_w_exp = max(len("Experiment"), max(len(r[0]) for r in rows))
    col_w_method = max(len("Method"), max(len(r[1]) for r in rows))
    metric_w = 10

    header = f"{'Experiment':<{col_w_exp}s}  {'Method':<{col_w_method}s}"
    for k in METRIC_KEYS:
        arrow = " ↓" if k in ERROR_METRICS else " ↑"
        header += f"  {k + arrow:>{metric_w}s}"
    print(header)
    print("=" * len(header))

    prev_exp = None
    for exp_name, method_name, metrics in rows:
        if prev_exp is not None and exp_name != prev_exp:
            print("-" * len(header))
        prev_exp = exp_name

        row = f"{exp_name:<{col_w_exp}s}  {method_name:<{col_w_method}s}"
        for k in METRIC_KEYS:
            v = metrics.get(k, float("nan"))
            row += f"  {v:>{metric_w}.4f}"
        print(row)

    print()


def find_best(experiments: list[tuple[str, dict]]):
    """Highlight the best method for each metric across all experiments."""
    all_rows = []
    for exp_name, data in experiments:
        for method_name, metrics in data.get("summary", {}).items():
            all_rows.append((f"{exp_name}/{method_name}", metrics))

    if not all_rows:
        return

    print("Best across all experiments:")
    for k in METRIC_KEYS:
        values = [(name, m.get(k, float("nan"))) for name, m in all_rows]
        values = [(n, v) for n, v in values if v == v]  # filter NaN
        if not values:
            continue
        if k in ERROR_METRICS:
            best_name, best_val = min(values, key=lambda x: x[1])
        else:
            best_name, best_val = max(values, key=lambda x: x[1])
        print(f"  {k:<12s}: {best_val:.4f}  ({best_name})")
    print()


def export_csv(experiments: list[tuple[str, dict]], path: str):
    """Write comparison table to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "method"] + list(METRIC_KEYS))
        for exp_name, data in experiments:
            for method_name, metrics in data.get("summary", {}).items():
                row = [exp_name, method_name]
                row += [f"{metrics.get(k, float('nan')):.6f}" for k in METRIC_KEYS]
                writer.writerow(row)
    print(f"CSV exported to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics across depth estimation experiments."
    )
    parser.add_argument(
        "results", nargs="+", type=str,
        help="Paths to results.json files from different experiments.",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export comparison table to CSV file.",
    )
    args = parser.parse_args()

    experiments = []
    for path in args.results:
        data = load_results(path)
        exp_name = data.get("config", {}).get("name", Path(path).parent.name)
        experiments.append((exp_name, data))

    print(f"\nComparing {len(experiments)} experiment(s):\n")
    print_comparison_table(experiments)
    find_best(experiments)

    if args.csv:
        export_csv(experiments, args.csv)


if __name__ == "__main__":
    main()
