#!/usr/bin/env python3
"""
Compare metrics across multiple experiments.

Usage:
    # All saved runs under outputs/experiments (default; every results.json found)
    python scripts/compare_experiments.py

    python scripts/compare_experiments.py \\
        outputs/experiments/baseline/results.json \\
        outputs/experiments/smooth_sigma/results.json

    python scripts/compare_experiments.py outputs/experiments/*/results.json

    python scripts/compare_experiments.py --root ./outputs/experiments

Prints a consolidated comparison table and optionally exports to CSV / Markdown.
Experiment rows are keyed by the **output directory name** (parent of results.json),
so different runs stay distinct even if config ``name`` matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")
ERROR_METRICS = {"abs_rel", "sq_rel", "rmse", "rmse_log"}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_comparison_table(
    experiments: list[tuple[str, dict]],
    only_methods: set[str] | None = None,
):
    """Print a formatted comparison table to stdout."""
    rows = []
    for exp_name, data in experiments:
        summary = data.get("summary", {})
        for method_name, metrics in summary.items():
            if only_methods is not None and method_name not in only_methods:
                continue
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


def find_best(
    experiments: list[tuple[str, dict]],
    only_methods: set[str] | None = None,
):
    """Highlight the best method for each metric across all experiments."""
    all_rows = []
    for exp_name, data in experiments:
        for method_name, metrics in data.get("summary", {}).items():
            if only_methods is not None and method_name not in only_methods:
                continue
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


def export_csv(
    experiments: list[tuple[str, dict]],
    path: str,
    only_methods: set[str] | None = None,
):
    """Write comparison table to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "method"] + list(METRIC_KEYS))
        for exp_name, data in experiments:
            for method_name, metrics in data.get("summary", {}).items():
                if only_methods is not None and method_name not in only_methods:
                    continue
                row = [exp_name, method_name]
                row += [f"{metrics.get(k, float('nan')):.6f}" for k in METRIC_KEYS]
                writer.writerow(row)
    print(f"CSV exported to {path}", file=sys.stderr)


def export_markdown(
    experiments: list[tuple[str, dict]],
    path: str,
    only_methods: set[str] | None = None,
    title: str | None = None,
) -> None:
    """Write comparison table as GitHub-flavored Markdown."""
    rows: list[tuple[str, str, dict[str, float]]] = []
    for exp_name, data in experiments:
        for method_name, metrics in data.get("summary", {}).items():
            if only_methods is not None and method_name not in only_methods:
                continue
            rows.append((exp_name, method_name, metrics))

    lines: list[str] = []
    if title:
        lines.append(f"## {title}\n")
    header = "| experiment | method | " + " | ".join(METRIC_KEYS) + " |"
    sep = "| --- | --- | " + " | ".join(["---"] * len(METRIC_KEYS)) + " |"
    lines.extend([header, sep])
    for exp_name, method_name, metrics in rows:
        cells = [exp_name, method_name]
        cells += [f"{metrics.get(k, float('nan')):.4f}" for k in METRIC_KEYS]
        lines.append("| " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown exported to {path}", file=sys.stderr)


def collect_result_paths(results: list[str], root: str) -> list[str]:
    """
    If *results* is empty, find every ``**/results.json`` under *root*.
    If *results* is a single existing directory, glob inside it.
    Otherwise return *results* as explicit file paths.
    """
    if not results:
        base = Path(root)
        if not base.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        found = sorted(base.glob("**/results.json"))
        if not found:
            raise SystemExit(f"No results.json under {base}")
        return [str(p) for p in found]

    if len(results) == 1:
        p = Path(results[0])
        if p.is_dir():
            found = sorted(p.glob("**/results.json"))
            if not found:
                raise SystemExit(f"No results.json under {p}")
            return [str(x) for x in found]

    out: list[str] = []
    for r in results:
        rp = Path(r)
        if not rp.is_file():
            raise SystemExit(f"Not a file: {r}")
        out.append(str(rp.resolve()))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics across depth estimation experiments.",
    )
    parser.add_argument(
        "results",
        nargs="*",
        type=str,
        default=[],
        help="Paths to results.json files. If omitted, scan --root for all **/results.json. "
        "A single directory scans that tree.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="outputs/experiments",
        help="When no result files are given: scan this directory for **/results.json "
        "(default: outputs/experiments).",
    )
    parser.add_argument(
        "--only-methods",
        type=str,
        nargs="+",
        default=None,
        help="Keep only these method keys (e.g. global local_bilateral inr_film).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export comparison table to CSV file.",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Export comparison table to a Markdown file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional heading for the Markdown export.",
    )
    args = parser.parse_args()

    paths = collect_result_paths(list(args.results), args.root)
    only = set(args.only_methods) if args.only_methods else None

    experiments: list[tuple[str, dict]] = []
    for path in paths:
        data = load_results(path)
        exp_id = Path(path).parent.name
        experiments.append((exp_id, data))

    print(f"\nLoaded {len(experiments)} experiment result file(s) ({len(paths)} path(s)).\n")
    print_comparison_table(experiments, only_methods=only)
    find_best(experiments, only_methods=only)

    if args.csv:
        export_csv(experiments, args.csv, only_methods=only)
    if args.markdown:
        export_markdown(experiments, args.markdown, only_methods=only, title=args.title)


if __name__ == "__main__":
    main()
