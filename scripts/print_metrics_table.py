#!/usr/bin/env python3
"""
Print a single unified metrics table from ``results.json`` (one experiment, several methods).

Also export CSV or Markdown for reports.

Example (one experiment, one ``results.json``)::

    uv run python scripts/print_metrics_table.py \\
        outputs/experiments/metrics_three_approaches/results.json \\
        --csv outputs/metrics_three.csv

To merge **all** runs under ``outputs/experiments``, use ``compare_experiments.py`` (no args).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from depth_estimation.evaluation.metrics import _METRIC_KEYS


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _print_console(summary: dict[str, dict[str, float]]) -> None:
    names = list(summary.keys())
    if not names:
        print("No methods in summary.", file=sys.stderr)
        return

    col_w = max(len("method"), max(len(n) for n in names))
    metric_w = 10
    header = f"{'method':<{col_w}s}"
    for k in _METRIC_KEYS:
        header += f"  {k:>{metric_w}s}"
    print(header)
    print("-" * len(header))

    for name in names:
        m = summary[name]
        row = f"{name:<{col_w}s}"
        for k in _METRIC_KEYS:
            v = m.get(k, float("nan"))
            row += f"  {v:>{metric_w}.4f}"
        print(row)


def _write_csv(path: str, summary: dict[str, dict[str, float]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(_METRIC_KEYS))
        for name in summary:
            m = summary[name]
            w.writerow([name] + [f"{m.get(k, float('nan')):.6f}" for k in _METRIC_KEYS])
    print(f"Wrote {path}", file=sys.stderr)


def _write_markdown(path: str, summary: dict[str, dict[str, float]], title: str | None) -> None:
    lines: list[str] = []
    if title:
        lines.append(f"## {title}\n")
    header = "| method | " + " | ".join(_METRIC_KEYS) + " |"
    sep = "|" + "|".join(["---"] * (1 + len(_METRIC_KEYS))) + "|"
    lines.append(header)
    lines.append(sep)
    for name in summary:
        m = summary[name]
        cells = [name] + [f"{m.get(k, float('nan')):.4f}" for k in _METRIC_KEYS]
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")
    print(f"Wrote {path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Print metrics table from experiment results.json.")
    p.add_argument("results_json", type=str, help="Path to results.json from ExperimentRunner.")
    p.add_argument("--csv", type=str, default=None, help="Write the same table to CSV.")
    p.add_argument("--markdown", type=str, default=None, help="Write a Markdown table.")
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional heading inside the Markdown file.",
    )
    args = p.parse_args()

    data = _load(args.results_json)
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise SystemExit("Invalid results.json: missing 'summary' object.")

    _print_console(summary)
    if args.csv:
        _write_csv(args.csv, summary)
    if args.markdown:
        _write_markdown(args.markdown, summary, args.title)


if __name__ == "__main__":
    main()
