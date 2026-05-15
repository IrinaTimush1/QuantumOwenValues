#!/usr/bin/env python3
"""Aggregate QSVM Owen outputs over the five local datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsvm_experiment_utils import aggregate_qsvm_dataset_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate QSVM Owen values over datasets.")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "qsvm_owen_all_datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = aggregate_qsvm_dataset_runs(args.output_dir)
    print("Saved aggregate outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
