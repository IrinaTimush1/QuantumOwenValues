#!/usr/bin/env python3
"""
Aggregate the thesis QSVM Owen outputs over the five dataset splits.

Run this after `run_qsvm_owen_all_datasets.py`. It reads the per-dataset,
per-r CSV files and writes aggregate mean/std tables and auxiliary plots under
`results/qsvm_owen_all_datasets/aggregate/`.
"""

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
