#!/usr/bin/env python3
"""Run exact QSVM Owen experiments for all five downloaded datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from qsvm_experiment_utils import ensure_dir, run_exact_owen, str_to_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QSVM Owen experiments for multiple dataset indices and r values."
    )
    parser.add_argument("--data-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--r-values", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "qsvm_owen_all_datasets")
    parser.add_argument("--svc-c", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--make-value-distribution", type=str_to_bool, default=False)
    parser.add_argument("--validate-simulator", type=str_to_bool, default=True)
    parser.add_argument("--silent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    summaries = []

    for data_index in args.data_indices:
        dataset_dir = output_dir / f"dataset_{data_index}"
        for r in args.r_values:
            print("=" * 88)
            print(f"Running QSVM Owen: dataset={data_index}, r={r}")
            sub_args = SimpleNamespace(
                data_index=int(data_index),
                data_dir=args.data_dir,
                train_path=None,
                test_path=None,
                output_dir=dataset_dir,
                svc_c=float(args.svc_c),
                n_jobs=1,
                force=bool(args.force),
                make_value_distribution=bool(args.make_value_distribution),
                validate_simulator=bool(args.validate_simulator),
                silent=bool(args.silent),
            )
            summary = run_exact_owen(sub_args, r=int(r))
            summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "all_run_summaries.csv"
    summary_df.to_csv(summary_path, index=False)
    print("=" * 88)
    print(f"Saved run summary to {summary_path}")
    print("Next run:")
    print(f"  python scripts/qsvm/aggregate_qsvm_owen_all_datasets.py --output-dir {output_dir}")


if __name__ == "__main__":
    main()
