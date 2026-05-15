#!/usr/bin/env python3
"""Create the combined QSVM Owen gate-value summary plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsvm_experiment_utils import plot_combined_gate_values, plot_combined_value_distributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined r=1,2,3 QSVM Owen gate values.")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "qsvm_owen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate_png, gate_pdf = plot_combined_gate_values(args.output_dir)
    print(f"Saved combined gate plot to {gate_png}")
    print(f"Saved combined gate plot to {gate_pdf}")
    dist_png, dist_pdf = plot_combined_value_distributions(args.output_dir)
    print(f"Saved combined distribution plot to {dist_png}")
    print(f"Saved combined distribution plot to {dist_pdf}")


if __name__ == "__main__":
    main()
