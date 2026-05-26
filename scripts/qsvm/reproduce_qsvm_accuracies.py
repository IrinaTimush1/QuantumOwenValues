#!/usr/bin/env python3
"""Reproduce the full-circuit QSVM accuracies from the SVQX setup."""

from __future__ import annotations

import argparse

from qsvm_experiment_utils import add_common_data_args, run_replication, str_to_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce full QSVM accuracies for r=1,2,3 using exact statevector kernels."
    )
    add_common_data_args(parser)
    parser.add_argument("--r-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--search-conventions", type=str_to_bool, default=False)
    return parser.parse_args()


def main() -> None:
    run_replication(parse_args())


if __name__ == "__main__":
    main()

