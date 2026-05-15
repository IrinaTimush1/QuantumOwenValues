#!/usr/bin/env python3
"""Compute exact QSVM Owen values for r=2 under the E/M/X partition."""

from __future__ import annotations

import argparse

from qsvm_experiment_utils import add_owen_args, run_exact_owen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact QSVM Owen values for r=2.")
    add_owen_args(parser)
    return parser.parse_args()


def main() -> None:
    run_exact_owen(parse_args(), r=2)


if __name__ == "__main__":
    main()

