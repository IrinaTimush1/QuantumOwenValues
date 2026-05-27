#!/usr/bin/env python3
"""Compute locked-passive QSVM gate-level Shapley values matching the Owen game."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from qsvm_common import (  # noqa: E402
    QSVM_EXPECTED_ACCURACY,
    load_qsvm_dataset,
    qsvm_active_gates,
    qsvm_accuracy,
    qsvm_gate_name,
    qsvm_passive_gates,
)
from shapley_estimator import estimate_shapley  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--dataset-index", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--r-values", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--alpha-r3", type=float, default=0.01)
    p.add_argument("--r3-runs", type=int, default=2)
    return p.parse_args()


def ensure_dirs(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for r in sorted({int(row["r"]) for row in rows}):
        gates = sorted({int(row["gate_index"]) for row in rows if int(row["r"]) == r})
        for gate in gates:
            vals = [
                float(row["shapley_value"])
                for row in rows
                if int(row["r"]) == r and int(row["gate_index"]) == gate
            ]
            out.append(
                {
                    "r": r,
                    "gate_index": gate,
                    "mean": float(np.mean(vals)) if vals else math.nan,
                    "std": float(np.std(vals, ddof=0)) if vals else math.nan,
                }
            )
    return out


def plot_qsvm(summary: List[Dict[str, object]], output_path: Path) -> None:
    r_values = sorted({int(row["r"]) for row in summary})
    fig, axes = plt.subplots(1, len(r_values), figsize=(5.3 * len(r_values), 4.4), sharey=False)
    if len(r_values) == 1:
        axes = [axes]
    colors = {1: "#4c78a8", 2: "#f58518", 3: "#54a24b"}

    for ax, r in zip(axes, r_values):
        rows = [row for row in summary if int(row["r"]) == r]
        gates = [int(row["gate_index"]) for row in rows]
        means = [float(row["mean"]) for row in rows]
        stds = [float(row["std"]) for row in rows]
        x = np.arange(len(gates))
        ax.bar(x, means, yerr=stds, capsize=3, color=colors.get(r, "#4c78a8"), edgecolor="black", linewidth=0.5)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"r={r}")
        ax.set_xlabel("Gate index")
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gates], rotation=0)
        for xpos, gate in zip(x, gates):
            ax.text(xpos, ax.get_ylim()[0], qsvm_gate_name(gate), ha="center", va="top", fontsize=7)
    axes[0].set_ylabel("Shapley value")
    fig.suptitle("QSVM Shapley reproduction (Heese Fig. 6 setting)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def qsvm_run_settings(r: int, args: argparse.Namespace) -> tuple[float, int]:
    if int(r) in {1, 2}:
        return 1.0, 1
    if int(r) == 3:
        return float(args.alpha_r3), int(args.r3_runs)
    raise ValueError(f"Unsupported r={r}")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dirs(args.output_dir)
    dataset = load_qsvm_dataset(args.data_dir, data_index=args.dataset_index)

    all_rows: List[Dict[str, object]] = []
    sanity_rows: List[Dict[str, object]] = []
    metadata = {
        "data_index": int(args.dataset_index),
        "train_path": str(dataset.train_path),
        "test_path": str(dataset.test_path),
        "runs": [],
    }

    start_all = time.time()
    for r in [int(v) for v in args.r_values]:
        players = qsvm_active_gates(r)
        passive = qsvm_passive_gates(r)
        print(f"QSVM r={r} passive gates: {passive}")
        print(f"QSVM r={r} active Shapley players: {players}")
        full_acc = qsvm_accuracy(dataset, r, players, passive_gates=passive)
        expected = QSVM_EXPECTED_ACCURACY.get(r)
        match = expected is not None and abs(full_acc - expected) <= 1e-9
        msg = "MATCH" if match else f"MISMATCH (got {full_acc:.3f}, expected {expected:.3f})"
        print(f"QSVM r={r} full-circuit accuracy: {full_acc:.3f}; {msg}")
        sanity_rows.append(
            {
                "r": r,
                "dataset_index": int(args.dataset_index),
                "full_circuit_accuracy": full_acc,
                "expected_accuracy": expected,
                "match": bool(match),
            }
        )

        alpha, num_runs = qsvm_run_settings(r, args)
        for run in range(num_runs):
            seed = int(args.base_seed) + run
            print(f"QSVM r={r} Shapley run {run} seed={seed} alpha={alpha}")

            def value_function(coalition):
                return qsvm_accuracy(dataset, r, coalition, passive_gates=passive)

            t0 = time.time()
            phi, cache, stats = estimate_shapley(
                players,
                value_function,
                alpha=alpha,
                k_repetitions=1,
                seed=seed,
                progress=lambda s, r=r: print(f"  r={r} {s}"),
            )
            runtime = time.time() - t0
            for gate in players:
                all_rows.append({"r": r, "gate_index": gate, "run": run, "shapley_value": float(phi[gate])})
            metadata["runs"].append(
                {
                    "r": r,
                    "run": run,
                    "seed": seed,
                    "alpha": alpha,
                    "active_gates": players,
                    "passive_gates": passive,
                    "runtime_seconds": runtime,
                    "stats": stats.__dict__,
                    "cache_size": len(cache),
                }
            )
            print(f"  finished r={r} run={run} in {runtime:.1f}s; cached {len(cache)} coalition means")

    summary = summarize(all_rows)
    write_csv(output_dir / "qsvm_shapley_per_gate.csv", all_rows, ["r", "gate_index", "run", "shapley_value"])
    write_csv(output_dir / "qsvm_shapley_summary.csv", summary, ["r", "gate_index", "mean", "std"])
    write_csv(
        output_dir / "qsvm_full_circuit_sanity.csv",
        sanity_rows,
        ["r", "dataset_index", "full_circuit_accuracy", "expected_accuracy", "match"],
    )
    plot_qsvm(summary, output_dir / "figures" / "qsvm_shapley_reproduction.png")
    metadata["runtime_seconds"] = time.time() - start_all
    with (output_dir / "qsvm_shapley_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {output_dir / 'qsvm_shapley_per_gate.csv'}")
    print(f"Wrote {output_dir / 'qsvm_shapley_summary.csv'}")
    print(f"Wrote {output_dir / 'figures' / 'qsvm_shapley_reproduction.png'}")


if __name__ == "__main__":
    main()
