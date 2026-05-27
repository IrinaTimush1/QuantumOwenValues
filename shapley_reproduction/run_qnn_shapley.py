#!/usr/bin/env python3
"""Compute locked-passive QNN gate-level Shapley values matching the Owen game."""

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

from qnn_common import (  # noqa: E402
    PAPER_THETA,
    QNN_ACTIVE_GATES,
    QNN_GATE_LABELS,
    QNN_PASSIVE_GATES,
    load_qnn_data,
    qnn_expected_accuracy,
    qnn_one_shot_accuracy,
    qnn_threshold_accuracy,
)
from shapley_estimator import estimate_shapley  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, default=ROOT / "data" / "qnn-data.csv")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    p.add_argument("--num-runs", type=int, default=5)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--sanity-trials", type=int, default=200)
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
    for gate in QNN_ACTIVE_GATES:
        vals = [float(row["shapley_value"]) for row in rows if int(row["gate_index"]) == gate]
        mean = float(np.mean(vals)) if vals else math.nan
        std = float(np.std(vals, ddof=0)) if vals else math.nan
        out.append({"gate_index": gate, "mean": mean, "std": std})
    return out


def plot_qnn(summary: List[Dict[str, object]], output_path: Path) -> None:
    gates = [int(row["gate_index"]) for row in summary]
    means = [float(row["mean"]) for row in summary]
    stds = [float(row["std"]) for row in summary]
    labels = [QNN_GATE_LABELS[g] for g in gates]

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(gates))
    ax.bar(x, means, yerr=stds, capsize=3, color="#4c78a8", edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Gate label, locked-passive active gates")
    ax.set_ylabel("Shapley value")
    ax.set_title("QNN locked-passive Shapley (K=32, alpha=0.01, 5 runs)")
    for xpos, gate in zip(x, gates):
        ax.text(xpos, ax.get_ylim()[0], str(gate), ha="center", va="top", fontsize=8)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dirs(args.output_dir)
    X, y = load_qnn_data(args.data_path)

    expected = qnn_expected_accuracy(QNN_ACTIVE_GATES, X, y, theta=PAPER_THETA, passive_gates=QNN_PASSIVE_GATES)
    threshold = qnn_threshold_accuracy(QNN_ACTIVE_GATES, X, y, theta=PAPER_THETA, passive_gates=QNN_PASSIVE_GATES)
    sanity_rng = np.random.default_rng(args.base_seed + 9999)
    one_shot_vals = [
        qnn_one_shot_accuracy(
            frozenset(QNN_ACTIVE_GATES),
            X,
            y,
            rng=sanity_rng,
            theta=PAPER_THETA,
            passive_gates=QNN_PASSIVE_GATES,
        )
        for _ in range(int(args.sanity_trials))
    ]
    sanity_mean = float(np.mean(one_shot_vals))
    heese_ref = 0.80
    sanity_match = abs(threshold - heese_ref) <= 1e-9
    print(
        "QNN full-circuit trained-accuracy sanity: "
        f"{threshold:.3f}; "
        + ("MATCH" if sanity_match else f"MISMATCH (got {threshold:.3f}, expected {heese_ref:.3f})")
    )
    print(
        "QNN one-shot value-function diagnostic: "
        f"expected_mean={expected:.3f}, sampled_mean={sanity_mean:.3f} "
        f"over {int(args.sanity_trials)} trials"
    )

    rows: List[Dict[str, object]] = []
    run_metadata = []
    start_all = time.time()
    for run in range(int(args.num_runs)):
        seed = int(args.base_seed) + run
        value_rng = np.random.default_rng(seed + 100000)

        def value_function(coalition):
            return qnn_one_shot_accuracy(
                coalition,
                X,
                y,
                rng=value_rng,
                theta=PAPER_THETA,
                passive_gates=QNN_PASSIVE_GATES,
            )

        print(f"QNN Shapley run {run} seed={seed}")
        t0 = time.time()
        phi, cache, stats = estimate_shapley(
            QNN_ACTIVE_GATES,
            value_function,
            alpha=float(args.alpha),
            k_repetitions=int(args.k),
            seed=seed,
            progress=lambda msg: print(f"  {msg}"),
        )
        for gate in QNN_ACTIVE_GATES:
            rows.append({"gate_index": gate, "run": run, "shapley_value": float(phi[gate])})
        meta = {
            "run": run,
            "seed": seed,
            "runtime_seconds": time.time() - t0,
            "stats": stats.__dict__,
            "cache_size": len(cache),
        }
        run_metadata.append(meta)
        print(
            f"  finished run {run} in {meta['runtime_seconds']:.1f}s; "
            f"cached {len(cache)} coalition means"
        )

    summary = summarize(rows)
    write_csv(output_dir / "qnn_shapley_per_gate.csv", rows, ["gate_index", "run", "shapley_value"])
    write_csv(output_dir / "qnn_shapley_summary.csv", summary, ["gate_index", "mean", "std"])
    plot_qnn(summary, output_dir / "figures" / "qnn_shapley_reproduction.png")

    payload = {
        "active_gates": QNN_ACTIVE_GATES,
        "passive_gates": QNN_PASSIVE_GATES,
        "baseline": "locked-passive Shapley baseline matching the QNN E/M/X Owen game",
        "theta": PAPER_THETA.tolist(),
        "K": int(args.k),
        "alpha": float(args.alpha),
        "num_runs": int(args.num_runs),
        "full_circuit_accuracy": {
            "heese_reference": heese_ref,
            "threshold_accuracy": threshold,
            "expected_one_shot_accuracy": expected,
            "sampled_one_shot_mean": sanity_mean,
            "sampled_one_shot_std": float(np.std(one_shot_vals, ddof=0)),
            "sanity_trials": int(args.sanity_trials),
            "match": bool(sanity_match),
        },
        "runtime_seconds": time.time() - start_all,
        "runs": run_metadata,
    }
    with (output_dir / "qnn_shapley_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {output_dir / 'qnn_shapley_per_gate.csv'}")
    print(f"Wrote {output_dir / 'qnn_shapley_summary.csv'}")
    print(f"Wrote {output_dir / 'figures' / 'qnn_shapley_reproduction.png'}")


if __name__ == "__main__":
    main()
