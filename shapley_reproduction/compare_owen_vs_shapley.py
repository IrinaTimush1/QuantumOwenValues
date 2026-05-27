#!/usr/bin/env python3
"""Compare thesis Owen values with reproduced Heese Shapley values."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SHAPLEY_DIR = Path(__file__).resolve().parent
if str(SHAPLEY_DIR) not in sys.path:
    sys.path.insert(0, str(SHAPLEY_DIR))

from qnn_common import QNN_GATE_LABELS  # noqa: E402
from qsvm_common import qsvm_gate_name  # noqa: E402


QNN_OWEN_ACTIVE = [2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapley-results", type=Path, default=SHAPLEY_DIR / "results")
    p.add_argument("--repo-results", type=Path, default=ROOT / "results")
    p.add_argument("--dataset-index", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=SHAPLEY_DIR / "results")
    return p.parse_args()


def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)


def corr_pair(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float, float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 2 or np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return math.nan, math.nan, math.nan, math.nan
    spearman = spearmanr(x_arr, y_arr)
    pearson = pearsonr(x_arr, y_arr)
    return float(spearman.statistic), float(spearman.pvalue), float(pearson.statistic), float(pearson.pvalue)


def load_qnn_comparison(repo_results: Path, shapley_results: Path) -> pd.DataFrame:
    shap = pd.read_csv(shapley_results / "qnn_shapley_summary.csv")
    owen = pd.read_csv(repo_results / "qnn_owen_emx_main" / "aggregated_owen_gate_values.csv")
    owen = owen[owen["K"] == 32].copy()
    owen = owen[["gate_1_based", "mean_owen"]].rename(columns={"gate_1_based": "gate_index", "mean_owen": "owen"})
    shap = shap[["gate_index", "mean"]].rename(columns={"mean": "shapley"})
    gates = sorted(QNN_OWEN_ACTIVE)
    print(f"QNN comparison uses the shared locked-passive active set: {gates}")
    print("QNN passive gates fixed in both Owen and Shapley: [1, 3, 8, 10]")
    df = pd.merge(owen, shap, on="gate_index", how="inner")
    return df[df["gate_index"].isin(gates)].sort_values("gate_index").reset_index(drop=True)


def load_qsvm_comparison(repo_results: Path, shapley_results: Path, dataset_index: int, r: int) -> pd.DataFrame:
    shap = pd.read_csv(shapley_results / "qsvm_shapley_summary.csv")
    shap = shap[shap["r"] == int(r)][["gate_index", "mean"]].rename(columns={"mean": "shapley"})
    owen_path = repo_results / "qsvm_owen_all_datasets" / f"dataset_{int(dataset_index)}" / f"r{int(r)}" / "gate_owen_values.csv"
    owen = pd.read_csv(owen_path)
    owen = owen[["gate_index_1based", "owen_value"]].rename(
        columns={"gate_index_1based": "gate_index", "owen_value": "owen"}
    )
    owen["owen"] = owen["owen"].fillna(0.0)
    return pd.merge(owen, shap, on="gate_index", how="inner").sort_values("gate_index").reset_index(drop=True)


def write_correlations(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["circuit", "n_gates_compared", "spearman_rho", "spearman_p", "pearson_r", "pearson_p"],
        )
        writer.writeheader()
        writer.writerows(rows)


def overlay_plot(df: pd.DataFrame, title: str, labels: List[str], output_path: Path) -> None:
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.5, 0.45 * len(df)), 4.5))
    ax.bar(x - width / 2, df["owen"], width=width, label="Owen", color="#4c78a8", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, df["shapley"], width=width, label="Shapley", color="#f58518", edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in df["gate_index"]])
    ax.set_xlabel("Gate index")
    ax.set_ylabel("Attribution value")
    ax.set_title(title)
    ax.legend(frameon=False)
    for xpos, label in zip(x, labels):
        ax.text(xpos, ax.get_ylim()[0], label, ha="center", va="top", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def add_corr_row(rows: List[Dict[str, object]], circuit: str, df: pd.DataFrame) -> None:
    sr, sp, pr, pp = corr_pair(df["owen"], df["shapley"])
    rows.append(
        {
            "circuit": circuit,
            "n_gates_compared": int(len(df)),
            "spearman_rho": sr,
            "spearman_p": sp,
            "pearson_r": pr,
            "pearson_p": pp,
        }
    )


def main() -> None:
    args = parse_args()
    ensure_dirs(args.output_dir)

    corr_rows: List[Dict[str, object]] = []

    qnn = load_qnn_comparison(args.repo_results, args.shapley_results)
    add_corr_row(corr_rows, "QNN", qnn)
    overlay_plot(
        qnn,
        "QNN Owen vs Shapley (intersection of active gates)",
        [QNN_GATE_LABELS[int(g)] for g in qnn["gate_index"]],
        args.output_dir / "figures" / "owen_vs_shapley_qnn.png",
    )

    print("QSVM comparison uses active gates only; passive H gates are fixed in both Owen and Shapley.")
    for r in [1, 2, 3]:
        df = load_qsvm_comparison(args.repo_results, args.shapley_results, args.dataset_index, r)
        add_corr_row(corr_rows, f"QSVM r={r}", df)
        overlay_plot(
            df,
            f"QSVM r={r} Owen vs Shapley",
            [qsvm_gate_name(int(g)) for g in df["gate_index"]],
            args.output_dir / "figures" / f"owen_vs_shapley_qsvm_r{r}.png",
        )

    out_csv = args.output_dir / "owen_vs_shapley_correlation.csv"
    write_correlations(out_csv, corr_rows)

    print("\nOwen vs Shapley correlation summary")
    print("-----------------------------------")
    for row in corr_rows:
        print(
            f"{row['circuit']:8s} n={row['n_gates_compared']:2d} "
            f"Spearman={row['spearman_rho']:.3g} (p={row['spearman_p']:.3g}), "
            f"Pearson={row['pearson_r']:.3g} (p={row['pearson_p']:.3g})"
        )
    print(f"\nWrote {out_csv}")
    print(f"Wrote overlay figures under {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
