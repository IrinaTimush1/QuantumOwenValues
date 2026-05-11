#!/usr/bin/env python3
"""
script_plot_estimator_validation.py

Create compact, paper-style plots for the Monte Carlo Owen estimator validation
WITHOUT recomputing any Owen values.

This script reads the CSV outputs already produced by the estimator-validation
experiment and makes a small number of informative plots for these metrics:

1. MAE
2. Max Absolute Error
3. RMSE
4. Normalized Error
5. Spearman rho
6. Runtime

Aggregation rule
----------------
1) Average over seeds within each circuit
2) Then average over circuits for each (property, n)

Important plotting choices
--------------------------
- All shaded bands show ONE STANDARD DEVIATION across circuits.
- Max Absolute Error is shown as a line only (no std shading).
- Additional tradeoff plots are produced:
    normalized error (y-axis) vs runtime (x-axis),
    one figure for magic and one for entanglement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = ROOT / "data" / "plots" / "estimator_validation_15"

DEFAULT_PER_CIRCUIT_CSV = DEFAULT_BASE_DIR / "per_circuit_summary.csv"
DEFAULT_PER_RUN_CSV = DEFAULT_BASE_DIR / "per_run_rows.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "paper_plots"

PROPERTIES = ["magic", "entanglement"]
N_VALUES = [100, 300, 500, 1000]

METRIC_CANDIDATES: Dict[str, List[str]] = {
    "MAE": ["gate_mae_mean", "gate_mae"],
    "Max Absolute Error": ["gate_max_ae_mean", "gate_max_ae"],
    "RMSE": ["gate_rmse_mean", "gate_rmse"],
    "Normalized Error": [
        "gate_normalized_mae_total_mean",
        "gate_norm_mae_total_mean",
        "gate_normalized_mae_total",
        "gate_norm_mae_total",
        "gate_normalized_mae_max_mean",
        "gate_norm_mae_max_mean",
        "gate_normalized_mae_max",
        "gate_norm_mae_max",
    ],
    "Spearman rho": ["gate_spearman_rho_mean", "gate_spearman_rho"],
    "Runtime (s)": ["runtime_mean", "runtime_s", "mean_runtime_s"],
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-circuit-csv", type=Path, default=DEFAULT_PER_CIRCUIT_CSV)
    p.add_argument("--per-run-csv", type=Path, default=DEFAULT_PER_RUN_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--style", type=str, default="seaborn-v0_8-whitegrid")
    return p.parse_args()


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def pick_existing_column(df: pd.DataFrame, candidates: List[str], metric_name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find a source column for metric '{metric_name}'. "
        f"Tried: {candidates}\nAvailable columns: {list(df.columns)}"
    )


def prettify_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])


def set_metric_ylim(ax: plt.Axes, metric: str) -> None:
    if metric == "Spearman rho":
        ax.set_ylim(0.0, 1.02)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=max(0.0, ymin))


def normalize_n_col(df: pd.DataFrame) -> pd.DataFrame:
    if "n" in df.columns and "n_samples" not in df.columns:
        df = df.rename(columns={"n": "n_samples"})
    return df


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_per_run_seed_averaged(per_run_csv: Path) -> pd.DataFrame:
    if not per_run_csv.exists():
        raise FileNotFoundError(f"Missing per_run_rows.csv: {per_run_csv}")

    df = pd.read_csv(per_run_csv)
    df = normalize_n_col(df)

    required = {"benchmark_id", "property", "n_samples"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{per_run_csv} is missing required columns: {sorted(missing)}"
        )

    group_cols = ["benchmark_id", "property", "n_samples"]
    if "family_role" in df.columns:
        group_cols.insert(1, "family_role")

    numeric_cols = []
    for cand_list in METRIC_CANDIDATES.values():
        for c in cand_list:
            if c in df.columns:
                numeric_cols.append(c)
    numeric_cols = sorted(set(numeric_cols))

    agg_dict = {c: "mean" for c in numeric_cols}
    out = df.groupby(group_cols, as_index=False).agg(agg_dict)

    rename_out = {}
    for c in out.columns:
        if c in group_cols:
            continue
        if not c.endswith("_mean"):
            rename_out[c] = f"{c}_mean"
    out = out.rename(columns=rename_out)
    return out


def load_seed_averaged_per_circuit(
    per_circuit_csv: Path,
    per_run_csv: Path,
) -> pd.DataFrame:
    """
    Build one row per (benchmark_id, family_role, property, n_samples),
    seed-averaged within circuit.

    Uses per_circuit_summary.csv where possible, and fills missing metrics
    (especially normalized error) from per_run_rows.csv.
    """
    per_run_seed_avg = load_per_run_seed_averaged(per_run_csv)

    if not per_circuit_csv.exists():
        return per_run_seed_avg

    per_circuit = pd.read_csv(per_circuit_csv)
    per_circuit = normalize_n_col(per_circuit)

    required = {"benchmark_id", "property", "n_samples"}
    missing = required - set(per_circuit.columns)
    if missing:
        raise ValueError(
            f"{per_circuit_csv} is missing required columns: {sorted(missing)}"
        )

    merge_keys = ["benchmark_id", "property", "n_samples"]
    if "family_role" in per_circuit.columns and "family_role" in per_run_seed_avg.columns:
        merge_keys = ["benchmark_id", "family_role", "property", "n_samples"]

    merged = per_circuit.copy()

    missing_cols = [
        c for c in per_run_seed_avg.columns
        if c not in merged.columns and c not in merge_keys
    ]
    if missing_cols:
        merged = merged.merge(
            per_run_seed_avg[merge_keys + missing_cols],
            on=merge_keys,
            how="left",
        )

    return merged


def build_plot_summary(seed_averaged_per_circuit: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary:
      one row per (property, n_samples, metric)
    with:
      mean, std, n_circuits
    """
    rows = []

    for metric_label, candidates in METRIC_CANDIDATES.items():
        source_col = pick_existing_column(seed_averaged_per_circuit, candidates, metric_label)

        for prop in sorted(seed_averaged_per_circuit["property"].unique()):
            sub_prop = seed_averaged_per_circuit.loc[
                seed_averaged_per_circuit["property"] == prop
            ].copy()

            for n in N_VALUES:
                sub = sub_prop.loc[sub_prop["n_samples"] == n, source_col].dropna()
                n_circuits = int(len(sub))
                if n_circuits == 0:
                    continue

                mean = float(sub.mean())
                std = float(sub.std(ddof=1)) if n_circuits > 1 else 0.0

                rows.append(
                    {
                        "property": prop,
                        "n_samples": int(n),
                        "metric": metric_label,
                        "source_column": source_col,
                        "n_circuits": n_circuits,
                        "mean": mean,
                        "std": std,
                        "lower_std": mean - std,
                        "upper_std": mean + std,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_metric_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    metric: str,
    properties: List[str],
) -> None:
    sub = plot_df.loc[
        (plot_df["metric"] == metric) &
        (plot_df["property"].isin(properties))
    ].copy()

    if sub.empty:
        ax.set_visible(False)
        return

    color_map = {
        "magic": "#ff7f0e",
        "entanglement": "#1f77b4",
    }

    for prop in properties:
        s = sub.loc[sub["property"] == prop].sort_values("n_samples")
        if s.empty:
            continue

        x = s["n_samples"].to_numpy(dtype=float)
        y = s["mean"].to_numpy(dtype=float)
        std = s["std"].to_numpy(dtype=float)

        ax.plot(
            x, y,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=prop,
            color=color_map.get(prop, None),
        )

        # Do not show std band for Max Absolute Error
        if metric != "Max Absolute Error":
            ax.fill_between(
                x,
                y - std,
                y + std,
                alpha=0.18,
                color=color_map.get(prop, None),
            )

    ax.set_title(metric, fontsize=13, pad=8)
    ax.set_xlabel("n samples")
    prettify_axes(ax)
    set_metric_ylim(ax, metric)


# ---------------------------------------------------------------------
# Main figures
# ---------------------------------------------------------------------

def plot_compact_dashboard(plot_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    metrics = [
        "MAE",
        "Max Absolute Error",
        "RMSE",
        "Normalized Error",
        "Spearman rho",
        "Runtime (s)",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        plot_metric_panel(ax, plot_df, metric, properties=PROPERTIES)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=2,
            frameon=False,
            fontsize=12,
        )

    fig.suptitle(
        "Monte Carlo Owen Estimator Validation\n"
        "Mean over circuits after seed-averaging within each circuit; shaded bands = std",
        fontsize=20,
        y=0.975,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.80])

    fig.savefig(output_dir / "estimator_validation_compact_dashboard.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "estimator_validation_compact_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_property_dashboard(
    plot_df: pd.DataFrame,
    property_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    metrics = [
        "MAE",
        "Max Absolute Error",
        "RMSE",
        "Normalized Error",
        "Spearman rho",
        "Runtime (s)",
    ]

    sub = plot_df.loc[plot_df["property"] == property_name].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.flatten()

    color = "#1f77b4" if property_name == "entanglement" else "#ff7f0e"

    for ax, metric in zip(axes, metrics):
        s = sub.loc[sub["metric"] == metric].sort_values("n_samples")
        if s.empty:
            ax.set_visible(False)
            continue

        x = s["n_samples"].to_numpy(dtype=float)
        y = s["mean"].to_numpy(dtype=float)
        std = s["std"].to_numpy(dtype=float)

        ax.plot(x, y, marker="o", linewidth=2.2, markersize=6, color=color)

        if metric != "Max Absolute Error":
            ax.fill_between(x, y - std, y + std, alpha=0.18, color=color, label="std")

        ax.set_title(metric, fontsize=13, pad=8)
        ax.set_xlabel("n samples")
        prettify_axes(ax)
        set_metric_ylim(ax, metric)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=2,
            frameon=False,
            fontsize=12,
        )

    fig.suptitle(
        f"{property_name.capitalize()} estimator validation\n"
        "Mean over circuits after seed-averaging within each circuit; shaded bands = std",
        fontsize=20,
        y=0.975,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.80])

    fig.savefig(output_dir / f"estimator_validation_{property_name}_dashboard.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"estimator_validation_{property_name}_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Tradeoff plots: normalized error vs runtime
# ---------------------------------------------------------------------

def plot_tradeoff(plot_df: pd.DataFrame, property_name: str, output_dir: Path, dpi: int) -> None:
    sub_err = plot_df.loc[
        (plot_df["property"] == property_name) &
        (plot_df["metric"] == "Normalized Error")
    ].sort_values("n_samples")

    sub_rt = plot_df.loc[
        (plot_df["property"] == property_name) &
        (plot_df["metric"] == "Runtime (s)")
    ].sort_values("n_samples")

    if sub_err.empty or sub_rt.empty:
        return

    merged = sub_err.merge(
        sub_rt,
        on=["property", "n_samples"],
        suffixes=("_err", "_rt"),
    )

    color = "#1f77b4" if property_name == "entanglement" else "#ff7f0e"

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    x = merged["mean_rt"].to_numpy(dtype=float)
    y = merged["mean_err"].to_numpy(dtype=float)
    xstd = merged["std_rt"].to_numpy(dtype=float)
    ystd = merged["std_err"].to_numpy(dtype=float)
    ns = merged["n_samples"].to_list()

    ax.plot(x, y, marker="o", linewidth=2.2, color=color)

    # horizontal and vertical std bars
    ax.errorbar(
        x, y,
        xerr=xstd,
        yerr=ystd,
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        alpha=0.6,
        capsize=4,
    )

    for xi, yi, n in zip(x, y, ns):
        ax.annotate(
            f"n={n}",
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
        )

    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Normalized Error")
    ax.set_title(f"{property_name.capitalize()}: normalized error vs runtime")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=max(0.0, ymin))
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(left=max(0.0, xmin))

    fig.tight_layout()
    fig.savefig(output_dir / f"tradeoff_normalized_error_vs_runtime_{property_name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"tradeoff_normalized_error_vs_runtime_{property_name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use(args.style)
    except OSError:
        pass

    seed_averaged_per_circuit = load_seed_averaged_per_circuit(
        per_circuit_csv=args.per_circuit_csv,
        per_run_csv=args.per_run_csv,
    )

    seed_averaged_per_circuit.to_csv(
        args.output_dir / "seed_averaged_per_circuit_metrics.csv",
        index=False,
    )

    plot_summary = build_plot_summary(seed_averaged_per_circuit)
    plot_summary.to_csv(args.output_dir / "paper_metric_summary.csv", index=False)

    plot_compact_dashboard(plot_summary, args.output_dir, args.dpi)

    for prop in PROPERTIES:
        plot_property_dashboard(plot_summary, prop, args.output_dir, args.dpi)
        plot_tradeoff(plot_summary, prop, args.output_dir, args.dpi)

    print(f"Saved seed-averaged per-circuit table to: {args.output_dir / 'seed_averaged_per_circuit_metrics.csv'}")
    print(f"Saved paper metric summary to: {args.output_dir / 'paper_metric_summary.csv'}")
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()