#!/usr/bin/env python3
"""
script_plot_estimator_validation.py

Paper-style plots for the FRACTION-BASED Monte Carlo Owen estimator validation.

This script reads already-computed CSVs from the fraction experiment:
    per_run_rows.csv
    per_circuit_summary.csv
    global_summary.csv

It creates:
    1. A compact dashboard where MAE and RMSE are replaced by
       Magic Pareto and Entanglement Pareto plots.
    2. Separate normalized-error-vs-runtime Pareto plots for magic and entanglement.

Expected fraction columns:
    sample_frac      e.g. 0.3
    sample_percent   e.g. 30

If only n_samples exists, it is treated as sample_percent for compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_BASE_DIR = PROJECT_ROOT / "results" / "estimator_validation_15"

PROPERTIES = ["magic", "entanglement"]

COLOR_MAP = {
    "magic": "#ff7f0e",
    "entanglement": "#1f77b4",
}

METRIC_CANDIDATES: Dict[str, List[str]] = {
    "Max Absolute Error": ["gate_max_ae_mean", "gate_max_ae"],
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

    p.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing per_run_rows.csv and per_circuit_summary.csv.",
    )

    p.add_argument(
        "--per-circuit-csv",
        type=Path,
        default=None,
        help="Optional explicit path to per_circuit_summary.csv.",
    )

    p.add_argument(
        "--per-run-csv",
        type=Path,
        default=None,
        help="Optional explicit path to per_run_rows.csv.",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to BASE_DIR/paper_plots_fraction.",
    )

    p.add_argument("--dpi", type=int, default=300)

    p.add_argument(
        "--style",
        type=str,
        default="seaborn-v0_8-whitegrid",
    )

    return p.parse_args()


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def normalize_fraction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sample_percent" not in df.columns:
        if "n_samples" in df.columns:
            df["sample_percent"] = df["n_samples"].astype(int)
        elif "n" in df.columns:
            df["sample_percent"] = df["n"].astype(int)
        else:
            raise ValueError(
                "Could not find sample_percent, n_samples, or n column."
            )

    if "sample_frac" not in df.columns:
        df["sample_frac"] = df["sample_percent"].astype(float) / 100.0

    if "label" not in df.columns:
        df["label"] = df["sample_percent"].astype(int).astype(str) + "%"

    if "n_samples" not in df.columns:
        df["n_samples"] = df["sample_percent"].astype(int)

    return df


def percent_label(percent: int) -> str:
    return f"{int(percent)}%"


def set_percent_xaxis(ax: plt.Axes, percents: List[int]) -> None:
    percents = sorted({int(p) for p in percents})
    ax.set_xticks(percents)
    ax.set_xticklabels([percent_label(p) for p in percents])


def pick_existing_column(df: pd.DataFrame, candidates: List[str], metric_name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find a source column for metric '{metric_name}'.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def prettify_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)


def set_metric_ylim(ax: plt.Axes, metric: str) -> None:
    if metric == "Spearman rho":
        ax.set_ylim(0.0, 1.02)
    else:
        ymin, _ = ax.get_ylim()
        ax.set_ylim(bottom=max(0.0, ymin))


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_per_run_seed_averaged(per_run_csv: Path) -> pd.DataFrame:
    if not per_run_csv.exists():
        raise FileNotFoundError(f"Missing per_run_rows.csv: {per_run_csv}")

    df = pd.read_csv(per_run_csv)
    df = normalize_fraction_columns(df)

    required = {"benchmark_id", "property", "sample_percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{per_run_csv} is missing required columns: {sorted(missing)}"
        )

    group_cols = ["benchmark_id", "property", "sample_frac", "sample_percent"]
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
    Build one row per:
        benchmark_id, family_role, property, sample_percent

    Seed-averaged within circuit.
    """
    per_run_seed_avg = load_per_run_seed_averaged(per_run_csv)

    if not per_circuit_csv.exists():
        return per_run_seed_avg

    per_circuit = pd.read_csv(per_circuit_csv)
    per_circuit = normalize_fraction_columns(per_circuit)

    required = {"benchmark_id", "property", "sample_percent"}
    missing = required - set(per_circuit.columns)
    if missing:
        raise ValueError(
            f"{per_circuit_csv} is missing required columns: {sorted(missing)}"
        )

    merge_keys = ["benchmark_id", "property", "sample_frac", "sample_percent"]

    if "family_role" in per_circuit.columns and "family_role" in per_run_seed_avg.columns:
        merge_keys = ["benchmark_id", "family_role", "property", "sample_frac", "sample_percent"]

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
    Build one row per:
        property, sample_percent, metric

    Aggregation:
        1. already seed-averaged inside each circuit
        2. now mean/std across circuits
    """
    rows = []

    percents = sorted(seed_averaged_per_circuit["sample_percent"].unique())

    for metric_label, candidates in METRIC_CANDIDATES.items():
        source_col = pick_existing_column(
            seed_averaged_per_circuit,
            candidates,
            metric_label,
        )

        for prop in sorted(seed_averaged_per_circuit["property"].unique()):
            sub_prop = seed_averaged_per_circuit.loc[
                seed_averaged_per_circuit["property"] == prop
            ].copy()

            for percent in percents:
                sub = sub_prop.loc[
                    sub_prop["sample_percent"] == percent,
                    source_col,
                ].dropna()

                n_circuits = int(len(sub))

                if n_circuits == 0:
                    continue

                mean = float(sub.mean())
                std = float(sub.std(ddof=1)) if n_circuits > 1 else 0.0

                rows.append({
                    "property": prop,
                    "sample_percent": int(percent),
                    "sample_frac": float(percent) / 100.0,
                    "label": percent_label(int(percent)),
                    "metric": metric_label,
                    "source_column": source_col,
                    "n_circuits": n_circuits,
                    "mean": mean,
                    "std": std,
                    "lower_std": mean - std,
                    "upper_std": mean + std,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Metric panels
# ---------------------------------------------------------------------

def plot_metric_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    metric: str,
    properties: List[str],
    sample_percents: List[int],
) -> None:
    sub = plot_df.loc[
        (plot_df["metric"] == metric)
        & (plot_df["property"].isin(properties))
    ].copy()

    if sub.empty:
        ax.set_visible(False)
        return

    for prop in properties:
        s = sub.loc[sub["property"] == prop].sort_values("sample_percent")

        if s.empty:
            continue

        x = s["sample_percent"].to_numpy(dtype=float)
        y = s["mean"].to_numpy(dtype=float)
        std = s["std"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=prop,
            color=COLOR_MAP.get(prop),
        )

        if metric != "Max Absolute Error":
            ax.fill_between(
                x,
                y - std,
                y + std,
                alpha=0.18,
                color=COLOR_MAP.get(prop),
            )

    ax.set_title(metric, fontsize=13, pad=8)
    ax.set_xlabel("sampled Owen fraction")
    set_percent_xaxis(ax, sample_percents)
    prettify_axes(ax)
    set_metric_ylim(ax, metric)


# ---------------------------------------------------------------------
# Pareto panels
# ---------------------------------------------------------------------

def get_tradeoff_df(plot_df: pd.DataFrame, property_name: str) -> pd.DataFrame:
    sub_err = plot_df.loc[
        (plot_df["property"] == property_name)
        & (plot_df["metric"] == "Normalized Error")
    ].sort_values("sample_percent")

    sub_rt = plot_df.loc[
        (plot_df["property"] == property_name)
        & (plot_df["metric"] == "Runtime (s)")
    ].sort_values("sample_percent")

    if sub_err.empty or sub_rt.empty:
        return pd.DataFrame()

    merged = sub_err.merge(
        sub_rt,
        on=["property", "sample_percent", "sample_frac", "label"],
        suffixes=("_err", "_rt"),
    )

    return merged.sort_values("sample_percent")


def plot_tradeoff_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    property_name: str,
) -> None:
    merged = get_tradeoff_df(plot_df, property_name)

    if merged.empty:
        ax.set_visible(False)
        return

    color = COLOR_MAP.get(property_name)

    x = merged["mean_rt"].to_numpy(dtype=float)
    y = merged["mean_err"].to_numpy(dtype=float)
    xstd = merged["std_rt"].fillna(0.0).to_numpy(dtype=float)
    ystd = merged["std_err"].fillna(0.0).to_numpy(dtype=float)
    labels = merged["label"].to_list()

    ax.plot(x, y, marker="o", linewidth=2.2, color=color)

    ax.errorbar(
        x,
        y,
        xerr=xstd,
        yerr=ystd,
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        alpha=0.6,
        capsize=4,
    )

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(
            lab,
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
        )

    ax.set_xlabel("runtime (s)")
    ax.set_ylabel("normalized error")
    ax.set_title(f"{property_name.capitalize()} Pareto", fontsize=13, pad=8)
    prettify_axes(ax)

    ymin, _ = ax.get_ylim()
    xmin, _ = ax.get_xlim()
    ax.set_ylim(bottom=max(0.0, ymin))
    ax.set_xlim(left=max(0.0, xmin))


def plot_tradeoff_single(
    plot_df: pd.DataFrame,
    property_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    merged = get_tradeoff_df(plot_df, property_name)

    if merged.empty:
        return

    color = COLOR_MAP.get(property_name)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    x = merged["mean_rt"].to_numpy(dtype=float)
    y = merged["mean_err"].to_numpy(dtype=float)
    xstd = merged["std_rt"].fillna(0.0).to_numpy(dtype=float)
    ystd = merged["std_err"].fillna(0.0).to_numpy(dtype=float)
    labels = merged["label"].to_list()

    ax.plot(x, y, marker="o", linewidth=2.2, color=color)

    ax.errorbar(
        x,
        y,
        xerr=xstd,
        yerr=ystd,
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        alpha=0.6,
        capsize=4,
    )

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(
            lab,
            (xi, yi),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=11,
        )

    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Normalized Error")
    ax.set_title(f"{property_name.capitalize()}: normalized error vs runtime")
    prettify_axes(ax)

    ymin, _ = ax.get_ylim()
    xmin, _ = ax.get_xlim()
    ax.set_ylim(bottom=max(0.0, ymin))
    ax.set_xlim(left=max(0.0, xmin))

    fig.tight_layout()

    fig.savefig(
        output_dir / f"tradeoff_normalized_error_vs_runtime_{property_name}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"tradeoff_normalized_error_vs_runtime_{property_name}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------

def plot_compact_dashboard_with_pareto(
    plot_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    sample_percents: List[int],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.flatten()

    # Replaces MAE and RMSE with Pareto plots.
    plot_tradeoff_panel(axes[0], plot_df, "magic")
    plot_metric_panel(axes[1], plot_df, "Max Absolute Error", PROPERTIES, sample_percents)
    plot_tradeoff_panel(axes[2], plot_df, "entanglement")

    plot_metric_panel(axes[3], plot_df, "Normalized Error", PROPERTIES, sample_percents)
    plot_metric_panel(axes[4], plot_df, "Spearman rho", PROPERTIES, sample_percents)
    plot_metric_panel(axes[5], plot_df, "Runtime (s)", PROPERTIES, sample_percents)

    handles, labels = axes[1].get_legend_handles_labels()

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
        "Fraction-based sampling; Pareto panels show normalized error vs runtime",
        fontsize=20,
        y=0.975,
    )

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.80])

    fig.savefig(
        output_dir / "estimator_validation_fraction_dashboard_with_pareto.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "estimator_validation_fraction_dashboard_with_pareto.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    per_circuit_csv = args.per_circuit_csv or (base_dir / "per_circuit_summary.csv")
    per_run_csv = args.per_run_csv or (base_dir / "per_run_rows.csv")
    output_dir = args.output_dir or (base_dir / "paper_plots_fraction")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use(args.style)
    except OSError:
        pass

    seed_averaged_per_circuit = load_seed_averaged_per_circuit(
        per_circuit_csv=per_circuit_csv,
        per_run_csv=per_run_csv,
    )

    seed_averaged_per_circuit = normalize_fraction_columns(seed_averaged_per_circuit)

    seed_averaged_per_circuit.to_csv(
        output_dir / "seed_averaged_per_circuit_metrics.csv",
        index=False,
    )

    plot_summary = build_plot_summary(seed_averaged_per_circuit)

    plot_summary.to_csv(
        output_dir / "paper_metric_summary.csv",
        index=False,
    )

    sample_percents = sorted(plot_summary["sample_percent"].unique().astype(int).tolist())

    plot_compact_dashboard_with_pareto(
        plot_df=plot_summary,
        output_dir=output_dir,
        dpi=args.dpi,
        sample_percents=sample_percents,
    )

    for prop in PROPERTIES:
        plot_tradeoff_single(
            plot_df=plot_summary,
            property_name=prop,
            output_dir=output_dir,
            dpi=args.dpi,
        )

    print(f"Read per-circuit CSV from: {per_circuit_csv}")
    print(f"Read per-run CSV from:     {per_run_csv}")
    print(f"Saved seed-averaged table to: {output_dir / 'seed_averaged_per_circuit_metrics.csv'}")
    print(f"Saved paper metric summary to: {output_dir / 'paper_metric_summary.csv'}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
