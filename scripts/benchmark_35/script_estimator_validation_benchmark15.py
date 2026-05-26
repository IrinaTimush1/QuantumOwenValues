#!/usr/bin/env python3
"""
script_estimator_validation_benchmark15.py

Estimator validation for the Monte Carlo Owen estimator on the exact-Owen
benchmark subset.

This version uses positive Owen sampling fractions instead of absolute n values.

Example:
    --sample-fracs 0.3 0.5 0.7 0.9

These are passed directly to qshaptools as:

    owen_sample_frac = 0.3
    owen_sample_frac = 0.5
    ...

This means the validation is percentage-based rather than using unrealistic
absolute sample counts such as 100, 300, 500, 1000.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from script_exact_owen_benchmark15 import (
    BENCHMARK_PKL,
    GATE_SPEC_CSV,
    ROOT,
    SUMMARY_CSV,
    _property_value_fun_factory,
    aggregate_group_scores,
    build_benchmark_circuits,
    get_exact_ids,
    load_benchmark_pickle,
    load_gate_spec,
    load_summary,
    run_exact_owen_for_circuit,
)

from qshaptools.qowen import QuantumOwenValues  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ESTIMATOR_OUTPUT_DIR = ROOT / "results" / "estimator_validation_15"
EXACT_RESULTS_JSON = ROOT / "results" / "exact_owen_15" / "exact_owen_results.json"

DEFAULT_SAMPLE_FRACS = [0.30, 0.50, 0.70, 0.90]
DEFAULT_N_SEEDS = 10

EPS = 1e-12
PROPERTIES = ["magic", "entanglement"]
FAMILY_ORDER = {"magic_axis": 0, "entanglement_axis": 1, "interior": 2}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--benchmark-pkl", type=Path, default=BENCHMARK_PKL)
    p.add_argument("--gate-spec-csv", type=Path, default=GATE_SPEC_CSV)
    p.add_argument("--exact-results-json", type=Path, default=EXACT_RESULTS_JSON)
    p.add_argument("--output-dir", type=Path, default=ESTIMATOR_OUTPUT_DIR)

    p.add_argument(
        "--sample-fracs",
        type=float,
        nargs="+",
        default=DEFAULT_SAMPLE_FRACS,
        help=(
            "Positive Owen sampling fractions to test. "
            "Use decimals, e.g. --sample-fracs 0.3 0.5 0.7 0.9. "
            "These are passed directly as positive owen_sample_frac values."
        ),
    )

    p.add_argument(
        "--n-seeds",
        "--seeds-per-fraction",
        dest="n_seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of random seeds per (circuit, property, fraction) cell.",
    )

    p.add_argument("--allow-order-fallback", action="store_true")

    p.add_argument(
        "--force-recompute-exact",
        action="store_true",
        help="Ignore exact_owen_results.json and recompute exact Owen.",
    )

    p.add_argument(
        "--rec-group-mae-threshold",
        type=float,
        default=0.02,
        help="Recommendation threshold on mean group MAE.",
    )

    p.add_argument(
        "--rec-top-group-agree-threshold",
        type=float,
        default=0.95,
        help="Recommendation threshold on top-group agreement rate.",
    )

    p.add_argument(
        "--benchmark-ids",
        nargs="+",
        default=None,
        help="Optional explicit benchmark IDs to validate, e.g. I11 I22 I33 I44 I55.",
    )

    p.add_argument(
        "--expected-count",
        type=int,
        default=15,
        help="Expected number of exact circuits when --benchmark-ids is not provided. Use -1 to disable.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_sample_fracs(sample_fracs: Sequence[float]) -> List[float]:
    out = []
    for frac in sample_fracs:
        frac = float(frac)
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                f"Invalid sample fraction {frac}. Use values in (0, 1], "
                f"for example 0.3 0.5 0.7 0.9."
            )
        out.append(frac)
    return out


def frac_to_percent(sample_frac: float) -> int:
    return int(round(100.0 * float(sample_frac)))


def percent_label(percent: int) -> str:
    return f"{int(percent)}%"


def set_percent_xaxis(ax, observed_values: Sequence[int] | None = None) -> None:
    if observed_values is None:
        ticks = [30, 50, 70, 90]
    else:
        ticks = sorted({int(x) for x in observed_values})

    ax.set_xticks(ticks)
    ax.set_xticklabels([percent_label(t) for t in ticks])


def get_exact_ids_compat(
    summary_df: pd.DataFrame,
    benchmark_ids: Sequence[str] | None,
    expected_count: int | None,
) -> List[str]:
    """
    Supports both versions of get_exact_ids:
    - old version: get_exact_ids(summary_df)
    - patched version: get_exact_ids(summary_df, benchmark_ids=..., expected_count=...)
    """
    try:
        return get_exact_ids(
            summary_df,
            benchmark_ids=benchmark_ids,
            expected_count=expected_count,
        )
    except TypeError:
        if benchmark_ids is not None and len(benchmark_ids) > 0:
            available = set(summary_df["benchmark_id"].astype(str))
            ids = [str(x) for x in benchmark_ids]
            missing = [x for x in ids if x not in available]
            if missing:
                raise ValueError(f"Requested benchmark IDs not found: {missing}")
            return ids

        ids = get_exact_ids(summary_df)

        if expected_count is not None and len(ids) != expected_count:
            raise ValueError(
                f"Expected {expected_count} exact circuits, found {len(ids)}: {ids}"
            )

        return ids


# ---------------------------------------------------------------------------
# Sampled Owen call
# ---------------------------------------------------------------------------

def run_sampled_owen(
    item,
    spec: Dict[str, Any],
    property_name: str,
    sample_frac: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Run one sampled QuantumOwenValues call.

    Important:
        positive owen_sample_frac = fraction mode
        negative owen_sample_frac = absolute sample-count mode

    This script intentionally uses positive fractions.
    """
    partition = spec["partition"]
    locked = spec["locked"]
    labels = spec["partition_labels"]
    value_fun = _property_value_fun_factory(property_name)

    t0 = time.perf_counter()

    qov = QuantumOwenValues(
        qc=item.qc,
        partition=partition,
        value_fun=value_fun,
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=locked,
        owen_sample_frac=float(sample_frac),
        owen_sample_reps=1,
        evaluate_value_only_once=False,
        owen_sample_seed=int(seed),
        name=f"{item.benchmark_id}_{property_name}_frac{sample_frac:.2f}_seed{seed}",
        silent=True,
    )

    phi = qov.run()

    t1 = time.perf_counter()

    phi_dict = {int(k): float(v) for k, v in phi.items()}
    group = aggregate_group_scores(phi_dict, partition, labels)

    return {
        "phi": phi_dict,
        "group": group,
        "runtime_s": float(t1 - t0),
    }


# ---------------------------------------------------------------------------
# Error / agreement metrics
# ---------------------------------------------------------------------------

def compute_per_gate_errors(
    phi_samp: Dict[int, float],
    phi_exact: Dict[int, float],
) -> Dict[str, float]:
    keys = sorted(phi_exact.keys())

    samp_arr = np.array([phi_samp[k] for k in keys], dtype=float)
    exact_arr = np.array([phi_exact[k] for k in keys], dtype=float)

    abs_err = np.abs(samp_arr - exact_arr)
    sq_err = (samp_arr - exact_arr) ** 2
    abs_exact = np.abs(exact_arr)

    total_mag = float(np.sum(abs_exact))
    max_mag = float(np.max(abs_exact)) if abs_exact.size > 0 else 0.0

    if samp_arr.size >= 2 and np.std(exact_arr) > EPS:
        rho, _ = spearmanr(samp_arr, exact_arr)
        if np.isnan(rho):
            rho = 1.0
    else:
        rho = 1.0

    return {
        "gate_mae": float(np.mean(abs_err)),
        "gate_max_ae": float(np.max(abs_err)) if abs_err.size else 0.0,
        "gate_rmse": float(np.sqrt(np.mean(sq_err))),
        "gate_total_abs_exact": total_mag,
        "gate_max_abs_exact": max_mag,
        "gate_normalized_mae_total": float(np.mean(abs_err) / (total_mag + EPS)),
        "gate_normalized_mae_max": float(np.mean(abs_err) / (max_mag + EPS)),
        "gate_spearman_rho": float(rho),
    }


def compute_group_errors(
    group_samp: Dict[str, float],
    group_exact: Dict[str, float],
) -> Dict[str, Any]:
    labels = sorted(group_exact.keys())

    samp_arr = np.array([group_samp[k] for k in labels], dtype=float)
    exact_arr = np.array([group_exact[k] for k in labels], dtype=float)

    abs_err = np.abs(samp_arr - exact_arr)

    top_exact = max(group_exact, key=group_exact.get)
    top_samp = max(group_samp, key=group_samp.get)

    bottom_exact = min(group_exact, key=group_exact.get)
    bottom_samp = min(group_samp, key=group_samp.get)

    order_exact = sorted(labels, key=lambda g: group_exact[g], reverse=True)
    order_samp = sorted(labels, key=lambda g: group_samp[g], reverse=True)

    full_order_match = bool(order_exact == order_samp)

    if len(labels) >= 2 and np.std(exact_arr) > EPS:
        tau, _ = kendalltau(samp_arr, exact_arr)
        if np.isnan(tau):
            tau = 1.0
    else:
        tau = 1.0

    return {
        "group_mae": float(np.mean(abs_err)),
        "group_max_ae": float(np.max(abs_err)) if abs_err.size else 0.0,
        "group_top_exact": top_exact,
        "group_top_samp": top_samp,
        "group_top_match": bool(top_exact == top_samp),
        "group_bottom_exact": bottom_exact,
        "group_bottom_samp": bottom_samp,
        "group_bottom_match": bool(bottom_exact == bottom_samp),
        "group_full_order_match": full_order_match,
        "group_kendall_tau": float(tau),
    }


def compute_efficiency_check(
    phi_samp: Dict[int, float],
    phi_exact: Dict[int, float],
) -> Dict[str, float]:
    sum_samp = float(sum(phi_samp.values()))
    sum_exact = float(sum(phi_exact.values()))

    return {
        "sum_phi_samp": sum_samp,
        "sum_phi_exact": sum_exact,
        "efficiency_abs_err": abs(sum_samp - sum_exact),
    }


# ---------------------------------------------------------------------------
# Load exact Owen results
# ---------------------------------------------------------------------------

def load_or_compute_exact(
    benchmark_circuits: Dict[str, Any],
    gate_spec: Dict[str, Dict[str, Any]],
    exact_ids: Sequence[str],
    exact_results_json: Path,
    force_recompute: bool,
) -> Dict[str, Dict[str, Any]]:
    expected_ids = set(str(x) for x in exact_ids)

    if exact_results_json.exists() and not force_recompute:
        with open(exact_results_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            ids_in_file = set(str(r.get("benchmark_id")) for r in data)

            if expected_ids.issubset(ids_in_file):
                print(f"Loaded exact Owen results from: {exact_results_json}")
                by_id = {str(r["benchmark_id"]): r for r in data}
                return {bid: by_id[bid] for bid in exact_ids}

        print("Existing exact results JSON looked invalid/incomplete; recomputing.")

    print("Recomputing exact Owen values...")

    results = []
    for bid in exact_ids:
        print(f"  exact Owen for {bid}")
        results.append(
            run_exact_owen_for_circuit(
                item=benchmark_circuits[bid],
                spec=gate_spec[bid],
            )
        )

    exact_results_json.parent.mkdir(parents=True, exist_ok=True)

    with open(exact_results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved exact Owen results to: {exact_results_json}")

    return {str(r["benchmark_id"]): r for r in results}


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def recommend_fraction(
    global_summary: pd.DataFrame,
    group_mae_threshold: float,
    top_agree_threshold: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "group_mae_threshold": group_mae_threshold,
        "top_agree_threshold": top_agree_threshold,
        "per_fraction_feasibility": [],
    }

    percent_values = sorted(global_summary["sample_percent"].unique())
    feasible: List[int] = []

    for percent in percent_values:
        sub = global_summary.loc[global_summary["sample_percent"] == percent]
        per_prop: Dict[str, Any] = {}

        for _, r in sub.iterrows():
            ok_mae = r["mean_group_mae"] <= group_mae_threshold
            ok_top = r["top_group_agree_rate"] >= top_agree_threshold

            per_prop[r["property"]] = {
                "mean_group_mae": float(r["mean_group_mae"]),
                "top_agree_rate": float(r["top_group_agree_rate"]),
                "passes_mae": bool(ok_mae),
                "passes_top_agree": bool(ok_top),
                "passes_both": bool(ok_mae and ok_top),
            }

        all_pass = all(v["passes_both"] for v in per_prop.values())

        out["per_fraction_feasibility"].append({
            "sample_percent": int(percent),
            "sample_frac": float(percent) / 100.0,
            "label": percent_label(int(percent)),
            "per_property": per_prop,
            "all_properties_pass": bool(all_pass),
        })

        if all_pass:
            feasible.append(int(percent))

    if feasible:
        best = min(feasible)
        out["recommended_sample_percent"] = int(best)
        out["recommended_sample_frac"] = float(best) / 100.0
        out["recommended_label"] = percent_label(best)
        out["reasoning"] = (
            f"Smallest sampling fraction with mean group MAE <= {group_mae_threshold} "
            f"and top-group agreement >= {top_agree_threshold} for both properties."
        )
    else:
        scores = []
        for percent in percent_values:
            sub = global_summary.loc[global_summary["sample_percent"] == percent]
            score = float(
                sub["top_group_agree_rate"].mean()
                - sub["mean_group_mae"].mean()
            )
            scores.append((score, int(percent)))

        scores.sort(reverse=True)
        best = scores[0][1]

        out["recommended_sample_percent"] = int(best)
        out["recommended_sample_frac"] = float(best) / 100.0
        out["recommended_label"] = percent_label(best)
        out["reasoning"] = (
            "No tested fraction satisfied both thresholds for both properties. "
            "Falling back to the fraction that maximises "
            "(mean top-group agreement - mean group MAE)."
        )

    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(
    rows_df: pd.DataFrame,
    gate_rows_df: pd.DataFrame,
    group_rows_df: pd.DataFrame,
    global_summary: pd.DataFrame,
    sample_percents: Sequence[int],
    out_dir: Path,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]

        plot_convergence(
            sub,
            "gate_mae",
            prop,
            plots_dir / f"{prop}_gate_mae_vs_fraction.png",
            ylabel="gate-level MAE",
            sample_percents=sample_percents,
        )

        plot_convergence(
            sub,
            "group_mae",
            prop,
            plots_dir / f"{prop}_group_mae_vs_fraction.png",
            ylabel="group-level MAE",
            sample_percents=sample_percents,
        )

        plot_convergence(
            sub,
            "gate_rmse",
            prop,
            plots_dir / f"{prop}_gate_rmse_vs_fraction.png",
            ylabel="gate-level RMSE",
            sample_percents=sample_percents,
        )

    plot_runtime_vs_fraction(rows_df, plots_dir / "runtime_vs_fraction.png", sample_percents)

    for prop in PROPERTIES:
        plot_error_runtime_pareto(
            rows_df,
            prop,
            plots_dir / f"{prop}_error_runtime_pareto.png",
        )

    plot_top_group_agreement(
        global_summary,
        plots_dir / "top_group_agreement_vs_fraction.png",
        sample_percents,
    )

    plot_full_order_agreement(
        rows_df,
        plots_dir / "full_order_agreement_vs_fraction.png",
        sample_percents,
    )

    plot_kendall_tau(
        global_summary,
        plots_dir / "group_kendall_tau_vs_fraction.png",
        sample_percents,
    )

    plot_spearman_rho(
        rows_df,
        plots_dir / "gate_spearman_rho_vs_fraction.png",
        sample_percents,
    )

    for prop in PROPERTIES:
        for percent in sample_percents:
            plot_group_scatter(
                group_rows_df,
                prop,
                percent,
                plots_dir / f"{prop}_group_scatter_{percent}pct.png",
            )

            plot_gate_scatter(
                gate_rows_df,
                prop,
                percent,
                plots_dir / f"{prop}_gate_scatter_{percent}pct.png",
            )

    for prop in PROPERTIES:
        plot_per_circuit_heatmap(
            rows_df,
            prop,
            plots_dir / f"{prop}_per_circuit_heatmap.png",
        )

    plot_efficiency_check(
        rows_df,
        plots_dir / "efficiency_check_vs_fraction.png",
        sample_percents,
    )

    print(f"Saved plots to: {plots_dir}")


def plot_convergence(
    sub: pd.DataFrame,
    metric: str,
    prop: str,
    out_path: Path,
    ylabel: str,
    sample_percents: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    agg = (
        sub.groupby("sample_percent")[metric]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("sample_percent")
    )

    xs = agg["sample_percent"].to_numpy(dtype=float)
    means = agg["mean"].to_numpy(dtype=float)
    stds = agg["std"].fillna(0.0).to_numpy(dtype=float)

    ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=4, label="mean ± std")

    ax.fill_between(
        xs,
        agg["min"].to_numpy(dtype=float),
        agg["max"].to_numpy(dtype=float),
        alpha=0.15,
        label="min–max",
    )

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{prop}: {ylabel} vs sampled fraction")
    set_percent_xaxis(ax, sample_percents)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_runtime_vs_fraction(
    rows_df: pd.DataFrame,
    out_path: Path,
    sample_percents: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]

        agg = (
            sub.groupby("sample_percent")["runtime_s"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("sample_percent")
        )

        ax.errorbar(
            agg["sample_percent"],
            agg["mean"],
            yerr=agg["std"].fillna(0.0),
            fmt="o-",
            capsize=4,
            label=prop,
        )

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel("runtime per call (s)")
    ax.set_title("Sampled Owen runtime vs sampled fraction")
    set_percent_xaxis(ax, sample_percents)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_runtime_pareto(
    rows_df: pd.DataFrame,
    prop: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    sub = rows_df.loc[rows_df["property"] == prop]

    agg = (
        sub.groupby("sample_percent")
        .agg(
            runtime=("runtime_s", "mean"),
            runtime_std=("runtime_s", "std"),
            err=("group_mae", "mean"),
            err_std=("group_mae", "std"),
        )
        .reset_index()
        .sort_values("sample_percent")
    )

    ax.errorbar(
        agg["runtime"],
        agg["err"],
        xerr=agg["runtime_std"].fillna(0.0),
        yerr=agg["err_std"].fillna(0.0),
        fmt="o-",
        capsize=4,
    )

    for _, r in agg.iterrows():
        ax.annotate(
            percent_label(int(r["sample_percent"])),
            (r["runtime"], r["err"]),
            textcoords="offset points",
            xytext=(8, 8),
        )

    ax.set_xlabel("mean runtime (s)")
    ax.set_ylabel("mean group MAE")
    ax.set_title(f"{prop}: error-runtime Pareto")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_group_agreement(
    global_summary: pd.DataFrame,
    out_path: Path,
    sample_percents: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for prop in PROPERTIES:
        sub = (
            global_summary.loc[global_summary["property"] == prop]
            .sort_values("sample_percent")
        )

        ax.plot(
            sub["sample_percent"],
            sub["top_group_agree_rate"],
            "o-",
            label=prop,
        )

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel("top-group agreement rate")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Top-group agreement: sampled vs exact")
    set_percent_xaxis(ax, sample_percents)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_full_order_agreement(
    rows_df: pd.DataFrame,
    out_path: Path,
    sample_percents: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]

        agg = (
            sub.groupby("sample_percent")["group_full_order_match"]
            .mean()
            .reset_index()
            .sort_values("sample_percent")
        )

        ax.plot(
            agg["sample_percent"],
            agg["group_full_order_match"],
            "o-",
            label=prop,
        )

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel("full group-order agreement rate")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Full group-ordering agreement vs sampled fraction")
    set_percent_xaxis(ax, sample_percents)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_kendall_tau(
    global_summary: pd.DataFrame,
    out_path: Path,
    sample_percents: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for prop in PROPERTIES:
        sub = (
            global_summary.loc[global_summary["property"] == prop]
            .sort_values("sample_percent")
        )

        ax.plot(
            sub["sample_percent"],
            sub["mean_kendall_tau"],
            "o-",
            label=prop,
        )

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel("mean Kendall tau")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Group-score rank agreement")
    set_percent_xaxis(ax, sample_percents)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spearman_rho(
    rows_df: pd.DataFrame,
    out_path: Path,
    sample_percents: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]

        agg = (
            sub.groupby("sample_percent")["gate_spearman_rho"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("sample_percent")
        )

        ax.errorbar(
            agg["sample_percent"],
            agg["mean"],
            yerr=agg["std"].fillna(0.0),
            fmt="o-",
            capsize=4,
            label=prop,
        )

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel("gate-level Spearman rho")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Gate-level rank correlation vs sampled fraction")
    set_percent_xaxis(ax, sample_percents)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_group_scatter(
    group_rows_df: pd.DataFrame,
    prop: str,
    sample_percent: int,
    out_path: Path,
) -> None:
    sub = group_rows_df.loc[
        (group_rows_df["property"] == prop)
        & (group_rows_df["sample_percent"] == sample_percent)
    ]

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(sub["score_exact"], sub["score_samp"], alpha=0.5, s=25)

    mn = float(min(sub["score_exact"].min(), sub["score_samp"].min()))
    mx = float(max(sub["score_exact"].max(), sub["score_samp"].max()))
    pad = 0.05 * max(mx - mn, 1e-6)

    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls="--", color="gray")

    ax.set_xlabel("exact group score")
    ax.set_ylabel("sampled group score")
    ax.set_title(
        f"{prop}: group-level sampled vs exact at {sample_percent}%"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gate_scatter(
    gate_rows_df: pd.DataFrame,
    prop: str,
    sample_percent: int,
    out_path: Path,
) -> None:
    sub = gate_rows_df.loc[
        (gate_rows_df["property"] == prop)
        & (gate_rows_df["sample_percent"] == sample_percent)
    ]

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(sub["phi_exact"], sub["phi_samp"], alpha=0.35, s=15)

    mn = float(min(sub["phi_exact"].min(), sub["phi_samp"].min()))
    mx = float(max(sub["phi_exact"].max(), sub["phi_samp"].max()))
    pad = 0.05 * max(mx - mn, 1e-6)

    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls="--", color="gray")

    ax.set_xlabel("exact phi")
    ax.set_ylabel("sampled phi")
    ax.set_title(
        f"{prop}: gate-level sampled vs exact at {sample_percent}%"
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_circuit_heatmap(
    rows_df: pd.DataFrame,
    prop: str,
    out_path: Path,
) -> None:
    sub = rows_df.loc[rows_df["property"] == prop]

    pivot = (
        sub.groupby(["benchmark_id", "sample_percent"])["group_mae"]
        .mean()
        .unstack("sample_percent")
    )

    families = (
        sub.drop_duplicates("benchmark_id")
        .set_index("benchmark_id")["family_role"]
    )

    order_key = [FAMILY_ORDER.get(families.loc[bid], 99) for bid in pivot.index]
    pivot = pivot.iloc[np.argsort(order_key)]

    fig, ax = plt.subplots(figsize=(7, max(5, 0.35 * len(pivot))))

    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([percent_label(int(c)) for c in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel("benchmark_id")
    ax.set_title(f"{prop}: group MAE heatmap")

    plt.colorbar(im, ax=ax, label="group MAE")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_efficiency_check(
    rows_df: pd.DataFrame,
    out_path: Path,
    sample_percents: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]

        agg = (
            sub.groupby("sample_percent")["efficiency_abs_err"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("sample_percent")
        )

        ax.errorbar(
            agg["sample_percent"],
            agg["mean"],
            yerr=agg["std"].fillna(0.0),
            fmt="o-",
            capsize=4,
            label=prop,
        )

    ax.set_xlabel("sampled Owen fraction")
    ax.set_ylabel("|sum(phi_samp) - sum(phi_exact)|")
    ax.set_title("Efficiency deviation vs sampled fraction")
    set_percent_xaxis(ax, sample_percents)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_fracs = validate_sample_fracs(args.sample_fracs)
    sample_percents = [frac_to_percent(f) for f in sample_fracs]

    summary_df = load_summary(args.summary_csv)

    expected_count = None if args.expected_count < 0 else args.expected_count

    exact_ids = get_exact_ids_compat(
        summary_df=summary_df,
        benchmark_ids=args.benchmark_ids,
        expected_count=expected_count,
    )

    benchmark_map = load_benchmark_pickle(
        args.benchmark_pkl,
        summary_df,
        allow_order_fallback=args.allow_order_fallback,
    )

    benchmark_circuits = build_benchmark_circuits(
        summary_df,
        benchmark_map,
        exact_ids,
    )

    gate_spec = load_gate_spec(
        args.gate_spec_csv,
        benchmark_circuits,
    )

    exact_by_id = load_or_compute_exact(
        benchmark_circuits=benchmark_circuits,
        gate_spec=gate_spec,
        exact_ids=exact_ids,
        exact_results_json=args.exact_results_json,
        force_recompute=args.force_recompute_exact,
    )

    seeds = list(range(int(args.n_seeds)))

    total = (
        len(exact_ids)
        * len(PROPERTIES)
        * len(sample_fracs)
        * len(seeds)
    )

    frac_labels = [percent_label(p) for p in sample_percents]

    print(
        f"\nRunning {total} sampled Owen configurations "
        f"(circuits={len(exact_ids)}, properties={len(PROPERTIES)}, "
        f"sample_fracs={frac_labels}, seeds={len(seeds)})."
    )

    rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    t_start = time.perf_counter()
    config_idx = 0

    for bid in exact_ids:
        item = benchmark_circuits[bid]
        spec = gate_spec[bid]
        exact_res = exact_by_id[bid]
        family_role = item.summary_row["family_role"]

        for prop in PROPERTIES:
            phi_exact = {
                int(k): float(v)
                for k, v in exact_res[f"phi_{prop}"].items()
            }

            group_exact = {
                str(k): float(v)
                for k, v in exact_res[f"group_{prop}"].items()
            }

            for sample_frac in sample_fracs:
                sample_percent = frac_to_percent(sample_frac)

                for seed in seeds:
                    config_idx += 1

                    samp = run_sampled_owen(
                        item=item,
                        spec=spec,
                        property_name=prop,
                        sample_frac=sample_frac,
                        seed=seed,
                    )

                    gate_errs = compute_per_gate_errors(
                        samp["phi"],
                        phi_exact,
                    )

                    group_errs = compute_group_errors(
                        samp["group"],
                        group_exact,
                    )

                    eff = compute_efficiency_check(
                        samp["phi"],
                        phi_exact,
                    )

                    rows.append({
                        "benchmark_id": bid,
                        "family_role": family_role,
                        "property": prop,
                        "sample_frac": float(sample_frac),
                        "sample_percent": int(sample_percent),
                        "n_samples": int(sample_percent),  # compatibility column
                        "seed": int(seed),
                        "runtime_s": samp["runtime_s"],
                        **gate_errs,
                        **group_errs,
                        **eff,
                    })

                    for k, v_samp in samp["phi"].items():
                        gate_rows.append({
                            "benchmark_id": bid,
                            "family_role": family_role,
                            "property": prop,
                            "sample_frac": float(sample_frac),
                            "sample_percent": int(sample_percent),
                            "n_samples": int(sample_percent),  # compatibility column
                            "seed": int(seed),
                            "gate_idx": int(k),
                            "phi_exact": float(phi_exact[int(k)]),
                            "phi_samp": float(v_samp),
                            "abs_err": abs(float(v_samp) - float(phi_exact[int(k)])),
                        })

                    for g, s_samp in samp["group"].items():
                        group_rows.append({
                            "benchmark_id": bid,
                            "family_role": family_role,
                            "property": prop,
                            "sample_frac": float(sample_frac),
                            "sample_percent": int(sample_percent),
                            "n_samples": int(sample_percent),  # compatibility column
                            "seed": int(seed),
                            "group": str(g),
                            "score_exact": float(group_exact[str(g)]),
                            "score_samp": float(s_samp),
                            "abs_err": abs(float(s_samp) - float(group_exact[str(g)])),
                        })

                    if config_idx % 25 == 0 or config_idx == total:
                        elapsed = time.perf_counter() - t_start
                        rate = config_idx / max(elapsed, 1e-9)
                        eta = (total - config_idx) / max(rate, 1e-9)

                        print(
                            f"  [{config_idx}/{total}] {bid} {prop} "
                            f"frac={sample_percent}% seed={seed} | "
                            f"{elapsed:.1f}s elapsed, ETA {eta:.0f}s"
                        )

    print(f"\nTotal wall-clock time: {time.perf_counter() - t_start:.1f}s")

    rows_df = pd.DataFrame(rows)
    gate_rows_df = pd.DataFrame(gate_rows)
    group_rows_df = pd.DataFrame(group_rows)

    rows_df.to_csv(out_dir / "per_run_rows.csv", index=False)
    gate_rows_df.to_csv(out_dir / "per_gate_rows.csv", index=False)
    group_rows_df.to_csv(out_dir / "per_group_rows.csv", index=False)

    per_circuit = (
        rows_df.groupby(
            ["benchmark_id", "family_role", "property", "sample_frac", "sample_percent"]
        )
        .agg(
            runtime_mean=("runtime_s", "mean"),
            runtime_std=("runtime_s", "std"),
            gate_mae_mean=("gate_mae", "mean"),
            gate_mae_std=("gate_mae", "std"),
            gate_max_ae_mean=("gate_max_ae", "mean"),
            gate_rmse_mean=("gate_rmse", "mean"),
            gate_spearman_rho_mean=("gate_spearman_rho", "mean"),
            group_mae_mean=("group_mae", "mean"),
            group_mae_std=("group_mae", "std"),
            group_max_ae_mean=("group_max_ae", "mean"),
            top_group_agree_rate=("group_top_match", "mean"),
            bottom_group_agree_rate=("group_bottom_match", "mean"),
            full_order_agree_rate=("group_full_order_match", "mean"),
            kendall_tau_mean=("group_kendall_tau", "mean"),
            efficiency_abs_err_mean=("efficiency_abs_err", "mean"),
        )
        .reset_index()
    )

    per_circuit.to_csv(out_dir / "per_circuit_summary.csv", index=False)

    global_rows: List[Dict[str, Any]] = []

    for (prop, sample_frac, sample_percent), sub in rows_df.groupby(
        ["property", "sample_frac", "sample_percent"]
    ):
        global_rows.append({
            "property": prop,
            "sample_frac": float(sample_frac),
            "sample_percent": int(sample_percent),
            "label": percent_label(int(sample_percent)),
            "n_circuits": int(sub["benchmark_id"].nunique()),
            "n_seeds": int(sub["seed"].nunique()),
            "mean_gate_mae": float(sub["gate_mae"].mean()),
            "std_gate_mae": float(sub["gate_mae"].std()),
            "mean_gate_max_ae": float(sub["gate_max_ae"].mean()),
            "mean_gate_rmse": float(sub["gate_rmse"].mean()),
            "mean_gate_norm_mae_total": float(sub["gate_normalized_mae_total"].mean()),
            "mean_gate_norm_mae_max": float(sub["gate_normalized_mae_max"].mean()),
            "mean_gate_spearman_rho": float(sub["gate_spearman_rho"].mean()),
            "mean_group_mae": float(sub["group_mae"].mean()),
            "std_group_mae": float(sub["group_mae"].std()),
            "mean_group_max_ae": float(sub["group_max_ae"].mean()),
            "top_group_agree_rate": float(sub["group_top_match"].mean()),
            "bottom_group_agree_rate": float(sub["group_bottom_match"].mean()),
            "full_order_agree_rate": float(sub["group_full_order_match"].mean()),
            "mean_kendall_tau": float(sub["group_kendall_tau"].mean()),
            "mean_efficiency_abs_err": float(sub["efficiency_abs_err"].mean()),
            "mean_runtime_s": float(sub["runtime_s"].mean()),
            "std_runtime_s": float(sub["runtime_s"].std()),
            "median_runtime_s": float(sub["runtime_s"].median()),
        })

    global_summary = (
        pd.DataFrame(global_rows)
        .sort_values(["property", "sample_percent"])
    )

    global_summary.to_csv(out_dir / "global_summary.csv", index=False)

    print("\n=== Global summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(global_summary.to_string(index=False))

    recommended = recommend_fraction(
        global_summary,
        group_mae_threshold=args.rec_group_mae_threshold,
        top_agree_threshold=args.rec_top_group_agree_threshold,
    )

    with open(out_dir / "recommended_fraction.json", "w", encoding="utf-8") as f:
        json.dump(recommended, f, indent=2)

    # Compatibility filename, so older report scripts looking for recommended_n.json still find something.
    with open(out_dir / "recommended_n.json", "w", encoding="utf-8") as f:
        json.dump(recommended, f, indent=2)

    print("\n=== Recommended sampling fraction ===")
    print(json.dumps(recommended, indent=2))

    make_plots(
        rows_df=rows_df,
        gate_rows_df=gate_rows_df,
        group_rows_df=group_rows_df,
        global_summary=global_summary,
        sample_percents=sample_percents,
        out_dir=out_dir,
    )

    print(f"\nAll outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
