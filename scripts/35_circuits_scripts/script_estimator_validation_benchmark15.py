 #!/usr/bin/env python3
"""
script_estimator_validation_benchmark15.py

Estimator validation for the Monte Carlo Owen estimator on the 15-circuit
exact-Owen benchmark subset.

For each of the 15 circuits and each property (magic, entanglement):
  for each n in N_LIST:
    for each seed in SEEDS:
      - run sampled QuantumOwenValues with owen_sample_frac = -n
      - measure wall-clock runtime
      - compute gate-level errors vs. the exact Owen values
      - compute group-level errors vs. the exact Owen values
      - compute rank agreement metrics (top-group match, full group ordering
        match, Kendall tau on group scores, Spearman rho on gate scores)
      - store raw (exact, sampled) pairs for diagnostic scatter plots
      - verify sampled efficiency (sum phi_samp ≈ sum phi_exact)

Aggregations:
  - per-run rows                (one row per (circuit, property, n, seed))
  - per-gate rows               (for gate-level scatter plots)
  - per-group rows              (for group-level scatter plots)
  - per-circuit summary         (mean/std across seeds, per n)
  - global summary              (mean/std across circuits + seeds, per n)

Outputs (data/plots/estimator_validation_15/):
  - per_run_rows.csv
  - per_gate_rows.csv
  - per_group_rows.csv
  - per_circuit_summary.csv
  - global_summary.csv
  - recommended_n.json
  - plots/*.png

Usage:
    python script_estimator_validation_benchmark15.py
    python script_estimator_validation_benchmark15.py --n-list 100 300 500 1000 --n-seeds 10
    python script_estimator_validation_benchmark15.py --n-seeds 3            # fast dev run
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

# Importing this module also prepends qshaptools/src to sys.path, so the
# qshaptools import below works without additional setup.
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

ESTIMATOR_OUTPUT_DIR = ROOT / "data" / "plots" / "estimator_validation_15"
EXACT_RESULTS_JSON = ROOT / "data" / "plots" / "exact_owen_15" / "exact_owen_results.json"

DEFAULT_N_LIST = [100, 300, 500, 1000]
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
    p.add_argument("--n-list", type=int, nargs="+", default=DEFAULT_N_LIST,
                   help="Sample counts to test, e.g. --n-list 100 300 500 1000")
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                   help="Number of random seeds per (circuit, property, n) cell.")
    p.add_argument("--allow-order-fallback", action="store_true")
    p.add_argument("--force-recompute-exact", action="store_true",
                   help="Ignore exact_owen_results.json and recompute exact Owen.")
    p.add_argument("--rec-group-mae-threshold", type=float, default=0.02,
                   help="Recommendation threshold on mean group MAE.")
    p.add_argument("--rec-top-group-agree-threshold", type=float, default=0.95,
                   help="Recommendation threshold on top-group agreement rate.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampled Owen call
# ---------------------------------------------------------------------------

def run_sampled_owen(
    item,
    spec: Dict[str, Any],
    property_name: str,
    n_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run one sampled QuantumOwenValues for (circuit, property, n, seed).

    Key choices:
      - owen_sample_frac = -n         : negative sign => absolute # of (R,T) pairs
      - owen_sample_reps = 1          : statevector sim is deterministic
      - evaluate_value_only_once = False : coalition-level caching
                                        (matches the exact script for fairness)
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
        owen_sample_frac=-int(n_samples),
        owen_sample_reps=1,
        evaluate_value_only_once=False,
        owen_sample_seed=int(seed),
        name=f"{item.benchmark_id}_{property_name}_samp_n{n_samples}_s{seed}",
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

    # Spearman rho between sampled and exact gate-level scores.
    # Degenerate if fewer than 2 gates or exact is constant.
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

    # Kendall tau on group-level scores. For 3 groups it's coarse
    # (tau ∈ {-1, -1/3, 1/3, 1}) but still useful.
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
    """Owen efficiency: sum(phi) should equal v(N) - v(∅), so sampled and exact
    sums must match for any n. Any deviation indicates a bug in sampling."""
    sum_samp = float(sum(phi_samp.values()))
    sum_exact = float(sum(phi_exact.values()))
    return {
        "sum_phi_samp": sum_samp,
        "sum_phi_exact": sum_exact,
        "efficiency_abs_err": abs(sum_samp - sum_exact),
    }


# ---------------------------------------------------------------------------
# Load exact Owen results (reuse if on disk)
# ---------------------------------------------------------------------------

def load_or_compute_exact(
    benchmark_circuits: Dict[str, Any],
    gate_spec: Dict[str, Dict[str, Any]],
    exact_ids: Sequence[str],
    exact_results_json: Path,
    force_recompute: bool,
) -> Dict[str, Dict[str, Any]]:
    if exact_results_json.exists() and not force_recompute:
        with open(exact_results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == len(exact_ids):
            print(f"Loaded exact Owen results from: {exact_results_json}")
            return {r["benchmark_id"]: r for r in data}
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
    return {r["benchmark_id"]: r for r in results}


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def recommend_n(
    global_summary: pd.DataFrame,
    group_mae_threshold: float,
    top_agree_threshold: float,
) -> Dict[str, Any]:
    """Smallest n s.t. both properties pass group-MAE and top-group-agreement."""
    out: Dict[str, Any] = {
        "group_mae_threshold": group_mae_threshold,
        "top_agree_threshold": top_agree_threshold,
        "per_n_feasibility": [],
    }
    n_values = sorted(global_summary["n_samples"].unique())
    feasible_ns: List[int] = []
    for n in n_values:
        sub = global_summary.loc[global_summary["n_samples"] == n]
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
        out["per_n_feasibility"].append({
            "n_samples": int(n),
            "per_property": per_prop,
            "all_properties_pass": bool(all_pass),
        })
        if all_pass:
            feasible_ns.append(int(n))

    if feasible_ns:
        out["recommended_n"] = int(min(feasible_ns))
        out["reasoning"] = (
            f"Smallest n with mean group MAE <= {group_mae_threshold} and "
            f"top-group agreement >= {top_agree_threshold} for both properties."
        )
    else:
        scores = []
        for n in n_values:
            sub = global_summary.loc[global_summary["n_samples"] == n]
            score = float(sub["top_group_agree_rate"].mean() - sub["mean_group_mae"].mean())
            scores.append((score, int(n)))
        scores.sort(reverse=True)
        out["recommended_n"] = int(scores[0][1])
        out["reasoning"] = (
            "No n satisfied both thresholds for both properties. "
            "Falling back to n that maximises (mean top-agree - mean group MAE). "
            "Consider raising n beyond the tested range or relaxing thresholds."
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
    n_list: Sequence[int],
    out_dir: Path,
) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]
        plot_convergence(sub, "gate_mae", prop,
                         plots_dir / f"{prop}_gate_mae_vs_n.png",
                         ylabel="gate-level MAE")
        plot_convergence(sub, "group_mae", prop,
                         plots_dir / f"{prop}_group_mae_vs_n.png",
                         ylabel="group-level MAE")
        plot_convergence(sub, "gate_rmse", prop,
                         plots_dir / f"{prop}_gate_rmse_vs_n.png",
                         ylabel="gate-level RMSE")

    plot_runtime_vs_n(rows_df, plots_dir / "runtime_vs_n.png")

    for prop in PROPERTIES:
        plot_error_runtime_pareto(rows_df, prop,
                                  plots_dir / f"{prop}_error_runtime_pareto.png")

    plot_top_group_agreement(global_summary,
                             plots_dir / "top_group_agreement_vs_n.png")
    plot_full_order_agreement(rows_df,
                              plots_dir / "full_order_agreement_vs_n.png")
    plot_kendall_tau(global_summary,
                     plots_dir / "group_kendall_tau_vs_n.png")
    plot_spearman_rho(rows_df,
                      plots_dir / "gate_spearman_rho_vs_n.png")

    for prop in PROPERTIES:
        for n in n_list:
            plot_group_scatter(group_rows_df, prop, n,
                               plots_dir / f"{prop}_group_scatter_n{n}.png")
            plot_gate_scatter(gate_rows_df, prop, n,
                              plots_dir / f"{prop}_gate_scatter_n{n}.png")

    for prop in PROPERTIES:
        plot_per_circuit_heatmap(rows_df, prop,
                                 plots_dir / f"{prop}_per_circuit_heatmap.png")

    plot_efficiency_check(rows_df, plots_dir / "efficiency_check_vs_n.png")
    print(f"Saved plots to: {plots_dir}")


def plot_convergence(sub: pd.DataFrame, metric: str, prop: str, out_path: Path,
                     ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    agg = sub.groupby("n_samples")[metric].agg(["mean", "std", "min", "max"]).reset_index()
    ns = agg["n_samples"].to_numpy(dtype=float)
    means = agg["mean"].to_numpy(dtype=float)
    stds = agg["std"].to_numpy(dtype=float)
    ax.errorbar(ns, means, yerr=stds, fmt="o-", capsize=4, label="mean ± std")
    ax.fill_between(ns,
                    agg["min"].to_numpy(dtype=float),
                    agg["max"].to_numpy(dtype=float),
                    alpha=0.15, label="min–max")
    if np.all(means > 0):
        ref = means[0] * np.sqrt(ns[0] / ns)
        ax.plot(ns, ref, ls="--", color="gray", alpha=0.7,
                label=r"$1/\sqrt{n}$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{prop}: {ylabel} vs n")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_runtime_vs_n(rows_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]
        agg = sub.groupby("n_samples")["runtime_s"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["n_samples"], agg["mean"], yerr=agg["std"], fmt="o-",
                    capsize=4, label=prop)
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel("runtime per call (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Sampled Owen runtime vs n")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_runtime_pareto(rows_df: pd.DataFrame, prop: str,
                              out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = rows_df.loc[rows_df["property"] == prop]
    agg = sub.groupby("n_samples").agg(
        runtime=("runtime_s", "mean"),
        runtime_std=("runtime_s", "std"),
        err=("group_mae", "mean"),
        err_std=("group_mae", "std"),
    ).reset_index()
    ax.errorbar(agg["runtime"], agg["err"],
                xerr=agg["runtime_std"], yerr=agg["err_std"],
                fmt="o-", capsize=4, color="tab:blue")
    for _, r in agg.iterrows():
        ax.annotate(f"n={int(r['n_samples'])}", (r["runtime"], r["err"]),
                    textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("mean runtime (s)")
    ax.set_ylabel("mean group MAE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{prop}: error–runtime Pareto")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_group_agreement(global_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for prop in PROPERTIES:
        sub = global_summary.loc[global_summary["property"] == prop].sort_values("n_samples")
        ax.plot(sub["n_samples"], sub["top_group_agree_rate"], "o-", label=prop)
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel("top-group agreement rate")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Top-group agreement: sampled vs exact")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_full_order_agreement(rows_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]
        agg = sub.groupby("n_samples")["group_full_order_match"].mean().reset_index()
        ax.plot(agg["n_samples"], agg["group_full_order_match"], "o-", label=prop)
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel("full group-order agreement rate")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Full group-ordering agreement vs n")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_kendall_tau(global_summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for prop in PROPERTIES:
        sub = global_summary.loc[global_summary["property"] == prop].sort_values("n_samples")
        ax.plot(sub["n_samples"], sub["mean_kendall_tau"], "o-", label=prop)
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel("mean Kendall tau (group scores)")
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Group-score rank agreement")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_spearman_rho(rows_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]
        agg = sub.groupby("n_samples")["gate_spearman_rho"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["n_samples"], agg["mean"], yerr=agg["std"], fmt="o-",
                    capsize=4, label=prop)
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel("gate-level Spearman rho")
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("Gate-level rank correlation vs n")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_group_scatter(group_rows_df: pd.DataFrame, prop: str, n: int,
                       out_path: Path) -> None:
    sub = group_rows_df.loc[
        (group_rows_df["property"] == prop) & (group_rows_df["n_samples"] == n)
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
    ax.set_title(f"{prop}: group-level sampled vs exact at n={n}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gate_scatter(gate_rows_df: pd.DataFrame, prop: str, n: int,
                      out_path: Path) -> None:
    sub = gate_rows_df.loc[
        (gate_rows_df["property"] == prop) & (gate_rows_df["n_samples"] == n)
    ]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub["phi_exact"], sub["phi_samp"], alpha=0.35, s=15)
    mn = float(min(sub["phi_exact"].min(), sub["phi_samp"].min()))
    mx = float(max(sub["phi_exact"].max(), sub["phi_samp"].max()))
    pad = 0.05 * max(mx - mn, 1e-6)
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls="--", color="gray")
    ax.set_xlabel("exact phi (per gate)")
    ax.set_ylabel("sampled phi (per gate)")
    ax.set_title(f"{prop}: gate-level sampled vs exact at n={n}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_circuit_heatmap(rows_df: pd.DataFrame, prop: str, out_path: Path) -> None:
    sub = rows_df.loc[rows_df["property"] == prop]
    pivot = (
        sub.groupby(["benchmark_id", "n_samples"])["group_mae"]
        .mean().unstack("n_samples")
    )
    families = sub.drop_duplicates("benchmark_id").set_index("benchmark_id")["family_role"]
    order_key = [FAMILY_ORDER.get(families.loc[bid], 99) for bid in pivot.index]
    pivot = pivot.iloc[np.argsort(order_key)]

    fig, ax = plt.subplots(figsize=(7, max(5, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel("benchmark_id (grouped by family)")
    ax.set_title(f"{prop}: group MAE heatmap (mean over seeds)")
    plt.colorbar(im, ax=ax, label="group MAE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_efficiency_check(rows_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for prop in PROPERTIES:
        sub = rows_df.loc[rows_df["property"] == prop]
        agg = sub.groupby("n_samples")["efficiency_abs_err"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["n_samples"], agg["mean"], yerr=agg["std"], fmt="o-",
                    capsize=4, label=prop)
    ax.set_xlabel("n (sampled coalitions)")
    ax.set_ylabel("|sum(phi_samp) - sum(phi_exact)|")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Efficiency deviation vs n")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
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

    summary_df = load_summary(args.summary_csv)
    exact_ids = get_exact_ids(summary_df)
    benchmark_map = load_benchmark_pickle(
        args.benchmark_pkl, summary_df,
        allow_order_fallback=args.allow_order_fallback,
    )
    benchmark_circuits = build_benchmark_circuits(summary_df, benchmark_map, exact_ids)
    gate_spec = load_gate_spec(args.gate_spec_csv, benchmark_circuits)

    exact_by_id = load_or_compute_exact(
        benchmark_circuits=benchmark_circuits,
        gate_spec=gate_spec,
        exact_ids=exact_ids,
        exact_results_json=args.exact_results_json,
        force_recompute=args.force_recompute_exact,
    )

    n_list = [int(n) for n in args.n_list]
    seeds = list(range(int(args.n_seeds)))
    total = len(exact_ids) * len(PROPERTIES) * len(n_list) * len(seeds)
    print(f"\nRunning {total} sampled Owen configurations "
          f"(circuits={len(exact_ids)}, properties={len(PROPERTIES)}, "
          f"n_list={n_list}, seeds={len(seeds)}).")

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
            phi_exact = {int(k): float(v) for k, v in exact_res[f"phi_{prop}"].items()}
            group_exact = {k: float(v) for k, v in exact_res[f"group_{prop}"].items()}

            for n in n_list:
                for seed in seeds:
                    config_idx += 1
                    samp = run_sampled_owen(item, spec, prop, n, seed)
                    gate_errs = compute_per_gate_errors(samp["phi"], phi_exact)
                    group_errs = compute_group_errors(samp["group"], group_exact)
                    eff = compute_efficiency_check(samp["phi"], phi_exact)

                    rows.append({
                        "benchmark_id": bid,
                        "family_role": family_role,
                        "property": prop,
                        "n_samples": int(n),
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
                            "n_samples": int(n),
                            "seed": int(seed),
                            "gate_idx": int(k),
                            "phi_exact": phi_exact[k],
                            "phi_samp": float(v_samp),
                            "abs_err": abs(float(v_samp) - phi_exact[k]),
                        })

                    for g, s_samp in samp["group"].items():
                        group_rows.append({
                            "benchmark_id": bid,
                            "family_role": family_role,
                            "property": prop,
                            "n_samples": int(n),
                            "seed": int(seed),
                            "group": g,
                            "score_exact": group_exact[g],
                            "score_samp": float(s_samp),
                            "abs_err": abs(float(s_samp) - group_exact[g]),
                        })

                    if config_idx % 25 == 0 or config_idx == total:
                        elapsed = time.perf_counter() - t_start
                        rate = config_idx / max(elapsed, 1e-9)
                        eta = (total - config_idx) / max(rate, 1e-9)
                        print(f"  [{config_idx}/{total}] {bid} {prop} "
                              f"n={n} seed={seed} | "
                              f"{elapsed:.1f}s elapsed, ETA {eta:.0f}s")

    print(f"\nTotal wall-clock time: {time.perf_counter() - t_start:.1f}s")

    rows_df = pd.DataFrame(rows)
    gate_rows_df = pd.DataFrame(gate_rows)
    group_rows_df = pd.DataFrame(group_rows)

    rows_df.to_csv(out_dir / "per_run_rows.csv", index=False)
    gate_rows_df.to_csv(out_dir / "per_gate_rows.csv", index=False)
    group_rows_df.to_csv(out_dir / "per_group_rows.csv", index=False)

    per_circuit = rows_df.groupby(
        ["benchmark_id", "family_role", "property", "n_samples"]
    ).agg(
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
    ).reset_index()
    per_circuit.to_csv(out_dir / "per_circuit_summary.csv", index=False)

    global_rows: List[Dict[str, Any]] = []
    for (prop, n), sub in rows_df.groupby(["property", "n_samples"]):
        global_rows.append({
            "property": prop,
            "n_samples": int(n),
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
    global_summary = pd.DataFrame(global_rows).sort_values(["property", "n_samples"])
    global_summary.to_csv(out_dir / "global_summary.csv", index=False)

    print("\n=== Global summary ===")
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(global_summary.to_string(index=False))

    recommended = recommend_n(
        global_summary,
        group_mae_threshold=args.rec_group_mae_threshold,
        top_agree_threshold=args.rec_top_group_agree_threshold,
    )
    with open(out_dir / "recommended_n.json", "w", encoding="utf-8") as f:
        json.dump(recommended, f, indent=2)

    print("\n=== Recommended n ===")
    print(json.dumps(recommended, indent=2))

    make_plots(rows_df, gate_rows_df, group_rows_df, global_summary, n_list, out_dir)

    print(f"\nAll outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()