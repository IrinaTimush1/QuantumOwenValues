#!/usr/bin/env python3
"""
script_faithfulness_ablation_benchmark15.py

Faithfulness-by-ablation experiment for the 15-circuit exact-Owen benchmark.

This script reuses the same 15 circuits, the same gate-spec CSV, and the same
value functions as script_exact_owen_benchmark15.py.

Experiment logic
----------------
For each circuit and for each property separately:
  1) obtain the exact Owen group scores,
  2) identify the highest-ranked group and lowest-ranked group,
  3) remove the highest-ranked group and recompute the target value function,
  4) remove the lowest-ranked group and recompute the target value function,
  5) compare the metric drops.

Why property-specific ranking?
------------------------------
Because the relevant group ranking for magic may differ from the relevant group
ranking for entanglement. Therefore the script performs two ablation analyses
per circuit:
  - one for the magic (SRE) value function,
  - one for the entanglement (Meyer-Wallach) value function.

Outputs
-------
Written to data/plots/faithfulness_ablation_15/ by default:
  - faithfulness_ablation_rows.csv
  - faithfulness_ablation_summary.csv
  - faithfulness_ablation_results.json
  - paired drop plots for magic and entanglement
  - scatter plots for top-drop vs bottom-drop
  - grouped family means plots

Usage
-----
    python script_faithfulness_ablation_benchmark15.py

Optional:
    python script_faithfulness_ablation_benchmark15.py \
        --exact-results-json data/plots/exact_owen_15/exact_owen_results.json

If the exact-results JSON is not supplied or does not exist, the script will
recompute exact Owen values using the exact benchmark code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from script_exact_owen_benchmark15 import (
    SUMMARY_CSV,
    BENCHMARK_PKL,
    GATE_SPEC_CSV,
    ROOT,
    load_summary,
    get_exact_ids,
    load_benchmark_pickle,
    build_benchmark_circuits,
    load_gate_spec,
    calculate_stabilizer_renyi_entropy_qiskit,
    calculate_meyer_wallach_entanglement_qiskit,
    run_exact_owen_for_circuit,
)
from qshaptools.tools import build_circuit

ABLATION_OUTPUT_DIR = ROOT / "data" / "plots" / "faithfulness_ablation_15"
EXACT_RESULTS_JSON = ROOT / "data" / "plots" / "exact_owen_15" / "exact_owen_results.json"

PROPERTY_TO_GROUP_KEY = {
    "magic": "group_magic",
    "entanglement": "group_entanglement",
}

TIE_TOL = 1e-12
EPS = 1e-15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--benchmark-pkl", type=Path, default=BENCHMARK_PKL)
    p.add_argument("--gate-spec-csv", type=Path, default=GATE_SPEC_CSV)
    p.add_argument("--exact-results-json", type=Path, default=EXACT_RESULTS_JSON)
    p.add_argument("--output-dir", type=Path, default=ABLATION_OUTPUT_DIR)
    p.add_argument(
        "--allow-order-fallback",
        action="store_true",
        help=(
            "Only use this if benchmark_35.pkl cannot be matched by benchmark_id or candidate_uid, "
            "but is known to be in the same order as benchmark_35_summary.csv."
        ),
    )
    p.add_argument(
        "--force-recompute-exact",
        action="store_true",
        help="Ignore any existing exact_owen_results.json and recompute exact Owen values.",
    )
    return p.parse_args()


def metric_value(qc, property_name: str) -> float:
    if property_name == "magic":
        return float(calculate_stabilizer_renyi_entropy_qiskit(qc))
    if property_name == "entanglement":
        return float(calculate_meyer_wallach_entanglement_qiskit(qc))
    raise ValueError(f"Unknown property_name={property_name!r}")


def build_reduced_qc(item, keep_gate_indices: Sequence[int]):
    qc, _ = build_circuit(
        qc_data=item.qc.data,
        num_qubits=item.qc.num_qubits,
        S=list(sorted(keep_gate_indices)),
        cl_bits=False,
    )
    return qc


def choose_extreme_groups(
    group_scores: Dict[str, float],
    tol: float = TIE_TOL,
) -> Dict[str, Any]:
    labels = list(group_scores.keys())
    vals = np.array([float(group_scores[k]) for k in labels], dtype=float)

    max_val = float(np.max(vals))
    min_val = float(np.min(vals))

    top_groups = sorted([lab for lab in labels if abs(group_scores[lab] - max_val) <= tol])
    bottom_groups = sorted([lab for lab in labels if abs(group_scores[lab] - min_val) <= tol])

    # Deterministic representative choice for ablation when there is a tie.
    chosen_top = top_groups[0]
    chosen_bottom = bottom_groups[0]

    return {
        "top_group": chosen_top,
        "top_groups_tied": top_groups,
        "top_score": max_val,
        "bottom_group": chosen_bottom,
        "bottom_groups_tied": bottom_groups,
        "bottom_score": min_val,
        "top_is_tie": len(top_groups) > 1,
        "bottom_is_tie": len(bottom_groups) > 1,
    }


def relative_drop(drop_abs: float, baseline: float) -> float:
    denom = max(abs(float(baseline)), EPS)
    return float(drop_abs / denom)


def run_ablation_for_property(
    item,
    spec: Dict[str, Any],
    exact_result: Dict[str, Any],
    property_name: str,
) -> Dict[str, Any]:
    group_scores = exact_result[PROPERTY_TO_GROUP_KEY[property_name]]
    extremes = choose_extreme_groups(group_scores)

    partition_dict = exact_result["partition"]
    group_to_gates = {k: list(v) for k, v in partition_dict.items()}

    active = set(spec["active"])
    locked = set(spec["locked"])

    baseline_metric = metric_value(item.qc, property_name)

    top_group = extremes["top_group"]
    bottom_group = extremes["bottom_group"]

    keep_without_top = sorted((active - set(group_to_gates[top_group])) | locked)
    keep_without_bottom = sorted((active - set(group_to_gates[bottom_group])) | locked)

    qc_without_top = build_reduced_qc(item, keep_without_top)
    qc_without_bottom = build_reduced_qc(item, keep_without_bottom)

    metric_without_top = metric_value(qc_without_top, property_name)
    metric_without_bottom = metric_value(qc_without_bottom, property_name)

    drop_top_abs = float(baseline_metric - metric_without_top)
    drop_bottom_abs = float(baseline_metric - metric_without_bottom)

    return {
        "benchmark_id": item.benchmark_id,
        "family_role": item.summary_row["family_role"],
        "property": property_name,
        "baseline_metric": float(baseline_metric),
        "group_scores": {k: float(v) for k, v in group_scores.items()},
        "top_group": top_group,
        "top_groups_tied": extremes["top_groups_tied"],
        "top_score": float(extremes["top_score"]),
        "top_is_tie": bool(extremes["top_is_tie"]),
        "bottom_group": bottom_group,
        "bottom_groups_tied": extremes["bottom_groups_tied"],
        "bottom_score": float(extremes["bottom_score"]),
        "bottom_is_tie": bool(extremes["bottom_is_tie"]),
        "removed_top_group_gates": group_to_gates[top_group],
        "removed_bottom_group_gates": group_to_gates[bottom_group],
        "metric_without_top": float(metric_without_top),
        "metric_without_bottom": float(metric_without_bottom),
        "drop_top_abs": drop_top_abs,
        "drop_bottom_abs": drop_bottom_abs,
        "drop_top_rel": relative_drop(drop_top_abs, baseline_metric),
        "drop_bottom_rel": relative_drop(drop_bottom_abs, baseline_metric),
        "top_beats_bottom_abs": bool(drop_top_abs > drop_bottom_abs + TIE_TOL),
        "top_equals_bottom_abs": bool(abs(drop_top_abs - drop_bottom_abs) <= TIE_TOL),
    }


def ensure_exact_results(
    benchmark_circuits: Dict[str, Any],
    gate_spec: Dict[str, Dict[str, Any]],
    exact_ids: Sequence[str],
    exact_results_json: Path,
    force_recompute: bool,
) -> List[Dict[str, Any]]:
    if exact_results_json.exists() and not force_recompute:
        with open(exact_results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) == len(exact_ids):
            print(f"Loaded exact Owen results from: {exact_results_json}")
            return data
        print("Existing exact results JSON looked invalid/incomplete; recomputing exact Owen.")

    results: List[Dict[str, Any]] = []
    for benchmark_id in exact_ids:
        print(f"Recomputing exact Owen for {benchmark_id} ...")
        results.append(
            run_exact_owen_for_circuit(
                item=benchmark_circuits[benchmark_id],
                spec=gate_spec[benchmark_id],
            )
        )

    exact_results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(exact_results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved exact Owen results to: {exact_results_json}")
    return results


def plot_paired_drops(df_prop: pd.DataFrame, property_name: str, out_path: Path) -> None:
    plot_df = df_prop.sort_values(["family_role", "benchmark_id"]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, plot_df["drop_top_abs"], width, label="remove top Owen group")
    ax.bar(x + width / 2, plot_df["drop_bottom_abs"], width, label="remove bottom Owen group")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["benchmark_id"], rotation=45, ha="right")
    ax.set_ylabel("metric drop (baseline - ablated)")
    ax.set_title(f"Faithfulness by ablation: {property_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter_top_vs_bottom(df_prop: pd.DataFrame, property_name: str, out_path: Path) -> None:
    vals = np.concatenate([
        df_prop["drop_top_abs"].to_numpy(dtype=float),
        df_prop["drop_bottom_abs"].to_numpy(dtype=float),
    ])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = 0.05 * max(hi - lo, 1.0)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df_prop["drop_bottom_abs"], df_prop["drop_top_abs"])
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("drop after removing bottom group")
    ax.set_ylabel("drop after removing top group")
    ax.set_title(f"Top-vs-bottom ablation drop: {property_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_family_means(df: pd.DataFrame, property_name: str, out_path: Path) -> None:
    sub = (
        df.loc[df["property"] == property_name]
        .groupby("family_role", as_index=False)[["drop_top_abs", "drop_bottom_abs"]]
        .mean()
    )
    family_order = [
        fam for fam in ["magic_axis", "entanglement_axis", "interior"]
        if fam in sub["family_role"].tolist()
    ]
    sub["family_role"] = pd.Categorical(sub["family_role"], categories=family_order, ordered=True)
    sub = sub.sort_values("family_role")

    x = np.arange(len(sub))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, sub["drop_top_abs"], width, label="remove top Owen group")
    ax.bar(x + width / 2, sub["drop_bottom_abs"], width, label="remove bottom Owen group")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["family_role"])
    ax.set_ylabel("mean metric drop")
    ax.set_title(f"Family mean ablation drops: {property_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for property_name in ["magic", "entanglement"]:
        sub = df.loc[df["property"] == property_name].copy()
        rows.append(
            {
                "property": property_name,
                "scope": "all_circuits",
                "n": int(len(sub)),
                "mean_drop_top_abs": float(sub["drop_top_abs"].mean()),
                "mean_drop_bottom_abs": float(sub["drop_bottom_abs"].mean()),
                "median_drop_top_abs": float(sub["drop_top_abs"].median()),
                "median_drop_bottom_abs": float(sub["drop_bottom_abs"].median()),
                "count_top_gt_bottom": int(sub["top_beats_bottom_abs"].sum()),
                "count_equal": int(sub["top_equals_bottom_abs"].sum()),
                "share_top_gt_bottom": float(sub["top_beats_bottom_abs"].mean()),
                "mean_advantage_top_minus_bottom": float((sub["drop_top_abs"] - sub["drop_bottom_abs"]).mean()),
            }
        )

        fam_group = sub.groupby("family_role")
        for family_role, fam_df in fam_group:
            rows.append(
                {
                    "property": property_name,
                    "scope": family_role,
                    "n": int(len(fam_df)),
                    "mean_drop_top_abs": float(fam_df["drop_top_abs"].mean()),
                    "mean_drop_bottom_abs": float(fam_df["drop_bottom_abs"].mean()),
                    "median_drop_top_abs": float(fam_df["drop_top_abs"].median()),
                    "median_drop_bottom_abs": float(fam_df["drop_bottom_abs"].median()),
                    "count_top_gt_bottom": int(fam_df["top_beats_bottom_abs"].sum()),
                    "count_equal": int(fam_df["top_equals_bottom_abs"].sum()),
                    "share_top_gt_bottom": float(fam_df["top_beats_bottom_abs"].mean()),
                    "mean_advantage_top_minus_bottom": float((fam_df["drop_top_abs"] - fam_df["drop_bottom_abs"]).mean()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    summary_df = load_summary(args.summary_csv)
    exact_ids = get_exact_ids(summary_df)
    benchmark_map = load_benchmark_pickle(
        args.benchmark_pkl,
        summary_df,
        allow_order_fallback=args.allow_order_fallback,
    )
    benchmark_circuits = build_benchmark_circuits(summary_df, benchmark_map, exact_ids)
    gate_spec = load_gate_spec(args.gate_spec_csv, benchmark_circuits)

    exact_results = ensure_exact_results(
        benchmark_circuits=benchmark_circuits,
        gate_spec=gate_spec,
        exact_ids=exact_ids,
        exact_results_json=args.exact_results_json,
        force_recompute=args.force_recompute_exact,
    )
    exact_by_id = {r["benchmark_id"]: r for r in exact_results}

    rows: List[Dict[str, Any]] = []
    for benchmark_id in exact_ids:
        item = benchmark_circuits[benchmark_id]
        spec = gate_spec[benchmark_id]
        exact_res = exact_by_id[benchmark_id]

        for property_name in ["magic", "entanglement"]:
            rows.append(run_ablation_for_property(item, spec, exact_res, property_name))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = out_dir / "faithfulness_ablation_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    rows_df = pd.DataFrame(rows)
    rows_csv = out_dir / "faithfulness_ablation_rows.csv"
    rows_df.to_csv(rows_csv, index=False)

    summary_df_out = build_summary_table(rows_df)
    summary_csv = out_dir / "faithfulness_ablation_summary.csv"
    summary_df_out.to_csv(summary_csv, index=False)

    for property_name in ["magic", "entanglement"]:
        sub = rows_df.loc[rows_df["property"] == property_name].copy()
        plot_paired_drops(sub, property_name, out_dir / f"{property_name}_paired_drops.png")
        plot_scatter_top_vs_bottom(sub, property_name, out_dir / f"{property_name}_top_vs_bottom_scatter.png")
        plot_family_means(rows_df, property_name, out_dir / f"{property_name}_family_means.png")

    print(f"Saved per-circuit ablation rows to: {rows_csv}")
    print(f"Saved summary statistics to: {summary_csv}")
    print(f"Saved JSON results to: {results_json}")
    print(f"Saved plots to: {out_dir}")

    print("\nHeadline statistics:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df_out.to_string(index=False))


if __name__ == "__main__":
    main()
