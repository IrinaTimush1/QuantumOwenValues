#!/usr/bin/env python3
"""
Random-partition control for the exact 15-circuit Owen benchmark.

The experiment asks whether the hand-defined E/M/X grouping has a stronger
dominance and faithfulness signal than random groupings with the same active
gates and the same non-empty group sizes.

Outputs are written to:
    results/random_partition_control_15/

Gate indices in CSV/JSON outputs follow the existing exact benchmark files:
0-based `gate_idx` values from `data/benchmark_15_exact_gate_spec.csv`.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from script_exact_owen_benchmark15 import (
    BENCHMARK_PKL,
    GATE_SPEC_CSV,
    ROOT,
    SUMMARY_CSV,
    BenchmarkCircuit,
    aggregate_group_scores,
    build_benchmark_circuits,
    calculate_meyer_wallach_entanglement_qiskit,
    calculate_stabilizer_renyi_entropy_qiskit,
    get_exact_ids,
    load_benchmark_pickle,
    load_gate_spec,
    load_summary,
)
from qshaptools.qowen import QuantumOwenValues
from qshaptools.tools import build_circuit


OUTPUT_DIR = ROOT / "results" / "random_partition_control_15"
VALUE_FUNCTIONS = ("magic", "entanglement")
EPS = 1e-12
GROUP_ORDER = ("E", "M", "X")


@dataclass
class PartitionMetrics:
    group_totals: Dict[str, float]
    dominance_abs: float
    dominance_norm: float
    top_group: str
    bottom_group: str
    top_drop: float
    bottom_drop: float
    faithfulness_gap: float
    faithfulness_ratio: float
    share_E: float
    share_M: float
    share_X: float
    resource_alignment: float
    expected_group_label: str
    expected_group_gates: List[int]
    expected_drop: float
    expected_drop_norm: float
    expected_metric_skip_reason: str


class CachedPropertyValueFunction:
    """Deterministic coalition value function with per-circuit memoization."""

    def __init__(self, item: BenchmarkCircuit, value_function: str):
        if value_function not in VALUE_FUNCTIONS:
            raise ValueError(f"Unknown value_function={value_function!r}")
        self.item = item
        self.value_function = value_function
        self.cache: Dict[Tuple[int, ...], float] = {}

    def value_for_indices(self, keep_gate_indices: Iterable[int]) -> float:
        key = tuple(sorted(int(g) for g in keep_gate_indices))
        if key in self.cache:
            return self.cache[key]

        qc, _ = build_circuit(
            qc_data=self.item.qc.data,
            num_qubits=self.item.qc.num_qubits,
            S=list(key),
            cl_bits=False,
        )
        if self.value_function == "magic":
            value = calculate_stabilizer_renyi_entropy_qiskit(qc)
        else:
            value = calculate_meyer_wallach_entanglement_qiskit(qc)
        self.cache[key] = float(value)
        return float(value)

    def __call__(
        self,
        qc_data: Any,
        num_qubits: int,
        S: Sequence[int],
        quantum_instance: Any = None,
        **kwargs: Any,
    ) -> float:
        # QuantumOwenValues passes S with locked gates already included.
        return self.value_for_indices(S)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--benchmark-pkl", type=Path, default=BENCHMARK_PKL)
    p.add_argument("--gate-spec-csv", type=Path, default=GATE_SPEC_CSV)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--n-random", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--benchmark-ids", nargs="+", default=None)
    p.add_argument(
        "--value-functions",
        nargs="+",
        choices=list(VALUE_FUNCTIONS),
        default=list(VALUE_FUNCTIONS),
    )
    p.add_argument("--max-resample-attempts", type=int, default=1000)
    p.add_argument(
        "--no-label-aware-metrics",
        action="store_true",
        help="Disable resource-alignment and expected-group ablation metrics.",
    )
    p.add_argument(
        "--allow-order-fallback",
        action="store_true",
        help=(
            "Only use this if benchmark_35_from_pool.pkl cannot be matched by "
            "benchmark_id or candidate_uid, but is known to be in summary order."
        ),
    )
    return p.parse_args()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def partition_to_dict(labels: Sequence[str], partition: Sequence[Sequence[int]]) -> Dict[str, List[int]]:
    return {str(label): sorted(int(g) for g in group) for label, group in zip(labels, partition)}


def nonempty_partition_from_spec(spec: Mapping[str, Any]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for label, group in zip(spec["partition_labels"], spec["partition"]):
        if len(group) > 0:
            out[str(label)] = sorted(int(g) for g in group)
    return out


def group_sizes(partition: Mapping[str, Sequence[int]]) -> Dict[str, int]:
    return {label: int(len(gates)) for label, gates in partition.items()}


def canonical_partition(partition: Mapping[str, Sequence[int]]) -> Dict[str, List[int]]:
    return {label: sorted(int(g) for g in partition[label]) for label in sorted(partition)}


def partition_equals(a: Mapping[str, Sequence[int]], b: Mapping[str, Sequence[int]]) -> bool:
    return canonical_partition(a) == canonical_partition(b)


def validate_random_partition(
    random_partition: Mapping[str, Sequence[int]],
    original_partition: Mapping[str, Sequence[int]],
    active: Sequence[int],
) -> None:
    active_sorted = sorted(int(g) for g in active)
    flat = sorted(int(g) for group in random_partition.values() for g in group)
    if flat != active_sorted:
        raise ValueError(f"Random partition does not cover active gates exactly: {flat} != {active_sorted}")
    if len(flat) != len(set(flat)):
        raise ValueError(f"Random partition has duplicate gates: {random_partition}")
    if group_sizes(random_partition) != group_sizes(original_partition):
        raise ValueError(
            f"Random partition group sizes changed: {group_sizes(random_partition)} "
            f"!= {group_sizes(original_partition)}"
        )
    if partition_equals(random_partition, original_partition):
        raise ValueError("Random partition exactly reproduces the original E/M/X partition.")


def make_random_partition(
    active: Sequence[int],
    original_partition: Mapping[str, Sequence[int]],
    random_seed: int,
    max_resample_attempts: int,
) -> Dict[str, List[int]]:
    labels = list(original_partition.keys())
    sizes = [len(original_partition[label]) for label in labels]
    rng = np.random.default_rng(int(random_seed))
    active_arr = np.array(sorted(int(g) for g in active), dtype=int)

    for _ in range(int(max_resample_attempts)):
        shuffled = active_arr.copy()
        rng.shuffle(shuffled)
        out: Dict[str, List[int]] = {}
        start = 0
        for label, size in zip(labels, sizes):
            out[label] = sorted(int(x) for x in shuffled[start : start + size])
            start += size
        if not partition_equals(out, original_partition):
            validate_random_partition(out, original_partition, active)
            return out

    raise RuntimeError(
        "Could not sample a non-original random partition after "
        f"{max_resample_attempts} attempts. active={list(active)}, original={original_partition}"
    )


def run_exact_owen_for_partition(
    item: BenchmarkCircuit,
    partition_dict: Mapping[str, Sequence[int]],
    locked: Sequence[int],
    value_function: str,
    cached_value_fun: CachedPropertyValueFunction,
    name_suffix: str,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    labels = list(partition_dict.keys())
    partition = [list(partition_dict[label]) for label in labels]
    qov = QuantumOwenValues(
        qc=item.qc,
        partition=partition,
        value_fun=cached_value_fun,
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=list(locked),
        owen_sample_frac=None,
        owen_sample_reps=1,
        evaluate_value_only_once=True,
        owen_sample_seed=123,
        name=f"{item.benchmark_id}_{value_function}_{name_suffix}",
        silent=True,
    )
    phi = {int(k): float(v) for k, v in qov.run().items()}
    group_totals = aggregate_group_scores(phi, partition, labels)
    return phi, {str(k): float(v) for k, v in group_totals.items()}


def choose_top_bottom(group_totals: Mapping[str, float]) -> Tuple[str, str]:
    labels = sorted(group_totals)
    top_group = max(labels, key=lambda lab: (float(group_totals[lab]), lab))
    bottom_group = min(labels, key=lambda lab: (float(group_totals[lab]), lab))
    return top_group, bottom_group


def expected_group_for_value_function(
    partition_dict: Mapping[str, Sequence[int]],
    value_function: str,
) -> Tuple[str, List[int], str]:
    """Return label, gates, and skip reason for the resource-expected group."""

    if value_function == "magic":
        gates = sorted(
            set(int(g) for g in partition_dict.get("M", []))
            | set(int(g) for g in partition_dict.get("X", []))
        )
        if not gates:
            return "M+X", [], "magic expected group M+X is empty"
        return "M+X", gates, ""
    if value_function == "entanglement":
        gates = sorted(int(g) for g in partition_dict.get("E", []))
        if not gates:
            return "E", [], "entanglement expected group E is empty"
        return "E", gates, ""
    raise ValueError(f"Unknown value_function={value_function!r}")


def group_abs_shares(group_totals: Mapping[str, float]) -> Dict[str, float]:
    abs_by_group = {label: abs(float(group_totals.get(label, 0.0))) for label in GROUP_ORDER}
    denom = sum(abs_by_group.values()) + EPS
    return {label: float(abs_by_group[label] / denom) for label in GROUP_ORDER}


def compute_partition_metrics(
    item: BenchmarkCircuit,
    active: Sequence[int],
    locked: Sequence[int],
    partition_dict: Mapping[str, Sequence[int]],
    value_function: str,
    cached_value_fun: CachedPropertyValueFunction,
    name_suffix: str,
    include_label_aware_metrics: bool = True,
) -> PartitionMetrics:
    _, group_totals = run_exact_owen_for_partition(
        item=item,
        partition_dict=partition_dict,
        locked=locked,
        value_function=value_function,
        cached_value_fun=cached_value_fun,
        name_suffix=name_suffix,
    )

    values = np.array([float(group_totals[label]) for label in group_totals], dtype=float)
    abs_values = np.abs(values)
    dominance_abs = float(np.max(abs_values) - np.min(abs_values))
    dominance_norm = float(dominance_abs / (float(np.sum(abs_values)) + EPS))

    top_group, bottom_group = choose_top_bottom(group_totals)
    active_set = set(int(g) for g in active)
    locked_set = set(int(g) for g in locked)
    full_keep = sorted(active_set | locked_set)
    keep_without_top = sorted((active_set - set(partition_dict[top_group])) | locked_set)
    keep_without_bottom = sorted((active_set - set(partition_dict[bottom_group])) | locked_set)

    full_value = cached_value_fun.value_for_indices(full_keep)
    value_without_top = cached_value_fun.value_for_indices(keep_without_top)
    value_without_bottom = cached_value_fun.value_for_indices(keep_without_bottom)
    top_drop = float(full_value - value_without_top)
    bottom_drop = float(full_value - value_without_bottom)
    faithfulness_gap = float(top_drop - bottom_drop)
    faithfulness_ratio = float(top_drop / (abs(bottom_drop) + EPS))

    shares = group_abs_shares(group_totals)
    if value_function == "magic":
        resource_alignment = float(shares["M"] + shares["X"])
    else:
        resource_alignment = float(shares["E"])

    expected_label, expected_gates, expected_skip = expected_group_for_value_function(
        partition_dict,
        value_function,
    )
    expected_drop = np.nan
    expected_drop_norm = np.nan
    if not include_label_aware_metrics:
        expected_skip = "label-aware metrics disabled"
        resource_alignment = np.nan
        expected_gates = []
    elif not expected_skip:
        keep_without_expected = sorted((active_set - set(expected_gates)) | locked_set)
        value_without_expected = cached_value_fun.value_for_indices(keep_without_expected)
        expected_drop = float(full_value - value_without_expected)
        expected_drop_norm = float(expected_drop / (abs(full_value) + EPS))

    return PartitionMetrics(
        group_totals={label: float(value) for label, value in group_totals.items()},
        dominance_abs=dominance_abs,
        dominance_norm=dominance_norm,
        top_group=top_group,
        bottom_group=bottom_group,
        top_drop=top_drop,
        bottom_drop=bottom_drop,
        faithfulness_gap=faithfulness_gap,
        faithfulness_ratio=faithfulness_ratio,
        share_E=float(shares["E"]),
        share_M=float(shares["M"]),
        share_X=float(shares["X"]),
        resource_alignment=resource_alignment,
        expected_group_label=expected_label,
        expected_group_gates=expected_gates,
        expected_drop=expected_drop,
        expected_drop_norm=expected_drop_norm,
        expected_metric_skip_reason=expected_skip,
    )


def metrics_to_row(metrics: Optional[PartitionMetrics]) -> Dict[str, Any]:
    if metrics is None:
        return {
            "group_totals_json": json_dumps({}),
            "dominance_abs": np.nan,
            "dominance_norm": np.nan,
            "top_group": "",
            "bottom_group": "",
            "top_drop": np.nan,
            "bottom_drop": np.nan,
            "faithfulness_gap": np.nan,
            "faithfulness_ratio": np.nan,
            "share_E": np.nan,
            "share_M": np.nan,
            "share_X": np.nan,
            "resource_alignment": np.nan,
            "expected_group_label": "",
            "expected_group_gates_json": json_dumps([]),
            "expected_drop": np.nan,
            "expected_drop_norm": np.nan,
            "expected_metric_skip_reason": "missing partition metrics",
        }
    return {
        "group_totals_json": json_dumps(metrics.group_totals),
        "dominance_abs": metrics.dominance_abs,
        "dominance_norm": metrics.dominance_norm,
        "top_group": metrics.top_group,
        "bottom_group": metrics.bottom_group,
        "top_drop": metrics.top_drop,
        "bottom_drop": metrics.bottom_drop,
        "faithfulness_gap": metrics.faithfulness_gap,
        "faithfulness_ratio": metrics.faithfulness_ratio,
        "share_E": metrics.share_E,
        "share_M": metrics.share_M,
        "share_X": metrics.share_X,
        "resource_alignment": metrics.resource_alignment,
        "expected_group_label": metrics.expected_group_label,
        "expected_group_gates_json": json_dumps(metrics.expected_group_gates),
        "expected_drop": metrics.expected_drop,
        "expected_drop_norm": metrics.expected_drop_norm,
        "expected_metric_skip_reason": metrics.expected_metric_skip_reason,
    }


def percentile_less(random_values: Sequence[float], emx_value: float) -> float:
    arr = np.asarray(random_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(emx_value):
        return float("nan")
    return float(np.sum(arr < float(emx_value)) / arr.size)


def distribution_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p95": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr, ddof=0)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p05": float(np.percentile(arr, 5)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
    }


def wilcoxon_percentile_p(percentiles: Sequence[float]) -> float:
    arr = np.asarray(percentiles, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        warnings.warn("Too few valid circuits for Wilcoxon signed-rank test; writing NaN.")
        return float("nan")
    diffs = arr - 0.5
    if np.allclose(diffs, 0.0):
        warnings.warn("All percentile differences from 0.5 are zero; writing NaN for Wilcoxon p-value.")
        return float("nan")
    try:
        return float(wilcoxon(diffs, alternative="greater").pvalue)
    except ValueError as exc:
        warnings.warn(f"Wilcoxon signed-rank test failed ({exc}); writing NaN.")
        return float("nan")


def build_summary(control_df: pd.DataFrame, requested_value_functions: Sequence[str], n_random: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_specs = [
        (
            "dominance_norm",
            "emx_dominance_norm",
            "random_dominance_norm_mean",
            "dominance_percentile",
        ),
        (
            "faithfulness_gap",
            "emx_faithfulness_gap",
            "random_faithfulness_gap_mean",
            "faithfulness_percentile",
        ),
        (
            "resource_alignment",
            "emx_resource_alignment",
            "random_resource_alignment_mean",
            "resource_alignment_percentile",
        ),
        (
            "expected_drop",
            "emx_expected_drop",
            "random_expected_drop_mean",
            "expected_drop_percentile",
        ),
        (
            "expected_drop_norm",
            "emx_expected_drop_norm",
            "random_expected_drop_norm_mean",
            "expected_drop_norm_percentile",
        ),
    ]
    for value_function in requested_value_functions:
        vf_sub = control_df.loc[control_df["value_function"] == value_function].copy()
        for metric, emx_col, random_col, percentile_col in metric_specs:
            if percentile_col not in vf_sub:
                pvals = np.array([])
                sub = vf_sub.iloc[0:0].copy()
            else:
                percentile_values = vf_sub[percentile_col].to_numpy(dtype=float) if len(vf_sub) else np.array([])
                finite_mask = np.isfinite(percentile_values)
                sub = vf_sub.loc[finite_mask].copy() if len(vf_sub) else vf_sub.copy()
                pvals = sub[percentile_col].to_numpy(dtype=float) if len(sub) else np.array([])
            rows.append(
                {
                    "value_function": value_function,
                    "metric": metric,
                    "n_valid_circuits": int(len(sub)),
                    "n_random": int(n_random),
                    "mean_emx": float_or_nan(sub[emx_col].mean()) if len(sub) else np.nan,
                    "mean_random": float_or_nan(sub[random_col].mean()) if len(sub) else np.nan,
                    "mean_percentile": float_or_nan(np.nanmean(pvals)) if len(pvals) else np.nan,
                    "median_percentile": float_or_nan(np.nanmedian(pvals)) if len(pvals) else np.nan,
                    "wilcoxon_p_greater_than_0_5": wilcoxon_percentile_p(pvals),
                }
            )
    return pd.DataFrame(rows)


def plot_percentile_strip(
    control_df: pd.DataFrame,
    percentile_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    rng = np.random.default_rng(2026)
    labels = list(VALUE_FUNCTIONS)
    for xpos, value_function in enumerate(labels):
        sub = control_df.loc[
            (control_df["value_function"] == value_function)
            & (control_df["skip_reason"].fillna("") == "")
        ]
        vals = sub[percentile_col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=vals.size)
        ax.scatter(
            np.full(vals.size, xpos, dtype=float) + jitter,
            vals,
            s=28,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.35,
        )
        ax.scatter(
            [xpos],
            [float(np.mean(vals))],
            marker="D",
            s=58,
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
            label="mean" if xpos == 0 else None,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(["magic", "entanglement"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dominance_summary(control_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = list(VALUE_FUNCTIONS)
    x = np.arange(len(labels))
    width = 0.34
    emx_means: List[float] = []
    emx_stds: List[float] = []
    rand_means: List[float] = []
    rand_stds: List[float] = []
    for value_function in labels:
        sub = control_df.loc[
            (control_df["value_function"] == value_function)
            & (control_df["skip_reason"].fillna("") == "")
        ]
        emx = sub["emx_dominance_norm"].dropna().to_numpy(dtype=float)
        rnd = sub["random_dominance_norm_mean"].dropna().to_numpy(dtype=float)
        emx_means.append(float(np.mean(emx)) if emx.size else np.nan)
        emx_stds.append(float(np.std(emx, ddof=0)) if emx.size else np.nan)
        rand_means.append(float(np.mean(rnd)) if rnd.size else np.nan)
        rand_stds.append(float(np.std(rnd, ddof=0)) if rnd.size else np.nan)

    ax.bar(x - width / 2, emx_means, width, yerr=emx_stds, capsize=3, label="E/M/X")
    ax.bar(x + width / 2, rand_means, width, yerr=rand_stds, capsize=3, label="matched random")
    ax.set_xticks(x)
    ax.set_xticklabels(["magic", "entanglement"])
    ax.set_ylabel("mean dominance_norm")
    ax.set_title("Dominance signal: E/M/X vs random partitions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metric_summary(
    control_df: pd.DataFrame,
    emx_col: str,
    random_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    valid_percentile_col: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = list(VALUE_FUNCTIONS)
    x = np.arange(len(labels))
    width = 0.34
    emx_means: List[float] = []
    emx_stds: List[float] = []
    rand_means: List[float] = []
    rand_stds: List[float] = []
    for value_function in labels:
        sub = control_df.loc[control_df["value_function"] == value_function]
        if valid_percentile_col is not None and valid_percentile_col in sub:
            sub = sub.dropna(subset=[valid_percentile_col])
        emx = sub[emx_col].dropna().to_numpy(dtype=float) if emx_col in sub else np.array([])
        rnd = sub[random_col].dropna().to_numpy(dtype=float) if random_col in sub else np.array([])
        emx_means.append(float(np.mean(emx)) if emx.size else np.nan)
        emx_stds.append(float(np.std(emx, ddof=0)) if emx.size else np.nan)
        rand_means.append(float(np.mean(rnd)) if rnd.size else np.nan)
        rand_stds.append(float(np.std(rnd, ddof=0)) if rnd.size else np.nan)

    ax.bar(x - width / 2, emx_means, width, yerr=emx_stds, capsize=3, label="E/M/X")
    ax.bar(x + width / 2, rand_means, width, yerr=rand_stds, capsize=3, label="matched random")
    ax.set_xticks(x)
    ax.set_xticklabels(["magic", "entanglement"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_example_histograms(
    control_df: pd.DataFrame,
    random_df: pd.DataFrame,
    out_path: Path,
) -> None:
    valid = control_df.loc[control_df["skip_reason"].fillna("") == ""].copy()
    valid = valid.dropna(subset=["dominance_percentile"])
    if valid.empty:
        return
    strong = valid.sort_values("dominance_percentile", ascending=False).iloc[0]
    weak = valid.assign(dist=(valid["dominance_percentile"] - 0.5).abs()).sort_values("dist").iloc[0]
    examples = [("strong", strong), ("near-median", weak)]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for ax, (label, row) in zip(axes, examples):
        sub = random_df.loc[
            (random_df["benchmark_id"] == row["benchmark_id"])
            & (random_df["value_function"] == row["value_function"])
        ]
        vals = sub["dominance_norm"].dropna().to_numpy(dtype=float)
        if vals.size:
            ax.hist(vals, bins=min(18, max(6, int(math.sqrt(vals.size)))), color="#9ecae1", edgecolor="black")
        ax.axvline(float(row["emx_dominance_norm"]), color="#d62728", linewidth=2.0, label="E/M/X")
        ax.set_title(f"{label}: {row['benchmark_id']} ({row['value_function']})")
        ax.set_xlabel("dominance_norm")
    axes[0].set_ylabel("random partitions")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_metric_example_histograms(
    control_df: pd.DataFrame,
    random_df: pd.DataFrame,
    percentile_col: str,
    emx_col: str,
    random_col: str,
    xlabel: str,
    out_path: Path,
) -> None:
    if percentile_col not in control_df or emx_col not in control_df or random_col not in random_df:
        return
    valid = control_df.dropna(subset=[percentile_col]).copy()
    if valid.empty:
        return
    strong = valid.sort_values(percentile_col, ascending=False).iloc[0]
    weak = valid.assign(dist=(valid[percentile_col] - 0.5).abs()).sort_values("dist").iloc[0]
    examples = [("strong", strong), ("near-median", weak)]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for ax, (label, row) in zip(axes, examples):
        sub = random_df.loc[
            (random_df["benchmark_id"] == row["benchmark_id"])
            & (random_df["value_function"] == row["value_function"])
        ]
        vals = sub[random_col].dropna().to_numpy(dtype=float)
        if vals.size:
            ax.hist(vals, bins=min(18, max(6, int(math.sqrt(vals.size)))), color="#c7e9c0", edgecolor="black")
        ax.axvline(float(row[emx_col]), color="#d62728", linewidth=2.0, label="E/M/X")
        ax.set_title(f"{label}: {row['benchmark_id']} ({row['value_function']})")
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("random partitions")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    random_rows: List[Dict[str, Any]],
    emx_rows: List[Dict[str, Any]],
    control_rows: List[Dict[str, Any]],
    summary_df: pd.DataFrame,
    discrimination_rows: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    random_df = pd.DataFrame(random_rows)
    emx_df = pd.DataFrame(emx_rows)
    control_df = pd.DataFrame(control_rows)
    discrim_df = pd.DataFrame(discrimination_rows)

    random_cols = [
        "benchmark_id",
        "family_role",
        "value_function",
        "random_index",
        "random_seed",
        "active_gates_json",
        "locked_gates_json",
        "original_group_sizes_json",
        "random_partition_json",
        "group_totals_json",
        "dominance_abs",
        "dominance_norm",
        "top_group",
        "bottom_group",
        "top_drop",
        "bottom_drop",
        "faithfulness_gap",
        "faithfulness_ratio",
        "share_E",
        "share_M",
        "share_X",
        "resource_alignment",
        "expected_group_label",
        "expected_group_gates_json",
        "expected_drop",
        "expected_drop_norm",
        "expected_metric_skip_reason",
    ]
    emx_cols = [
        "benchmark_id",
        "family_role",
        "value_function",
        "active_gates_json",
        "locked_gates_json",
        "emx_partition_json",
        "group_sizes_json",
        "group_totals_json",
        "dominance_abs",
        "dominance_norm",
        "top_group",
        "bottom_group",
        "top_drop",
        "bottom_drop",
        "faithfulness_gap",
        "faithfulness_ratio",
        "share_E",
        "share_M",
        "share_X",
        "resource_alignment",
        "expected_group_label",
        "expected_group_gates_json",
        "expected_drop",
        "expected_drop_norm",
        "expected_metric_skip_reason",
        "skip_reason",
    ]
    control_cols = [
        "benchmark_id",
        "family_role",
        "value_function",
        "n_active",
        "n_nonempty_groups",
        "n_random",
        "emx_dominance_norm",
        "random_dominance_norm_mean",
        "random_dominance_norm_std",
        "random_dominance_norm_median",
        "random_dominance_norm_p05",
        "random_dominance_norm_p95",
        "dominance_percentile",
        "emx_faithfulness_gap",
        "random_faithfulness_gap_mean",
        "random_faithfulness_gap_std",
        "random_faithfulness_gap_median",
        "random_faithfulness_gap_p05",
        "random_faithfulness_gap_p95",
        "faithfulness_percentile",
        "emx_resource_alignment",
        "random_resource_alignment_mean",
        "random_resource_alignment_std",
        "random_resource_alignment_median",
        "random_resource_alignment_p05",
        "random_resource_alignment_p95",
        "resource_alignment_percentile",
        "emx_expected_drop",
        "random_expected_drop_mean",
        "random_expected_drop_std",
        "random_expected_drop_median",
        "random_expected_drop_p05",
        "random_expected_drop_p95",
        "expected_drop_percentile",
        "emx_expected_drop_norm",
        "random_expected_drop_norm_mean",
        "random_expected_drop_norm_std",
        "random_expected_drop_norm_median",
        "random_expected_drop_norm_p05",
        "random_expected_drop_norm_p95",
        "expected_drop_norm_percentile",
        "skip_reason",
    ]
    summary_cols = [
        "value_function",
        "metric",
        "n_valid_circuits",
        "n_random",
        "mean_emx",
        "mean_random",
        "mean_percentile",
        "median_percentile",
        "wilcoxon_p_greater_than_0_5",
    ]
    discrim_cols = [
        "benchmark_id",
        "family_role",
        "emx_top_magic_group",
        "emx_top_entanglement_group",
        "emx_switch",
        "random_switch_rate",
        "n_random",
        "skip_reason",
    ]

    random_df.reindex(columns=random_cols).to_csv(output_dir / "random_partition_draws.csv", index=False)
    emx_df.reindex(columns=emx_cols).to_csv(output_dir / "emx_partition_metrics.csv", index=False)
    control_df.reindex(columns=control_cols).to_csv(output_dir / "random_control_by_circuit.csv", index=False)
    summary_df.reindex(columns=summary_cols).to_csv(output_dir / "random_control_summary.csv", index=False)
    discrim_df.reindex(columns=discrim_cols).to_csv(output_dir / "value_function_discrimination.csv", index=False)
    return random_df, control_df


def make_emx_row(
    item: BenchmarkCircuit,
    value_function: str,
    active: Sequence[int],
    locked: Sequence[int],
    emx_partition: Mapping[str, Sequence[int]],
    metrics: Optional[PartitionMetrics],
    skip_reason: str,
) -> Dict[str, Any]:
    row = {
        "benchmark_id": item.benchmark_id,
        "family_role": item.summary_row["family_role"],
        "value_function": value_function,
        "active_gates_json": json_dumps(sorted(int(g) for g in active)),
        "locked_gates_json": json_dumps(sorted(int(g) for g in locked)),
        "emx_partition_json": json_dumps(canonical_partition(emx_partition)),
        "group_sizes_json": json_dumps(group_sizes(emx_partition)),
        "skip_reason": skip_reason,
    }
    row.update(metrics_to_row(metrics))
    return row


def make_random_row(
    item: BenchmarkCircuit,
    value_function: str,
    random_index: int,
    random_seed: int,
    active: Sequence[int],
    locked: Sequence[int],
    original_group_sizes: Mapping[str, int],
    random_partition: Mapping[str, Sequence[int]],
    metrics: PartitionMetrics,
) -> Dict[str, Any]:
    row = {
        "benchmark_id": item.benchmark_id,
        "family_role": item.summary_row["family_role"],
        "value_function": value_function,
        "random_index": int(random_index),
        "random_seed": int(random_seed),
        "active_gates_json": json_dumps(sorted(int(g) for g in active)),
        "locked_gates_json": json_dumps(sorted(int(g) for g in locked)),
        "original_group_sizes_json": json_dumps(dict(original_group_sizes)),
        "random_partition_json": json_dumps(canonical_partition(random_partition)),
    }
    row.update(metrics_to_row(metrics))
    return row


def make_control_row(
    item: BenchmarkCircuit,
    value_function: str,
    active: Sequence[int],
    n_nonempty_groups: int,
    n_random: int,
    emx_metrics: Optional[PartitionMetrics],
    random_metrics: Sequence[PartitionMetrics],
    skip_reason: str,
) -> Dict[str, Any]:
    dominance_values = [m.dominance_norm for m in random_metrics]
    faithfulness_values = [m.faithfulness_gap for m in random_metrics]
    resource_alignment_values = [m.resource_alignment for m in random_metrics]
    expected_drop_values = [m.expected_drop for m in random_metrics]
    expected_drop_norm_values = [m.expected_drop_norm for m in random_metrics]
    row: Dict[str, Any] = {
        "benchmark_id": item.benchmark_id,
        "family_role": item.summary_row["family_role"],
        "value_function": value_function,
        "n_active": int(len(active)),
        "n_nonempty_groups": int(n_nonempty_groups),
        "n_random": int(len(random_metrics) if not skip_reason else 0),
        "emx_dominance_norm": emx_metrics.dominance_norm if emx_metrics else np.nan,
        "dominance_percentile": (
            percentile_less(dominance_values, emx_metrics.dominance_norm)
            if emx_metrics and not skip_reason
            else np.nan
        ),
        "emx_faithfulness_gap": emx_metrics.faithfulness_gap if emx_metrics else np.nan,
        "faithfulness_percentile": (
            percentile_less(faithfulness_values, emx_metrics.faithfulness_gap)
            if emx_metrics and not skip_reason
            else np.nan
        ),
        "emx_resource_alignment": emx_metrics.resource_alignment if emx_metrics else np.nan,
        "resource_alignment_percentile": (
            percentile_less(resource_alignment_values, emx_metrics.resource_alignment)
            if emx_metrics
            else np.nan
        ),
        "emx_expected_drop": emx_metrics.expected_drop if emx_metrics else np.nan,
        "expected_drop_percentile": (
            percentile_less(expected_drop_values, emx_metrics.expected_drop)
            if emx_metrics
            else np.nan
        ),
        "emx_expected_drop_norm": emx_metrics.expected_drop_norm if emx_metrics else np.nan,
        "expected_drop_norm_percentile": (
            percentile_less(expected_drop_norm_values, emx_metrics.expected_drop_norm)
            if emx_metrics
            else np.nan
        ),
        "skip_reason": skip_reason,
    }
    row.update(distribution_stats(dominance_values, "random_dominance_norm"))
    row.update(distribution_stats(faithfulness_values, "random_faithfulness_gap"))
    row.update(distribution_stats(resource_alignment_values, "random_resource_alignment"))
    row.update(distribution_stats(expected_drop_values, "random_expected_drop"))
    row.update(distribution_stats(expected_drop_norm_values, "random_expected_drop_norm"))
    return row


def main() -> None:
    args = parse_args()
    if args.n_random < 1:
        raise ValueError("--n-random must be >= 1")

    summary_df = load_summary(args.summary_csv)
    exact_ids = get_exact_ids(summary_df, benchmark_ids=args.benchmark_ids, expected_count=None if args.benchmark_ids else 15)
    benchmark_map = load_benchmark_pickle(
        args.benchmark_pkl,
        summary_df,
        allow_order_fallback=args.allow_order_fallback,
    )
    benchmark_circuits = build_benchmark_circuits(summary_df, benchmark_map, exact_ids)
    gate_spec = load_gate_spec(args.gate_spec_csv, benchmark_circuits)

    master_rng = np.random.default_rng(int(args.seed))
    random_rows: List[Dict[str, Any]] = []
    emx_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    discrimination_rows: List[Dict[str, Any]] = []

    random_metric_lookup: Dict[Tuple[str, int, str], PartitionMetrics] = {}
    emx_metric_lookup: Dict[Tuple[str, str], Optional[PartitionMetrics]] = {}
    random_seed_lookup: Dict[Tuple[str, int], int] = {}
    random_partition_lookup: Dict[Tuple[str, int], Dict[str, List[int]]] = {}

    requested_value_functions = list(dict.fromkeys(args.value_functions))
    include_label_aware_metrics = not bool(args.no_label_aware_metrics)
    print(f"Running random partition control for {len(exact_ids)} circuits")
    print(f"Value functions: {requested_value_functions}")
    print(f"Random partitions per valid circuit: {args.n_random}")
    print(f"Label-aware metrics: {'enabled' if include_label_aware_metrics else 'disabled'}")

    for benchmark_id in exact_ids:
        item = benchmark_circuits[benchmark_id]
        spec = gate_spec[benchmark_id]
        active = sorted(int(g) for g in spec["active"])
        locked = sorted(int(g) for g in spec["locked"])
        emx_partition = nonempty_partition_from_spec(spec)
        n_groups = len(emx_partition)
        skip_reason = ""
        if n_groups < 2:
            skip_reason = "fewer than two non-empty active E/M/X groups"

        print(f"  {benchmark_id}: active={len(active)}, groups={group_sizes(emx_partition)}")

        for random_index in range(args.n_random):
            if skip_reason:
                continue
            random_seed = int(master_rng.integers(0, 2**31 - 1))
            random_seed_lookup[(benchmark_id, random_index)] = random_seed
            random_partition_lookup[(benchmark_id, random_index)] = make_random_partition(
                active=active,
                original_partition=emx_partition,
                random_seed=random_seed,
                max_resample_attempts=args.max_resample_attempts,
            )

        for value_function in requested_value_functions:
            cached_value_fun = CachedPropertyValueFunction(item, value_function)
            emx_metrics: Optional[PartitionMetrics] = None
            if n_groups >= 1:
                emx_metrics = compute_partition_metrics(
                    item=item,
                    active=active,
                    locked=locked,
                    partition_dict=emx_partition,
                    value_function=value_function,
                    cached_value_fun=cached_value_fun,
                    name_suffix="emx",
                    include_label_aware_metrics=include_label_aware_metrics,
                )
            emx_metric_lookup[(benchmark_id, value_function)] = emx_metrics
            emx_rows.append(
                make_emx_row(
                    item=item,
                    value_function=value_function,
                    active=active,
                    locked=locked,
                    emx_partition=emx_partition,
                    metrics=emx_metrics,
                    skip_reason=skip_reason,
                )
            )

            random_metrics_for_control: List[PartitionMetrics] = []
            if not skip_reason:
                for random_index in range(args.n_random):
                    random_seed = random_seed_lookup[(benchmark_id, random_index)]
                    random_partition = random_partition_lookup[(benchmark_id, random_index)]
                    validate_random_partition(random_partition, emx_partition, active)
                    metrics = compute_partition_metrics(
                        item=item,
                        active=active,
                        locked=locked,
                        partition_dict=random_partition,
                        value_function=value_function,
                        cached_value_fun=cached_value_fun,
                        name_suffix=f"random_{random_index}",
                        include_label_aware_metrics=include_label_aware_metrics,
                    )
                    random_metric_lookup[(benchmark_id, random_index, value_function)] = metrics
                    random_metrics_for_control.append(metrics)
                    random_rows.append(
                        make_random_row(
                            item=item,
                            value_function=value_function,
                            random_index=random_index,
                            random_seed=random_seed,
                            active=active,
                            locked=locked,
                            original_group_sizes=group_sizes(emx_partition),
                            random_partition=random_partition,
                            metrics=metrics,
                        )
                    )

            control_rows.append(
                make_control_row(
                    item=item,
                    value_function=value_function,
                    active=active,
                    n_nonempty_groups=n_groups,
                    n_random=args.n_random,
                    emx_metrics=emx_metrics,
                    random_metrics=random_metrics_for_control,
                    skip_reason=skip_reason,
                )
            )

        if set(VALUE_FUNCTIONS).issubset(set(requested_value_functions)):
            discrim_skip = skip_reason
            emx_magic = emx_metric_lookup.get((benchmark_id, "magic"))
            emx_ent = emx_metric_lookup.get((benchmark_id, "entanglement"))
            random_switches: List[int] = []
            if not discrim_skip and emx_magic is not None and emx_ent is not None:
                for random_index in range(args.n_random):
                    rm = random_metric_lookup[(benchmark_id, random_index, "magic")]
                    re = random_metric_lookup[(benchmark_id, random_index, "entanglement")]
                    random_switches.append(int(rm.top_group != re.top_group))
            else:
                discrim_skip = discrim_skip or "requires valid magic and entanglement metrics"
            discrimination_rows.append(
                {
                    "benchmark_id": item.benchmark_id,
                    "family_role": item.summary_row["family_role"],
                    "emx_top_magic_group": emx_magic.top_group if emx_magic and not discrim_skip else "",
                    "emx_top_entanglement_group": emx_ent.top_group if emx_ent and not discrim_skip else "",
                    "emx_switch": (
                        int(emx_magic.top_group != emx_ent.top_group)
                        if emx_magic and emx_ent and not discrim_skip
                        else np.nan
                    ),
                    "random_switch_rate": float(np.mean(random_switches)) if random_switches else np.nan,
                    "n_random": int(len(random_switches)),
                    "skip_reason": discrim_skip,
                }
            )

    control_df_for_summary = pd.DataFrame(control_rows)
    summary_out = build_summary(control_df_for_summary, requested_value_functions, args.n_random)
    random_df, control_df = write_outputs(
        output_dir=args.output_dir,
        random_rows=random_rows,
        emx_rows=emx_rows,
        control_rows=control_rows,
        summary_df=summary_out,
        discrimination_rows=discrimination_rows,
    )

    fig_dir = args.output_dir / "figures"
    plot_percentile_strip(
        control_df,
        "dominance_percentile",
        "Dominance percentile of E/M/X vs matched random partitions",
        "dominance percentile",
        fig_dir / "random_control_dominance_percentiles.png",
    )
    plot_percentile_strip(
        control_df,
        "faithfulness_percentile",
        "Faithfulness percentile of E/M/X vs matched random partitions",
        "faithfulness percentile",
        fig_dir / "random_control_faithfulness_percentiles.png",
    )
    plot_dominance_summary(control_df, fig_dir / "random_control_dominance_summary.png")
    plot_example_histograms(control_df, random_df, fig_dir / "random_control_example_histograms.png")
    plot_percentile_strip(
        control_df,
        "resource_alignment_percentile",
        "Resource-alignment percentile vs matched random partitions",
        "resource-alignment percentile",
        fig_dir / "random_control_resource_alignment_percentiles.png",
    )
    plot_metric_summary(
        control_df,
        "emx_resource_alignment",
        "random_resource_alignment_mean",
        "resource_alignment",
        "Resource-alignment share: E/M/X vs random",
        fig_dir / "random_control_resource_alignment_summary.png",
        valid_percentile_col="resource_alignment_percentile",
    )
    plot_percentile_strip(
        control_df,
        "expected_drop_norm_percentile",
        "Expected-group ablation percentile vs matched random partitions",
        "expected_drop_norm percentile",
        fig_dir / "random_control_expected_drop_percentiles.png",
    )
    plot_metric_summary(
        control_df,
        "emx_expected_drop_norm",
        "random_expected_drop_norm_mean",
        "expected_drop_norm",
        "Expected-group ablation drop: E/M/X vs random",
        fig_dir / "random_control_expected_drop_summary.png",
        valid_percentile_col="expected_drop_norm_percentile",
    )
    plot_metric_example_histograms(
        control_df,
        random_df,
        "resource_alignment_percentile",
        "emx_resource_alignment",
        "resource_alignment",
        "resource_alignment",
        fig_dir / "random_control_alignment_example_histograms.png",
    )

    print(f"\nSaved random partition control outputs to: {args.output_dir}")
    print("Terminal summary:")
    headline_metrics = ["dominance_norm", "faithfulness_gap", "resource_alignment", "expected_drop_norm"]
    summary_print = summary_out.loc[summary_out["metric"].isin(headline_metrics)].copy()
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(
            summary_print[
                [
                    "value_function",
                    "metric",
                    "mean_emx",
                    "mean_random",
                    "mean_percentile",
                    "wilcoxon_p_greater_than_0_5",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
