#!/usr/bin/env python3
"""
Run QSVM Owen values for two appendix alternative partitions.

The main thesis QSVM experiment uses the E/M/X partition. This script keeps
the same datasets, passive Hadamard gates, feature-map convention, and exact
Owen evaluator, but changes the active-gate grouping to:

* feature_semantics: F1, F2, F12
* repetition_blocks: B1, B2, B3 where available

Outputs are written to results/qsvm_owen_alternative_partitions/ by default.
All saved gate indices are thesis-style 1-based indices.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qsvm_experiment_utils import (
    DEFAULT_CONVENTION,
    ensure_dir,
    import_qiskit_and_qowen,
    infer_repo_root,
    load_qsvm_dataset,
    qsvm_accuracy,
    validate_manual_simulator_random_subsets,
    build_dummy_full_qsvm_circuit,
)


EPS = 1e-12
PARTITION_NAMES = ["feature_semantics", "repetition_blocks"]
GROUP_COLORS = {
    "F1": "#4c78a8",
    "F2": "#f58518",
    "F12": "#54a24b",
    "B1": "#4c78a8",
    "B2": "#f58518",
    "B3": "#54a24b",
}


@dataclass(frozen=True)
class AlternativePartition:
    partition_name: str
    r: int
    passive_gates: List[int]
    groups: Dict[str, List[int]]
    group_descriptions: Dict[str, str]

    @property
    def active_gates(self) -> List[int]:
        return sorted(g for gates in self.groups.values() for g in gates)

    @property
    def passive_gates_0_based(self) -> List[int]:
        return [g - 1 for g in self.passive_gates]

    @property
    def groups_0_based(self) -> Dict[str, List[int]]:
        return {label: [g - 1 for g in gates] for label, gates in self.groups.items()}


class CachedQSVMValueFunction:
    """In-memory deterministic v(S) cache shared across partitions."""

    def __init__(self, dataset: Any, r: int, passive_gates: Sequence[int], svc_c: float = 1.0):
        self.dataset = dataset
        self.r = int(r)
        self.passive_gates = sorted(int(g) for g in passive_gates)
        self.svc_c = float(svc_c)
        self.cache: Dict[Tuple[int, ...], float] = {}

    def value_for_active_gates(self, active_gates_1_based: Iterable[int]) -> float:
        active = tuple(sorted(int(g) for g in active_gates_1_based if int(g) not in self.passive_gates))
        if active in self.cache:
            return self.cache[active]
        value = qsvm_accuracy(
            dataset=self.dataset,
            r=self.r,
            active_gates=active,
            passive_gates=self.passive_gates,
            convention=DEFAULT_CONVENTION,
            svc_c=self.svc_c,
        )
        self.cache[active] = float(value)
        return float(value)

    def __call__(
        self,
        qc_data: Any = None,
        num_qubits: Optional[int] = None,
        S: Optional[Sequence[int]] = None,
        S_list: Optional[Sequence[Sequence[int]]] = None,
        quantum_instance: Any = None,
        **kwargs: Any,
    ) -> float | List[float]:
        if S_list is not None:
            return [self._value_from_qowen_indices(s) for s in S_list]
        if S is None:
            raise ValueError("CachedQSVMValueFunction requires S or S_list.")
        return self._value_from_qowen_indices(S)

    def _value_from_qowen_indices(self, S: Sequence[int]) -> float:
        active_1_based = [idx + 1 for idx in S if (idx + 1) not in self.passive_gates]
        return self.value_for_active_gates(active_1_based)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "qsvm_owen_alternative_partitions")
    parser.add_argument("--r-values", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--partition-names", nargs="+", default=PARTITION_NAMES, choices=PARTITION_NAMES)
    parser.add_argument("--dataset-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--svc-c", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--validate-simulator", action="store_true")
    return parser.parse_args()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def passive_gates_for_r(r: int) -> List[int]:
    out: List[int] = []
    for block in range(1, int(r) + 1):
        offset = 7 * (block - 1)
        out.extend([1 + offset, 3 + offset])
    return sorted(out)


def make_feature_semantics_partition(r: int) -> AlternativePartition:
    groups = {"F1": [], "F2": [], "F12": []}
    for block in range(1, int(r) + 1):
        offset = 7 * (block - 1)
        groups["F1"].append(2 + offset)
        groups["F2"].append(4 + offset)
        groups["F12"].extend([5 + offset, 6 + offset, 7 + offset])
    return AlternativePartition(
        partition_name="feature_semantics",
        r=int(r),
        passive_gates=passive_gates_for_r(r),
        groups={k: sorted(v) for k, v in groups.items()},
        group_descriptions={
            "F1": "local phase encoding of feature x1",
            "F2": "local phase encoding of feature x2",
            "F12": "two-feature interaction motif",
        },
    )


def make_repetition_blocks_partition(r: int) -> AlternativePartition:
    groups: Dict[str, List[int]] = {}
    descriptions: Dict[str, str] = {}
    for block in range(1, int(r) + 1):
        offset = 7 * (block - 1)
        label = f"B{block}"
        groups[label] = [2 + offset, 4 + offset, 5 + offset, 6 + offset, 7 + offset]
        descriptions[label] = f"active gates in repetition block {block}"
    return AlternativePartition(
        partition_name="repetition_blocks",
        r=int(r),
        passive_gates=passive_gates_for_r(r),
        groups={k: sorted(v) for k, v in groups.items()},
        group_descriptions=descriptions,
    )


def make_partition(partition_name: str, r: int) -> AlternativePartition:
    if partition_name == "feature_semantics":
        return make_feature_semantics_partition(r)
    if partition_name == "repetition_blocks":
        return make_repetition_blocks_partition(r)
    raise ValueError(f"Unknown partition_name={partition_name!r}")


def validate_partition(partition: AlternativePartition) -> None:
    all_gates = set(range(1, 7 * partition.r + 1))
    passive = set(partition.passive_gates)
    active = partition.active_gates
    active_set = set(active)
    if passive & active_set:
        raise ValueError(f"{partition.partition_name} r={partition.r}: passive/active overlap")
    if passive | active_set != all_gates:
        missing = sorted(all_gates - (passive | active_set))
        extra = sorted((passive | active_set) - all_gates)
        raise ValueError(f"{partition.partition_name} r={partition.r}: missing={missing}, extra={extra}")
    if len(active) != len(active_set):
        raise ValueError(f"{partition.partition_name} r={partition.r}: duplicate active gates")


def rank_desc(values: Mapping[str, float]) -> Dict[str, int]:
    ordered = sorted(values, key=lambda k: (-float(values[k]), k))
    return {label: i + 1 for i, label in enumerate(ordered)}


def run_owen_for_partition(
    partition: AlternativePartition,
    evaluator: CachedQSVMValueFunction,
    QuantumOwenValues: Any,
    *,
    silent: bool = True,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    qc = build_dummy_full_qsvm_circuit(partition.r)
    qowen_partition = [partition.groups_0_based[label] for label in partition.groups]
    qov = QuantumOwenValues(
        qc=qc,
        partition=qowen_partition,
        value_fun=evaluator,
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=partition.passive_gates_0_based,
        owen_sample_frac=None,
        owen_sample_reps=1,
        evaluate_value_only_once=True,
        sample_in_memory=True,
        owen_sample_seed=123,
        owen_batch_size=None,
        name=f"qsvm_{partition.partition_name}_r{partition.r}",
        silent=bool(silent),
    )
    phi_0_based = qov.run()
    phi_1_based = {int(k) + 1: float(v) for k, v in phi_0_based.items()}
    group_values = {
        label: float(sum(phi_1_based[g] for g in gates))
        for label, gates in partition.groups.items()
    }
    return phi_1_based, group_values


def group_rows_for_dataset(
    partition: AlternativePartition,
    group_values: Mapping[str, float],
    full_accuracy: float,
    baseline_accuracy: float,
    dataset_index: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, gates in partition.groups.items():
        rows.append(
            {
                "partition_name": partition.partition_name,
                "r": partition.r,
                "dataset_index": int(dataset_index),
                "group": group,
                "group_description": partition.group_descriptions[group],
                "gates_json": json_dumps(gates),
                "passive_gates_json": json_dumps(partition.passive_gates),
                "group_owen_value": float(group_values[group]),
                "full_accuracy": float(full_accuracy),
                "baseline_accuracy_if_available": float(baseline_accuracy),
                "n_active_gates": int(len(partition.active_gates)),
                "n_passive_gates": int(len(partition.passive_gates)),
            }
        )
    return rows


def gate_rows_for_dataset(
    partition: AlternativePartition,
    phi_1_based: Mapping[int, float],
    dataset_index: int,
) -> List[Dict[str, Any]]:
    gate_to_group = {
        gate: group
        for group, gates in partition.groups.items()
        for gate in gates
    }
    return [
        {
            "partition_name": partition.partition_name,
            "r": partition.r,
            "dataset_index": int(dataset_index),
            "gate_index": int(gate),
            "group": gate_to_group[gate],
            "gate_owen_value": float(phi_1_based[gate]),
        }
        for gate in sorted(gate_to_group)
    ]


def add_group_shares(group_df: pd.DataFrame) -> pd.DataFrame:
    df = group_df.copy()
    denom = (
        df.assign(abs_value=df["group_owen_value"].abs())
        .groupby(["partition_name", "r", "dataset_index"])["abs_value"]
        .transform("sum")
        + EPS
    )
    df["abs_group_owen"] = df["group_owen_value"].abs()
    df["share_abs_group_owen"] = df["abs_group_owen"] / denom
    return df


def build_group_stats(group_df: pd.DataFrame) -> pd.DataFrame:
    df = add_group_shares(group_df)
    rows: List[Dict[str, Any]] = []
    for (partition_name, r, group), sub in df.groupby(["partition_name", "r", "group"], sort=True):
        first = sub.iloc[0]
        rows.append(
            {
                "partition_name": partition_name,
                "r": int(r),
                "group": group,
                "group_description": first["group_description"],
                "gates_json": first["gates_json"],
                "mean_group_owen": float(sub["group_owen_value"].mean()),
                "std_group_owen": float(sub["group_owen_value"].std(ddof=0)),
                "mean_abs_group_owen": float(sub["abs_group_owen"].mean()),
                "std_abs_group_owen": float(sub["abs_group_owen"].std(ddof=0)),
                "mean_share_abs_group_owen": float(sub["share_abs_group_owen"].mean()),
                "std_share_abs_group_owen": float(sub["share_abs_group_owen"].std(ddof=0)),
            }
        )
    stats = pd.DataFrame(rows)
    if stats.empty:
        stats["rank_by_mean_group_owen"] = []
        stats["rank_by_mean_abs_group_owen"] = []
        return stats
    stats["rank_by_mean_group_owen"] = (
        stats.groupby(["partition_name", "r"])["mean_group_owen"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    stats["rank_by_mean_abs_group_owen"] = (
        stats.groupby(["partition_name", "r"])["mean_abs_group_owen"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return stats.sort_values(["partition_name", "r", "rank_by_mean_abs_group_owen"]).reset_index(drop=True)


def build_gate_stats(gate_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (partition_name, r, gate_index, group), sub in gate_df.groupby(
        ["partition_name", "r", "gate_index", "group"], sort=True
    ):
        rows.append(
            {
                "partition_name": partition_name,
                "r": int(r),
                "gate_index": int(gate_index),
                "group": group,
                "mean_gate_owen": float(sub["gate_owen_value"].mean()),
                "std_gate_owen": float(sub["gate_owen_value"].std(ddof=0)),
                "mean_abs_gate_owen": float(sub["gate_owen_value"].abs().mean()),
            }
        )
    stats = pd.DataFrame(rows)
    if stats.empty:
        stats["rank_by_mean_abs_gate_owen"] = []
        return stats
    stats["rank_by_mean_abs_gate_owen"] = (
        stats.groupby(["partition_name", "r"])["mean_abs_gate_owen"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return stats.sort_values(["partition_name", "r", "rank_by_mean_abs_gate_owen"]).reset_index(drop=True)


def interpretation_short(partition_name: str, r: int, dominant_abs: str, n_groups: int) -> str:
    if partition_name == "feature_semantics":
        if dominant_abs == "F12":
            return "pairwise feature-interaction motif has the largest mean absolute contribution"
        if dominant_abs == "F1":
            return "feature x1 local phase gates have the largest mean absolute contribution"
        if dominant_abs == "F2":
            return "feature x2 local phase gates have the largest mean absolute contribution"
    if partition_name == "repetition_blocks":
        if n_groups < 2:
            return "only one active repetition block is available"
        return f"repetition block {dominant_abs[1:]} has the largest mean absolute contribution"
    return ""


def build_partition_summary(group_stats: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (partition_name, r), sub in group_stats.groupby(["partition_name", "r"], sort=True):
        n_groups = int(len(sub))
        signed = {str(row["group"]): float(row["mean_group_owen"]) for _, row in sub.iterrows()}
        abs_vals = {label: abs(value) for label, value in signed.items()}
        dominant_signed = max(signed, key=lambda g: (signed[g], g)) if signed else ""
        dominant_abs = max(abs_vals, key=lambda g: (abs_vals[g], g)) if abs_vals else ""
        skip_reason = ""
        if n_groups < 2:
            dominance_gap_abs = np.nan
            dominance_gap_norm = np.nan
            skip_reason = "single active group; dominance comparison is not meaningful"
        else:
            vals = np.array(list(abs_vals.values()), dtype=float)
            dominance_gap_abs = float(np.max(vals) - np.min(vals))
            dominance_gap_norm = float(dominance_gap_abs / (np.sum(vals) + EPS))
        rows.append(
            {
                "partition_name": partition_name,
                "r": int(r),
                "n_groups": n_groups,
                "dominant_group_by_signed_mean": dominant_signed,
                "dominant_group_by_abs_mean": dominant_abs,
                "dominance_gap_abs": dominance_gap_abs,
                "dominance_gap_norm": dominance_gap_norm,
                "interpretation_short": interpretation_short(partition_name, int(r), dominant_abs, n_groups),
                "skip_reason": skip_reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["partition_name", "r"]).reset_index(drop=True)


def plot_group_metric(
    group_stats: pd.DataFrame,
    partition_name: str,
    metric_col: str,
    err_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    sub = group_stats.loc[group_stats["partition_name"] == partition_name].copy()
    if sub.empty:
        return
    r_values = sorted(sub["r"].unique())
    if partition_name == "feature_semantics":
        group_order = ["F1", "F2", "F12"]
    else:
        group_order = ["B1", "B2", "B3"]
    x = np.arange(len(r_values), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, num=len(group_order))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for offset, group in zip(offsets, group_order):
        means: List[float] = []
        errs: List[float] = []
        for r in r_values:
            row = sub.loc[(sub["r"] == r) & (sub["group"] == group)]
            if row.empty:
                means.append(np.nan)
                errs.append(np.nan)
            else:
                means.append(float(row.iloc[0][metric_col]))
                errs.append(float(row.iloc[0][err_col]))
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            label=group,
            color=GROUP_COLORS.get(group),
            edgecolor="black",
            linewidth=0.45,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in r_values])
    ax.set_xlabel("feature-map repetitions r")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_plots(group_stats: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = ensure_dir(output_dir / "figures")
    plot_group_metric(
        group_stats,
        "feature_semantics",
        "mean_group_owen",
        "std_group_owen",
        "mean group Owen value",
        "QSVM feature-semantics partition",
        fig_dir / "qsvm_feature_semantics_group_values.png",
    )
    plot_group_metric(
        group_stats,
        "feature_semantics",
        "mean_share_abs_group_owen",
        "std_share_abs_group_owen",
        "mean absolute attribution share",
        "QSVM feature-semantics attribution shares",
        fig_dir / "qsvm_feature_semantics_group_shares.png",
    )
    plot_group_metric(
        group_stats,
        "repetition_blocks",
        "mean_group_owen",
        "std_group_owen",
        "mean group Owen value",
        "QSVM repetition-block partition",
        fig_dir / "qsvm_repetition_blocks_group_values.png",
    )
    plot_group_metric(
        group_stats,
        "repetition_blocks",
        "mean_share_abs_group_owen",
        "std_share_abs_group_owen",
        "mean absolute attribution share",
        "QSVM repetition-block attribution shares",
        fig_dir / "qsvm_repetition_blocks_group_shares.png",
    )


def format_share_text(stats_sub: pd.DataFrame) -> str:
    rows = stats_sub.sort_values("group")
    return ", ".join(
        f"{row['group']}={float(row['mean_share_abs_group_owen']):.3f}"
        for _, row in rows.iterrows()
    )


def build_interpretation_text(group_stats: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("QSVM alternative partition interpretation")
    lines.append("=" * 45)
    lines.append("")
    lines.append("Feature-semantics partition:")
    fs = summary_df.loc[summary_df["partition_name"] == "feature_semantics"].sort_values("r")
    for _, row in fs.iterrows():
        r = int(row["r"])
        sub = group_stats.loc[(group_stats["partition_name"] == "feature_semantics") & (group_stats["r"] == r)]
        dominant = str(row["dominant_group_by_abs_mean"])
        shares = format_share_text(sub)
        if dominant == "F12":
            meaning = "This suggests that the pairwise feature interaction motif is important."
        elif dominant == "F1":
            meaning = "This suggests that local single-feature encoding of x1 contributes strongly."
        elif dominant == "F2":
            meaning = "This suggests that local single-feature encoding of x2 contributes strongly."
        else:
            meaning = "The result indicates a distributed feature contribution."
        lines.append(f"r={r}: dominant group = {dominant}; shares: {shares}. {meaning}")

    lines.append("")
    lines.append("Repetition-block partition:")
    rb = summary_df.loc[summary_df["partition_name"] == "repetition_blocks"].sort_values("r")
    for _, row in rb.iterrows():
        r = int(row["r"])
        sub = group_stats.loc[(group_stats["partition_name"] == "repetition_blocks") & (group_stats["r"] == r)]
        shares = format_share_text(sub)
        if int(row["n_groups"]) < 2:
            lines.append(
                f"r={r}: only one active block, group comparison not meaningful; shares: {shares}. "
                "For r=1 the block partition is degenerate because only one active block exists."
            )
            continue
        dominant = str(row["dominant_group_by_abs_mean"])
        if dominant == "B1":
            meaning = "This suggests that the first repetition carries most of the attribution."
        else:
            meaning = "This suggests that additional feature-map repetitions add explanatory contribution."
        lines.append(f"r={r}: dominant block = {dominant}; shares: {shares}. {meaning}")

    lines.append("")
    lines.append("These summaries are descriptive controls and should be interpreted alongside the main E/M/X QSVM analysis.")
    return "\n".join(lines) + "\n"


def print_human_summary(group_stats: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\nFeature-semantics partition:")
    fs = summary_df.loc[summary_df["partition_name"] == "feature_semantics"].sort_values("r")
    for _, row in fs.iterrows():
        r = int(row["r"])
        sub = group_stats.loc[(group_stats["partition_name"] == "feature_semantics") & (group_stats["r"] == r)]
        print(f"r={r}: dominant group = {row['dominant_group_by_abs_mean']}; shares: {format_share_text(sub)}")

    print("\nRepetition-block partition:")
    rb = summary_df.loc[summary_df["partition_name"] == "repetition_blocks"].sort_values("r")
    for _, row in rb.iterrows():
        r = int(row["r"])
        sub = group_stats.loc[(group_stats["partition_name"] == "repetition_blocks") & (group_stats["r"] == r)]
        if int(row["n_groups"]) < 2:
            print(f"r={r}: only one active block, group comparison not meaningful; shares: {format_share_text(sub)}")
        else:
            print(f"r={r}: dominant block = {row['dominant_group_by_abs_mean']}; shares: {format_share_text(sub)}")


def prepare_output_dir(output_dir: Path, force: bool) -> Path:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")
    return output_dir


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = prepare_output_dir(args.output_dir, bool(args.force))
    repo_root = infer_repo_root()
    _, _, QuantumOwenValues = import_qiskit_and_qowen(repo_root)

    group_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    for r in [int(v) for v in args.r_values]:
        if bool(args.validate_simulator):
            report = validate_manual_simulator_random_subsets(r=r, num_points=8, num_random_subsets=10)
            print(f"Manual-vs-Qiskit simulator validation for r={r}: {report}")
            if float(report["max_abs_difference"]) > 1e-9:
                raise ValueError(f"Manual simulator validation failed for r={r}: {report}")

        partitions = [make_partition(name, r) for name in args.partition_names]
        for partition in partitions:
            validate_partition(partition)

        for dataset_index in [int(v) for v in args.dataset_indices]:
            print("=" * 88)
            print(f"Running QSVM alternative partitions: dataset={dataset_index}, r={r}")
            dataset = load_qsvm_dataset(data_index=dataset_index, data_dir=args.data_dir)
            passive = passive_gates_for_r(r)
            evaluator = CachedQSVMValueFunction(dataset=dataset, r=r, passive_gates=passive, svc_c=args.svc_c)
            active_all = [g for g in range(1, 7 * r + 1) if g not in passive]
            full_accuracy = evaluator.value_for_active_gates(active_all)
            baseline_accuracy = evaluator.value_for_active_gates([])
            print(f"Full active accuracy: {full_accuracy:.4f}; H-only baseline: {baseline_accuracy:.4f}")

            for partition in partitions:
                if partition.passive_gates != passive:
                    raise ValueError(f"Passive gates mismatch for {partition.partition_name} r={r}")
                phi, group_values = run_owen_for_partition(
                    partition=partition,
                    evaluator=evaluator,
                    QuantumOwenValues=QuantumOwenValues,
                    silent=True,
                )
                total_group = float(sum(group_values.values()))
                total_gate = float(sum(phi.values()))
                efficiency_target = float(full_accuracy - baseline_accuracy)
                if abs(total_group - total_gate) > 1e-9:
                    raise ValueError(
                        f"Group/gate sum mismatch for {partition.partition_name} dataset={dataset_index} r={r}: "
                        f"{total_group} vs {total_gate}"
                    )
                if abs(total_group - efficiency_target) > 1e-7:
                    print(
                        "Warning: Owen efficiency gap "
                        f"{total_group - efficiency_target:.3e} for {partition.partition_name}, "
                        f"dataset={dataset_index}, r={r}"
                    )

                group_rows.extend(
                    group_rows_for_dataset(
                        partition=partition,
                        group_values=group_values,
                        full_accuracy=full_accuracy,
                        baseline_accuracy=baseline_accuracy,
                        dataset_index=dataset_index,
                    )
                )
                gate_rows.extend(gate_rows_for_dataset(partition, phi, dataset_index))
                run_rows.append(
                    {
                        "partition_name": partition.partition_name,
                        "r": r,
                        "dataset_index": dataset_index,
                        "full_accuracy": float(full_accuracy),
                        "baseline_accuracy_if_available": float(baseline_accuracy),
                        "n_cached_value_evaluations": int(len(evaluator.cache)),
                    }
                )

    group_by_dataset = pd.DataFrame(group_rows)
    gate_by_dataset = pd.DataFrame(gate_rows)
    group_stats = build_group_stats(group_by_dataset)
    gate_stats = build_gate_stats(gate_by_dataset)
    summary = build_partition_summary(group_stats)

    group_by_dataset.to_csv(output_dir / "qsvm_alt_partition_group_values_by_dataset.csv", index=False)
    group_stats.to_csv(output_dir / "qsvm_alt_partition_group_stats.csv", index=False)
    summary.to_csv(output_dir / "qsvm_alt_partition_summary.csv", index=False)
    gate_by_dataset.to_csv(output_dir / "qsvm_alt_partition_gate_values_by_dataset.csv", index=False)
    gate_stats.to_csv(output_dir / "qsvm_alt_partition_gate_stats.csv", index=False)
    pd.DataFrame(run_rows).to_csv(output_dir / "qsvm_alt_partition_run_summaries.csv", index=False)

    if not bool(args.no_plots):
        make_plots(group_stats, output_dir)

    interpretation = build_interpretation_text(group_stats, summary)
    (output_dir / "qsvm_alt_partition_interpretation.txt").write_text(interpretation, encoding="utf-8")
    print_human_summary(group_stats, summary)
    print(f"\nSaved QSVM alternative partition outputs to {output_dir}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
