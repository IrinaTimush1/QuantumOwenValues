#!/usr/bin/env python3
"""
script_full35_estimated_study.py

Full 35-circuit estimated Owen study at fixed Owen sampling fraction
(default: 0.70).

What this script does
---------------------
1. Loads the full 35-circuit benchmark and a full gate-spec CSV.
2. Computes estimated Owen values for BOTH value functions:
      - magic
      - entanglement
   using the existing QuantumOwenValues code from the repo.
3. Saves:
      - raw JSON results
      - gate-level CSV
      - group-level CSV
      - wide circuit-summary CSV
4. Produces the thesis Figure 4-style dominant-coalition position plot,
   with zero-target value cells marked as not applicable.
5. Writes a small markdown report with the key trend statistics:
      - does M rise along the magic axis?
      - does E rise along the entanglement axis?
      - where does X become important?
      - which circuits are dominated by each coalition, excluding not-applicable cells?

Notes
-----
- This script REUSES the code already present in the repo:
    - QuantumOwenValues
    - value-function factories
    - benchmark loading
    - gate-spec loading
- It does NOT recompute exact Owen.
- It assumes you already have a FULL gate-spec CSV for all 35 circuits.
- If the gate-spec is incomplete and you pass --scaffold-missing-gates,
  the script writes a scaffold CSV for the missing circuits and exits.

Recommended use
---------------
python scripts/benchmark_35/script_full35_estimated_study.py
python scripts/benchmark_35/script_full35_estimated_study.py --sample-frac 0.7 --seed 123
python scripts/benchmark_35/script_full35_estimated_study.py --sample-frac 0.7 --repeats 5 --seed 123

If you use repeats > 1:
- the script reruns the sampled estimator with different seeds
- then averages the estimated gate Owen values across repeats
- this gives smoother group trends for the full 35-circuit study
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # repository root

IMPORT_PATHS = [
    SCRIPT_DIR,
    REPO_ROOT,
    REPO_ROOT / "qshaptools",
    REPO_ROOT / "qshaptools" / "src",
    REPO_ROOT / "qshaptools" / "src" / "qshaptools",
]

for p in IMPORT_PATHS:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

ROOT = REPO_ROOT

# Reuse existing experiment / qshaptools integration from the repo.
from script_exact_owen_benchmark15 import (  # type: ignore
    SUMMARY_CSV,
    BENCHMARK_PKL,
    BenchmarkCircuit,
    QuantumOwenValues,
    _property_value_fun_factory,
    aggregate_group_scores,
    build_benchmark_circuits,
    load_benchmark_pickle,
    load_summary,
)
from script_plot_full35_dominant_positions import (  # type: ignore
    add_display_top_group_columns,
    plot_dominant_group_positions,
)

DEFAULT_GATE_SPEC = ROOT / "data" / "benchmark_35_gate_spec.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "estimated_full35_frac70"
DEFAULT_DOMINANT_PLOT = "fig_dominant_group_positions.png"

ENTANGLERS = {"cx", "cz"}
PARAMETRIC_OR_MAGICISH = {
    "rx", "ry", "rz",
    "u", "u1", "u2", "u3",
    "p", "phase",
    "t", "tdg",
}

PROPERTY_NAMES = ["magic", "entanglement"]
GROUP_LABELS = ["E", "M", "X"]
OBSOLETE_PLOT_FILENAMES = (
    "full35_group_score_heatmaps.png",
    "full35_group_score_heatmaps.pdf",
    "full35_top_group_maps.png",
    "full35_top_group_maps.pdf",
    "full35_axis_trends.png",
    "full35_axis_trends.pdf",
    "full35_dominance_margin_maps.png",
    "full35_dominance_margin_maps.pdf",
    "dominant_coalition_positionsnew.png",
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--benchmark-pkl", type=Path, default=BENCHMARK_PKL)
    p.add_argument("--gate-spec-csv", type=Path, default=DEFAULT_GATE_SPEC)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--sample-frac",
        type=float,
        default=0.70,
        help=(
            "Fraction of the Owen sampling space to sample. "
            "Positive values use fraction-based sampling, e.g. 0.70. "
            "This replaces the old fixed-n setting."
        ),
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated sampled runs per circuit/property; estimated gate Owen values are averaged across repeats.",
    )
    p.add_argument("--allow-order-fallback", action="store_true")
    p.add_argument(
        "--scaffold-missing-gates",
        action="store_true",
        help="If the gate-spec CSV is missing some circuits, write a scaffold CSV for the missing circuits and exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Benchmark IDs / grid helpers
# ---------------------------------------------------------------------

def benchmark_sort_key(bid: str) -> Tuple[int, int, int]:
    if bid.startswith("M"):
        return (0, int(bid[1:]), 0)
    if bid.startswith("E"):
        return (1, int(bid[1:]), 0)
    if bid.startswith("I"):
        return (2, int(bid[1]), int(bid[2]))
    return (99, 0, 0)


def all_benchmark_ids(summary_df: pd.DataFrame) -> List[str]:
    ids = [str(x) for x in summary_df["benchmark_id"].tolist()]
    return sorted(ids, key=benchmark_sort_key)


def parse_grid_position(benchmark_id: str) -> Tuple[int, int]:
    """
    Map benchmark IDs to a 6x6 display grid:
      row 0, col 1..5: M1..M5
      row 1..5, col 0: E1..E5
      row 1..5, col 1..5: I11..I55
    """
    if benchmark_id.startswith("M"):
        return (0, int(benchmark_id[1:]))
    if benchmark_id.startswith("E"):
        return (int(benchmark_id[1:]), 0)
    if benchmark_id.startswith("I"):
        return (int(benchmark_id[1]), int(benchmark_id[2]))
    raise ValueError(f"Unrecognized benchmark id: {benchmark_id}")


# ---------------------------------------------------------------------
# Automatic scaffold for missing gate specs
# ---------------------------------------------------------------------

def _instr_name(instr: Any) -> str:
    return str(instr.operation.name).lower()


def _instr_qubits(instr: Any) -> List[int]:
    return [int(q._index) for q in instr.qubits]


def _shares_qubit(qs1: Sequence[int], qs2: Sequence[int]) -> bool:
    return bool(set(qs1).intersection(qs2))


def _is_adjacent_to_entangler(qc: Any, idx: int, qubits: Sequence[int]) -> bool:
    neighbors = []
    if idx - 1 >= 0:
        neighbors.append(idx - 1)
    if idx + 1 < len(qc.data):
        neighbors.append(idx + 1)

    for j in neighbors:
        instr = qc.data[j]
        if _instr_name(instr) in ENTANGLERS and _shares_qubit(qubits, _instr_qubits(instr)):
            return True
    return False


def scaffold_rows_for_circuit(item: BenchmarkCircuit) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, instr in enumerate(item.qc.data):
        name = _instr_name(instr)
        qubits = _instr_qubits(instr)
        num_qubits_acted_on = len(qubits)

        suggested_group = ""
        is_active = 0
        group = ""
        notes = ""

        if name in ENTANGLERS:
            suggested_group = "E"
            is_active = 1
            group = "E"
        elif name in PARAMETRIC_OR_MAGICISH:
            if _is_adjacent_to_entangler(item.qc, idx, qubits):
                suggested_group = "X"
                is_active = 1
                group = "X"
            else:
                suggested_group = "M"
                is_active = 1
                group = "M"
        else:
            # passive by default
            suggested_group = ""
            is_active = 0
            group = ""

        rows.append(
            {
                "benchmark_id": item.benchmark_id,
                "gate_idx": idx,
                "gate_name": name,
                "qubits": json.dumps(qubits),
                "num_qubits_acted_on": num_qubits_acted_on,
                "suggested_group": suggested_group,
                "is_active": is_active,
                "group": group,
                "notes": notes,
            }
        )
    return rows


def maybe_write_missing_gate_scaffold(
    gate_spec_csv: Path,
    benchmark_circuits: Mapping[str, BenchmarkCircuit],
) -> Optional[Path]:
    if not gate_spec_csv.exists():
        missing_ids = list(benchmark_circuits.keys())
        scaffold_rows = []
        for bid in missing_ids:
            scaffold_rows.extend(scaffold_rows_for_circuit(benchmark_circuits[bid]))
        out = gate_spec_csv.parent / "benchmark_35_gate_spec_scaffold.csv"
        pd.DataFrame(scaffold_rows).to_csv(out, index=False)
        return out

    existing = pd.read_csv(gate_spec_csv)
    have_ids = set(str(x) for x in existing["benchmark_id"].astype(str).unique())
    missing_ids = [bid for bid in benchmark_circuits.keys() if bid not in have_ids]

    if not missing_ids:
        return None

    scaffold_rows = []
    for bid in missing_ids:
        scaffold_rows.extend(scaffold_rows_for_circuit(benchmark_circuits[bid]))
    out = gate_spec_csv.parent / "benchmark_35_missing_gate_spec_scaffold.csv"
    pd.DataFrame(scaffold_rows).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------
# Estimator run
# ---------------------------------------------------------------------

def average_phi_dicts(phi_dicts: Sequence[Mapping[int, float]], gate_order: Sequence[int]) -> Dict[int, float]:
    if not phi_dicts:
        return {}
    out: Dict[int, float] = {}
    for g in gate_order:
        vals = [float(d.get(int(g), 0.0)) for d in phi_dicts]
        out[int(g)] = float(np.mean(vals))
    return out


def run_estimated_once(
    item: BenchmarkCircuit,
    spec: Dict[str, Any],
    sample_frac: float,
    seed: int,
    property_name: str,
) -> Tuple[Dict[int, float], float]:
    t0 = time.perf_counter()
    qov = QuantumOwenValues(
        qc=item.qc,
        partition=spec["partition"],
        value_fun=_property_value_fun_factory(property_name),
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=spec["locked"],
        owen_sample_frac=float(sample_frac), # existing repo convention: negative = absolute sample count
        owen_sample_reps=1,
        evaluate_value_only_once=False,
        owen_sample_seed=seed,
        name=f"{item.benchmark_id}_{property_name}_frac{sample_frac:.2f}_seed{seed}",
        silent=True,
    )
    phi = qov.run()
    runtime_s = time.perf_counter() - t0
    return {int(k): float(v) for k, v in phi.items()}, float(runtime_s)


def run_estimated_for_circuit(
    item: BenchmarkCircuit,
    spec: Dict[str, Any],
    sample_frac: float,
    seed: int,
    repeats: int,
) -> Dict[str, Any]:
    gate_order = sorted(int(x) for x in spec["active"])

    phi_magic_runs: List[Dict[int, float]] = []
    phi_ent_runs: List[Dict[int, float]] = []
    runtimes_magic: List[float] = []
    runtimes_ent: List[float] = []

    for r in range(repeats):
        s_magic = seed + r
        s_ent = seed + 10_000 + r

        
        phi_m, rt_m = run_estimated_once(item, spec, sample_frac, s_magic, "magic")
        phi_e, rt_e = run_estimated_once(item, spec, sample_frac, s_ent, "entanglement")
 
        phi_magic_runs.append(phi_m)
        phi_ent_runs.append(phi_e)
        runtimes_magic.append(rt_m)
        runtimes_ent.append(rt_e)

    phi_magic = average_phi_dicts(phi_magic_runs, gate_order)
    phi_ent = average_phi_dicts(phi_ent_runs, gate_order)

    group_magic = aggregate_group_scores(phi_magic, spec["partition"], spec["partition_labels"])
    group_ent = aggregate_group_scores(phi_ent, spec["partition"], spec["partition_labels"])

    def abs_share(scores: Mapping[str, float]) -> Dict[str, float]:
        denom = sum(abs(float(scores.get(g, 0.0))) for g in GROUP_LABELS)
        if denom == 0.0:
            return {g: 0.0 for g in GROUP_LABELS}
        return {g: abs(float(scores.get(g, 0.0))) / denom for g in GROUP_LABELS}

    return {
        "benchmark_id": item.benchmark_id,
        "family_role": item.summary_row["family_role"],
        "candidate_uid": item.summary_row.get("candidate_uid", ""),
        "sre": float(item.summary_row.get("sre", np.nan)),
        "entanglement_q": float(item.summary_row.get("entanglement_q", np.nan)),
        "sre_norm": float(item.summary_row.get("sre_norm", np.nan)),
        "ent_norm": float(item.summary_row.get("ent_norm", np.nan)),
        "sample_frac": float(sample_frac),
        "seed": int(seed),
        "repeats": int(repeats),
        "phi_magic": phi_magic,
        "phi_entanglement": phi_ent,
        "group_magic": group_magic,
        "group_entanglement": group_ent,
        "group_magic_abs_share": abs_share(group_magic),
        "group_entanglement_abs_share": abs_share(group_ent),
        "runtime_magic_mean_s": float(np.mean(runtimes_magic)),
        "runtime_entanglement_mean_s": float(np.mean(runtimes_ent)),
        "runtime_total_mean_s": float(np.mean(runtimes_magic) + np.mean(runtimes_ent)),
        "runtime_magic_std_s": float(np.std(runtimes_magic, ddof=1)) if repeats > 1 else 0.0,
        "runtime_entanglement_std_s": float(np.std(runtimes_ent, ddof=1)) if repeats > 1 else 0.0,
    }


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def top_and_margin(scores: Mapping[str, float]) -> Tuple[str, float]:
    ordered = sorted(((g, float(scores.get(g, 0.0))) for g in GROUP_LABELS), key=lambda x: x[1], reverse=True)
    top_group = ordered[0][0]
    top_score = ordered[0][1]
    second_score = ordered[1][1]
    return top_group, float(top_score - second_score)


def build_gate_rows(
    results: Sequence[Mapping[str, Any]],
    gate_spec: Mapping[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for res in results:
        bid = str(res["benchmark_id"])
        spec = gate_spec[bid]
        group_by_gate = {}
        for group_label, group_members in zip(spec["partition_labels"], spec["partition"]):
            for g in group_members:
                group_by_gate[int(g)] = str(group_label)

        for gate_idx, phi in res["phi_magic"].items():
            rows.append(
                {
                    "benchmark_id": bid,
                    "family_role": res["family_role"],
                    "property": "magic",
                    "gate_idx": int(gate_idx),
                    "group": group_by_gate.get(int(gate_idx), ""),
                    "phi": float(phi),
                }
            )
        for gate_idx, phi in res["phi_entanglement"].items():
            rows.append(
                {
                    "benchmark_id": bid,
                    "family_role": res["family_role"],
                    "property": "entanglement",
                    "gate_idx": int(gate_idx),
                    "group": group_by_gate.get(int(gate_idx), ""),
                    "phi": float(phi),
                }
            )
    return pd.DataFrame(rows)


def build_group_rows(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for res in results:
        bid = str(res["benchmark_id"])
        for prop_name, key_scores, key_share in [
            ("magic", "group_magic", "group_magic_abs_share"),
            ("entanglement", "group_entanglement", "group_entanglement_abs_share"),
        ]:
            scores = res[key_scores]
            shares = res[key_share]
            for g in GROUP_LABELS:
                rows.append(
                    {
                        "benchmark_id": bid,
                        "family_role": res["family_role"],
                        "property": prop_name,
                        "group": g,
                        "score": float(scores.get(g, 0.0)),
                        "abs_share": float(shares.get(g, 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def build_circuit_summary(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for res in results:
        bid = str(res["benchmark_id"])
        row, col = parse_grid_position(bid)

        top_magic, margin_magic = top_and_margin(res["group_magic"])
        top_ent, margin_ent = top_and_margin(res["group_entanglement"])

        row_dict = {
            "benchmark_id": bid,
            "family_role": res["family_role"],
            "grid_row": row,
            "grid_col": col,
            "candidate_uid": res["candidate_uid"],
            "sre": res["sre"],
            "entanglement_q": res["entanglement_q"],
            "sre_norm": res["sre_norm"],
            "ent_norm": res["ent_norm"],
            "sample_frac": res["sample_frac"],
            "repeats": res["repeats"],
            "runtime_magic_mean_s": res["runtime_magic_mean_s"],
            "runtime_entanglement_mean_s": res["runtime_entanglement_mean_s"],
            "runtime_total_mean_s": res["runtime_total_mean_s"],
            "top_group_magic": top_magic,
            "top_margin_magic": margin_magic,
            "top_group_entanglement": top_ent,
            "top_margin_entanglement": margin_ent,
        }

        for g in GROUP_LABELS:
            row_dict[f"magic_{g}"] = float(res["group_magic"].get(g, 0.0))
            row_dict[f"entanglement_{g}"] = float(res["group_entanglement"].get(g, 0.0))
            row_dict[f"magic_abs_share_{g}"] = float(res["group_magic_abs_share"].get(g, 0.0))
            row_dict[f"entanglement_abs_share_{g}"] = float(res["group_entanglement_abs_share"].get(g, 0.0))

        rows.append(row_dict)

    return pd.DataFrame(rows).sort_values(["grid_row", "grid_col"])


# ---------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------

def spearman_simple(x: Sequence[float], y: Sequence[float]) -> float:
    sx = pd.Series(list(x)).rank(method="average")
    sy = pd.Series(list(y)).rank(method="average")
    return float(sx.corr(sy, method="pearson"))


def analyze_trends(summary_df: pd.DataFrame) -> str:
    display_df = add_display_top_group_columns(summary_df)

    lines: List[str] = []
    lines.append("# Full 35-circuit estimated Owen study")
    lines.append("")
    lines.append(f"- Number of circuits: {len(summary_df)}")
    lines.append(f"- Owen sampling fraction: {float(summary_df['sample_frac'].iloc[0]):.2f}")
    lines.append(f"- Repeats per circuit/property: {int(summary_df['repeats'].iloc[0])}")
    lines.append("")

    # Magic axis: M1..M5, magic value function, M coalition
    magic_axis = summary_df.loc[summary_df["benchmark_id"].str.startswith("M")].copy()
    magic_axis["axis_idx"] = magic_axis["benchmark_id"].str[1:].astype(int)
    magic_axis = magic_axis.sort_values("axis_idx")

    rho_magic = spearman_simple(magic_axis["axis_idx"], magic_axis["magic_M"])
    lines.append("## Axis trend checks")
    lines.append("")
    lines.append(f"- Spearman correlation between magic-axis position and **Magic coalition score for the magic value**: **{rho_magic:.4f}**")

    ent_axis = summary_df.loc[summary_df["benchmark_id"].str.startswith("E")].copy()
    ent_axis["axis_idx"] = ent_axis["benchmark_id"].str[1:].astype(int)
    ent_axis = ent_axis.sort_values("axis_idx")
    rho_ent = spearman_simple(ent_axis["axis_idx"], ent_axis["entanglement_E"])
    lines.append(f"- Spearman correlation between entanglement-axis position and **Entanglement coalition score for the entanglement value**: **{rho_ent:.4f}**")
    lines.append("")

    # Top-group counts. Use the same zero-target/not-applicable rule as the
    # thesis Figure 4 plot so the report and visual agree.
    lines.append("## Dominant coalition counts")
    lines.append("")
    lines.append("Counts exclude circuits whose target value is zero and therefore not applicable.")
    lines.append("")
    mg_counts = display_df["display_top_group_magic"].value_counts().reindex(GROUP_LABELS, fill_value=0)
    eg_counts = display_df["display_top_group_entanglement"].value_counts().reindex(GROUP_LABELS, fill_value=0)
    mg_na = int((display_df["display_top_group_magic"] == "NA").sum())
    eg_na = int((display_df["display_top_group_entanglement"] == "NA").sum())

    lines.append("### Magic value")
    for g in GROUP_LABELS:
        lines.append(f"- {g}: {int(mg_counts[g])} circuits")
    lines.append(f"- Not applicable: {mg_na} circuits")
    lines.append("")
    lines.append("### Entanglement value")
    for g in GROUP_LABELS:
        lines.append(f"- {g}: {int(eg_counts[g])} circuits")
    lines.append(f"- Not applicable: {eg_na} circuits")
    lines.append("")

    # Where does X become important?
    lines.append("## Where does Mix (X) become important?")
    lines.append("")
    top_magic_x = summary_df.sort_values("magic_X", ascending=False).head(5)[["benchmark_id", "magic_X", "magic_abs_share_X"]]
    top_ent_x = summary_df.sort_values("entanglement_X", ascending=False).head(5)[["benchmark_id", "entanglement_X", "entanglement_abs_share_X"]]

    lines.append("### Highest X score for the magic value")
    for _, row in top_magic_x.iterrows():
        lines.append(f"- {row['benchmark_id']}: score={row['magic_X']:.6f}, abs-share={row['magic_abs_share_X']:.3f}")
    lines.append("")
    lines.append("### Highest X score for the entanglement value")
    for _, row in top_ent_x.iterrows():
        lines.append(f"- {row['benchmark_id']}: score={row['entanglement_X']:.6f}, abs-share={row['entanglement_abs_share_X']:.3f}")
    lines.append("")

    # Strongest dominance margins
    lines.append("## Strongest dominance margins")
    lines.append("")
    top_dom_magic = summary_df.sort_values("top_margin_magic", ascending=False).head(5)[["benchmark_id", "top_group_magic", "top_margin_magic"]]
    top_dom_ent = summary_df.sort_values("top_margin_entanglement", ascending=False).head(5)[["benchmark_id", "top_group_entanglement", "top_margin_entanglement"]]

    lines.append("### Magic value")
    for _, row in top_dom_magic.iterrows():
        lines.append(f"- {row['benchmark_id']}: top={row['top_group_magic']}, margin={row['top_margin_magic']:.6f}")
    lines.append("")
    lines.append("### Entanglement value")
    for _, row in top_dom_ent.iterrows():
        lines.append(f"- {row['benchmark_id']}: top={row['top_group_entanglement']}, margin={row['top_margin_entanglement']:.6f}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Thesis plot housekeeping
# ---------------------------------------------------------------------

def remove_obsolete_plot_files(output_dir: Path) -> None:
    for filename in OBSOLETE_PLOT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def load_gate_spec_full_study(
    gate_spec_csv: Path,
    benchmark_circuits: Mapping[str, BenchmarkCircuit],
) -> Dict[str, Dict[str, Any]]:
    """
    Load gate-spec CSV for the full estimated 35-circuit study.

    Unlike the exact-15 loader, this function does NOT enforce a small active-player limit.
    It only validates consistency and constructs the partition/locked-gate metadata needed by
    QuantumOwenValues.
    """
    if not gate_spec_csv.exists():
        raise FileNotFoundError(f"Missing gate-spec CSV: {gate_spec_csv}")

    df = pd.read_csv(gate_spec_csv)
    required_cols = {
        "benchmark_id",
        "gate_idx",
        "gate_name",
        "is_active",
        "group",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Gate-spec CSV is missing required columns: {sorted(missing)}"
        )

    out: Dict[str, Dict[str, Any]] = {}

    for benchmark_id, item in benchmark_circuits.items():
        sub = df.loc[df["benchmark_id"].astype(str) == str(benchmark_id)].copy()
        if sub.empty:
            raise ValueError(f"Missing gate-spec rows for circuit {benchmark_id}")

        sub["gate_idx"] = sub["gate_idx"].astype(int)
        sub["is_active"] = sub["is_active"].fillna(0).astype(int)
        sub["group"] = sub["group"].fillna("").astype(str).str.strip()

        sub = sub.sort_values("gate_idx").reset_index(drop=True)

        expected_gate_indices = list(range(len(item.qc.data)))
        actual_gate_indices = sub["gate_idx"].tolist()
        if actual_gate_indices != expected_gate_indices:
            raise ValueError(
                f"{benchmark_id}: gate_idx mismatch.\n"
                f"Expected: {expected_gate_indices}\n"
                f"Actual:   {actual_gate_indices}"
            )

        active = sub.loc[sub["is_active"] == 1, "gate_idx"].astype(int).tolist()
        if len(active) == 0:
            raise ValueError(f"{benchmark_id}: no active players found.")

        # Build partition in fixed E/M/X order, skipping empty groups if needed.
        partition_labels: List[str] = []
        partition: List[List[int]] = []

        for label in ["E", "M", "X"]:
            members = sub.loc[
                (sub["is_active"] == 1) & (sub["group"] == label),
                "gate_idx"
            ].astype(int).tolist()
            if members:
                partition_labels.append(label)
                partition.append(members)

        assigned_active = sorted(g for group in partition for g in group)
        if sorted(active) != assigned_active:
            missing_active = sorted(set(active) - set(assigned_active))
            extra_assigned = sorted(set(assigned_active) - set(active))
            raise ValueError(
                f"{benchmark_id}: active/group assignment mismatch.\n"
                f"Active gates: {sorted(active)}\n"
                f"Assigned in groups: {assigned_active}\n"
                f"Missing active assignments: {missing_active}\n"
                f"Extra assigned gates: {extra_assigned}"
            )

        locked = [int(i) for i in expected_gate_indices if i not in active]

        out[benchmark_id] = {
            "active": active,
            "locked": locked,
            "partition": partition,
            "partition_labels": partition_labels,
            "gate_spec_df": sub.copy(),
        }

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary(args.summary_csv)
    benchmark_ids = all_benchmark_ids(summary_df)

    benchmark_map = load_benchmark_pickle(
        args.benchmark_pkl,
        summary_df,
        allow_order_fallback=args.allow_order_fallback,
    )
    benchmark_circuits = build_benchmark_circuits(summary_df, benchmark_map, benchmark_ids)

    scaffold_path = maybe_write_missing_gate_scaffold(args.gate_spec_csv, benchmark_circuits)
    if scaffold_path is not None and args.scaffold_missing_gates:
        print(f"Wrote missing gate-spec scaffold to: {scaffold_path}")
        print("Please review/merge it into your full gate-spec CSV, then rerun.")
        return

    if not args.gate_spec_csv.exists():
        raise FileNotFoundError(
            f"Missing full gate-spec CSV: {args.gate_spec_csv}\n"
            "Use --scaffold-missing-gates once to write a scaffold, then review it."
        )
    
    gate_spec = load_gate_spec_full_study(args.gate_spec_csv, benchmark_circuits)

    results: List[Dict[str, Any]] = []
    for benchmark_id in benchmark_ids:
        print(f"Running estimated Owen for {benchmark_id} ...")
        result = run_estimated_for_circuit(
            item=benchmark_circuits[benchmark_id],
            spec=gate_spec[benchmark_id],
            sample_frac=args.sample_frac,
            seed=args.seed,
            repeats=args.repeats,
        )
        results.append(result)

    # Save raw JSON
    raw_json = args.output_dir / "estimated_full35_results.json"
    raw_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save tabular outputs
    gate_rows = build_gate_rows(results, gate_spec)
    gate_csv = args.output_dir / "estimated_full35_gate_scores.csv"
    gate_rows.to_csv(gate_csv, index=False)

    group_rows = build_group_rows(results)
    group_csv = args.output_dir / "estimated_full35_group_scores.csv"
    group_rows.to_csv(group_csv, index=False)

    summary_wide = build_circuit_summary(results)
    summary_csv = args.output_dir / "estimated_full35_circuit_summary.csv"
    summary_wide.to_csv(summary_csv, index=False)

    # Analysis report
    report_text = analyze_trends(summary_wide)
    report_md = args.output_dir / "estimated_full35_trend_report.md"
    report_md.write_text(report_text, encoding="utf-8")

    # Thesis Figure 4 plot. Remove older exploratory plot outputs first so a
    # rerun leaves only the submission figure in the result directory.
    remove_obsolete_plot_files(args.output_dir)
    dominant_plot = args.output_dir / DEFAULT_DOMINANT_PLOT
    plot_dominant_group_positions(summary_wide, dominant_plot)

    print(f"\nSaved raw JSON to: {raw_json}")
    print(f"Saved gate CSV to: {gate_csv}")
    print(f"Saved group CSV to: {group_csv}")
    print(f"Saved circuit summary CSV to: {summary_csv}")
    print(f"Saved trend report to: {report_md}")
    print(f"Saved dominant-position plot to: {dominant_plot}")
    print("\nDone.")


if __name__ == "__main__":
    main()
