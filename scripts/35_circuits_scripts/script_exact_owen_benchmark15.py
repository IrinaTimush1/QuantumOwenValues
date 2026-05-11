#!/usr/bin/env python3
"""
script_exact_owen_benchmark15.py

Exact-Owen experiment runner for the 15-circuit "small" benchmark subset.

What this script does
---------------------
1) Reads `data/benchmark_35_summary.csv` and extracts the 15 circuits with
   `selection_mode == "exact"`.
2) Loads `data/benchmark_35.pkl` and reconstructs a Qiskit circuit for each
   selected benchmark item.
3) Writes a *manual gate-spec template CSV* if one does not exist yet.
   This is where you decide:
      - which gates are ACTIVE players,
      - which gates are PASSIVE / always-on locked gates,
      - which ACTIVE gates belong to group E / M / X.
4) After you fill in that CSV, the script validates the specification.
5) For each circuit, it computes:
      - exact Owen values for the magic value function (SRE),
      - exact Owen values for the entanglement value function (Meyer-Wallach),
      - per-group sums,
      - per-circuit plots,
      - CSV/JSON outputs.

Design principle
----------------
This script does *not* silently infer your thesis partition. It only offers
a weak *suggested_group* column in the scaffold CSV. The final active/passive
and E/M/X decisions must be written explicitly by you in the CSV.

Usage
-----
First run:
    python script_exact_owen_benchmark15.py --scaffold-only

This creates:
    data/benchmark_15_exact_gate_spec.csv

Then edit that CSV manually.

Then run:
    python script_exact_owen_benchmark15.py

Outputs will be written to:
    data/plots/exact_owen_15/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qiskit.converters import circuit_to_dag

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
QSHAPTOOLS_SRC = ROOT / "qshaptools" / "src"
if str(QSHAPTOOLS_SRC) not in sys.path:
    sys.path.insert(0, str(QSHAPTOOLS_SRC))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402

from qshaptools.qowen import QuantumOwenValues  # noqa: E402
from qshaptools.tools import build_circuit  # noqa: E402

SUMMARY_CSV = ROOT / "data" / "benchmark_35_from_pool_summary.csv"
BENCHMARK_PKL = ROOT / "data" / "benchmark_35_from_pool.pkl"
GATE_SPEC_CSV = ROOT / "data" / "benchmark_15_exact_gate_spec.csv"
OUTPUT_DIR = ROOT / "data" / "plots" / "exact_owen_15"

VALID_GROUPS = {"E", "M", "X"}

ENTANGLER_GATES = {"cx", "cz"}
LOCAL_CLIFFORD_1Q = {"h", "s", "sdg", "x", "y", "z"}


@dataclass
class BenchmarkCircuit:
    benchmark_id: str
    summary_row: Dict[str, Any]
    circuit_info: Dict[str, Any]
    qc: QuantumCircuit


def load_summary(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")

    df = pd.read_csv(summary_csv)

    required = {
        "benchmark_id",
        "selection_mode",
        "family_role",
        "candidate_uid",
        "sre",
        "entanglement_q",
        "sre_norm",
        "ent_norm",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Summary CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if "gate_count" not in df.columns:
        if "num_gates" in df.columns:
            df = df.copy()
            df["gate_count"] = df["num_gates"]
        else:
            raise ValueError(
                "Summary CSV must contain either 'gate_count' or 'num_gates'."
            )

    return df


def get_exact_ids(summary_df: pd.DataFrame) -> List[str]:
    exact_df = summary_df.loc[summary_df["selection_mode"].astype(str).str.lower() == "exact"]
    ids = exact_df["benchmark_id"].astype(str).tolist()
    if len(ids) != 15:
        raise ValueError(
            f"Expected exactly 15 exact benchmark circuits, found {len(ids)}: {ids}"
        )
    return ids


def _normalize_pickled_record(record: Any) -> Dict[str, Any]:
    """
    Normalize one benchmark pickle item into a dict-like object.

    Supported input shapes:
      - raw circuit dict with qasm/num_qubits
      - tuple(raw_dict, sre)
      - tuple(raw_dict, sre, entanglement)
      - dict with nested circuit info under common keys
    """
    if isinstance(record, tuple):
        if len(record) >= 1 and isinstance(record[0], dict):
            base = dict(record[0])
            if len(record) >= 2:
                base.setdefault("sre", record[1])
            if len(record) >= 3:
                base.setdefault("entanglement_q", record[2])
            return base
        raise ValueError(
            "Unsupported tuple record in benchmark pickle. "
            "Expected (circuit_dict, ...)."
        )

    if isinstance(record, dict):
        if "qasm" in record and "num_qubits" in record:
            return dict(record)

        for nested_key in ("circuit_info", "circuit", "record", "payload", "raw"):
            nested = record.get(nested_key)
            if isinstance(nested, dict) and "qasm" in nested and "num_qubits" in nested:
                merged = dict(nested)
                for k, v in record.items():
                    if k != nested_key:
                        merged.setdefault(k, v)
                return merged

    raise ValueError(
        "Could not normalize benchmark pickle record. "
        "Expected a dict containing qasm/num_qubits, or a tuple whose first item is such a dict."
    )


def load_benchmark_pickle(
    benchmark_pkl: Path,
    summary_df: pd.DataFrame,
    allow_order_fallback: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a map benchmark_id -> normalized circuit_info.

    Matching strategy:
      1) benchmark_id in record
      2) candidate_uid in record matched through summary CSV
      3) optional order fallback (disabled by default)
    """
    if not benchmark_pkl.exists():
        raise FileNotFoundError(f"Missing benchmark pickle: {benchmark_pkl}")

    with open(benchmark_pkl, "rb") as f:
        payload = pickle.load(f)

    normalized_records: List[Dict[str, Any]] = []
    if isinstance(payload, Mapping):
        iterator = payload.values()
    elif isinstance(payload, Sequence):
        iterator = payload
    else:
        raise ValueError(
            f"Unsupported benchmark pickle top-level type: {type(payload).__name__}"
        )

    for raw in iterator:
        normalized_records.append(_normalize_pickled_record(raw))

    summary_by_candidate = {
        str(row["candidate_uid"]): str(row["benchmark_id"])
        for _, row in summary_df.iterrows()
    }

    result: Dict[str, Dict[str, Any]] = {}
    unmatched: List[Dict[str, Any]] = []

    for rec in normalized_records:
        rec_benchmark_id = rec.get("benchmark_id")
        rec_candidate_uid = rec.get("candidate_uid")

        if rec_benchmark_id is not None:
            result[str(rec_benchmark_id)] = rec
            continue

        if rec_candidate_uid is not None and str(rec_candidate_uid) in summary_by_candidate:
            result[summary_by_candidate[str(rec_candidate_uid)]] = rec
            continue

        unmatched.append(rec)

    if unmatched:
        if not allow_order_fallback:
            sample_keys = sorted(unmatched[0].keys()) if unmatched else []
            raise ValueError(
                "Some benchmark pickle records could not be matched to benchmark_id "
                "via benchmark_id or candidate_uid.\n"
                f"Unmatched count: {len(unmatched)}\n"
                f"Example keys: {sample_keys}\n"
                "If you know benchmark_35.pkl is in the same order as benchmark_35_summary.csv, "
                "rerun with --allow-order-fallback."
            )

        if len(normalized_records) != len(summary_df):
            raise ValueError(
                "Order fallback requested, but pickle length does not match summary length: "
                f"{len(normalized_records)} vs {len(summary_df)}"
            )

        result.clear()
        for (_, row), rec in zip(summary_df.iterrows(), normalized_records):
            result[str(row["benchmark_id"])] = rec

    return result


def build_benchmark_circuits(
    summary_df: pd.DataFrame,
    benchmark_map: Dict[str, Dict[str, Any]],
    selected_ids: Sequence[str],
) -> Dict[str, BenchmarkCircuit]:
    summary_index = summary_df.set_index("benchmark_id", drop=False)
    out: Dict[str, BenchmarkCircuit] = {}

    for benchmark_id in selected_ids:
        if benchmark_id not in benchmark_map:
            raise KeyError(
                f"benchmark_id={benchmark_id} not found in benchmark pickle after matching."
            )
        if benchmark_id not in summary_index.index:
            raise KeyError(f"benchmark_id={benchmark_id} not found in summary CSV.")

        summary_row = summary_index.loc[benchmark_id].to_dict()
        circuit_info = benchmark_map[benchmark_id]
        qasm = circuit_info.get("qasm")
        if not isinstance(qasm, str) or not qasm.strip():
            raise ValueError(f"{benchmark_id}: missing or empty qasm in benchmark record.")

        qc = QuantumCircuit.from_qasm_str(qasm)
        out[benchmark_id] = BenchmarkCircuit(
            benchmark_id=benchmark_id,
            summary_row=summary_row,
            circuit_info=circuit_info,
            qc=qc,
        )
    return out


def qubit_indices(qc: QuantumCircuit, qargs: Sequence[Any]) -> List[int]:
    idxs: List[int] = []
    for q in qargs:
        try:
            idxs.append(int(qc.find_bit(q).index))
        except Exception:
            idxs.append(int(getattr(q, "index")))
    return idxs


def is_non_clifford_single_qubit_gate(instr_name: str, instr_params: Sequence[Any]) -> bool:
    """
    Very conservative heuristic for scaffold suggestions only.

    We do NOT use this as the final thesis partition.
    """
    name = instr_name.lower()
    if name in {"t", "tdg"}:
        return True
    if name in {"rx", "ry", "rz", "p", "u", "u1", "u2", "u3"}:
        # Treat parameterized one-qubit rotations/phases as potentially magic-contributing.
        return True
    return False


def suggest_group_for_gate(
    qc: QuantumCircuit,
    idx: int,
) -> str:
    """
    Weak suggestion for the scaffold CSV.

    Rules:
      - 2-qubit entanglers -> E
      - non-Clifford-like 1-qubit rotations -> M
      - if such a 1-qubit gate is immediately adjacent to a 2-qubit gate -> X
      - else blank
    """
    instr, qargs, _ = qc.data[idx]
    q_count = len(qargs)
    name = instr.name.lower()

    if q_count >= 2:
        return "E"

    if not is_non_clifford_single_qubit_gate(name, instr.params):
        return ""

    prev_twoq = idx > 0 and len(qc.data[idx - 1][1]) >= 2
    next_twoq = idx + 1 < len(qc.data) and len(qc.data[idx + 1][1]) >= 2
    if prev_twoq or next_twoq:
        return "X"
    return "M"


def write_gate_spec_template(
    benchmark_circuits: Dict[str, BenchmarkCircuit],
    spec_csv: Path,
    overwrite: bool = False,
) -> None:
    if spec_csv.exists() and not overwrite:
        print(f"Gate-spec CSV already exists, not overwriting: {spec_csv}")
        return

    rows: List[Dict[str, Any]] = []
    for benchmark_id, item in benchmark_circuits.items():
        qc = item.qc
        for idx, (instr, qargs, cargs) in enumerate(qc.data):
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "gate_idx": idx,
                    "gate_name": instr.name,
                    "qubits": json.dumps(qubit_indices(qc, qargs)),
                    "num_qubits_acted_on": len(qargs),
                    "suggested_group": suggest_group_for_gate(qc, idx),
                    "is_active": "",
                    "group": "",
                    "notes": "",
                }
            )

    spec_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(spec_csv, index=False)
    print(f"Wrote manual gate-spec template to: {spec_csv}")
    print(
        "Next step: fill in `is_active` with 0/1 and set `group` to E/M/X for active gates."
    )


def parse_bool_like(x: Any) -> Optional[bool]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    if s == "":
        return None
    raise ValueError(f"Could not parse boolean-like value: {x!r}")


def load_gate_spec(
    spec_csv: Path,
    benchmark_circuits: Dict[str, BenchmarkCircuit],
) -> Dict[str, Dict[str, Any]]:
    if not spec_csv.exists():
        raise FileNotFoundError(
            f"Missing gate spec CSV: {spec_csv}\n"
            "Run this script once with --scaffold-only, then fill in the CSV."
        )

    spec_df = pd.read_csv(spec_csv)
    required = {"benchmark_id", "gate_idx", "is_active", "group"}
    missing = required - set(spec_df.columns)
    if missing:
        raise ValueError(
            "Gate-spec CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    spec_by_circuit: Dict[str, Dict[str, Any]] = {}
    for benchmark_id, item in benchmark_circuits.items():
        qc = item.qc
        sub = spec_df.loc[spec_df["benchmark_id"].astype(str) == str(benchmark_id)].copy()
        if len(sub) != len(qc.data):
            raise ValueError(
                f"{benchmark_id}: gate-spec row count ({len(sub)}) does not match "
                f"number of circuit instructions ({len(qc.data)})."
            )

        sub["gate_idx"] = sub["gate_idx"].astype(int)
        sub = sub.sort_values("gate_idx").reset_index(drop=True)

        expected = list(range(len(qc.data)))
        got = sub["gate_idx"].tolist()
        if got != expected:
            raise ValueError(
                f"{benchmark_id}: gate_idx must be exactly {expected}, got {got}"
            )

        active: List[int] = []
        locked: List[int] = []
        partition_map: Dict[str, List[int]] = {"E": [], "M": [], "X": []}

        for _, row in sub.iterrows():
            idx = int(row["gate_idx"])
            is_active = parse_bool_like(row["is_active"])
            group_raw = "" if pd.isna(row["group"]) else str(row["group"]).strip().upper()

            if is_active is None:
                raise ValueError(
                    f"{benchmark_id}: gate {idx} has empty is_active. Fill 0 or 1."
                )

            if is_active:
                if group_raw not in VALID_GROUPS:
                    raise ValueError(
                        f"{benchmark_id}: gate {idx} is active but group={group_raw!r} "
                        f"is not one of {sorted(VALID_GROUPS)}."
                    )
                active.append(idx)
                partition_map[group_raw].append(idx)
            else:
                if group_raw not in {"", "NAN"}:
                    raise ValueError(
                        f"{benchmark_id}: gate {idx} is passive/locked but has non-empty "
                        f"group={group_raw!r}. Leave group blank for passive gates."
                    )
                locked.append(idx)

        if not (1 <= len(active) <= 8):
            raise ValueError(
                f"{benchmark_id}: expected 1..8 active players for exact Owen, found {len(active)}."
                )
        if len(active) < 6:
            print(
                f"Warning: {benchmark_id} has only {len(active)} active players. "
                "This is still fine for exact Owen, but it differs from the original 6-8 target."
                )
        
        partition_labels = [lab for lab in ["E", "M", "X"] if len(partition_map[lab]) > 0]
        partition = [partition_map[lab] for lab in partition_labels]

        covered = sorted(active)
        flat_partition = sorted([g for grp in partition for g in grp])
        if covered != flat_partition:
            raise ValueError(
                f"{benchmark_id}: partition does not exactly cover active gates.\n"
                f"active={covered}\npartition={flat_partition}"
            )

        spec_by_circuit[benchmark_id] = {
            "active": active,
            "locked": locked,
            "partition": partition,
            "partition_labels": partition_labels,
            "spec_table": sub,
        }   
    return spec_by_circuit


def calculate_stabilizer_renyi_entropy_qiskit(
    circuit: QuantumCircuit,
    alpha: int = 2,
) -> float:
    """
    Mirrors the QC_dataset_generation implementation for consistency.

    For n qubits with d = 2**n:
      xi_P = (1/d) * <psi|P|psi>^2
      A = sum_P xi_P^alpha
      S_alpha = (1/(1-alpha)) * log(A) - log(d)
    """
    n = circuit.num_qubits
    d = 2 ** n
    pauli_gates = ["I", "X", "Y", "Z"]
    statevector = Statevector.from_instruction(circuit)

    A = 0.0
    for combination in product(pauli_gates, repeat=n):
        pauli_circuit = QuantumCircuit(n)
        for qubit, gate in enumerate(combination):
            if gate == "X":
                pauli_circuit.x(qubit)
            elif gate == "Y":
                pauli_circuit.y(qubit)
            elif gate == "Z":
                pauli_circuit.z(qubit)
        pauli_operator = Operator(pauli_circuit)
        exp_val = float(np.real(statevector.expectation_value(pauli_operator)))
        xi_p = (1.0 / d) * (exp_val ** 2)
        A += xi_p ** alpha

    entropy = (1.0 / (1.0 - alpha)) * np.log(A) - np.log(d)
    return float(entropy)


def _single_qubit_reduced_density_matrix(
    statevector: np.ndarray,
    num_qubits: int,
    target_qubit: int,
) -> np.ndarray:
    psi = np.asarray(statevector, dtype=np.complex128)
    expected_len = 2 ** num_qubits
    if psi.ndim != 1 or psi.size != expected_len:
        raise ValueError(
            f"Statevector must have length 2**num_qubits = {expected_len}, got {psi.shape}."
        )
    tensor = psi.reshape([2] * num_qubits)
    psi_target_first = np.moveaxis(tensor, target_qubit, 0).reshape(2, -1)
    rho = psi_target_first @ np.conjugate(psi_target_first.T)
    return rho


def calculate_meyer_wallach_entanglement_qiskit(circuit: QuantumCircuit) -> float:
    statevector = Statevector.from_instruction(circuit).data
    n = circuit.num_qubits
    purities: List[float] = []
    for q in range(n):
        rho_q = _single_qubit_reduced_density_matrix(statevector, n, q)
        purities.append(float(np.real(np.trace(rho_q @ rho_q))))
    q_value = 2.0 * (1.0 - float(np.mean(purities)))
    return float(np.clip(q_value, 0.0, 1.0))


def _property_value_fun_factory(kind: str):
    kind = kind.lower().strip()
    if kind not in {"magic", "entanglement"}:
        raise ValueError(f"Unknown property kind: {kind}")

    def _value_fun(
        qc_data,
        num_qubits: int,
        S: Sequence[int],
        quantum_instance=None,
        **kwargs,
    ) -> float:
        # In QuantumOwenValues, S already includes locked gates before the callback is invoked.
        qc, _ = build_circuit(qc_data=qc_data, num_qubits=num_qubits, S=list(S), cl_bits=False)
        if kind == "magic":
            return calculate_stabilizer_renyi_entropy_qiskit(qc)
        return calculate_meyer_wallach_entanglement_qiskit(qc)

    return _value_fun


def aggregate_group_scores(
    phi_dict: Dict[int, float],
    partition: Sequence[Sequence[int]],
    labels: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, group in zip(labels, partition):
        out[label] = float(sum(phi_dict[g] for g in group))
    return out


def plot_group_bars(
    benchmark_id: str,
    magic_group_scores: Dict[str, float],
    ent_group_scores: Dict[str, float],
    out_path: Path,
) -> None:
    labels = list(magic_group_scores.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, [magic_group_scores[k] for k in labels], width, label="magic (SRE)")
    ax.bar(x + width / 2, [ent_group_scores[k] for k in labels], width, label="entanglement (MW)")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("group-summed Owen value")
    ax.set_title(f"{benchmark_id}: exact Owen group scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_exact_owen_for_circuit(
    item: BenchmarkCircuit,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    partition = spec["partition"]
    locked = spec["locked"]
    labels = spec["partition_labels"]

    qov_magic = QuantumOwenValues(
        qc=item.qc,
        partition=partition,
        value_fun=_property_value_fun_factory("magic"),
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=locked,
        owen_sample_frac=None,   # exact mode
        owen_sample_reps=1,
        evaluate_value_only_once=True,
        owen_sample_seed=123,
        name=f"{item.benchmark_id}_magic_exact",
        silent=True,
    )
    phi_magic = qov_magic.run()
    group_magic = aggregate_group_scores(phi_magic, partition, labels)

    qov_ent = QuantumOwenValues(
        qc=item.qc,
        partition=partition,
        value_fun=_property_value_fun_factory("entanglement"),
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=locked,
        owen_sample_frac=None,   # exact mode
        owen_sample_reps=1,
        evaluate_value_only_once=True,
        owen_sample_seed=123,
        name=f"{item.benchmark_id}_ent_exact",
        silent=True,
    )
    phi_ent = qov_ent.run()
    group_ent = aggregate_group_scores(phi_ent, partition, labels)

    return {
        "benchmark_id": item.benchmark_id,
        "family_role": item.summary_row["family_role"],
        "gate_count": int(item.summary_row["gate_count"]),
        "active_count": len(spec["active"]),
        "active": spec["active"],
        "locked": locked,
        "partition": {
            label: group
            for label, group in zip(spec["partition_labels"], spec["partition"])
        },
        "summary_targets": {
            "target_magic_level": item.summary_row.get("target_magic_level"),
            "target_ent_level": item.summary_row.get("target_ent_level"),
            "sre": item.summary_row.get("sre"),
            "entanglement_q": item.summary_row.get("entanglement_q"),
            "sre_norm": item.summary_row.get("sre_norm"),
            "ent_norm": item.summary_row.get("ent_norm"),
        },
        "phi_magic": {int(k): float(v) for k, v in phi_magic.items()},
        "phi_entanglement": {int(k): float(v) for k, v in phi_ent.items()},
        "group_magic": group_magic,
        "group_entanglement": group_ent,
        "top_group_magic": max(group_magic, key=group_magic.get),
        "top_group_entanglement": max(group_ent, key=group_ent.get),
    }


def save_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
    benchmark_circuit_lookup: Dict[str, BenchmarkCircuit],
    gate_spec_lookup: Dict[str, Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "exact_owen_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    group_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []

    for item in results:
        benchmark_id = item["benchmark_id"]
        spec = gate_spec_lookup[benchmark_id]
        benchmark_obj = benchmark_circuit_lookup[benchmark_id]

        save_partitioned_circuit_text(
            item=benchmark_obj,
            spec=spec,
            result=item,
            out_dir=output_dir / "partitioned_circuits",
        )
        bid = item["benchmark_id"]
        for prop_name, group_scores in [
            ("magic", item["group_magic"]),
            ("entanglement", item["group_entanglement"]),
        ]:
            row = {
                "benchmark_id": bid,
                "family_role": item["family_role"],
                "property": prop_name,
                "top_group": item[f"top_group_{prop_name}"],
            }
            row.update(group_scores)
            group_rows.append(row)

        for gate_idx, val in item["phi_magic"].items():
            gate_rows.append(
                {
                    "benchmark_id": bid,
                    "property": "magic",
                    "gate_idx": gate_idx,
                    "owen_value": val,
                }
            )
        for gate_idx, val in item["phi_entanglement"].items():
            gate_rows.append(
                {
                    "benchmark_id": bid,
                    "property": "entanglement",
                    "gate_idx": gate_idx,
                    "owen_value": val,
                }
            )

        plot_group_bars(
            benchmark_id=bid,
            magic_group_scores=item["group_magic"],
            ent_group_scores=item["group_entanglement"],
            out_path=output_dir / f"{bid}_group_bars.png",
        )

    pd.DataFrame(group_rows).to_csv(output_dir / "exact_owen_group_scores.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output_dir / "exact_owen_gate_scores.csv", index=False)

    print(f"Saved JSON results to: {json_path}")
    print(f"Saved group CSV to: {output_dir / 'exact_owen_group_scores.csv'}")
    print(f"Saved gate CSV to: {output_dir / 'exact_owen_gate_scores.csv'}")
    print(f"Saved plots to: {output_dir}")


def print_selection_summary(summary_df: pd.DataFrame) -> None:
    exact_df = summary_df.loc[summary_df["selection_mode"].astype(str).str.lower() == "exact"].copy()
    exact_df = exact_df[[
        "benchmark_id",
        "family_role",
        "gate_count",
        "candidate_uid",
        "sre",
        "entanglement_q",
        "sre_norm",
        "ent_norm",
    ]]
    print("\nExact subset from benchmark_35_summary.csv:")
    print(exact_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--benchmark-pkl", type=Path, default=BENCHMARK_PKL)
    p.add_argument("--gate-spec-csv", type=Path, default=GATE_SPEC_CSV)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "--allow-order-fallback",
        action="store_true",
        help=(
            "Only use this if benchmark_35.pkl cannot be matched by benchmark_id or candidate_uid, "
            "but is known to be in the same order as benchmark_35_summary.csv."
        ),
    )
    p.add_argument(
        "--overwrite-spec",
        action="store_true",
        help="Overwrite an existing gate-spec scaffold CSV.",
    )
    p.add_argument(
        "--scaffold-only",
        action="store_true",
        help="Only write the gate-spec scaffold CSV and stop.",
    )
    return p.parse_args()


def is_entangler(instr_name: str) -> bool:
    return instr_name.lower() in ENTANGLER_GATES

def is_local_clifford_1q(instr_name: str) -> bool:
    return instr_name.lower() in LOCAL_CLIFFORD_1Q


def gate_qubit_set(qc: QuantumCircuit, qargs: Sequence[Any]) -> set[int]:
    return set(qubit_indices(qc, qargs))


def adjacent_shared_entangler(qc: QuantumCircuit, idx: int) -> bool:
    instr, qargs, _ = qc.data[idx]
    this_qubits = gate_qubit_set(qc, qargs)

    neighbors = []
    if idx > 0:
        neighbors.append(qc.data[idx - 1])
    if idx + 1 < len(qc.data):
        neighbors.append(qc.data[idx + 1])

    for n_instr, n_qargs, _ in neighbors:
        if is_entangler(n_instr.name):
            n_qubits = gate_qubit_set(qc, n_qargs)
            if this_qubits & n_qubits:
                return True
    return False


def suggest_group_for_gate(
    qc: QuantumCircuit,
    idx: int,
) -> str:
    instr, qargs, _ = qc.data[idx]
    name = instr.name.lower()

    if is_entangler(name):
        return "E"

    if len(qargs) != 1:
        return ""

    shared_adj_ent = adjacent_shared_entangler(qc, idx)

    # Local Clifford gates near entanglers are entanglement-preparation candidates
    if is_local_clifford_1q(name):
        if shared_adj_ent:
            return "E"
        return ""

    # Non-Clifford local gates
    if is_non_clifford_single_qubit_gate(name, instr.params):
        if shared_adj_ent:
            return "X"
        return "M"

    return ""


def save_partitioned_circuit_text(
    item: BenchmarkCircuit,
    spec: Dict[str, Any],
    result: Dict[str, Any],
    out_dir: Path,
) -> None:
    """
    Save a text inspection file for one circuit showing:
      - gate index
      - gate name
      - qubits
      - partition group
      - exact Owen value for magic
      - exact Owen value for entanglement
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    qc = item.qc
    phi_magic = result["phi_magic"]
    phi_ent = result["phi_entanglement"]

    group_by_gate = {}
    for label, group in zip(spec["partition_labels"], spec["partition"]):
        for g in group:
            group_by_gate[g] = label
    for g in spec["locked"]:
        group_by_gate[g] = "PASSIVE"

    lines = []
    lines.append(f"benchmark_id: {item.benchmark_id}")
    lines.append(f"family_role: {item.summary_row['family_role']}")
    lines.append("")
    lines.append("Circuit:")
    lines.append(str(qc.draw(output='text')))
    lines.append("")
    lines.append("Gate table:")
    lines.append(
        "idx | gate | qubits | group | phi_magic | phi_entanglement"
    )
    lines.append("-" * 80)

    for idx, (instr, qargs, _) in enumerate(qc.data):
        qbs = qubit_indices(qc, qargs)
        group = group_by_gate.get(idx, "UNASSIGNED")
        pm = phi_magic.get(idx, float("nan"))
        pe = phi_ent.get(idx, float("nan"))
        lines.append(
            f"{idx:>3} | {instr.name:<4} | {str(qbs):<8} | {group:<7} | "
            f"{pm:> .12f} | {pe:> .12f}"
        )

    out_path = out_dir / f"{item.benchmark_id}_partitioned_circuit.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    summary_df = load_summary(args.summary_csv)
    exact_ids = get_exact_ids(summary_df)
    print_selection_summary(summary_df)

    benchmark_map = load_benchmark_pickle(
        args.benchmark_pkl,
        summary_df,
        allow_order_fallback=args.allow_order_fallback,
    )
    benchmark_circuits = build_benchmark_circuits(summary_df, benchmark_map, exact_ids)

    write_gate_spec_template(
        benchmark_circuits=benchmark_circuits,
        spec_csv=args.gate_spec_csv,
        overwrite=args.overwrite_spec,
    )

    if args.scaffold_only:
        print("\nScaffold-only mode complete.")
        return

    gate_spec = load_gate_spec(args.gate_spec_csv, benchmark_circuits)

    results: List[Dict[str, Any]] = []
    for benchmark_id in exact_ids:
        print(f"Running exact Owen for {benchmark_id} ...")
        result = run_exact_owen_for_circuit(
            item=benchmark_circuits[benchmark_id],
            spec=gate_spec[benchmark_id],
        )
        results.append(result)

    save_results(results,args.output_dir,benchmark_circuits,gate_spec,)
    print("\nDone.")


if __name__ == "__main__":
    main()
