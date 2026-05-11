#!/usr/bin/env python3
"""
script_noisy_robustness_axis4.py

Reduced noisy-robustness experiment on 4 representative axis circuits:
    M1, M5, E1, E5

Design
------
- Uses the exact clean baseline already saved in:
      data/plots/exact_owen_15/exact_owen_gate_scores.csv
- Reuses the manual gate partition from:
      data/benchmark_15_exact_gate_spec.csv
- Runs only the noisy sampled Owen estimator
- Keeps n fixed at the chosen value (default: 300)
- Uses K in {20, 50, 100} to average noisy coalition-value evaluations
- No outer repeats: one full Owen run per (circuit, property, K)

Outputs
-------
data/plots/noisy_robustness_axis4/
    - per_circuit_property_metrics.csv
    - summary_metrics.csv
    - noisy_robustness_results.json
    - plot_noisy_robustness_axis4.png

Main figure
-----------
One figure with 2 panels:
    - magic property
    - entanglement property

Each panel shows:
    - circuit-level gate MAE curves for each of the 4 circuits
    - a thick mean curve across the 4 circuits

This lets you see both:
    - the overall robustness trend with K
    - which circuits are more fragile than others
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script_exact_owen_benchmark15 import (  # type: ignore
    SUMMARY_CSV,
    BENCHMARK_PKL,
    GATE_SPEC_CSV,
    BenchmarkCircuit,
    QuantumOwenValues,
    aggregate_group_scores,
    build_benchmark_circuits,
    build_circuit,
    load_benchmark_pickle,
    load_gate_spec,
    load_summary,
)

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


EXACT_GATE_CSV = ROOT / "data" / "plots" / "exact_owen_15" / "exact_owen_gate_scores.csv"
OUTPUT_DIR = ROOT / "data" / "plots" / "noisy_robustness_axis4"

FOUR_IDS = ["M1", "M5", "E1", "E5"]
GROUP_LABELS = ["E", "M", "X"]
PAULIS = ["I", "X", "Y", "Z"]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    p.add_argument("--benchmark-pkl", type=Path, default=BENCHMARK_PKL)
    p.add_argument("--gate-spec-csv", type=Path, default=GATE_SPEC_CSV)
    p.add_argument("--exact-gate-csv", type=Path, default=EXACT_GATE_CSV)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    p.add_argument("--n", type=int, default=300)
    p.add_argument("--Ks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--seed", type=int, default=123)

    # generic hardware-like noise model
    p.add_argument("--one-q-error", type=float, default=0.001)
    p.add_argument("--two-q-error", type=float, default=0.01)
    p.add_argument("--readout-error", type=float, default=0.02)

    p.add_argument("--allow-order-fallback", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------
# Exact baseline loading
# ---------------------------------------------------------------------

def load_exact_gate_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exact gate baseline CSV: {path}\n"
            "Run script_exact_owen_benchmark15.py first."
        )
    df = pd.read_csv(path)
    required = {"benchmark_id", "property", "gate_idx", "owen_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Exact gate CSV missing required columns: {sorted(missing)}")

    df["benchmark_id"] = df["benchmark_id"].astype(str)
    df["property"] = df["property"].astype(str)
    df["gate_idx"] = df["gate_idx"].astype(int)
    df["owen_value"] = df["owen_value"].astype(float)
    return df


def build_exact_baseline(
    exact_gate_df: pd.DataFrame,
    gate_spec: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    baseline: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for bid in FOUR_IDS:
        baseline[bid] = {}
        spec = gate_spec[bid]

        gate_to_group: Dict[int, str] = {}
        for label, members in zip(spec["partition_labels"], spec["partition"]):
            for g in members:
                gate_to_group[int(g)] = str(label)

        for prop in ["magic", "entanglement"]:
            sub = exact_gate_df[
                (exact_gate_df["benchmark_id"] == bid) &
                (exact_gate_df["property"] == prop)
            ].copy()

            phi = {int(r["gate_idx"]): float(r["owen_value"]) for _, r in sub.iterrows()}
            gate_order = sorted(phi.keys())

            group_scores = {lab: 0.0 for lab in GROUP_LABELS}
            for g, v in phi.items():
                lab = gate_to_group[int(g)]
                group_scores[lab] += float(v)

            baseline[bid][prop] = {
                "phi": phi,
                "group": group_scores,
                "gate_order": gate_order,
                "top_group": max(GROUP_LABELS, key=lambda lab: group_scores.get(lab, 0.0)),
            }

    return baseline


# ---------------------------------------------------------------------
# Sample-fraction logic
# ---------------------------------------------------------------------

def owen_term_count_per_group(partition: Sequence[Sequence[int]]) -> Dict[int, int]:
    m = len(partition)
    counts: Dict[int, int] = {}
    for group in partition:
        s = len(group)
        if s == 0:
            continue
        terms = (2 ** (m - 1)) * (2 ** (s - 1))
        for player in group:
            counts[int(player)] = int(terms)
    return counts


def sample_frac_from_n(partition: Sequence[Sequence[int]], n: int) -> Tuple[float, int]:
    counts = owen_term_count_per_group(partition)
    if not counts:
        return 1.0, 0
    t_max = max(counts.values())
    alpha = min(1.0, float(n) / float(t_max))
    return alpha, t_max


# ---------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------

def build_generic_noise_model(
    one_q_error: float,
    two_q_error: float,
    readout_error: float,
) -> NoiseModel:
    noise_model = NoiseModel()

    err_1 = depolarizing_error(one_q_error, 1)
    err_2 = depolarizing_error(two_q_error, 2)

    one_q_gates = [
        "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "sx", "sxdg", "rx", "ry", "rz", "p", "u", "u1", "u2", "u3"
    ]
    two_q_gates = ["cx", "cz"]

    for gate in one_q_gates:
        try:
            noise_model.add_all_qubit_quantum_error(err_1, gate)
        except Exception:
            pass

    for gate in two_q_gates:
        try:
            noise_model.add_all_qubit_quantum_error(err_2, gate)
        except Exception:
            pass

    ro = ReadoutError([
        [1.0 - readout_error, readout_error],
        [readout_error, 1.0 - readout_error],
    ])
    noise_model.add_all_qubit_readout_error(ro)
    return noise_model


# ---------------------------------------------------------------------
# Shot-based noisy value estimators
# ---------------------------------------------------------------------

def add_basis_measurements(qc: QuantumCircuit, basis: Sequence[str]) -> QuantumCircuit:
    n = qc.num_qubits
    out = QuantumCircuit(n, n)
    out.compose(qc, inplace=True)

    for q, b in enumerate(basis):
        if b == "X":
            out.h(q)
        elif b == "Y":
            out.sdg(q)
            out.h(q)
        elif b == "Z":
            pass
        else:
            raise ValueError(f"Unknown basis: {b}")

    out.measure(range(n), range(n))
    return out


def run_counts(
    sim: AerSimulator,
    qc: QuantumCircuit,
    shots: int,
    seed: int,
) -> Dict[str, int]:
    job = sim.run(qc, shots=shots, seed_simulator=seed)
    result = job.result()
    return result.get_counts()


def bit_parity_from_string_little_endian(bitstring_big_endian: str, qubits: Sequence[int]) -> int:
    bits = bitstring_big_endian[::-1]
    return sum(int(bits[q]) for q in qubits) % 2


def expectation_from_counts_for_pauli(
    counts: Mapping[str, int],
    pauli_string: Sequence[str],
) -> float:
    non_i = [q for q, p in enumerate(pauli_string) if p != "I"]
    shots = sum(counts.values())
    if shots == 0:
        return 0.0

    total = 0.0
    for bitstring, c in counts.items():
        parity = bit_parity_from_string_little_endian(bitstring, non_i)
        eigen = 1.0 if parity == 0 else -1.0
        total += eigen * c

    return float(total / shots)


def all_full_bases(n: int) -> List[Tuple[str, ...]]:
    return list(product(["X", "Y", "Z"], repeat=n))


def full_basis_for_pauli(pauli: Sequence[str]) -> Tuple[str, ...]:
    return tuple("Z" if p == "I" else p for p in pauli)


def estimate_all_pauli_expectations_from_shots(
    qc: QuantumCircuit,
    sim: AerSimulator,
    shots: int,
    seed: int,
) -> Dict[Tuple[str, ...], float]:
    n = qc.num_qubits

    basis_to_counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
    for idx, basis in enumerate(all_full_bases(n)):
        meas_qc = add_basis_measurements(qc, basis)
        basis_to_counts[basis] = run_counts(sim, meas_qc, shots=shots, seed=seed + idx)

    exps: Dict[Tuple[str, ...], float] = {}
    for pauli in product(PAULIS, repeat=n):
        basis = full_basis_for_pauli(pauli)
        counts = basis_to_counts[basis]
        exps[tuple(pauli)] = expectation_from_counts_for_pauli(counts, pauli)

    return exps


def noisy_sre_from_pauli_expectations(
    exps: Mapping[Tuple[str, ...], float],
    n: int,
    alpha: int = 2,
) -> float:
    d = 2 ** n
    A = 0.0
    for p in product(PAULIS, repeat=n):
        exp_val = float(exps[tuple(p)])
        xi_p = (1.0 / d) * (exp_val ** 2)
        A += xi_p ** alpha
    entropy = (1.0 / (1.0 - alpha)) * np.log(max(A, 1e-300)) - np.log(d)
    return float(entropy)


def noisy_meyer_wallach_from_pauli_expectations(
    exps: Mapping[Tuple[str, ...], float],
    n: int,
) -> float:
    purities = []
    for q in range(n):
        px = ["I"] * n
        py = ["I"] * n
        pz = ["I"] * n
        px[q] = "X"
        py[q] = "Y"
        pz[q] = "Z"

        rx = float(exps[tuple(px)])
        ry = float(exps[tuple(py)])
        rz = float(exps[tuple(pz)])

        purity = 0.5 * (1.0 + rx * rx + ry * ry + rz * rz)
        purities.append(purity)

    q_value = 2.0 * (1.0 - float(np.mean(purities)))
    return float(np.clip(q_value, 0.0, 1.0))


def make_noisy_value_fun(
    kind: str,
    sim: AerSimulator,
    shots: int,
    K: int,
    base_seed: int,
):
    cache: Dict[Tuple[int, ...], float] = {}

    def _value_fun(
        qc_data,
        num_qubits: int,
        S: Sequence[int],
        quantum_instance=None,
        **kwargs,
    ) -> float:
        key = tuple(sorted(int(x) for x in S))
        if key in cache:
            return cache[key]

        qc_reduced, _ = build_circuit(
            qc_data=qc_data,
            num_qubits=num_qubits,
            S=list(S),
            cl_bits=False,
        )

        vals = []
        subset_seed_base = base_seed + 1009 * sum(key) + 7919 * len(key)

        for rep in range(K):
            rep_seed = subset_seed_base + rep * 104729
            exps = estimate_all_pauli_expectations_from_shots(
                qc=qc_reduced,
                sim=sim,
                shots=shots,
                seed=rep_seed,
            )
            if kind == "magic":
                vals.append(noisy_sre_from_pauli_expectations(exps, qc_reduced.num_qubits))
            elif kind == "entanglement":
                vals.append(noisy_meyer_wallach_from_pauli_expectations(exps, qc_reduced.num_qubits))
            else:
                raise ValueError(f"Unknown kind: {kind}")

        cache[key] = float(np.mean(vals))
        return cache[key]

    return _value_fun


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def dense_gate_vector(phi: Mapping[int, float], gate_order: Sequence[int]) -> np.ndarray:
    return np.array([float(phi.get(int(g), 0.0)) for g in gate_order], dtype=float)


def dense_group_vector(group_scores: Mapping[str, float]) -> np.ndarray:
    return np.array([float(group_scores.get(g, 0.0)) for g in GROUP_LABELS], dtype=float)


def top_group_label(group_scores: Mapping[str, float]) -> str:
    return max(GROUP_LABELS, key=lambda lab: float(group_scores.get(lab, 0.0)))


def compare_property(
    benchmark_id: str,
    family_role: str,
    property_name: str,
    K: int,
    runtime_s: float,
    exact_phi: Mapping[int, float],
    est_phi: Mapping[int, float],
    exact_group: Mapping[str, float],
    est_group: Mapping[str, float],
    gate_order: Sequence[int],
) -> Dict[str, Any]:
    exact_gate = dense_gate_vector(exact_phi, gate_order)
    est_gate = dense_gate_vector(est_phi, gate_order)

    exact_group_vec = dense_group_vector(exact_group)
    est_group_vec = dense_group_vector(est_group)

    gate_abs = np.abs(est_gate - exact_gate)
    group_abs = np.abs(est_group_vec - exact_group_vec)

    rho = spearmanr(exact_gate, est_gate).statistic if len(gate_order) >= 2 else np.nan
    if pd.isna(rho):
        rho = 1.0

    exact_top = top_group_label(exact_group)
    est_top = top_group_label(est_group)

    return {
        "benchmark_id": benchmark_id,
        "family_role": family_role,
        "property": property_name,
        "K": int(K),
        "runtime_s": float(runtime_s),
        "gate_mae": float(np.mean(gate_abs)),
        "gate_max_ae": float(np.max(gate_abs)),
        "group_mae": float(np.mean(group_abs)),
        "group_max_ae": float(np.max(group_abs)),
        "top_group_exact": exact_top,
        "top_group_est": est_top,
        "top_group_match": bool(exact_top == est_top),
        "gate_spearman_rho": float(rho),
        "sum_phi_exact": float(np.sum(exact_gate)),
        "sum_phi_est": float(np.sum(est_gate)),
        "efficiency_abs_err": float(abs(np.sum(est_gate) - np.sum(exact_gate))),
    }


# ---------------------------------------------------------------------
# Noisy Owen run
# ---------------------------------------------------------------------

def run_noisy_sampled_for_circuit(
    item: BenchmarkCircuit,
    spec: Dict[str, Any],
    n: int,
    seed: int,
    K: int,
    shots: int,
    sim: AerSimulator,
) -> Dict[str, Any]:
    partition = spec["partition"]
    locked = spec["locked"]
    labels = spec["partition_labels"]
    alpha, _ = sample_frac_from_n(partition, n)

    t0 = time.perf_counter()
    qov_magic = QuantumOwenValues(
        qc=item.qc,
        partition=partition,
        value_fun=make_noisy_value_fun(
            kind="magic",
            sim=sim,
            shots=shots,
            K=K,
            base_seed=seed,
        ),
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=locked,
        owen_sample_frac=alpha,
        owen_sample_reps=1,
        evaluate_value_only_once=False,
        owen_sample_seed=seed,
        name=f"{item.benchmark_id}_noisy_magic_n{n}_K{K}",
        silent=True,
    )
    phi_magic = qov_magic.run()
    rt_magic = time.perf_counter() - t0

    t1 = time.perf_counter()
    qov_ent = QuantumOwenValues(
        qc=item.qc,
        partition=partition,
        value_fun=make_noisy_value_fun(
            kind="entanglement",
            sim=sim,
            shots=shots,
            K=K,
            base_seed=seed + 10_000,
        ),
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=locked,
        owen_sample_frac=alpha,
        owen_sample_reps=1,
        evaluate_value_only_once=False,
        owen_sample_seed=seed + 10_000,
        name=f"{item.benchmark_id}_noisy_ent_n{n}_K{K}",
        silent=True,
    )
    phi_ent = qov_ent.run()
    rt_ent = time.perf_counter() - t1

    return {
        "phi_magic": {int(k): float(v) for k, v in phi_magic.items()},
        "phi_entanglement": {int(k): float(v) for k, v in phi_ent.items()},
        "group_magic": aggregate_group_scores(phi_magic, partition, labels),
        "group_entanglement": aggregate_group_scores(phi_ent, partition, labels),
        "runtime_magic_s": float(rt_magic),
        "runtime_entanglement_s": float(rt_ent),
    }


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_summary(per_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    props = ["magic", "entanglement"]
    circuits = FOUR_IDS

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, prop in zip(axes, props):
        sub = per_df[per_df["property"] == prop].copy()

        # thin per-circuit lines
        for bid in circuits:
            s = sub[sub["benchmark_id"] == bid].sort_values("K")
            ax.plot(
                s["K"].to_numpy(),
                s["gate_mae"].to_numpy(),
                marker="o",
                linewidth=1.5,
                alpha=0.55,
                label=bid,
            )

        # thick mean line
        smean = summary_df[summary_df["property"] == prop].sort_values("K")
        ax.plot(
            smean["K"].to_numpy(),
            smean["gate_mae_mean"].to_numpy(),
            marker="o",
            linewidth=3.2,
            color="black",
            label="mean gate MAE",
        )
        ax.plot(
            smean["K"].to_numpy(),
            smean["group_mae_mean"].to_numpy(),
            marker="s",
            linewidth=3.2,
            color="tab:red",
            label="mean group MAE",
        )

        ax.set_title(f"{prop.capitalize()} value")
        ax.set_xlabel("K noisy coalition-value averages")
        ax.set_ylabel("error vs exact clean Owen")
        ax.grid(alpha=0.3)

        ymax = max(
            np.max(sub["gate_mae"].to_numpy()) if len(sub) else 0.0,
            np.max(smean["group_mae_mean"].to_numpy()) if len(smean) else 0.0,
        )

        for _, row in smean.iterrows():
            ax.text(
                row["K"],
                ymax * 1.03 if ymax > 0 else 0.01,
                f"TG={row['top_group_agree_rate']:.2f}\nρ={row['spearman_mean']:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_ylim(top=(ymax * 1.28 if ymax > 0 else 0.05))

    handles, labels = axes[0].get_legend_handles_labels()
    # keep only unique labels
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)

    fig.legend(
        uniq_handles,
        uniq_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle(
        "Noisy robustness on 4 representative axis circuits (n=300)\n"
        "Thin lines: per-circuit gate MAE, thick lines: mean gate/group MAE",
        fontsize=15,
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary(args.summary_csv)
    benchmark_map = load_benchmark_pickle(
        args.benchmark_pkl,
        summary_df,
        allow_order_fallback=args.allow_order_fallback,
    )
    benchmark_circuits = build_benchmark_circuits(summary_df, benchmark_map, FOUR_IDS)

    gate_spec = load_gate_spec(args.gate_spec_csv, benchmark_circuits)
    exact_gate_df = load_exact_gate_baseline(args.exact_gate_csv)
    exact_baseline = build_exact_baseline(exact_gate_df, gate_spec)

    noise_model = build_generic_noise_model(
        one_q_error=args.one_q_error,
        two_q_error=args.two_q_error,
        readout_error=args.readout_error,
    )
    noisy_sim = AerSimulator(noise_model=noise_model)

    per_rows: List[Dict[str, Any]] = []
    raw_payload: List[Dict[str, Any]] = []

    for bid in FOUR_IDS:
        item = benchmark_circuits[bid]
        spec = gate_spec[bid]

        for K in args.Ks:
            print(f"Running noisy robustness for {bid}, K={K} ...")
            noisy = run_noisy_sampled_for_circuit(
                item=item,
                spec=spec,
                n=args.n,
                seed=args.seed,
                K=K,
                shots=args.shots,
                sim=noisy_sim,
            )
            raw_payload.append({
                "benchmark_id": bid,
                "K": int(K),
                "result": noisy,
            })

            for prop in ["magic", "entanglement"]:
                exact_phi = exact_baseline[bid][prop]["phi"]
                exact_group = exact_baseline[bid][prop]["group"]
                gate_order = exact_baseline[bid][prop]["gate_order"]

                if prop == "magic":
                    est_phi = noisy["phi_magic"]
                    est_group = noisy["group_magic"]
                    runtime_s = noisy["runtime_magic_s"]
                else:
                    est_phi = noisy["phi_entanglement"]
                    est_group = noisy["group_entanglement"]
                    runtime_s = noisy["runtime_entanglement_s"]

                per_rows.append(compare_property(
                    benchmark_id=bid,
                    family_role=item.summary_row["family_role"],
                    property_name=prop,
                    K=K,
                    runtime_s=runtime_s,
                    exact_phi=exact_phi,
                    est_phi=est_phi,
                    exact_group=exact_group,
                    est_group=est_group,
                    gate_order=gate_order,
                ))

    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(args.output_dir / "per_circuit_property_metrics.csv", index=False)

    summary = (
        per_df
        .groupby(["property", "K"], as_index=False)
        .agg(
            gate_mae_mean=("gate_mae", "mean"),
            group_mae_mean=("group_mae", "mean"),
            top_group_agree_rate=("top_group_match", "mean"),
            spearman_mean=("gate_spearman_rho", "mean"),
            runtime_mean=("runtime_s", "mean"),
        )
    )
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)

    (args.output_dir / "noisy_robustness_results.json").write_text(
        json.dumps(raw_payload, indent=2),
        encoding="utf-8",
    )

    plot_summary(per_df, summary, args.output_dir / "plot_noisy_robustness_axis4.png")

    print(f"Saved per-circuit metrics to: {args.output_dir / 'per_circuit_property_metrics.csv'}")
    print(f"Saved summary metrics to: {args.output_dir / 'summary_metrics.csv'}")
    print(f"Saved raw results to: {args.output_dir / 'noisy_robustness_results.json'}")
    print(f"Saved plot to: {args.output_dir / 'plot_noisy_robustness_axis4.png'}")


if __name__ == "__main__":
    main()