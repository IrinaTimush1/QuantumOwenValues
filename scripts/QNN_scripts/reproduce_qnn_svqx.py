#!/usr/bin/env python3
"""
Reproduce the QNN SVQX experiment from Section 4.2 / Figures 13--14 of
Heese et al., "Explaining Quantum Circuits with Shapley Values".


It uses the repository's existing qshaptools implementation rather than
reimplementing Shapley values from scratch.

Paper mapping:
- QNN circuit: Figure 12, 19 indexed gates, 2 qubits.
- Dataset: 20 points, same data for train/test, loaded from qnn-data.csv.
- Fixed paper parameters: theta ~= (3.860, -1.070, -1.583, 0.860).
- Active gates: A = {2, 4, 5, ..., 19} in paper 1-based indexing.
- Locked/passive gates: R = {1, 3} in paper 1-based indexing.
- Python/Qiskit instruction indices are 0-based, so locked gates are [0, 2].
- Value function: one-shot test accuracy without retraining.
- Simulator: shot/qasm simulator, shots=1.
- Figure 13: K=32, alpha=0.01, five independent runs.
- Figure 14: K in {1, 8, 16, 32}, alpha=0.01, five independent runs.

Typical usage from the OVQX repo root:
    python scripts/reproduce_qnn_svqx.py --mode fig13 --use-paper-theta --n-jobs 5
    python scripts/reproduce_qnn_svqx.py --mode fig14 --use-paper-theta --n-jobs 5
    python scripts/reproduce_qnn_svqx.py --mode both  --use-paper-theta --n-jobs 5

If you place this file in the repository root instead of scripts/, it will also work.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend so the script works in parallel/headless runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants from the paper
# ---------------------------------------------------------------------------

PAPER_THETA = np.array([3.860, -1.070, -1.583, 0.860], dtype=float)
PAPER_LOCKED_INSTRUCTIONS_0_BASED = [0, 2]  # paper R={1,3}
PAPER_ACTIVE_GATES_1_BASED = [2] + list(range(4, 20))
PAPER_ACTIVE_INSTRUCTIONS_0_BASED = [g - 1 for g in PAPER_ACTIVE_GATES_1_BASED]
PAPER_GATE_NAMES_1_BASED = {
    1: "H",  2: "P",  3: "H",  4: "P",  5: "CX", 6: "P",  7: "CX",
    8: "H",  9: "P", 10: "H", 11: "P", 12: "CX", 13: "P", 14: "CX",
    15: "RY", 16: "RY", 17: "CX", 18: "RY", 19: "RY",
}
FEATURE_PARAM_NAMES = [f"feat_p{i}" for i in range(6)]
THETA_PARAM_NAMES = [f"theta_{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# Robust project / dependency setup
# ---------------------------------------------------------------------------

def infer_repo_root() -> Path:
    """Infer OVQX root whether script lives in repo root or scripts/."""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for c in candidates:
        if (c / "qshaptools" / "src" / "qshaptools").exists():
            return c
    # Last fallback: current working directory. Import step will fail clearly.
    return Path.cwd()


def setup_import_paths(repo_root: Path) -> None:
    """Add OVQX qshaptools path to sys.path."""
    qshaptools_path = repo_root / "qshaptools" / "src" / "qshaptools"
    if not qshaptools_path.exists():
        raise FileNotFoundError(
            f"Cannot find qshaptools at {qshaptools_path}. Run this script from the OVQX repo "
            "or place it under OVQX/scripts/."
        )
    sys.path.insert(0, str(qshaptools_path))


def patch_qiskit_bit_index_compatibility() -> None:
    """
    qshaptools in this repo uses qubit.index / clbit.index, which existed in older
    Qiskit versions. Newer Qiskit stores this as a private _index. This small
    compatibility patch keeps the repository's original code usable without editing it.
    """
    try:
        from qiskit.circuit import Qubit, Clbit
    except Exception:
        return

    def _get_index(self):  # type: ignore[no-untyped-def]
        if hasattr(self, "_index"):
            return self._index
        raise AttributeError("Qiskit bit object has neither .index nor ._index")

    if not hasattr(Qubit, "index"):
        Qubit.index = property(_get_index)  # type: ignore[attr-defined]
    if not hasattr(Clbit, "index"):
        Clbit.index = property(_get_index)  # type: ignore[attr-defined]


def import_qiskit_and_qshaptools() -> Tuple[Any, Any, Any, Any, Any]:
    """Import Qiskit and existing qshaptools objects with clear errors."""
    patch_qiskit_bit_index_compatibility()
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
    except Exception as exc:
        raise ImportError("Could not import qiskit. Install qiskit and qiskit-aer.") from exc

    try:
        from qiskit_aer import Aer
    except Exception:
        try:
            from qiskit import Aer  # older Qiskit
        except Exception as exc:
            raise ImportError("Could not import Aer. Install qiskit-aer.") from exc

    try:
        from qshap import QuantumShapleyValues
        from qvalues import value_callable
    except Exception as exc:
        raise ImportError(
            "Could not import QuantumShapleyValues/value_callable from qshaptools. "
            "Check that qshaptools/src/qshaptools is on sys.path."
        ) from exc

    return QuantumCircuit, Parameter, Aer, QuantumShapleyValues, value_callable


# ---------------------------------------------------------------------------
# Simple qasm quantum instance wrapper
# ---------------------------------------------------------------------------

class SimpleQasmQuantumInstance:
    """
    Minimal object with .execute(circuits), matching what qshaptools value functions need.

    We intentionally avoid a fixed seed for every backend.run call. Instead, this object
    owns an RNG and draws a fresh seed per execute call, making repeated K evaluations
    genuinely noisy but reproducible from the run seed.
    """

    is_statevector = False

    def __init__(self, backend: Any, shots: int = 1, seed: Optional[int] = None):
        self.backend = backend
        self.shots = int(shots)
        self.rng = np.random.default_rng(seed)

    def execute(self, circuits: Any) -> Any:
        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]
        seed_simulator = int(self.rng.integers(0, 2**31 - 1))
        try:
            job = self.backend.run(circuits, shots=self.shots, seed_simulator=seed_simulator)
            return job.result()
        except TypeError:
            # Older backend.run implementations may not accept seed_simulator.
            job = self.backend.run(circuits, shots=self.shots)
            return job.result()


# ---------------------------------------------------------------------------
# Data and circuit
# ---------------------------------------------------------------------------

def load_or_create_dataset(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the paper QNN toy dataset from CSV."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "The paper experiment uses qnn-data.csv. Pass --data-path /path/to/qnn-data.csv."
        )

    # Your existing scripts use sep=';'. This also tolerates comma CSVs.
    try:
        df = pd.read_csv(data_path, sep=";")
        if not {"x0", "x1", "y"}.issubset(df.columns):
            df = pd.read_csv(data_path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset at {data_path}") from exc

    if not {"x0", "x1", "y"}.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns x0, x1, y. Found: {list(df.columns)}"
        )

    X = df[["x0", "x1"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    if len(y) != 20:
        logging.warning("Paper dataset has 20 points, but loaded %d rows from %s", len(y), data_path)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1.")
    return X, y


def build_qnn_circuit() -> Any:
    """Build the exact 19-gate QNN circuit from paper Figure 12."""
    QuantumCircuit, Parameter, _, _, _ = import_qiskit_and_qshaptools()

    qc = QuantumCircuit(2, 2)
    fp = [Parameter(name) for name in FEATURE_PARAM_NAMES]
    tp = [Parameter(name) for name in THETA_PARAM_NAMES]

    # First feature-map block, paper gates 1--7.
    qc.h(0)           # 1
    qc.p(fp[0], 0)    # 2: P(phi_1(x_1))
    qc.h(1)           # 3
    qc.p(fp[1], 1)    # 4: P(phi_2(x_2))
    qc.cx(0, 1)       # 5
    qc.p(fp[2], 1)    # 6: P(phi(x))
    qc.cx(0, 1)       # 7

    # Second feature-map block, paper gates 8--14.
    qc.h(0)           # 8
    qc.p(fp[3], 0)    # 9: P(phi_1(x_1))
    qc.h(1)           # 10
    qc.p(fp[4], 1)    # 11: P(phi_2(x_2))
    qc.cx(0, 1)       # 12
    qc.p(fp[5], 1)    # 13: P(phi(x))
    qc.cx(0, 1)       # 14

    # Trainable layer, paper gates 15--19.
    qc.ry(tp[0], 0)   # 15: RY(theta_1) in paper notation
    qc.ry(tp[1], 1)   # 16: RY(theta_2)
    qc.cx(0, 1)       # 17
    qc.ry(tp[2], 0)   # 18: RY(theta_3)
    qc.ry(tp[3], 1)   # 19: RY(theta_4)
    return qc


def instruction_qubit_indices(qc: Any, qargs: Sequence[Any]) -> List[int]:
    """Version-safe qubit index extraction for logging."""
    out: List[int] = []
    for q in qargs:
        if hasattr(q, "index"):
            out.append(int(q.index))
        else:
            out.append(int(qc.find_bit(q).index))
    return out


def validate_qnn_circuit(qc: Any) -> None:
    """Fail fast if the Figure 12 circuit / active-passive indexing is wrong."""
    if len(qc.data) != 19:
        raise ValueError(f"Expected 19 instructions/gates, got {len(qc.data)}")
    if qc.num_qubits != 2:
        raise ValueError(f"Expected 2 qubits, got {qc.num_qubits}")

    observed = [instr.operation.name if hasattr(instr, "operation") else instr[0].name for instr in qc.data]
    expected = [PAPER_GATE_NAMES_1_BASED[i].lower() for i in range(1, 20)]
    # Qiskit uses 'p', 'cx', 'ry', 'h'.
    expected = ["p" if x == "p" else "cx" if x == "cx" else "ry" if x == "ry" else "h" for x in expected]
    if observed != expected:
        raise ValueError(f"Gate sequence mismatch.\nObserved: {observed}\nExpected: {expected}")

    active = sorted(set(range(19)) - set(PAPER_LOCKED_INSTRUCTIONS_0_BASED))
    if active != PAPER_ACTIVE_INSTRUCTIONS_0_BASED:
        raise ValueError(f"Active gate mismatch: {active} != {PAPER_ACTIVE_INSTRUCTIONS_0_BASED}")


# ---------------------------------------------------------------------------
# Binding, readout, and value function
# ---------------------------------------------------------------------------

def feature_values_for_x(x0: float, x1: float) -> Dict[str, float]:
    """Paper feature functions: phi_i(x_i)=2x_i and phi(x)=2(pi-x_1)(pi-x_2)."""
    phi1 = 2.0 * float(x0)
    phi2 = 2.0 * float(x1)
    phi_cross = 2.0 * (math.pi - float(x0)) * (math.pi - float(x1))
    return {
        "feat_p0": phi1,
        "feat_p1": phi2,
        "feat_p2": phi_cross,
        "feat_p3": phi1,
        "feat_p4": phi2,
        "feat_p5": phi_cross,
    }


def bind_qnn_parameters(qc: Any, x0: float, x1: float, theta: Sequence[float]) -> Any:
    values = feature_values_for_x(x0, x1)
    for i, t in enumerate(theta):
        values[f"theta_{i}"] = float(t)
    bindings = {p: values[p.name] for p in qc.parameters}
    try:
        return qc.assign_parameters(bindings, inplace=False)
    except TypeError:
        return qc.assign_parameters(bindings)


def predict_first_qubit_from_counts(counts: Mapping[str, int]) -> int:
    """
    Prediction from first qubit. With measure([0,1],[0,1]), Qiskit bitstrings are
    c1c0, so qubit 0 is the rightmost character.
    """
    bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
    return int(bitstring[-1])


def qnn_one_shot_accuracy_eval_fun(
    quantum_instance: Any,
    qc: Any,
    param_def_dict: Mapping[Any, Any],
    X: np.ndarray,
    y: np.ndarray,
    theta: Sequence[float],
) -> float:
    """
    qshaptools-compatible value evaluation:
        eval_fun(quantum_instance, qc, param_def_dict, **kwargs) -> float

    One value-function call evaluates v(S) once:
    - for each data point, bind x and fixed theta,
    - run exactly one shot,
    - read out the first qubit,
    - compute accuracy over the dataset.
    """
    circuits = []
    for x0, x1 in X:
        bound = bind_qnn_parameters(qc, float(x0), float(x1), theta)
        measured = bound.copy()
        measured.measure(range(measured.num_qubits), range(measured.num_qubits))
        circuits.append(measured)

    result = quantum_instance.execute(circuits)
    correct = 0
    for i, circuit in enumerate(circuits):
        counts = result.get_counts(circuit)
        pred = predict_first_qubit_from_counts(counts)
        if pred == int(y[i]):
            correct += 1
    return correct / len(y)


# ---------------------------------------------------------------------------
# Optional training
# ---------------------------------------------------------------------------

def estimate_accuracy_many_trials(
    qc: Any,
    X: np.ndarray,
    y: np.ndarray,
    theta: Sequence[float],
    seed: int,
    n_trials: int = 50,
) -> Tuple[float, float, List[float]]:
    """Repeated full-circuit one-shot accuracy sanity check."""
    _, _, Aer, _, _ = import_qiskit_and_qshaptools()
    backend = Aer.get_backend("qasm_simulator")
    qi = SimpleQasmQuantumInstance(backend, shots=1, seed=seed)
    accs: List[float] = []
    for _ in range(n_trials):
        accs.append(qnn_one_shot_accuracy_eval_fun(qi, qc, {}, X, y, theta))
    return float(np.mean(accs)), float(np.std(accs, ddof=0)), accs


def train_qnn_if_requested(
    qc: Any,
    X: np.ndarray,
    y: np.ndarray,
    base_seed: int,
    maxiter: int = 250,
) -> np.ndarray:
    """
    Optional COBYLA training. The paper reports trained theta and does not provide all
    optimizer details, so --use-paper-theta is the recommended exact-reproduction mode.

    This training uses a finite-shot stochastic loss; results may not exactly match the paper.
    """
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        raise ImportError("--train requires scipy.") from exc

    logging.warning(
        "Training mode is provided for convenience, but exact paper reproduction should use "
        "--use-paper-theta because optimizer details/randomness are not fully specified."
    )

    rng = np.random.default_rng(base_seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=4)

    def objective(theta: np.ndarray) -> float:
        # Average several one-shot accuracies to make COBYLA less noisy.
        mean_acc, _, _ = estimate_accuracy_many_trials(
            qc, X, y, theta, seed=int(rng.integers(0, 2**31 - 1)), n_trials=16
        )
        return 1.0 - mean_acc

    res = minimize(objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 1.0})
    if not res.success:
        logging.warning("COBYLA did not report success: %s", res.message)
    return np.asarray(res.x, dtype=float)


# ---------------------------------------------------------------------------
# SVQX experiment runner
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    K: int
    run_id: int
    seed: int
    alpha: float
    data_path: str
    theta: List[float]
    output_dir: str
    silent: bool = True
    sample_in_memory: bool = True
    shap_batch_size: Optional[int] = None


def make_jsonable(obj: Any) -> Any:
    """Convert qshap metadata to JSON-serializable structures."""
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def run_single_svqx_experiment(config: RunConfig) -> Dict[str, Any]:
    """One independent SVQX run for a fixed K and seed."""
    start = time.time()
    repo_root = infer_repo_root()
    setup_import_paths(repo_root)
    QuantumCircuit, Parameter, Aer, QuantumShapleyValues, value_callable = import_qiskit_and_qshaptools()

    X, y = load_or_create_dataset(Path(config.data_path).expanduser())
    qc = build_qnn_circuit()
    validate_qnn_circuit(qc)

    backend = Aer.get_backend("qasm_simulator")
    qi = SimpleQasmQuantumInstance(backend, shots=1, seed=config.seed)

    qsv = QuantumShapleyValues(
        qc=qc,
        value_fun=value_callable,
        value_kwargs_dict={
            "eval_fun": qnn_one_shot_accuracy_eval_fun,
            "X": X,
            "y": y,
            "theta": list(config.theta),
        },
        quantum_instance=qi,
        locked_instructions=PAPER_LOCKED_INSTRUCTIONS_0_BASED,
        shap_sample_frac=float(config.alpha),
        shap_sample_reps=int(config.K),
        evaluate_value_only_once=False,  # critical: K repeated noisy value-function evaluations matter
        sample_in_memory=bool(config.sample_in_memory),
        shap_sample_seed=int(config.seed),
        shap_batch_size=config.shap_batch_size,
        name=f"qnn_K{config.K}_run{config.run_id}",
        silent=bool(config.silent),
    )

    phi_dict_raw = qsv.run()
    phi_dict = {int(k): float(v) for k, v in phi_dict_raw.items()}

    # Store useful metadata if available. S_gen_ can be large, but for this experiment it is reasonable.
    summary = qsv.get_summary_dict() if hasattr(qsv, "get_summary_dict") else {}
    sampled_coalitions = getattr(qsv, "S_gen_", None)
    num_samples_dict = getattr(qsv, "num_samples_dict_", None)

    result = {
        "K": int(config.K),
        "run_id": int(config.run_id),
        "seed": int(config.seed),
        "alpha": float(config.alpha),
        "theta": list(map(float, config.theta)),
        "phi_dict_0_based": phi_dict,
        "phi_dict_1_based": {int(k) + 1: float(v) for k, v in phi_dict.items()},
        "summary": make_jsonable(summary),
        "sampled_coalitions_0_based": make_jsonable(sampled_coalitions),
        "num_samples_dict": make_jsonable(num_samples_dict),
        "elapsed_seconds": time.time() - start,
    }

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"raw_K{config.K}_run{config.run_id}_seed{config.seed}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["raw_result_path"] = str(out_file)
    return result


def run_parallel_experiments(
    K_values: Sequence[int],
    num_runs: int,
    n_jobs: int,
    alpha: float,
    base_seed: int,
    data_path: Path,
    theta: Sequence[float],
    output_dir: Path,
    silent: bool,
) -> List[Dict[str, Any]]:
    """Run the independent repetitions in parallel for each K."""
    configs: List[RunConfig] = []
    for K in K_values:
        for run_id in range(num_runs):
            # Different deterministic seed for every (K, run_id).
            seed = int(base_seed + 100_000 * int(K) + run_id)
            configs.append(
                RunConfig(
                    K=int(K),
                    run_id=int(run_id),
                    seed=seed,
                    alpha=float(alpha),
                    data_path=str(data_path),
                    theta=list(map(float, theta)),
                    output_dir=str(output_dir),
                    silent=silent,
                )
            )

    results: List[Dict[str, Any]] = []
    max_workers = max(1, int(n_jobs))
    logging.info("Launching %d independent jobs with n_jobs=%d", len(configs), max_workers)

    if max_workers == 1:
        for c in configs:
            logging.info("Running K=%s run=%s seed=%s", c.K, c.run_id, c.seed)
            results.append(run_single_svqx_experiment(c))
    else:
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {executor.submit(run_single_svqx_experiment, c): c for c in configs}
            for fut in cf.as_completed(future_to_config):
                c = future_to_config[fut]
                try:
                    res = fut.result()
                    logging.info(
                        "Finished K=%s run=%s seed=%s in %.1fs",
                        c.K, c.run_id, c.seed, res.get("elapsed_seconds", float("nan")),
                    )
                    results.append(res)
                except Exception as exc:
                    logging.error("FAILED K=%s run=%s seed=%s", c.K, c.run_id, c.seed)
                    logging.error("%s", traceback.format_exc())
                    raise exc

    results = sorted(results, key=lambda r: (int(r["K"]), int(r["run_id"])))
    all_file = output_dir / "all_raw_results.json"
    with all_file.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(results), f, indent=2)
    logging.info("Saved combined raw results to %s", all_file)
    return results


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------

def aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Aggregate mean/std over independent runs for each K and active gate."""
    by_k: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_k.setdefault(int(r["K"]), []).append(r)

    aggregated: Dict[int, Dict[str, Any]] = {}
    for K, rows in sorted(by_k.items()):
        active_gates_1 = PAPER_ACTIVE_GATES_1_BASED
        values_by_gate: Dict[int, List[float]] = {g: [] for g in active_gates_1}
        for r in rows:
            phi_1 = {int(k): float(v) for k, v in r["phi_dict_1_based"].items()}
            for g in active_gates_1:
                values_by_gate[g].append(float(phi_1[g]))
        aggregated[K] = {
            "K": K,
            "num_runs": len(rows),
            "gate_indices_1_based": active_gates_1,
            "gate_names": [PAPER_GATE_NAMES_1_BASED[g] for g in active_gates_1],
            "mean": {g: float(np.mean(values_by_gate[g])) for g in active_gates_1},
            "std": {g: float(np.std(values_by_gate[g], ddof=0)) for g in active_gates_1},
            "values": {g: list(map(float, values_by_gate[g])) for g in active_gates_1},
        }
    return aggregated


def write_aggregated_csv(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for K, info in aggregated.items():
        for g in info["gate_indices_1_based"]:
            rows.append({
                "K": K,
                "gate_1_based": g,
                "gate_0_based": g - 1,
                "gate_name": PAPER_GATE_NAMES_1_BASED[g],
                "mean_phi": info["mean"][g],
                "std_phi": info["std"][g],
                "num_runs": info["num_runs"],
                "run_values": json.dumps(info["values"][g]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "aggregated_svqx.csv", index=False)


def plot_fig13_style(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path, K: int = 32) -> Path:
    if K not in aggregated:
        raise ValueError(f"Cannot plot Figure 13 style: K={K} not found in results.")
    info = aggregated[K]
    gates = info["gate_indices_1_based"]
    means = [info["mean"][g] for g in gates]
    stds = [info["std"][g] for g in gates]
    names = [PAPER_GATE_NAMES_1_BASED[g] for g in gates]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.errorbar(gates, means, yerr=stds, fmt="o", capsize=3, markersize=5, label=f"K = {K}")
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("gate index g")
    ax.set_ylabel(r"$\Phi^{(g)}$")
    ax.set_title("QNN SVQX reproduction — Figure 13 style")
    ax.set_xticks(gates)
    ax.legend()

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(gates)
    ax_top.set_xticklabels(names, fontsize=8)

    fig.tight_layout()
    path = output_dir / f"fig13_style_K{K}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_fig14_style(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path, K_values: Sequence[int]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    gates = PAPER_ACTIVE_GATES_1_BASED
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, K in enumerate(K_values):
        if K not in aggregated:
            logging.warning("Skipping K=%s in Figure 14 plot because it is missing.", K)
            continue
        info = aggregated[K]
        means = [info["mean"][g] for g in gates]
        stds = [info["std"][g] for g in gates]
        ax.errorbar(
            gates,
            means,
            yerr=stds,
            fmt=markers[i % len(markers)],
            capsize=3,
            markersize=5,
            linestyle="None",
            label=f"K = {K}",
        )

    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("gate index g")
    ax.set_ylabel(r"$\Phi^{(g)}$")
    ax.set_title("QNN SVQX reproduction — Figure 14 style")
    ax.set_xticks(gates)
    ax.legend()

    names = [PAPER_GATE_NAMES_1_BASED[g] for g in gates]
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(gates)
    ax_top.set_xticklabels(names, fontsize=8)

    fig.tight_layout()
    path = output_dir / "fig14_style_K_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = infer_repo_root()
    default_data_path = Path("/Users/iratimush/xqml-thesis/data/qnn-data.csv")
    if not default_data_path.exists():
        default_data_path = repo_root / "data" / "qnn-data.csv"

    p = argparse.ArgumentParser(
        description="Reproduce the QNN SVQX experiment from Heese et al. Section 4.2 / Figures 13--14."
    )
    p.add_argument("--mode", choices=["fig13", "fig14", "both"], default="both")
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 8, 16, 32])
    p.add_argument("--num-runs", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--n-jobs", type=int, default=5)
    p.add_argument("--data-path", type=Path, default=default_data_path)
    p.add_argument("--output-dir", type=Path, default=repo_root / "results" / "qnn_svqx")
    p.add_argument("--use-paper-theta", action="store_true", help="Use theta reported in the paper. Recommended.")
    p.add_argument("--train", action="store_true", help="Train theta with COBYLA instead of using paper theta.")
    p.add_argument("--train-maxiter", type=int, default=250)
    p.add_argument("--no-silent", action="store_true", help="Show qshaptools tqdm bars inside worker processes.")
    p.add_argument("--sanity-trials", type=int, default=50)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    if args.train and args.use_paper_theta:
        p.error("Use either --train or --use-paper-theta, not both.")
    if not args.train and not args.use_paper_theta:
        logging.warning("Neither --train nor --use-paper-theta specified; defaulting to --use-paper-theta.")
        args.use_paper_theta = True

    if args.mode == "fig13":
        args.k_values = [32]
    elif args.mode == "fig14":
        # Keep user custom K values if explicitly passed; argparse cannot tell easily, so normalize default-like behavior.
        if args.k_values is None:
            args.k_values = [1, 8, 16, 32]
    elif args.mode == "both":
        # Need K=32 for fig13, and the comparison set for fig14.
        args.k_values = sorted(set(args.k_values + [32]))

    if args.num_runs < 1:
        p.error("--num-runs must be >= 1")
    if args.n_jobs < 1:
        p.error("--n-jobs must be >= 1")
    if args.alpha <= 0:
        p.error("--alpha must be > 0")
    return args


def save_config(args: argparse.Namespace, theta: Sequence[float], output_dir: Path) -> None:
    config = {
        "mode": args.mode,
        "k_values": args.k_values,
        "num_runs": args.num_runs,
        "alpha": args.alpha,
        "base_seed": args.base_seed,
        "n_jobs": args.n_jobs,
        "data_path": str(args.data_path),
        "output_dir": str(output_dir),
        "use_paper_theta": bool(args.use_paper_theta),
        "train": bool(args.train),
        "theta": list(map(float, theta)),
        "paper_locked_instructions_0_based": PAPER_LOCKED_INSTRUCTIONS_0_BASED,
        "paper_active_gates_1_based": PAPER_ACTIVE_GATES_1_BASED,
        "value_function": "one-shot test accuracy; first-qubit readout; no retraining",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def log_circuit(qc: Any, output_dir: Path) -> None:
    lines = []
    lines.append("Instruction order check: Python index is 0-based, paper gate index is 1-based.\n")
    for idx, item in enumerate(qc.data):
        if hasattr(item, "operation"):
            instr = item.operation
            qargs = item.qubits
            cargs = item.clbits
        else:
            instr, qargs, cargs = item
        qubits = instruction_qubit_indices(qc, qargs)
        params = [str(p) for p in getattr(instr, "params", [])]
        locked = idx in PAPER_LOCKED_INSTRUCTIONS_0_BASED
        lines.append(
            f"python_idx={idx:2d} | paper_gate={idx+1:2d} | "
            f"name={instr.name:3s} | qubits={qubits} | params={params} | "
            f"{'LOCKED/PASSIVE' if locked else 'ACTIVE'}"
        )
    text = "\n".join(lines)
    logging.info("\n%s", text)
    with (output_dir / "qnn_circuit_gate_index_log.txt").open("w", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    repo_root = infer_repo_root()
    setup_import_paths(repo_root)
    import_qiskit_and_qshaptools()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Repo root: %s", repo_root)
    logging.info("Output dir: %s", output_dir)
    logging.info("Data path: %s", args.data_path)

    X, y = load_or_create_dataset(args.data_path.expanduser())
    logging.info("Loaded dataset: X shape=%s, y shape=%s, label counts=%s", X.shape, y.shape, dict(pd.Series(y).value_counts()))

    qc = build_qnn_circuit()
    validate_qnn_circuit(qc)
    log_circuit(qc, output_dir)

    if args.train:
        theta = train_qnn_if_requested(qc, X, y, base_seed=args.base_seed, maxiter=args.train_maxiter)
    else:
        theta = PAPER_THETA.copy()
    logging.info("Using theta = %s", np.array2string(np.asarray(theta), precision=6))

    mean_acc, std_acc, sanity_accs = estimate_accuracy_many_trials(
        qc, X, y, theta, seed=args.base_seed + 999, n_trials=args.sanity_trials
    )
    logging.info(
        "Full-circuit one-shot accuracy sanity check: %.3f ± %.3f over %d trials. Paper reports about 0.80.",
        mean_acc, std_acc, args.sanity_trials,
    )
    with (output_dir / "full_circuit_sanity_accuracy.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": mean_acc, "std": std_acc, "trials": sanity_accs}, f, indent=2)

    save_config(args, theta, output_dir)

    results = run_parallel_experiments(
        K_values=args.k_values,
        num_runs=args.num_runs,
        n_jobs=args.n_jobs,
        alpha=args.alpha,
        base_seed=args.base_seed,
        data_path=args.data_path.expanduser(),
        theta=theta,
        output_dir=output_dir,
        silent=not args.no_silent,
    )

    aggregated = aggregate_results(results)
    with (output_dir / "aggregated_svqx.json").open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(aggregated), f, indent=2)
    write_aggregated_csv(aggregated, output_dir)

    if args.mode in {"fig13", "both"}:
        fig13 = plot_fig13_style(aggregated, output_dir, K=32)
        logging.info("Saved Figure 13-style plot: %s", fig13)
    if args.mode in {"fig14", "both"}:
        fig14 = plot_fig14_style(aggregated, output_dir, K_values=args.k_values)
        logging.info("Saved Figure 14-style plot: %s", fig14)

    logging.info("Done.")


if __name__ == "__main__":
    main()
