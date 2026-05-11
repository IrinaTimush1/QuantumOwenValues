#!/usr/bin/env python3
"""
Run the QNN one-shot-accuracy Owen-value experiment for several coalition
structures.

This script reproduces the QNN setup from Heese et al. Section 4.2, but replaces
ordinary gate-level Shapley/SVQX with gate-level Owen values under one of three
alternative coalition structures:

1. architecture
   - feature_map_block_1
   - feature_map_block_2
   - trainable_classifier

2. feature_interaction
   - local_feature_encoding
   - cross_feature_entangling_motifs
   - trainable_classifier

3. qubit_role
   - q0_local_readout_path
   - q1_local_hidden_path
   - two_qubit_couplers
   - final_q1_post_readout_rotation

The active/passive setup, trained theta, one-shot value function, K sweep, and
five independent runs follow the QNN/SVQX reproduction. By default, alpha=1.0 is
used because for these small Owen coalition structures this evaluates the full
structured Owen coalition space; K is still needed because one-shot accuracy is
stochastic.

Typical usage from the repo root:

    python scripts/reproduce_qnn_owen_multiple_partitions.py \
      --partition all \
      --mode both \
      --use-paper-theta \
      --data-path /Users/iratimush/xqml-thesis/data/qnn-data.csv \
      --output-dir results/qnn_owen_partitions \
      --num-runs 5 \
      --alpha 1.0 \
      --n-jobs 5
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

PAPER_THETA = np.array([3.860, -1.070, -1.583, 0.860], dtype=float)

# Paper gate numbers are 1-based. Qiskit instruction indices are 0-based.
PAPER_LOCKED_GATES_1_BASED = [1, 3]
PAPER_LOCKED_INSTRUCTIONS_0_BASED = [g - 1 for g in PAPER_LOCKED_GATES_1_BASED]
PAPER_ACTIVE_GATES_1_BASED = [2] + list(range(4, 20))
PAPER_ACTIVE_INSTRUCTIONS_0_BASED = [g - 1 for g in PAPER_ACTIVE_GATES_1_BASED]

PAPER_GATE_NAMES_1_BASED = {
    1: "H",  2: "P",  3: "H",  4: "P",  5: "CX", 6: "P",  7: "CX",
    8: "H",  9: "P", 10: "H", 11: "P", 12: "CX", 13: "P", 14: "CX",
    15: "RY", 16: "RY", 17: "CX", 18: "RY", 19: "RY",
}

FEATURE_PARAM_NAMES = [f"feat_p{i}" for i in range(6)]
THETA_PARAM_NAMES = [f"theta_{i}" for i in range(4)]


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    title: str
    partition_1_based: Dict[str, List[int]]
    long_names: Dict[str, str]
    short_labels: Dict[str, str]
    description: Dict[str, str]

    @property
    def labels(self) -> List[str]:
        return list(self.partition_1_based.keys())

    @property
    def partition_0_based(self) -> Dict[str, List[int]]:
        return {label: [g - 1 for g in gates] for label, gates in self.partition_1_based.items()}


# Important: the architecture list the user previously wrote as
# [1,3,4,5,6], [7,...,13], [14,...,18] was the 0-based version. Since the
# present experiment is described in paper 1-based indices, the corresponding
# 1-based architecture partition is [2,4,5,6,7], [8,...,14], [15,...,19].
PARTITION_SPECS: Dict[str, PartitionSpec] = {
    "architecture": PartitionSpec(
        name="architecture",
        title="architecture partition",
        partition_1_based={
            "B1": [2, 4, 5, 6, 7],
            "B2": [8, 9, 10, 11, 12, 13, 14],
            "C": [15, 16, 17, 18, 19],
        },
        long_names={
            "B1": "feature map block 1",
            "B2": "feature map block 2",
            "C": "trainable classifier",
        },
        short_labels={"B1": "B1", "B2": "B2", "C": "C"},
        description={
            "B1": "first feature-map block: early local phases plus first CX-P-CX interaction motif",
            "B2": "second feature-map block: second encoding layer plus second CX-P-CX interaction motif",
            "C": "trainable classifier: final variational RY/CX/RY layer",
        },
    ),
    "feature_interaction": PartitionSpec(
        name="feature_interaction",
        title="feature-interaction partition",
        partition_1_based={
            "L": [2, 4, 8, 9, 10, 11],
            "XFE": [5, 6, 7, 12, 13, 14],
            "C": [15, 16, 17, 18, 19],
        },
        long_names={
            "L": "local feature encoding",
            "XFE": "cross-feature entangling motifs",
            "C": "trainable classifier",
        },
        short_labels={"L": "L", "XFE": "X", "C": "C"},
        description={
            "L": "single-qubit feature encoding and local basis-preparation gates",
            "XFE": "CX-P(phi(x))-CX motifs that encode cross-feature interaction through entangling structure",
            "C": "trainable classifier: final variational RY/CX/RY layer",
        },
    ),
    "qubit_role": PartitionSpec(
        name="qubit_role",
        title="qubit-role partition",
        partition_1_based={
            "Q0": [2, 8, 9, 15, 18],
            "Q1": [4, 6, 10, 11, 13, 16],
            "2Q": [5, 7, 12, 14, 17],
            "Q1F": [19],
        },
        long_names={
            "Q0": "q0 local readout path",
            "Q1": "q1 local hidden path",
            "2Q": "two-qubit couplers",
            "Q1F": "final q1 post-readout rotation",
        },
        short_labels={"Q0": "q0", "Q1": "q1", "2Q": "2q", "Q1F": "q1f"},
        description={
            "Q0": "local gates acting on measured/readout qubit q0",
            "Q1": "local gates acting on hidden/unmeasured qubit q1",
            "2Q": "two-qubit CX gates that couple readout and hidden paths",
            "Q1F": "final local rotation on q1; expected to be close to irrelevant for q0 readout",
        },
    ),
}


# ---------------------------------------------------------------------------
# Repository / dependency setup
# ---------------------------------------------------------------------------

def infer_repo_root() -> Path:
    """Infer OVQX root whether this script lives in repo root or scripts/."""
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for c in candidates:
        if (c / "qshaptools" / "src" / "qshaptools").exists():
            return c
    return Path.cwd()


def setup_import_paths(repo_root: Path) -> None:
    """Add both qshaptools/src and qshaptools/src/qshaptools to sys.path."""
    src = repo_root / "qshaptools" / "src"
    pkg = src / "qshaptools"
    if not pkg.exists():
        raise FileNotFoundError(
            f"Cannot find qshaptools at {pkg}. Run this script from the OVQX repo "
            "or place it under OVQX/scripts/."
        )
    for p in (str(src), str(pkg)):
        if p not in sys.path:
            sys.path.insert(0, p)


def patch_qiskit_bit_index_compatibility() -> None:
    """Patch .index property for newer Qiskit bit objects if needed."""
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


def import_qiskit_and_qowen() -> Tuple[Any, Any, Any, Any, Any]:
    """Import Qiskit and existing qshaptools Owen implementation."""
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
        from qshaptools.qowen import QuantumOwenValues
        from qshaptools.tools import build_circuit
    except Exception:
        try:
            from qowen import QuantumOwenValues  # type: ignore
            from tools import build_circuit  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Could not import QuantumOwenValues/build_circuit from qshaptools. "
                "Check that this script is inside the OVQX repository."
            ) from exc

    return QuantumCircuit, Parameter, Aer, QuantumOwenValues, build_circuit


# ---------------------------------------------------------------------------
# Minimal QASM quantum instance wrapper
# ---------------------------------------------------------------------------

class SimpleQasmQuantumInstance:
    """Minimal object with .execute(circuits), compatible with qshaptools.

    Each execute call receives a fresh simulator seed drawn from this object's RNG.
    This keeps K repeated value-function evaluations genuinely noisy while still
    reproducible from the run seed.
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
        except TypeError:
            job = self.backend.run(circuits, shots=self.shots)
        return job.result()


# ---------------------------------------------------------------------------
# Data and QNN circuit
# ---------------------------------------------------------------------------

def load_or_create_dataset(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load qnn-data.csv with columns x0, x1, y."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Pass --data-path /Users/iratimush/xqml-thesis/data/qnn-data.csv"
        )
    try:
        df = pd.read_csv(data_path, sep=";")
        if not {"x0", "x1", "y"}.issubset(df.columns):
            df = pd.read_csv(data_path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset at {data_path}") from exc

    required = {"x0", "x1", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {sorted(required)}. Found: {list(df.columns)}")
    X = df[["x0", "x1"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    if len(y) != 20:
        logging.warning("Paper QNN dataset has 20 rows, but loaded %d rows.", len(y))
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1.")
    return X, y


def build_qnn_circuit() -> Any:
    """Build paper Figure 12 QNN, with the 19 gate order preserved exactly."""
    QuantumCircuit, Parameter, _, _, _ = import_qiskit_and_qowen()
    qc = QuantumCircuit(2, 2)
    fp = [Parameter(name) for name in FEATURE_PARAM_NAMES]
    tp = [Parameter(name) for name in THETA_PARAM_NAMES]

    # Feature-map block 1, paper gates 1--7.
    qc.h(0)           # 1
    qc.p(fp[0], 0)    # 2: P(phi_1(x_1))
    qc.h(1)           # 3
    qc.p(fp[1], 1)    # 4: P(phi_2(x_2))
    qc.cx(0, 1)       # 5
    qc.p(fp[2], 1)    # 6: P(phi(x))
    qc.cx(0, 1)       # 7

    # Feature-map block 2, paper gates 8--14.
    qc.h(0)           # 8
    qc.p(fp[3], 0)    # 9: P(phi_1(x_1))
    qc.h(1)           # 10
    qc.p(fp[4], 1)    # 11: P(phi_2(x_2))
    qc.cx(0, 1)       # 12
    qc.p(fp[5], 1)    # 13: P(phi(x))
    qc.cx(0, 1)       # 14

    # Trainable layer, paper gates 15--19.
    qc.ry(tp[0], 0)   # 15
    qc.ry(tp[1], 1)   # 16
    qc.cx(0, 1)       # 17
    qc.ry(tp[2], 0)   # 18
    qc.ry(tp[3], 1)   # 19
    return qc


def instruction_qubit_indices(qc: Any, qargs: Sequence[Any]) -> List[int]:
    out: List[int] = []
    for q in qargs:
        if hasattr(q, "index"):
            out.append(int(q.index))
        else:
            out.append(int(qc.find_bit(q).index))
    return out


def validate_qnn_circuit_and_partition(qc: Any, spec: PartitionSpec) -> None:
    """Fail fast if the QNN or selected partition is inconsistent."""
    if len(qc.data) != 19:
        raise ValueError(f"Expected 19 instructions/gates, got {len(qc.data)}")
    if qc.num_qubits != 2:
        raise ValueError(f"Expected 2 qubits, got {qc.num_qubits}")

    observed = []
    for item in qc.data:
        instr = item.operation if hasattr(item, "operation") else item[0]
        observed.append(instr.name.lower())
    expected = [PAPER_GATE_NAMES_1_BASED[i].lower() for i in range(1, 20)]
    expected = ["p" if x == "p" else "cx" if x == "cx" else "ry" if x == "ry" else "h" for x in expected]
    if observed != expected:
        raise ValueError(f"Gate sequence mismatch.\nObserved: {observed}\nExpected: {expected}")

    active = sorted(set(range(19)) - set(PAPER_LOCKED_INSTRUCTIONS_0_BASED))
    if active != PAPER_ACTIVE_INSTRUCTIONS_0_BASED:
        raise ValueError(f"Active gate mismatch: {active} != {PAPER_ACTIVE_INSTRUCTIONS_0_BASED}")

    flat: List[int] = []
    for label in spec.labels:
        flat.extend(spec.partition_0_based[label])
    if sorted(flat) != sorted(PAPER_ACTIVE_INSTRUCTIONS_0_BASED):
        raise ValueError(
            f"Partition '{spec.name}' must exactly cover active gates.\n"
            f"active={PAPER_ACTIVE_INSTRUCTIONS_0_BASED}\n"
            f"covered={sorted(flat)}\n"
            f"partition_1_based={spec.partition_1_based}"
        )
    if len(flat) != len(set(flat)):
        raise ValueError(f"Partition '{spec.name}' contains duplicate gates.")
    if set(flat) & set(PAPER_LOCKED_INSTRUCTIONS_0_BASED):
        raise ValueError(f"Partition '{spec.name}' contains locked/passive gates.")


def log_circuit_and_partition(qc: Any, output_dir: Path, spec: PartitionSpec) -> None:
    lines: List[str] = []
    lines.append("QNN gate order. Python index is 0-based; paper gate index is 1-based.\n")
    gate_to_group = {g: label for label, gates in spec.partition_0_based.items() for g in gates}
    for idx, item in enumerate(qc.data):
        if hasattr(item, "operation"):
            instr = item.operation
            qargs = item.qubits
        else:
            instr, qargs, _ = item
        qubits = instruction_qubit_indices(qc, qargs)
        params = [str(p) for p in getattr(instr, "params", [])]
        if idx in PAPER_LOCKED_INSTRUCTIONS_0_BASED:
            role = "LOCKED/PASSIVE"
        else:
            role = f"ACTIVE group={gate_to_group.get(idx, '?')}"
        lines.append(
            f"python_idx={idx:2d} | paper_gate={idx+1:2d} | "
            f"name={instr.name:3s} | qubits={qubits} | params={params} | {role}"
        )
    lines.append(f"\nSelected partition: {spec.name} ({spec.title})")
    lines.append("Partition, 1-based paper gate indices:")
    for label in spec.labels:
        lines.append(f"  {label} ({spec.long_names[label]}): {spec.partition_1_based[label]}")
    lines.append("\nPartition, 0-based Python instruction indices:")
    for label in spec.labels:
        lines.append(f"  {label} ({spec.long_names[label]}): {spec.partition_0_based[label]}")
    lines.append("\nPartition semantics:")
    for label in spec.labels:
        lines.append(f"  {label}: {spec.description[label]}")

    text = "\n".join(lines)
    logging.info("\n%s", text)
    with (output_dir / "qnn_owen_gate_partition_log.txt").open("w", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------------------------------------------------------------------
# Binding and one-shot value function
# ---------------------------------------------------------------------------

def feature_values_for_x(x0: float, x1: float) -> Dict[str, float]:
    """Paper feature functions: phi_i(x_i)=2*x_i and phi(x)=2(pi-x1)(pi-x2)."""
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
    """With measure([0,1],[0,1]), Qiskit bitstrings are c1c0; q0 is rightmost."""
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
    """One noisy value-function realization v(S) for a coalition circuit."""
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


def owen_value_callable(
    qc_data: Sequence[Any],
    num_qubits: int,
    S: Sequence[int],
    quantum_instance: Any,
    eval_fun: Any,
    build_circuit_fun: Any,
    **eval_fun_kwargs: Any,
) -> float:
    """Small local wrapper equivalent to qvalues.value_callable."""
    qc, param_def_dict = build_circuit_fun(qc_data, num_qubits, S)
    return float(eval_fun(quantum_instance, qc, param_def_dict, **eval_fun_kwargs))


# ---------------------------------------------------------------------------
# Optional training; paper theta remains recommended
# ---------------------------------------------------------------------------

def estimate_accuracy_many_trials(
    qc: Any,
    X: np.ndarray,
    y: np.ndarray,
    theta: Sequence[float],
    seed: int,
    n_trials: int = 50,
) -> Tuple[float, float, List[float]]:
    _, _, Aer, _, _ = import_qiskit_and_qowen()
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
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        raise ImportError("--train requires scipy.") from exc

    logging.warning(
        "Training mode is provided for convenience. Exact reproduction should use "
        "--use-paper-theta because the paper's optimizer randomness/details are not fully specified."
    )
    rng = np.random.default_rng(base_seed)
    theta0 = rng.uniform(-math.pi, math.pi, size=4)

    def objective(theta: np.ndarray) -> float:
        mean_acc, _, _ = estimate_accuracy_many_trials(
            qc, X, y, theta, seed=int(rng.integers(0, 2**31 - 1)), n_trials=16
        )
        return 1.0 - mean_acc

    res = minimize(objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 1.0})
    if not res.success:
        logging.warning("COBYLA did not report success: %s", res.message)
    return np.asarray(res.x, dtype=float)


# ---------------------------------------------------------------------------
# Owen experiment runner
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    partition_name: str
    K: int
    run_id: int
    seed: int
    alpha: float
    data_path: str
    theta: List[float]
    output_dir: str
    silent: bool = True
    owen_batch_size: Optional[int] = None


def make_jsonable(obj: Any) -> Any:
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


def aggregate_group_scores(
    phi_dict_0_based: Mapping[int, float],
    spec: PartitionSpec,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label in spec.labels:
        vals = [float(phi_dict_0_based[g]) for g in spec.partition_0_based[label]]
        out[label] = {
            "total": float(np.sum(vals)),
            "mean_per_gate": float(np.mean(vals)) if vals else 0.0,
            "num_gates": int(len(vals)),
        }
    return out


def run_single_owen_experiment(config: RunConfig) -> Dict[str, Any]:
    """One independent Owen run for fixed partition, K, and seed."""
    start = time.time()
    repo_root = infer_repo_root()
    setup_import_paths(repo_root)
    _, _, Aer, QuantumOwenValues, build_circuit_fun = import_qiskit_and_qowen()
    spec = PARTITION_SPECS[config.partition_name]

    X, y = load_or_create_dataset(Path(config.data_path).expanduser())
    qc = build_qnn_circuit()
    validate_qnn_circuit_and_partition(qc, spec)

    backend = Aer.get_backend("qasm_simulator")
    qi = SimpleQasmQuantumInstance(backend, shots=1, seed=config.seed)

    partition = [spec.partition_0_based[label] for label in spec.labels]

    qov = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=owen_value_callable,
        value_kwargs_dict={
            "eval_fun": qnn_one_shot_accuracy_eval_fun,
            "build_circuit_fun": build_circuit_fun,
            "X": X,
            "y": y,
            "theta": list(config.theta),
        },
        quantum_instance=qi,
        locked_instructions=PAPER_LOCKED_INSTRUCTIONS_0_BASED,
        owen_sample_frac=float(config.alpha),
        owen_sample_reps=int(config.K),
        evaluate_value_only_once=False,  # fresh K noisy evaluations per sampled/enumerated visit
        sample_in_memory=True,
        owen_sample_seed=int(config.seed),
        owen_batch_size=config.owen_batch_size,
        name=f"qnn_owen_{spec.name}_K{config.K}_run{config.run_id}",
        silent=bool(config.silent),
    )

    phi_dict_raw = qov.run()
    phi_dict = {int(k): float(v) for k, v in phi_dict_raw.items()}
    group_scores = aggregate_group_scores(phi_dict, spec)

    summary = qov.get_summary_dict() if hasattr(qov, "get_summary_dict") else {}
    eval_plan = getattr(qov, "_eval_plan", None)
    num_pairs = getattr(qov, "_num_pairs", None)

    result = {
        "partition_name": spec.name,
        "partition_title": spec.title,
        "K": int(config.K),
        "run_id": int(config.run_id),
        "seed": int(config.seed),
        "alpha": float(config.alpha),
        "theta": list(map(float, config.theta)),
        "partition_labels": spec.labels,
        "partition_long_names": spec.long_names,
        "partition_short_labels": spec.short_labels,
        "partition_0_based": spec.partition_0_based,
        "partition_1_based": spec.partition_1_based,
        "phi_dict_0_based": phi_dict,
        "phi_dict_1_based": {int(k) + 1: float(v) for k, v in phi_dict.items()},
        "group_scores": group_scores,
        "summary": make_jsonable(summary),
        "num_pairs": make_jsonable(num_pairs),
        "eval_plan": make_jsonable(eval_plan),
        "elapsed_seconds": time.time() - start,
    }

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"raw_owen_{spec.name}_K{config.K}_run{config.run_id}_seed{config.seed}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["raw_result_path"] = str(out_file)
    return result


def run_parallel_experiments(
    partition_name: str,
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
    configs: List[RunConfig] = []
    # Offset by partition index so the same K/run pair uses distinct seeds across partitions.
    partition_offset = list(PARTITION_SPECS).index(partition_name) * 10_000_000
    for K in K_values:
        for run_id in range(num_runs):
            seed = int(base_seed + partition_offset + 100_000 * int(K) + run_id)
            configs.append(
                RunConfig(
                    partition_name=partition_name,
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
    logging.info("Launching %d Owen jobs for partition=%s with n_jobs=%d", len(configs), partition_name, max_workers)

    if max_workers == 1:
        for c in configs:
            logging.info("Running partition=%s K=%s run=%s seed=%s", c.partition_name, c.K, c.run_id, c.seed)
            results.append(run_single_owen_experiment(c))
    else:
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {executor.submit(run_single_owen_experiment, c): c for c in configs}
            for fut in cf.as_completed(future_to_config):
                c = future_to_config[fut]
                try:
                    res = fut.result()
                    logging.info(
                        "Finished partition=%s K=%s run=%s seed=%s in %.1fs",
                        c.partition_name, c.K, c.run_id, c.seed, res.get("elapsed_seconds", float("nan")),
                    )
                    results.append(res)
                except Exception as exc:
                    logging.error("FAILED partition=%s K=%s run=%s seed=%s", c.partition_name, c.K, c.run_id, c.seed)
                    logging.error("%s", traceback.format_exc())
                    raise exc

    results = sorted(results, key=lambda r: (int(r["K"]), int(r["run_id"])))
    with (output_dir / "all_raw_owen_results.json").open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(results), f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------

def aggregate_results(results: Sequence[Dict[str, Any]], spec: PartitionSpec) -> Dict[int, Dict[str, Any]]:
    by_k: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_k.setdefault(int(r["K"]), []).append(r)

    aggregated: Dict[int, Dict[str, Any]] = {}
    for K, rows in sorted(by_k.items()):
        values_by_gate: Dict[int, List[float]] = {g: [] for g in PAPER_ACTIVE_GATES_1_BASED}
        for r in rows:
            phi_1 = {int(k): float(v) for k, v in r["phi_dict_1_based"].items()}
            for g in PAPER_ACTIVE_GATES_1_BASED:
                values_by_gate[g].append(float(phi_1[g]))

        group_totals: Dict[str, List[float]] = {label: [] for label in spec.labels}
        group_means: Dict[str, List[float]] = {label: [] for label in spec.labels}
        for r in rows:
            gs = r["group_scores"]
            for label in spec.labels:
                group_totals[label].append(float(gs[label]["total"]))
                group_means[label].append(float(gs[label]["mean_per_gate"]))

        gate_groups = {
            g: next(label for label in spec.labels if g in spec.partition_1_based[label])
            for g in PAPER_ACTIVE_GATES_1_BASED
        }
        aggregated[K] = {
            "partition_name": spec.name,
            "partition_title": spec.title,
            "K": K,
            "num_runs": len(rows),
            "gate_indices_1_based": PAPER_ACTIVE_GATES_1_BASED,
            "gate_names": [PAPER_GATE_NAMES_1_BASED[g] for g in PAPER_ACTIVE_GATES_1_BASED],
            "gate_groups": gate_groups,
            "mean": {g: float(np.mean(values_by_gate[g])) for g in PAPER_ACTIVE_GATES_1_BASED},
            "std": {g: float(np.std(values_by_gate[g], ddof=0)) for g in PAPER_ACTIVE_GATES_1_BASED},
            "values": {g: list(map(float, values_by_gate[g])) for g in PAPER_ACTIVE_GATES_1_BASED},
            "group_total_mean": {label: float(np.mean(group_totals[label])) for label in spec.labels},
            "group_total_std": {label: float(np.std(group_totals[label], ddof=0)) for label in spec.labels},
            "group_mean_per_gate_mean": {label: float(np.mean(group_means[label])) for label in spec.labels},
            "group_mean_per_gate_std": {label: float(np.std(group_means[label], ddof=0)) for label in spec.labels},
            "group_total_values": {label: list(map(float, group_totals[label])) for label in spec.labels},
            "group_mean_per_gate_values": {label: list(map(float, group_means[label])) for label in spec.labels},
        }
    return aggregated


def write_aggregated_csvs(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path, spec: PartitionSpec) -> None:
    gate_rows = []
    group_rows = []
    for K, info in aggregated.items():
        for g in info["gate_indices_1_based"]:
            label = info["gate_groups"][g]
            gate_rows.append({
                "partition": spec.name,
                "K": K,
                "gate_1_based": g,
                "gate_0_based": g - 1,
                "gate_name": PAPER_GATE_NAMES_1_BASED[g],
                "group": label,
                "group_name": spec.long_names[label],
                "mean_owen": info["mean"][g],
                "std_owen": info["std"][g],
                "num_runs": info["num_runs"],
                "run_values": json.dumps(info["values"][g]),
            })
        for label in spec.labels:
            group_rows.append({
                "partition": spec.name,
                "K": K,
                "group": label,
                "group_name": spec.long_names[label],
                "gates_1_based": json.dumps(spec.partition_1_based[label]),
                "num_gates": len(spec.partition_1_based[label]),
                "mean_group_total": info["group_total_mean"][label],
                "std_group_total": info["group_total_std"][label],
                "mean_group_mean_per_gate": info["group_mean_per_gate_mean"][label],
                "std_group_mean_per_gate": info["group_mean_per_gate_std"][label],
                "num_runs": info["num_runs"],
                "group_total_run_values": json.dumps(info["group_total_values"][label]),
                "group_mean_per_gate_run_values": json.dumps(info["group_mean_per_gate_values"][label]),
            })
    pd.DataFrame(gate_rows).to_csv(output_dir / "aggregated_owen_gate_values.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "aggregated_owen_group_values.csv", index=False)


def plot_fig13_style(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path, spec: PartitionSpec, K: int = 32) -> Path:
    if K not in aggregated:
        raise ValueError(f"Cannot plot K={K}: not present in results.")
    info = aggregated[K]
    gates = info["gate_indices_1_based"]
    means = [info["mean"][g] for g in gates]
    stds = [info["std"][g] for g in gates]
    names = [PAPER_GATE_NAMES_1_BASED[g] for g in gates]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.errorbar(gates, means, yerr=stds, fmt="o", capsize=3, markersize=5, label=f"K = {K}")
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xlabel("gate index g")
    ax.set_ylabel(r"$Ow^{(g)}$")
    ax.set_title(f"QNN Owen values — {spec.title}")
    ax.set_xticks(gates)
    ax.legend()

    # Add compact group labels below each tick.
    ymin, ymax = ax.get_ylim()
    for g in gates:
        group = info["gate_groups"][g]
        ax.text(g, ymin, spec.short_labels[group], ha="center", va="bottom", fontsize=8, alpha=0.75)
    ax.set_ylim(ymin, ymax)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(gates)
    ax_top.set_xticklabels(names, fontsize=8)

    fig.tight_layout()
    path = output_dir / f"fig13_style_owen_{spec.name}_K{K}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_fig14_style(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path, spec: PartitionSpec, K_values: Sequence[int]) -> Path:
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
    ax.set_ylabel(r"$Ow^{(g)}$")
    ax.set_title(f"QNN Owen values — K comparison, {spec.title}")
    ax.set_xticks(gates)
    ax.legend()

    names = [PAPER_GATE_NAMES_1_BASED[g] for g in gates]
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(gates)
    ax_top.set_xticklabels(names, fontsize=8)

    fig.tight_layout()
    path = output_dir / f"fig14_style_owen_K_comparison_{spec.name}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_group_totals(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path, spec: PartitionSpec, K: int = 32) -> Path:
    """Extra diagnostic plot: group total Owen values at K=32."""
    if K not in aggregated:
        raise ValueError(f"Cannot plot group totals for K={K}: not present.")
    info = aggregated[K]
    labels = spec.labels
    x = np.arange(len(labels))
    means = [info["group_total_mean"][l] for l in labels]
    stds = [info["group_total_std"][l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(labels)), 4.5))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=4, markersize=6)
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{spec.long_names[l]}" for l in labels])
    ax.set_ylabel("sum of gate-level Owen values in group")
    ax.set_title(f"QNN Owen group totals, K={K}, {spec.title}")
    fig.tight_layout()
    path = output_dir / f"owen_group_totals_{spec.name}_K{K}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_group_mean_per_gate(aggregated: Mapping[int, Dict[str, Any]], output_dir: Path, spec: PartitionSpec, K: int = 32) -> Path:
    """Extra diagnostic plot: mean Owen value per gate in each group."""
    if K not in aggregated:
        raise ValueError(f"Cannot plot group means for K={K}: not present.")
    info = aggregated[K]
    labels = spec.labels
    x = np.arange(len(labels))
    means = [info["group_mean_per_gate_mean"][l] for l in labels]
    stds = [info["group_mean_per_gate_std"][l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(labels)), 4.5))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=4, markersize=6)
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{spec.long_names[l]}" for l in labels])
    ax.set_ylabel("mean gate-level Owen value per gate")
    ax.set_title(f"QNN Owen group means per gate, K={K}, {spec.title}")
    fig.tight_layout()
    path = output_dir / f"owen_group_mean_per_gate_{spec.name}_K{K}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = infer_repo_root()
    default_data_path = Path("/Users/iratimush/xqml-thesis/data/qnn-data.csv")
    if not default_data_path.exists():
        default_data_path = repo_root / "data" / "qnn-data.csv"

    p = argparse.ArgumentParser(
        description="Run QNN Owen values for architecture/feature-interaction/qubit-role partitions."
    )
    p.add_argument("--partition", choices=list(PARTITION_SPECS.keys()) + ["all"], default="all")
    p.add_argument("--mode", choices=["fig13", "fig14", "both"], default="both")
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 8, 16, 32])
    p.add_argument("--num-runs", type=int, default=5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--base-seed", type=int, default=123)
    p.add_argument("--n-jobs", type=int, default=5)
    p.add_argument("--data-path", type=Path, default=default_data_path)
    p.add_argument("--output-dir", type=Path, default=repo_root / "results" / "qnn_owen_partitions")
    p.add_argument("--use-paper-theta", action="store_true", help="Use theta reported in the paper. Recommended.")
    p.add_argument("--train", action="store_true", help="Train theta with COBYLA instead of using paper theta.")
    p.add_argument("--train-maxiter", type=int, default=250)
    p.add_argument("--no-silent", action="store_true", help="Show qshaptools progress bars inside worker processes.")
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
    elif args.mode == "both":
        args.k_values = sorted(set(args.k_values + [32]))
    if args.num_runs < 1:
        p.error("--num-runs must be >= 1")
    if args.n_jobs < 1:
        p.error("--n-jobs must be >= 1")
    if args.alpha <= 0:
        p.error("--alpha must be > 0")
    return args


def save_config(args: argparse.Namespace, theta: Sequence[float], output_dir: Path, spec: PartitionSpec) -> None:
    config = {
        "experiment": f"QNN Owen values with {spec.name} coalition structure",
        "partition": spec.name,
        "partition_title": spec.title,
        "mode": args.mode,
        "k_values": args.k_values,
        "num_runs": args.num_runs,
        "alpha": args.alpha,
        "alpha_note": "alpha=1.0 evaluates the full structured Owen coalition space for these small partitions; K still averages shot noise.",
        "base_seed": args.base_seed,
        "n_jobs": args.n_jobs,
        "data_path": str(args.data_path),
        "output_dir": str(output_dir),
        "use_paper_theta": bool(args.use_paper_theta),
        "train": bool(args.train),
        "theta": list(map(float, theta)),
        "paper_locked_gates_1_based": PAPER_LOCKED_GATES_1_BASED,
        "paper_locked_instructions_0_based": PAPER_LOCKED_INSTRUCTIONS_0_BASED,
        "paper_active_gates_1_based": PAPER_ACTIVE_GATES_1_BASED,
        "partition_1_based": spec.partition_1_based,
        "partition_0_based": spec.partition_0_based,
        "partition_semantics": spec.description,
        "value_function": "one-shot test accuracy; first-qubit readout; no retraining",
        "estimator_note": (
            "For each Owen visit (R,T), v(Q_R union T) and "
            "v(Q_R union T union {i}) are each evaluated K fresh times; the two "
            "K-means are differenced and then averaged over visits."
        ),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def log_effective_owen_context_counts(spec: PartitionSpec, alpha: float) -> None:
    """Log the number of structured Owen contexts implied by a partition."""
    m = len(spec.labels)
    logging.info("Effective Owen contexts for partition=%s, alpha=%s", spec.name, alpha)
    for label in spec.labels:
        group_size = len(spec.partition_1_based[label])
        total_contexts = (2 ** (m - 1)) * (2 ** (group_size - 1))
        sampled_contexts = math.ceil(alpha * total_contexts)
        logging.info(
            "  %s (%s): group_size=%d, total_contexts_per_gate=%d, selected_contexts≈%d",
            label, spec.long_names[label], group_size, total_contexts, sampled_contexts,
        )


def run_for_partition(args: argparse.Namespace, spec: PartitionSpec, theta: Sequence[float], qc: Any, X: np.ndarray, y: np.ndarray) -> None:
    partition_output_dir = args.output_dir.expanduser().resolve() / spec.name
    partition_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("\n%s", "=" * 90)
    logging.info("Running partition: %s (%s)", spec.name, spec.title)
    logging.info("Output dir: %s", partition_output_dir)

    validate_qnn_circuit_and_partition(qc, spec)
    log_circuit_and_partition(qc, partition_output_dir, spec)
    save_config(args, theta, partition_output_dir, spec)
    log_effective_owen_context_counts(spec, args.alpha)

    results = run_parallel_experiments(
        partition_name=spec.name,
        K_values=args.k_values,
        num_runs=args.num_runs,
        n_jobs=args.n_jobs,
        alpha=args.alpha,
        base_seed=args.base_seed,
        data_path=args.data_path.expanduser(),
        theta=theta,
        output_dir=partition_output_dir,
        silent=not args.no_silent,
    )

    aggregated = aggregate_results(results, spec)
    with (partition_output_dir / "aggregated_owen.json").open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(aggregated), f, indent=2)
    write_aggregated_csvs(aggregated, partition_output_dir, spec)

    if args.mode in {"fig13", "both"}:
        fig13 = plot_fig13_style(aggregated, partition_output_dir, spec, K=32)
        logging.info("Saved Figure 13-style Owen plot: %s", fig13)
        group_plot = plot_group_totals(aggregated, partition_output_dir, spec, K=32)
        logging.info("Saved Owen group-total plot: %s", group_plot)
        group_mean_plot = plot_group_mean_per_gate(aggregated, partition_output_dir, spec, K=32)
        logging.info("Saved Owen group-mean-per-gate plot: %s", group_mean_plot)
    if args.mode in {"fig14", "both"}:
        fig14 = plot_fig14_style(aggregated, partition_output_dir, spec, K_values=args.k_values)
        logging.info("Saved Figure 14-style Owen plot: %s", fig14)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    repo_root = infer_repo_root()
    setup_import_paths(repo_root)
    import_qiskit_and_qowen()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Repo root: %s", repo_root)
    logging.info("Output root: %s", output_dir)
    logging.info("Data path: %s", args.data_path)

    X, y = load_or_create_dataset(args.data_path.expanduser())
    logging.info("Loaded dataset: X shape=%s, y shape=%s, label counts=%s", X.shape, y.shape, dict(pd.Series(y).value_counts()))

    qc = build_qnn_circuit()

    if args.train:
        theta = train_qnn_if_requested(qc, X, y, base_seed=args.base_seed, maxiter=args.train_maxiter)
    else:
        theta = PAPER_THETA.copy()
    logging.info("Using theta = %s", np.array2string(np.asarray(theta), precision=6))

    mean_acc, std_acc, sanity_accs = estimate_accuracy_many_trials(
        qc, X, y, theta, seed=args.base_seed + 999, n_trials=args.sanity_trials
    )
    logging.info(
        "Full-circuit one-shot accuracy sanity check: %.3f +/- %.3f over %d trials. Paper reports about 0.80.",
        mean_acc, std_acc, args.sanity_trials,
    )
    with (output_dir / "full_circuit_sanity_accuracy.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": mean_acc, "std": std_acc, "trials": sanity_accs}, f, indent=2)

    if args.partition == "all":
        partition_names = list(PARTITION_SPECS.keys())
    else:
        partition_names = [args.partition]

    for partition_name in partition_names:
        run_for_partition(args, PARTITION_SPECS[partition_name], theta, qc, X, y)

    logging.info("Done.")


if __name__ == "__main__":
    main()
