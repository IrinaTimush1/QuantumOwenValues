#!/usr/bin/env python3
"""
Shared utilities for the QSVM Owen-value experiments.

The code in this module keeps the paper/QSVM-specific machinery out of the
entrypoint scripts:

* robust loading of the downloaded QSVM CSV files,
* construction and validation of the two-qubit SVQX feature map,
* exact deterministic statevector kernel evaluation,
* a CSV-backed QSVM accuracy value function for qshaptools QuantumOwenValues,
* output writers and plotting helpers.

All public gate indices in saved files and plots are paper-style 1-based
indices. qshaptools receives the same gates as Qiskit instruction indices,
which are 0-based.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_ACCURACIES: Dict[int, float] = {1: 0.842, 2: 0.986, 3: 0.913}
GROUP_LABELS: List[str] = ["E", "M", "X"]
GROUP_COLORS: Dict[str, str] = {
    "E": "#386cb0",
    "M": "#fdb462",
    "X": "#7fc97f",
    "passive": "#bdbdbd",
}
PAPER_STYLE: Dict[str, Any] = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
}
PAPER_SERIES_STYLE: Dict[int, Dict[str, Any]] = {
    1: {"color": "#004663", "marker": "o", "label": r"$r=1$, exact"},
    2: {"color": "#be5590", "marker": "s", "label": r"$r=2$, exact"},
    3: {"color": "#ff9900", "marker": "^", "label": r"$r=3$, exact"},
}


# ---------------------------------------------------------------------------
# Repository and dependency setup
# ---------------------------------------------------------------------------


def infer_repo_root() -> Path:
    """Infer the thesis repository root from this file or the CWD."""
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "qshaptools" / "src" / "qshaptools").exists():
            return candidate
    return Path.cwd()


def setup_import_paths(repo_root: Path) -> None:
    """Add qshaptools/src to sys.path for local, non-installed use."""
    src = repo_root / "qshaptools" / "src"
    if not (src / "qshaptools").exists():
        raise FileNotFoundError(f"Cannot find qshaptools package under {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def patch_qiskit_bit_index_compatibility() -> None:
    """Provide .index on Qiskit bits when running with newer Qiskit objects."""
    try:
        from qiskit.circuit import Clbit, Qubit
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


def import_qiskit_and_qowen(repo_root: Optional[Path] = None) -> Tuple[Any, Any, Any]:
    """Import Qiskit and the existing qshaptools QuantumOwenValues class."""
    if repo_root is None:
        repo_root = infer_repo_root()
    setup_import_paths(repo_root)
    patch_qiskit_bit_index_compatibility()
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except Exception as exc:
        raise ImportError("Could not import qiskit. Use the repository virtualenv.") from exc
    try:
        from qshaptools.qowen import QuantumOwenValues
    except Exception as exc:
        raise ImportError("Could not import qshaptools.qowen.QuantumOwenValues.") from exc
    return QuantumCircuit, Statevector, QuantumOwenValues


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def str_to_bool(value: Any) -> bool:
    """Parse argparse-friendly boolean strings."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def coalition_key(gates_1_based: Iterable[int]) -> str:
    """Return a stable CSV key for a sorted 1-based active-gate coalition."""
    gates = sorted(int(g) for g in gates_1_based)
    return "-".join(str(g) for g in gates) if gates else "empty"


def parse_coalition_key(key: Any) -> List[int]:
    """Inverse of coalition_key."""
    text = str(key).strip()
    if text in {"", "empty", "nan", "None"}:
        return []
    return [int(part) for part in text.split("-") if part]


def gate_list_string(gates: Iterable[int]) -> str:
    """Compact, deterministic representation of a gate-index list."""
    return "-".join(str(int(g)) for g in sorted(gates))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, indent=2, sort_keys=True)


def make_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_jsonable(v) for v in obj]
    return obj



def file_sha256(path: Path) -> str:
    """Small-content hash used to prevent stale QSVM value-cache reuse."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_cache_metadata(dataset: "QSVMDataset") -> Dict[str, str]:
    """Metadata that must match before a cached coalition value is reused."""
    return {
        "train_path": str(Path(dataset.train_path).expanduser().resolve()),
        "test_path": str(Path(dataset.test_path).expanduser().resolve()),
        "train_shape": "x".join(str(x) for x in dataset.X_train.shape),
        "test_shape": "x".join(str(x) for x in dataset.X_test.shape),
        "feature_columns": "|".join(dataset.feature_columns),
        "label_column": str(dataset.label_column),
        "train_sha256": file_sha256(Path(dataset.train_path)),
        "test_sha256": file_sha256(Path(dataset.test_path)),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QSVMDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_path: Path
    test_path: Path
    data_index: int
    feature_columns: Tuple[str, str]
    label_column: str

    @property
    def train_shape(self) -> Tuple[int, int]:
        return tuple(self.X_train.shape)  # type: ignore[return-value]

    @property
    def test_shape(self) -> Tuple[int, int]:
        return tuple(self.X_test.shape)  # type: ignore[return-value]


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a semicolon- or comma-separated CSV, preferring semicolon."""
    errors: List[str] = []
    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                return df
            errors.append(f"sep={sep!r} produced one column")
        except Exception as exc:
            errors.append(f"sep={sep!r}: {exc}")
    raise ValueError(f"Could not parse CSV at {path}. Attempts: {'; '.join(errors)}")


def infer_feature_and_label_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """Infer x0/x1/y columns when names are not exactly canonical."""
    lower_to_col = {str(c).strip().lower(): c for c in df.columns}

    if {"x0", "x1", "y"}.issubset(lower_to_col):
        return [lower_to_col["x0"], lower_to_col["x1"]], lower_to_col["y"]

    label_candidates = ["y", "label", "class", "target"]
    label_col: Optional[str] = None
    for name in label_candidates:
        if name in lower_to_col:
            label_col = lower_to_col[name]
            break

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric_non_index = [
        c for c in numeric_cols
        if str(c).strip().lower() not in {"idx", "index", "id"}
    ]

    if label_col is None:
        if len(numeric_non_index) < 3:
            raise ValueError(
                "Could not infer a label column. Expected y/label/class or at least "
                f"three numeric non-index columns, found {list(df.columns)}."
            )
        label_col = numeric_non_index[2]

    feature_cols = [c for c in numeric_non_index if c != label_col][:2]
    if len(feature_cols) < 2:
        raise ValueError(
            "Could not infer two numeric feature columns. "
            f"Columns were {list(df.columns)}, label inferred as {label_col!r}."
        )
    return feature_cols, label_col


def load_qsvm_dataset(
    data_index: int = 0,
    data_dir: Path | str = Path("data"),
    train_path: Optional[Path | str] = None,
    test_path: Optional[Path | str] = None,
    swap_features: bool = False,
) -> QSVMDataset:
    """Load train and test CSVs for a QSVM dataset index."""
    data_dir = Path(data_dir)
    train_p = Path(train_path) if train_path is not None else data_dir / f"qsvm-traindata-{data_index}.csv"
    test_p = Path(test_path) if test_path is not None else data_dir / f"qsvm-testdata-{data_index}.csv"

    if not train_p.exists():
        raise FileNotFoundError(f"Training data not found: {train_p}")
    if not test_p.exists():
        raise FileNotFoundError(f"Test data not found: {test_p}")

    train_df = read_csv_robust(train_p)
    test_df = read_csv_robust(test_p)
    train_features, train_label = infer_feature_and_label_columns(train_df)
    test_features, test_label = infer_feature_and_label_columns(test_df)

    if swap_features:
        train_features = [train_features[1], train_features[0]]
        test_features = [test_features[1], test_features[0]]

    X_train = train_df[train_features].to_numpy(dtype=float)
    y_train = train_df[train_label].to_numpy()
    X_test = test_df[test_features].to_numpy(dtype=float)
    y_test = test_df[test_label].to_numpy()

    return QSVMDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_path=train_p,
        test_path=test_p,
        data_index=int(data_index),
        feature_columns=(str(train_features[0]), str(train_features[1])),
        label_column=str(train_label),
    )


def print_dataset_report(dataset: QSVMDataset, max_rows: int = 5) -> None:
    """Print the diagnostics requested for the loaded dataset."""
    print(f"Train path: {dataset.train_path}")
    print(f"Test path:  {dataset.test_path}")
    print(f"Train shape: {dataset.X_train.shape}, labels: {dataset.y_train.shape}")
    print(f"Test shape:  {dataset.X_test.shape}, labels: {dataset.y_test.shape}")

    def dist(y: np.ndarray) -> Dict[Any, int]:
        unique, counts = np.unique(y, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts)}

    print(f"Train label distribution: {dist(dataset.y_train)}")
    print(f"Test label distribution:  {dist(dataset.y_test)}")
    all_x = np.vstack([dataset.X_train, dataset.X_test])
    mins = np.min(all_x, axis=0)
    maxs = np.max(all_x, axis=0)
    print(f"Feature min: x0={mins[0]:.6g}, x1={mins[1]:.6g}")
    print(f"Feature max: x0={maxs[0]:.6g}, x1={maxs[1]:.6g}")
    preview = pd.DataFrame(
        np.column_stack([dataset.X_train[:max_rows], dataset.y_train[:max_rows]]),
        columns=[dataset.feature_columns[0], dataset.feature_columns[1], dataset.label_column],
    )
    print("First training rows:")
    print(preview.to_string(index=False))


# ---------------------------------------------------------------------------
# Feature-map construction and partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QSVMConvention:
    """Feature-map convention switches used by the optional search mode."""

    cnot_control: int = 0
    cnot_target: int = 1
    cross_phase_on: str = "target"  # "target" or "control"
    feature_order: str = "normal"  # "normal" or "swapped"
    simulator: str = "manual"  # "manual" or "qiskit"

    @property
    def cross_phase_qubit(self) -> int:
        if self.cross_phase_on == "target":
            return self.cnot_target
        if self.cross_phase_on == "control":
            return self.cnot_control
        raise ValueError(f"Unknown cross_phase_on={self.cross_phase_on!r}")

    @property
    def name(self) -> str:
        direction = f"{self.cnot_control}->{self.cnot_target}"
        return (
            f"cnot_{direction},cross_on_{self.cross_phase_on},"
            f"features_{self.feature_order},sim_{self.simulator}"
        )


DEFAULT_CONVENTION = QSVMConvention()


@dataclass(frozen=True)
class QSVMPartition:
    r: int
    passive: List[int]
    groups: Dict[str, List[int]]

    @property
    def active_gates(self) -> List[int]:
        out: List[int] = []
        for label in GROUP_LABELS:
            out.extend(self.groups[label])
        return sorted(out)

    @property
    def all_gates(self) -> List[int]:
        return list(range(1, 7 * self.r + 1))

    @property
    def groups_0_based(self) -> Dict[str, List[int]]:
        return {label: [g - 1 for g in gates] for label, gates in self.groups.items()}

    @property
    def passive_0_based(self) -> List[int]:
        return [g - 1 for g in self.passive]


def qsvm_gate_name(gate_index_1_based: int) -> str:
    pattern = ["H", "P", "H", "P", "CX", "P", "CX"]
    return pattern[(gate_index_1_based - 1) % 7]


def make_qsvm_partition(r: int) -> QSVMPartition:
    """Return the requested E/M/X partition for the QSVM feature map."""
    if r not in {1, 2, 3}:
        raise ValueError(f"This QSVM experiment supports r in {{1,2,3}}, got {r}")
    passive: List[int] = []
    groups: Dict[str, List[int]] = {"E": [], "M": [], "X": []}
    for rep in range(r):
        offset = 7 * rep
        passive.extend([offset + 1, offset + 3])
        groups["M"].extend([offset + 2, offset + 4])
        groups["E"].extend([offset + 5, offset + 7])
        groups["X"].append(offset + 6)
    partition = QSVMPartition(r=r, passive=sorted(passive), groups={k: sorted(v) for k, v in groups.items()})
    validate_qsvm_partition(partition)
    return partition


def validate_qsvm_partition(partition: QSVMPartition) -> None:
    """Fail fast if passive gates and E/M/X do not partition 1..7r."""
    all_gates = set(partition.all_gates)
    passive = set(partition.passive)
    group_sets = {label: set(partition.groups[label]) for label in GROUP_LABELS}
    active = set().union(*group_sets.values())

    if passive & active:
        raise ValueError(f"Passive gates overlap active groups: {sorted(passive & active)}")
    for a, b in itertools.combinations(GROUP_LABELS, 2):
        overlap = group_sets[a] & group_sets[b]
        if overlap:
            raise ValueError(f"Groups {a} and {b} overlap: {sorted(overlap)}")
    if passive | active != all_gates:
        missing = sorted(all_gates - (passive | active))
        extra = sorted((passive | active) - all_gates)
        raise ValueError(f"Partition does not cover 1..7r. missing={missing}, extra={extra}")
    bad_passive = [g for g in passive if qsvm_gate_name(g) != "H"]
    if bad_passive:
        raise ValueError(f"Only H gates may be passive, got non-H passive gates {bad_passive}")
    for g in group_sets["E"]:
        if qsvm_gate_name(g) != "CX":
            raise ValueError(f"E group must contain CX only, found gate {g}={qsvm_gate_name(g)}")
    for g in group_sets["M"] | group_sets["X"]:
        if qsvm_gate_name(g) != "P":
            raise ValueError(f"M/X groups must contain P only, found gate {g}={qsvm_gate_name(g)}")


def gate_subtype(gate_index_1_based: int, partition: QSVMPartition) -> str:
    if gate_index_1_based in partition.groups["M"]:
        return "local_P"
    if gate_index_1_based in partition.groups["X"]:
        return "cross_P"
    return ""


def gate_group(gate_index_1_based: int, partition: QSVMPartition) -> str:
    if gate_index_1_based in partition.passive:
        return "passive"
    for label in GROUP_LABELS:
        if gate_index_1_based in partition.groups[label]:
            return label
    raise KeyError(f"Gate {gate_index_1_based} not in partition")


def qsvm_gate_metadata(r: int) -> List[Dict[str, Any]]:
    partition = make_qsvm_partition(r)
    rows: List[Dict[str, Any]] = []
    for gate in partition.all_gates:
        rows.append(
            {
                "r": r,
                "gate_index_1based": gate,
                "gate_name": qsvm_gate_name(gate),
                "group": gate_group(gate, partition),
                "gate_subtype": gate_subtype(gate, partition),
            }
        )
    return rows


def print_partition_report(partition: QSVMPartition) -> None:
    print(f"QSVM r={partition.r} partition")
    print(f"  passive H: {partition.passive}")
    for label in GROUP_LABELS:
        print(f"  {label}: {partition.groups[label]}")
    print(f"  active gates: {partition.active_gates}")


def _feature_angles(x: Sequence[float], convention: QSVMConvention) -> Tuple[float, float, float]:
    if convention.feature_order == "swapped":
        x0, x1 = float(x[1]), float(x[0])
    elif convention.feature_order == "normal":
        x0, x1 = float(x[0]), float(x[1])
    else:
        raise ValueError(f"Unknown feature_order={convention.feature_order!r}")
    phi_1 = 2.0 * x0
    phi_2 = 2.0 * x1
    phi_cross = 2.0 * (math.pi - x0) * (math.pi - x1)
    return phi_1, phi_2, phi_cross


def build_qsvm_feature_map(
    x: Sequence[float],
    r: int,
    active_gates: Optional[Set[int] | Sequence[int]] = None,
    passive_gates: Optional[Set[int] | Sequence[int]] = None,
    convention: QSVMConvention = DEFAULT_CONVENTION,
) -> Any:
    """Build the two-qubit second-order Pauli-Z feature map.

    Gate indices are paper-style 1-based. If active_gates is None, all gates
    are included. Otherwise only active_gates plus passive_gates are included.
    """
    QuantumCircuit, _, _ = import_qiskit_and_qowen()
    qc = QuantumCircuit(2)
    active_set = None if active_gates is None else set(int(g) for g in active_gates)
    passive_set = set(int(g) for g in (passive_gates or []))

    def include(gate_index: int) -> bool:
        return active_set is None or gate_index in active_set or gate_index in passive_set

    phi_1, phi_2, phi_cross = _feature_angles(x, convention)
    for rep in range(r):
        offset = 7 * rep
        if include(offset + 1):
            qc.h(0)
        if include(offset + 2):
            qc.p(phi_1, 0)
        if include(offset + 3):
            qc.h(1)
        if include(offset + 4):
            qc.p(phi_2, 1)
        if include(offset + 5):
            qc.cx(convention.cnot_control, convention.cnot_target)
        if include(offset + 6):
            qc.p(phi_cross, convention.cross_phase_qubit)
        if include(offset + 7):
            qc.cx(convention.cnot_control, convention.cnot_target)
    return qc


# ---------------------------------------------------------------------------
# Exact statevector kernel and QSVM value function
# ---------------------------------------------------------------------------


def _apply_h(states: np.ndarray, qubit: int) -> None:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    if qubit == 0:
        a = states[:, 0].copy()
        b = states[:, 1].copy()
        states[:, 0] = (a + b) * inv_sqrt2
        states[:, 1] = (a - b) * inv_sqrt2
        a = states[:, 2].copy()
        b = states[:, 3].copy()
        states[:, 2] = (a + b) * inv_sqrt2
        states[:, 3] = (a - b) * inv_sqrt2
    elif qubit == 1:
        a = states[:, 0].copy()
        b = states[:, 2].copy()
        states[:, 0] = (a + b) * inv_sqrt2
        states[:, 2] = (a - b) * inv_sqrt2
        a = states[:, 1].copy()
        b = states[:, 3].copy()
        states[:, 1] = (a + b) * inv_sqrt2
        states[:, 3] = (a - b) * inv_sqrt2
    else:
        raise ValueError(f"Invalid qubit {qubit}")


def _apply_phase(states: np.ndarray, qubit: int, angles: np.ndarray) -> None:
    phase = np.exp(1j * angles)
    if qubit == 0:
        states[:, 1] *= phase
        states[:, 3] *= phase
    elif qubit == 1:
        states[:, 2] *= phase
        states[:, 3] *= phase
    else:
        raise ValueError(f"Invalid qubit {qubit}")


def _apply_cx(states: np.ndarray, control: int, target: int) -> None:
    if control == 0 and target == 1:
        states[:, [1, 3]] = states[:, [3, 1]]
    elif control == 1 and target == 0:
        states[:, [2, 3]] = states[:, [3, 2]]
    else:
        raise ValueError(f"Only two-qubit CX 0->1 or 1->0 is supported, got {control}->{target}")


def compute_statevectors_manual(
    X: np.ndarray,
    r: int,
    active_gates: Optional[Set[int] | Sequence[int]],
    passive_gates: Optional[Set[int] | Sequence[int]],
    convention: QSVMConvention = DEFAULT_CONVENTION,
) -> np.ndarray:
    """Fast exact statevectors for the two-qubit QSVM feature map."""
    X = np.asarray(X, dtype=float)
    states = np.zeros((X.shape[0], 4), dtype=np.complex128)
    states[:, 0] = 1.0
    active_set = None if active_gates is None else set(int(g) for g in active_gates)
    passive_set = set(int(g) for g in (passive_gates or []))

    if convention.feature_order == "swapped":
        x0 = X[:, 1]
        x1 = X[:, 0]
    elif convention.feature_order == "normal":
        x0 = X[:, 0]
        x1 = X[:, 1]
    else:
        raise ValueError(f"Unknown feature_order={convention.feature_order!r}")

    phi_1 = 2.0 * x0
    phi_2 = 2.0 * x1
    phi_cross = 2.0 * (math.pi - x0) * (math.pi - x1)

    def include(gate_index: int) -> bool:
        return active_set is None or gate_index in active_set or gate_index in passive_set

    for rep in range(r):
        offset = 7 * rep
        if include(offset + 1):
            _apply_h(states, 0)
        if include(offset + 2):
            _apply_phase(states, 0, phi_1)
        if include(offset + 3):
            _apply_h(states, 1)
        if include(offset + 4):
            _apply_phase(states, 1, phi_2)
        if include(offset + 5):
            _apply_cx(states, convention.cnot_control, convention.cnot_target)
        if include(offset + 6):
            _apply_phase(states, convention.cross_phase_qubit, phi_cross)
        if include(offset + 7):
            _apply_cx(states, convention.cnot_control, convention.cnot_target)
    return states


def compute_statevectors_qiskit(
    X: np.ndarray,
    r: int,
    active_gates: Optional[Set[int] | Sequence[int]],
    passive_gates: Optional[Set[int] | Sequence[int]],
    convention: QSVMConvention = DEFAULT_CONVENTION,
) -> np.ndarray:
    """Reference exact statevectors using Qiskit Statevector.from_instruction."""
    _, Statevector, _ = import_qiskit_and_qowen()
    rows: List[np.ndarray] = []
    for x in np.asarray(X, dtype=float):
        qc = build_qsvm_feature_map(
            x=x,
            r=r,
            active_gates=active_gates,
            passive_gates=passive_gates,
            convention=convention,
        )
        rows.append(np.asarray(Statevector.from_instruction(qc).data, dtype=np.complex128))
    return np.vstack(rows)


def compute_statevectors(
    X: np.ndarray,
    r: int,
    active_gates: Optional[Set[int] | Sequence[int]],
    passive_gates: Optional[Set[int] | Sequence[int]],
    convention: QSVMConvention = DEFAULT_CONVENTION,
) -> np.ndarray:
    """Compute exact statevectors with the selected backend."""
    if convention.simulator == "manual":
        return compute_statevectors_manual(X, r, active_gates, passive_gates, convention)
    if convention.simulator == "qiskit":
        return compute_statevectors_qiskit(X, r, active_gates, passive_gates, convention)
    raise ValueError(f"Unknown simulator={convention.simulator!r}")


def compute_kernel_from_statevectors(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return K[i,j] = abs(<B[j] | A[i]>)**2 with shape len(A) x len(B)."""
    overlaps = A @ B.conj().T
    kernel = np.abs(overlaps) ** 2
    return np.clip(np.real_if_close(kernel), 0.0, 1.0).astype(float)


def require_sklearn_svc() -> Any:
    try:
        from sklearn.svm import SVC
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required for QSVM training. In this repo, try "
            "`.venv/bin/python -m pip install scikit-learn` and rerun."
        ) from exc
    return SVC


def qsvm_accuracy(
    dataset: QSVMDataset,
    r: int,
    active_gates: Optional[Set[int] | Sequence[int]],
    passive_gates: Optional[Set[int] | Sequence[int]],
    convention: QSVMConvention = DEFAULT_CONVENTION,
    svc_c: float = 1.0,
    check_kernel: bool = True,
) -> float:
    """Train a precomputed-kernel SVC and return deterministic test accuracy."""
    SVC = require_sklearn_svc()

    train_states = compute_statevectors(dataset.X_train, r, active_gates, passive_gates, convention)
    test_states = compute_statevectors(dataset.X_test, r, active_gates, passive_gates, convention)
    K_train = compute_kernel_from_statevectors(train_states, train_states)
    K_test = compute_kernel_from_statevectors(test_states, train_states)

    if check_kernel:
        diag = np.diag(K_train)
        if not np.allclose(diag, 1.0, atol=1e-8):
            raise ValueError(
                f"Training kernel diagonal is not close to 1. min={diag.min()}, max={diag.max()}"
            )
        if not np.allclose(K_train, K_train.T, atol=1e-8):
            raise ValueError("Training kernel is not symmetric within tolerance.")

    clf = SVC(kernel="precomputed", C=float(svc_c))
    clf.fit(K_train, dataset.y_train)
    pred = clf.predict(K_test)
    return float(np.mean(pred == dataset.y_test))


def validate_manual_simulator(r: int = 3, num_points: int = 8) -> float:
    """Backward-compatible full-circuit manual-vs-Qiskit validation."""
    report = validate_manual_simulator_random_subsets(r=r, num_points=num_points, num_random_subsets=0)
    return float(report["max_abs_difference"])


def validate_manual_simulator_random_subsets(
    r: int = 3,
    num_points: int = 8,
    num_random_subsets: int = 10,
    seed: int = 123,
) -> Dict[str, float | int]:
    """Compare the fast manual simulator against Qiskit on random coalitions.

    The previous check only compared the full circuit. This also checks the
    H-only coalition and random active-gate subsets, which is the relevant
    case for Owen values. The comparison is global-phase invariant.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 2.0 * math.pi, size=(num_points, 2))
    partition = make_qsvm_partition(r)
    active = list(partition.active_gates)

    subsets: List[Tuple[int, ...]] = [tuple(), tuple(active)]
    for _ in range(int(num_random_subsets)):
        k = int(rng.integers(0, len(active) + 1))
        subset = tuple(sorted(rng.choice(active, size=k, replace=False).tolist()))
        subsets.append(subset)
    subsets = sorted(set(subsets), key=lambda x: (len(x), x))

    qiskit_conv = QSVMConvention(simulator="qiskit")
    max_abs = 0.0
    max_infidelity = 0.0
    for subset in subsets:
        manual = compute_statevectors_manual(X, r, subset, partition.passive, DEFAULT_CONVENTION)
        qiskit = compute_statevectors_qiskit(X, r, subset, partition.passive, qiskit_conv)
        for m_state, q_state in zip(manual, qiskit):
            overlap = np.vdot(q_state, m_state)
            max_infidelity = max(max_infidelity, float(1.0 - min(1.0, abs(overlap) ** 2)))
            if abs(overlap) > 1e-14:
                m_state = m_state / (overlap / abs(overlap))
            max_abs = max(max_abs, float(np.max(np.abs(m_state - q_state))))

    return {
        "r": int(r),
        "num_points": int(num_points),
        "num_checked_subsets": int(len(subsets)),
        "max_abs_difference": float(max_abs),
        "max_infidelity": float(max_infidelity),
    }


class QSVMAccuracyValueFunction:
    """CSV-backed deterministic value function v(S) for exact Owen runs."""

    cache_fields = [
        "r",
        "coalition_key",
        "active_gate_indices",
        "included_gate_indices_total",
        "n_active_gates",
        "n_total_gates",
        "value_accuracy",
        "data_index",
        "convention",
        "svc_c",
        "train_path",
        "test_path",
        "train_shape",
        "test_shape",
        "feature_columns",
        "label_column",
        "train_sha256",
        "test_sha256",
    ]

    def __init__(
        self,
        dataset: QSVMDataset,
        partition: QSVMPartition,
        cache_path: Path,
        convention: QSVMConvention = DEFAULT_CONVENTION,
        svc_c: float = 1.0,
        force: bool = False,
    ):
        self.dataset = dataset
        self.partition = partition
        self.cache_path = cache_path
        self.convention = convention
        self.svc_c = float(svc_c)
        self.cache: Dict[str, float] = {}
        ensure_dir(cache_path.parent)
        if force and cache_path.exists():
            cache_path.unlink()
        self._load_cache()

    def _cache_row_matches_current_run(self, row: pd.Series) -> bool:
        """Return True only if a persisted value belongs to this exact run setup."""
        if int(row.get("r", -1)) != self.partition.r:
            return False
        if int(row.get("data_index", -1)) != self.dataset.data_index:
            return False
        if str(row.get("convention", "")) != self.convention.name:
            return False
        try:
            if not math.isclose(float(row.get("svc_c", np.nan)), self.svc_c, rel_tol=0.0, abs_tol=1e-12):
                return False
        except Exception:
            return False

        expected = dataset_cache_metadata(self.dataset)
        for key, expected_value in expected.items():
            if key not in row.index:
                # Old cache format: do not trust it silently. Recompute instead.
                return False
            if str(row.get(key, "")) != str(expected_value):
                return False
        return True

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        df = pd.read_csv(self.cache_path)
        if "coalition_key" not in df.columns or "value_accuracy" not in df.columns:
            return
        n_loaded = 0
        n_skipped = 0
        for _, row in df.iterrows():
            if not self._cache_row_matches_current_run(row):
                n_skipped += 1
                continue
            self.cache[str(row["coalition_key"])] = float(row["value_accuracy"])
            n_loaded += 1
        if n_skipped:
            print(
                f"Cache {self.cache_path} contained {n_skipped} stale/incompatible rows; "
                f"loaded {n_loaded} matching rows."
            )

    def _append_cache_row(self, active_gates: Sequence[int], value: float) -> None:
        key = coalition_key(active_gates)
        total = sorted(set(active_gates) | set(self.partition.passive))
        row = {
            "r": self.partition.r,
            "coalition_key": key,
            "active_gate_indices": gate_list_string(active_gates),
            "included_gate_indices_total": gate_list_string(total),
            "n_active_gates": len(active_gates),
            "n_total_gates": len(total),
            "value_accuracy": float(value),
            "data_index": self.dataset.data_index,
            "convention": self.convention.name,
            "svc_c": self.svc_c,
            **dataset_cache_metadata(self.dataset),
        }
        exists = self.cache_path.exists()
        with self.cache_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.cache_fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def value_for_active_gates(self, active_gates_1_based: Iterable[int]) -> float:
        active = sorted(int(g) for g in active_gates_1_based if int(g) not in self.partition.passive)
        key = coalition_key(active)
        if key in self.cache:
            return float(self.cache[key])
        value = qsvm_accuracy(
            dataset=self.dataset,
            r=self.partition.r,
            active_gates=active,
            passive_gates=self.partition.passive,
            convention=self.convention,
            svc_c=self.svc_c,
        )
        self.cache[key] = float(value)
        self._append_cache_row(active, float(value))
        return float(value)

    def value_from_qowen_S(self, S: Sequence[int]) -> float:
        """Value callback for QuantumOwenValues. S contains 0-based indices plus locked gates."""
        active_1_based = sorted(
            idx + 1 for idx in S
            if (idx + 1) not in self.partition.passive
        )
        return self.value_for_active_gates(active_1_based)

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
            return [self.value_from_qowen_S(s) for s in S_list]
        if S is None:
            raise ValueError("QSVMAccuracyValueFunction requires S or S_list.")
        return self.value_from_qowen_S(S)

    def make_qowen_memory(self) -> Dict[Tuple[int, ...], List[List[Optional[float]]]]:
        """Convert persisted 1-based cache to qshaptools' exact-mode memory format."""
        memory: Dict[Tuple[int, ...], List[List[Optional[float]]]] = {}
        for key, value in self.cache.items():
            gates_1 = parse_coalition_key(key)
            gates_0 = tuple(sorted(g - 1 for g in gates_1))
            memory[gates_0] = [[None, float(value)]]
        return memory


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def exact_memory_to_rows(
    qowen_memory: Mapping[Tuple[int, ...], Sequence[Sequence[Any]]],
    exact_coalitions_0_based: Iterable[Tuple[int, ...]],
    partition: QSVMPartition,
    data_index: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for coalition_0 in sorted(set(tuple(c) for c in exact_coalitions_0_based)):
        values = qowen_memory.get(coalition_0)
        if not values:
            raise KeyError(f"Missing qowen memory for coalition {coalition_0}")
        value = float(np.mean([float(v[1]) for v in values]))
        active = sorted(idx + 1 for idx in coalition_0)
        total = sorted(set(active) | set(partition.passive))
        rows.append(
            {
                "r": partition.r,
                "coalition_key": coalition_key(active),
                "active_gate_indices": gate_list_string(active),
                "included_gate_indices_total": gate_list_string(total),
                "n_active_gates": len(active),
                "n_total_gates": len(total),
                "value_accuracy": value,
                "data_index": data_index,
            }
        )
    return rows


def make_gate_owen_dataframe(
    phi_1_based: Mapping[int, float],
    partition: QSVMPartition,
    full_accuracy: float,
    empty_active_accuracy: float,
    data_index: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for meta in qsvm_gate_metadata(partition.r):
        gate = int(meta["gate_index_1based"])
        group = str(meta["group"])
        value = None if group == "passive" else float(phi_1_based[gate])
        rows.append(
            {
                **meta,
                "owen_value": value,
                "full_accuracy": float(full_accuracy),
                "empty_active_accuracy": float(empty_active_accuracy),
                "data_index": int(data_index),
            }
        )
    return pd.DataFrame(rows)


def make_group_owen_dataframe(
    phi_1_based: Mapping[int, float],
    partition: QSVMPartition,
    full_accuracy: float,
    empty_active_accuracy: float,
    data_index: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label in GROUP_LABELS:
        gates = partition.groups[label]
        rows.append(
            {
                "r": partition.r,
                "group": label,
                "n_gates": len(gates),
                "gate_indices": gate_list_string(gates),
                "group_owen_value": float(sum(phi_1_based[g] for g in gates)),
                "full_accuracy": float(full_accuracy),
                "empty_active_accuracy": float(empty_active_accuracy),
                "data_index": int(data_index),
            }
        )
    return pd.DataFrame(rows)


def append_distribution_row(path: Path, row: Mapping[str, Any]) -> None:
    fields = [
        "r",
        "subset_key",
        "active_gate_indices",
        "included_gate_indices_total",
        "n_active_gates",
        "n_total_gates",
        "value_accuracy",
        "data_index",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def compute_all_subset_distribution(
    evaluator: QSVMAccuracyValueFunction,
    output_path: Path,
    force: bool = False,
    progress_every: int = 250,
) -> pd.DataFrame:
    """Evaluate v(S) for every active-gate subset, with resume from CSV."""
    partition = evaluator.partition
    ensure_dir(output_path.parent)
    if force and output_path.exists():
        output_path.unlink()

    done: Set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if "subset_key" in existing.columns:
            done = {str(k) for k in existing["subset_key"]}

    active = partition.active_gates
    total_count = 2 ** len(active)
    n_done_start = len(done)
    n_new = 0
    t0 = time.time()
    print(
        f"All-subset value distribution for r={partition.r}: "
        f"{total_count} subsets, {n_done_start} already present."
    )

    for k in range(len(active) + 1):
        for subset in itertools.combinations(active, k):
            key = coalition_key(subset)
            if key in done:
                continue
            value = evaluator.value_for_active_gates(subset)
            total = sorted(set(subset) | set(partition.passive))
            row = {
                "r": partition.r,
                "subset_key": key,
                "active_gate_indices": gate_list_string(subset),
                "included_gate_indices_total": gate_list_string(total),
                "n_active_gates": len(subset),
                "n_total_gates": len(total),
                "value_accuracy": float(value),
                "data_index": evaluator.dataset.data_index,
            }
            append_distribution_row(output_path, row)
            done.add(key)
            n_new += 1
            if progress_every and n_new % progress_every == 0:
                elapsed = time.time() - t0
                print(f"  evaluated {len(done)}/{total_count} subsets ({elapsed:.1f}s)")

    df = pd.read_csv(output_path)
    df = df.drop_duplicates(subset=["subset_key"], keep="last")
    df = df.sort_values(["n_active_gates", "subset_key"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_figure(fig: Any, png_path: Path, pdf_path: Optional[Path] = None) -> None:
    ensure_dir(png_path.parent)
    fig.tight_layout(pad=1.2)
    fig.savefig(png_path, dpi=220)
    if pdf_path is not None:
        fig.savefig(pdf_path)
    plt.close(fig)


def plot_qsvm_data(dataset: QSVMDataset, out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    labels = sorted(set(dataset.y_train.tolist()) | set(dataset.y_test.tolist()), key=str)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 2)))
    color_map = {label: colors[i] for i, label in enumerate(labels)}

    for label in labels:
        train_mask = dataset.y_train == label
        test_mask = dataset.y_test == label
        ax.scatter(
            dataset.X_test[test_mask, 0],
            dataset.X_test[test_mask, 1],
            s=14,
            color=color_map[label],
            alpha=0.25,
            marker="o",
            linewidths=0,
            label=f"test y={label}",
        )
        ax.scatter(
            dataset.X_train[train_mask, 0],
            dataset.X_train[train_mask, 1],
            s=42,
            color=color_map[label],
            alpha=0.95,
            marker="^",
            edgecolors="black",
            linewidths=0.45,
            label=f"train y={label}",
        )

    all_x = np.vstack([dataset.X_train, dataset.X_test])
    if np.min(all_x) >= -1e-9 and np.max(all_x) <= 2 * math.pi + 1e-9:
        ax.set_xlim(0, 2 * math.pi)
        ax.set_ylim(0, 2 * math.pi)
        ticks = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        ticklabels = ["0", "pi/2", "pi", "3pi/2", "2pi"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"QSVM dataset {dataset.data_index}: train and test points")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=True)
    save_figure(fig, out_png, out_pdf)


def _add_top_gate_axis(ax: Any, gates: Sequence[int], names: Sequence[str], fontsize: int = 9) -> Any:
    """Add QNN Figure-13-style gate names on a top x-axis."""
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(list(gates))
    ax_top.set_xticklabels(list(names), fontsize=fontsize)
    return ax_top


def _set_qnn_style_spines(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _force_times_1e_minus_2_clean(ax: Any) -> None:
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(
        -0.065,
        1.015,
        r"$\times 10^{-2}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=plt.rcParams.get("ytick.labelsize", 11),
        clip_on=False,
    )


def plot_gate_owen_values(gate_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    r = int(gate_df["r"].iloc[0])
    full_accuracy = float(gate_df["full_accuracy"].iloc[0])
    xs = gate_df["gate_index_1based"].to_numpy(dtype=int)
    active_df = gate_df[gate_df["group"] != "passive"].copy()
    passive_df = gate_df[gate_df["group"] == "passive"].copy()

    active_xs = active_df["gate_index_1based"].to_numpy(dtype=int)
    active_values = active_df["owen_value"].to_numpy(dtype=float)
    passive_xs = passive_df["gate_index_1based"].to_numpy(dtype=int)

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(max(6.4, 0.42 * len(xs) + 3.5), 3.7))
        style = PAPER_SERIES_STYLE.get(r, PAPER_SERIES_STYLE[1])
        ax.scatter(
            active_xs,
            active_values,
            marker=style["marker"],
            s=34,
            color=style["color"],
            label=rf"$r={r}$, exact Owen",
        )
        if len(passive_xs) > 0:
            ax.scatter(
                passive_xs,
                np.zeros_like(passive_xs, dtype=float),
                marker="x",
                s=32,
                color="0.45",
                linewidths=1.0,
                label="passive H",
                zorder=3,
            )
        ax.axhline(0.0, linewidth=0.7, color="0.6")
        ax.grid(True, color="0.75", linewidth=0.55)
        ax.set_xlabel(r"$g$")
        ax.set_ylabel(r"$\Phi_{\mathrm{Owen}}^{(g)}$")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs])
        ax.set_xlim(0.5, 7 * r + 0.5)
        _force_times_1e_minus_2_clean(ax)
        ax.legend(title=f"full acc. = {full_accuracy:.3f}", loc="best", frameon=True, fancybox=False)
        _add_top_gate_axis(ax, xs, gate_df["gate_name"].tolist(), fontsize=12)
        _set_qnn_style_spines(ax)
        save_figure(fig, out_png, out_pdf)


def plot_group_owen_values(group_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    r = int(group_df["r"].iloc[0])
    labels = [
        f"{row.group} ({int(row.n_gates)})"
        for row in group_df.itertuples(index=False)
    ]
    values = group_df["group_owen_value"].to_numpy(dtype=float)
    colors = [GROUP_COLORS[str(g)] for g in group_df["group"]]

    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.5")
        ax.set_ylabel(r"$\sum_{g \in G}\Phi_{\mathrm{Owen}}^{(g)}$")
        ax.set_title(f"Group-level exact Owen values for QSVM, r={r}")
        ax.grid(axis="y", color="0.82", linewidth=0.55)
        _set_qnn_style_spines(ax)
        save_figure(fig, out_png, out_pdf)


def plot_value_distribution(dist_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    if dist_df.empty:
        return
    r = int(dist_df["r"].iloc[0])
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        _plot_value_distribution_axis(
            ax=ax,
            dist_df=dist_df,
            r=r,
            panel_label=None,
            show_ylabel=True,
            show_passive_note=True,
        )
        save_figure(fig, out_png, out_pdf)


def _plot_value_distribution_axis(
    ax: Any,
    dist_df: pd.DataFrame,
    r: int,
    panel_label: Optional[str],
    show_ylabel: bool,
    show_passive_note: bool,
) -> None:
    grouped = [(int(k), g["value_accuracy"].to_numpy(dtype=float)) for k, g in dist_df.groupby("n_total_gates")]
    grouped.sort(key=lambda item: item[0], reverse=True)
    k_values = [k for k, _ in grouped]
    values = [vals for _, vals in grouped]
    maxima = [float(np.max(vals)) for vals in values]
    total_gates = 7 * r
    positions = np.array([total_gates - k for k in k_values], dtype=float)
    all_tick_positions = np.arange(total_gates + 1)
    all_tick_labels = [str(k) for k in range(total_gates, -1, -1)]

    color = PAPER_SERIES_STYLE.get(r, PAPER_SERIES_STYLE[1])["color"]
    ax.boxplot(
        values,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": color, "alpha": 0.18, "edgecolor": color, "linewidth": 0.7},
        medianprops={"color": color, "linewidth": 1.0},
        whiskerprops={"color": color, "linewidth": 0.7},
        capprops={"color": color, "linewidth": 0.7},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": color,
            "markersize": 3.0,
            "alpha": 0.8,
        },
    )
    ax.plot(positions, maxima, linestyle=(0, (1, 2.5)), color="black", linewidth=1.1)
    ax.set_xlim(-0.5, total_gates + 0.5)
    ax.set_xticks(all_tick_positions)
    ax.set_xticklabels(all_tick_labels)
    ax.set_xlabel(r"number of gates $k$")
    if show_ylabel:
        ax.set_ylabel(r"$\mathcal{W}_k$")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, color="0.75", linewidth=0.55)
    if show_passive_note:
        passive = len(make_qsvm_partition(r).passive)
        ax.set_title(f"r={r}, H passive ({passive} gates always included)", fontsize=12)
    if panel_label is not None:
        ax.text(
            0.5,
            -0.34,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
    _set_qnn_style_spines(ax)


# ---------------------------------------------------------------------------
# Aggregation over the five QSVM datasets
# ---------------------------------------------------------------------------


def aggregate_qsvm_dataset_runs(output_dir: Path) -> Dict[str, Path]:
    """Aggregate per-dataset QSVM Owen outputs into mean/std CSVs and plots.

    Expected layout:
        output_dir/dataset_0/r1/gate_owen_values.csv
        output_dir/dataset_0/r1/group_owen_values.csv
        ...
    """
    output_dir = Path(output_dir)
    aggregate_dir = ensure_dir(output_dir / "aggregate")

    gate_paths = sorted(output_dir.glob("dataset_*/r*/gate_owen_values.csv"))
    group_paths = sorted(output_dir.glob("dataset_*/r*/group_owen_values.csv"))
    if not gate_paths:
        raise FileNotFoundError(f"No dataset_*/r*/gate_owen_values.csv files found under {output_dir}")
    if not group_paths:
        raise FileNotFoundError(f"No dataset_*/r*/group_owen_values.csv files found under {output_dir}")

    gate_all = pd.concat([pd.read_csv(path) for path in gate_paths], ignore_index=True)
    group_all = pd.concat([pd.read_csv(path) for path in group_paths], ignore_index=True)
    gate_all_path = aggregate_dir / "gate_owen_values_all_datasets.csv"
    group_all_path = aggregate_dir / "group_owen_values_all_datasets.csv"
    gate_all.to_csv(gate_all_path, index=False)
    group_all.to_csv(group_all_path, index=False)

    active_gate_all = gate_all[gate_all["group"] != "passive"].copy()
    gate_summary = (
        active_gate_all
        .groupby(["r", "gate_index_1based", "gate_name", "group", "gate_subtype"], dropna=False)
        .agg(
            owen_mean=("owen_value", "mean"),
            owen_std=("owen_value", "std"),
            n_datasets=("data_index", "nunique"),
            full_accuracy_mean=("full_accuracy", "mean"),
            full_accuracy_std=("full_accuracy", "std"),
            empty_active_accuracy_mean=("empty_active_accuracy", "mean"),
            empty_active_accuracy_std=("empty_active_accuracy", "std"),
        )
        .reset_index()
        .sort_values(["r", "gate_index_1based"])
    )
    gate_summary_path = aggregate_dir / "gate_owen_values_mean_std.csv"
    gate_summary.to_csv(gate_summary_path, index=False)

    group_summary = (
        group_all
        .groupby(["r", "group", "n_gates", "gate_indices"], dropna=False)
        .agg(
            group_owen_mean=("group_owen_value", "mean"),
            group_owen_std=("group_owen_value", "std"),
            n_datasets=("data_index", "nunique"),
            full_accuracy_mean=("full_accuracy", "mean"),
            full_accuracy_std=("full_accuracy", "std"),
            empty_active_accuracy_mean=("empty_active_accuracy", "mean"),
            empty_active_accuracy_std=("empty_active_accuracy", "std"),
        )
        .reset_index()
        .sort_values(["r", "group"])
    )
    group_summary_path = aggregate_dir / "group_owen_values_mean_std.csv"
    group_summary.to_csv(group_summary_path, index=False)

    accuracy_summary = (
        gate_all[["data_index", "r", "full_accuracy", "empty_active_accuracy"]]
        .drop_duplicates()
        .groupby("r")
        .agg(
            full_accuracy_mean=("full_accuracy", "mean"),
            full_accuracy_std=("full_accuracy", "std"),
            empty_active_accuracy_mean=("empty_active_accuracy", "mean"),
            empty_active_accuracy_std=("empty_active_accuracy", "std"),
            n_datasets=("data_index", "nunique"),
        )
        .reset_index()
        .sort_values("r")
    )
    accuracy_summary_path = aggregate_dir / "accuracy_mean_std.csv"
    accuracy_summary.to_csv(accuracy_summary_path, index=False)

    gate_plot_png, gate_plot_pdf = plot_aggregate_gate_owen_values(gate_summary, aggregate_dir)
    group_plot_png, group_plot_pdf = plot_aggregate_group_owen_values(group_summary, aggregate_dir)

    return {
        "gate_all": gate_all_path,
        "group_all": group_all_path,
        "gate_summary": gate_summary_path,
        "group_summary": group_summary_path,
        "accuracy_summary": accuracy_summary_path,
        "gate_plot_png": gate_plot_png,
        "gate_plot_pdf": gate_plot_pdf,
        "group_plot_png": group_plot_png,
        "group_plot_pdf": group_plot_pdf,
    }


def plot_aggregate_gate_owen_values(gate_summary: pd.DataFrame, aggregate_dir: Path) -> Tuple[Path, Path]:
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(11.8, 4.2))
        for r in sorted(gate_summary["r"].unique()):
            df = gate_summary[gate_summary["r"] == r].copy()
            style = PAPER_SERIES_STYLE.get(int(r), PAPER_SERIES_STYLE[1])
            ax.errorbar(
                df["gate_index_1based"].to_numpy(dtype=int),
                df["owen_mean"].to_numpy(dtype=float),
                yerr=df["owen_std"].fillna(0.0).to_numpy(dtype=float),
                fmt=style["marker"],
                color=style["color"],
                markersize=5,
                elinewidth=0.9,
                capsize=2.5,
                linestyle="none",
                label=rf"$r={int(r)}$, mean $\pm$ std",
                zorder=3,
            )
        passive_h = [g for g in range(1, 22) if qsvm_gate_name(g) == "H"]
        ax.scatter(passive_h, np.zeros(len(passive_h)), marker="x", s=28, color="0.45", label="passive H", zorder=3)
        ax.axhline(0.0, linewidth=0.7, color="0.6")
        ax.grid(True, color="0.75", linewidth=0.55)
        ax.set_xlim(0.5, 21.5)
        ax.set_xticks(list(range(1, 22)))
        ax.set_xticklabels([str(i) for i in range(1, 22)])
        ax.set_xlabel(r"$g$")
        ax.set_ylabel(r"Mean Owen value over datasets")
        _force_times_1e_minus_2_clean(ax)
        ax.legend(loc="best", frameon=True, fancybox=False)
        _add_top_gate_axis(ax, list(range(1, 22)), [qsvm_gate_name(i) for i in range(1, 22)], fontsize=12)
        _set_qnn_style_spines(ax)
        png = aggregate_dir / "fig_qsvm_gate_owen_mean_std_over_datasets.png"
        pdf = aggregate_dir / "fig_qsvm_gate_owen_mean_std_over_datasets.pdf"
        save_figure(fig, png, pdf)
        return png, pdf


def plot_aggregate_group_owen_values(group_summary: pd.DataFrame, aggregate_dir: Path) -> Tuple[Path, Path]:
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        labels: List[str] = []
        values: List[float] = []
        errors: List[float] = []
        colors: List[str] = []
        for r in sorted(group_summary["r"].unique()):
            df = group_summary[group_summary["r"] == r].copy()
            for label in GROUP_LABELS:
                row = df[df["group"] == label].iloc[0]
                labels.append(f"r={int(r)}\n{label}")
                values.append(float(row["group_owen_mean"]))
                errors.append(0.0 if pd.isna(row["group_owen_std"]) else float(row["group_owen_std"]))
                colors.append(GROUP_COLORS[label])
        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.5")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"Mean group Owen total over datasets")
        ax.grid(axis="y", color="0.82", linewidth=0.55)
        _set_qnn_style_spines(ax)
        png = aggregate_dir / "fig_qsvm_group_owen_mean_std_over_datasets.png"
        pdf = aggregate_dir / "fig_qsvm_group_owen_mean_std_over_datasets.pdf"
        save_figure(fig, png, pdf)
        return png, pdf

# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def build_dummy_full_qsvm_circuit(r: int) -> Any:
    """Build a concrete full circuit solely so QuantumOwenValues can inspect gates."""
    return build_qsvm_feature_map([0.123, 0.456], r=r, active_gates=None, passive_gates=None)


def run_replication(args: argparse.Namespace) -> pd.DataFrame:
    repo_root = infer_repo_root()
    import_qiskit_and_qowen(repo_root)
    out_dir = ensure_dir(Path(args.output_dir) / "replication")

    data_index = int(args.data_index)
    convention = DEFAULT_CONVENTION
    dataset = load_qsvm_dataset(
        data_index=data_index,
        data_dir=Path(args.data_dir),
        train_path=args.train_path,
        test_path=args.test_path,
    )
    print_dataset_report(dataset)

    data_png = out_dir / f"fig_qsvm_data_dataset{data_index}.png"
    data_pdf = out_dir / f"fig_qsvm_data_dataset{data_index}.pdf"
    plot_qsvm_data(dataset, data_png, data_pdf)
    print(f"Saved data scatter plot to {data_png} and {data_pdf}")

    rows: List[Dict[str, Any]] = []
    for r in [int(x) for x in args.r_values]:
        partition = make_qsvm_partition(r)
        active = partition.active_gates
        acc = qsvm_accuracy(
            dataset=dataset,
            r=r,
            active_gates=active,
            passive_gates=partition.passive,
            convention=convention,
            svc_c=float(args.svc_c),
        )
        target = TARGET_ACCURACIES.get(r)
        diff = None if target is None else abs(acc - target)
        rows.append(
            {
                "data_index": data_index,
                "r": r,
                "full_accuracy": acc,
                "target_accuracy": target,
                "absolute_difference": diff,
                "train_path": str(dataset.train_path),
                "test_path": str(dataset.test_path),
                "convention": convention.name,
                "svc_c": float(args.svc_c),
            }
        )
        if target is None:
            print(f"r={r}: full accuracy = {acc:.4f}")
        else:
            print(f"r={r}: full accuracy = {acc:.4f}; target ~= {target:.4f}; abs diff = {diff:.4f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "qsvm_full_accuracies.csv"
    json_path = out_dir / "qsvm_full_accuracies.json"
    df.to_csv(csv_path, index=False)
    save_json(json_path, {"results": rows})
    print(f"Saved replication CSV to {csv_path}")
    print(f"Saved replication JSON to {json_path}")

    large_diffs = [row for row in rows if row["absolute_difference"] is not None and row["absolute_difference"] > 0.05]
    if large_diffs:
        print("One or more accuracies differ from the paper targets by more than 0.05.")
        print("Diagnostics to check:")
        print("  - CNOT direction: default is control 0, target 1.")
        print("  - Cross-phase qubit: default is the CNOT target.")
        print("  - Phi convention: local phi_i=2*x_i and cross phi=2*(pi-x1)*(pi-x2).")
        print("  - CSV labels and feature column order.")
        print("  - SVC parameters: default SVC(kernel='precomputed', C=1.0).")
        if not bool(args.search_conventions):
            print("  - Re-run with --search-conventions true for a small convention sweep.")

    if bool(args.search_conventions):
        run_convention_search(dataset, args, out_dir)

    return df


def run_convention_search(dataset: QSVMDataset, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    """Try common feature-map convention alternatives and rank by target error."""
    conventions: List[QSVMConvention] = []
    for control, target in [(0, 1), (1, 0)]:
        for cross_on in ["target", "control"]:
            for feature_order in ["normal", "swapped"]:
                conventions.append(
                    QSVMConvention(
                        cnot_control=control,
                        cnot_target=target,
                        cross_phase_on=cross_on,
                        feature_order=feature_order,
                        simulator=DEFAULT_CONVENTION.simulator,
                    )
                )

    rows: List[Dict[str, Any]] = []
    print("Convention search:")
    for convention in conventions:
        total_error = 0.0
        result: Dict[str, Any] = {"convention": convention.name}
        for r in [int(x) for x in args.r_values]:
            partition = make_qsvm_partition(r)
            acc = qsvm_accuracy(
                dataset=dataset,
                r=r,
                active_gates=partition.active_gates,
                passive_gates=partition.passive,
                convention=convention,
                svc_c=float(args.svc_c),
            )
            target_acc = TARGET_ACCURACIES.get(r)
            diff = abs(acc - target_acc) if target_acc is not None else np.nan
            if target_acc is not None:
                total_error += diff
            result[f"accuracy_r{r}"] = acc
            result[f"abs_diff_r{r}"] = diff
        result["sum_abs_diff"] = total_error
        rows.append(result)
        print(f"  {convention.name}: sum abs diff={total_error:.4f}")

    df = pd.DataFrame(rows).sort_values("sum_abs_diff")
    path = out_dir / "qsvm_convention_search.csv"
    df.to_csv(path, index=False)
    print(f"Saved convention search to {path}")
    print(df.head(8).to_string(index=False))
    return df


def run_exact_owen(args: argparse.Namespace, r: int) -> Dict[str, Any]:
    repo_root = infer_repo_root()
    _, _, QuantumOwenValues = import_qiskit_and_qowen(repo_root)
    out_dir = ensure_dir(Path(args.output_dir) / f"r{r}")
    t0 = time.time()

    dataset = load_qsvm_dataset(
        data_index=int(args.data_index),
        data_dir=Path(args.data_dir),
        train_path=args.train_path,
        test_path=args.test_path,
    )
    print_dataset_report(dataset)
    partition = make_qsvm_partition(r)
    print_partition_report(partition)

    simulator_validation = None
    if bool(getattr(args, "validate_simulator", True)):
        simulator_validation = validate_manual_simulator_random_subsets(r=r, num_points=8, num_random_subsets=10)
        print(f"Manual-vs-Qiskit simulator validation: {simulator_validation}")
        if float(simulator_validation["max_abs_difference"]) > 1e-9:
            raise ValueError(
                "Manual simulator does not match Qiskit on random QSVM coalitions. "
                f"Report: {simulator_validation}"
            )

    if int(args.n_jobs) != 1:
        print("Note: --n-jobs is accepted for CLI compatibility; exact qowen evaluation runs sequentially.")

    cache_path = out_dir / "qsvm_value_cache.csv"
    evaluator = QSVMAccuracyValueFunction(
        dataset=dataset,
        partition=partition,
        cache_path=cache_path,
        convention=DEFAULT_CONVENTION,
        svc_c=float(args.svc_c),
        force=bool(args.force),
    )

    full_accuracy = evaluator.value_for_active_gates(partition.active_gates)
    empty_active_accuracy = evaluator.value_for_active_gates([])
    print(f"Full active accuracy: {full_accuracy:.4f}")
    print(f"Empty-active H-only accuracy: {empty_active_accuracy:.4f}")

    qc = build_dummy_full_qsvm_circuit(r)
    qowen_partition = [partition.groups_0_based[label] for label in GROUP_LABELS]
    memory = evaluator.make_qowen_memory()
    qov = QuantumOwenValues(
        qc=qc,
        partition=qowen_partition,
        value_fun=evaluator,
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=partition.passive_0_based,
        owen_sample_frac=None,
        owen_sample_reps=1,
        evaluate_value_only_once=True,
        sample_in_memory=True,
        owen_sample_seed=123,
        owen_batch_size=None,
        memory=memory,
        name=f"qsvm_r{r}_exact_owen",
        silent=bool(args.silent),
    )
    phi_0_based = qov.run()
    phi_1_based = {int(k) + 1: float(v) for k, v in phi_0_based.items()}

    # Owen efficiency check over active players.
    total_phi = float(sum(phi_1_based.values()))
    efficiency_gap = total_phi - (full_accuracy - empty_active_accuracy)
    print(f"Sum Owen values: {total_phi:.8f}; full-empty: {full_accuracy - empty_active_accuracy:.8f}")
    print(f"Efficiency gap: {efficiency_gap:.3e}")

    exact_coalitions = getattr(qov, "_all_coalitions", set())
    exact_rows = exact_memory_to_rows(
        qowen_memory=qov.memory,
        exact_coalitions_0_based=exact_coalitions,
        partition=partition,
        data_index=dataset.data_index,
    )
    exact_df = pd.DataFrame(exact_rows)
    exact_path = out_dir / "coalition_values_exact_owen.csv"
    exact_df.to_csv(exact_path, index=False)

    gate_df = make_gate_owen_dataframe(
        phi_1_based=phi_1_based,
        partition=partition,
        full_accuracy=full_accuracy,
        empty_active_accuracy=empty_active_accuracy,
        data_index=dataset.data_index,
    )
    gate_path = out_dir / "gate_owen_values.csv"
    gate_df.to_csv(gate_path, index=False)

    group_df = make_group_owen_dataframe(
        phi_1_based=phi_1_based,
        partition=partition,
        full_accuracy=full_accuracy,
        empty_active_accuracy=empty_active_accuracy,
        data_index=dataset.data_index,
    )
    group_path = out_dir / "group_owen_values.csv"
    group_df.to_csv(group_path, index=False)

    plot_gate_owen_values(
        gate_df,
        out_dir / f"fig_qsvm_owen_gate_values_r{r}.png",
        out_dir / f"fig_qsvm_owen_gate_values_r{r}.pdf",
    )
    plot_group_owen_values(
        group_df,
        out_dir / f"fig_qsvm_owen_group_values_r{r}.png",
        out_dir / f"fig_qsvm_owen_group_values_r{r}.pdf",
    )

    dist_evaluations = 0
    dist_path = out_dir / "all_subset_value_distribution.csv"
    if bool(args.make_value_distribution):
        dist_df = compute_all_subset_distribution(evaluator, dist_path, force=bool(args.force))
        dist_evaluations = int(len(dist_df))
        plot_value_distribution(
            dist_df,
            out_dir / f"fig_qsvm_value_distribution_r{r}.png",
            out_dir / f"fig_qsvm_value_distribution_r{r}.pdf",
        )
    elif dist_path.exists():
        dist_df = pd.read_csv(dist_path)
        dist_evaluations = int(len(dist_df))
        plot_value_distribution(
            dist_df,
            out_dir / f"fig_qsvm_value_distribution_r{r}.png",
            out_dir / f"fig_qsvm_value_distribution_r{r}.pdf",
        )

    target = TARGET_ACCURACIES.get(r)
    runtime = time.time() - t0
    summary = {
        "r": r,
        "train_path": str(dataset.train_path),
        "test_path": str(dataset.test_path),
        "data_index": dataset.data_index,
        "train_shape": list(dataset.X_train.shape),
        "test_shape": list(dataset.X_test.shape),
        "full_accuracy": full_accuracy,
        "empty_active_accuracy": empty_active_accuracy,
        "partition": partition.groups,
        "passive_gates": partition.passive,
        "number_of_exact_owen_value_evaluations": len(exact_df),
        "number_of_all_subset_distribution_evaluations": dist_evaluations,
        "runtime_seconds": runtime,
        "target_accuracy": target,
        "obtained_accuracy": full_accuracy,
        "absolute_difference_from_target": None if target is None else abs(full_accuracy - target),
        "efficiency_gap": efficiency_gap,
        "convention": DEFAULT_CONVENTION.name,
        "svc_c": float(args.svc_c),
        "cache_path": str(cache_path),
        "simulator_validation": simulator_validation,
    }
    summary_path = out_dir / "summary.json"
    save_json(summary_path, summary)

    print(f"Saved gate Owen CSV to {gate_path}")
    print(f"Saved group Owen CSV to {group_path}")
    print(f"Saved exact coalition CSV to {exact_path}")
    if bool(args.make_value_distribution):
        print(f"Saved all-subset distribution CSV to {dist_path}")
    print(f"Saved summary JSON to {summary_path}")
    print(f"Runtime: {runtime:.1f}s")
    return summary


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "qsvm_owen_all_datasets")
    parser.add_argument("--svc-c", type=float, default=1.0)
