#!/usr/bin/env python3
"""QSVM feature-map helpers for the Heese Shapley reproduction."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


QSVM_EXPECTED_ACCURACY = {1: 0.842, 2: 0.986, 3: 0.913}


@dataclass(frozen=True)
class QSVMDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_path: Path
    test_path: Path
    data_index: int


def qsvm_gate_name(gate_index: int) -> str:
    return ["H", "P", "H", "P", "CX", "P", "CX"][(int(gate_index) - 1) % 7]


def qsvm_passive_gates(r: int) -> list[int]:
    """Hadamard gates locked passive in the thesis QSVM E/M/X Owen game."""

    return sorted([1 + 7 * rep for rep in range(int(r))] + [3 + 7 * rep for rep in range(int(r))])


def qsvm_active_gates(r: int) -> list[int]:
    passive = set(qsvm_passive_gates(r))
    return [g for g in range(1, 7 * int(r) + 1) if g not in passive]


def load_qsvm_dataset(data_dir: Path, data_index: int = 0) -> QSVMDataset:
    train_path = Path(data_dir) / f"qsvm-traindata-{int(data_index)}.csv"
    test_path = Path(data_dir) / f"qsvm-testdata-{int(data_index)}.csv"
    X_train, y_train = _load_xy(train_path)
    X_test, y_test = _load_xy(test_path)
    return QSVMDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_path=train_path,
        test_path=test_path,
        data_index=int(data_index),
    )


def _load_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append((float(row["x0"]), float(row["x1"]), int(float(row["y"]))))
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return (
        np.asarray([[r[0], r[1]] for r in rows], dtype=float),
        np.asarray([r[2] for r in rows], dtype=int),
    )


def _apply_h(states: np.ndarray, qubit: int) -> None:
    inv = 1.0 / math.sqrt(2.0)
    if qubit == 0:
        pairs = [(0, 1), (2, 3)]
    elif qubit == 1:
        pairs = [(0, 2), (1, 3)]
    else:
        raise ValueError(f"invalid qubit {qubit}")
    for a_idx, b_idx in pairs:
        a = states[:, a_idx].copy()
        b = states[:, b_idx].copy()
        states[:, a_idx] = (a + b) * inv
        states[:, b_idx] = (a - b) * inv


def _apply_phase(states: np.ndarray, qubit: int, angles: np.ndarray) -> None:
    phase = np.exp(1j * angles)
    if qubit == 0:
        states[:, 1] *= phase
        states[:, 3] *= phase
    elif qubit == 1:
        states[:, 2] *= phase
        states[:, 3] *= phase
    else:
        raise ValueError(f"invalid qubit {qubit}")


def _apply_cx(states: np.ndarray, control: int = 0, target: int = 1) -> None:
    if control == 0 and target == 1:
        states[:, [1, 3]] = states[:, [3, 1]]
    elif control == 1 and target == 0:
        states[:, [2, 3]] = states[:, [3, 2]]
    else:
        raise ValueError(f"unsupported two-qubit CX {control}->{target}")


def compute_qsvm_statevectors(
    X: np.ndarray,
    r: int,
    active_gates: Optional[Iterable[int]],
) -> np.ndarray:
    """Compute exact two-qubit statevectors for the Heese feature map.

    Heese's Shapley experiment treats every gate as active; for a coalition
    value v(S), exactly the gates in S are included.
    """

    X = np.asarray(X, dtype=float)
    states = np.zeros((X.shape[0], 4), dtype=np.complex128)
    states[:, 0] = 1.0

    active = None if active_gates is None else set(int(g) for g in active_gates)

    def include(g: int) -> bool:
        return active is None or g in active

    x0 = X[:, 0]
    x1 = X[:, 1]
    phi_1 = 2.0 * x0
    phi_2 = 2.0 * x1
    phi_cross = 2.0 * (math.pi - x0) * (math.pi - x1)

    for rep in range(int(r)):
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
            _apply_cx(states, 0, 1)
        if include(offset + 6):
            _apply_phase(states, 1, phi_cross)
        if include(offset + 7):
            _apply_cx(states, 0, 1)

    return states


def compute_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    overlaps = A @ B.conj().T
    kernel = np.abs(overlaps) ** 2
    return np.clip(np.real_if_close(kernel), 0.0, 1.0).astype(float)


def qsvm_accuracy(
    dataset: QSVMDataset,
    r: int,
    active_gates: Iterable[int],
    *,
    passive_gates: Iterable[int] = (),
    svc_c: float = 1.0,
) -> float:
    from sklearn.svm import SVC

    included = set(int(g) for g in active_gates) | set(int(g) for g in passive_gates)
    train_states = compute_qsvm_statevectors(dataset.X_train, int(r), included)
    test_states = compute_qsvm_statevectors(dataset.X_test, int(r), included)
    K_train = compute_kernel(train_states, train_states)
    K_test = compute_kernel(test_states, train_states)
    clf = SVC(kernel="precomputed", C=float(svc_c))
    clf.fit(K_train, dataset.y_train)
    pred = clf.predict(K_test)
    return float(np.mean(pred == dataset.y_test))
