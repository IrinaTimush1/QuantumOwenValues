#!/usr/bin/env python3
"""QNN circuit and value-function helpers for the Shapley reproduction."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import FrozenSet, Iterable, List, Sequence, Tuple

import numpy as np


PAPER_THETA = np.array([3.860, -1.070, -1.583, 0.860], dtype=float)

# Locked-passive Shapley baseline matching the thesis QNN E/M/X Owen game.
QNN_PASSIVE_GATES = [1, 3, 8, 10]
QNN_ACTIVE_GATES = [2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
QNN_GATE_LABELS = {
    1: "H",
    2: "P",
    3: "H",
    4: "P",
    5: "CX",
    6: "P",
    7: "CX",
    8: "H",
    9: "P",
    10: "H",
    11: "P",
    12: "CX",
    13: "P",
    14: "CX",
    15: "RY",
    16: "RY",
    17: "CX",
    18: "RY",
    19: "RY",
}


def load_qnn_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[Tuple[float, float, int]] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append((float(row["x0"]), float(row["x1"]), int(float(row["y"]))))
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    X = np.asarray([[r[0], r[1]] for r in rows], dtype=float)
    y = np.asarray([r[2] for r in rows], dtype=int)
    return X, y


def _apply_h(state: np.ndarray, qubit: int) -> None:
    inv = 1.0 / math.sqrt(2.0)
    if qubit == 0:
        pairs = [(0, 1), (2, 3)]
    elif qubit == 1:
        pairs = [(0, 2), (1, 3)]
    else:
        raise ValueError(f"invalid qubit {qubit}")
    for a_idx, b_idx in pairs:
        a = state[a_idx]
        b = state[b_idx]
        state[a_idx] = (a + b) * inv
        state[b_idx] = (a - b) * inv


def _apply_phase(state: np.ndarray, qubit: int, angle: float) -> None:
    phase = np.exp(1j * float(angle))
    if qubit == 0:
        state[1] *= phase
        state[3] *= phase
    elif qubit == 1:
        state[2] *= phase
        state[3] *= phase
    else:
        raise ValueError(f"invalid qubit {qubit}")


def _apply_ry(state: np.ndarray, qubit: int, angle: float) -> None:
    c = math.cos(float(angle) / 2.0)
    s = math.sin(float(angle) / 2.0)
    if qubit == 0:
        pairs = [(0, 1), (2, 3)]
    elif qubit == 1:
        pairs = [(0, 2), (1, 3)]
    else:
        raise ValueError(f"invalid qubit {qubit}")
    for a_idx, b_idx in pairs:
        a = state[a_idx]
        b = state[b_idx]
        state[a_idx] = c * a - s * b
        state[b_idx] = s * a + c * b


def _apply_cx(state: np.ndarray, control: int, target: int) -> None:
    if control == 0 and target == 1:
        state[[1, 3]] = state[[3, 1]]
    elif control == 1 and target == 0:
        state[[2, 3]] = state[[3, 2]]
    else:
        raise ValueError(f"unsupported two-qubit CX {control}->{target}")


def feature_values(x0: float, x1: float) -> Tuple[float, float, float]:
    phi_1 = 2.0 * float(x0)
    phi_2 = 2.0 * float(x1)
    phi_cross = 2.0 * (math.pi - float(x0)) * (math.pi - float(x1))
    return phi_1, phi_2, phi_cross


def qnn_state(
    x: Sequence[float],
    theta: Sequence[float],
    included_gates: Iterable[int],
) -> np.ndarray:
    included = set(int(g) for g in included_gates)
    x0, x1 = float(x[0]), float(x[1])
    phi_1, phi_2, phi_cross = feature_values(x0, x1)
    th = [float(v) for v in theta]

    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0

    if 1 in included:
        _apply_h(state, 0)
    if 2 in included:
        _apply_phase(state, 0, phi_1)
    if 3 in included:
        _apply_h(state, 1)
    if 4 in included:
        _apply_phase(state, 1, phi_2)
    if 5 in included:
        _apply_cx(state, 0, 1)
    if 6 in included:
        _apply_phase(state, 1, phi_cross)
    if 7 in included:
        _apply_cx(state, 0, 1)

    if 8 in included:
        _apply_h(state, 0)
    if 9 in included:
        _apply_phase(state, 0, phi_1)
    if 10 in included:
        _apply_h(state, 1)
    if 11 in included:
        _apply_phase(state, 1, phi_2)
    if 12 in included:
        _apply_cx(state, 0, 1)
    if 13 in included:
        _apply_phase(state, 1, phi_cross)
    if 14 in included:
        _apply_cx(state, 0, 1)

    if 15 in included:
        _apply_ry(state, 0, th[0])
    if 16 in included:
        _apply_ry(state, 1, th[1])
    if 17 in included:
        _apply_cx(state, 0, 1)
    if 18 in included:
        _apply_ry(state, 0, th[2])
    if 19 in included:
        _apply_ry(state, 1, th[3])
    return state


def q0_one_probability(state: np.ndarray) -> float:
    return float(np.abs(state[1]) ** 2 + np.abs(state[3]) ** 2)


def qnn_one_shot_accuracy(
    coalition: FrozenSet[int],
    X: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    theta: Sequence[float] = PAPER_THETA,
    passive_gates: Sequence[int] = QNN_PASSIVE_GATES,
) -> float:
    included = set(int(g) for g in coalition) | set(int(g) for g in passive_gates)
    correct = 0
    for x, label in zip(X, y):
        p_one = q0_one_probability(qnn_state(x, theta, included))
        pred = int(rng.random() < p_one)
        correct += int(pred == int(label))
    return float(correct / len(y))


def qnn_expected_accuracy(
    coalition: Iterable[int],
    X: np.ndarray,
    y: np.ndarray,
    *,
    theta: Sequence[float] = PAPER_THETA,
    passive_gates: Sequence[int] = QNN_PASSIVE_GATES,
) -> float:
    included = set(int(g) for g in coalition) | set(int(g) for g in passive_gates)
    probs = []
    for x, label in zip(X, y):
        p_one = q0_one_probability(qnn_state(x, theta, included))
        probs.append(p_one if int(label) == 1 else 1.0 - p_one)
    return float(np.mean(probs))


def qnn_threshold_accuracy(
    coalition: Iterable[int],
    X: np.ndarray,
    y: np.ndarray,
    *,
    theta: Sequence[float] = PAPER_THETA,
    passive_gates: Sequence[int] = QNN_PASSIVE_GATES,
) -> float:
    included = set(int(g) for g in coalition) | set(int(g) for g in passive_gates)
    correct = 0
    for x, label in zip(X, y):
        p_one = q0_one_probability(qnn_state(x, theta, included))
        pred = int(p_one >= 0.5)
        correct += int(pred == int(label))
    return float(correct / len(y))
