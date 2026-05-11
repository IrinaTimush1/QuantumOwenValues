"""
test_quantum_owen_workflow.py
=============================

End-to-end sanity test for the Quantum Owen Values framework.

This script:
1. adds the local qshaptools/src folder to Python's import path,
2. builds a small quantum circuit,
3. locks selected gates,
4. defines a manual partition over the unlocked gates,
5. defines a simple deterministic value function,
6. runs exact Owen values,
7. runs sampled Owen values,
8. optionally runs a noisy sampled Owen test,
9. aggregates gate-level Owen values into module-level scores.

This is a workflow/integration test for your Owen framework.
It is not a physics-faithfulness test for your thesis value function.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Make local src-layout package importable
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "qshaptools" / "src"
sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

from qiskit.circuit import QuantumCircuit
from qshaptools.qowen import QuantumOwenValues


# ---------------------------------------------------------------------
# Local helper: inspect a circuit
# ---------------------------------------------------------------------

def inspect_circuit_local(qc, locked_instructions=None):
    """
    Print a simple readable table of all gates in the circuit.
    """
    if locked_instructions is None:
        locked_instructions = []

    print(f"{'idx':>4} {'locked':>8} {'name':<12} {'qubits':<15} {'#q':>3}")
    print("-" * 55)

    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        qubits = []
        for q in qargs:
            try:
                qubits.append(qc.find_bit(q).index)
            except Exception:
                qubits.append(getattr(q, "index", "?"))

        is_locked = idx in locked_instructions
        lock_str = "LOCKED" if is_locked else ""
        print(f"{idx:4d} {lock_str:>8} {instr.name:<12} {str(qubits):<15} {len(qubits):3d}")

    unlocked = [i for i in range(len(qc.data)) if i not in locked_instructions]
    print("\nSummary:")
    print(f"  total gates : {len(qc.data)}")
    print(f"  locked      : {locked_instructions}")
    print(f"  unlocked    : {unlocked}")


# ---------------------------------------------------------------------
# Local helper: validate a partition
# ---------------------------------------------------------------------

def validate_partition_local(partition, qc, locked_instructions=None):
    """
    Check that the partition is a valid partition of the unlocked gates.
    """
    if locked_instructions is None:
        locked_instructions = []

    num_gates = len(qc.data)
    all_indices = set(range(num_gates))
    locked_set = set(locked_instructions)
    unlocked_set = all_indices - locked_set

    flat = []
    for group in partition:
        flat.extend(group)

    flat_set = set(flat)

    missing = unlocked_set - flat_set
    extra = flat_set - unlocked_set

    if missing:
        raise ValueError(
            f"Partition is missing unlocked gates: {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"Partition contains invalid/locked/out-of-range gates: {sorted(extra)}"
        )
    if len(flat) != len(flat_set):
        raise ValueError("Partition contains duplicate gate indices.")

    print(f"✓ Partition is valid: {len(partition)} groups covering {len(unlocked_set)} unlocked gates.")


# ---------------------------------------------------------------------
# Step 1: Build a toy circuit
# ---------------------------------------------------------------------

def build_test_circuit():
    """
    Build a tiny 2-qubit circuit with clear modular structure.

    Gate indices:
        0: h(0)
        1: h(1)
        2: rz(0.5, 0)
        3: rz(0.3, 1)
        4: cx(0, 1)
        5: ry(0.7, 0)
        6: ry(0.2, 1)
        7: cx(1, 0)
        8: ry(0.9, 0)
        9: ry(0.4, 1)
    """
    qc = QuantumCircuit(2)

    qc.h(0)          # 0
    qc.h(1)          # 1
    qc.rz(0.5, 0)    # 2
    qc.rz(0.3, 1)    # 3
    qc.cx(0, 1)      # 4
    qc.ry(0.7, 0)    # 5
    qc.ry(0.2, 1)    # 6
    qc.cx(1, 0)      # 7
    qc.ry(0.9, 0)    # 8
    qc.ry(0.4, 1)    # 9

    return qc


# ---------------------------------------------------------------------
# Step 2: Deterministic toy value function
# ---------------------------------------------------------------------

def toy_value_fun(qc_data, num_qubits, S, quantum_instance=None, **kw):
    """
    Simple deterministic value function for framework testing.

    It rewards:
    - active gates in general,
    - 2-qubit gates more strongly,
    - slightly later gates a little more.

    This is only to test the Owen machinery.
    """
    score = 0.0
    active = set(S)

    for idx, (instr, qargs, cargs, opts) in enumerate(qc_data):
        if idx in active:
            score += 1.0
            if len(qargs) >= 2:
                score += 2.0
            score += 0.01 * idx

    return float(score)


# ---------------------------------------------------------------------
# Optional noisy toy value function
# ---------------------------------------------------------------------

def noisy_toy_value_fun(qc_data, num_qubits, S, quantum_instance=None, noise_std=0.1, **kw):
    """
    Noisy version of toy_value_fun to test repeated evaluations.
    """
    import numpy as np

    base = toy_value_fun(
        qc_data=qc_data,
        num_qubits=num_qubits,
        S=S,
        quantum_instance=quantum_instance,
        **kw,
    )
    return float(base + np.random.randn() * noise_std)


# ---------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------

def print_gate_scores(phi_dict, title="Gate-level Owen values"):
    print(f"\n{title}:")
    for gate_idx in sorted(phi_dict):
        print(f"  gate {gate_idx:2d}: {phi_dict[gate_idx]: .6f}")


def aggregate_module_scores(phi_dict, partition, labels):
    """
    Sum gate-level Owen values inside each module/group.
    """
    module_scores = {}
    for label, group in zip(labels, partition):
        module_scores[label] = float(sum(phi_dict[g] for g in group))
    return module_scores


def print_module_scores(module_scores, title="Module-level Owen values"):
    print(f"\n{title}:")
    for label, score in module_scores.items():
        print(f"  {label:<16s}: {score: .6f}")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    print("=" * 72)
    print("QUANTUM OWEN VALUES WORKFLOW TEST")
    print("=" * 72)

    # 1) Build circuit
    qc = build_test_circuit()

    # 2) Decide which gates are locked
    #    Here the initial H gates are always active and excluded from the game.
    locked = [0, 1]

    # 3) Define a manual partition over UNLOCKED gates
    #    unlocked gates are [2,3,4,5,6,7,8,9]
    partition = [
        [2, 3],          # data_encoding
        [4, 7],          # entangling
        [5, 6, 8, 9],    # variational
    ]
    labels = ["data_encoding", "entangling", "variational"]

    print("\n--- CIRCUIT ---")
    inspect_circuit_local(qc, locked_instructions=locked)

    print("\n--- PARTITION CHECK ---")
    validate_partition_local(partition, qc, locked_instructions=locked)

    print("\nManual partition used:")
    for label, group in zip(labels, partition):
        print(f"  {label:<16s}: {group}")

    # 4) Define value function
    value_fun = toy_value_fun

    # 5) Create and run exact QuantumOwenValues
    print("\n" + "=" * 72)
    print("RUN EXACT OWEN")
    print("=" * 72)

    qov_exact = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=value_fun,
        value_kwargs_dict={},
        quantum_instance=None,        # toy value function does not use it
        locked_instructions=locked,
        owen_sample_frac=None,        # exact mode
        owen_sample_reps=1,
        owen_sample_seed=123,
        name="exact_owen",
        silent=False,
    )

    phi_exact = qov_exact.run()
    print_gate_scores(phi_exact, title="Exact gate-level Owen values")

    module_scores_exact = aggregate_module_scores(phi_exact, partition, labels)
    print_module_scores(module_scores_exact, title="Exact module-level Owen values")

    # 6) Create and run sampled QuantumOwenValues
    print("\n" + "=" * 72)
    print("RUN SAMPLED OWEN")
    print("=" * 72)

    qov_sampled = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=value_fun,
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=locked,
        owen_sample_frac=-50,         # sample exactly 50 (R,T) visits per player
        owen_sample_reps=1,
        owen_sample_seed=123,
        name="sampled_owen",
        silent=False,
    )

    phi_sampled = qov_sampled.run()
    print_gate_scores(phi_sampled, title="Sampled gate-level Owen values")

    module_scores_sampled = aggregate_module_scores(phi_sampled, partition, labels)
    print_module_scores(module_scores_sampled, title="Sampled module-level Owen values")

    # 7) Compare exact vs sampled
    print("\n" + "=" * 72)
    print("EXACT VS SAMPLED")
    print("=" * 72)
    for g in sorted(phi_exact):
        diff = phi_sampled[g] - phi_exact[g]
        print(f"  gate {g:2d}: sampled - exact = {diff: .6f}")

    # 8) Optional noisy sampled test
    print("\n" + "=" * 72)
    print("RUN NOISY SAMPLED OWEN (OPTIONAL)")
    print("=" * 72)

    qov_noisy = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=noisy_toy_value_fun,
        value_kwargs_dict={"noise_std": 0.2},
        quantum_instance=None,
        locked_instructions=locked,
        owen_sample_frac=-50,
        owen_sample_reps=8,           # repeated evals per sampled visit
        owen_sample_seed=123,
        name="noisy_sampled_owen",
        silent=False,
    )

    phi_noisy = qov_noisy.run()
    print_gate_scores(phi_noisy, title="Noisy sampled gate-level Owen values")

    module_scores_noisy = aggregate_module_scores(phi_noisy, partition, labels)
    print_module_scores(module_scores_noisy, title="Noisy sampled module-level Owen values")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()