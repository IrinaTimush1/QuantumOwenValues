# partition_util.py - Partition helpers for Owen values
# Part of OVQX (https://github.com/IrinaTimush1/OVQX), accompanying the B.Sc.
# thesis "Group-Structured Attributions for Explainable Quantum Machine
# Learning" by Irina Timus (Maastricht University, 2026). MIT-licensed.
"""
partition_utils.py — Utilities for defining Owen-value partitions
==================================================================

This module provides helper functions to inspect a quantum circuit and
build partitions (groupings of gates) for Owen-value computation.

The core idea:
  - Every gate in a Qiskit QuantumCircuit has an INDEX (its position
    in qc.data, starting from 0).
  - Some gates are "locked" (always active, excluded from the game).
  - The remaining "unlocked" gate indices are the PLAYERS.
  - A PARTITION groups these unlocked indices into disjoint sets.

This module helps you go from a circuit to a partition.
"""

from collections import defaultdict


# =========================================================================
# Step 1: Inspect a circuit — see what you're working with
# =========================================================================

def inspect_circuit(qc, locked_instructions=None):
    """
    Print a human-readable table of every gate in the circuit.

    This is the first thing you should call when designing a partition.
    It shows you each gate's index, name, parameters, which qubits it
    acts on, and whether it's locked.

    Parameters
    ----------
    qc : QuantumCircuit
    locked_instructions : list[int] or None
        Gate indices that will be locked (always active).

    Returns
    -------
    info_list : list[dict]
        One dict per gate with keys: index, name, params, qubits,
        num_qubits, is_locked, is_entangling.
    """
    if locked_instructions is None:
        locked_instructions = []

    info_list = []
    print(f"{'idx':>4} {'locked':>6} {'name':<12} {'qubits':<15} {'#q':>3} {'entangling':>10} {'params'}")
    print("-" * 75)

    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        qubits = [q.index for q in qargs]
        is_locked = idx in locked_instructions
        is_entangling = len(qubits) >= 2  # 2+ qubit gate = entangling
        params = [float(p) if hasattr(p, '__float__') else str(p) for p in instr.params]

        info = {
            "index": idx,
            "name": instr.name,
            "params": params,
            "qubits": qubits,
            "num_qubits": len(qubits),
            "is_locked": is_locked,
            "is_entangling": is_entangling,
        }
        info_list.append(info)

        lock_str = "LOCKED" if is_locked else ""
        ent_str = "yes" if is_entangling else ""
        param_str = str(params) if params else ""
        print(f"{idx:4d} {lock_str:>6} {instr.name:<12} {str(qubits):<15} {len(qubits):3d} {ent_str:>10} {param_str}")

    # Summary
    unlocked = [i["index"] for i in info_list if not i["is_locked"]]
    print(f"\nTotal gates: {len(info_list)}")
    print(f"Locked: {locked_instructions}")
    print(f"Unlocked (players): {unlocked}")
    return info_list


# =========================================================================
# Step 2: Build partitions — multiple strategies
# =========================================================================

def partition_by_gate_type(qc, locked_instructions=None):
    """
    Group unlocked gates by their NAME (h, cx, ry, rz, etc.).

    Example output for a circuit with H, RY, and CX gates:
        [[0, 3, 6],    # all H gates
         [1, 4, 7],    # all RY gates
         [2, 5, 8]]    # all CX gates

    This is useful when you want to ask: "How important is the
    class of entangling gates vs rotation gates vs Hadamard gates?"
    """
    if locked_instructions is None:
        locked_instructions = []

    groups = defaultdict(list)
    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        if idx not in locked_instructions:
            groups[instr.name].append(idx)

    partition = list(groups.values())
    labels = list(groups.keys())
    return partition, labels


def partition_entangling_vs_local(qc, locked_instructions=None):
    """
    Two groups: entangling gates (2+ qubits) vs local gates (1 qubit).

    This directly answers: "How much do entangling gates contribute
    compared to single-qubit rotations?"

    Returns
    -------
    partition : list[list[int]]
        [local_gate_indices, entangling_gate_indices]
        (empty groups are omitted)
    labels : list[str]
        ["local", "entangling"] (matching partition order)
    """
    if locked_instructions is None:
        locked_instructions = []

    local = []
    entangling = []
    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        if idx not in locked_instructions:
            if len(qargs) >= 2:
                entangling.append(idx)
            else:
                local.append(idx)

    partition = []
    labels = []
    if local:
        partition.append(local)
        labels.append("local")
    if entangling:
        partition.append(entangling)
        labels.append("entangling")

    return partition, labels


def partition_by_layer(qc, layer_boundaries, locked_instructions=None):
    """
    Group gates into layers defined by explicit index boundaries.

    Parameters
    ----------
    qc : QuantumCircuit
    layer_boundaries : list[int]
        Gate indices where each new layer STARTS.
        Example: [0, 5, 10] means:
          Layer 0 = gates 0,1,2,3,4
          Layer 1 = gates 5,6,7,8,9
          Layer 2 = gates 10,11,...,end

    Returns
    -------
    partition : list[list[int]]
    labels : list[str]
    """
    if locked_instructions is None:
        locked_instructions = []

    num_gates = len(qc.data)
    boundaries = sorted(layer_boundaries) + [num_gates]

    partition = []
    labels = []
    for layer_idx in range(len(boundaries) - 1):
        start = boundaries[layer_idx]
        end = boundaries[layer_idx + 1]
        group = [g for g in range(start, end) if g not in locked_instructions]
        if group:
            partition.append(group)
            labels.append(f"layer_{layer_idx}")

    return partition, labels


def partition_by_repetition(qc, gates_per_rep, locked_instructions=None,
                            num_locked_prefix=0):
    """
    For circuits with a repeating structure (common in VQAs), group
    gates by repetition index.

    Parameters
    ----------
    qc : QuantumCircuit
    gates_per_rep : int
        Number of gates in each repetition block.
    locked_instructions : list[int] or None
    num_locked_prefix : int
        Number of gates at the start that are locked (e.g., initial H layer).
        These are skipped before counting repetitions.

    Example
    -------
    For the QSVM feature map (Fig. 4 in the paper) with r=2 repetitions
    and 7 gates per repetition:

        partition_by_repetition(qc, gates_per_rep=7, num_locked_prefix=0)
        # → [[0,1,2,3,4,5,6], [7,8,9,10,11,12,13]]

    For the QAOA circuit (Fig. 22) with cost+mix layers:
        # Each "repetition" has 2 layers (cost + mix)
        partition_by_repetition(qc, gates_per_rep=2, num_locked_prefix=7)
        # → [[7,8], [9,10], [11,12], ...]
    """
    if locked_instructions is None:
        locked_instructions = []

    num_gates = len(qc.data)
    partition = []
    labels = []
    rep = 0
    g = num_locked_prefix

    while g < num_gates:
        group = []
        for offset in range(gates_per_rep):
            idx = g + offset
            if idx < num_gates and idx not in locked_instructions:
                group.append(idx)
        if group:
            partition.append(group)
            labels.append(f"rep_{rep}")
        g += gates_per_rep
        rep += 1

    return partition, labels


def partition_by_qubit(qc, locked_instructions=None):
    """
    Group gates by which qubit(s) they act on.

    Single-qubit gates go into the group for their qubit.
    Multi-qubit gates go into a separate "multi" group.

    This answers: "Which qubit's gates contribute most?"
    """
    if locked_instructions is None:
        locked_instructions = []

    qubit_groups = defaultdict(list)
    multi_group = []

    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        if idx not in locked_instructions:
            qubits = [q.index for q in qargs]
            if len(qubits) == 1:
                qubit_groups[qubits[0]].append(idx)
            else:
                multi_group.append(idx)

    partition = []
    labels = []
    for q in sorted(qubit_groups.keys()):
        partition.append(qubit_groups[q])
        labels.append(f"qubit_{q}")
    if multi_group:
        partition.append(multi_group)
        labels.append("multi_qubit")

    return partition, labels


def partition_manual(groups, locked_instructions=None):
    """
    Validate and return a manually specified partition.

    Parameters
    ----------
    groups : dict[str, list[int]]
        Named groups.  Example:
        {
            "data_encoding": [0, 1, 2, 3],
            "entangling":    [4, 5, 6],
            "variational":   [7, 8, 9, 10],
        }

    Returns
    -------
    partition : list[list[int]]
    labels : list[str]
    """
    if locked_instructions is None:
        locked_instructions = []

    partition = []
    labels = []
    all_indices = []

    for name, indices in groups.items():
        clean = [i for i in indices if i not in locked_instructions]
        if clean:
            partition.append(clean)
            labels.append(name)
            all_indices.extend(clean)

    # Check no duplicates
    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Duplicate gate indices across groups!")

    return partition, labels


# =========================================================================
# Step 3: Validate a partition before passing to OwenValues
# =========================================================================

def validate_partition(partition, qc, locked_instructions=None):
    """
    Check that a partition is valid for the given circuit.

    Raises ValueError with a descriptive message if invalid.
    Returns the set of unlocked indices for convenience.
    """
    if locked_instructions is None:
        locked_instructions = []

    num_gates = len(qc.data)
    all_indices = set(range(num_gates))
    locked_set = set(locked_instructions)
    unlocked_set = all_indices - locked_set

    # Flatten partition
    partition_flat = []
    for group in partition:
        partition_flat.extend(group)

    partition_set = set(partition_flat)

    # Check coverage
    missing = unlocked_set - partition_set
    extra = partition_set - unlocked_set

    if missing:
        raise ValueError(
            f"Partition does not cover these unlocked gates: {sorted(missing)}\n"
            f"Every unlocked gate must appear in exactly one group."
        )
    if extra:
        raise ValueError(
            f"Partition contains indices not in unlocked gates: {sorted(extra)}\n"
            f"These might be locked gates or out-of-range indices."
        )
    if len(partition_flat) != len(partition_set):
        raise ValueError("Partition contains duplicate gate indices.")

    print(f"✓ Partition is valid: {len(partition)} groups covering "
          f"{len(unlocked_set)} unlocked gates.")
    return unlocked_set


# =========================================================================
# Demo: putting it all together
# =========================================================================

if __name__ == "__main__":
    from qiskit.circuit import QuantumCircuit
    import numpy as np

    print("=" * 60)
    print("DEMO: Building partitions for a QNN-style circuit")
    print("=" * 60)

    # Build a circuit similar to Fig. 12 in the paper
    qc = QuantumCircuit(2, 2)

    # Layer 1: Hadamard (data prep)
    qc.h(0)                    # gate 0
    qc.h(1)                    # gate 1

    # Layer 2: Data encoding
    qc.rz(0.5, 0)             # gate 2
    qc.rz(0.3, 1)             # gate 3

    # Layer 3: Entangling
    qc.cx(0, 1)               # gate 4

    # Layer 4: Variational rotations
    qc.ry(0.7, 0)             # gate 5
    qc.ry(0.2, 1)             # gate 6

    # Layer 5: More entangling
    qc.cx(1, 0)               # gate 7

    # Layer 6: More variational
    qc.ry(0.9, 0)             # gate 8
    qc.ry(0.4, 1)             # gate 9

    # ── Lock the initial H gates (they're always active) ────────────
    locked = [0, 1]

    print("\n--- Circuit inspection ---")
    info = inspect_circuit(qc, locked)

    # ── Strategy A: Entangling vs Local ──────────────────────────────
    print("\n--- Strategy A: Entangling vs Local ---")
    part_a, labels_a = partition_entangling_vs_local(qc, locked)
    for label, group in zip(labels_a, part_a):
        print(f"  {label}: gate indices {group}")

    # ── Strategy B: By gate type ─────────────────────────────────────
    print("\n--- Strategy B: By gate type ---")
    part_b, labels_b = partition_by_gate_type(qc, locked)
    for label, group in zip(labels_b, part_b):
        print(f"  {label}: gate indices {group}")

    # ── Strategy C: Manual functional grouping ───────────────────────
    print("\n--- Strategy C: Manual functional grouping ---")
    part_c, labels_c = partition_manual({
        "data_encoding": [2, 3],
        "entangling":    [4, 7],
        "variational":   [5, 6, 8, 9],
    }, locked)
    for label, group in zip(labels_c, part_c):
        print(f"  {label}: gate indices {group}")

    # ── Strategy D: By qubit ─────────────────────────────────────────
    print("\n--- Strategy D: By qubit ---")
    part_d, labels_d = partition_by_qubit(qc, locked)
    for label, group in zip(labels_d, part_d):
        print(f"  {label}: gate indices {group}")

    # ── Validate one of them ─────────────────────────────────────────
    print("\n--- Validation ---")
    validate_partition(part_c, qc, locked)

    print("\n--- How to use with QuantumOwenValues ---")
    print("""
    from qowen import QuantumOwenValues

    qov = QuantumOwenValues(
        qc=qc,
        partition=[[2, 3], [4, 7], [5, 6, 8, 9]],
        value_fun=my_value_function,
        value_kwargs_dict={...},
        quantum_instance=qi,
        locked_instructions=[0, 1],
    )
    qov.run()
    # qov.phi_dict → {2: ..., 3: ..., 4: ..., 5: ..., ...}
    #
    # To get MODULE-level importance, sum within groups:
    # data_encoding_importance = qov.phi_dict[2] + qov.phi_dict[3]
    # entangling_importance    = qov.phi_dict[4] + qov.phi_dict[7]
    # variational_importance   = qov.phi_dict[5] + qov.phi_dict[6]
    #                          + qov.phi_dict[8] + qov.phi_dict[9]
    """)
