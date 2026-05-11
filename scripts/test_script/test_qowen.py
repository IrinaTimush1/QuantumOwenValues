from pathlib import Path
import sys
import numpy as np

# ---------------------------------------------------------------------
# Make qshaptools importable in the SAME style as the original repo
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
QSHAPTOOLS_SRC = ROOT / "qshaptools" / "src"
QSHAPTOOLS_PKG = ROOT / "qshaptools" / "src" / "qshaptools"
sys.path.insert(0, str(QSHAPTOOLS_SRC))
sys.path.insert(0, str(QSHAPTOOLS_PKG))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qshaptools.qowen import QuantumOwenValues
from qshaptools.qshap import QuantumShapleyValues
from qshaptools.tools import build_circuit
from qshaptools.tools import extract_from_circuit
from qshaptools.uowen import OwenValues

 

# ---------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------
PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition, msg):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ PASS  {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  ✗ FAIL  {msg}")


def close(a, b, atol=1e-9):
    return abs(float(a) - float(b)) < atol


def dicts_close(d1, d2, atol=1e-9):
    if set(d1.keys()) != set(d2.keys()):
        return False
    return all(close(d1[k], d2[k], atol) for k in d1)


# ---------------------------------------------------------------------
# Deterministic quantum value function
# ---------------------------------------------------------------------
def value_prob_q0_one(S, qc_data, num_qubits, **kwargs):
    """
    Deterministic value function:
    v(S) = probability that qubit 0 is measured as 1
    in the statevector of the coalition circuit.

    No simulator backend needed.
    """
    qc, _ = build_circuit(qc_data, num_qubits, S=S, cl_bits=False)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()

    # Qiskit basis ordering for 2 qubits: |q1 q0>
    # q0 = 1 corresponds to odd indices
    p_q0_1 = float(np.sum(probs[1::2]))
    return p_q0_1


# ---------------------------------------------------------------------
# Small test circuits
# ---------------------------------------------------------------------
def make_test_circuit_1():
    """
    2-qubit circuit with 5 gates, all unlocked by default.
    """
    qc = QuantumCircuit(2)
    qc.h(0)       # 0
    qc.cx(0, 1)   # 1
    qc.ry(0.3, 0) # 2
    qc.rz(0.5, 1) # 3
    qc.cx(1, 0)   # 4
    return qc


def make_test_circuit_2():
    """
    2-qubit circuit where some gates can be locked.
    """
    qc = QuantumCircuit(2)
    qc.h(0)        # 0
    qc.h(1)        # 1
    qc.cx(0, 1)    # 2
    qc.ry(0.7, 0)  # 3
    qc.ry(0.2, 1)  # 4
    qc.cx(1, 0)    # 5
    return qc


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_singleton_partition_equals_qshap():
    print("\n═══ Test 1: Quantum singleton partition = Quantum Shapley ═══")
    qc = make_test_circuit_1()
    partition = [[0], [1], [2], [3], [4]]

    qov = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=value_prob_q0_one,
        value_kwargs_dict={},
        quantum_instance=None,
        silent=True,
    )
    qov.run()

    qsv = QuantumShapleyValues(
        qc=qc,
        value_fun=value_prob_q0_one,
        value_kwargs_dict={},
        quantum_instance=None,
        silent=True,
    )
    qsv()

    check(dicts_close(qov.phi_dict, qsv.phi_dict, atol=1e-9),
          "Quantum Owen == Quantum Shapley for singleton partition")


def test_grand_partition_equals_qshap():
    print("\n═══ Test 2: Quantum grand partition = Quantum Shapley ═══")
    qc = make_test_circuit_1()
    partition = [[0, 1, 2, 3, 4]]

    qov = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=value_prob_q0_one,
        value_kwargs_dict={},
        quantum_instance=None,
        silent=True,
    )
    qov.run()

    qsv = QuantumShapleyValues(
        qc=qc,
        value_fun=value_prob_q0_one,
        value_kwargs_dict={},
        quantum_instance=None,
        silent=True,
    )
    qsv()

    check(dicts_close(qov.phi_dict, qsv.phi_dict, atol=1e-9),
          "Quantum Owen == Quantum Shapley for grand partition")


def test_locked_instructions_respected():
    print("\n═══ Test 3: Locked instructions respected ═══")
    qc = make_test_circuit_2()
    locked = [0, 1]
    partition = [[2, 3], [4, 5]]

    qov = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=value_prob_q0_one,
        value_kwargs_dict={},
        quantum_instance=None,
        locked_instructions=locked,
        silent=True,
    )
    qov.run()

    expected_unlocked = [2, 3, 4, 5]
    check(qov._unlocked_instructions == expected_unlocked,
          f"Unlocked instructions are {expected_unlocked}")
    check(qov._locked_instructions == locked,
          f"Locked instructions are {locked}")
    check(set(qov.phi_dict.keys()) == set(expected_unlocked),
          "Owen values are returned only for unlocked instructions")


def test_partition_validation():
    print("\n═══ Test 4: Partition validation catches invalid partition ═══")
    qc = make_test_circuit_1()

    # Duplicate gate 1, missing gate 4
    bad_partition = [[0, 1], [1, 2, 3]]

    try:
        _ = QuantumOwenValues(
            qc=qc,
            partition=bad_partition,
            value_fun=value_prob_q0_one,
            value_kwargs_dict={},
            quantum_instance=None,
            silent=True,
        )
        check(False, "Invalid partition should raise ValueError")
    except ValueError:
        check(True, "Invalid partition raises ValueError")


def test_sampled_qowen_convergence():
    print("\n═══ Test 5: Sampled Quantum Owen converges toward exact Quantum Owen ═══")
    qc = make_test_circuit_1()
    partition = [[0, 1], [2, 3, 4]]

    qov_exact = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=value_prob_q0_one,
        value_kwargs_dict={},
        quantum_instance=None,
        silent=True,
    )
    qov_exact.run()
    exact = qov_exact.phi_dict

    sample_counts = [10, 50, 200, 1000]
    errors = []

    for L in sample_counts:
        trials = []
        for seed in range(10):
            qov_samp = QuantumOwenValues(
                qc=qc,
                partition=partition,
                value_fun=value_prob_q0_one,
                value_kwargs_dict={},
                quantum_instance=None,
                owen_sample_frac=-L,   # absolute number of sampled (R,T) pairs
                owen_sample_seed=seed,
                silent=True,
            )
            qov_samp.run()
            trials.append(qov_samp.phi_dict)

        avg = {i: np.mean([t[i] for t in trials]) for i in exact.keys()}
        err = max(abs(avg[i] - exact[i]) for i in exact.keys())
        errors.append(err)
        print(f"    L={L:5d}  max|err|={err:.6f}")

    check(errors[-1] < errors[0],
          f"Sampled error decreases: {errors[0]:.6f} → {errors[-1]:.6f}")


def test_summary_and_aliases():
    print("\n═══ Test 6: Summary / helper methods ═══")
    qc = make_test_circuit_1()
    partition = [[0, 1], [2, 3, 4]]

    qov = QuantumOwenValues(
        qc=qc,
        partition=partition,
        value_fun=value_prob_q0_one,
        value_kwargs_dict={},
        quantum_instance=None,
        name="qowen_test",
        silent=True,
    )
    qov.run()

    s = str(qov)
    d = qov.get_summary_dict()

    check(len(s) > 0, "__str__ returns non-empty string")
    check("partition" in d, "summary contains partition")
    check("phi_dict" in d, "summary contains phi_dict")
    check("num_qubits" in d, "summary contains num_qubits")
    check(d["num_qubits"] == 2, "num_qubits == 2")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Quantum Owen Wrapper Verification Suite")
    print("=" * 70)

    test_singleton_partition_equals_qshap()
    test_grand_partition_equals_qshap()
    test_locked_instructions_respected()
    test_partition_validation()
    test_sampled_qowen_convergence()
    test_summary_and_aliases()

    print("\n" + "=" * 70)
    print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 70)

    sys.exit(1 if FAIL_COUNT > 0 else 0)