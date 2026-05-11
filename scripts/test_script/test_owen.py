"""
test_owen.py — Verification suite for the Owen-value implementation
====================================================================

This script runs a battery of mathematical correctness tests to ensure
that the Owen value code is correct.  Each test prints PASS/FAIL.

Tests performed
---------------
1. **Additive game**: for v(S) = Σ w_i, Owen(i) = w_i for ANY partition.
2. **Singleton partition**: partition={{0},{1},...,{N-1}} => Owen = Shapley.
3. **Grand partition**: partition={{0,1,...,N-1}} => Owen = Shapley.
4. **Efficiency axiom**: Σ_i Ow_i = v(N) - v(∅) for several games.
5. **Known Owen ≠ Shapley example**: 3-player majority-style game with
   partition {{0,1},{2}}.  Hand-computed Owen values are verified.
6. **Sampled Owen convergence**: sampled estimator converges to exact
   Owen as the number of samples grows.
7. **Symmetry within group**: symmetric players in the same group get
   equal Owen values.
8. **Null player**: a player that never contributes gets Owen = 0.

All tests use ClassicalOwenValues (no quantum backend needed).
"""

import sys
import numpy as np

# ── make sure the module is importable ────────────────────────────────────
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
QSHAPTOOLS_PATH = ROOT / "qshaptools" / "src" 
sys.path.insert(0, str(QSHAPTOOLS_PATH))

from qshaptools.cowen import ClassicalOwenValues

# Also import Shapley for comparison tests
# We re-implement a tiny exact Shapley here so the test is self-contained.
import math
from itertools import chain, combinations


# =========================================================================
# Utility: brute-force Shapley (for comparison)
# =========================================================================

def _powerset(iterable):
    """Return all subsets as tuples."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def brute_force_shapley(N, v_func):
    """
    Exact Shapley values by enumeration for players {0, ..., N-1}.

    Parameters
    ----------
    N : int
    v_func : callable  — v_func(S_sorted_list) -> float

    Returns
    -------
    dict {player: shapley_value}
    """
    players = list(range(N))
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        val = 0.0
        for S_tuple in _powerset(others):
            S = sorted(list(S_tuple))
            Si = sorted(S + [i])
            w = (math.factorial(len(S))
                 * math.factorial(N - len(S) - 1)
                 / math.factorial(N))
            val += w * (v_func(Si) - v_func(S))
        phi[i] = val
    return phi


def brute_force_owen(N, partition, v_func):
    """
    Exact Owen values by direct double-sum enumeration.

    This is a *reference* implementation used ONLY for verification.
    It does not share any code with uowen.py.

    Parameters
    ----------
    N : int
    partition : list[list[int]]
    v_func : callable  — v_func(S_sorted_list) -> float

    Returns
    -------
    dict {player: owen_value}
    """
    m = len(partition)
    group_indices = list(range(m))

    # player -> group index
    p2g = {}
    for gi, grp in enumerate(partition):
        for p in grp:
            p2g[p] = gi

    phi = {}
    for i in range(N):
        k = p2g[i]
        Ck = partition[k]
        Ck_without_i = [p for p in Ck if p != i]
        other_groups = [g for g in group_indices if g != k]

        val = 0.0
        for R_tuple in _powerset(other_groups):
            R = list(R_tuple)
            Q_R = []
            for r in R:
                Q_R.extend(partition[r])
            alpha = (math.factorial(len(R))
                     * math.factorial(m - len(R) - 1)
                     / math.factorial(m))

            for T_tuple in _powerset(Ck_without_i):
                T = list(T_tuple)
                beta = (math.factorial(len(T))
                        * math.factorial(len(Ck) - len(T) - 1)
                        / math.factorial(len(Ck)))

                S = sorted(Q_R + T)
                Si = sorted(Q_R + T + [i])
                val += alpha * beta * (v_func(Si) - v_func(S))

        phi[i] = val
    return phi


# =========================================================================
# Value functions for tests
# =========================================================================

def vf_additive(S, weights, **kw):
    """v(S) = Σ_{i ∈ S} weights[i]."""
    return sum(weights[i] for i in S)


def vf_cardinality_squared(S, **kw):
    """v(S) = |S|^2."""
    return len(S) ** 2


def vf_majority(S, **kw):
    """
    3-player game:
    v(S) = 1 if |S| >= 2 else 0.
    """
    return 1.0 if len(S) >= 2 else 0.0


def vf_interaction(S, **kw):
    """
    4-player game with interaction:
    v(S) = |S|  +  10 * I({0,1} ⊆ S)  +  10 * I({2,3} ⊆ S)
    Players 0,1 and 2,3 each have synergy.
    """
    val = len(S)
    s = set(S)
    if {0, 1} <= s:
        val += 10
    if {2, 3} <= s:
        val += 10
    return float(val)


def vf_null_player(S, null_idx=3, **kw):
    """
    4-player game where player `null_idx` contributes nothing.
    v(S) = sum of non-null players in S.
    """
    return float(sum(1 for i in S if i != null_idx))


# =========================================================================
# Test harness
# =========================================================================

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
    return abs(a - b) < atol


def dicts_close(d1, d2, atol=1e-9):
    if set(d1.keys()) != set(d2.keys()):
        return False
    return all(close(d1[k], d2[k], atol) for k in d1)


# =========================================================================
# Tests
# =========================================================================

def test_additive_game():
    """
    TEST 1: Additive game  v(S) = Σ w_i
    ─────────────────────────────────────
    For an additive game, every allocation method gives Ow(i) = w_i,
    regardless of the partition.
    """
    print("\n═══ Test 1: Additive game ═══")
    N = 4
    weights = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0}

    partitions = [
        [[0], [1], [2], [3]],           # singletons
        [[0, 1], [2, 3]],               # pairs
        [[0, 1, 2, 3]],                 # grand
        [[0], [1, 2, 3]],               # asymmetric
    ]

    for part in partitions:
        ov = ClassicalOwenValues(
            N=N, partition=part,
            value_fun=vf_additive,
            value_kwargs_dict={"weights": weights},
            silent=True,
        )
        ov.run()
        for i in range(N):
            check(close(ov.phi_dict[i], weights[i]),
                  f"partition={part}  Ow({i})={ov.phi_dict[i]:.6f} == {weights[i]}")


def test_singleton_partition_equals_shapley():
    """
    TEST 2: Singleton partition  ⇒  Owen = Shapley
    ────────────────────────────────────────────────
    When each player is its own group, the outer sum degenerates to the
    standard Shapley formula.
    """
    print("\n═══ Test 2: Singleton partition = Shapley ═══")
    N = 4
    partition = [[i] for i in range(N)]

    for vf, label in [
        (vf_cardinality_squared, "|S|^2"),
        (vf_interaction, "interaction"),
    ]:
        ov = ClassicalOwenValues(
            N=N, partition=partition,
            value_fun=vf, silent=True,
        )
        ov.run()

        shap = brute_force_shapley(N, lambda S: vf(S=S))
        check(dicts_close(ov.phi_dict, shap, atol=1e-9),
              f"Singleton Owen == Shapley for v={label}")


def test_grand_partition_equals_shapley():
    """
    TEST 3: Grand partition  ⇒  Owen = Shapley
    ────────────────────────────────────────────
    When all players share one group, the inner sum IS the Shapley
    formula (the outer sum has only one term with weight 1).
    """
    print("\n═══ Test 3: Grand partition = Shapley ═══")
    N = 4
    partition = [list(range(N))]

    for vf, label in [
        (vf_cardinality_squared, "|S|^2"),
        (vf_interaction, "interaction"),
    ]:
        ov = ClassicalOwenValues(
            N=N, partition=partition,
            value_fun=vf, silent=True,
        )
        ov.run()

        shap = brute_force_shapley(N, lambda S: vf(S=S))
        check(dicts_close(ov.phi_dict, shap, atol=1e-9),
              f"Grand Owen == Shapley for v={label}")


def test_efficiency():
    """
    TEST 4: Efficiency axiom   Σ_i Ow_i = v(N) - v(∅)
    ───────────────────────────────────────────────────
    The Owen value satisfies the same efficiency axiom as Shapley.
    """
    print("\n═══ Test 4: Efficiency axiom ═══")
    test_cases = [
        (4, [[0, 1], [2, 3]], vf_cardinality_squared, "|S|^2"),
        (4, [[0, 1], [2, 3]], vf_interaction, "interaction"),
        (3, [[0, 1], [2]], vf_majority, "majority"),
        (4, [[0], [1, 2, 3]], vf_cardinality_squared, "|S|^2 asym"),
    ]
    for N, partition, vf, label in test_cases:
        ov = ClassicalOwenValues(
            N=N, partition=partition,
            value_fun=vf, silent=True,
        )
        ov.run()
        total = sum(ov.phi_dict.values())
        vN = vf(S=list(range(N)))
        v0 = vf(S=[])
        expected = vN - v0
        check(close(total, expected),
              f"Σ Ow_i = {total:.6f} == v(N)-v(∅) = {expected:.6f}  [{label}]")


def test_known_owen_ne_shapley():
    """
    TEST 5: Known example where Owen ≠ Shapley
    ────────────────────────────────────────────
    Game: N={0,1,2}, v(S) = 1 if |S|≥2 else 0.
    Partition: {{0,1}, {2}}.

    Hand-computed Owen values:
        Ow(0) = 1/2,  Ow(1) = 1/2,  Ow(2) = 0.

    Shapley values:
        Sh(0) = Sh(1) = Sh(2) = 1/3.
    """
    print("\n═══ Test 5: Known Owen ≠ Shapley ═══")
    N = 3
    partition = [[0, 1], [2]]

    ov = ClassicalOwenValues(
        N=N, partition=partition,
        value_fun=vf_majority, silent=True,
    )
    ov.run()

    check(close(ov.phi_dict[0], 0.5),
          f"Ow(0) = {ov.phi_dict[0]:.6f} == 0.5")
    check(close(ov.phi_dict[1], 0.5),
          f"Ow(1) = {ov.phi_dict[1]:.6f} == 0.5")
    check(close(ov.phi_dict[2], 0.0),
          f"Ow(2) = {ov.phi_dict[2]:.6f} == 0.0")

    shap = brute_force_shapley(N, lambda S: vf_majority(S=S))
    check(not dicts_close(ov.phi_dict, shap, atol=0.01),
          f"Owen ≠ Shapley  (Shapley = {shap})")


def test_reference_implementation_agreement():
    """
    TEST 6: Agreement with independent reference implementation
    ────────────────────────────────────────────────────────────
    Compare ClassicalOwenValues against a brute-force Owen computation
    that shares no code with uowen.py.
    """
    print("\n═══ Test 6: Agreement with reference Owen ═══")
    test_cases = [
        (4, [[0, 1], [2, 3]], vf_cardinality_squared, "|S|^2"),
        (4, [[0, 1], [2, 3]], vf_interaction, "interaction"),
        (4, [[0], [1, 2, 3]], vf_interaction, "interaction asym"),
        (3, [[0, 1], [2]], vf_majority, "majority"),
        (5, [[0, 1], [2], [3, 4]], vf_cardinality_squared, "|S|^2 5-player"),
    ]
    for N, partition, vf, label in test_cases:
        ov = ClassicalOwenValues(
            N=N, partition=partition,
            value_fun=vf, silent=True,
        )
        ov.run()

        ref = brute_force_owen(N, partition, lambda S: vf(S=S))
        check(dicts_close(ov.phi_dict, ref, atol=1e-9),
              f"Matches reference for {label}, partition={partition}")


def test_sampled_owen_convergence():
    """
    TEST 7: Sampled Owen estimator converges to exact Owen
    ───────────────────────────────────────────────────────
    With increasing sample count, sampled Owen should approach exact
    Owen.  We verify that the error shrinks.
    """
    print("\n═══ Test 7: Sampled Owen convergence ═══")
    N = 4
    partition = [[0, 1], [2, 3]]

    # Exact
    ov_exact = ClassicalOwenValues(
        N=N, partition=partition,
        value_fun=vf_interaction, silent=True,
    )
    ov_exact.run()
    exact = ov_exact.phi_dict

    # Sampled with increasing L
    errors = []
    sample_counts = [10, 50, 200, 1000]
    for L in sample_counts:
        # Average over several seeds for stability
        trials = []
        for seed in range(10):
            ov_samp = ClassicalOwenValues(
                N=N, partition=partition,
                value_fun=vf_interaction,
                owen_sample_frac=-L,      # negative int = absolute count
                owen_sample_seed=seed,
                silent=True,
            )
            ov_samp.run()
            trials.append(ov_samp.phi_dict)

        avg = {i: np.mean([t[i] for t in trials]) for i in range(N)}
        err = max(abs(avg[i] - exact[i]) for i in range(N))
        errors.append(err)
        print(f"    L={L:5d}  max|err|={err:.6f}")

    # Check that error decreases (at least last < first)
    check(errors[-1] < errors[0],
          f"Error decreases: {errors[0]:.6f} → {errors[-1]:.6f}")
    check(errors[-1] < 0.5,
          f"Error at L={sample_counts[-1]} is small: {errors[-1]:.6f}")


def test_within_group_symmetry():
    """
    TEST 8: Symmetric players in the same group get equal Owen values
    ──────────────────────────────────────────────────────────────────
    If players i, j are in the same group and are interchangeable in v,
    then Ow(i) = Ow(j).
    """
    print("\n═══ Test 8: Within-group symmetry ═══")
    # v(S) = |S|^2: all players are symmetric
    N = 4
    partition = [[0, 1], [2, 3]]
    ov = ClassicalOwenValues(
        N=N, partition=partition,
        value_fun=vf_cardinality_squared, silent=True,
    )
    ov.run()

    check(close(ov.phi_dict[0], ov.phi_dict[1]),
          f"Ow(0)={ov.phi_dict[0]:.6f} == Ow(1)={ov.phi_dict[1]:.6f}")
    check(close(ov.phi_dict[2], ov.phi_dict[3]),
          f"Ow(2)={ov.phi_dict[2]:.6f} == Ow(3)={ov.phi_dict[3]:.6f}")
    # Also across groups since the game is fully symmetric
    check(close(ov.phi_dict[0], ov.phi_dict[2]),
          f"Ow(0)={ov.phi_dict[0]:.6f} == Ow(2)={ov.phi_dict[2]:.6f}  (symmetric game)")


def test_null_player():
    """
    TEST 9: Null player axiom
    ──────────────────────────
    A player whose marginal contribution is always zero gets Ow = 0.
    """
    print("\n═══ Test 9: Null player ═══")
    N = 4
    partition = [[0, 1], [2, 3]]

    ov = ClassicalOwenValues(
        N=N, partition=partition,
        value_fun=vf_null_player,
        value_kwargs_dict={"null_idx": 3},
        silent=True,
    )
    ov.run()
    check(close(ov.phi_dict[3], 0.0),
          f"Ow(null=3) = {ov.phi_dict[3]:.6f} == 0")


def test_owen_with_noisy_reps():
    """
    TEST 10: owen_sample_reps (K) denoising
    ────────────────────────────────────────
    When the value function has additive noise, increasing K should
    reduce the variance of the Owen estimates.
    """
    print("\n═══ Test 10: Noisy repetitions (K) ═══")
    N = 3
    partition = [[0, 1], [2]]
    noise_std = 0.5

    def vf_noisy(S, **kw):
        return vf_majority(S=S) + np.random.randn() * noise_std

    # Run with K=1 and K=32, each with multiple seeds
    def run_many(K, n_seeds=20):
        results = []
        for seed in range(n_seeds):
            np.random.seed(seed * 1000 + K)
            ov = ClassicalOwenValues(
                N=N, partition=partition,
                value_fun=vf_noisy,
                owen_sample_reps=K,
                evaluate_value_only_once=False,
                silent=True,
            )
            ov.run()
            results.append(ov.phi_dict[0])
        return np.std(results)

    std_K1 = run_many(1)
    std_K32 = run_many(32)
    print(f"    std(Ow(0)) at K=1:  {std_K1:.4f}")
    print(f"    std(Ow(0)) at K=32: {std_K32:.4f}")
    check(std_K32 < std_K1,
          f"K=32 variance < K=1 variance ({std_K32:.4f} < {std_K1:.4f})")


def test_interaction_game_owen_vs_shapley():
    """
    TEST 11: Interaction game — Owen captures group synergy
    ────────────────────────────────────────────────────────
    v(S) = |S| + 10·I({0,1}⊆S) + 10·I({2,3}⊆S)
    Partition = {{0,1}, {2,3}} aligns with the synergy structure.

    Owen should give DIFFERENT values than Shapley here because the
    partition respects the synergy structure.
    """
    print("\n═══ Test 11: Interaction game (Owen vs Shapley) ═══")
    N = 4
    partition = [[0, 1], [2, 3]]

    ov = ClassicalOwenValues(
        N=N, partition=partition,
        value_fun=vf_interaction, silent=True,
    )
    ov.run()

    shap = brute_force_shapley(N, lambda S: vf_interaction(S=S))

    print(f"    Owen:    {ov.phi_dict}")
    print(f"    Shapley: {shap}")

    # Verify efficiency for both
    ov_sum = sum(ov.phi_dict.values())
    sh_sum = sum(shap.values())
    vN = vf_interaction(S=[0, 1, 2, 3])
    check(close(ov_sum, vN), f"Owen efficient: {ov_sum:.4f} == {vN}")
    check(close(sh_sum, vN), f"Shapley efficient: {sh_sum:.4f} == {vN}")

    # Verify agreement with reference
    ref = brute_force_owen(N, partition, lambda S: vf_interaction(S=S))
    check(dicts_close(ov.phi_dict, ref),
          "Owen matches reference for interaction game")


def test_display_and_summary():
    """
    TEST 12: __str__ and get_summary_dict don't crash
    """
    print("\n═══ Test 12: Display and summary ═══")
    N = 4
    partition = [[0, 1], [2, 3]]
    ov = ClassicalOwenValues(
        N=N, partition=partition,
        value_fun=vf_cardinality_squared, silent=True,
        name="test_display",
    )
    ov.run()

    s = str(ov)
    check(len(s) > 0, "__str__ returns non-empty string")

    d = ov.get_summary_dict()
    check("partition" in d, "summary contains 'partition'")
    check("phi_dict" in d, "summary contains 'phi_dict'")
    check(d["N"] == 4, "summary N == 4")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Owen Value Verification Suite")
    print("=" * 70)

    test_additive_game()
    test_singleton_partition_equals_shapley()
    test_grand_partition_equals_shapley()
    test_efficiency()
    test_known_owen_ne_shapley()
    test_reference_implementation_agreement()
    test_sampled_owen_convergence()
    test_within_group_symmetry()
    test_null_player()
    test_owen_with_noisy_reps()
    test_interaction_game_owen_vs_shapley()
    test_display_and_summary()

    print("\n" + "=" * 70)
    print(f"Results:  {PASS_COUNT} passed,  {FAIL_COUNT} failed")
    print("=" * 70)

    sys.exit(1 if FAIL_COUNT > 0 else 0)