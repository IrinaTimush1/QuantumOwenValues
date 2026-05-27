#!/usr/bin/env python3
"""Direct gate-level Shapley estimator used by the reproduction scripts.

The estimator follows the Shapley value convention used by Heese et al.:

    phi_i = 1/N * sum_S [v(S union {i}) - v(S)] / binom(N - 1, |S|)

For alpha=1 it enumerates all contexts S subset N\\{i}. For alpha<1 it
samples contexts from the equivalent distribution: choose a coalition size
uniformly from 0..N-1, then choose a subset of that size uniformly.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


Coalition = FrozenSet[int]
ValueFunction = Callable[[Coalition], float]
ProgressCallback = Callable[[str], None]


@dataclass
class ShapleyRunStats:
    """Small metadata bundle returned with an estimator run."""

    n_players: int
    alpha: float
    k_repetitions: int
    contexts_per_player: int
    exact: bool
    seed: int
    unique_value_evaluations: int
    cached_mean_values: int


def _powerset(items: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    for size in range(len(items) + 1):
        yield from itertools.combinations(items, size)


def _sample_context(others: Sequence[int], rng: random.Random) -> Coalition:
    size = rng.randrange(0, len(others) + 1)
    if size == 0:
        return frozenset()
    return frozenset(rng.sample(list(others), size))


def _contexts_for_player(
    players: Sequence[int],
    player: int,
    *,
    alpha: float,
    rng: random.Random,
) -> Tuple[List[Coalition], bool]:
    others = [p for p in players if p != player]
    if alpha >= 1.0:
        return [frozenset(s) for s in _powerset(others)], True

    n_contexts = max(1, int(math.ceil(float(alpha) * (2 ** len(others)))))
    return [_sample_context(others, rng) for _ in range(n_contexts)], False


def estimate_shapley(
    players: Sequence[int],
    value_function: ValueFunction,
    *,
    alpha: float = 1.0,
    k_repetitions: int = 1,
    seed: int = 0,
    value_cache: MutableMapping[Coalition, float] | None = None,
    progress: ProgressCallback | None = None,
) -> Tuple[Dict[int, float], MutableMapping[Coalition, float], ShapleyRunStats]:
    """Estimate Shapley values for a list of players.

    Parameters
    ----------
    players:
        Gate indices participating in the game.
    value_function:
        Callable returning v(S) for a frozenset of active gate indices.
    alpha:
        If 1, enumerate all contexts exactly. If below 1, sample
        ceil(alpha * 2^(N-1)) contexts per gate.
    k_repetitions:
        Number of repeated value-function calls used to form the cached mean
        value of each coalition. This is K in the uncertain-Shapley estimator.
    seed:
        Seed for Monte Carlo context sampling. The stochastic value function,
        if any, should use its own seeded RNG.
    value_cache:
        Optional mapping from coalition to the cached K-mean v(S).
    progress:
        Optional callback receiving one-line status strings.
    """

    ordered_players = [int(p) for p in players]
    if len(ordered_players) != len(set(ordered_players)):
        raise ValueError("players contains duplicates")
    if not ordered_players:
        raise ValueError("players must be non-empty")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    if int(k_repetitions) < 1:
        raise ValueError("k_repetitions must be >= 1")

    rng = random.Random(int(seed))
    cache: MutableMapping[Coalition, float] = value_cache if value_cache is not None else {}
    unique_value_evaluations = 0

    def mean_value(coalition: Coalition) -> float:
        nonlocal unique_value_evaluations
        key = frozenset(int(x) for x in coalition)
        if key in cache:
            return float(cache[key])
        vals = [float(value_function(key)) for _ in range(int(k_repetitions))]
        unique_value_evaluations += int(k_repetitions)
        mean = sum(vals) / len(vals)
        cache[key] = mean
        return mean

    phi: Dict[int, float] = {}
    contexts_per_player = 0
    exact = alpha >= 1.0

    for player_idx, player in enumerate(ordered_players, start=1):
        contexts, is_exact = _contexts_for_player(
            ordered_players,
            player,
            alpha=float(alpha),
            rng=rng,
        )
        contexts_per_player = len(contexts)
        exact = exact and is_exact
        if progress is not None:
            progress(
                f"gate {player} ({player_idx}/{len(ordered_players)}): "
                f"{len(contexts)} contexts"
            )

        if is_exact:
            total = 0.0
            n = len(ordered_players)
            for context in contexts:
                marginal = mean_value(frozenset(set(context) | {player})) - mean_value(context)
                total += marginal / (n * math.comb(n - 1, len(context)))
            phi[player] = total
        else:
            if not contexts:
                phi[player] = 0.0
                continue
            total = 0.0
            for context in contexts:
                total += mean_value(frozenset(set(context) | {player})) - mean_value(context)
            phi[player] = total / len(contexts)

    stats = ShapleyRunStats(
        n_players=len(ordered_players),
        alpha=float(alpha),
        k_repetitions=int(k_repetitions),
        contexts_per_player=int(contexts_per_player),
        exact=bool(exact),
        seed=int(seed),
        unique_value_evaluations=int(unique_value_evaluations),
        cached_mean_values=len(cache),
    )
    return phi, cache, stats


def summarize_gate_runs(rows: Sequence[Mapping[str, float | int]]) -> List[Dict[str, float | int]]:
    """Summarize rows containing gate_index, run, shapley_value."""

    by_gate: Dict[int, List[float]] = {}
    for row in rows:
        gate = int(row["gate_index"])
        by_gate.setdefault(gate, []).append(float(row["shapley_value"]))

    summary: List[Dict[str, float | int]] = []
    for gate in sorted(by_gate):
        vals = by_gate[gate]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        summary.append({"gate_index": gate, "mean": mean, "std": math.sqrt(var)})
    return summary

