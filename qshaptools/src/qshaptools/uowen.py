# uowen.py - Uncertain Owen Values
# Part of OVQX (https://github.com/IrinaTimush1/OVQX), accompanying the B.Sc.
# thesis "Group-Structured Attributions for Explainable Quantum Machine
# Learning" by Irina Timus (Maastricht University, 2026). MIT-licensed.
"""
Owen Values for Quantum Circuit Explanations (OVQXs)

This module implements Owen values as an extension of Shapley values
for explaining quantum circuits with an a priori coalition structure
(partition of gates into groups/modules).

Mathematical notes:
For player i in group C_k, the exact Owen value is

    Ow_i(N, v, C) = sum_{R ⊆ M \\ {k}} sum_{T ⊆ C_k \\ {i}}
                    alpha_k(R) * beta_i(T)
                    * [ v(Q_R ∪ T ∪ {i}) - v(Q_R ∪ T) ]

where
    M = {1, ..., m} is the index set of groups,
    Q_R = union of all groups indexed by R,
    alpha_k(R) = |R|!(m-|R|-1)! / m!,
    beta_i(T)  = |T|!(|C_k|-|T|-1)! / |C_k|!.

Estimator modes:
1) Exact Owen:
   - enumerate all (R, T),
   - compute the doubly weighted sum exactly.

2) Sampled Owen:
   - sample (R^(t), T^(t)) from alpha_k ⊗ beta_i,
   - for each sampled visit t, compute K repeated evaluations of
         v(Q_R^(t) ∪ T^(t))
     and
         v(Q_R^(t) ∪ T^(t) ∪ {i}),
   - average within each visit,
   - average the resulting marginal contributions across visits.

The crucial point is that in sampled mode, repeated visits to the same
coalition are treated as DISTINCT visits with fresh evaluations. This is
the part that must be done to match the sampled product-space estimator.

References:
[1] Owen (1977), "Values of Games with a Priori Unions"
[2] Kotsiopoulos et al. (2024), "Approximation of group explainers with
    coalition structure using Monte Carlo sampling on the product space
    of coalitions and features"
"""

import math
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # noqa: N801
        def __init__(self, *a, **kw):
            self.n = 0
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            self.n += n

from qshaptools.tools import powerset


# Helper: Owen-specific weight functions

def owen_weight_outer(R_size, num_groups):
    """
    Outer (between-group) Shapley weight alpha_k(R).

    Parameters
    ----------
    R_size : int
        Cardinality |R| of the outer coalition.
    num_groups : int
        Number of groups m.

    Returns
    -------
    float
        |R|! (m - |R| - 1)! / m!
    """
    return (
        math.factorial(R_size)
        * math.factorial(num_groups - R_size - 1)
        / math.factorial(num_groups)
    )


def owen_weight_inner(T_size, group_size):
    """
    Inner (within-group) Shapley weight beta_i(T).

    Parameters
    ----------
    T_size : int
        Cardinality |T| of the inner coalition.
    group_size : int
        Size |C_k| of the focal player's group.

    Returns
    -------
    float
        |T|! (|C_k| - |T| - 1)! / |C_k|!
    """
    return (
        math.factorial(T_size)
        * math.factorial(group_size - T_size - 1)
        / math.factorial(group_size)
    )


# ---------------------------------------------------------------------------
# OwenValues class
# ---------------------------------------------------------------------------

class OwenValues:
    """
    Compute Owen values for a cooperative game with a priori coalition structure.

    Design goals
    ------------
    This class mirrors the style of the Shapley implementation used in the
    benchmark toolbox:
      - locked instructions are always appended before evaluation,
      - value functions may be evaluated one-by-one or in batches,
      - exact mode supports coalition caching via self._memory,
      - sampled mode treats each sampled visit independently.

    Important implementation note
    -----------------------------
    The exact path and the sampled path are intentionally separated.

    Exact mode:
      - deduplicate coalitions,
      - optionally cache/reuse coalition values,
      - compute weighted exact Owen sums.

    Sampled mode:
      - DO NOT deduplicate sampled visits,
      - each sampled visit gets fresh evaluations,
      - this is required to match the sampled estimator properly.

    Parameters
    ----------
    unlocked_instructions : list[int]
        Players that are attributed Owen values.
    locked_instructions : list[int] or None
        Players always present in every coalition.
    partition : list[list[int]]
        Partition of unlocked_instructions.
    value_fun : callable
        If owen_batch_size is None:
            value_fun(S=..., **value_kwargs_dict) -> float
        Else:
            value_fun(S_list=..., **value_kwargs_dict) -> list[float]
    value_kwargs_dict : dict
        Extra keyword arguments forwarded to value_fun.
    owen_sample_frac : float | None | negative int
        None      -> exact enumeration
        float > 0 -> sample this fraction of all (R, T) pairs per player
        int < 0   -> sample exactly abs(int) pairs per player
    owen_sample_reps : int
        Number K of repeated evaluations per coalition or per sampled visit.
    owen_batch_size : int | None
        Batch size for value-function evaluation.
    evaluate_value_only_once : bool
        Exact mode only:
            if True, unique coalitions are evaluated once and reused from memory.
        Sampled mode:
            this must remain False to preserve visit-level sampling semantics.
    sample_in_memory : bool
        Kept for interface compatibility.
    owen_sample_seed : int | None
        RNG seed for sampling.
    memory : dict | None
        Coalition cache used in exact mode.
    callback : callable | None
        callback(S) called before value_fun evaluation.
    delta_exponent : int
        Exponent on marginal contributions. Standard Owen uses 1.
    name : str | None
        Progress-bar label.
    silent : bool
        Suppress progress bars.
    """

    def __init__(
        self,
        unlocked_instructions,
        locked_instructions,
        partition,
        value_fun,
        value_kwargs_dict,
        owen_sample_frac=None,
        owen_sample_reps=1,
        owen_batch_size=None,
        evaluate_value_only_once=False,
        sample_in_memory=True,
        owen_sample_seed=None,
        memory=None,
        callback=None,
        delta_exponent=1,
        name=None,
        silent=False,
    ):
        # ---- instructions ------------------------------------------------
        if locked_instructions is None:
            locked_instructions = []
        self._locked_instructions = sorted(list(locked_instructions))
        self._unlocked_instructions = list(unlocked_instructions)

        if len(self._unlocked_instructions) == 0:
            raise ValueError("At least one unlocked instruction is required.")

        # ---- partition ---------------------------------------------------
        self._partition = [list(g) for g in partition]
        self._validate_partition()
        self._num_groups = len(self._partition)
        self._group_indices = list(range(self._num_groups))

        # player -> group lookup
        self._player_to_group = {}
        for g_idx, group in enumerate(self._partition):
            for player in group:
                self._player_to_group[player] = g_idx

        # ---- sampling config ---------------------------------------------
        if owen_sample_frac is not None:
            if isinstance(owen_sample_frac, int) and owen_sample_frac < 0:
                self._owen_sample_abs = abs(owen_sample_frac)
                self._owen_sample_frac_value = None
            elif owen_sample_frac == 0:
                raise ValueError("owen_sample_frac must not be 0.")
            else:
                self._owen_sample_abs = None
                self._owen_sample_frac_value = float(owen_sample_frac)
        else:
            self._owen_sample_abs = None
            self._owen_sample_frac_value = None

        self._use_sampling = owen_sample_frac is not None

        # ---- hyper-parameters --------------------------------------------
        self._owen_sample_reps = max(1, int(owen_sample_reps))
        self._evaluate_value_only_once = bool(evaluate_value_only_once)
        self._sample_in_memory = sample_in_memory
        self._owen_sample_seed = owen_sample_seed
        self._value_fun = value_fun
        self._value_kwargs_dict = value_kwargs_dict
        self._owen_batch_size = owen_batch_size
        self._callback = callback
        self._delta_exponent = delta_exponent
        self._silent = bool(silent)
        self._name = None if name is not None and len(str(name)) == 0 else name

        # ---- memory/cache ------------------------------------------------
        if memory is None:
            memory = {}
        self._memory = memory

        # sampled-mode visit storage:
        # key: (player_i, visit_id, "S") or (player_i, visit_id, "Si")
        # value: list of repeated evaluations for that specific visit
        self._sampled_visit_values = {}

        # In sampled mode, caching/reuse across visits breaks the estimator semantics.
        if self._use_sampling and self._evaluate_value_only_once:
            raise ValueError(
                "In sampled mode, evaluate_value_only_once must be False. "
                "The sampled estimator requires fresh evaluations for each sampled visit."
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_partition(self):
        """Validate that partition is an exact partition of unlocked players."""
        all_players = []
        for group in self._partition:
            all_players.extend(group)

        if sorted(all_players) != sorted(self._unlocked_instructions):
            raise ValueError(
                "partition must be an exact partition of unlocked_instructions.\n"
                f"  unlocked = {sorted(self._unlocked_instructions)}\n"
                f"  partition covers = {sorted(all_players)}"
            )

        if len(all_players) != len(set(all_players)):
            raise ValueError("partition contains duplicate players.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phi_dict(self):
        """Return computed Owen values, or None if not yet computed."""
        return self.phi_dict_ if hasattr(self, "phi_dict_") else None

    @property
    def memory(self):
        """Coalition cache used in exact mode."""
        return self._memory

    @property
    def partition(self):
        return self._partition

    @property
    def name(self):
        return self._name if self._name is not None else "owen"

    # ------------------------------------------------------------------
    # Value-function evaluation
    # ------------------------------------------------------------------

    def _evaluate_value_fun(self, S):
        """
        Evaluate the user-supplied value function.

        Parameters
        ----------
        S : list[int] or list[list[int]]
            If batch mode is off, S is a single coalition.
            If batch mode is on, S is a list of coalitions.

        Returns
        -------
        float or list[float]
        """
        if self._callback is not None:
            self._callback(S)

        if self._owen_batch_size is None:
            return self._value_fun(S=S, **self._value_kwargs_dict)
        else:
            return self._value_fun(S_list=S, **self._value_kwargs_dict)

    # ------------------------------------------------------------------
    # Coalition enumeration helpers
    # ------------------------------------------------------------------

    def _outer_coalitions(self, k):
        """
        Enumerate all outer coalitions R ⊆ M \\ {k}.

        Returns
        -------
        list[tuple[int]]
        """
        other_groups = [g for g in self._group_indices if g != k]
        P, _ = powerset(other_groups)
        return list(P)

    def _inner_coalitions(self, k, i):
        """
        Enumerate all inner coalitions T ⊆ C_k \\ {i}.

        Returns
        -------
        list[tuple[int]]
        """
        group_without_i = [p for p in self._partition[k] if p != i]
        P, _ = powerset(group_without_i)
        return list(P)

    def _build_coalition(self, R, T):
        """
        Build coalition Q_R ∪ T as a sorted tuple of player indices.

        Parameters
        ----------
        R : tuple[int]
            Group indices fully included.
        T : tuple[int]
            Players from the focal player's own group.

        Returns
        -------
        tuple[int]
        """
        players = list(T)
        for r in R:
            players.extend(self._partition[r])
        return tuple(sorted(players))

    # ------------------------------------------------------------------
    # Sampling (R, T) ~ alpha_k ⊗ beta_i
    # ------------------------------------------------------------------

    def _sample_RT_pairs(self, k, i, rng):
        """
        Sample (R, T) pairs from the Owen product distribution.

        Returns
        -------
        pairs : list[tuple[tuple[int], tuple[int]]]
        L : int
            Number of sampled visits.
        """
        all_R = self._outer_coalitions(k)
        all_T = self._inner_coalitions(k, i)

        m = self._num_groups
        gsize = len(self._partition[k])

        R_w = np.array([owen_weight_outer(len(R), m) for R in all_R], dtype=float)
        T_w = np.array([owen_weight_inner(len(T), gsize) for T in all_T], dtype=float)

        R_p = R_w / R_w.sum()
        T_p = T_w / T_w.sum()

        total_pairs = len(all_R) * len(all_T)

        if self._owen_sample_abs is not None:
            L = self._owen_sample_abs
        else:
            L = int(np.ceil(self._owen_sample_frac_value * total_pairs))

        ri = rng.choice(len(all_R), size=L, replace=True, p=R_p)
        ti = rng.choice(len(all_T), size=L, replace=True, p=T_p)

        pairs = [(all_R[r], all_T[t]) for r, t in zip(ri, ti)]
        return pairs, L

    # ------------------------------------------------------------------
    # Phase 1: build evaluation plans
    # ------------------------------------------------------------------

    def _collect_coalitions(self):
        """
        Build evaluation plans for exact or sampled mode.

        Exact mode
        ----------
        self._eval_plan[i] is a list of dicts with keys:
            R, T, S, Si
        and self._all_coalitions stores all unique coalitions needed.

        Sampled mode
        ------------
        self._eval_plan[i] is a list of dicts with keys:
            visit_id, R, T, S, Si

        The important change is that in sampled mode we preserve VISIT identity,
        because repeated visits to the same coalition must remain distinct.
        """
        self._eval_plan = {}
        self._num_pairs = {}
        self._all_coalitions = set()

        desc = f"{self.name}:plan"
        with tqdm(desc=desc, total=len(self._unlocked_instructions), disable=self._silent) as prog:
            for i in self._unlocked_instructions:
                k = self._player_to_group[i]

                if self._use_sampling:
                    pairs, L = self._sample_RT_pairs(k, i, self.rng_)
                    visit_list = []
                    for visit_id, (R, T) in enumerate(pairs):
                        S = self._build_coalition(R, T)
                        Si = tuple(sorted(list(S) + [i]))
                        visit_list.append({
                            "visit_id": visit_id,
                            "R": R,
                            "T": T,
                            "S": S,
                            "Si": Si,
                        })
                    self._eval_plan[i] = visit_list
                    self._num_pairs[i] = L

                else:
                    all_R = self._outer_coalitions(k)
                    all_T = self._inner_coalitions(k, i)
                    exact_items = []

                    for R in all_R:
                        for T in all_T:
                            S = self._build_coalition(R, T)
                            Si = tuple(sorted(list(S) + [i]))
                            exact_items.append({
                                "R": R,
                                "T": T,
                                "S": S,
                                "Si": Si,
                            })
                            self._all_coalitions.add(S)
                            self._all_coalitions.add(Si)

                    self._eval_plan[i] = exact_items
                    self._num_pairs[i] = len(exact_items)

                prog.update(1)

    # ------------------------------------------------------------------
    # Phase 2A: exact mode evaluation with caching/deduplication
    # ------------------------------------------------------------------

    def _evaluate_all_exact(self):
        """
        Evaluate all UNIQUE coalitions needed in exact mode.

        Each coalition is evaluated:
            - once if evaluate_value_only_once=True and not already cached,
            - K times otherwise, where K = owen_sample_reps.

        Results are stored in self._memory under coalition tuple keys.
        """
        locked = self._locked_instructions

        eval_list = []
        for S_tuple in sorted(self._all_coalitions):
            key = S_tuple
            if self._evaluate_value_only_once and key in self._memory:
                continue

            reps = 1 if self._evaluate_value_only_once else self._owen_sample_reps
            for _ in range(reps):
                eval_list.append(list(S_tuple))

        total = len(eval_list)

        # batch path
        if self._owen_batch_size is not None:
            batches = [
                eval_list[j:j + self._owen_batch_size]
                for j in range(0, total, self._owen_batch_size)
            ]

            with tqdm(desc=f"{self.name}:eval", total=total, disable=self._silent) as prog:
                for batch in batches:
                    S_batch = [sorted(s + locked) for s in batch]
                    vals = self._evaluate_value_fun(S_batch)

                    for s, v in zip(batch, vals):
                        key = tuple(sorted(s))
                        self._memory.setdefault(key, []).append([None, v])
                        prog.update(1)
            return

        # sequential path
        with tqdm(desc=f"{self.name}:eval", total=total, disable=self._silent) as prog:
            for s in eval_list:
                sl = sorted(s + locked)
                v = self._evaluate_value_fun(sl)
                key = tuple(sorted(s))
                self._memory.setdefault(key, []).append([None, v])
                prog.update(1)

    # ------------------------------------------------------------------
    # Phase 2B: sampled mode evaluation with VISIT-SPECIFIC storage
    # ------------------------------------------------------------------

    def _evaluate_all_sampled(self):
        """
        Evaluate all sampled visits in sampled mode.

        This is the mathematically important change.

        For each player i and each sampled visit t:
            - evaluate S exactly K fresh times,
            - evaluate Si exactly K fresh times,
            - store these repetitions under (i, visit_id, 'S') and (i, visit_id, 'Si').

        Repeated visits to the same coalition are NOT merged.
        """
        locked = self._locked_instructions
        self._sampled_visit_values = {}

        total = 0
        for i in self._unlocked_instructions:
            total += 2 * self._owen_sample_reps * len(self._eval_plan[i])

        with tqdm(desc=f"{self.name}:eval", total=total, disable=self._silent) as prog:
            # Batch path for sampled mode:
            # to preserve visit identity, we batch only within a given visit.
            if self._owen_batch_size is not None:
                for i in self._unlocked_instructions:
                    for item in self._eval_plan[i]:
                        visit_id = item["visit_id"]
                        S = list(item["S"])
                        Si = list(item["Si"])

                        S_vals = []
                        Si_vals = []

                        # evaluate repeated S values in batches
                        remaining = self._owen_sample_reps
                        while remaining > 0:
                            b = min(self._owen_batch_size, remaining)
                            batch = [sorted(S + locked) for _ in range(b)]
                            vals = self._evaluate_value_fun(batch)
                            S_vals.extend(vals)
                            remaining -= b
                            prog.update(b)

                        # evaluate repeated Si values in batches
                        remaining = self._owen_sample_reps
                        while remaining > 0:
                            b = min(self._owen_batch_size, remaining)
                            batch = [sorted(Si + locked) for _ in range(b)]
                            vals = self._evaluate_value_fun(batch)
                            Si_vals.extend(vals)
                            remaining -= b
                            prog.update(b)

                        self._sampled_visit_values[(i, visit_id, "S")] = S_vals
                        self._sampled_visit_values[(i, visit_id, "Si")] = Si_vals
                return

            # Sequential path
            for i in self._unlocked_instructions:
                for item in self._eval_plan[i]:
                    visit_id = item["visit_id"]
                    S = list(item["S"])
                    Si = list(item["Si"])

                    S_vals = []
                    Si_vals = []

                    for _ in range(self._owen_sample_reps):
                        vS = self._evaluate_value_fun(sorted(S + locked))
                        S_vals.append(vS)
                        prog.update(1)

                    for _ in range(self._owen_sample_reps):
                        vSi = self._evaluate_value_fun(sorted(Si + locked))
                        Si_vals.append(vSi)
                        prog.update(1)

                    self._sampled_visit_values[(i, visit_id, "S")] = S_vals
                    self._sampled_visit_values[(i, visit_id, "Si")] = Si_vals

    # ------------------------------------------------------------------
    # Phase 2 dispatcher
    # ------------------------------------------------------------------

    def _evaluate_all(self):
        """Dispatch to exact or sampled evaluation logic."""
        if self._use_sampling:
            self._evaluate_all_sampled()
        else:
            self._evaluate_all_exact()

    # ------------------------------------------------------------------
    # Phase 3A: exact Owen aggregation
    # ------------------------------------------------------------------

    def _compute_owen_exact(self):
        """
        Compute exact Owen values from cached coalition evaluations.

        For each exact (R, T) term:
            - retrieve mean value of S from memory,
            - retrieve mean value of Si from memory,
            - apply standard Owen weights,
            - accumulate weighted marginal contribution.
        """
        m = self._num_groups
        self.phi_dict_ = {}

        total = sum(self._num_pairs.values())
        with tqdm(desc=f"{self.name}:sums", total=total, disable=self._silent) as prog:
            for i in self._unlocked_instructions:
                k = self._player_to_group[i]
                gsize = len(self._partition[k])

                phi = 0.0

                for item in self._eval_plan[i]:
                    R = item["R"]
                    T = item["T"]
                    S = item["S"]
                    Si = item["Si"]

                    v_S = np.mean([x[1] for x in self._memory[S]])
                    v_Si = np.mean([x[1] for x in self._memory[Si]])

                    d = (v_Si - v_S) ** self._delta_exponent
                    w = (
                        owen_weight_outer(len(R), m)
                        * owen_weight_inner(len(T), gsize)
                    )
                    phi += w * d
                    prog.update(1)

                self.phi_dict_[i] = phi

    # ------------------------------------------------------------------
    # Phase 3B: sampled Owen aggregation
    # ------------------------------------------------------------------

    def _compute_owen_sampled(self):
        """
        Compute sampled Owen values from VISIT-SPECIFIC repeated evaluations.

        For each sampled visit t:
            - average S-values within the visit,
            - average Si-values within the visit,
            - compute visit-level marginal contribution,
            - average these across sampled visits.

        This matches the intended sampled estimator structure.
        """
        self.phi_dict_ = {}

        total = sum(self._num_pairs.values())
        with tqdm(desc=f"{self.name}:sums", total=total, disable=self._silent) as prog:
            for i in self._unlocked_instructions:
                phi = 0.0
                n_terms = 0

                for item in self._eval_plan[i]:
                    visit_id = item["visit_id"]

                    vals_S = self._sampled_visit_values[(i, visit_id, "S")]
                    vals_Si = self._sampled_visit_values[(i, visit_id, "Si")]

                    v_S = float(np.mean(vals_S))
                    v_Si = float(np.mean(vals_Si))

                    d = (v_Si - v_S) ** self._delta_exponent
                    phi += d
                    n_terms += 1
                    prog.update(1)

                self.phi_dict_[i] = phi / n_terms if n_terms > 0 else 0.0

    # ------------------------------------------------------------------
    # Phase 3 dispatcher
    # ------------------------------------------------------------------

    def _compute_owen(self):
        """Dispatch to exact or sampled Owen aggregation."""
        if self._use_sampling:
            self._compute_owen_sampled()
        else:
            self._compute_owen_exact()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self):
        """
        Run the full Owen-value computation pipeline.

        Returns
        -------
        dict[int, float]
            Player -> Owen value
        """
        self.rng_ = np.random.RandomState(self._owen_sample_seed)
        self._collect_coalitions()
        self._evaluate_all()
        self._compute_owen()
        return self.phi_dict

    def run(self):
        """Alias for __call__."""
        return self()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def clear_memory(self):
        """Clear exact-mode coalition cache."""
        self._memory = {}

    def eval_S_list(self, S_list, recall=False):
        """
        Evaluate value functions for an explicit list of coalitions.

        This helper uses exact-mode memory only.
        It is intended mainly for debugging and inspection.
        """
        locked = self._locked_instructions
        values = []

        for S in S_list:
            S = sorted(list(S))
            Sl = sorted(S + locked)
            key = tuple(S)

            if recall and key in self._memory:
                value = np.mean([vi[1] for vi in self._memory[key]])
            else:
                if self._owen_batch_size is None:
                    value = self._evaluate_value_fun(Sl)
                else:
                    value = self._evaluate_value_fun([Sl])[0]

            values.append(value)

        return values

    def get_summary_dict(self, property_list=None):
        """Return a summary of the current configuration and results."""
        if property_list is None:
            property_list = []

        def ga(n):
            return getattr(self, n) if hasattr(self, n) else None

        summary = {
            "name": self.name,
            "value_fun": ga("_value_fun"),
            "unlocked_instructions": ga("_unlocked_instructions"),
            "locked_instructions": ga("_locked_instructions"),
            "partition": ga("_partition"),
            "num_groups": ga("_num_groups"),
            "delta_exponent": ga("_delta_exponent"),
            "owen_sample_frac_value": ga("_owen_sample_frac_value"),
            "owen_sample_abs": ga("_owen_sample_abs"),
            "owen_sample_reps": ga("_owen_sample_reps"),
            "evaluate_value_only_once": ga("_evaluate_value_only_once"),
            "owen_sample_seed": ga("_owen_sample_seed"),
            "owen_batch_size": ga("_owen_batch_size"),
            "num_pairs": ga("_num_pairs"),
            "phi_dict": ga("phi_dict_"),
        }

        for n in property_list:
            summary[n] = ga(n)

        return summary

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    def __str__(self):
        N = len(self._unlocked_instructions)
        M = len(self._locked_instructions)
        m = self._num_groups
        gsizes = [len(g) for g in self._partition]

        rep = f"[{self.name}]\n"
        rep += f"value_fun:                   {self._value_fun}\n"
        rep += f"unlocked_instructions [{N:3d}]: {self._unlocked_instructions}\n"
        rep += f"locked_instructions   [{M:3d}]: {self._locked_instructions}\n"
        rep += f"partition ({m} groups):        {self._partition}\n"
        rep += f"group sizes:                   {gsizes}\n"
        rep += f"delta_exponent:               {self._delta_exponent}\n"
        rep += f"owen_sample_frac_value:       {self._owen_sample_frac_value}\n"
        rep += f"owen_sample_abs:              {self._owen_sample_abs}\n"
        rep += f"owen_sample_reps:             {self._owen_sample_reps}\n"
        rep += f"evaluate_value_only_once:     {self._evaluate_value_only_once}\n"
        rep += f"owen_sample_seed:             {self._owen_sample_seed}\n"
        rep += f"owen_batch_size:              {self._owen_batch_size}\n"

        exact_total = sum(
            2 ** (m - 1) * 2 ** (len(self._partition[self._player_to_group[i]]) - 1)
            for i in self._unlocked_instructions
        )
        rep += f"exact total (R,T) pairs:      {exact_total}\n"

        if hasattr(self, "_num_pairs"):
            used = sum(self._num_pairs.values())
            rep += f"used (R,T) pairs:             {used}"

        return rep
