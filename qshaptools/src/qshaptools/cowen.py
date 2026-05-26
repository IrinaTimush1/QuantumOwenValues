# cowen.py - Classical Owen Values
# Part of OVQX (https://github.com/IrinaTimush1/OVQX), accompanying the B.Sc.
# thesis "Group-Structured Attributions for Explainable Quantum Machine
# Learning" by Irina Timus (Maastricht University, 2026). MIT-licensed.
"""
Classical Owen Values
=====================

Thin wrapper around :class:`uowen.OwenValues` for *non-quantum* cooperative
games.  This mirrors :class:`cshap.ClassicalShapleyValues` so that the
interface is consistent.

Usage
-----
>>> from cowen import ClassicalOwenValues
>>> def v(S, **kw):
...     return len(S)        # simple additive game
>>> ov = ClassicalOwenValues(
...     N=4,
...     partition=[[0,1],[2,3]],
...     value_fun=v,
... )
>>> ov.run()
"""

from qshaptools.uowen import OwenValues


class ClassicalOwenValues(OwenValues):
    """
    Owen values for a classical (non-quantum) cooperative game.

    Parameters
    ----------
    N : int
        Number of players.  Players are indexed ``0, 1, ..., N-1``.
    partition : list[list[int]]
        A priori coalition structure — a partition of ``range(N)``.
    value_fun : callable
        ``value_fun(S=..., **value_kwargs_dict) -> float``
        where *S* is a sorted list of player indices.
    value_kwargs_dict : dict
        Extra keyword arguments forwarded to every ``value_fun`` call.
    owen_sample_frac : float | None | negative int
        See :class:`uowen.OwenValues`.
    owen_sample_reps : int
        See :class:`uowen.OwenValues`.
    evaluate_value_only_once : bool
        See :class:`uowen.OwenValues`.
    sample_in_memory : bool
        Kept for interface compatibility.
    owen_sample_seed : int | None
        Random seed for coalition sampling.
    owen_batch_size : int | None
        Batch-evaluate value functions.
    memory : dict | None
        Pre-existing memory cache.
    callback : callable | None
        Called before each evaluation.
    delta_exponent : int
        Exponent on marginal contributions (1 = Owen values).
    name : str | None
        Label for progress bars.
    silent : bool
        Suppress progress bars.
    """

    def __init__(
        self,
        N,
        partition,
        value_fun,
        value_kwargs_dict=None,
        owen_sample_frac=None,
        owen_sample_reps=1,
        evaluate_value_only_once=False,
        sample_in_memory=True,
        owen_sample_seed=None,
        owen_batch_size=None,
        memory=None,
        callback=None,
        delta_exponent=1,
        name=None,
        silent=False,
    ):
        if value_kwargs_dict is None:
            value_kwargs_dict = {}
        self._N = int(N)
        unlocked_instructions = list(range(self._N))
        locked_instructions = []

        super().__init__(
            unlocked_instructions=unlocked_instructions,
            locked_instructions=locked_instructions,
            partition=partition,
            value_fun=value_fun,
            value_kwargs_dict=value_kwargs_dict,
            owen_sample_frac=owen_sample_frac,
            owen_sample_reps=owen_sample_reps,
            owen_batch_size=owen_batch_size,
            evaluate_value_only_once=evaluate_value_only_once,
            sample_in_memory=sample_in_memory,
            owen_sample_seed=owen_sample_seed,
            memory=memory,
            callback=callback,
            delta_exponent=delta_exponent,
            name=name,
            silent=silent,
        )

    def get_summary_dict(self, property_list=None):
        if property_list is None:
            property_list = []
        summary = super().get_summary_dict(property_list)
        summary["N"] = getattr(self, "_N", None)
        return summary
