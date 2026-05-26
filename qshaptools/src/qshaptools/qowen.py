# qowen.py - Quantum Owen Values
# Part of OVQX (https://github.com/IrinaTimush1/QuantumOwenValues), accompanying the B.Sc.
# thesis "Group-Structured Attributions for Explainable Quantum Machine
# Learning" by Irina Timus (Maastricht University, 2026). MIT-licensed.
"""
Quantum Owen Values
===================

Wrapper around :class:`uowen.OwenValues` for *quantum-circuit* cooperative
games.  This mirrors :class:`qshap.QuantumShapleyValues` so that:

  * a Qiskit ``QuantumCircuit`` is accepted directly,
  * gates are extracted via ``tools.extract_from_circuit``,
  * locked gates are excluded from the game,
  * the ``partition`` is defined over *unlocked* gate indices.

Usage
-----
>>> from qowen import QuantumOwenValues
>>> qov = QuantumOwenValues(
...     qc=my_circuit,
...     partition=[[0,1,2], [3,4,5]],    # groups of unlocked gate indices
...     value_fun=my_value_function,
...     value_kwargs_dict={...},
...     quantum_instance=qi,
... )
>>> qov.run()
"""
from qshaptools.tools import extract_from_circuit
from qshaptools.uowen import OwenValues


class QuantumOwenValues(OwenValues):
    """
    Owen values for a quantum circuit with modular structure.

    Parameters
    ----------
    qc : qiskit.circuit.QuantumCircuit
        Quantum circuit of interest.
    partition : list[list[int]]
        A priori coalition structure over the *unlocked* gate indices.
        Each inner list groups gate indices that form a logical module.
    value_fun : callable
        ``(qc_data, num_qubits, S, quantum_instance, **kw) -> float``
        (single) or ``(qc_data, num_qubits, S_list, quantum_instance,
        **kw) -> list[float]`` (batch).
    value_kwargs_dict : dict
        Extra keyword arguments forwarded to ``value_fun``.
    quantum_instance : qiskit.utils.QuantumInstance
        Backend wrapper passed to ``value_fun``.
    owen_sample_frac : float | None | negative int
        Coalition-sampling control (see ``uowen.OwenValues``).
    owen_sample_reps : int
        Noisy-repetition count K.
    evaluate_value_only_once : bool
        De-duplicate coalition evaluations.
    sample_in_memory : bool
        Kept for interface compatibility.
    owen_sample_seed : int | None
        Random seed.
    owen_batch_size : int | None
        Batch evaluation size.
    qc_preprocessing_fun : callable | None
        ``qc -> qc`` preprocessing of the circuit.
    locked_instructions : list | None
        Gate indices that are always active (not part of the game).
    memory : dict | None
        Pre-existing memory cache.
    callback : callable | None
        Called before each evaluation.
    delta_exponent : int
        Exponent on marginal contributions.
    name : str | None
        Label for progress bars.
    silent : bool
        Suppress progress bars.
    """

    def __init__(
        self,
        qc,
        partition,
        value_fun,
        value_kwargs_dict,
        quantum_instance,
        owen_sample_frac=None,
        owen_sample_reps=1,
        evaluate_value_only_once=False,
        sample_in_memory=True,
        owen_sample_seed=None,
        owen_batch_size=None,
        qc_preprocessing_fun=None,
        locked_instructions=None,
        memory=None,
        callback=None,
        delta_exponent=1,
        name=None,
        silent=False,
    ):
        # ---- preprocess circuit ------------------------------------------
        self._qc_preprocessing_fun = qc_preprocessing_fun
        if self._qc_preprocessing_fun is not None:
            qc = self._qc_preprocessing_fun(qc)
        self._num_qubits, self._qc_data = extract_from_circuit(
            qc, locked_instructions
        )

        # identify unlocked gate indices
        unlocked_instructions = [
            idx
            for idx, (instr, qargs, cargs, opts) in enumerate(self._qc_data)
            if not opts["lock"]
        ]

        # ---- value-function kwargs (same pattern as qshap.py) ------------
        self._quantum_instance = quantum_instance
        effective_kwargs = {
            "qc_data": self._qc_data,
            "num_qubits": self._num_qubits,
            "quantum_instance": self._quantum_instance,
        }
        effective_kwargs.update(value_kwargs_dict)

        # ---- initialise base class ---------------------------------------
        super().__init__(
            unlocked_instructions=unlocked_instructions,
            locked_instructions=locked_instructions,
            partition=partition,
            value_fun=value_fun,
            value_kwargs_dict=effective_kwargs,
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

    # convenience methods (same signatures as qshap.py) --------------------

    def run(self):
        """Evaluate Owen values.  Alias for ``__call__``."""
        return self()

    def get_values(self, S_list, recall=False):
        """Evaluate value functions for explicit coalitions."""
        return self.eval_S_list(S_list, recall)

    def disp(self):
        """Print settings."""
        print(self.__str__())

    def get_summary_dict(self, property_list=None):
        if property_list is None:
            property_list = []

        def ga(n):
            return getattr(self, n) if hasattr(self, n) else None

        summary = super().get_summary_dict(property_list)
        summary.update(
            {
                "quantum_instance": ga("_quantum_instance"),
                "qc_preprocessing_fun": ga("_qc_preprocessing_fun"),
                "num_qubits": ga("_num_qubits"),
            }
        )
        return summary
