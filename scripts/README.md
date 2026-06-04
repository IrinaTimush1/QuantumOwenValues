# Script Index

Run scripts from the repository root after `pip install -e .`.

## `benchmark_35/`

- `script_exact_owen_benchmark15.py` - exact Owen values on the 15-circuit
  subset; produces Table I and feeds Fig. 2 and Table II.
- `script_faithfulness_ablation_benchmark15.py` - group-ablation experiment
  (Fig. 2, Table II). The legacy filename is kept; thesis Section V-B calls
  this "Group-Ablation Support".
- `script_random_partition_control_benchmark15.py` - label-aware
  random-partition control on the 15-circuit subset (Tables VII-VIII,
  Appendix A-B).
- `script_estimator_validation_benchmark15.py` - Monte Carlo Owen estimator
  validation over sampling fractions (feeds Table IX).
- `script_plot_estimator_validation.py` - paper-style estimator-validation
  plots (Fig. 6).
- `script_full35_estimated_study.py` - sampled Owen values for all 35
  benchmark circuits and the Fig. 3 dominant-coalition map.
- `script_plot_full35_study.py` - auxiliary full-35 exploratory plots.
- `script_plot_full35_dominant_positions.py` - standalone re-renderer for
  the Fig. 3 dominant-coalition map.

## `qnn/`

- `reproduce_qnn_owen.py` - main QNN E/M/X Owen experiment (Fig. 4, Fig. 8,
  Fig. 9, Table X).
- `reproduce_qnn_owen_multiple_partitions.py` - alternative QNN partitions
  (Tables XII, XIII; Appendix A-H).
- `reproduce_qnn_svqx.py` - SVQX gate-level baseline used in Discussion
  Section VI.

## `qsvm/`

- `qsvm_experiment_utils.py` - shared QSVM data, circuit, Owen, aggregation,
  and plotting helpers.
- `run_qsvm_owen_all_datasets.py` - main multi-dataset QSVM E/M/X runner
  (Fig. 5, Tables XIV-XVI).
- `aggregate_qsvm_owen_all_datasets.py` - aggregate QSVM outputs across the
  five splits.
- `plot_qsvm_paper_style.py` - paper-style QSVM figures and statistics.
- `reproduce_qsvm_accuracies.py` - full-circuit QSVM accuracy check
  (Table XV setup check).
- `run_qsvm_owen_alternative_partitions.py` - feature-semantics and
  repetition-block partitions (Table XVII, Appendix A-K).

## `tests/`

- `test_owen.py` - classical Owen-value axiom tests (47 cases).
- `test_qowen.py` - quantum Owen / Shapley consistency checks (12 cases).
- `test_quantum_owen_workflow.py` - end-to-end quantum Owen workflow test.

## Shapley reproduction

The locked-passive Shapley baseline used in thesis Appendix A-G (Table XI,
Figs 10-13) lives in the separate top-level folder `shapley_reproduction/`,
not under `scripts/`. See `shapley_reproduction/README.md` for invocation
details.
