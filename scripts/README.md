# Script Index

Run scripts from the repository root after `pip install -e .`.

## `benchmark_35/`

- `script_exact_owen_benchmark15.py` - exact Owen values for the 15-circuit
  subset; produces Table I, Table II, and Fig. 2 outputs.
- `script_faithfulness_ablation_benchmark15.py` - ablation faithfulness checks
  for the 15-circuit subset.
- `script_estimator_validation_benchmark15.py` - Monte Carlo Owen estimator
  validation over sampling fractions.
- `script_plot_estimator_validation.py` - paper-style estimator-validation
  plots.
- `script_full35_estimated_study.py` - sampled Owen values for all 35 benchmark
  circuits and the thesis Fig. 4 dominant-coalition map.
- `script_plot_full35_study.py` - auxiliary full-35 exploratory plots.
- `script_plot_full35_dominant_positions.py` - standalone re-renderer for the
  Fig. 4 dominant-coalition map.

## `qnn/`

- `reproduce_qnn_owen.py` - main QNN E/M/X Owen experiment for Fig. 5, Fig. 8,
  Fig. 9, and Table VIII.
- `reproduce_qnn_owen_multiple_partitions.py` - alternative QNN partitions for
  Table X.
- `reproduce_qnn_svqx.py` - SVQX gate-level baseline used for comparison.

## `qsvm/`

- `qsvm_experiment_utils.py` - shared QSVM data, circuit, Owen, aggregation,
  and plotting helpers.
- `run_qsvm_owen_all_datasets.py` - main multi-dataset QSVM runner.
- `aggregate_qsvm_owen_all_datasets.py` - aggregate QSVM outputs across the
  five splits.
- `plot_qsvm_paper_style.py` - paper-style QSVM figures and statistics.
- `reproduce_qsvm_accuracies.py` - full-circuit QSVM accuracy check.
- `README.md` - QSVM-specific workflow notes and output layout.

## `tests/`

- `test_owen.py` - classical Owen-value axiom tests.
- `test_qowen.py` - quantum Owen/Shapley consistency checks.
- `test_quantum_owen_workflow.py` - end-to-end quantum Owen workflow test.
