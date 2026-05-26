# Data Files

`data/` contains only input data used by the thesis scripts.

- `benchmark_35_from_pool.pkl` - the 35 selected three-qubit benchmark circuits,
  sampled from a 10,000-circuit random pool.
- `benchmark_35_from_pool_summary.csv` - normalized magic and entanglement
  labels, family roles, and metadata for the selected 35 circuits.
- `benchmark_35_gate_spec.csv` - manual E/M/X gate-level annotations for the
  full 35-circuit estimated Owen study.
- `benchmark_15_exact_gate_spec.csv` - manual E/M/X gate-level annotations for
  the 15-circuit exact Owen subset.
- `qnn-data.csv` - the 20-point binary classification dataset from Heese et
  al. used in the QNN case study.
- `qsvm-traindata-{0..4}.csv` - five QSVM training splits, with 40 points each.
- `qsvm-testdata-{0..4}.csv` - matching QSVM test splits, with 1000 points each.

Generated figures and tables are kept under `results/`.
