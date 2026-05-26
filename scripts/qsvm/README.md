# QSVM Scripts

Run all commands from the repository root after installing the environment with
`pip install -r requirements.txt` and `pip install -e .`.

## Thesis Workflow

The QSVM thesis results are reproduced by three scripts:

```bash
python scripts/qsvm/run_qsvm_owen_all_datasets.py \
    --data-dir data \
    --output-dir results/qsvm_owen_all_datasets

python scripts/qsvm/aggregate_qsvm_owen_all_datasets.py \
    --output-dir results/qsvm_owen_all_datasets

python scripts/qsvm/plot_qsvm_paper_style.py \
    --results-dir results/qsvm_owen_all_datasets
```

This produces the outputs used for Fig. 6 and Tables XI-XIII.

## Script Index

- `run_qsvm_owen_all_datasets.py` is the canonical runner. It computes exact
  Owen values for dataset splits `0..4` and repetition depths `r=1,2,3`.
- `aggregate_qsvm_owen_all_datasets.py` collects the per-split outputs into
  aggregate mean/std CSVs and auxiliary aggregate plots.
- `plot_qsvm_paper_style.py` creates the paper-style QSVM figures and
  statistics tables from the committed result folders.
- `reproduce_qsvm_accuracies.py` is an optional full-circuit accuracy sanity
  check for the unablated QSVM circuits. It does not compute Owen values.
- `qsvm_experiment_utils.py` contains the shared implementation: data loading,
  feature-map construction, exact statevector kernel evaluation, value
  function caching, Owen execution, aggregation helpers, and plotting helpers.

Optional full-circuit accuracy check:

```bash
python scripts/qsvm/reproduce_qsvm_accuracies.py \
    --data-dir data \
    --output-dir results/qsvm_owen_all_datasets
```

Older single-dataset wrappers were removed. Their functionality is covered by
`run_qsvm_owen_all_datasets.py`; for a small targeted run, use for example:

```bash
python scripts/qsvm/run_qsvm_owen_all_datasets.py \
    --data-indices 0 \
    --r-values 1 \
    --data-dir data \
    --output-dir /tmp/qsvm_smoke
```

## Output Layout

`results/qsvm_owen_all_datasets/` contains:

- `dataset_<i>/r<r>/gate_owen_values.csv`
- `dataset_<i>/r<r>/group_owen_values.csv`
- `dataset_<i>/r<r>/coalition_values_exact_owen.csv`
- `dataset_<i>/r<r>/qsvm_value_cache.csv`
- `dataset_<i>/r<r>/summary.json`
- `all_run_summaries.csv`
- `aggregate/` outputs from `aggregate_qsvm_owen_all_datasets.py`
- `paperstyle_plots/` outputs from `plot_qsvm_paper_style.py`
