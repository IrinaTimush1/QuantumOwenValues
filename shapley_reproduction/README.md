# Shapley Reproduction

This folder computes gate-level Shapley baselines for the same locked-passive
active-player games used by the thesis Owen-value experiments, for comparison
against the Owen-value results in the thesis repository.

The scripts are intentionally separate from the main Owen code. They add no
imports or edits to `qshaptools/`, `scripts/`, or `analysis/`.

## What Is Reproduced

- QNN locked-passive setting: active gates
  `{2,4,5,6,7,9,11,12,13,14,15,16,17,18,19}`, passive gates `{1,3,8,10}`,
  paper theta `(3.860, -1.070, -1.583, 0.860)`, one-shot accuracy, `K=32`,
  `alpha=0.01`, seeds `0..4`.
- QSVM locked-passive setting: in each repetition, gates `1+7(j-1)` and
  `3+7(j-1)` are passive Hadamards; all remaining gates are Shapley players.
  The default dataset is
  `qsvm-traindata-0.csv` / `qsvm-testdata-0.csv`, which matches the Heese
  reference accuracies in the cleaned thesis repository.

The direct estimator is in `shapley_estimator.py`. For `alpha=1` it enumerates
all contexts. For `alpha<1` it samples contexts by choosing the coalition size
uniformly and then a subset uniformly within that size class.

## Commands

Run from the repository root after installing the thesis requirements:

```bash
python shapley_reproduction/run_qnn_shapley.py
python shapley_reproduction/run_qsvm_shapley.py
python shapley_reproduction/compare_owen_vs_shapley.py
```

Outputs are written under `shapley_reproduction/results/`:

- `qnn_shapley_per_gate.csv`
- `qnn_shapley_summary.csv`
- `qsvm_shapley_per_gate.csv`
- `qsvm_shapley_summary.csv`
- `owen_vs_shapley_correlation.csv`
- figures under `shapley_reproduction/results/figures/`

## Sanity Checks

The runners print full-circuit accuracy checks before estimating Shapley
values:

- QNN reference: `0.800`
- QSVM references: `r=1: 0.842`, `r=2: 0.986`, `r=3: 0.913`

If a check prints `MISMATCH`, the loaded data or value-function convention is
not exactly the Heese convention; the produced Shapley values are then still
valid for the local dataset, but not a literal reproduction of the published
figure.

## Runtime Notes

Approximate runtimes depend strongly on CPU and scikit-learn version. On the
local Apple Silicon environment used while adding this folder:

- QNN locked-passive reproduction: about 1.2 minutes.
- QSVM `r=1` exact: below 1 second for the estimator.
- QSVM `r=2` exact: below 1 second for the estimator.
- QSVM `r=3`, `alpha=0.01`, two runs: about 4 seconds for the estimator.

For a quick QSVM smoke test, restrict the run:

```bash
python shapley_reproduction/run_qsvm_shapley.py --r-values 1
```
