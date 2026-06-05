# OVQX - Owen Values for Quantum Circuit Explanations

Code accompanying the B.Sc. thesis **"Group-Structured Attributions for
Explainable Quantum Machine Learning"** by Irina Timus, Department of Advanced
Computing Sciences, Maastricht University, 2026.

Supervisors: Menica Dibenedetto and Tjitze Rienstra.

## What this repository contains

OVQX extends the SVQX framework of Heese et al. from gate-level Shapley
attributions to group-structured Owen attributions over predefined gate
partitions. The repository reproduces:

- the controlled 35-circuit benchmark (Fig. 1) and the 15-circuit exact Owen
  study with group-ablation support (Table I, Fig. 2, Table II);
- the label-aware random-partition control (Tables VII-VIII);
- the Monte Carlo estimator validation (Fig. 6, Table IX) and the full
  35-circuit estimated study (Fig. 3);
- the QNN case study with E/M/X and alternative partitions (Fig. 4, Figs 8-9,
  Tables X, XII, XIII);
- the QSVM case study with E/M/X and alternative partitions (Fig. 5,
  Tables XIV-XVI, Table XVII);
- the locked-passive Shapley comparison used in Appendix A-G (Table XI,
  Figs 10-13).

The thesis PDF is included as `thesis.pdf`.

## Repository layout

```text
QuantumOwenValues/
├── qshaptools/                # upstream qshaptools plus OVQX Owen extensions
├── data/                      # input CSV/PKL data only
├── scripts/
│   ├── benchmark_35/          # 35-circuit benchmark, estimator, ablation, random-partition control
│   ├── qnn/                   # QNN Owen, alternative partitions, and SVQX baseline
│   ├── qsvm/                  # QSVM Owen runners, alternative partitions, aggregation, plots
│   └── tests/                 # Owen-value verification scripts
├── shapley_reproduction/      # locked-passive Shapley baseline for Owen-vs-Shapley comparison
├── results/                   # pruned thesis outputs and figures
└── thesis.pdf
```

## Installation

The GitHub repository for this thesis artifact is private.

### From the submitted ZIP archive

```bash
unzip QuantumOwenValues.zip
cd QuantumOwenValues          # or the extracted folder name, e.g. QuantumOwenValues-main
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="$PWD/qshaptools/src:$PWD/qshaptools/src/qshaptools:$PYTHONPATH"
pip install -r requirements.txt
pip install -e .
```

### From GitHub, if access has been granted

```bash
git clone https://github.com/IrinaTimush1/QuantumOwenValues.git
cd QuantumOwenValues
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="$PWD/qshaptools/src:$PWD/qshaptools/src/qshaptools:$PYTHONPATH"
pip install -r requirements.txt
pip install -e .
```

The pinned environment uses Qiskit 0.46 APIs such as `Aer` and
`QuantumInstance`; Qiskit 1.0 or newer is not compatible with these scripts.
Use Python 3.12 for the submitted environment. Installing from the ZIP still
requires ordinary `pip` access to the packages listed in `requirements.txt`;
it does not require access to the private GitHub repository.

## Reproducing the thesis figures

Run commands from the repository root. By default, outputs are written under
`results/`; most scripts also accept an explicit `--output-dir`.

### Fig. 1 - Benchmark plane

The selected benchmark is stored in `data/benchmark_35_from_pool.pkl` and
`data/benchmark_35_from_pool_summary.csv`. The saved thesis figure is at:

```text
results/benchmark_35/benchmark_35_normalized.png
```

### Table I, Fig. 2, Table II - 15-circuit exact Owen and group ablation

```bash
python scripts/benchmark_35/script_exact_owen_benchmark15.py
python scripts/benchmark_35/script_faithfulness_ablation_benchmark15.py
```

The second filename retains the earlier `faithfulness_ablation` name for
backward compatibility; thesis Section V-B calls this experiment
"Group-Ablation Support".

### Tables VII-VIII, Appendix A-B - Label-aware random-partition control

```bash
python scripts/benchmark_35/script_random_partition_control_benchmark15.py \
    --n-random 100 --seed 12345
```

This writes generic dominance/faithfulness metrics and label-aware
resource-alignment controls to `results/random_partition_control_15/`.
Tables VII-VIII use the `random_control_summary.csv` rows for
`magic/resource_alignment`, `magic/expected_drop_norm`,
`entanglement/resource_alignment`, and `entanglement/expected_drop_norm`.

### Fig. 6, Table IX, Appendix A-C - Monte Carlo estimator validation

```bash
python scripts/benchmark_35/script_estimator_validation_benchmark15.py \
    --sample-fracs 0.3 0.5 0.7 0.9 --seeds-per-fraction 10
python scripts/benchmark_35/script_plot_estimator_validation.py
```

### Fig. 3 - Full 35-circuit estimated Owen

```bash
python scripts/benchmark_35/script_full35_estimated_study.py \
    --sample-frac 0.7 --repeats 5
```

This writes the thesis Fig. 3 dominant-coalition plot directly to
`results/estimated_full35_frac70/fig_dominant_group_positions.png`.

### Fig. 4, Fig. 8, Fig. 9, Table X - QNN E/M/X

```bash
python scripts/qnn/reproduce_qnn_owen.py \
    --data-path data/qnn-data.csv \
    --use-paper-theta \
    --k-values 1 8 16 32 --num-runs 5 --alpha 1.0 \
    --output-dir results/qnn_owen_emx_main
```

This command produces the QNN main bar chart, K-comparison, gate-level figure,
and Table X outputs.

### Tables XII-XIII, Appendix A-H - Alternative QNN partitions

```bash
python scripts/qnn/reproduce_qnn_owen_multiple_partitions.py \
    --data-path data/qnn-data.csv \
    --use-paper-theta \
    --output-dir results/qnn_owen_partitions
```

### Fig. 5, Tables XIV-XVI - QSVM E/M/X

```bash
python scripts/qsvm/run_qsvm_owen_all_datasets.py \
    --data-dir data --output-dir results/qsvm_owen_all_datasets
python scripts/qsvm/aggregate_qsvm_owen_all_datasets.py \
    --output-dir results/qsvm_owen_all_datasets
python scripts/qsvm/plot_qsvm_paper_style.py \
    --results-dir results/qsvm_owen_all_datasets
```

### Table XVII, Appendix A-K - Alternative QSVM partitions

```bash
python scripts/qsvm/run_qsvm_owen_alternative_partitions.py \
    --data-dir data \
    --output-dir results/qsvm_owen_alternative_partitions \
    --r-values 1 2 3
```

This writes appendix-ready comparisons for `feature_semantics` (F1/F2/F12)
and `repetition_blocks` (B1/B2/B3).

### Table XI, Figs 10-13, Appendix A-G - Locked-passive Shapley comparison

```bash
python shapley_reproduction/run_qnn_shapley.py
python shapley_reproduction/run_qsvm_shapley.py
python shapley_reproduction/compare_owen_vs_shapley.py
```

These commands write to `shapley_reproduction/results/`. The comparison script
produces Table XI and Figs 10-13. This folder is intentionally separate because
the standalone Shapley estimator does not import from the Owen modules, as
documented in `shapley_reproduction/README.md`.

### SVQX baseline (Discussion Section VI)

```bash
python scripts/qnn/reproduce_qnn_svqx.py \
    --data-path data/qnn-data.csv \
    --use-paper-theta \
    --output-dir results/qnn_svqx
```

## Verification

```bash
python scripts/tests/test_owen.py
python scripts/tests/test_qowen.py
python scripts/tests/test_quantum_owen_workflow.py
```

These scripts check Owen-value axioms, reduction to Shapley values under
singleton and grand partitions, null-player behavior, and sampled-estimator
convergence on small deterministic games.

## Runtime expectations

The committed `results/` directory contains the pruned outputs used in the
thesis. Approximate runtimes on a 2026 Apple Silicon laptop are:

- Quick tests: under 2 minutes.
- QNN E/M/X reproduction (`K=1,8,16,32`, 5 runs, `n_jobs=5`): about 8 minutes.
- QSVM full reproduction over five dataset splits and `r=1,2,3`: about 5-15 minutes depending on cache state.
- Full 35-circuit estimated study (`sample_frac=0.7`, 5 repeats): about 1.5-2 hours.

Exact timings vary with CPU, simulator version, and parallelism settings.

## License and attribution

This repository is MIT-licensed. It builds on `qshaptools` by Heese et al.,
also MIT-licensed. See `LICENSE`, `NOTICE.md`, and `qshaptools/LICENSE` for
the full attribution trail.

## Citing this work

If you use this code, please cite:

> Timus, I. (2026). *Group-Structured Attributions for Explainable Quantum
> Machine Learning* (B.Sc. thesis). Maastricht University.

`CITATION.cff` is included for reference managers.
