# OVQX - Owen Values for Quantum Circuit Explanations

Code accompanying the B.Sc. thesis **"Group-Structured Attributions for
Explainable Quantum Machine Learning"** by Irina Timus, Department of Advanced
Computing Sciences, Maastricht University, 2026.

Supervisors: Menica Dibenedetto and Tjitze Rienstra.

## What this repository contains

OVQX extends the SVQX framework of Heese et al. from gate-level Shapley
attributions to group-structured Owen attributions over predefined gate
partitions. The repository reproduces the controlled 35-circuit benchmark, the
15-circuit exact Owen study and faithfulness ablation, the Monte Carlo estimator
validation, the full 35-circuit estimated study, and the QNN/QSVM case studies.

The thesis PDF is included as `thesis.pdf`.

## Repository layout

```text
OVQX/
├── qshaptools/        # upstream qshaptools plus OVQX Owen extensions
├── data/              # input CSV/PKL data only
├── scripts/
│   ├── benchmark_35/  # 35-circuit benchmark, estimator, faithfulness scripts
│   ├── qnn/           # QNN Owen, alternative partitions, and SVQX baseline
│   ├── qsvm/          # QSVM Owen runners, aggregation, and plots
│   └── tests/         # Owen-value verification scripts
├── results/           # pruned thesis outputs and figures
└── thesis.pdf
```

## Installation

```bash
git clone https://github.com/IrinaTimush1/QuantumOwenValues.git
cd OVQX
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The pinned environment uses Qiskit 0.46 APIs such as `Aer` and
`QuantumInstance`; Qiskit 1.0 or newer is not compatible with these scripts.

## Reproducing the thesis figures

Run commands from the repository root. By default, outputs are written under
`results/`; most scripts also accept an explicit `--output-dir`.

### Fig. 1 - Benchmark plane

The selected benchmark is stored in `data/benchmark_35_from_pool.pkl` and
`data/benchmark_35_from_pool_summary.csv`. The saved thesis figure is at:

```text
results/benchmark_35/benchmark_35_normalized.png
```

### Table I, Fig. 2, Table II - 15-circuit exact Owen and faithfulness

```bash
python scripts/benchmark_35/script_exact_owen_benchmark15.py
python scripts/benchmark_35/script_faithfulness_ablation_benchmark15.py
```

### Fig. 3, Table VII - Estimator validation

```bash
python scripts/benchmark_35/script_estimator_validation_benchmark15.py \
    --sample-fracs 0.3 0.5 0.7 0.9 --seeds-per-fraction 10
python scripts/benchmark_35/script_plot_estimator_validation.py
```

### Fig. 4 - Full 35-circuit estimated Owen

```bash
python scripts/benchmark_35/script_full35_estimated_study.py \
    --sample-frac 0.7 --repeats 5
python scripts/benchmark_35/script_plot_full35_dominant_positions.py
```

### Fig. 5, Fig. 8, Fig. 9, Table VIII - QNN E/M/X

```bash
python scripts/qnn/reproduce_qnn_owen.py \
    --data-path data/qnn-data.csv \
    --use-paper-theta \
    --k-values 1 8 16 32 --num-runs 5 --alpha 1.0 \
    --output-dir results/qnn_owen_emx_main
```

### Table X - Alternative QNN partitions

```bash
python scripts/qnn/reproduce_qnn_owen_multiple_partitions.py \
    --data-path data/qnn-data.csv \
    --use-paper-theta \
    --output-dir results/qnn_owen_partitions
```

### Fig. 6, Tables XI-XIII - QSVM E/M/X

```bash
python scripts/qsvm/run_qsvm_owen_all_datasets.py \
    --data-dir data --output-dir results/qsvm_owen_all_datasets
python scripts/qsvm/aggregate_qsvm_owen_all_datasets.py
python scripts/qsvm/plot_qsvm_paper_style.py
```

### SVQX baseline

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
thesis. The unit tests run quickly. The full 35-circuit, QNN, and QSVM
reproduction commands are computational experiments and can take from minutes
to several hours depending on CPU, simulator, and parallelism settings.

## License and attribution

This repository is MIT-licensed. It builds on `qshaptools` by Heese et al.,
also MIT-licensed. See `LICENSE`, `NOTICE.md`, and `qshaptools/LICENSE` for
the full attribution trail.

## Citing this work

If you use this code, please cite:

> Timus, I. (2026). *Group-Structured Attributions for Explainable Quantum
> Machine Learning* (B.Sc. thesis). Maastricht University.

`CITATION.cff` is included for reference managers.
