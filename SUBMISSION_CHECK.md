# Submission Check

Generated on 2026-05-27 from a fresh Python 3.12 virtual environment outside
the repository:

```bash
python3.12 -m venv /tmp/ovqx-submission-check-venv
source /tmp/ovqx-submission-check-venv/bin/activate
export PYTHONPATH="$PWD/qshaptools/src:$PWD/qshaptools/src/qshaptools:$PYTHONPATH"
```

## Environment

Command:

```bash
python --version
```

Output:

```text
Python 3.12.13
```

## Install

Command:

```bash
pip install -r requirements.txt
```

Output excerpt:

```text
Successfully installed contourpy-1.3.3 cycler-0.12.1 dill-0.4.1 fonttools-4.63.0 joblib-1.5.3 kiwisolver-1.5.0 matplotlib-3.10.8 mpmath-1.3.0 numpy-2.4.2 packaging-26.2 pandas-3.0.1 pillow-12.2.0 ply-3.11 psutil-7.2.2 pyparsing-3.3.2 python-dateutil-2.9.0.post0 qiskit-0.46.3 qiskit-aer-0.14.2 qiskit-terra-0.46.3 rustworkx-0.17.1 scikit-learn-1.8.0 scipy-1.17.1 six-1.17.0 stevedore-5.8.0 symengine-0.14.1 sympy-1.14.0 threadpoolctl-3.6.0 tqdm-4.67.3
```

Command:

```bash
pip install -e .
```

Output excerpt:

```text
Successfully built ovqx
Successfully installed ovqx-0.1.0
```

Command:

```bash
pip check
```

Output:

```text
No broken requirements found.
```

## Tests

Command:

```bash
python scripts/tests/test_owen.py
```

Output excerpt:

```text
Results:  47 passed,  0 failed
```

Command:

```bash
python scripts/tests/test_qowen.py
```

Output excerpt:

```text
Results: 12 passed, 0 failed
```

Command:

```bash
python scripts/tests/test_quantum_owen_workflow.py
```

Output excerpt:

```text
========================================================================
DONE
========================================================================
```

## QSVM Smoke Test

Command:

```bash
python scripts/qsvm/run_qsvm_owen_all_datasets.py \
    --data-indices 0 \
    --r-values 1 \
    --data-dir data \
    --output-dir /tmp/qsvm_smoke
```

Output excerpt:

```text
Running QSVM Owen: dataset=0, r=1
Full active accuracy: 0.8420
Empty-active H-only accuracy: 0.5000
Sum Owen values: 0.34200000; full-empty: 0.34200000
Efficiency gap: 0.000e+00
Saved gate Owen CSV to /tmp/qsvm_smoke/dataset_0/r1/gate_owen_values.csv
Saved group Owen CSV to /tmp/qsvm_smoke/dataset_0/r1/group_owen_values.csv
Saved exact coalition CSV to /tmp/qsvm_smoke/dataset_0/r1/coalition_values_exact_owen.csv
Saved summary JSON to /tmp/qsvm_smoke/dataset_0/r1/summary.json
Runtime: 8.0s
Saved run summary to /tmp/qsvm_smoke/all_run_summaries.csv
```

All commands exited with status 0.
