# Third-party code and attribution

This repository builds on the `qshaptools` package by Raoul Heese et al.,
published under the MIT License. See `qshaptools/LICENSE` and the upstream
README at `qshaptools/README.rst`.

## Upstream code: Heese et al., 2023, MIT

The following files under `qshaptools/src/qshaptools/` are from the upstream
`qshaptools` package:

- `cshap.py`
- `qshap.py`
- `ushap.py`
- `qvalues.py`
- `tools.py`
- `values.py`
- `postprocessing.py`

Upstream paper:
Heese, R., Gerlach, T., Mucke, S., Muller, S., Jakobs, M., & Piatkowski, N.
(2025). Explaining quantum circuits with Shapley values: towards explainable
quantum machine learning. *Quantum Machine Intelligence*, 7(27).
https://doi.org/10.1007/s42484-025-00254-8

## New thesis code: Timus, 2026, MIT

The following files under `qshaptools/src/qshaptools/` are new contributions
implementing Owen values for quantum circuit explanations:

- `uowen.py` - core uncertain Owen-value engine (exact and Monte Carlo)
- `qowen.py` - Qiskit wrapper, mirroring `qshap.QuantumShapleyValues`
- `cowen.py` - classical wrapper, mirroring `cshap.ClassicalShapleyValues`
- `partition_util.py` - helpers for designing gate-level partitions

All scripts under `scripts/` are new thesis code, including the controlled
35-circuit benchmark workflow, the Monte Carlo estimator validation, the
faithfulness ablation, and the QNN/QSVM case studies.

## Benchmark generation

The 3-qubit random-circuit pool generation and SRE labelling code is adapted
from the repository accompanying Lipardi et al. (2025):

Lipardi, V., Dibenedetto, D., Stamoulis, G., & Winands, M. H. M. (2025).
A study on stabilizer Renyi entropy estimation using machine learning.
arXiv:2509.16799.
