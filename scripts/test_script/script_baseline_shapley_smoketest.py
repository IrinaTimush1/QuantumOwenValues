import sys
from pathlib import Path

# Add vendored qshaptools to import path
sys.path.append(str(Path(__file__).parent / "third_party" / "qshaptools"))

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import QAOAAnsatz
from qiskit.opflow import PauliSumOp

from qshap import QuantumShapleyValues
from qvalues import value_H
from tools import visualize_shapleys

def main():
    # 1) tiny circuit
    H = PauliSumOp.from_list([("ZZI", 1), ("ZII", 2), ("ZIZ", -3)])
    qc = QAOAAnsatz(cost_operator=H, reps=1)
    qc = qc.decompose().decompose().decompose()
    qc = qc.assign_parameters([0] * len(qc.parameters))

    # 2) simulator
    qi = QuantumInstance(backend=Aer.get_backend("statevector_simulator"))

    # 3) Shapley wrapper
    qsv = QuantumShapleyValues(
        qc,
        value_fun=value_H,
        value_kwargs_dict=dict(H=H),
        quantum_instance=qi
    )

    print(qsv)

    # 4) run Shapley
    qsv()

    # 5) show results
    print("phi_dict =", qsv.phi_dict)

    # 6) visualize + save
    qc_vis = visualize_shapleys(qc, phi_dict=qsv.phi_dict)
    fig = qc_vis.draw(output="mpl", fold=-1)
    fig.savefig("baseline_shapley_plot.png", dpi=200)
    print("Saved: baseline_shapley_plot.png")

if __name__ == "__main__":
    main()