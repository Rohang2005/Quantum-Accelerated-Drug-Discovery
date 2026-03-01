from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

coeffs = [
    -7.498946,
     0.171201,
    -0.222796,
     0.168623,
     0.120546,
     0.165868,
     0.165868,
     0.174349,
     0.120546
]

paulis = [
    "IIII",
    "ZIII",
    "IZII",
    "IIZI",
    "IIIZ",
    "ZZII",
    "IIZZ",
    "XXII",
    "YYII"
]

H = SparsePauliOp(paulis, coeffs)

def ansatz(params):
    qc = QuantumCircuit(4)
    for i in range(4):
        qc.ry(params[i], i)
    for i in range(3):
        qc.cx(i, i + 1)
    return qc

def run_vqe_hardware():

    service = QiskitRuntimeService(channel="ibm_quantum_platform")

    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=4
    )

    print("Using backend:", backend.name)

    params = np.random.uniform(-np.pi, np.pi, 4)
    circuit = ansatz(params)

    job = service.run(
        program_id="estimator",
        options={
            "backend_name": backend.name,
            "shots": 1024
        },
        inputs={
            "circuits": [circuit],
            "observables": [H]
        }
    )

    result = job.result()
    energy = result["values"][0]

    return energy


if __name__ == "__main__":
    energy = run_vqe_hardware()
    print("Hardware VQE Energy:", energy)
