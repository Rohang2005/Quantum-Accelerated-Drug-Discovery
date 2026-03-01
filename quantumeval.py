import numpy as np
import pennylane as qml


def run_vqe():
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

    ops = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(2),
        qml.PauliZ(3),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(2) @ qml.PauliZ(3),
        qml.PauliX(0) @ qml.PauliX(1),
        qml.PauliY(0) @ qml.PauliY(1),
    ]

    H = qml.Hamiltonian(coeffs, ops)
    n_qubits = 4
    dev = qml.device("lightning.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(params):
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return qml.expval(H)

    params = np.random.uniform(
        low=-np.pi,
        high=np.pi,
        size=(n_qubits,)
    )

    params = qml.numpy.array(params, requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=0.1)
    steps = 50

    for _ in range(steps):
        params, energy = optimizer.step_and_cost(circuit, params)
    return float(energy)

if __name__ == "__main__":
    score = run_vqe()
    print("Quantum Evaluation Score:", score)
