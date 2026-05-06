"""Standalone VQE smoke test on Lithium Hydride (LiH).

Run before the full pipeline to verify PennyLane + PySCF are correctly
installed.  Output should converge to about ``-7.86`` Ha for LiH at 1.6 Bohr
with a 2-electron / 2-orbital active space and STO-3G.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pennylane as qml
from pennylane import numpy as pnp

from src.config import QUANTUM_CONFIG
from src.logging_utils import get_logger

logger = get_logger("smoke")


def main() -> None:
    symbols = ["Li", "H"]
    coordinates = pnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]])

    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates,
        charge=0, mult=1, basis="sto-3g",
        active_electrons=QUANTUM_CONFIG.get("test_active_electrons", 2),
        active_orbitals=QUANTUM_CONFIG.get("test_active_orbitals", 2),
        load_data=True,
    )
    logger.info("Built LiH Hamiltonian with %d qubits", n_qubits)

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def cost(params):
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(H)

    pnp.random.seed(42)
    params = pnp.array(pnp.random.normal(0, pnp.pi, n_qubits), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=QUANTUM_CONFIG.get("test_vqe_stepsize", 0.1))

    steps = QUANTUM_CONFIG.get("test_vqe_steps", 40)
    energy = float("inf")
    for step in range(steps):
        params, energy = optimizer.step_and_cost(cost, params)
        if step % 10 == 0:
            logger.info("Step %3d | energy = %.6f Ha", step, energy)

    logger.info("FINAL LiH energy: %.6f Ha", energy)


if __name__ == "__main__":
    main()
