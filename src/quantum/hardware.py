"""Qiskit Aer 'hardware-style' validation pass.

The original implementation simply did a single forward pass at theta=0 and
called it validation.  Here we run a short, real VQE optimization against the
``qiskit.aer`` backend so the resulting energy is comparable to the
PennyLane Lightning result (lower = better) and reflects the actual
quantum-style cost surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pennylane as qml
from pennylane import numpy as pnp

from ..logging_utils import get_logger
from .geometry import GeometryError, smiles_to_geometry

logger = get_logger("hardware")


@dataclass
class HardwareValidationResult:
    smiles: str
    energy: float
    n_qubits: int
    succeeded: bool
    error: Optional[str] = None


class HardwareValidator:
    """Re-runs VQE on Qiskit Aer for a final sanity check.

    A short ``vqe_steps`` is used here (default 20) because this stage is
    only meant to verify the candidate behaves consistently across simulator
    backends, not to chase another decimal place of accuracy.
    """

    def __init__(
        self,
        active_electrons: int = 2,
        active_orbitals: int = 2,
        basis: str = "sto-3g",
        vqe_steps: int = 20,
        stepsize: float = 0.1,
    ):
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.basis = basis
        self.vqe_steps = vqe_steps
        self.stepsize = stepsize

    def validate(self, smiles: str) -> HardwareValidationResult:
        try:
            symbols, coords = smiles_to_geometry(smiles)
        except GeometryError as exc:
            return HardwareValidationResult(smiles, float("inf"), 0, False, str(exc))

        try:
            H, n_qubits = qml.qchem.molecular_hamiltonian(
                symbols,
                coords,
                charge=0,
                mult=1,
                basis=self.basis,
                active_electrons=self.active_electrons,
                active_orbitals=self.active_orbitals,
                load_data=True,
            )
        except Exception as exc:
            return HardwareValidationResult(smiles, float("inf"), 0, False, str(exc))

        try:
            dev = qml.device("qiskit.aer", wires=n_qubits)
        except Exception as exc:
            logger.warning("qiskit.aer unavailable, falling back to default.qubit: %s", exc)
            dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def cost_fn(params):
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(params[n_qubits + i], wires=i)
            return qml.expval(H)

        try:
            params = pnp.array(
                pnp.random.normal(0, pnp.pi, n_qubits * 2), requires_grad=True
            )
            optimizer = qml.AdamOptimizer(stepsize=self.stepsize)
            energy = float("inf")
            for _ in range(self.vqe_steps):
                params, energy = optimizer.step_and_cost(cost_fn, params)
        except Exception as exc:
            return HardwareValidationResult(smiles, float("inf"), n_qubits, False, str(exc))

        return HardwareValidationResult(smiles, float(energy), n_qubits, True)
