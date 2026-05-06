"""SMILES -> 3D geometry resolution via RDKit.

Returns ``(symbols, coordinates_in_bohr)`` ready for PennyLane / PySCF
``molecular_hamiltonian``.  PennyLane expects coordinates in atomic units
(Bohr); RDKit produces angstroms, so we convert.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..logging_utils import get_logger

logger = get_logger("geometry")

ANGSTROM_TO_BOHR = 1.8897259886

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class GeometryError(RuntimeError):
    """Raised when a SMILES cannot be turned into a 3D geometry."""


def smiles_to_geometry(
    smiles: str,
    optimize: bool = True,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """Convert a SMILES string into atomic symbols + Bohr coordinates.

    Parameters
    ----------
    smiles
        The SMILES string to embed.
    optimize
        Run UFF optimization after ETKDG embedding.  Disable to save time on
        large batches where geometry quality matters less.
    seed
        Random seed handed to ETKDG to make embeddings deterministic.
    """
    if not RDKIT_AVAILABLE:
        raise GeometryError("RDKit is not installed; cannot embed SMILES.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise GeometryError(f"Invalid SMILES: {smiles!r}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) == -1:
        # Retry with random coords, otherwise give up.
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) == -1:
            raise GeometryError(f"ETKDG failed for {smiles!r}")

    if optimize:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception as exc:
            logger.debug("UFF optimization failed for %s: %s", smiles, exc)

    conf = mol.GetConformer()
    symbols, coords = [], []
    for i, atom in enumerate(mol.GetAtoms()):
        symbols.append(atom.GetSymbol())
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])

    return symbols, np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
