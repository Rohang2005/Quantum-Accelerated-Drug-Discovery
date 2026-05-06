"""SMILES <-> molecular-graph tensor conversion.

The original repo trained a MolGAN on random uniform tensors, which means the
generator never saw any chemistry.  This module fixes that: we encode each
SMILES string into a fixed-size pair of one-hot tensors (atoms, bonds) and
provide a robust decoder that argmaxes generator output back into RDKit
``Mol`` objects.

Conventions
-----------
* ``MAX_ATOMS`` slots per molecule.  Unused slots are tagged with the
  ``PAD`` atom type and connected via ``NoBond`` edges.
* ``ATOM_TYPES`` = 7  (C, N, O, F, S, Cl, PAD)
* ``BOND_TYPES`` = 5  (NoBond, Single, Double, Triple, Aromatic)

The encode/decode routines round-trip cleanly for every SMILES that survives
:func:`smiles_to_graph` (i.e. one whose atoms are all in ``ATOM_MAP`` and which
has at most ``MAX_ATOMS`` heavy atoms).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .logging_utils import get_logger

logger = get_logger("smiles_graph")

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None  # type: ignore[assignment]


MAX_ATOMS = 12
ATOM_MAP = [6, 7, 8, 9, 16, 17]  # C, N, O, F, S, Cl
PAD_INDEX = len(ATOM_MAP)        # index 6 == PAD
ATOM_TYPES = len(ATOM_MAP) + 1   # 7
BOND_TYPES = 5                   # NoBond, Single, Double, Triple, Aromatic

if RDKIT_AVAILABLE:
    BOND_MAP = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    _BOND_TO_IDX = {b: i for i, b in enumerate(BOND_MAP) if b is not None}
else:
    BOND_MAP = [None] * BOND_TYPES
    _BOND_TO_IDX = {}


def smiles_to_graph(
    smiles: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Encode a SMILES string into ``(nodes, edges)`` one-hot tensors.

    Returns ``None`` when the molecule is invalid, contains an unsupported
    atom, or exceeds ``MAX_ATOMS`` heavy atoms.

    Shapes
    ------
    nodes : ``(MAX_ATOMS, ATOM_TYPES)``
    edges : ``(MAX_ATOMS, MAX_ATOMS, BOND_TYPES)``
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for SMILES encoding.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumHeavyAtoms() > MAX_ATOMS:
        return None

    nodes = np.zeros((MAX_ATOMS, ATOM_TYPES), dtype=np.float32)
    edges = np.zeros((MAX_ATOMS, MAX_ATOMS, BOND_TYPES), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        if z not in ATOM_MAP:
            return None
        nodes[i, ATOM_MAP.index(z)] = 1.0

    for j in range(mol.GetNumAtoms(), MAX_ATOMS):
        nodes[j, PAD_INDEX] = 1.0

    edges[:, :, 0] = 1.0
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt not in _BOND_TO_IDX:
            continue
        idx = _BOND_TO_IDX[bt]
        edges[a, b, 0] = 0.0
        edges[b, a, 0] = 0.0
        edges[a, b, idx] = 1.0
        edges[b, a, idx] = 1.0

    return nodes, edges


def encode_smiles_dataset(
    smiles_list: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Encode a list of SMILES, dropping anything that fails to encode.

    Returns
    -------
    nodes_tensor : shape ``(N, MAX_ATOMS, ATOM_TYPES)``
    edges_tensor : shape ``(N, MAX_ATOMS, MAX_ATOMS, BOND_TYPES)``
    kept_smiles  : the canonical SMILES that survived encoding
    """
    nodes, edges, kept = [], [], []
    for smi in smiles_list:
        result = smiles_to_graph(smi)
        if result is None:
            continue
        n, e = result
        nodes.append(n)
        edges.append(e)
        kept.append(smi)

    if not nodes:
        raise ValueError("No SMILES could be encoded into the graph format.")

    nodes_tensor = torch.from_numpy(np.stack(nodes))
    edges_tensor = torch.from_numpy(np.stack(edges))
    logger.info("Encoded %d/%d SMILES into graph tensors", len(kept), len(smiles_list))
    return nodes_tensor, edges_tensor, kept


def graph_to_mol(nodes: np.ndarray, edges: np.ndarray) -> Optional["Chem.Mol"]:
    """Decode argmax tensors into an RDKit ``Mol`` (or ``None`` if invalid).

    ``nodes`` must be shape ``(MAX_ATOMS, ATOM_TYPES)`` and ``edges`` must be
    shape ``(MAX_ATOMS, MAX_ATOMS, BOND_TYPES)`` (logits or probabilities; we
    argmax internally).
    """
    if not RDKIT_AVAILABLE:
        return None

    node_idx = np.argmax(nodes, axis=-1)  # (V,)
    edge_idx = np.argmax(edges, axis=-1)  # (V, V)

    rw = Chem.RWMol()
    rdkit_indices: List[Optional[int]] = []
    for v in range(MAX_ATOMS):
        atype = int(node_idx[v])
        if atype == PAD_INDEX:
            rdkit_indices.append(None)
            continue
        atom = Chem.Atom(ATOM_MAP[atype])
        rdkit_indices.append(rw.AddAtom(atom))

    for i in range(MAX_ATOMS):
        if rdkit_indices[i] is None:
            continue
        for j in range(i + 1, MAX_ATOMS):
            if rdkit_indices[j] is None:
                continue
            bidx = int(edge_idx[i, j])
            if bidx == 0 or bidx >= len(BOND_MAP) or BOND_MAP[bidx] is None:
                continue
            rw.AddBond(rdkit_indices[i], rdkit_indices[j], BOND_MAP[bidx])

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return _attempt_recovery(rw)
    if mol.GetNumAtoms() == 0:
        return None
    return mol


def _attempt_recovery(rw: "Chem.RWMol") -> Optional["Chem.Mol"]:
    """Try to keep just the largest connected fragment that sanitizes."""
    try:
        mol = rw.GetMol()
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return None

    best = None
    for frag in frags:
        try:
            Chem.SanitizeMol(frag)
        except Exception:
            continue
        if frag.GetNumAtoms() == 0:
            continue
        if best is None or frag.GetNumAtoms() > best.GetNumAtoms():
            best = frag
    return best


def graph_to_smiles(nodes: np.ndarray, edges: np.ndarray) -> Optional[str]:
    """Decode tensors to a canonical SMILES of the largest *connected* fragment.

    Disconnected molecules (e.g. ``"C.CCC"``) are rejected at the fragment
    level because PySCF cannot build a single Hamiltonian for unconnected
    systems, which makes them dead weight for the VQE step.  We keep just
    the largest sanitizable fragment with at least 2 heavy atoms.
    """
    mol = graph_to_mol(nodes, edges)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    except Exception:
        return None

    best = None
    for frag in frags:
        try:
            Chem.SanitizeMol(frag)
        except Exception:
            continue
        if frag.GetNumHeavyAtoms() < 2:
            continue
        if best is None or frag.GetNumHeavyAtoms() > best.GetNumHeavyAtoms():
            best = frag

    if best is None:
        return None
    try:
        smi = Chem.MolToSmiles(best, canonical=True)
    except Exception:
        return None
    if not smi or "." in smi:
        return None
    return smi


def batch_decode_to_smiles(
    nodes_batch: torch.Tensor,
    edges_batch: torch.Tensor,
) -> List[Optional[str]]:
    """Argmax-decode a whole batch of generator outputs."""
    nodes_np = nodes_batch.detach().cpu().numpy()
    edges_np = edges_batch.detach().cpu().numpy()
    out: List[Optional[str]] = []
    for i in range(nodes_np.shape[0]):
        out.append(graph_to_smiles(nodes_np[i], edges_np[i]))
    return out
