"""Chemistry-aware feature extraction from top-scoring molecules.

We use RDKit's :mod:`rdkit.Chem.Scaffolds.MurckoScaffold` to find recurring
core skeletons and a small SMARTS library to flag common functional groups.
The resulting list is fed both to (a) the LLM explanation prompt and (b) the
feedback module so the GAN can be biased toward fragments that consistently
score well.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

from ..config import SIMULATION_CONFIG
from ..logging_utils import get_logger
from ..quantum.vqe import VQEResult

logger = get_logger("features")

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


_FUNCTIONAL_GROUPS: List[Tuple[str, str]] = [
    ("hydroxyl",       "[OX2H]"),
    ("primary_amine",  "[NX3;H2;!$(NC=O)]"),
    ("secondary_amine","[NX3;H1;!$(NC=O)]"),
    ("amide",          "[NX3][CX3](=[OX1])"),
    ("carbonyl",       "[CX3]=[OX1]"),
    ("carboxyl",       "[CX3](=O)[OX2H1]"),
    ("ether",          "[OD2]([#6])[#6]"),
    ("nitrile",        "[CX2]#[NX1]"),
    ("aromatic_ring",  "a1aaaaa1"),
    ("halide",         "[F,Cl,Br,I]"),
    ("thiol",          "[SX2H]"),
]


def extract_scaffolds(
    results: Iterable[VQEResult],
    top_k: int = None,
) -> List[str]:
    """Return the ``top_k`` most common Bemis-Murcko scaffold SMILES."""
    if top_k is None:
        top_k = SIMULATION_CONFIG.get("top_k_features", 3)
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit unavailable; skipping scaffold extraction.")
        return []

    counts: Counter = Counter()
    for r in results:
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaf, canonical=True) if scaf else ""
        except Exception:
            continue
        if scaf_smi:
            counts[scaf_smi] += 1

    return [s for s, _ in counts.most_common(top_k)]


def extract_functional_groups(
    results: Iterable[VQEResult],
    top_k: int = 5,
) -> List[str]:
    """Return human-readable names of functional groups common to top molecules."""
    if not RDKIT_AVAILABLE:
        return []

    counts: Counter = Counter()
    for r in results:
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            continue
        for name, smarts in _FUNCTIONAL_GROUPS:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None and mol.HasSubstructMatch(patt):
                counts[name] += 1

    return [name for name, _ in counts.most_common(top_k)]
