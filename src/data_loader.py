"""Load and filter molecules from the ZINC dataset.

The Variational Quantum Eigensolver in :mod:`src.quantum.vqe` builds a
molecular Hamiltonian via PySCF.  Even with an aggressive active-space
approximation, molecules with more than ~12 heavy atoms blow past 8 GB of RAM
on a laptop, so we filter the dataset down before anything quantum touches it.

The loader auto-detects any CSV under ``data/`` containing a ``smiles``-like
column, so users can drop in ``Zinc_250K.csv``, ``250k_rndm_zinc_drugs_clean.csv``,
or any custom export without renaming.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import DATA_DIR
from .logging_utils import get_logger

logger = get_logger("data")

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


_FALLBACK_SMILES = [
    "CC(C)C", "CCO", "CCN", "CCC", "C=O", "CC=O", "CCOC", "CCCC", "CCCN",
    "C1CCCCC1", "c1ccccc1", "CC(=O)O", "CCN(CC)CC", "CC#N", "CO",
]

_SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "Smiles", "canonical_smiles", "smi")


def discover_dataset(data_dir: Path = DATA_DIR) -> Optional[Path]:
    """Return the first CSV/TSV under ``data_dir`` that looks like a SMILES dump.

    Preference order:
      1. ``Zinc_250K.csv`` (the documented default).
      2. Any other ``*.csv`` containing a SMILES-like column.
      3. ``*.smi`` / ``*.tsv`` with the same heuristic.
    """
    if not data_dir.exists():
        return None

    preferred = data_dir / "Zinc_250K.csv"
    if preferred.exists():
        return preferred

    for pattern in ("*.csv", "*.tsv", "*.smi"):
        for path in sorted(data_dir.glob(pattern)):
            try:
                head = pd.read_csv(path, nrows=5, sep=None, engine="python")
            except Exception:
                continue
            for col in head.columns:
                if str(col).strip().lower() in {c.lower() for c in _SMILES_COLUMN_CANDIDATES}:
                    return path
    return None


def load_zinc_subset(
    n: int = 10_000,
    max_heavy_atoms: int = 12,
    csv_name: Optional[str] = None,
) -> List[str]:
    """Return up to ``n`` SMILES strings filtered to small drug-like fragments.

    Parameters
    ----------
    n
        Maximum number of SMILES to return.
    max_heavy_atoms
        Drop any molecule whose heavy-atom count exceeds this.  Match this to
        the active space you intend to use in VQE; 12 is a safe default for
        STO-3G with a 2-electron / 2-orbital active space.
    csv_name
        Optional explicit file name inside ``DATA_DIR``.  When omitted we
        auto-detect the first SMILES-like CSV in the data directory.
    """
    if csv_name:
        file_path = DATA_DIR / csv_name
        if not file_path.exists():
            logger.warning("%s not found; trying auto-detect.", file_path)
            file_path = discover_dataset() or file_path
    else:
        file_path = discover_dataset()

    if file_path is None or not file_path.exists():
        logger.warning(
            "No SMILES dataset found in %s; falling back to %d built-in SMILES.",
            DATA_DIR, len(_FALLBACK_SMILES),
        )
        return _FALLBACK_SMILES[:n] if n < len(_FALLBACK_SMILES) else list(_FALLBACK_SMILES)

    try:
        df = pd.read_csv(file_path, nrows=250_000, sep=None, engine="python")
    except Exception as exc:
        logger.error("Failed to read %s: %s", file_path, exc)
        return list(_FALLBACK_SMILES)

    smiles_col = _resolve_smiles_column(df)
    if smiles_col is None:
        logger.error("CSV %s has no SMILES-like column.", file_path)
        return list(_FALLBACK_SMILES)
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})

    df["smiles"] = (
        df["smiles"].astype(str).str.replace("\n", "", regex=False).str.strip()
    )
    df = df[df["smiles"].apply(lambda x: bool(x) and '"' not in x)]

    valid: List[str] = []
    for raw in df["smiles"]:
        smi = _validate_and_filter(raw, max_heavy_atoms)
        if smi is not None:
            valid.append(smi)
            if len(valid) >= n:
                break

    if not valid:
        logger.warning("No molecules survived filtering; using fallback list.")
        return list(_FALLBACK_SMILES)

    logger.info("Loaded %d filtered SMILES from %s", len(valid), file_path.name)
    return valid


def dataset_preview(
    n: int = 20,
    max_heavy_atoms: int = 12,
) -> List[str]:
    """Cheap helper for UIs: return up to ``n`` filtered SMILES."""
    return load_zinc_subset(n=n, max_heavy_atoms=max_heavy_atoms)


def _resolve_smiles_column(df: pd.DataFrame) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in _SMILES_COLUMN_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _validate_and_filter(smi: str, max_heavy_atoms: int) -> Optional[str]:
    if not smi:
        return None
    if not RDKIT_AVAILABLE:
        return smi if len(smi) < 15 else None
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumHeavyAtoms() > max_heavy_atoms:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
