"""Generate analysis graphs for the Quantum-AI Drug Discovery pipeline.

Produces a set of PNG figures inside the ``graphs/`` folder at the project
root. Each figure is self-contained and is regenerated every time you run
this script.

Sections
--------
1. Dataset analysis (ZINC SMILES corpus)
   - Heavy-atom distribution
   - Molecular-weight distribution
   - LogP distribution (drug-likeness proxy)
   - Ring-count distribution
   - Atom-type frequencies
   - SMILES length distribution
   - Pairwise descriptor correlation heatmap

2. MolGAN generative analysis
   - Validity rate per sampling attempt
   - Heavy-atom distribution: real vs generated
   - Atom-type frequencies: real vs generated

3. VQE / quantum scoring
   - VQE energy bar chart for the curated drug-fragment pool
   - Heavy atoms vs energy scatter
   - Energy distribution histogram

4. Pipeline summary
   - Stage timing breakdown for a small inference run

Usage
-----
    python scripts/generate_graphs.py                   # everything
    python scripts/generate_graphs.py --skip-molgan     # no GAN sampling
    python scripts/generate_graphs.py --skip-vqe        # no quantum scoring
    python scripts/generate_graphs.py --dataset-rows 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import discover_dataset  # noqa: E402
from src.inference import SMALL_MOLECULE_FALLBACK_POOL  # noqa: E402
from src.logging_utils import get_logger  # noqa: E402

logger = get_logger("graphs")

GRAPHS_DIR = PROJECT_ROOT / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "font.size": 11,
})

PALETTE_PRIMARY = "#3a86ff"
PALETTE_SECONDARY = "#ff006e"
PALETTE_ACCENT = "#06d6a0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> Path:
    out = GRAPHS_DIR / name
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out.relative_to(PROJECT_ROOT))
    return out


_SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "Smiles",
                             "canonical_smiles", "smi")


def _fast_load_smiles(n: int, max_heavy_atoms: int) -> List[str]:
    """Fast ZINC SMILES loader: C-engine CSV read with chunked filtering.

    Bypasses the project's default loader (which uses ``engine='python'`` with
    ``sep=None``; that path takes ~hours on a 22 MB CSV on Windows). We read
    the file in chunks and stop as soon as we've collected ``n`` SMILES that
    pass the heavy-atom filter.
    """
    from rdkit import Chem

    path = discover_dataset()
    if path is None or not path.exists():
        logger.warning("No dataset found; using fallback pool only.")
        return list(SMALL_MOLECULE_FALLBACK_POOL)

    logger.info("Streaming dataset from %s", path)
    smiles: List[str] = []
    chunksize = 5000
    total_seen = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, engine="c"):
        col = next((c for c in _SMILES_COLUMN_CANDIDATES if c in chunk.columns),
                   None)
        if col is None:
            stripped = {c: c.strip().lower() for c in chunk.columns}
            for orig, low in stripped.items():
                if low in {c.lower() for c in _SMILES_COLUMN_CANDIDATES}:
                    col = orig
                    break
        if col is None:
            logger.error("Could not find a SMILES column in %s", path)
            return list(SMALL_MOLECULE_FALLBACK_POOL)

        for raw in chunk[col].dropna().astype(str):
            total_seen += 1
            smi = raw.strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if mol.GetNumHeavyAtoms() > max_heavy_atoms:
                continue
            smiles.append(Chem.MolToSmiles(mol, canonical=True))
            if len(smiles) >= n:
                break
        if len(smiles) >= n:
            break

    logger.info("Selected %d/%d SMILES (heavy_atoms <= %d).",
                len(smiles), total_seen, max_heavy_atoms)
    return smiles


def _rdkit_descriptors(smiles_list: Sequence[str]) -> pd.DataFrame:
    """Compute a compact descriptor frame: heavy atoms, MW, LogP, rings, length."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            rows.append({
                "smiles": smi,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "mol_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "num_rings": Lipinski.RingCount(mol),
                "h_donors": Lipinski.NumHDonors(mol),
                "h_acceptors": Lipinski.NumHAcceptors(mol),
                "rot_bonds": Lipinski.NumRotatableBonds(mol),
                "smiles_len": len(smi),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _atom_frequencies(smiles_list: Sequence[str]) -> Counter:
    from rdkit import Chem
    counter: Counter = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            counter[atom.GetSymbol()] += 1
    return counter


# ---------------------------------------------------------------------------
# Dataset graphs
# ---------------------------------------------------------------------------

def graph_dataset_overview(df: pd.DataFrame) -> None:
    """Four-panel overview: heavy atoms, MW, LogP, ring count."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("ZINC dataset: drug-likeness descriptor distributions", fontsize=15)

    sns.histplot(
        df["heavy_atoms"], bins=range(0, int(df["heavy_atoms"].max()) + 2),
        color=PALETTE_PRIMARY, ax=axes[0, 0], edgecolor="white",
    )
    axes[0, 0].set_title("Heavy atom count")
    axes[0, 0].set_xlabel("Heavy atoms")
    axes[0, 0].axvline(12, color=PALETTE_SECONDARY, linestyle="--",
                       label="VQE budget (<=12)")
    axes[0, 0].legend()

    sns.histplot(df["mol_weight"], bins=40, color=PALETTE_ACCENT,
                 ax=axes[0, 1], edgecolor="white")
    axes[0, 1].set_title("Molecular weight (Da)")
    axes[0, 1].set_xlabel("g/mol")
    axes[0, 1].axvline(500, color=PALETTE_SECONDARY, linestyle="--",
                       label="Lipinski MW <= 500")
    axes[0, 1].legend()

    sns.histplot(df["logp"], bins=40, color=PALETTE_PRIMARY,
                 ax=axes[1, 0], edgecolor="white")
    axes[1, 0].set_title("LogP (lipophilicity)")
    axes[1, 0].set_xlabel("LogP")
    axes[1, 0].axvline(5, color=PALETTE_SECONDARY, linestyle="--",
                       label="Lipinski LogP <= 5")
    axes[1, 0].legend()

    rings_max = max(int(df["num_rings"].max()), 1)
    sns.histplot(df["num_rings"], bins=range(0, rings_max + 2),
                 color=PALETTE_ACCENT, ax=axes[1, 1], edgecolor="white", discrete=True)
    axes[1, 1].set_title("Ring count")
    axes[1, 1].set_xlabel("Rings per molecule")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "01_dataset_overview.png")


def graph_atom_frequencies(df: pd.DataFrame) -> None:
    """Atom-type frequency bar chart (per-molecule average)."""
    counts = _atom_frequencies(df["smiles"].tolist())
    if not counts:
        return
    n_mols = max(1, len(df))
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v / n_mols for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=PALETTE_PRIMARY, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Average atom-type count per molecule (ZINC subset)")
    ax.set_ylabel("Atoms per molecule")
    ax.set_xlabel("Element")
    fig.tight_layout()
    _save(fig, "02_atom_frequencies.png")


def graph_smiles_length(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(df["smiles_len"], bins=40, color=PALETTE_SECONDARY,
                 ax=ax, edgecolor="white")
    ax.axvline(df["smiles_len"].median(), color="black", linestyle="--",
               label=f"median = {df['smiles_len'].median():.0f}")
    ax.set_title("SMILES string length distribution")
    ax.set_xlabel("Characters")
    ax.legend()
    fig.tight_layout()
    _save(fig, "03_smiles_length.png")


def graph_descriptor_correlation(df: pd.DataFrame) -> None:
    cols = ["heavy_atoms", "mol_weight", "logp", "num_rings",
            "h_donors", "h_acceptors", "rot_bonds", "smiles_len"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title("Descriptor correlation heatmap")
    fig.tight_layout()
    _save(fig, "04_descriptor_correlation.png")


# ---------------------------------------------------------------------------
# MolGAN graphs
# ---------------------------------------------------------------------------

def graph_molgan_validity_curve(checkpoint_path: Optional[Path]) -> List[str]:
    """Sample multiple batches; track unique-valid count per attempt."""
    from src.inference import load_pretrained_molgan
    from src.smiles_graph import batch_decode_to_smiles
    import torch

    molgan = load_pretrained_molgan(checkpoint_path)
    n_attempts = 8
    batch_size = 64

    cumulative_unique: List[int] = []
    cumulative_valid: List[int] = []
    seen_valid: set = set()
    total_decoded = 0
    total_valid = 0

    molgan.generator.eval()
    for attempt in range(1, n_attempts + 1):
        with torch.no_grad():
            z = torch.randn(batch_size, molgan.cfg.z_dim, device=molgan.device)
            edges, nodes = molgan.generator(z)
            decoded = batch_decode_to_smiles(nodes, edges)

        for smi in decoded:
            total_decoded += 1
            if smi:
                total_valid += 1
                seen_valid.add(smi)
        cumulative_unique.append(len(seen_valid))
        cumulative_valid.append(total_valid)
        logger.info("MolGAN attempt %d: %d unique valid SMILES",
                    attempt, len(seen_valid))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(1, n_attempts + 1)
    validity_rate = [v / (i * batch_size) * 100 for i, v in
                     enumerate(cumulative_valid, start=1)]
    ax.bar(x - 0.18, cumulative_unique, width=0.36,
           color=PALETTE_PRIMARY, label="Unique valid SMILES (cumulative)")
    ax.bar(x + 0.18, cumulative_valid, width=0.36,
           color=PALETTE_ACCENT, label="Valid SMILES (cumulative)")
    ax.set_xlabel("Sampling attempt")
    ax.set_ylabel("Count")
    ax.set_title("MolGAN sampling progression")
    ax.set_xticks(x)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(x, validity_rate, color=PALETTE_SECONDARY, marker="o",
             linewidth=2, label="Validity rate (%)")
    ax2.set_ylabel("Validity rate (%)", color=PALETTE_SECONDARY)
    ax2.tick_params(axis="y", colors=PALETTE_SECONDARY)
    ax2.grid(False)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "05_molgan_sampling_progression.png")

    return list(seen_valid)


def graph_real_vs_generated(real_smiles: Sequence[str],
                            gen_smiles: Sequence[str]) -> None:
    if not gen_smiles:
        logger.warning("No generated SMILES to compare; skipping real-vs-generated.")
        return

    real_df = _rdkit_descriptors(real_smiles)
    gen_df = _rdkit_descriptors(gen_smiles)
    if real_df.empty or gen_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    bins = range(0, int(max(real_df["heavy_atoms"].max(),
                             gen_df["heavy_atoms"].max())) + 2)
    axes[0].hist(real_df["heavy_atoms"], bins=bins, alpha=0.6,
                 color=PALETTE_PRIMARY, label=f"ZINC (n={len(real_df)})",
                 density=True, edgecolor="white")
    axes[0].hist(gen_df["heavy_atoms"], bins=bins, alpha=0.6,
                 color=PALETTE_SECONDARY, label=f"MolGAN (n={len(gen_df)})",
                 density=True, edgecolor="white")
    axes[0].set_title("Heavy-atom distribution")
    axes[0].set_xlabel("Heavy atoms")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    real_atoms = _atom_frequencies(real_smiles)
    gen_atoms = _atom_frequencies(gen_smiles)
    elements = sorted(set(real_atoms) | set(gen_atoms))
    real_freq = np.array([real_atoms.get(e, 0) for e in elements], dtype=float)
    gen_freq = np.array([gen_atoms.get(e, 0) for e in elements], dtype=float)
    real_freq = real_freq / max(1, real_freq.sum())
    gen_freq = gen_freq / max(1, gen_freq.sum())

    x = np.arange(len(elements))
    axes[1].bar(x - 0.2, real_freq, width=0.4, color=PALETTE_PRIMARY,
                label="ZINC", edgecolor="white")
    axes[1].bar(x + 0.2, gen_freq, width=0.4, color=PALETTE_SECONDARY,
                label="MolGAN", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(elements)
    axes[1].set_title("Atom-type composition (normalized)")
    axes[1].set_xlabel("Element")
    axes[1].set_ylabel("Fraction of atoms")
    axes[1].legend()

    fig.suptitle("Real (ZINC) vs MolGAN-generated molecules", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "06_real_vs_generated.png")


# ---------------------------------------------------------------------------
# VQE graphs
# ---------------------------------------------------------------------------

def graph_vqe_pool(skip: bool = False) -> Optional[List[Tuple[str, float, int]]]:
    """Score the curated fallback pool with VQE; produce energy graphs."""
    if skip:
        logger.info("Skipping VQE scoring section.")
        return None

    from src.quantum.vqe import VQEScorer
    from rdkit import Chem

    scorer = VQEScorer()
    scored: List[Tuple[str, float, int]] = []
    for smi in SMALL_MOLECULE_FALLBACK_POOL:
        result = scorer.score(smi)
        if not result.succeeded:
            logger.info("Skipped %s (status=%s)", smi, result.status)
            continue
        mol = Chem.MolFromSmiles(smi)
        n_heavy = mol.GetNumHeavyAtoms() if mol else 0
        scored.append((smi, float(result.energy), n_heavy))

    if not scored:
        logger.warning("VQE produced no successful scores; skipping VQE graphs.")
        return None

    scored.sort(key=lambda t: t[1])
    smis = [s for s, _, _ in scored]
    energies = np.array([e for _, e, _ in scored])
    heavy = np.array([h for _, _, h in scored])

    fig, ax = plt.subplots(figsize=(13, 6.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(scored)))
    bars = ax.barh(smis, energies, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("VQE / HF ground-state energy (Hartree)")
    ax.set_title(
        "Curated drug-fragment pool: VQE energies (lower = more stable)")
    for bar, e in zip(bars, energies):
        ax.text(e, bar.get_y() + bar.get_height() / 2,
                f" {e:+.2f}", va="center", ha="left", fontsize=9)
    fig.tight_layout()
    _save(fig, "07_vqe_energy_ranking.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(heavy, energies, c=energies, cmap="viridis",
                    s=120, edgecolor="black", linewidth=0.6)
    for x, y, label in zip(heavy, energies, smis):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, alpha=0.85)
    ax.set_xlabel("Heavy atoms")
    ax.set_ylabel("VQE energy (Hartree)")
    ax.set_title("Molecule size vs ground-state energy")
    plt.colorbar(sc, ax=ax, label="Energy (Ha)")
    fig.tight_layout()
    _save(fig, "08_vqe_size_vs_energy.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(energies, bins=12, color=PALETTE_PRIMARY,
                 edgecolor="white", ax=ax, kde=True)
    ax.axvline(energies.mean(), color=PALETTE_SECONDARY, linestyle="--",
               label=f"mean = {energies.mean():.2f} Ha")
    ax.axvline(np.median(energies), color="black", linestyle=":",
               label=f"median = {np.median(energies):.2f} Ha")
    ax.set_title("VQE energy distribution (curated fragment pool)")
    ax.set_xlabel("Energy (Hartree)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "09_vqe_energy_distribution.png")

    return scored


# ---------------------------------------------------------------------------
# Pipeline timing graph
# ---------------------------------------------------------------------------

def graph_pipeline_timing(real_smiles: Sequence[str],
                          checkpoint_path: Optional[Path]) -> None:
    """Time each major pipeline stage and chart the breakdown."""
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    from src.smiles_graph import encode_smiles_dataset
    sample_real = list(real_smiles[:500])
    try:
        encode_smiles_dataset(sample_real)
    except Exception as exc:
        logger.warning("Encoding subset failed: %s", exc)
    timings["SMILES -> graph encode"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from src.inference import load_pretrained_molgan
    molgan = load_pretrained_molgan(checkpoint_path)
    timings["MolGAN load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    samples = molgan.sample_smiles(n=8, max_tries=4, max_heavy_atoms=7)
    timings["MolGAN sample (8 mols)"] = time.perf_counter() - t0
    logger.info("MolGAN produced %d samples for timing graph", len(samples))

    t0 = time.perf_counter()
    try:
        from src.quantum.vqe import VQEScorer
        scorer = VQEScorer()
        for smi in SMALL_MOLECULE_FALLBACK_POOL[:5]:
            scorer.score(smi)
    except Exception as exc:
        logger.warning("VQE timing failed: %s", exc)
    timings["VQE score (5 mols, cached)"] = time.perf_counter() - t0

    labels = list(timings.keys())
    values = [timings[k] for k in labels]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(labels, values, color=PALETTE_ACCENT, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time (seconds)")
    ax.set_title("Pipeline stage timing breakdown")
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y() + bar.get_height() / 2,
                f" {v:.2f}s", va="center", ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, "10_pipeline_timing.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-rows", type=int, default=4000,
                        help="Number of ZINC SMILES to load for analysis "
                             "(default: 4000).")
    parser.add_argument("--max-heavy-atoms", type=int, default=12,
                        help="Heavy-atom filter for the loaded subset.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="MolGAN checkpoint path "
                             "(default: checkpoints/molgan.pt).")
    parser.add_argument("--skip-molgan", action="store_true",
                        help="Skip MolGAN sampling graphs.")
    parser.add_argument("--skip-vqe", action="store_true",
                        help="Skip VQE scoring graphs.")
    parser.add_argument("--skip-timing", action="store_true",
                        help="Skip the pipeline-timing graph.")
    args = parser.parse_args()

    logger.info("Output directory: %s", GRAPHS_DIR)

    dataset_path = discover_dataset()
    logger.info("Dataset path: %s", dataset_path)

    logger.info("Loading up to %d SMILES (heavy_atoms <= %d)...",
                args.dataset_rows, args.max_heavy_atoms)
    smiles = _fast_load_smiles(n=args.dataset_rows,
                               max_heavy_atoms=args.max_heavy_atoms)
    logger.info("Loaded %d SMILES.", len(smiles))

    df = _rdkit_descriptors(smiles)
    if df.empty:
        logger.error("No valid molecules parsed; aborting.")
        return 1
    logger.info("Computed descriptors for %d molecules.", len(df))

    graph_dataset_overview(df)
    graph_atom_frequencies(df)
    graph_smiles_length(df)
    graph_descriptor_correlation(df)

    gen_smiles: List[str] = []
    if not args.skip_molgan:
        try:
            gen_smiles = graph_molgan_validity_curve(args.checkpoint)
        except Exception as exc:
            logger.error("MolGAN sampling graph failed: %s", exc)
        try:
            graph_real_vs_generated(df["smiles"].tolist(), gen_smiles)
        except Exception as exc:
            logger.error("Real-vs-generated graph failed: %s", exc)

    if not args.skip_vqe:
        try:
            graph_vqe_pool(skip=False)
        except Exception as exc:
            logger.error("VQE graphs failed: %s", exc)

    if not args.skip_timing:
        try:
            graph_pipeline_timing(df["smiles"].tolist(), args.checkpoint)
        except Exception as exc:
            logger.error("Timing graph failed: %s", exc)

    logger.info("All graphs written to %s", GRAPHS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
