"""High-level inference helpers for the quantum-AI drug discovery pipeline.

This module is the bridge between a *trained* MolGAN checkpoint and the
downstream scoring/feedback stages.  The Streamlit UI in ``app.py`` consumes
these helpers; you can also call them directly from a notebook.

Typical flow::

    from src.inference import (
        load_pretrained_molgan,
        generate_candidates,
        score_candidates,
        run_inference,
    )

    molgan = load_pretrained_molgan("checkpoints/molgan.pt")
    smiles = generate_candidates(molgan, n=20)
    results = score_candidates(smiles)

Or, end-to-end::

    out = run_inference(checkpoint_path="checkpoints/molgan.pt", n_molecules=20)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .config import QUANTUM_CONFIG, SIMULATION_CONFIG
from .data_loader import load_zinc_subset
from .logging_utils import get_logger
from .models.molgan import MolGAN
from .pipeline.classifier import classify_molecules
from .pipeline.feature_extractor import extract_functional_groups, extract_scaffolds
from .pipeline.feedback import build_feedback_dataset
from .quantum.hardware import HardwareValidationResult, HardwareValidator
from .quantum.vqe import VQEResult, VQEScorer

logger = get_logger("inference")

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "molgan.pt"

ProgressCallback = Callable[[str, float], None]
"""Signature: ``progress("message", fraction_complete in [0, 1])``."""


# Curated drug-fragment SMILES that all fit comfortably under a 7-heavy-atom
# VQE budget. Used as a fallback so the UI always has something to score
# even when MolGAN's distribution skews larger than the budget.
SMALL_MOLECULE_FALLBACK_POOL: List[str] = [
    "CCO",          # ethanol
    "CO",           # methanol
    "CC(=O)O",      # acetic acid
    "CCN",          # ethylamine
    "CCOC",         # methoxyethane
    "C=O",          # formaldehyde
    "CC=O",         # acetaldehyde
    "CCC",          # propane
    "CCCO",         # 1-propanol
    "CC(C)O",       # 2-propanol
    "CCCN",         # propylamine
    "OCCO",         # ethylene glycol
    "NCC(=O)O",     # glycine
    "CC#N",         # acetonitrile
    "C1CCCC1",      # cyclopentane
    "c1ccccc1",     # benzene
    "C1CCNCC1",     # piperidine
    "OC=O",         # formic acid
    "NC=O",         # formamide
    "CSC",          # dimethyl sulfide
]


@dataclass
class InferenceResult:
    """Structured output of :func:`run_inference`."""

    candidates: List[VQEResult] = field(default_factory=list)
    top_candidates: List[VQEResult] = field(default_factory=list)
    scaffolds: List[str] = field(default_factory=list)
    functional_groups: List[str] = field(default_factory=list)
    hardware_validation: Optional[HardwareValidationResult] = None
    explanation: Optional[str] = None
    elapsed_sec: float = 0.0
    iterations_run: int = 0


def load_pretrained_molgan(
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> MolGAN:
    """Load a MolGAN from a checkpoint.

    Falls back to a freshly-initialized untrained MolGAN with a clear warning
    if the checkpoint is missing, so notebook explorations don't crash hard.
    """
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not path.exists():
        logger.warning(
            "No checkpoint at %s; returning an UNTRAINED MolGAN. "
            "Run `python scripts/train.py` first.", path,
        )
        return MolGAN(device=device)
    return MolGAN.load_checkpoint(path, device=device)


def generate_candidates(
    molgan: MolGAN,
    n: int = 20,
    max_tries: int = 10,
    progress: Optional[ProgressCallback] = None,
    max_heavy_atoms: Optional[int] = None,
    use_fallback_pool: bool = True,
    fallback_pool_first: int = 0,
) -> List[str]:
    """Sample ``n`` unique sanitizable SMILES from the generator.

    Parameters
    ----------
    max_heavy_atoms
        If set, only return candidates whose heavy-atom count is at or below
        this limit. Aligns sampling with the downstream VQE budget so the
        scorer doesn't reject everything.
    use_fallback_pool
        If MolGAN can't produce enough small molecules within ``max_tries``,
        top up from a curated drug-fragment pool so the user always gets
        scorable candidates. Set to ``False`` to disable.
    fallback_pool_first
        Take this many candidates from the curated pool *before* asking
        MolGAN to fill the rest. Useful for fast demos where you want to
        guarantee cache-hit-friendly molecules in the result.
    progress
        Optional callback ``progress(message, fraction)`` for UIs.
    """

    def _filter_small(smi: str) -> bool:
        if max_heavy_atoms is None:
            return True
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            return mol is not None and mol.GetNumHeavyAtoms() <= max_heavy_atoms
        except ImportError:
            return True

    candidates: List[str] = []
    seen: set = set()

    fallback_first = min(max(0, int(fallback_pool_first)), n)
    if fallback_first:
        for smi in SMALL_MOLECULE_FALLBACK_POOL:
            if smi in seen or not _filter_small(smi):
                continue
            candidates.append(smi)
            seen.add(smi)
            if len(candidates) >= fallback_first:
                break
        if progress is not None:
            progress(
                f"Seeded {len(candidates)} curated small molecules.",
                len(candidates) / max(1, n),
            )

    def _bridge(collected: int, target: int, attempt: int, max_attempts: int) -> None:
        if progress is None:
            return
        total_so_far = len(candidates) + collected
        frac = max(total_so_far / max(1, n), attempt / max(1, max_attempts))
        progress(
            f"MolGAN sampling: {total_so_far}/{n} candidates (attempt {attempt}/{max_attempts})",
            min(1.0, frac),
        )

    deficit_after_seed = n - len(candidates)
    if deficit_after_seed > 0:
        gan_samples = molgan.sample_smiles(
            n=deficit_after_seed, max_tries=max_tries, progress=_bridge,
            max_heavy_atoms=max_heavy_atoms,
        )
        for smi in gan_samples:
            if smi in seen:
                continue
            candidates.append(smi)
            seen.add(smi)
            if len(candidates) >= n:
                break

    if use_fallback_pool and len(candidates) < n:
        deficit = n - len(candidates)
        topup: List[str] = []
        for smi in SMALL_MOLECULE_FALLBACK_POOL:
            if smi in seen or not _filter_small(smi):
                continue
            topup.append(smi)
            seen.add(smi)
            if len(topup) >= deficit:
                break
        if topup:
            logger.info(
                "Topped up %d candidates from the curated small-molecule fallback pool.",
                len(topup),
            )
            if progress is not None:
                progress(
                    f"Topped up {len(topup)} curated small molecules so VQE has something to score.",
                    1.0,
                )
            candidates = candidates + topup

    return candidates


def score_candidates(
    smiles_list: Sequence[str],
    scorer: Optional[VQEScorer] = None,
    progress: Optional[ProgressCallback] = None,
    **scorer_kwargs,
) -> List[VQEResult]:
    """Score each SMILES with VQE.

    SMILES are pre-screened by the scorer's heavy-atom budget so molecules
    that would push PySCF into a multi-minute SCF are rejected immediately
    (with a ``MOLECULE_TOO_LARGE`` status) instead of stalling the loop.

    The ``progress`` callback fires *before* each VQE attempt so UIs can
    show "scoring X..." rather than waiting for completion.
    """
    scorer = scorer or VQEScorer(**scorer_kwargs)
    results: List[VQEResult] = []
    total = max(1, len(smiles_list))

    n_skipped_upfront = 0
    for smi in smiles_list:
        ok, _ = scorer.screen(smi)
        if not ok:
            n_skipped_upfront += 1
    if n_skipped_upfront and progress is not None:
        progress(
            f"{n_skipped_upfront}/{total} candidates exceed the heavy-atom "
            f"budget and will be skipped (no VQE).",
            0.0,
        )

    for i, smi in enumerate(smiles_list):
        if progress is not None:
            progress(f"Scoring {i + 1}/{total}: {smi}", i / total)
        result = scorer.score(smi)
        results.append(result)
        if progress is not None:
            tag = "ok" if result.succeeded else result.status.value
            progress(f"Scored {i + 1}/{total} [{tag}]: {smi}", (i + 1) / total)
    return results


def validate_on_hardware(smiles: str, **kwargs) -> HardwareValidationResult:
    """Re-run a short VQE on Qiskit Aer for a single candidate."""
    validator = HardwareValidator(**kwargs)
    return validator.validate(smiles)


def run_inference(
    *,
    checkpoint_path: Optional[Path] = None,
    n_molecules: int = 20,
    iterations: int = 1,
    fine_tune_epochs: Optional[int] = None,
    base_corpus: Optional[List[str]] = None,
    top_k: int = 5,
    enable_hardware_validation: bool = True,
    enable_explanation: bool = False,
    scorer_kwargs: Optional[dict] = None,
    progress: Optional[ProgressCallback] = None,
    fallback_pool_first: int = 0,
) -> InferenceResult:
    """End-to-end inference: generate -> VQE-score -> (optionally) feedback -> validate -> explain.

    Parameters
    ----------
    checkpoint_path
        Path to a MolGAN checkpoint.  If absent we use an untrained MolGAN
        (mostly useful for unit testing).
    n_molecules
        Number of candidate molecules sampled per iteration.
    iterations
        How many generate -> score -> fine-tune cycles to run.  ``1`` (the
        default) is pure inference: no fine-tuning, just sample + score.
    fine_tune_epochs
        Used only when ``iterations > 1``.
    base_corpus
        Reference SMILES list to mix into the feedback dataset.  Auto-loaded
        from the ZINC subset if not provided.
    top_k
        Number of best (lowest-energy) candidates to keep at the end.
    enable_hardware_validation
        Re-run VQE on Qiskit Aer for the top candidate as a cross-backend
        sanity check.
    enable_explanation
        Call Gemini to produce a natural-language explanation.  Requires
        ``GOOGLE_API_KEY`` in ``.env``.
    scorer_kwargs
        Forwarded to ``VQEScorer`` (e.g. override ``vqe_steps``).
    progress
        Optional ``progress(msg, fraction)`` callback for UIs.

    Returns
    -------
    InferenceResult
        Structured payload ready for rendering in a UI.
    """
    started = time.time()
    fine_tune_epochs = fine_tune_epochs or SIMULATION_CONFIG.get("fine_tune_epochs", 20)
    scorer_kwargs = dict(scorer_kwargs or {})

    def _emit(msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, max(0.0, min(1.0, frac)))

    _emit("Loading MolGAN checkpoint...", 0.02)
    molgan = load_pretrained_molgan(checkpoint_path)

    scorer = VQEScorer(**scorer_kwargs)
    all_scored: List[VQEResult] = []
    iter_total = max(1, iterations)
    iter_done = 0

    # Align MolGAN sampling with the VQE budget so we don't generate hundreds
    # of oversize candidates that all get rejected.
    sampling_size_limit = scorer.max_heavy_atoms

    for it in range(1, iter_total + 1):
        base_progress = 0.05 + 0.7 * (it - 1) / iter_total
        sampling_span = 0.15 / iter_total
        _emit(f"Iteration {it}/{iter_total}: sampling candidates...", base_progress)

        def _sampling_progress(msg: str, frac: float, _bp=base_progress, _it=it) -> None:
            _emit(f"Iter {_it}: {msg}", _bp + sampling_span * frac)

        candidates = generate_candidates(
            molgan, n=n_molecules, progress=_sampling_progress,
            max_heavy_atoms=sampling_size_limit,
            fallback_pool_first=fallback_pool_first,
        )
        if not candidates:
            logger.warning("Iteration %d produced no valid SMILES.", it)
            iter_done += 1
            continue

        def _scoring_progress(msg: str, frac: float, _bp=base_progress, _it=it) -> None:
            span = 0.55 / iter_total
            _emit(f"Iter {_it}: {msg}", _bp + sampling_span + span * frac)

        results = score_candidates(candidates, scorer=scorer, progress=_scoring_progress)
        all_scored.extend(results)
        iter_done += 1

        if iterations > 1 and it < iter_total:
            good, _ = classify_molecules(results)
            good.sort(key=lambda r: r.energy)
            if not good:
                continue
            corpus_for_feedback = base_corpus if base_corpus is not None else load_zinc_subset(n=2000)
            feedback_data = build_feedback_dataset(good[: max(3, len(good) // 2)], corpus_for_feedback)
            if feedback_data:
                _emit(f"Iter {it}: fine-tuning generator...", base_progress + 0.7 / iter_total * 0.95)
                molgan.fine_tune(feedback_data, epochs=fine_tune_epochs)

    successful = sorted(
        (r for r in all_scored if r.succeeded),
        key=lambda r: r.energy,
    )
    deduped: List[VQEResult] = []
    seen: set = set()
    for r in successful:
        if r.smiles in seen:
            continue
        seen.add(r.smiles)
        deduped.append(r)
    top_candidates = deduped[:top_k]

    scaffolds = extract_scaffolds(top_candidates) if top_candidates else []
    groups = extract_functional_groups(top_candidates) if top_candidates else []

    out = InferenceResult(
        candidates=all_scored,
        top_candidates=top_candidates,
        scaffolds=scaffolds,
        functional_groups=groups,
        iterations_run=iter_done,
    )

    if enable_hardware_validation and top_candidates:
        _emit("Validating best candidate on Qiskit Aer...", 0.85)
        try:
            out.hardware_validation = validate_on_hardware(
                top_candidates[0].smiles,
                active_electrons=QUANTUM_CONFIG.get("test_active_electrons", 2),
                active_orbitals=QUANTUM_CONFIG.get("test_active_orbitals", 2),
                vqe_steps=QUANTUM_CONFIG.get("test_vqe_steps", 20),
                stepsize=QUANTUM_CONFIG.get("test_vqe_stepsize", 0.1),
            )
        except Exception as exc:
            logger.error("Hardware validation failed: %s", exc)

    if enable_explanation and top_candidates:
        _emit("Generating explanation via Gemini...", 0.95)
        try:
            from .llm.explain import explain_selection

            out.explanation = explain_selection(
                top_molecules=top_candidates,
                scaffolds=scaffolds,
                functional_groups=groups,
                hardware_result=out.hardware_validation,
                iterations=iter_total,
            )
        except Exception as exc:
            logger.error("Gemini call failed: %s", exc)
            out.explanation = f"[Explanation unavailable: {exc}]"

    out.elapsed_sec = time.time() - started
    _emit("Done.", 1.0)
    return out


def smiles_to_image_bytes(smiles: str, size: int = 320) -> Optional[bytes]:
    """Render a SMILES string to a PNG byte string for UI display.

    Returns ``None`` if RDKit is unavailable or the SMILES is invalid.
    """
    try:
        from io import BytesIO

        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(size, size))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


__all__ = [
    "DEFAULT_CHECKPOINT",
    "InferenceResult",
    "ProgressCallback",
    "SMALL_MOLECULE_FALLBACK_POOL",
    "generate_candidates",
    "load_pretrained_molgan",
    "run_inference",
    "score_candidates",
    "smiles_to_image_bytes",
    "validate_on_hardware",
]
