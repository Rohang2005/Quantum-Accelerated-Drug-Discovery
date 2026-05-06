"""End-to-end orchestrator for the quantum-AI drug discovery pipeline.

Stages
------
1.  Load and filter SMILES from ZINC (or fallback list).
2.  Train MolGAN on real graph encodings of those SMILES.
3.  Iterative feedback loop:
      - sample candidate SMILES from MolGAN
      - score each with VQE on PennyLane Lightning
      - classify good/bad by energy percentile
      - extract scaffolds + functional groups for explanation
      - build feedback dataset and fine-tune MolGAN
4.  Pick the global top-K by VQE energy.
5.  Run a second-opinion validation on Qiskit Aer for the best candidate.
6.  Hand everything to Gemini for a natural-language explanation.

The orchestrator returns a structured ``PipelineResult`` so callers (such
as a notebook or a UI) can inspect every stage rather than parsing logs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import SIMULATION_CONFIG
from ..data_loader import load_zinc_subset
from ..logging_utils import get_logger
from ..models.molgan import MolGAN
from ..quantum.hardware import HardwareValidationResult, HardwareValidator
from ..quantum.vqe import VQEResult, VQEScorer
from .classifier import classify_molecules
from .feature_extractor import extract_functional_groups, extract_scaffolds
from .feedback import build_feedback_dataset

logger = get_logger("pipeline")


@dataclass
class IterationReport:
    index: int
    candidates: List[VQEResult]
    good: List[VQEResult]
    scaffolds: List[str]
    functional_groups: List[str]


@dataclass
class PipelineResult:
    base_corpus_size: int
    iterations: List[IterationReport] = field(default_factory=list)
    final_candidates: List[VQEResult] = field(default_factory=list)
    hardware_validation: Optional[HardwareValidationResult] = None
    explanation: Optional[str] = None
    elapsed_sec: float = 0.0


def run_pipeline(
    *,
    dataset_size: int = 10_000,
    molgan_epochs: int = None,
    fine_tune_epochs: int = None,
    iterations: int = None,
    molecules_per_iter: int = None,
    final_candidates_count: int = None,
    enable_explanation: bool = True,
) -> PipelineResult:
    iterations = iterations or SIMULATION_CONFIG.get("iterations", 3)
    molecules_per_iter = molecules_per_iter or SIMULATION_CONFIG.get("molecules_per_iter", 20)
    final_candidates_count = final_candidates_count or SIMULATION_CONFIG.get(
        "final_candidates_count", 5
    )
    molgan_epochs = molgan_epochs or SIMULATION_CONFIG.get("molgan_epochs", 100)
    fine_tune_epochs = fine_tune_epochs or SIMULATION_CONFIG.get("fine_tune_epochs", 20)

    started = time.time()

    logger.info("=== Stage 1: data loading ===")
    base_corpus = load_zinc_subset(n=dataset_size)
    result = PipelineResult(base_corpus_size=len(base_corpus))

    logger.info("=== Stage 2: MolGAN pretraining ===")
    molgan = MolGAN()
    molgan.train(base_corpus, epochs=molgan_epochs)

    scorer = VQEScorer()
    all_scored: List[VQEResult] = []

    for it in range(1, iterations + 1):
        logger.info("=== Stage 3.%d/%d: feedback iteration ===", it, iterations)

        candidates_smiles = molgan.sample_smiles(molecules_per_iter)
        if not candidates_smiles:
            logger.warning("Iteration %d produced no valid candidates; skipping.", it)
            continue

        logger.info("Sampled %d candidate SMILES from MolGAN", len(candidates_smiles))
        for smi in candidates_smiles:
            logger.debug("  candidate: %s", smi)

        candidate_results = scorer.score_many(candidates_smiles)
        for r in candidate_results:
            all_scored.append(r)
            tag = "OK " if r.succeeded else r.status.value
            logger.info("  [%s] %-40s energy=%s", tag, r.smiles, _fmt_energy(r.energy))

        good, _bad = classify_molecules(candidate_results)
        good.sort(key=lambda r: r.energy)
        scaffolds = extract_scaffolds(good)
        groups = extract_functional_groups(good)

        report = IterationReport(
            index=it,
            candidates=candidate_results,
            good=good,
            scaffolds=scaffolds,
            functional_groups=groups,
        )
        result.iterations.append(report)
        logger.info(
            "Iter %d: %d good / %d total | scaffolds=%s | groups=%s",
            it, len(good), len(candidate_results), scaffolds, groups,
        )

        feedback_data = build_feedback_dataset(good[: max(3, len(good) // 2)], base_corpus)
        if feedback_data:
            molgan.fine_tune(feedback_data, epochs=fine_tune_epochs)

    successful = [r for r in all_scored if r.succeeded]
    successful.sort(key=lambda r: r.energy)
    seen = set()
    deduped = []
    for r in successful:
        if r.smiles in seen:
            continue
        seen.add(r.smiles)
        deduped.append(r)
    result.final_candidates = deduped[:final_candidates_count]

    logger.info("=== Stage 4: top candidates ===")
    if not result.final_candidates:
        logger.warning("No successful VQE results; aborting downstream stages.")
        result.elapsed_sec = time.time() - started
        return result

    for r in result.final_candidates:
        logger.info("  %-40s energy=%s qubits=%d", r.smiles, _fmt_energy(r.energy), r.n_qubits)

    logger.info("=== Stage 5: Qiskit Aer hardware validation ===")
    validator = HardwareValidator()
    best = result.final_candidates[0]
    result.hardware_validation = validator.validate(best.smiles)
    hv = result.hardware_validation
    if hv.succeeded:
        logger.info("  hardware_energy(%s) = %.4f", hv.smiles, hv.energy)
    else:
        logger.warning("  hardware validation failed: %s", hv.error)

    if enable_explanation:
        logger.info("=== Stage 6: Gemini explanation ===")
        try:
            from ..llm.explain import explain_selection

            scaffolds_all = []
            groups_all = []
            for rep in result.iterations:
                scaffolds_all.extend(rep.scaffolds)
                groups_all.extend(rep.functional_groups)

            result.explanation = explain_selection(
                top_molecules=result.final_candidates,
                scaffolds=list(dict.fromkeys(scaffolds_all)),
                functional_groups=list(dict.fromkeys(groups_all)),
                hardware_result=result.hardware_validation,
                iterations=iterations,
            )
            logger.info("Explanation generated (%d chars)", len(result.explanation or ""))
        except Exception as exc:
            logger.error("Explanation step failed: %s", exc)
            result.explanation = None

    result.elapsed_sec = time.time() - started
    logger.info("Pipeline finished in %.1fs", result.elapsed_sec)
    return result


def _fmt_energy(e: float) -> str:
    if e == float("inf") or e != e:
        return "  failed"
    return f"{e:+.4f}"
