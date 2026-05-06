"""Feedback construction: turn top-scoring molecules into a fine-tune set.

The original ``feedback.py`` was a one-line identity stub.  In the new
pipeline the feedback step builds an *actual* dataset that the MolGAN's
:py:meth:`~src.models.molgan.MolGAN.fine_tune` consumes between iterations.

Strategy
--------
1.  Take the top-K molecules from the previous iteration.
2.  Oversample them ``boost_factor`` times so the GAN sees them more often
    than the rest of the corpus.
3.  Mix in a random sample from the original training corpus to prevent
    catastrophic forgetting / mode collapse onto a tiny set.
"""

from __future__ import annotations

import random
from typing import List

from ..config import SIMULATION_CONFIG
from ..logging_utils import get_logger
from ..quantum.vqe import VQEResult

logger = get_logger("feedback")


def build_feedback_dataset(
    top_results: List[VQEResult],
    base_corpus: List[str],
    boost_factor: int = None,
    base_sample_size: int = None,
) -> List[str]:
    """Combine boosted top molecules with a random slice of the base corpus."""
    if not top_results:
        logger.info("Feedback: no top molecules; skipping.")
        return []

    boost_factor = boost_factor or SIMULATION_CONFIG.get("feedback_boost_factor", 8)
    base_sample_size = base_sample_size or SIMULATION_CONFIG.get(
        "feedback_base_sample_size", 200
    )

    boosted: List[str] = []
    for r in top_results:
        boosted.extend([r.smiles] * boost_factor)

    if base_corpus:
        sample_n = min(base_sample_size, len(base_corpus))
        boosted.extend(random.sample(base_corpus, sample_n))

    random.shuffle(boosted)
    logger.info(
        "Feedback dataset: %d top mols x%d boost + %d base samples = %d total",
        len(top_results), boost_factor,
        min(base_sample_size, len(base_corpus)) if base_corpus else 0,
        len(boosted),
    )
    return boosted
