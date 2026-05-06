"""Percentile-based classification of scored molecules."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..config import SIMULATION_CONFIG
from ..quantum.vqe import VQEResult


def classify_molecules(
    results: List[VQEResult],
    percentile: float = None,
) -> Tuple[List[VQEResult], List[VQEResult]]:
    """Split scored molecules into ``(good, bad)`` by VQE energy percentile.

    Lower energies are better, so molecules in the bottom ``percentile``%
    of the score distribution are the "good" ones.  Failed scorings (infinite
    energy) are always classified as bad.
    """
    if percentile is None:
        percentile = SIMULATION_CONFIG.get("classification_percentile", 30)

    finite = [r for r in results if math.isfinite(r.energy)]
    failed = [r for r in results if not math.isfinite(r.energy)]

    if not finite:
        return [], failed

    energies = np.array([r.energy for r in finite])
    threshold = float(np.percentile(energies, percentile))

    good = [r for r in finite if r.energy <= threshold]
    bad = [r for r in finite if r.energy > threshold] + failed
    return good, bad
