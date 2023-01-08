"""Small, dependency-free statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    if not values:
        return 0.0
    if not 0 <= pct <= 100:
        raise ValueError("pct must be within [0, 100]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def bootstrap_ci(
    values: Sequence[float], *, iterations: int = 1000, alpha: float = 0.05, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean (deterministic seed)."""
    import random

    if len(values) < 2:
        m = mean(values)
        return m, m
    rng = random.Random(seed)
    n = len(values)
    means = [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iterations)]
    return percentile(means, 100 * alpha / 2), percentile(means, 100 * (1 - alpha / 2))
