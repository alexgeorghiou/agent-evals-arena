"""Build rubric / pairwise judges from a CLI spec."""

from __future__ import annotations

import os

from evals_arena.judge import (
    HeuristicJudge,
    HeuristicPairwiseJudge,
    Judge,
    LLMJudge,
    LLMPairwiseJudge,
    PairwiseJudge,
)
from evals_arena.providers import create_provider

HEURISTIC = "heuristic"


def resolve_spec(spec: str | None) -> str:
    return (spec or os.getenv("ARENA_JUDGE") or HEURISTIC).strip()


def make_judge(spec: str | None) -> Judge:
    spec = resolve_spec(spec)
    if spec == HEURISTIC:
        return HeuristicJudge()
    return LLMJudge(create_provider(spec))


def make_pairwise_judge(spec: str | None) -> PairwiseJudge:
    spec = resolve_spec(spec)
    if spec == HEURISTIC:
        return HeuristicPairwiseJudge()
    return LLMPairwiseJudge(create_provider(spec))
