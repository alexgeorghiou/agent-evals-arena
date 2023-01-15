"""LLM-as-judge: rubric grading and pairwise preference.

Two families of judges share one interface:

* :class:`LLMJudge` / :class:`LLMPairwiseJudge` prompt a model for strict JSON.
* :class:`HeuristicJudge` / :class:`HeuristicPairwiseJudge` are deterministic,
  offline stand-ins (keyword and reference-overlap based) so suites run in CI
  without API keys. They are a smoke test of the plumbing, not a quality signal.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from evals_arena.providers.base import Message, Provider, Usage
from evals_arena.schema import Rubric

Preference = Literal["A", "B", "tie"]

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")
_STOPWORDS = frozenset(
    "a an the and or of to in on at for is are was were be it this that with by as from".split()
)


@dataclass
class CriterionScore:
    name: str
    score: float
    weight: float
    reason: str = ""


@dataclass
class JudgeVerdict:
    scores: list[CriterionScore]
    normalized: float
    rationale: str = ""
    usage: Usage = field(default_factory=Usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized": round(self.normalized, 4),
            "rationale": self.rationale,
            "scores": [vars(s) for s in self.scores],
        }


class JudgeParseError(ValueError):
    pass


def normalise(scores: list[CriterionScore], rubric: Rubric) -> float:
    span = rubric.scale_max - rubric.scale_min
    total_weight = sum(s.weight for s in scores) or 1.0
    value = sum(((s.score - rubric.scale_min) / span) * s.weight for s in scores) / total_weight
    return max(0.0, min(1.0, value))


def _balanced_objects(text: str) -> list[str]:
    """Return every top-level ``{...}`` span, honouring JSON string escapes."""
    spans: list[str] = []
    depth, start, in_str, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates fences and prose)."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for raw in [*fenced, *_balanced_objects(text)]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise JudgeParseError(f"no JSON object found in judge reply: {text[:200]!r}")


def parse_rubric_reply(text: str, rubric: Rubric) -> tuple[list[CriterionScore], str]:
    data = extract_json(text)
    items = data.get("criteria") or data.get("scores") or []
    if isinstance(items, dict):  # {"accuracy": 4, ...}
        items = [{"name": k, "score": v} for k, v in items.items()]
    by_name = {
        str(item.get("name", "")).strip().lower(): item for item in items if isinstance(item, dict)
    }
    scores: list[CriterionScore] = []
    for criterion in rubric.criteria:
        item = by_name.get(criterion.name.lower())
        if item is None:
            scores.append(
                CriterionScore(criterion.name, rubric.scale_min, criterion.weight, "not scored")
            )
            continue
        try:
            raw = float(item.get("score"))
        except (TypeError, ValueError) as exc:
            raise JudgeParseError(f"criterion '{criterion.name}' has no numeric score") from exc
        clamped = max(rubric.scale_min, min(rubric.scale_max, raw))
        scores.append(
            CriterionScore(criterion.name, clamped, criterion.weight, str(item.get("reason", "")))
        )
    return scores, str(data.get("rationale", ""))


def parse_preference(text: str) -> Preference:
    data = extract_json(text)
    winner = str(data.get("winner", "")).strip().upper()
    if winner in {"A", "B"}:
        return winner  # type: ignore[return-value]
    if winner in {"TIE", "DRAW", "EQUAL"}:
        return "tie"
    raise JudgeParseError(f"unexpected winner value: {winner!r}")


# --------------------------------------------------------------------- interfaces


class Judge(abc.ABC):
    @abc.abstractmethod
    def grade(
        self, rubric: Rubric, *, task: str, output: str, reference: str | None = None
    ) -> JudgeVerdict: ...


class PairwiseJudge(abc.ABC):
    @abc.abstractmethod
    def prefer(
        self, *, task: str, output_a: str, output_b: str, reference: str | None = None
    ) -> Preference: ...


def prefer_with_swap(
    judge: PairwiseJudge, *, task: str, output_a: str, output_b: str, reference: str | None = None
) -> Preference:
    """Ask twice with positions swapped; disagreement (position bias) becomes a tie."""
    first = judge.prefer(task=task, output_a=output_a, output_b=output_b, reference=reference)
    swapped = judge.prefer(task=task, output_a=output_b, output_b=output_a, reference=reference)
    second: Preference = {"A": "B", "B": "A", "tie": "tie"}[swapped]  # type: ignore[assignment]
    return first if first == second else "tie"


# ---------------------------------------------------------------------- LLM judges

RUBRIC_PROMPT = """You are a strict evaluator. Grade the RESPONSE to the TASK against each
criterion on an integer scale from {lo} to {hi}.
{instructions}
Criteria:
{criteria}

TASK:
{task}
{reference_block}
RESPONSE:
{output}

Reply with JSON only, shaped exactly like:
{{"criteria": [{{"name": "<criterion name>", "score": <int>, "reason": "<short reason>"}}],
 "rationale": "<one sentence overall>"}}"""

PAIRWISE_PROMPT = """You compare two responses to the same TASK and pick the better one.
Judge correctness first, then helpfulness and concision. Ignore response order and length.

TASK:
{task}
{reference_block}
RESPONSE A:
{a}

RESPONSE B:
{b}

Reply with JSON only: {{"winner": "A" | "B" | "tie", "reason": "<short reason>"}}"""


def _reference_block(reference: str | None) -> str:
    return f"\nREFERENCE ANSWER (for grading only):\n{reference}\n" if reference else ""


class LLMJudge(Judge):
    def __init__(self, provider: Provider, *, max_tokens: int = 2048) -> None:
        self.provider = provider
        self.max_tokens = max_tokens

    def grade(
        self, rubric: Rubric, *, task: str, output: str, reference: str | None = None
    ) -> JudgeVerdict:
        criteria = "\n".join(
            f"- {c.name} (weight {c.weight:g}): {c.description}" for c in rubric.criteria
        )
        prompt = RUBRIC_PROMPT.format(
            lo=rubric.scale_min,
            hi=rubric.scale_max,
            instructions=rubric.instructions,
            criteria=criteria,
            task=task,
            reference_block=_reference_block(reference),
            output=output or "<empty>",
        )
        completion = self.provider.complete(
            [Message.user(prompt)], tools=[], max_tokens=self.max_tokens
        )
        scores, rationale = parse_rubric_reply(completion.text, rubric)
        return JudgeVerdict(scores, normalise(scores, rubric), rationale, completion.usage)


class LLMPairwiseJudge(PairwiseJudge):
    def __init__(self, provider: Provider, *, max_tokens: int = 1024) -> None:
        self.provider = provider
        self.max_tokens = max_tokens

    def prefer(
        self, *, task: str, output_a: str, output_b: str, reference: str | None = None
    ) -> Preference:
        prompt = PAIRWISE_PROMPT.format(
            task=task, reference_block=_reference_block(reference), a=output_a, b=output_b
        )
        completion = self.provider.complete(
            [Message.user(prompt)], tools=[], max_tokens=self.max_tokens
        )
        return parse_preference(completion.text)


# --------------------------------------------------------------- heuristic judges


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def reference_recall(output: str, reference: str) -> float:
    ref = tokens(reference)
    return len(ref & tokens(output)) / len(ref) if ref else 0.0


class HeuristicJudge(Judge):
    """Keyword/reference-overlap judge. Deterministic; for offline runs only."""

    def grade(
        self, rubric: Rubric, *, task: str, output: str, reference: str | None = None
    ) -> JudgeVerdict:
        lowered = output.lower()
        span = rubric.scale_max - rubric.scale_min
        scores: list[CriterionScore] = []
        for c in rubric.criteria:
            if c.keywords:
                hits = [k for k in c.keywords if k.lower() in lowered]
                frac = len(hits) / len(c.keywords)
                reason = f"matched {len(hits)}/{len(c.keywords)} keywords"
            elif reference:
                frac = reference_recall(output, reference)
                reason = f"reference recall {frac:.2f}"
            else:
                frac = 0.5 if output.strip() else 0.0
                reason = "no keywords or reference; neutral score"
            scores.append(
                CriterionScore(c.name, round(rubric.scale_min + frac * span, 2), c.weight, reason)
            )
        return JudgeVerdict(scores, normalise(scores, rubric), "heuristic judge")


class HeuristicPairwiseJudge(PairwiseJudge):
    def __init__(self, margin: float = 0.05) -> None:
        self.margin = margin

    def prefer(
        self, *, task: str, output_a: str, output_b: str, reference: str | None = None
    ) -> Preference:
        basis = reference or task
        a, b = reference_recall(output_a, basis), reference_recall(output_b, basis)
        if abs(a - b) <= self.margin:
            return "tie"
        return "A" if a > b else "B"
