"""Result models produced by a run and consumed by gating, the Elo arena and the UI."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from evals_arena.stats import bootstrap_ci, mean, percentile


class AssertionResult(BaseModel):
    type: str
    name: str
    passed: bool
    score: float = Field(ge=0, le=1)
    weight: float = 1.0
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    score: float
    output: str
    input: str = ""
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    assertions: list[AssertionResult] = Field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    stop_reason: str = ""
    error: str | None = None
    tags: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    cases: int
    passed: int
    pass_rate: float
    score: float
    score_ci_low: float
    score_ci_high: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    total_cost_usd: float
    input_tokens: int
    output_tokens: int

    @classmethod
    def from_cases(cls, cases: list[CaseResult]) -> RunSummary:
        scores = [c.score for c in cases]
        latencies = [c.latency_ms for c in cases]
        low, high = bootstrap_ci(scores)
        passed = sum(c.passed for c in cases)
        return cls(
            cases=len(cases),
            passed=passed,
            pass_rate=passed / len(cases) if cases else 0.0,
            score=mean(scores),
            score_ci_low=low,
            score_ci_high=high,
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            latency_mean_ms=mean(latencies),
            total_cost_usd=sum(c.cost_usd for c in cases),
            input_tokens=sum(c.input_tokens for c in cases),
            output_tokens=sum(c.output_tokens for c in cases),
        )


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunResult(BaseModel):
    run_id: str
    suite: str
    suite_version: str = "1"
    model: str
    provider: str
    started_at: str = Field(default_factory=utcnow)
    finished_at: str | None = None
    cases: list[CaseResult] = Field(default_factory=list)
    summary: RunSummary | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def finalise(self) -> RunResult:
        self.summary = RunSummary.from_cases(self.cases)
        self.finished_at = utcnow()
        return self

    def case(self, case_id: str) -> CaseResult | None:
        return next((c for c in self.cases if c.case_id == case_id), None)
