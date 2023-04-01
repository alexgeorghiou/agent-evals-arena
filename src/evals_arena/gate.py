"""Regression gating: compare fresh runs against a committed baseline.

The baseline is a small JSON file checked into the repo. CI re-runs the suites
and fails the job when a suite's score or pass rate drops beyond the policy, or
when a case that used to pass now fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from evals_arena.results import RunResult, utcnow

Status = Literal["pass", "fail", "new"]


class CaseBaseline(BaseModel):
    score: float
    passed: bool


class SuiteBaseline(BaseModel):
    score: float
    pass_rate: float
    cost_usd: float = 0.0
    latency_p95_ms: float = 0.0
    cases: dict[str, CaseBaseline] = Field(default_factory=dict)


class Baseline(BaseModel):
    version: int = 1
    updated_at: str = Field(default_factory=utcnow)
    suites: dict[str, dict[str, SuiteBaseline]] = Field(default_factory=dict)
    """suite name -> model -> baseline"""

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.model_validate(json.loads(p.read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")

    def get(self, suite: str, model: str) -> SuiteBaseline | None:
        return self.suites.get(suite, {}).get(model)

    def record(self, run: RunResult) -> None:
        summary = run.summary or run.finalise().summary
        assert summary is not None
        self.suites.setdefault(run.suite, {})[run.model] = SuiteBaseline(
            score=round(summary.score, 4),
            pass_rate=round(summary.pass_rate, 4),
            cost_usd=round(summary.total_cost_usd, 6),
            latency_p95_ms=round(summary.latency_p95_ms, 1),
            cases={c.case_id: CaseBaseline(score=c.score, passed=c.passed) for c in run.cases},
        )
        self.updated_at = utcnow()


class GatePolicy(BaseModel):
    max_score_drop: float = Field(default=0.02, ge=0)
    max_pass_rate_drop: float = Field(default=0.0, ge=0)
    fail_on_case_regression: bool = True
    max_cost_increase_pct: float | None = Field(default=None, ge=0)
    max_latency_increase_pct: float | None = Field(default=None, ge=0)
    min_score: float | None = Field(default=None, ge=0, le=1)


@dataclass
class SuiteGate:
    suite: str
    model: str
    status: Status
    score: float
    pass_rate: float
    baseline_score: float | None = None
    baseline_pass_rate: float | None = None
    regressed_cases: list[str] = field(default_factory=list)
    improved_cases: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def score_delta(self) -> float | None:
        return None if self.baseline_score is None else self.score - self.baseline_score


@dataclass
class GateResult:
    suites: list[SuiteGate]

    @property
    def passed(self) -> bool:
        return all(s.status != "fail" for s in self.suites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "suites": [
                {**vars(s), "score_delta": s.score_delta} for s in self.suites
            ],
        }


def _pct_increase(new: float, old: float) -> float:
    return 0.0 if old <= 0 else (new - old) / old * 100


def gate_run(run: RunResult, baseline: SuiteBaseline | None, policy: GatePolicy) -> SuiteGate:
    summary = run.summary or run.finalise().summary
    assert summary is not None
    result = SuiteGate(run.suite, run.model, "pass", summary.score, summary.pass_rate)
    if policy.min_score is not None and summary.score < policy.min_score:
        result.reasons.append(f"score {summary.score:.3f} below floor {policy.min_score:.3f}")
    if baseline is None:
        result.status = "fail" if result.reasons else "new"
        return result

    result.baseline_score = baseline.score
    result.baseline_pass_rate = baseline.pass_rate
    drop = baseline.score - summary.score
    if drop > policy.max_score_drop + 1e-9:
        result.reasons.append(
            f"score dropped {drop:.3f} ({baseline.score:.3f} -> {summary.score:.3f}), "
            f"allowed {policy.max_score_drop:.3f}"
        )
    pr_drop = baseline.pass_rate - summary.pass_rate
    if pr_drop > policy.max_pass_rate_drop + 1e-9:
        result.reasons.append(
            f"pass rate dropped {pr_drop:.1%} ({baseline.pass_rate:.1%} -> {summary.pass_rate:.1%})"
        )
    for case in run.cases:
        before = baseline.cases.get(case.case_id)
        if before is None:
            continue
        if before.passed and not case.passed:
            result.regressed_cases.append(case.case_id)
        elif not before.passed and case.passed:
            result.improved_cases.append(case.case_id)
    if policy.fail_on_case_regression and result.regressed_cases:
        result.reasons.append(f"cases now failing: {', '.join(result.regressed_cases)}")
    if policy.max_cost_increase_pct is not None:
        inc = _pct_increase(summary.total_cost_usd, baseline.cost_usd)
        if inc > policy.max_cost_increase_pct:
            result.reasons.append(f"cost up {inc:.0f}% (allowed {policy.max_cost_increase_pct:.0f}%)")
    if policy.max_latency_increase_pct is not None:
        inc = _pct_increase(summary.latency_p95_ms, baseline.latency_p95_ms)
        if inc > policy.max_latency_increase_pct:
            result.reasons.append(
                f"p95 latency up {inc:.0f}% (allowed {policy.max_latency_increase_pct:.0f}%)"
            )
    result.status = "fail" if result.reasons else "pass"
    return result


def gate(
    runs: list[RunResult],
    baseline: Baseline,
    policy: GatePolicy,
    *,
    baseline_model: str | None = None,
) -> GateResult:
    """Gate each run against the baseline of its own model.

    ``baseline_model`` compares every run against another model's baseline instead,
    which answers "would switching to this model regress anything?".
    """
    return GateResult(
        [gate_run(r, baseline.get(r.suite, baseline_model or r.model), policy) for r in runs]
    )
