"""Execute a suite against one provider and score every case."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

from evals_arena.agent import run_agent
from evals_arena.assertions import EvalContext, evaluate
from evals_arena.judge import Judge
from evals_arena.pricing import PriceTable
from evals_arena.providers.base import Provider, RequestContext
from evals_arena.results import AssertionResult, CaseResult, RunResult
from evals_arena.sandbox import MockToolSandbox
from evals_arena.schema import Case, Suite

ProgressFn = Callable[[CaseResult], None]


def new_run_id(suite: str, model: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"{suite}--{slug}--{stamp}-{uuid.uuid4().hex[:6]}"


def score_case(results: list[AssertionResult]) -> float:
    total = sum(r.weight for r in results)
    if total == 0:
        return 1.0 if all(r.passed for r in results) else 0.0
    return sum(r.score * r.weight for r in results) / total


@dataclass
class Runner:
    provider: Provider
    judge: Judge | None = None
    prices: PriceTable = field(default_factory=PriceTable)
    concurrency: int = 1
    tags: set[str] = field(default_factory=set)

    def select(self, suite: Suite) -> list[Case]:
        if not self.tags:
            return list(suite.cases)
        return [c for c in suite.cases if self.tags & set(c.tags)]

    def run_case(self, suite: Suite, case: Case) -> CaseResult:
        sandbox = MockToolSandbox(suite.tools_for(case))
        trace = run_agent(
            self.provider,
            case.input,
            sandbox,
            suite.agent,
            RequestContext(suite=suite.name, case_id=case.id),
        )
        cost = self.prices.cost(self.provider.model, trace.usage)
        ctx = EvalContext(
            suite=suite,
            case=case,
            output=trace.output,
            trajectory=trace.trajectory,
            latency_ms=trace.latency_ms,
            cost_usd=cost,
            judge=self.judge,
        )
        assertion_results = [evaluate(a, ctx) for a in suite.assertions_for(case)]
        passed = trace.error is None and all(r.passed for r in assertion_results)
        score = 0.0 if trace.error else score_case(assertion_results)
        return CaseResult(
            case_id=case.id,
            input=case.input,
            passed=passed,
            score=round(score, 4),
            output=trace.output,
            trajectory=[inv.to_dict() for inv in trace.trajectory],
            assertions=assertion_results,
            latency_ms=round(trace.latency_ms, 2),
            input_tokens=trace.usage.input_tokens,
            output_tokens=trace.usage.output_tokens,
            cost_usd=round(cost, 8),
            turns=trace.turns,
            stop_reason=trace.stop_reason,
            error=trace.error,
            tags=list(case.tags),
        )

    def run(self, suite: Suite, progress: ProgressFn | None = None) -> RunResult:
        run = RunResult(
            run_id=new_run_id(suite.name, self.provider.model),
            suite=suite.name,
            suite_version=suite.version,
            model=self.provider.model,
            provider=self.provider.vendor,
            metadata={"judge": type(self.judge).__name__ if self.judge else None},
        )
        cases = self.select(suite)

        def work(case: Case) -> CaseResult:
            result = self.run_case(suite, case)
            if progress:
                progress(result)
            return result

        if self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                run.cases = list(pool.map(work, cases))
        else:
            run.cases = [work(c) for c in cases]
        return run.finalise()
