"""Assertion evaluators.

Each evaluator receives the assertion model plus an :class:`EvalContext` and
returns an :class:`AssertionResult` with a pass flag and a 0..1 score. Binary
checks score 0 or 1; the judge assertion contributes its normalised rubric score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jsonschema

from evals_arena.judge import Judge, JudgeParseError
from evals_arena.matching import explain_mismatch, matches
from evals_arena.results import AssertionResult
from evals_arena.sandbox import ToolInvocation
from evals_arena.schema import (
    Case,
    ContainsAssertion,
    CostAssertion,
    EqualsAssertion,
    JsonSchemaAssertion,
    JudgeAssertion,
    LatencyAssertion,
    MaxToolCallsAssertion,
    NoToolErrorsAssertion,
    NotContainsAssertion,
    RegexAssertion,
    SequenceStep,
    Suite,
    ToolCalledAssertion,
    ToolNotCalledAssertion,
    ToolSequenceAssertion,
)


@dataclass
class EvalContext:
    suite: Suite
    case: Case
    output: str
    trajectory: list[ToolInvocation]
    latency_ms: float
    cost_usd: float
    judge: Judge | None = None


Evaluator = Callable[[Any, EvalContext], AssertionResult]
_REGISTRY: dict[type, Evaluator] = {}


def evaluator(kind: type) -> Callable[[Evaluator], Evaluator]:
    def register(fn: Evaluator) -> Evaluator:
        _REGISTRY[kind] = fn
        return fn

    return register


def _result(a: Any, passed: bool, message: str = "", **details: Any) -> AssertionResult:
    return AssertionResult(
        type=a.type,
        name=a.name or a.type,
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=a.weight,
        message=message,
        details=details,
    )


def evaluate(assertion: Any, ctx: EvalContext) -> AssertionResult:
    fn = _REGISTRY.get(type(assertion))
    if fn is None:  # pragma: no cover - guarded by the schema union
        raise TypeError(f"no evaluator registered for {type(assertion).__name__}")
    return fn(assertion, ctx)


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


# ------------------------------------------------------------------ output checks


@evaluator(ContainsAssertion)
def _contains(a: ContainsAssertion, ctx: EvalContext) -> AssertionResult:
    haystack = ctx.output if a.case_sensitive else ctx.output.lower()
    needles = _as_list(a.value)
    found = [n for n in needles if (n if a.case_sensitive else n.lower()) in haystack]
    missing = [n for n in needles if n not in found]
    passed = bool(found) if a.mode == "any" else not missing
    return _result(a, passed, "" if passed else f"missing: {missing}", found=found)


@evaluator(NotContainsAssertion)
def _not_contains(a: NotContainsAssertion, ctx: EvalContext) -> AssertionResult:
    haystack = ctx.output if a.case_sensitive else ctx.output.lower()
    present = [n for n in _as_list(a.value) if (n if a.case_sensitive else n.lower()) in haystack]
    return _result(a, not present, f"unexpected: {present}" if present else "")


@evaluator(EqualsAssertion)
def _equals(a: EqualsAssertion, ctx: EvalContext) -> AssertionResult:
    actual = ctx.output.strip() if a.strip else ctx.output
    expected = a.value.strip() if a.strip else a.value
    return _result(a, actual == expected, "" if actual == expected else f"got {actual[:80]!r}")


_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


@evaluator(RegexAssertion)
def _regex(a: RegexAssertion, ctx: EvalContext) -> AssertionResult:
    flags = 0
    for f in a.flags:
        flags |= _FLAGS[f]
    match = re.search(a.pattern, ctx.output, flags)
    return _result(a, match is not None, "" if match else f"/{a.pattern}/ did not match")


def _json_payload(text: str) -> Any:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    return json.loads(fence.group(1) if fence else stripped)


@evaluator(JsonSchemaAssertion)
def _json_schema(a: JsonSchemaAssertion, ctx: EvalContext) -> AssertionResult:
    try:
        payload = _json_payload(ctx.output)
    except json.JSONDecodeError as exc:
        return _result(a, False, f"output is not JSON: {exc.msg}")
    try:
        jsonschema.validate(payload, a.schema_)
    except jsonschema.ValidationError as exc:
        return _result(a, False, f"schema violation at {list(exc.absolute_path)}: {exc.message}")
    return _result(a, True)


# -------------------------------------------------------------- trajectory checks


def _step_matches(step: SequenceStep, inv: ToolInvocation) -> bool:
    return inv.name == step.tool and (step.args is None or matches(step.args, inv.arguments))


def _calls_to(tool: str, args: dict[str, Any] | None, ctx: EvalContext) -> list[ToolInvocation]:
    return [
        inv
        for inv in ctx.trajectory
        if inv.name == tool and (args is None or matches(args, inv.arguments))
    ]


@evaluator(ToolCalledAssertion)
def _tool_called(a: ToolCalledAssertion, ctx: EvalContext) -> AssertionResult:
    hits = _calls_to(a.tool, a.args, ctx)
    too_few = len(hits) < a.min_times
    too_many = a.max_times is not None and len(hits) > a.max_times
    if too_few and a.args is not None:
        near = [inv for inv in ctx.trajectory if inv.name == a.tool]
        reason = (
            explain_mismatch(a.args, near[0].arguments) if near else f"'{a.tool}' never called"
        )
        return _result(a, False, f"no call matched args ({reason})", calls=len(hits))
    if too_few or too_many:
        bound = f">= {a.min_times}" + (f" and <= {a.max_times}" if a.max_times is not None else "")
        return _result(a, False, f"'{a.tool}' called {len(hits)}x, expected {bound}")
    return _result(a, True, calls=len(hits))


@evaluator(ToolNotCalledAssertion)
def _tool_not_called(a: ToolNotCalledAssertion, ctx: EvalContext) -> AssertionResult:
    hits = _calls_to(a.tool, a.args, ctx)
    return _result(a, not hits, f"'{a.tool}' was called {len(hits)}x" if hits else "")


def sequence_matches(
    steps: list[SequenceStep], trajectory: list[ToolInvocation], mode: str
) -> bool:
    if mode == "exact":
        return len(steps) == len(trajectory) and all(
            _step_matches(s, inv) for s, inv in zip(steps, trajectory, strict=True)
        )
    if mode == "contiguous":
        n = len(steps)
        return any(
            all(_step_matches(s, inv) for s, inv in zip(steps, trajectory[i : i + n], strict=True))
            for i in range(len(trajectory) - n + 1)
        )
    # subsequence: greedy matching is optimal for "is a subsequence" checks
    it = iter(trajectory)
    return all(any(_step_matches(step, inv) for inv in it) for step in steps)


@evaluator(ToolSequenceAssertion)
def _tool_sequence(a: ToolSequenceAssertion, ctx: EvalContext) -> AssertionResult:
    steps = a.normalised_steps()
    ok = sequence_matches(steps, ctx.trajectory, a.mode)
    actual = [inv.name for inv in ctx.trajectory]
    expected = [s.tool for s in steps]
    return _result(
        a, ok, "" if ok else f"{a.mode} {expected} not found in {actual}", actual=actual
    )


@evaluator(MaxToolCallsAssertion)
def _max_tool_calls(a: MaxToolCallsAssertion, ctx: EvalContext) -> AssertionResult:
    calls = [inv for inv in ctx.trajectory if a.tool is None or inv.name == a.tool]
    ok = len(calls) <= a.max
    return _result(a, ok, "" if ok else f"{len(calls)} calls > max {a.max}", calls=len(calls))


@evaluator(NoToolErrorsAssertion)
def _no_tool_errors(a: NoToolErrorsAssertion, ctx: EvalContext) -> AssertionResult:
    errors = [f"{inv.name}: {inv.error}" for inv in ctx.trajectory if inv.error]
    return _result(a, not errors, "; ".join(errors))


# ---------------------------------------------------------------- budget checks


@evaluator(LatencyAssertion)
def _latency(a: LatencyAssertion, ctx: EvalContext) -> AssertionResult:
    ok = ctx.latency_ms <= a.max_ms
    msg = "" if ok else f"{ctx.latency_ms:.0f}ms > {a.max_ms:.0f}ms"
    return _result(a, ok, msg, latency_ms=ctx.latency_ms)


@evaluator(CostAssertion)
def _cost(a: CostAssertion, ctx: EvalContext) -> AssertionResult:
    ok = ctx.cost_usd <= a.max_usd
    msg = "" if ok else f"${ctx.cost_usd:.5f} > ${a.max_usd:.5f}"
    return _result(a, ok, msg, cost_usd=ctx.cost_usd)


# ------------------------------------------------------------------------- judge


@evaluator(JudgeAssertion)
def _judge(a: JudgeAssertion, ctx: EvalContext) -> AssertionResult:
    if ctx.judge is None:
        return _result(a, False, "no judge configured")
    rubric = ctx.suite.resolve_rubric(a.rubric)
    try:
        verdict = ctx.judge.grade(
            rubric, task=ctx.case.input, output=ctx.output, reference=ctx.case.reference
        )
    except JudgeParseError as exc:
        return _result(a, False, f"judge reply unparseable: {exc}")
    passed = verdict.normalized >= a.min_score
    return AssertionResult(
        type=a.type,
        name=a.name or (a.rubric if isinstance(a.rubric, str) else "judge"),
        passed=passed,
        score=verdict.normalized,
        weight=a.weight,
        message=f"{verdict.normalized:.2f} vs min {a.min_score:.2f}",
        details=verdict.to_dict(),
    )
