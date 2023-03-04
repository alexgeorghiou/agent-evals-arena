from typing import Any

import pytest

from evals_arena.assertions import EvalContext, evaluate, sequence_matches
from evals_arena.judge import HeuristicJudge
from evals_arena.sandbox import ToolInvocation
from evals_arena.schema import Case, SequenceStep

from .conftest import make_suite


def ctx(output: str = "", trajectory=None, latency=100.0, cost=0.001, judge=None, **case):
    suite = make_suite()
    return EvalContext(
        suite=suite,
        case=Case(id="c1", input="q", **case),
        output=output,
        trajectory=trajectory or [],
        latency_ms=latency,
        cost_usd=cost,
        judge=judge,
    )


def check(assertion: dict[str, Any], context: EvalContext):
    case = Case.model_validate({"id": "x", "input": "x", "assertions": [assertion]})
    return evaluate(case.assertions[0], context)


def inv(name, turn=0, error=None, **args):
    return ToolInvocation(turn=turn, name=name, arguments=args, error=error)


TRAJ = [inv("search", q="tea"), inv("read", path="a.md"), inv("search", q="tea cakes")]


@pytest.mark.parametrize(
    ("assertion", "output", "passed"),
    [
        ({"type": "contains", "value": "Purr"}, "cats purr", True),
        ({"type": "contains", "value": "Purr", "case_sensitive": True}, "cats purr", False),
        ({"type": "contains", "value": ["a", "zzz"], "mode": "any"}, "a", True),
        ({"type": "contains", "value": ["a", "zzz"]}, "a", False),
        ({"type": "not_contains", "value": ["sorry"]}, "Sorry!", False),
        ({"type": "equals", "value": "42"}, " 42\n", True),
        ({"type": "regex", "pattern": "^total: \\d+$", "flags": ["m"]}, "x\ntotal: 7", True),
        ({"type": "regex", "pattern": "^total"}, "x\ntotal: 7", False),
    ],
)
def test_output_assertions(assertion, output, passed):
    assert check(assertion, ctx(output)).passed is passed


def test_json_schema_assertion_accepts_fenced_json():
    schema = {"type": "object", "required": ["grams"], "properties": {"grams": {"type": "number"}}}
    good = check({"type": "json_schema", "schema": schema}, ctx('```json\n{"grams": 250}\n```'))
    bad = check({"type": "json_schema", "schema": schema}, ctx('{"grams": "lots"}'))
    junk = check({"type": "json_schema", "schema": schema}, ctx("two hundred grams"))
    assert good.passed
    assert not bad.passed
    assert "grams" in bad.message
    assert "not JSON" in junk.message


def test_tool_called_with_args_and_counts():
    c = ctx(trajectory=TRAJ)
    assert check({"type": "tool_called", "tool": "search", "args": {"q": "tea*"}}, c).passed
    exact = check({"type": "tool_called", "tool": "search", "args": {"q": "coffee"}}, c)
    assert not exact.passed
    assert "expected 'coffee'" in exact.message
    assert not check({"type": "tool_called", "tool": "search", "max_times": 1}, c).passed
    assert check({"type": "tool_called", "tool": "search", "min_times": 2}, c).passed


def test_tool_not_called():
    c = ctx(trajectory=TRAJ)
    assert check({"type": "tool_not_called", "tool": "delete"}, c).passed
    assert not check({"type": "tool_not_called", "tool": "read"}, c).passed
    assert check({"type": "tool_not_called", "tool": "read", "args": {"path": "b*"}}, c).passed


@pytest.mark.parametrize(
    ("steps", "mode", "expected"),
    [
        (["search", "search"], "subsequence", True),
        (["read", "search"], "subsequence", True),
        (["read", "read"], "subsequence", False),
        (["search", "read"], "contiguous", True),
        (["search", "search"], "contiguous", False),
        (["search", "read", "search"], "exact", True),
        (["search", "read"], "exact", False),
    ],
)
def test_sequence_modes(steps, mode, expected):
    normalised = [SequenceStep(tool=s) for s in steps]
    assert sequence_matches(normalised, TRAJ, mode) is expected


def test_sequence_with_args():
    c = ctx(trajectory=TRAJ)
    steps = [{"tool": "search", "args": {"q": "tea cakes"}}, {"tool": "read"}]
    assert not check({"type": "tool_sequence", "steps": steps}, c).passed
    assert check({"type": "tool_sequence", "steps": steps[::-1]}, c).passed


def test_max_calls_and_tool_errors():
    c = ctx(trajectory=[*TRAJ, inv("read", error="denied")])
    assert not check({"type": "max_tool_calls", "max": 3}, c).passed
    assert check({"type": "max_tool_calls", "max": 2, "tool": "search"}, c).passed
    errors = check({"type": "no_tool_errors"}, c)
    assert not errors.passed
    assert "denied" in errors.message


def test_latency_and_cost_budgets():
    c = ctx(latency=1500, cost=0.02)
    assert not check({"type": "latency", "max_ms": 1000}, c).passed
    assert check({"type": "latency", "max_ms": 2000}, c).passed
    assert not check({"type": "cost", "max_usd": 0.01}, c).passed


def test_judge_assertion_uses_normalised_score():
    c = ctx("cats purr loudly", judge=HeuristicJudge())
    result = check({"type": "judge", "rubric": "basic", "min_score": 0.5, "weight": 2}, c)
    assert result.passed
    assert result.score == 1.0
    assert result.weight == 2
    assert result.details["scores"][0]["name"] == "mentions"


def test_judge_assertion_without_judge_fails():
    assert not check({"type": "judge", "rubric": "basic"}, ctx("x")).passed
