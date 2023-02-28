from pathlib import Path

import pytest

from evals_arena.agent import run_agent
from evals_arena.providers import (
    Completion,
    FakeProvider,
    Message,
    Provider,
    ProviderError,
    RequestContext,
    ToolCall,
    load_fake_scripts,
)
from evals_arena.providers.fake import FALLBACK_TEXT
from evals_arena.sandbox import MockToolSandbox
from evals_arena.schema import AgentConfig

CTX = RequestContext(suite="unit", case_id="c1")


def scripted(*steps):
    return FakeProvider("fake-test", {"unit": {"c1": list(steps)}})


def test_sandbox_matches_responses_and_records(suite):
    box = MockToolSandbox(suite.tools)
    ok = box.invoke(ToolCall("1", "lookup", {"q": "cats and dogs"}), 0)
    err = box.invoke(ToolCall("2", "lookup", {"q": "broken"}), 0)
    default = box.invoke(ToolCall("3", "lookup", {"q": "zebras"}), 1)
    assert ok.result == {"answer": "cats purr"}
    assert err.error == "backend unavailable"
    assert default.result == {"answer": "nothing"}
    assert [r.name for r in box.trajectory] == ["lookup"] * 3


def test_sandbox_validates_arguments_and_unknown_tools(suite):
    box = MockToolSandbox(suite.tools)
    bad = box.invoke(ToolCall("1", "lookup", {"limit": 3}), 0)
    unknown = box.invoke(ToolCall("2", "teleport", {}), 0)
    assert "invalid arguments" in bad.error
    assert "unknown tool" in unknown.error
    assert MockToolSandbox.render(bad).startswith('{"error"')


def test_sandbox_returns_copies(suite):
    box = MockToolSandbox(suite.tools)
    first = box.invoke(ToolCall("1", "lookup", {"q": "cats"}), 0)
    first.result["answer"] = "mutated"
    assert box.invoke(ToolCall("2", "lookup", {"q": "cats"}), 0).result == {"answer": "cats purr"}


def test_agent_loop_runs_tools_and_renders_placeholders(suite):
    provider = scripted(
        {"call": "lookup", "args": {"q": "cats"}, "latency_ms": 100},
        {"say": "Fact: {{lookup.answer}}", "latency_ms": 50},
    )
    trace = run_agent(provider, "cats?", MockToolSandbox(suite.tools), AgentConfig(), CTX)
    assert trace.output == "Fact: cats purr"
    assert trace.turns == 2
    assert trace.latency_ms == 150
    assert trace.usage.input_tokens > 0
    assert [m.role for m in trace.messages] == ["user", "assistant", "tool", "assistant"]


def test_parallel_calls_in_one_turn(suite):
    provider = scripted(
        {"calls": [{"call": "lookup", "args": {"q": "cats"}}, {"call": "noop"}]},
        {"say": "done"},
    )
    trace = run_agent(provider, "x", MockToolSandbox(suite.tools), AgentConfig(), CTX)
    assert [(i.turn, i.name) for i in trace.trajectory] == [(0, "lookup"), (0, "noop")]


def test_max_turns_stops_loop(suite):
    provider = scripted(*[{"call": "noop"} for _ in range(10)])
    trace = run_agent(provider, "x", MockToolSandbox(suite.tools), AgentConfig(max_turns=3), CTX)
    assert trace.stop_reason == "max_turns"
    assert len(trace.trajectory) == 3


def test_unscripted_case_falls_back(suite):
    trace = run_agent(FakeProvider(), "x", MockToolSandbox(suite.tools), AgentConfig(), CTX)
    assert trace.output == FALLBACK_TEXT


def test_provider_errors_are_captured(suite):
    class Boom(Provider):
        vendor = "boom"

        def complete(self, messages, **kwargs) -> Completion:
            raise ProviderError("rate limited")

    trace = run_agent(Boom("m"), "x", MockToolSandbox(suite.tools), AgentConfig(), CTX)
    assert trace.stop_reason == "error"
    assert "rate limited" in trace.error


def test_fake_scripts_support_inheritance(tmp_path: Path):
    (tmp_path / "s.fake.yaml").write_text(
        "suite: s\nmodels:\n  base:\n    a: [{say: A}]\n    b: [{say: B}]\n"
        "  child:\n    _inherit: base\n    b: [{say: B2}]\n"
    )
    book = load_fake_scripts([tmp_path], "child")
    assert book["s"]["a"] == [{"say": "A"}]
    assert book["s"]["b"] == [{"say": "B2"}]


def test_fake_scripts_detect_inheritance_cycles(tmp_path: Path):
    (tmp_path / "s.fake.yaml").write_text(
        "suite: s\nmodels:\n  a: {_inherit: b}\n  b: {_inherit: a}\n"
    )
    with pytest.raises(ValueError, match="cyclic"):
        load_fake_scripts([tmp_path], "a")


def test_message_helpers():
    msg = Message.tool_result("id1", "boom", is_error=True)
    assert msg.role == "tool"
    assert msg.is_error
