"""Adapter tests use stub clients shaped like the vendor SDK responses."""

import json
from types import SimpleNamespace as NS

import pytest

from evals_arena.providers import (
    FakeProvider,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    create_provider,
    parse_spec,
)
from evals_arena.providers.anthropic_provider import (
    AnthropicProvider,
    to_anthropic_messages,
    to_anthropic_tools,
)
from evals_arena.providers.openai_provider import (
    OpenAIProvider,
    to_openai_messages,
    to_openai_tools,
)

TOOLS = [ToolSpec("get_weather", "weather", {"type": "object", "properties": {}})]
HISTORY = [
    Message.user("weather in Oslo and Lisbon?"),
    Message(
        role="assistant",
        content="Checking.",
        tool_calls=[
            ToolCall("t1", "get_weather", {"city": "Oslo"}),
            ToolCall("t2", "get_weather", {"city": "Lisbon"}),
        ],
    ),
    Message.tool_result("t1", '{"temp_c": -3}'),
    Message.tool_result("t2", "boom", is_error=True),
]


class Block(NS):
    def model_dump(self, exclude_none=False):
        return dict(vars(self))


class StubAnthropic:
    def __init__(self, response):
        self.kwargs = None
        self.messages = NS(create=self._create)
        self._response = response

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self._response


def test_anthropic_message_conversion_groups_tool_results():
    out = to_anthropic_messages(HISTORY)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0] == {"type": "text", "text": "Checking."}
    assert out[1]["content"][2]["input"] == {"city": "Lisbon"}
    results = out[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[1]["is_error"] is True
    assert to_anthropic_tools(TOOLS)[0]["input_schema"] == {"type": "object", "properties": {}}


def test_anthropic_replays_provider_payload_verbatim():
    payload = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    msg = Message(role="assistant", content="", provider_payload=payload)
    assert to_anthropic_messages([msg])[0]["content"] is payload


def test_anthropic_provider_parses_response():
    response = NS(
        content=[
            Block(type="thinking", thinking="", signature="s"),
            Block(type="text", text="Let me check."),
            Block(type="tool_use", id="tu_1", name="get_weather", input={"city": "Paris"}),
        ],
        usage=NS(input_tokens=120, output_tokens=30),
        stop_reason="tool_use",
    )
    client = StubAnthropic(response)
    provider = AnthropicProvider("claude-opus-5-5", client=client)
    completion = provider.complete(HISTORY[:1], tools=TOOLS, system="be brief", max_tokens=512)
    assert completion.text == "Let me check."
    assert completion.tool_calls == [ToolCall("tu_1", "get_weather", {"city": "Paris"})]
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (120, 30)
    assert completion.raw[0]["type"] == "thinking"
    assert client.kwargs["system"] == "be brief"
    assert client.kwargs["model"] == "claude-opus-5-5"
    assert "temperature" not in client.kwargs


def test_anthropic_errors_are_wrapped():
    def boom(**kwargs):
        raise RuntimeError("overloaded")

    provider = AnthropicProvider("claude-opus-5-5", client=NS(messages=NS(create=boom)))
    with pytest.raises(ProviderError, match="overloaded"):
        provider.complete([Message.user("hi")], tools=[])


def test_openai_message_conversion():
    out = to_openai_messages(HISTORY, "sys")
    assert out[0] == {"role": "system", "content": "sys"}
    call = out[2]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Oslo"}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"temp_c": -3}'}
    assert to_openai_tools(TOOLS)[0]["type"] == "function"


def test_openai_provider_parses_tool_calls_and_bad_json():
    calls = [
        NS(id="c1", function=NS(name="get_weather", arguments='{"city": "Rome"}')),
        NS(id="c2", function=NS(name="get_weather", arguments="{broken")),
    ]
    response = NS(
        choices=[NS(message=NS(content=None, tool_calls=calls), finish_reason="tool_calls")],
        usage=NS(prompt_tokens=50, completion_tokens=12),
    )
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return response

    client = NS(chat=NS(completions=NS(create=create)))
    completion = OpenAIProvider("gpt-4.1", client=client).complete(HISTORY[:1], tools=TOOLS)
    assert completion.text == ""
    assert completion.tool_calls[0].arguments == {"city": "Rome"}
    assert completion.tool_calls[1].arguments == {"_raw": "{broken"}
    assert completion.usage.total_tokens == 62
    assert seen["max_completion_tokens"] == 4096


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("anthropic:claude-sonnet-5-5", ("anthropic", "claude-sonnet-5-5")),
        ("claude-opus-5-5", ("anthropic", "claude-opus-5-5")),
        ("gpt-4.1", ("openai", "gpt-4.1")),
        ("fake-sloppy", ("fake", "fake-sloppy")),
        ("OpenAI:", ("openai", None)),
    ],
)
def test_parse_spec(spec, expected):
    assert parse_spec(spec) == expected


def test_parse_spec_uses_env(monkeypatch):
    monkeypatch.setenv("ARENA_MODEL", "fake:fake-fast")
    assert parse_spec(None) == ("fake", "fake-fast")


def test_create_provider_fake_and_unknown():
    provider = create_provider("fake:fake-sloppy")
    assert isinstance(provider, FakeProvider)
    assert provider.label == "fake:fake-sloppy"
    with pytest.raises(ProviderError, match="unknown provider"):
        create_provider("mystery:model")
