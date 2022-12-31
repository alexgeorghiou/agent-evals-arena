"""Provider-neutral chat types and the provider interface.

The agent loop talks to every model through :class:`Provider`. Adapters convert
these neutral messages into vendor formats, which keeps the harness (and the
trajectory recorder) independent of any single SDK.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=text)

    @classmethod
    def tool_result(cls, call_id: str, content: str, *, is_error: bool = False) -> Message:
        return cls(role="tool", content=content, tool_call_id=call_id, is_error=is_error)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    latency_ms: float | None = None
    """Provider-reported latency. Fakes use this to simulate time deterministically."""


@dataclass(frozen=True)
class RequestContext:
    """Metadata about the request. Real providers ignore it; fakes use ``case_id``."""

    suite: str = ""
    case_id: str = ""


class ProviderError(RuntimeError):
    pass


class Provider(abc.ABC):
    """A chat model that supports tool calling."""

    #: vendor key, e.g. ``anthropic`` / ``openai`` / ``fake``
    vendor: str = "abstract"

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def label(self) -> str:
        return f"{self.vendor}:{self.model}"

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec],
        system: str | None = None,
        max_tokens: int = 4096,
        context: RequestContext | None = None,
    ) -> Completion:
        """Run one model turn and return text and/or tool calls."""
