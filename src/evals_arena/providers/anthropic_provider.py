"""Anthropic Messages API adapter (``pip install agent-evals-arena[anthropic]``)."""

from __future__ import annotations

import os
import time
from typing import Any

from evals_arena.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderError,
    RequestContext,
    ToolCall,
    ToolSpec,
    Usage,
)

DEFAULT_MODEL = "claude-opus-5-5"


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages; consecutive tool results are merged into one user turn."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            if m.provider_payload is not None:
                out.append({"role": "assistant", "content": m.provider_payload})
                continue
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                for tc in m.tool_calls
            )
            out.append({"role": "assistant", "content": blocks or m.content})
        else:
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
                "is_error": m.is_error,
            }
            prev = out[-1] if out else None
            if (
                prev is not None
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and all(b.get("type") == "tool_result" for b in prev["content"])
            ):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def from_anthropic_response(resp: Any, latency_ms: float) -> Completion:
    texts: list[str] = []
    calls: list[ToolCall] = []
    raw: list[dict[str, Any]] = []
    for block in resp.content:
        raw.append(block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block)
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    usage = Usage(int(resp.usage.input_tokens), int(resp.usage.output_tokens))
    return Completion(
        text="".join(texts),
        tool_calls=calls,
        usage=usage,
        stop_reason=str(resp.stop_reason or "end_turn"),
        latency_ms=latency_ms,
        raw=raw,
    )


class AnthropicProvider(Provider):
    """Calls Claude models through the official ``anthropic`` SDK.

    Server-side refusal fallbacks are intentionally *not* enabled: an eval harness
    must score the model it was asked to score, so a refusal is recorded as such
    instead of being silently answered by a different model.
    """

    vendor = "anthropic"

    def __init__(self, model: str | None = None, *, client: Any = None) -> None:
        super().__init__(model or os.getenv("ARENA_ANTHROPIC_MODEL", DEFAULT_MODEL))
        self.effort = os.getenv("ARENA_ANTHROPIC_EFFORT")  # low|medium|high|xhigh|max
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ProviderError(
                    "anthropic SDK not installed: pip install 'agent-evals-arena[anthropic]'"
                ) from exc
            client = anthropic.Anthropic()
        self._client = client

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec],
        system: str | None = None,
        max_tokens: int = 4096,
        context: RequestContext | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        start = time.perf_counter()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # SDK raises typed errors; surface them uniformly
            raise ProviderError(f"anthropic request failed: {exc}") from exc
        return from_anthropic_response(resp, (time.perf_counter() - start) * 1000)
