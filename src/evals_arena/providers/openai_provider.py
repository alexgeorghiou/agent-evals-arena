"""OpenAI Chat Completions adapter (``pip install agent-evals-arena[openai]``)."""

from __future__ import annotations

import json
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

DEFAULT_MODEL = "gpt-4.1"


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def to_openai_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        else:
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_value": value}


def from_openai_response(resp: Any, latency_ms: float) -> Completion:
    choice = resp.choices[0]
    msg = choice.message
    calls = [
        ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_arguments(tc.function.arguments))
        for tc in (msg.tool_calls or [])
    ]
    usage = Usage(
        int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        int(getattr(resp.usage, "completion_tokens", 0) or 0),
    )
    return Completion(
        text=msg.content or "",
        tool_calls=calls,
        usage=usage,
        stop_reason=str(choice.finish_reason or "stop"),
        latency_ms=latency_ms,
    )


class OpenAIProvider(Provider):
    vendor = "openai"

    def __init__(self, model: str | None = None, *, client: Any = None) -> None:
        super().__init__(model or os.getenv("ARENA_OPENAI_MODEL") or DEFAULT_MODEL)
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ProviderError(
                    "openai SDK not installed: pip install 'agent-evals-arena[openai]'"
                ) from exc
            client = openai.OpenAI()
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
            "messages": to_openai_messages(messages, system),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ProviderError(f"openai request failed: {exc}") from exc
        return from_openai_response(resp, (time.perf_counter() - start) * 1000)
