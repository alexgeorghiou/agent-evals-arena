"""A minimal, provider-agnostic tool-calling agent loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from evals_arena.providers.base import Message, Provider, ProviderError, RequestContext, Usage
from evals_arena.sandbox import MockToolSandbox, ToolInvocation
from evals_arena.schema import AgentConfig


@dataclass
class AgentTrace:
    output: str = ""
    messages: list[Message] = field(default_factory=list)
    trajectory: list[ToolInvocation] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    turns: int = 0
    stop_reason: str = "end_turn"
    error: str | None = None


def run_agent(
    provider: Provider,
    user_input: str,
    sandbox: MockToolSandbox,
    config: AgentConfig,
    context: RequestContext | None = None,
) -> AgentTrace:
    """Drive ``provider`` until it answers without tool calls or ``max_turns`` is hit.

    Latency is the sum of model latencies (provider-reported when available, else
    wall clock). Mock tools are instantaneous, so they never skew the numbers.
    """
    trace = AgentTrace(messages=[Message.user(user_input)])
    specs = sandbox.specs()
    for turn in range(config.max_turns):
        started = time.perf_counter()
        try:
            completion = provider.complete(
                trace.messages,
                tools=specs,
                system=config.system_prompt,
                max_tokens=config.max_tokens,
                context=context,
            )
        except ProviderError as exc:
            trace.error = str(exc)
            trace.stop_reason = "error"
            break
        elapsed = (time.perf_counter() - started) * 1000
        trace.latency_ms += completion.latency_ms if completion.latency_ms is not None else elapsed
        trace.usage = trace.usage + completion.usage
        trace.turns = turn + 1
        trace.messages.append(
            Message(
                role="assistant",
                content=completion.text,
                tool_calls=list(completion.tool_calls),
                provider_payload=completion.raw,
            )
        )
        if not completion.tool_calls:
            trace.output = completion.text
            trace.stop_reason = completion.stop_reason
            break
        for call in completion.tool_calls:
            record = sandbox.invoke(call, turn)
            trace.messages.append(
                Message.tool_result(call.id, sandbox.render(record), is_error=not record.ok)
            )
    else:
        trace.stop_reason = "max_turns"
        trace.output = trace.messages[-1].content if trace.messages else ""
    trace.trajectory = list(sandbox.trajectory)
    return trace
