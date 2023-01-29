"""A deterministic, scripted provider used for tests, demos and CI.

Scripts live next to suites in ``*.fake.yaml`` files::

    suite: weather-agent
    models:
      fake-strong:
        paris-today:
          - call: get_weather
            args: {city: Paris}
            latency_ms: 420
          - say: "It is {{get_weather.temp_c}} C and {{get_weather.condition}} in Paris."

The provider is stateless: the step to play is derived from how many assistant
turns already exist in the conversation, so a run can be replayed exactly.
``{{tool.path}}`` placeholders are filled from the most recent result of that tool,
which means the scripted answer reflects whatever the mock tool returned.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from evals_arena.providers.base import (
    Completion,
    Message,
    Provider,
    RequestContext,
    ToolCall,
    ToolSpec,
    Usage,
)

Script = list[dict[str, Any]]
ScriptBook = dict[str, dict[str, Script]]  # suite -> case_id -> steps

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][\w\-]*)((?:\.[\w\-]+)*)\s*\}\}")
FALLBACK_TEXT = "I'm not able to help with that request."


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token); good enough for deterministic fakes."""
    return max(1, (len(text) + 3) // 4)


def load_fake_scripts(paths: Iterable[str | Path], model: str) -> ScriptBook:
    """Collect scripts for ``model`` from every ``*.fake.yaml`` under ``paths``."""
    book: ScriptBook = {}
    for item in paths:
        p = Path(item)
        files = sorted(p.rglob("*.fake.yaml")) if p.is_dir() else [p]
        for f in files:
            if not f.name.endswith(".fake.yaml"):
                sibling = f.with_name(f.stem + ".fake.yaml")
                if not sibling.exists():
                    continue
                f = sibling
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            suite = str(data.get("suite", ""))
            cases = _resolve_model(data.get("models") or {}, model)
            book.setdefault(suite, {}).update(cases)
    return book


def _resolve_model(models: dict[str, Any], model: str, seen: tuple[str, ...] = ()) -> Script | Any:
    """Return ``model``'s case scripts, following ``_inherit`` chains."""
    if model in seen:
        raise ValueError(f"cyclic _inherit chain: {' -> '.join((*seen, model))}")
    own = dict(models.get(model) or {})
    parent = own.pop("_inherit", None)
    if parent is None:
        return own
    return {**_resolve_model(models, str(parent), (*seen, model)), **own}


class FakeProvider(Provider):
    vendor = "fake"

    def __init__(
        self,
        model: str = "fake-strong",
        scripts: ScriptBook | None = None,
        *,
        default_latency_ms: float = 250.0,
    ) -> None:
        super().__init__(model)
        self.scripts: ScriptBook = scripts or {}
        self.default_latency_ms = default_latency_ms
        self.calls = 0

    # ------------------------------------------------------------------ helpers
    def _script_for(self, context: RequestContext | None) -> Script | None:
        if context is None:
            return None
        by_suite = self.scripts.get(context.suite) or self.scripts.get("") or {}
        return by_suite.get(context.case_id)

    @staticmethod
    def _tool_results(messages: list[Message]) -> dict[str, Any]:
        names: dict[str, str] = {}
        latest: dict[str, Any] = {}
        for m in messages:
            for tc in m.tool_calls:
                names[tc.id] = tc.name
            if m.role == "tool" and m.tool_call_id in names:
                try:
                    value: Any = json.loads(m.content)
                except json.JSONDecodeError:
                    value = m.content
                latest[names[m.tool_call_id]] = value
        return latest

    @classmethod
    def _render(cls, template: str, messages: list[Message]) -> str:
        results = cls._tool_results(messages)

        def repl(match: re.Match[str]) -> str:
            value: Any = results.get(match.group(1))
            for key in filter(None, match.group(2).split(".")):
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
                    value = value[int(key)]
                else:
                    value = None
            if value is None:
                return "unknown"
            return value if isinstance(value, str) else json.dumps(value)

        return _PLACEHOLDER.sub(repl, template)

    # -------------------------------------------------------------- interface
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec],
        system: str | None = None,
        max_tokens: int = 4096,
        context: RequestContext | None = None,
    ) -> Completion:
        self.calls += 1
        turn = sum(1 for m in messages if m.role == "assistant")
        script = self._script_for(context)
        step: dict[str, Any] = (
            script[turn] if script is not None and turn < len(script) else {"say": FALLBACK_TEXT}
        )

        tool_calls: list[ToolCall] = []
        raw_calls = step.get("calls") or ([step] if "call" in step else [])
        for i, raw in enumerate(raw_calls):
            name = raw.get("call") or raw.get("name")
            tool_calls.append(
                ToolCall(id=f"call_{turn}_{i}", name=str(name), arguments=dict(raw.get("args", {})))
            )
        text = self._render(str(step.get("say", "")), messages) if not tool_calls else str(
            step.get("say", "")
        )

        prompt = (system or "") + "".join(m.content for m in messages)
        prompt += json.dumps([t.name for t in tools])
        generated = text + "".join(json.dumps(tc.arguments) + tc.name for tc in tool_calls)
        usage = Usage(estimate_tokens(prompt), estimate_tokens(generated))
        return Completion(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason="tool_use" if tool_calls else "end_turn",
            latency_ms=float(step.get("latency_ms", self.default_latency_ms)),
        )
