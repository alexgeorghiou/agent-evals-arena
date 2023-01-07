"""Mock tool sandbox: serves canned responses and records the trajectory."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import jsonschema

from evals_arena.matching import matches
from evals_arena.providers.base import ToolCall, ToolSpec
from evals_arena.schema import MockTool


@dataclass
class ToolInvocation:
    """One recorded tool call - the unit that trajectory assertions reason about."""

    turn: int
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MockToolSandbox:
    tools: list[MockTool]
    trajectory: list[ToolInvocation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_name = {t.name: t for t in self.tools}

    def specs(self) -> list[ToolSpec]:
        return [ToolSpec(t.name, t.description, t.parameters) for t in self.tools]

    def invoke(self, call: ToolCall, turn: int) -> ToolInvocation:
        record = ToolInvocation(turn=turn, name=call.name, arguments=dict(call.arguments))
        tool = self._by_name.get(call.name)
        if tool is None:
            record.error = f"unknown tool '{call.name}'"
        else:
            try:
                jsonschema.validate(call.arguments, tool.parameters)
            except jsonschema.ValidationError as exc:
                record.error = f"invalid arguments: {exc.message}"
            else:
                self._respond(tool, record)
        self.trajectory.append(record)
        return record

    @staticmethod
    def _respond(tool: MockTool, record: ToolInvocation) -> None:
        for response in tool.responses:
            if matches(response.when, record.arguments):
                if response.error:
                    record.error = response.error
                else:
                    record.result = copy.deepcopy(response.returns)
                return
        if tool.default is None:
            record.error = f"no mock response for arguments {json.dumps(record.arguments)}"
        else:
            record.result = copy.deepcopy(tool.default)

    @staticmethod
    def render(record: ToolInvocation) -> str:
        if record.error:
            return json.dumps({"error": record.error})
        if isinstance(record.result, str):
            return record.result
        return json.dumps(record.result, ensure_ascii=False)
