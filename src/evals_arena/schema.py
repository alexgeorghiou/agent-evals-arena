"""Pydantic models describing a YAML test suite.

A suite bundles mock tools, rubrics and cases. Every case is executed against
an agent and checked with a list of assertions. Assertions are a discriminated
union on ``type`` so that YAML files get precise validation errors.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------- tools


class MockResponse(_Model):
    """A canned response, returned when the tool arguments match ``when``."""

    when: dict[str, Any] = Field(default_factory=dict)
    returns: Any = None
    error: str | None = None


class MockTool(_Model):
    name: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_\-]{0,63}$")
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    responses: list[MockResponse] = Field(default_factory=list)
    default: Any = None

    @field_validator("parameters")
    @classmethod
    def _object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type", "object") != "object":
            raise ValueError("tool parameters must be a JSON schema of type 'object'")
        value.setdefault("type", "object")
        return value


# ------------------------------------------------------------------------- rubrics


class Criterion(_Model):
    name: str
    description: str
    weight: float = Field(default=1.0, gt=0)
    keywords: list[str] = Field(
        default_factory=list,
        description="Optional hints; used by the offline heuristic judge.",
    )


class Rubric(_Model):
    criteria: list[Criterion] = Field(min_length=1)
    scale_min: int = 1
    scale_max: int = 5
    instructions: str = ""

    @model_validator(mode="after")
    def _scale(self) -> Rubric:
        if self.scale_max <= self.scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        return self


# ---------------------------------------------------------------------- assertions


class _Assertion(_Model):
    weight: float = Field(default=1.0, ge=0)
    name: str | None = None


class ContainsAssertion(_Assertion):
    type: Literal["contains"]
    value: str | list[str]
    case_sensitive: bool = False
    mode: Literal["all", "any"] = "all"


class NotContainsAssertion(_Assertion):
    type: Literal["not_contains"]
    value: str | list[str]
    case_sensitive: bool = False


class EqualsAssertion(_Assertion):
    type: Literal["equals"]
    value: str
    strip: bool = True


class RegexAssertion(_Assertion):
    type: Literal["regex"]
    pattern: str
    flags: list[Literal["i", "m", "s"]] = Field(default_factory=list)


class JsonSchemaAssertion(_Assertion):
    type: Literal["json_schema"]
    schema_: dict[str, Any] = Field(alias="schema")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ToolCalledAssertion(_Assertion):
    type: Literal["tool_called"]
    tool: str
    args: dict[str, Any] | None = None
    min_times: int = Field(default=1, ge=0)
    max_times: int | None = Field(default=None, ge=0)


class ToolNotCalledAssertion(_Assertion):
    type: Literal["tool_not_called"]
    tool: str
    args: dict[str, Any] | None = None


class SequenceStep(_Model):
    tool: str
    args: dict[str, Any] | None = None


class ToolSequenceAssertion(_Assertion):
    """Check the order of tool calls.

    ``mode``:
      * ``subsequence`` - steps appear in order, other calls may be interleaved
      * ``contiguous``  - steps appear in order with nothing in between
      * ``exact``       - the full trajectory equals the steps
    """

    type: Literal["tool_sequence"]
    steps: list[SequenceStep | str] = Field(min_length=1)
    mode: Literal["subsequence", "contiguous", "exact"] = "subsequence"

    def normalised_steps(self) -> list[SequenceStep]:
        return [SequenceStep(tool=s) if isinstance(s, str) else s for s in self.steps]


class MaxToolCallsAssertion(_Assertion):
    type: Literal["max_tool_calls"]
    max: int = Field(ge=0)
    tool: str | None = None


class NoToolErrorsAssertion(_Assertion):
    type: Literal["no_tool_errors"]


class LatencyAssertion(_Assertion):
    type: Literal["latency"]
    max_ms: float = Field(gt=0)


class CostAssertion(_Assertion):
    type: Literal["cost"]
    max_usd: float = Field(gt=0)


class JudgeAssertion(_Assertion):
    type: Literal["judge"]
    rubric: str | Rubric
    min_score: float = Field(default=0.7, ge=0, le=1)


Assertion = Annotated[
    ContainsAssertion
    | NotContainsAssertion
    | EqualsAssertion
    | RegexAssertion
    | JsonSchemaAssertion
    | ToolCalledAssertion
    | ToolNotCalledAssertion
    | ToolSequenceAssertion
    | MaxToolCallsAssertion
    | NoToolErrorsAssertion
    | LatencyAssertion
    | CostAssertion
    | JudgeAssertion,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- cases


class Case(_Model):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-\.]*$")
    description: str = ""
    input: str
    tools: list[str] | None = Field(
        default=None, description="Subset of suite tools; defaults to all suite tools."
    )
    reference: str | None = Field(default=None, description="Gold answer for judges.")
    assertions: list[Assertion] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class AgentConfig(_Model):
    system_prompt: str = "You are a helpful assistant. Use the tools when they help."
    max_turns: int = Field(default=8, ge=1, le=50)
    max_tokens: int = Field(default=4096, ge=1)


class Suite(_Model):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    version: str = "1"
    description: str = ""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: list[MockTool] = Field(default_factory=list)
    rubrics: dict[str, Rubric] = Field(default_factory=dict)
    default_assertions: list[Assertion] = Field(default_factory=list)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def _references(self) -> Suite:
        tool_names = [t.name for t in self.tools]
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("duplicate tool names in suite")
        case_ids = [c.id for c in self.cases]
        dupes = sorted({c for c in case_ids if case_ids.count(c) > 1})
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
        known = set(tool_names)
        for case in self.cases:
            for name in case.tools or []:
                if name not in known:
                    raise ValueError(f"case '{case.id}' references unknown tool '{name}'")
            for assertion in [*self.default_assertions, *case.assertions]:
                if isinstance(assertion, JudgeAssertion) and isinstance(assertion.rubric, str):
                    if assertion.rubric not in self.rubrics:
                        raise ValueError(
                            f"case '{case.id}' references unknown rubric '{assertion.rubric}'"
                        )
        return self

    def tools_for(self, case: Case) -> list[MockTool]:
        if case.tools is None:
            return list(self.tools)
        wanted = set(case.tools)
        return [t for t in self.tools if t.name in wanted]

    def assertions_for(self, case: Case) -> list[Any]:
        return [*self.default_assertions, *case.assertions]

    def resolve_rubric(self, rubric: str | Rubric) -> Rubric:
        return self.rubrics[rubric] if isinstance(rubric, str) else rubric
