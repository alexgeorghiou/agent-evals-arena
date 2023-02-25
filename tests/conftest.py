from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals_arena.schema import Suite

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "suites"


def make_suite(**overrides: Any) -> Suite:
    data: dict[str, Any] = {
        "name": "unit",
        "tools": [
            {
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["q"],
                },
                "responses": [
                    {"when": {"q": "cats*"}, "returns": {"answer": "cats purr"}},
                    {"when": {"q": "broken"}, "error": "backend unavailable"},
                ],
                "default": {"answer": "nothing"},
            },
            {"name": "noop", "description": "Does nothing", "default": "ok"},
        ],
        "rubrics": {
            "basic": {
                "criteria": [
                    {"name": "mentions", "description": "mentions purr", "keywords": ["purr"]}
                ]
            }
        },
        "cases": [{"id": "c1", "input": "tell me about cats"}],
    }
    data.update(overrides)
    return Suite.model_validate(data)


@pytest.fixture
def suite() -> Suite:
    return make_suite()


@pytest.fixture
def suites_dir() -> Path:
    return SUITES
