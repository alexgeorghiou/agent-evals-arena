"""Load and validate suites from YAML files or directories."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError

from evals_arena.schema import Suite

SUITE_GLOBS = ("*.yaml", "*.yml")


class SuiteLoadError(Exception):
    """Raised when a suite file cannot be parsed or validated."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def _format_validation_error(err: ValidationError) -> str:
    lines = []
    for item in err.errors():
        loc = ".".join(str(p) for p in item["loc"]) or "<root>"
        lines.append(f"  - {loc}: {item['msg']}")
    return "invalid suite:\n" + "\n".join(lines)


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteLoadError(path, f"YAML parse error: {exc}") from exc
    except OSError as exc:
        raise SuiteLoadError(path, str(exc)) from exc
    if not isinstance(raw, dict):
        raise SuiteLoadError(path, "top-level document must be a mapping")
    try:
        return Suite.model_validate(raw)
    except ValidationError as exc:
        raise SuiteLoadError(path, _format_validation_error(exc)) from exc


def discover(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of suite files.

    Files under a ``fakes`` directory or ending in ``.fake.yaml`` are scripts for the
    offline provider, not suites, and are skipped.
    """
    found: set[Path] = set()
    for item in paths:
        p = Path(item)
        if p.is_dir():
            for pattern in SUITE_GLOBS:
                found.update(p.glob(pattern))
        elif p.exists():
            found.add(p)
        else:
            raise SuiteLoadError(p, "no such file or directory")
    return sorted(f for f in found if not f.name.endswith((".fake.yaml", ".fake.yml")))


def load_suites(paths: Iterable[str | Path]) -> list[tuple[Path, Suite]]:
    return [(p, load_suite(p)) for p in discover(paths)]
