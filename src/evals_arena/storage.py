"""Persist runs as JSON documents with a SQLite index for fast listing."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals_arena.results import RunResult

DEFAULT_ROOT = Path(".arena")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    suite        TEXT NOT NULL,
    model        TEXT NOT NULL,
    provider     TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    cases        INTEGER NOT NULL,
    passed       INTEGER NOT NULL,
    pass_rate    REAL NOT NULL,
    score        REAL NOT NULL,
    cost_usd     REAL NOT NULL,
    latency_p50  REAL NOT NULL,
    latency_p95  REAL NOT NULL,
    tokens       INTEGER NOT NULL,
    path         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_suite_model ON runs (suite, model, started_at);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


class ResultStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.db_path = self.root / "arena.sqlite"
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, run: RunResult) -> Path:
        if run.summary is None:
            run.finalise()
        assert run.summary is not None
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        s = run.summary
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run.run_id,
                    run.suite,
                    run.model,
                    run.provider,
                    run.started_at,
                    run.finished_at,
                    s.cases,
                    s.passed,
                    s.pass_rate,
                    s.score,
                    s.total_cost_usd,
                    s.latency_p50_ms,
                    s.latency_p95_ms,
                    s.input_tokens + s.output_tokens,
                    str(path.relative_to(self.root)),
                ),
            )
        return path

    def load(self, run_id: str) -> RunResult:
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            matches = sorted(self.runs_dir.glob(f"{run_id}*.json"))
            if len(matches) != 1:
                raise KeyError(f"run '{run_id}' not found" if not matches else "ambiguous run id")
            path = matches[0]
        return RunResult.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list(
        self, *, suite: str | None = None, model: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []
        if suite:
            sql += " AND suite = ?"
            params.append(suite)
        if model:
            sql += " AND model = ?"
            params.append(model)
        sql += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def latest(self, suite: str, model: str) -> RunResult | None:
        rows = self.list(suite=suite, model=model, limit=1)
        return self.load(rows[0]["run_id"]) if rows else None
