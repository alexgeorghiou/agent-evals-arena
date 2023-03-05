"""Elo ratings and a SQLite-backed leaderboard for pairwise model comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evals_arena.results import utcnow
from evals_arena.storage import connect

Outcome = Literal["A", "B", "tie"]
DEFAULT_RATING = 1000.0
DEFAULT_K = 32.0

_SCORES: dict[str, float] = {"A": 1.0, "B": 0.0, "tie": 0.5}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the Elo logistic model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update(
    rating_a: float, rating_b: float, outcome: Outcome, k: float = DEFAULT_K
) -> tuple[float, float]:
    """Return new ``(rating_a, rating_b)``. Zero-sum: the total rating is conserved."""
    exp_a = expected_score(rating_a, rating_b)
    delta = k * (_SCORES[outcome] - exp_a)
    return rating_a + delta, rating_b - delta


@dataclass
class Standing:
    model: str
    rating: float
    games: int
    wins: int
    losses: int
    ties: int

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0


SCHEMA = """
CREATE TABLE IF NOT EXISTS ratings (
    model      TEXT PRIMARY KEY,
    rating     REAL NOT NULL,
    games      INTEGER NOT NULL DEFAULT 0,
    wins       INTEGER NOT NULL DEFAULT 0,
    losses     INTEGER NOT NULL DEFAULT 0,
    ties       INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS matches (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    suite          TEXT NOT NULL,
    case_id        TEXT NOT NULL,
    model_a        TEXT NOT NULL,
    model_b        TEXT NOT NULL,
    winner         TEXT NOT NULL CHECK (winner IN ('A', 'B', 'tie')),
    rating_a_after REAL NOT NULL,
    rating_b_after REAL NOT NULL,
    created_at     TEXT NOT NULL
);
"""


class Leaderboard:
    def __init__(self, db_path: str | Path, *, k: float = DEFAULT_K) -> None:
        self.db_path = Path(db_path)
        self.k = k
        conn = connect(self.db_path)
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    def rating(self, model: str) -> float:
        conn = connect(self.db_path)
        try:
            row = conn.execute("SELECT rating FROM ratings WHERE model = ?", (model,)).fetchone()
        finally:
            conn.close()
        return float(row["rating"]) if row else DEFAULT_RATING

    def record(self, suite: str, case_id: str, model_a: str, model_b: str, winner: Outcome) -> None:
        if model_a == model_b:
            raise ValueError("a model cannot play against itself")
        new_a, new_b = update(self.rating(model_a), self.rating(model_b), winner, self.k)
        now = utcnow()
        conn = connect(self.db_path)
        try:
            for model, rating, result in (
                (model_a, new_a, {"A": "wins", "B": "losses", "tie": "ties"}[winner]),
                (model_b, new_b, {"A": "losses", "B": "wins", "tie": "ties"}[winner]),
            ):
                conn.execute(
                    "INSERT INTO ratings (model, rating, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(model) DO UPDATE SET rating = excluded.rating, "
                    "updated_at = excluded.updated_at",
                    (model, rating, now),
                )
                conn.execute(
                    f"UPDATE ratings SET games = games + 1, {result} = {result} + 1 "  # noqa: S608
                    "WHERE model = ?",
                    (model,),
                )
            conn.execute(
                "INSERT INTO matches (suite, case_id, model_a, model_b, winner, "
                "rating_a_after, rating_b_after, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (suite, case_id, model_a, model_b, winner, new_a, new_b, now),
            )
            conn.commit()
        finally:
            conn.close()

    def standings(self) -> list[Standing]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute("SELECT * FROM ratings ORDER BY rating DESC, model").fetchall()
        finally:
            conn.close()
        return [
            Standing(
                r["model"], float(r["rating"]), r["games"], r["wins"], r["losses"], r["ties"]
            )
            for r in rows
        ]

    def matches(self, limit: int = 100) -> list[dict[str, object]]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM matches ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]
