"""Head-to-head comparison of two models on a suite, feeding the Elo leaderboard."""

from __future__ import annotations

from dataclasses import dataclass, field

from evals_arena.elo import Leaderboard, Outcome
from evals_arena.judge import PairwiseJudge, prefer_with_swap
from evals_arena.results import CaseResult, RunResult
from evals_arena.schema import Suite


@dataclass
class Duel:
    case_id: str
    winner: Outcome
    judge_verdict: Outcome
    score_a: float
    score_b: float
    decided_by: str


@dataclass
class ArenaReport:
    suite: str
    model_a: str
    model_b: str
    duels: list[Duel] = field(default_factory=list)
    rating_a: float = 0.0
    rating_b: float = 0.0

    def tally(self) -> dict[str, int]:
        counts = {"A": 0, "B": 0, "tie": 0}
        for d in self.duels:
            counts[d.winner] += 1
        return counts


def decide(
    judge: PairwiseJudge,
    task: str,
    a: CaseResult,
    b: CaseResult,
    reference: str | None,
    *,
    score_margin: float = 0.1,
) -> Duel:
    """Pick a winner for one case.

    Errors lose outright. Otherwise the pairwise judge (run twice with swapped
    positions) decides; if it calls a tie, a clear gap in assertion scores breaks it.
    """
    if a.error or b.error:
        if a.error and b.error:
            winner: Outcome = "tie"
        else:
            winner = "B" if a.error else "A"
        return Duel(a.case_id, winner, "tie", a.score, b.score, "error")
    verdict = prefer_with_swap(judge, task=task, output_a=a.output, output_b=b.output,
                               reference=reference)
    if verdict != "tie":
        return Duel(a.case_id, verdict, verdict, a.score, b.score, "judge")
    if abs(a.score - b.score) > score_margin:
        winner = "A" if a.score > b.score else "B"
        return Duel(a.case_id, winner, verdict, a.score, b.score, "assertions")
    return Duel(a.case_id, "tie", verdict, a.score, b.score, "tie")


def compare_runs(
    suite: Suite,
    run_a: RunResult,
    run_b: RunResult,
    judge: PairwiseJudge,
    leaderboard: Leaderboard | None = None,
) -> ArenaReport:
    report = ArenaReport(suite.name, run_a.model, run_b.model)
    for case in suite.cases:
        a, b = run_a.case(case.id), run_b.case(case.id)
        if a is None or b is None:
            continue
        duel = decide(judge, case.input, a, b, case.reference)
        report.duels.append(duel)
        if leaderboard is not None:
            leaderboard.record(suite.name, case.id, run_a.model, run_b.model, duel.winner)
    if leaderboard is not None:
        report.rating_a = leaderboard.rating(run_a.model)
        report.rating_b = leaderboard.rating(run_b.model)
    return report
