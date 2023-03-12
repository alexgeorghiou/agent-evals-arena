"""Markdown renderers - suitable for PR comments and ``$GITHUB_STEP_SUMMARY``."""

from __future__ import annotations

from evals_arena.arena import ArenaReport
from evals_arena.elo import Standing
from evals_arena.gate import GateResult
from evals_arena.results import RunResult

_ICON = {"pass": "PASS", "fail": "FAIL", "new": "NEW"}


def _fmt_delta(delta: float | None) -> str:
    if delta is None:
        return "-"
    return f"{delta:+.3f}"


def run_markdown(run: RunResult, *, max_failures: int = 20) -> str:
    s = run.summary
    assert s is not None
    lines = [
        f"### `{run.suite}` on `{run.model}`",
        "",
        "| score | pass rate | cases | p50 latency | p95 latency | cost | tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {s.score:.3f} ({s.score_ci_low:.2f}-{s.score_ci_high:.2f}) | {s.pass_rate:.0%} "
        f"| {s.passed}/{s.cases} | {s.latency_p50_ms:.0f} ms | {s.latency_p95_ms:.0f} ms "
        f"| ${s.total_cost_usd:.4f} | {s.input_tokens + s.output_tokens:,} |",
    ]
    failing = [c for c in run.cases if not c.passed]
    if failing:
        lines += ["", "<details><summary>Failing cases</summary>", ""]
        lines += ["| case | score | failed assertions |", "|---|---:|---|"]
        for case in failing[:max_failures]:
            reasons = "; ".join(
                f"`{a.name}` {a.message}".strip() for a in case.assertions if not a.passed
            ) or (case.error or "")
            lines.append(f"| `{case.case_id}` | {case.score:.2f} | {reasons.replace('|', '/')} |")
        lines += ["", "</details>"]
    return "\n".join(lines)


def gate_markdown(result: GateResult) -> str:
    verdict = "passed" if result.passed else "FAILED"
    lines = [
        f"## Eval gate {verdict}",
        "",
        "| status | suite | model | score | baseline | delta | pass rate | notes |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for s in result.suites:
        base = "-" if s.baseline_score is None else f"{s.baseline_score:.3f}"
        notes = "; ".join(s.reasons) or (
            f"improved: {', '.join(s.improved_cases)}" if s.improved_cases else ""
        )
        lines.append(
            f"| {_ICON[s.status]} | `{s.suite}` | `{s.model}` | {s.score:.3f} | {base} "
            f"| {_fmt_delta(s.score_delta)} | {s.pass_rate:.0%} | {notes} |"
        )
    return "\n".join(lines)


def arena_markdown(report: ArenaReport) -> str:
    t = report.tally()
    lines = [
        f"### `{report.model_a}` vs `{report.model_b}` on `{report.suite}`",
        "",
        f"Wins A: **{t['A']}**, wins B: **{t['B']}**, ties: **{t['tie']}**",
        "",
        "| case | winner | judge | decided by | score A | score B |",
        "|---|---|---|---|---:|---:|",
    ]
    names = {"A": report.model_a, "B": report.model_b, "tie": "tie"}
    for d in report.duels:
        lines.append(
            f"| `{d.case_id}` | {names[d.winner]} | {d.judge_verdict} | {d.decided_by} "
            f"| {d.score_a:.2f} | {d.score_b:.2f} |"
        )
    return "\n".join(lines)


def leaderboard_markdown(standings: list[Standing]) -> str:
    lines = ["| # | model | Elo | games | W | L | T | win rate |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for i, s in enumerate(standings, 1):
        lines.append(
            f"| {i} | `{s.model}` | {s.rating:.0f} | {s.games} | {s.wins} | {s.losses} "
            f"| {s.ties} | {s.win_rate:.0%} |"
        )
    return "\n".join(lines)
