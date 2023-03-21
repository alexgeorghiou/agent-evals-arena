"""``arena`` command line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from evals_arena import __version__
from evals_arena.arena import compare_runs
from evals_arena.elo import DEFAULT_K, Leaderboard
from evals_arena.gate import Baseline, GatePolicy, gate
from evals_arena.judges_factory import make_judge, make_pairwise_judge
from evals_arena.loader import SuiteLoadError, load_suites
from evals_arena.pricing import PriceTable
from evals_arena.providers import create_provider
from evals_arena.report import arena_markdown, gate_markdown, leaderboard_markdown, run_markdown
from evals_arena.results import CaseResult, RunResult
from evals_arena.runner import Runner
from evals_arena.schema import Suite
from evals_arena.storage import DEFAULT_ROOT, ResultStore

app = typer.Typer(
    help="Evaluate tool-using LLM agents: run suites, compare models, gate regressions.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

PathsArg = Annotated[list[Path], typer.Argument(help="Suite files or directories.")]
ModelOpt = Annotated[
    str, typer.Option("--model", "-m", help="vendor:model, e.g. anthropic:claude-opus-5-5.")
]
JudgeOpt = Annotated[
    str | None,
    typer.Option(help="'heuristic' (offline) or a provider spec for LLM-as-judge."),
]
StoreOpt = Annotated[Path, typer.Option(help="Directory for run JSON and SQLite.")]
PricingOpt = Annotated[Path | None, typer.Option(help="YAML price overrides.")]
TagOpt = Annotated[list[str] | None, typer.Option("--tag", "-t", help="Only run tagged cases.")]


def _version(value: bool) -> None:
    if value:
        console.print(f"agent-evals-arena {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """agent-evals-arena"""


def _load(paths: list[Path]) -> list[tuple[Path, Suite]]:
    try:
        suites = load_suites(paths)
    except SuiteLoadError as exc:
        console.print(f"[red]error[/red] {exc}")
        raise typer.Exit(2) from exc
    if not suites:
        console.print("[red]error[/red] no suites found")
        raise typer.Exit(2)
    return suites


def _fake_paths(suites: list[tuple[Path, Suite]]) -> list[Path]:
    return sorted({p.parent for p, _ in suites})


def _progress(case: CaseResult) -> None:
    mark = "[green]pass[/green]" if case.passed else "[red]fail[/red]"
    console.print(
        f"  {mark} {case.case_id:<28} score={case.score:.2f} "
        f"{case.latency_ms:>7.0f}ms ${case.cost_usd:.5f}"
    )


def _execute(
    suites: list[tuple[Path, Suite]],
    model: str,
    judge: str | None,
    pricing: Path | None,
    concurrency: int,
    tags: list[str] | None,
    store: ResultStore | None,
    quiet: bool = False,
) -> list[RunResult]:
    provider = create_provider(model, fake_script_paths=_fake_paths(suites))
    runner = Runner(
        provider,
        judge=make_judge(judge),
        prices=PriceTable.load(pricing),
        concurrency=concurrency,
        tags=set(tags or []),
    )
    runs = []
    for _, suite in suites:
        if not quiet:
            console.print(f"[bold]{suite.name}[/bold] on [cyan]{provider.label}[/cyan]")
        run = runner.run(suite, progress=None if quiet else _progress)
        if store is not None:
            store.save(run)
        runs.append(run)
    return runs


def _summary_table(runs: list[RunResult]) -> Table:
    table = Table(title="Results")
    for col in ("suite", "model", "score", "pass", "p50 ms", "p95 ms", "cost $", "tokens"):
        table.add_column(col, justify="right" if col not in ("suite", "model") else "left")
    for r in runs:
        s = r.summary
        assert s is not None
        table.add_row(
            r.suite,
            r.model,
            f"{s.score:.3f}",
            f"{s.passed}/{s.cases}",
            f"{s.latency_p50_ms:.0f}",
            f"{s.latency_p95_ms:.0f}",
            f"{s.total_cost_usd:.4f}",
            f"{s.input_tokens + s.output_tokens:,}",
        )
    return table


def _write_summary(markdown: str, summary_file: Path | None) -> None:
    target = summary_file or (
        Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.getenv("GITHUB_STEP_SUMMARY") else None
    )
    if target:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(markdown + "\n\n")


@app.command()
def validate(paths: PathsArg) -> None:
    """Validate suite files without running them."""
    for path, suite in _load(paths):
        console.print(f"[green]ok[/green] {path} - {suite.name} ({len(suite.cases)} cases)")


@app.command()
def run(
    paths: PathsArg,
    model: ModelOpt = "fake:fake-strong",
    judge: JudgeOpt = None,
    store: StoreOpt = DEFAULT_ROOT,
    pricing: PricingOpt = None,
    concurrency: Annotated[int, typer.Option(min=1, max=64)] = 1,
    tag: TagOpt = None,
    fail_under: Annotated[float | None, typer.Option(help="Exit 1 if any score < this.")] = None,
    summary_file: Annotated[Path | None, typer.Option(help="Append markdown here.")] = None,
) -> None:
    """Run suites against one model and store the results."""
    runs = _execute(_load(paths), model, judge, pricing, concurrency, tag, ResultStore(store))
    console.print(_summary_table(runs))
    _write_summary("\n\n".join(run_markdown(r) for r in runs), summary_file)
    if fail_under is not None and any(r.summary and r.summary.score < fail_under for r in runs):
        console.print(f"[red]score below {fail_under}[/red]")
        raise typer.Exit(1)


@app.command()
def baseline(
    paths: PathsArg,
    model: ModelOpt = "fake:fake-strong",
    out: Annotated[Path, typer.Option(help="Baseline JSON file.")] = Path("baselines/baseline.json"),
    judge: JudgeOpt = None,
    pricing: PricingOpt = None,
) -> None:
    """Run suites and record the results as the new baseline."""
    runs = _execute(_load(paths), model, judge, pricing, 1, None, None, quiet=True)
    base = Baseline.load(out)
    for r in runs:
        base.record(r)
    base.save(out)
    console.print(_summary_table(runs))
    console.print(f"[green]baseline written[/green] {out}")


@app.command(name="gate")
def gate_cmd(
    paths: PathsArg,
    model: ModelOpt = "fake:fake-strong",
    baseline_file: Annotated[Path, typer.Option("--baseline")] = Path("baselines/baseline.json"),
    baseline_model: Annotated[
        str | None, typer.Option(help="Gate against another model's baseline entry.")
    ] = None,
    judge: JudgeOpt = None,
    pricing: PricingOpt = None,
    store: StoreOpt = DEFAULT_ROOT,
    max_score_drop: float = 0.02,
    max_pass_rate_drop: float = 0.0,
    allow_case_regressions: bool = False,
    max_cost_increase_pct: float | None = None,
    max_latency_increase_pct: float | None = None,
    min_score: float | None = None,
    summary_file: Annotated[Path | None, typer.Option(help="Append markdown here.")] = None,
    json_out: Annotated[Path | None, typer.Option(help="Write the gate result as JSON.")] = None,
) -> None:
    """Run suites and fail (exit 1) when results regress against the baseline."""
    runs = _execute(_load(paths), model, judge, pricing, 1, None, ResultStore(store), quiet=True)
    policy = GatePolicy(
        max_score_drop=max_score_drop,
        max_pass_rate_drop=max_pass_rate_drop,
        fail_on_case_regression=not allow_case_regressions,
        max_cost_increase_pct=max_cost_increase_pct,
        max_latency_increase_pct=max_latency_increase_pct,
        min_score=min_score,
    )
    result = gate(runs, Baseline.load(baseline_file), policy, baseline_model=baseline_model)
    markdown = gate_markdown(result)
    typer.echo(markdown)
    _write_summary(markdown + "\n\n" + "\n\n".join(run_markdown(r) for r in runs), summary_file)
    if json_out:
        json_out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if not result.passed:
        raise typer.Exit(1)


@app.command()
def compare(
    paths: PathsArg,
    model_a: Annotated[str, typer.Option("--model-a", "-a")],
    model_b: Annotated[str, typer.Option("--model-b", "-b")],
    judge: JudgeOpt = None,
    pricing: PricingOpt = None,
    store: StoreOpt = DEFAULT_ROOT,
    k: Annotated[float, typer.Option(help="Elo K-factor.")] = DEFAULT_K,
    summary_file: Annotated[Path | None, typer.Option(help="Append markdown here.")] = None,
) -> None:
    """Pairwise comparison of two models; updates the Elo leaderboard."""
    suites = _load(paths)
    results = ResultStore(store)
    runs_a = _execute(suites, model_a, judge, pricing, 1, None, results, quiet=True)
    runs_b = _execute(suites, model_b, judge, pricing, 1, None, results, quiet=True)
    board = Leaderboard(results.db_path, k=k)
    pairwise = make_pairwise_judge(judge)
    chunks = []
    for (_, suite), ra, rb in zip(suites, runs_a, runs_b, strict=True):
        report = compare_runs(suite, ra, rb, pairwise, board)
        chunks.append(arena_markdown(report))
        tally = report.tally()
        console.print(
            f"[bold]{suite.name}[/bold]: {ra.model} {tally['A']} - {tally['B']} {rb.model} "
            f"({tally['tie']} ties)"
        )
    console.print(_leaderboard_table(board))
    _write_summary("\n\n".join(chunks), summary_file)


def _leaderboard_table(board: Leaderboard) -> Table:
    table = Table(title="Elo leaderboard")
    for col in ("#", "model", "Elo", "games", "W", "L", "T", "win rate"):
        table.add_column(col, justify="left" if col == "model" else "right")
    for i, s in enumerate(board.standings(), 1):
        table.add_row(
            str(i), s.model, f"{s.rating:.0f}", str(s.games), str(s.wins), str(s.losses),
            str(s.ties), f"{s.win_rate:.0%}",
        )
    return table


@app.command()
def leaderboard(
    store: StoreOpt = DEFAULT_ROOT,
    markdown: Annotated[bool, typer.Option(help="Print markdown instead of a table.")] = False,
) -> None:
    """Show Elo standings."""
    board = Leaderboard(ResultStore(store).db_path)
    if markdown:
        typer.echo(leaderboard_markdown(board.standings()))
    else:
        console.print(_leaderboard_table(board))


@app.command()
def runs(
    store: StoreOpt = DEFAULT_ROOT,
    suite: str | None = None,
    model: str | None = None,
    limit: int = 20,
) -> None:
    """List stored runs."""
    table = Table(title="Runs")
    for col in ("run id", "suite", "model", "score", "pass rate", "cost $", "started"):
        table.add_column(col)
    for row in ResultStore(store).list(suite=suite, model=model, limit=limit):
        table.add_row(
            row["run_id"], row["suite"], row["model"], f"{row['score']:.3f}",
            f"{row['pass_rate']:.0%}", f"{row['cost_usd']:.4f}", row["started_at"],
        )
    console.print(table)


@app.command()
def show(run_id: str, store: StoreOpt = DEFAULT_ROOT) -> None:
    """Print a stored run as markdown."""
    try:
        result = ResultStore(store).load(run_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    typer.echo(run_markdown(result))


if __name__ == "__main__":  # pragma: no cover
    app()
